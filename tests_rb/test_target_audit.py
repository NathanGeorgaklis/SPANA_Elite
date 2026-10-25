# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R1-F independent code review + test audit (2026-10-25) of commit 3f9dd51; R1-F instruction
"""R1-F audit tests: probes that closed gaps found by mutation testing, regressions for the defects the
independent review reproduced (H1 newcomer-in-contact swap, H2 merge resolved by an unrelated newcomer, M1 scale
across a cut, L1 half-initialised seed), and the documented uncontrasted-reacquisition limitation. Neutral
TARGET / PERSON_* / TWIN names only; no benchmark knowledge."""
import unittest

from player_tracker_rb import population as P
from player_tracker_rb import target as T
from player_tracker_rb.config import TrackerConfig
from player_tracker_rb.appearance import AppearanceExtractor
from player_tracker_rb.camera import CUT, FIRST_FRAME, identity_motion, motion_from_matrix
from player_tracker_rb.detector import PersonCandidate
from tests_rb import _synth_scene as SS
from tests_rb._synth_scene import FPS, SceneFrame, box_at, make_people, run, still, linear

CFG = TrackerConfig.default()
KIT_A = (200, 60, 40)
KIT_B = (40, 40, 210)
SAME_KIT_COS = 0.75


def people(*others):
    return make_people([("TARGET", KIT_A, 1.0)] + list(others))


def _gap_then_return(gap, back, total_after):
    return [SceneFrame({"TARGET": b}, drop=["TARGET"] if gap[0] <= k < gap[1] else [])
            for k, b in enumerate(still(back + total_after, (500, 400)))]


class ProbeConfidence(unittest.TestCase):
    def test_untrusted_confidence_strictly_decays_below_the_ceiling(self):
        r = run(_gap_then_return((10, 50), 50, 10), people())
        last = r.frames[9].identity_confidence_rb
        hl = CFG.get("confidence_half_life_s")
        for f in r.frames[10:50]:
            expect = min(CFG.get("untrusted_identity_confidence_ceiling"),
                         last * 0.5 ** (f.time_since_confirmed_s / hl))
            self.assertAlmostEqual(f.identity_confidence_rb, expect, places=9)
        self.assertLess(r.frames[49].identity_confidence_rb, 0.5 * CFG.get("untrusted_identity_confidence_ceiling"))

    def test_trusted_confidence_is_decided_by_the_weakest_evidence(self):
        tt = run([SceneFrame({"TARGET": box_at(500, 400, 40, 100)})], people()).tracker
        m = CFG.get("contrastive_margin")
        ev = T.CandidateEvaluation(
            detection_index=1, tracklet_id="T1", box=(0, 0, 1, 1), role=T.CONTINUATION, in_gate=True,
            distance_h=0.1, merged=False, motion=None, quality_flags=(), target_similarity={"composite": 0.999},
            best_competitor="X", best_competitor_similarity=0.999 - m, contrast=m, margin=m, unknown_contacts=(),
            passed=True, details=())
        # high appearance, contrast exactly at the margin -> k ~ 0 -> confidence ~ 0.5
        self.assertLess(tt._confidence_trusted(ev), 0.51)


class ProbeAmbiguousAging(unittest.TestCase):
    def test_persistent_ambiguity_ages_to_lost(self):
        """A person first detected already in contact with the tracked target stays there for longer than
        max_ambiguous_s: the target must not stay AMBIGUOUS indefinitely; it ages to LOST (AGED_OUT)."""
        ppl = people(("PERSON_A", KIT_B, 0.2))
        frames = [SceneFrame({"TARGET": box_at(500, 400, 40, 100)}) for _ in range(15)]
        frames += [SceneFrame({"TARGET": box_at(500, 400, 40, 100), "PERSON_A": box_at(530, 400, 40, 100)},
                              order=["TARGET", "PERSON_A"]) for _ in range(60)]
        r = run(frames, ppl)
        self.assertEqual(r.swaps(), [])
        self.assertIn(T.AMBIGUOUS, r.states()[15:])
        lim = CFG.get("max_ambiguous_s") + 1.5 / FPS
        late = [f for f in r.frames[15:] if f.time_since_confirmed_s is not None and f.time_since_confirmed_s > lim]
        self.assertTrue(late)
        for f in late:
            self.assertNotEqual(f.rb_state, T.AMBIGUOUS, f.rel_frame)
        self.assertTrue(any(T.AGED_OUT in f.identity_uncertainty_reasons for f in late))


class ProbeNewcomerInContact(unittest.TestCase):
    def test_newcomer_first_seen_in_contact_never_becomes_the_target(self):
        """FAILS ON 3f9dd51 (real silent swap). A same-kit person (embedding cosine 0.8, inside the realistic
        same-kit range) is first detected already overlapping the tracked target and stands beside it. The
        target's continuation is blocked (unknown competitor in contact); reacquisition must not then accept the
        newcomer: the target's own observed box is a physically separate body right beside it."""
        for cos in (0.8, 0.9, 1.0):
            with self.subTest(cos=cos):
                ppl = people(("PERSON_A", KIT_A, cos))
                frames = [SceneFrame({"TARGET": box_at(500, 400, 40, 100)}) for _ in range(15)]
                frames += [SceneFrame({"TARGET": box_at(500, 400, 40, 100), "PERSON_A": box_at(530, 400, 40, 100)},
                                      order=["TARGET", "PERSON_A"]) for _ in range(60)]
                r = run(frames, ppl)
                self.assertEqual(r.swaps(), [], "silent identity swap onto the newcomer")


class ProbeGate(unittest.TestCase):
    def test_gate_frozen_while_ambiguous(self):
        tt = run([SceneFrame({"TARGET": box_at(500, 400, 40, 100)})], people()).tracker
        tt._last_confirm_t = 0.0
        tt._state = T.AMBIGUOUS
        tt._ambiguous_since_t = 0.2
        g = tt._gate(0.9)
        self.assertAlmostEqual(g, min(CFG.get("gate_base_h") + CFG.get("gate_growth_h_per_s") * 0.2,
                                      CFG.get("gate_max_h")))
        tt._state, tt._ambiguous_since_t = T.OCCLUDED, None      # as _enter_uncertain does on leaving AMBIGUOUS
        self.assertGreater(tt._gate(0.9), g)


class ProbeVerification(unittest.TestCase):
    def test_flickering_candidate_is_not_accepted(self):
        """After a long absence the returning box is detected in only one frame of three: the hit fraction stays
        below reacq_min_hit_fraction, so it must not be accepted while that persists."""
        frames = []
        for k in range(130):
            drop = (10 <= k < 50) or (k >= 50 and (k - 50) % 3 != 0)
            frames.append(SceneFrame({"TARGET": box_at(500, 400, 40, 100)}, drop=["TARGET"] if drop else []))
        r = run(frames, people())
        self.assertEqual([k for k in r.tracking_frames() if k >= 10], [])

    def test_return_far_away_after_long_absence_is_reacquired(self):
        """After more than lost_long_s the position is stale: the target is searched anywhere (appearance only)."""
        n_gap = int((CFG.get("lost_long_s") + 0.5) * FPS)
        frames = [SceneFrame({"TARGET": box_at(300, 400, 40, 100)}) for _ in range(10)]
        frames += [SceneFrame({}) for _ in range(n_gap)]
        frames += [SceneFrame({"TARGET": box_at(1000, 400, 40, 100)}) for _ in range(40)]
        r = run(frames, people())
        self.assertEqual(r.frames[-1].rb_state, T.TRACKING)

    def test_rival_in_accepting_frame_blocks_acceptance(self):
        """Verification has started on the true target; an identical twin then appears in the gate. The accepting
        frame has a rival within ambiguity_margin: no acceptance while the twin is present."""
        ppl = people(("TWIN", KIT_A, 1.0))
        frames = [SceneFrame({"TARGET": box_at(500, 400, 40, 100)}) for _ in range(10)]
        frames += [SceneFrame({}) for _ in range(40)]
        frames += [SceneFrame({"TARGET": box_at(500, 400, 40, 100)}) for _ in range(5)]
        frames += [SceneFrame({"TARGET": box_at(500, 400, 40, 100), "TWIN": box_at(580, 400, 40, 100)})
                   for _ in range(40)]
        r = run(frames, ppl)
        self.assertEqual([k for k in r.tracking_frames() if k >= 10], [])


class ProbeMotionVeto(unittest.TestCase):
    def test_implausibly_sized_lookalike_is_not_reacquired_soon_after_loss(self):
        """The target vanishes; 0.1 s later a same-kit look-alike 2.5x its height stands at the same place. Within
        0.8 s no body can change apparent size that much (motion SCALE veto), so it is never accepted."""
        ppl = people(("PERSON_A", KIT_A, 0.9))
        frames = [SceneFrame({"TARGET": box_at(500, 400, 40, 100)}) for _ in range(15)]
        frames += [SceneFrame({}) for _ in range(3)]
        frames += [SceneFrame({"PERSON_A": box_at(500, 400, 100, 250)}) for _ in range(21)]
        r = run(frames, ppl)
        self.assertEqual(r.swaps(), [])
        self.assertEqual([k for k in r.tracking_frames() if k >= 15], [])


class MotionVetoApplied(unittest.TestCase):
    def test_box_too_far_for_one_frame_is_vetoed_and_logged(self):
        """One frame after the target's last confirmation, a same-kit look-alike appears 0.65 target heights away:
        inside the search gate (0.75 h) but beyond what a body can move in 1/30 s (motion bound ~0.52 h). The
        candidate carries MOTION_VETO, does not pass, and the target is not in a verifying state on that frame.
        (Defence in depth: population tracklets and the reacquisition window also protect; this pins the veto.)"""
        ppl = people(("PERSON_A", KIT_A, 0.9))
        frames = [SceneFrame({"TARGET": box_at(500, 400, 40, 100)}) for _ in range(15)]
        frames += [SceneFrame({"PERSON_A": box_at(565, 400, 40, 100)}) for _ in range(3)]
        r = run(frames, ppl)
        c = r.frames[15].decision["candidates"]
        self.assertEqual(len(c), 1)
        self.assertTrue(c[0]["in_gate"])
        self.assertIn(T.MOTION_VETO, c[0]["details"])
        self.assertFalse(c[0]["passed"])
        self.assertNotEqual(r.frames[15].rb_state, T.REACQUIRING)
        self.assertEqual(r.swaps(), [])


class BankIsolationRule(unittest.TestCase):
    def test_trusted_but_touching_box_is_refused_by_the_target_rule(self):
        """The target stays TRACKING (an opponent with its own bank stands slightly overlapping it), but every bank
        offer while the boxes overlap above bank_contact_iou is refused by the TARGET rule itself (NOT_ISOLATED),
        not only by AppearanceBank's own check (defence in depth)."""
        from tests_rb._synth_scene import iou
        ppl = people(("PERSON_A", KIT_B, 0.0))
        frames = [SceneFrame({"TARGET": box_at(500, 400, 40, 100),
                              "PERSON_A": box_at(max(600 - 2 * k, 532), 400, 40, 100)}) for k in range(60)]
        r = run(frames, ppl, pose_gain=0.5)
        touching = [f for f in r.frames if len(r.dets[f.rel_frame]) == 2
                    and iou(r.dets[f.rel_frame][0].box, r.dets[f.rel_frame][1].box) > CFG.get("bank_contact_iou")
                    and f.decision.get("bank") and "SAMPLE_NOT_DUE" not in f.decision["bank"]["refusals"]]
        self.assertTrue(touching)
        for f in touching:
            self.assertEqual(f.rb_state, T.TRACKING)
            self.assertEqual(f.decision["bank"]["status"], "REFUSED")
            self.assertIn("NOT_ISOLATED", f.decision["bank"]["refusals"], f.rel_frame)


class ProbeBankOnlyOnTrusted(unittest.TestCase):
    def test_bank_never_offered_on_an_untrusted_frame(self):
        """An identical-looking person stands clear of the target inside its gate (target AMBIGUOUS, its own box
        isolated). The target bank must not even be offered a sample on any non-TRACKING frame."""
        from player_tracker_rb import appearance as A
        calls = []
        orig = A.AppearanceBank.add

        def add(self_, feat, ctx):
            calls.append(self_)
            return orig(self_, feat, ctx)
        ppl = people(("TWIN", KIT_A, 1.0))
        frames = [SceneFrame({"TARGET": box_at(500, 400, 40, 100)}) for _ in range(15)]
        frames += [SceneFrame({"TARGET": box_at(500, 400, 40, 100), "TWIN": box_at(570, 400, 40, 100)})
                   for _ in range(20)]
        A.AppearanceBank.add = add
        per_frame = []
        try:
            from tests_rb import _synth_scene as SS
            orig_update = T.TargetTracker.update

            def update(self_, res, image):
                n0 = sum(1 for b in calls if b is self_.bank)
                f = orig_update(self_, res, image)
                per_frame.append((f.rb_state, sum(1 for b in calls if b is self_.bank) - n0))
                return f
            T.TargetTracker.update = update
            try:
                r = run(frames, ppl)
            finally:
                T.TargetTracker.update = orig_update
        finally:
            A.AppearanceBank.add = orig
        self.assertIn(T.AMBIGUOUS, r.states())
        for state, n in per_frame:
            if state != T.TRACKING:
                self.assertEqual(n, 0, state)


class ReviewRegressions(unittest.TestCase):
    """Defects found by the independent R1-F review/audit on 3f9dd51 (each reproduced before the fix)."""

    def test_identical_unbanked_neighbour_beside_tracked_target_never_becomes_target(self):
        """H1/A1: an identical-looking person appears 50 px beside the tracked target at the image edge (edge crop:
        no bank yet). Reacquisition must not accept the neighbour while the target's own box is observed beside
        it (the target's observed tracklet is a competitor, not excluded as 'self')."""
        ppl = people(("TWIN", KIT_A, 1.0))
        frames = [SceneFrame({"TARGET": box_at(70, 400, 40, 100)}) for _ in range(15)]
        frames += [SceneFrame({"TARGET": box_at(70, 400, 40, 100), "TWIN": box_at(20, 400, 40, 100)})
                   for _ in range(40)]
        r = run(frames, ppl)
        self.assertEqual(r.swaps(), [])

    def test_newcomer_appearing_in_contact_after_being_hidden_never_becomes_target(self):
        """H1: a same-kit person, undetected until it stands in contact beside the visible target, is never
        reacquired as the target (cosines across and above the realistic same-kit range)."""
        for cos in (0.78, 0.9, 1.0):
            with self.subTest(cos=cos):
                ppl = people(("PERSON_A", KIT_A, cos))
                frames = []
                for k in range(70):
                    boxes = {"TARGET": box_at(400, 400, 40, 100)}
                    if k >= 20:
                        boxes["PERSON_A"] = box_at(434, 400, 40, 100)
                    frames.append(SceneFrame(boxes, order=["TARGET", "PERSON_A"]))
                r = run(frames, ppl)
                self.assertEqual(r.swaps(), [])

    def test_unrelated_newcomer_beside_a_merged_box_does_not_let_the_target_track_it(self):
        """H2: the target and a same-kit person merge into one detection; an unrelated third person then appears
        just beside the merged box. The merge must not be 'resolved' by that newcomer: the target never TRACKS the
        two-person box and its bank stays pure."""
        ppl = people(("PERSON_A", KIT_A, 0.78), ("PERSON_B", KIT_B, 0.0))
        frames = []
        for k in range(150):
            boxes = {"TARGET": box_at(400, 400, 40, 100), "PERSON_A": box_at(440 - min(k, 25), 400, 40, 100)}
            fr = SceneFrame(boxes, order=["TARGET", "PERSON_A", "PERSON_B"])
            if k >= 25:
                fr.merge = [("TARGET", "PERSON_A")]
            if k >= 34:
                boxes["PERSON_B"] = box_at(470 + 12 * (k - 34), 400, 40, 100)
            frames.append(fr)
        r = run(frames, ppl, cfg=TrackerConfig.with_overrides({"population_duplicate_iou": 0.99}))
        self.assertEqual(r.swaps(), [])
        self.assertTrue(r.bank_is_pure())

    def test_cut_resets_scale_so_a_zoomed_target_can_be_reacquired(self):
        """M1: after a hard cut the target reappears 2.5x taller (new shot, closer camera). Size history from the
        previous shot means nothing: no scale veto, and the target is reacquired by appearance."""
        frames = [SceneFrame({"TARGET": box_at(500, 400, 40, 100)}) for _ in range(20)]
        cut = identity_motion(CUT, reasons=("CUT",), continuity_broken=True)
        frames.append(SceneFrame({"TARGET": box_at(700, 380, 100, 250)}, camera=cut))
        frames += [SceneFrame({"TARGET": box_at(700, 380, 100, 250)}) for _ in range(40)]
        r = run(frames, people())
        self.assertEqual(r.frames[-1].rb_state, T.TRACKING)
        for f in r.frames[20:]:
            for c in f.decision.get("candidates", []):
                self.assertNotIn(T.MOTION_VETO, c["details"], f.rel_frame)
            self.assertNotEqual(f.target_height_px, 100.0, f"rel {f.rel_frame}: old-shot height after the cut")
        acc = r.tracker.reacquisition_events[0].reacq_accepted_frame
        self.assertEqual(r.states()[acc:], [T.TRACKING] * (len(frames) - acc), "the new scale must be its own reference")

    def test_seed_on_a_merged_detection_is_refused_and_tracker_stays_unusable(self):
        """L1 + seed contamination (found while writing this test): on the first frame the population has no size
        history, so a two-person detection is not yet merge-flagged. The human's seed box is the size reference:
        a detection merge-sized against it is refused, and the tracker then refuses update()."""
        ppl = people(("PERSON_A", KIT_B, 0.0))
        a, b = box_at(300, 400, 40, 100), box_at(330, 400, 40, 100)
        fr = SceneFrame({"TARGET": a, "PERSON_A": b}, merge=[("TARGET", "PERSON_A")])
        pop = P.Population(CFG, SS.W, SS.H)
        tt = T.TargetTracker(CFG, pop, AppearanceExtractor(CFG, SS.SignatureEmbedding(ppl)))
        cands = [PersonCandidate(d.box, 0.9, False, "synthetic") for d in SS.detect(fr)]
        res = pop.update(0, 0.0, cands, identity_motion(FIRST_FRAME))
        with self.assertRaises(T.SeedError):
            tt.seed(res, SS.render(fr, ppl), a)        # the human boxes the target; the detection holds two
        res2 = pop.update(1, 1 / FPS, cands, motion_from_matrix([[1, 0, 0], [0, 1, 0]], SS.W, SS.H))
        with self.assertRaises(T.TargetError):
            tt.update(res2, SS.render(fr, ppl))

    def test_interrupted_verification_never_carries_over_to_a_new_box(self):
        """Verification starts on a returning box, which then vanishes before the window completes. A later box
        (a new tracklet) never inherits that verification: its window starts from zero, nothing is TRACKED in
        between, and, while the interrupted target-looking tracklet is still remembered as a recent competitor,
        the new box cannot pass on contrast (conservative: KNOWN coverage cost until population_closed_memory_s)."""
        frames = [SceneFrame({"TARGET": box_at(500, 400, 40, 100)}) for _ in range(10)]
        frames += [SceneFrame({}) for _ in range(40)]
        frames += [SceneFrame({"TARGET": box_at(500, 400, 40, 100)}) for _ in range(6)]   # verification starts
        frames += [SceneFrame({}) for _ in range(30)]                                     # and is interrupted
        frames += [SceneFrame({"TARGET": box_at(500, 400, 40, 100)}) for _ in range(190)]
        r = run(frames, people())
        ver = [(k, f.decision["reacquisition"]) for k, f in enumerate(r.frames)
               if f.decision.get("reacquisition", {}).get("verifying")]
        first = {d["verifying"] for k, d in ver if k < 86}
        later = [(k, d) for k, d in ver if d["verifying"] not in first]
        self.assertEqual(len(first), 1)
        self.assertTrue(later, "the returning box must eventually be verified")
        self.assertEqual(later[0][1]["elapsed_s"], 0.0)
        self.assertEqual([k for k in r.tracking_frames() if 10 <= k < later[0][0]], [])
        self.assertEqual(r.frames[-1].rb_state, T.TRACKING)
        self.assertEqual(r.swaps(), [])


class VerificationSwitch(unittest.TestCase):
    def test_verification_moves_to_a_clearly_better_candidate_at_once(self):
        """A same-kit look-alike at the image edge (edge crop: never bank-eligible, so it cannot be contrasted
        against) starts verification. When the true target reappears inside the gate, clearly more similar and
        passing, verification restarts on the target in that frame (not after a wasted window), and the target, not
        the look-alike, is accepted after reacq_window_s."""
        ppl = people(("PERSON_A", KIT_A, 0.8))
        frames = [SceneFrame({"TARGET": box_at(110, 400, 40, 100)}) for _ in range(10)]
        frames += [SceneFrame({}) for _ in range(40)]
        frames += [SceneFrame({"PERSON_A": box_at(20, 400, 40, 100)}) for _ in range(6)]
        frames += [SceneFrame({"PERSON_A": box_at(20, 400, 40, 100), "TARGET": box_at(170, 400, 40, 100)})
                   for _ in range(40)]
        r = run(frames, ppl)
        ver = lambda k: r.frames[k].decision.get("reacquisition", {}).get("verifying")
        self.assertIsNotNone(ver(55))
        self.assertNotEqual(ver(56), ver(55))
        self.assertEqual(r.frames[56].decision["reacquisition"]["elapsed_s"], 0.0)
        self.assertEqual(r.swaps(), [])
        first = min(r.tracking_frames()[10:] or [len(frames)])
        self.assertLessEqual(first, 56 + int(CFG.get("reacq_window_s") * FPS) + 1)


class UnverifiableRivalsAndContacts(unittest.TestCase):
    def test_bigger_unknown_body_touching_the_target_blocks_it(self):
        """Contact is tested in BOTH directions: a newly detected, larger body (no bank yet) covers 12% of the
        target's box while the target covers only ~8% of it. That is contact: the target cannot be confirmed."""
        ppl = people(("PERSON_A", KIT_B, 0.0))
        tb = box_at(500, 400, 40, 100)
        big = (tb[2] - 4.8, tb[1], tb[2] - 4.8 + 50, tb[1] + 125)
        frames = [SceneFrame({"TARGET": tb}) for _ in range(15)]
        frames += [SceneFrame({"TARGET": tb, "PERSON_A": big}, order=["TARGET", "PERSON_A"]) for _ in range(3)]
        r = run(frames, ppl)
        cont = [c for c in r.frames[15].decision["candidates"] if c["role"] == T.CONTINUATION]
        self.assertEqual(len(cont), 1)
        self.assertIn(T.UNKNOWN_COMPETITOR_IN_CONTACT, cont[0]["details"])
        self.assertNotEqual(r.frames[15].rb_state, T.TRACKING)

    def test_rival_present_through_the_window_prevents_acceptance(self):
        """Verification of the true target is under way when an almost identical person (no bank: edge crop)
        appears in the gate, within ambiguity_margin. While the rival stays, the target is never accepted."""
        ppl = people(("TWIN", KIT_A, 0.97))
        frames = [SceneFrame({"TARGET": box_at(110, 400, 40, 100)}) for _ in range(10)]
        frames += [SceneFrame({}) for _ in range(40)]
        frames += [SceneFrame({"TARGET": box_at(170, 400, 40, 100)}) for _ in range(5)]
        frames += [SceneFrame({"TARGET": box_at(170, 400, 40, 100), "TWIN": box_at(20, 400, 40, 100)})
                   for _ in range(40)]
        r = run(frames, ppl)
        self.assertTrue(any(r.frames[k].decision.get("reacquisition", {}).get("rivals") for k in range(56, 60)))
        self.assertEqual([k for k in r.tracking_frames() if k >= 10], [])
        self.assertEqual(r.swaps(), [])


class ContactMemory(unittest.TestCase):
    def test_closed_competitor_stops_counting_after_contact_memory(self):
        """The interrupted-verification scene: a closed target-looking tracklet blocks the real return by contrast.
        It may block only for contact_memory_s after closing: with a shorter contact memory the return is verified
        much earlier than with the default (equal to population_closed_memory_s)."""
        frames = [SceneFrame({"TARGET": box_at(500, 400, 40, 100)}) for _ in range(10)]
        frames += [SceneFrame({}) for _ in range(40)]
        frames += [SceneFrame({"TARGET": box_at(500, 400, 40, 100)}) for _ in range(6)]
        frames += [SceneFrame({}) for _ in range(30)]
        frames += [SceneFrame({"TARGET": box_at(500, 400, 40, 100)}) for _ in range(120)]

        def first_new_verification(cfg):
            r = run(frames, people(), cfg=cfg)
            return min(k for k, f in enumerate(r.frames) if k >= 86
                       and f.decision.get("reacquisition", {}).get("verifying"))
        short = first_new_verification(TrackerConfig.with_overrides({"contact_memory_s": 0.5}))
        default = first_new_verification(None)
        self.assertLessEqual(short, 86 + int(FPS / 10))
        self.assertGreater(default, short + int(FPS / 2))


class KnownOthersGate(unittest.TestCase):
    def test_body_seen_briefly_beside_a_just_reacquired_target_is_not_ruled_out(self):
        """M2: right after a reacquisition the target has been on its new tracklet for less than
        target_known_other_min_s. A body seen beside it then is NOT recorded as a known other (one trusted frame
        cannot rule a body out for its whole tracklet), so it stays a reacquisition candidate later."""
        base = [SceneFrame({"TARGET": box_at(500, 400, 40, 100)}) for _ in range(10)]
        base += [SceneFrame({}) for _ in range(40)]
        base += [SceneFrame({"TARGET": box_at(500, 400, 40, 100)}) for _ in range(30)]
        acc = run(base, people()).tracker.reacquisition_events[0].reacq_accepted_frame
        ppl = people(("PERSON_A", KIT_B, 0.0))
        frames = base[:acc + 1]
        frames += [SceneFrame({"TARGET": box_at(500, 400, 40, 100), "PERSON_A": box_at(600, 400, 40, 100)})
                   for _ in range(2)]
        frames += [SceneFrame({"PERSON_A": box_at(600, 400, 40, 100)}) for _ in range(20)]
        r = run(frames, ppl)
        self.assertEqual(r.tracker.reacquisition_events[0].reacq_accepted_frame, acc)
        self.assertTrue(all(r.frames[k].rb_state == T.TRACKING for k in range(acc, acc + 3)))
        later = [r.frames[k].decision.get("reacquisition", {}) for k in range(acc + 3, len(frames))]
        self.assertTrue(any("excluded_known_others" in d for d in later))
        for d in later:
            self.assertEqual(d.get("excluded_known_others", []), [])


class H1Layers(unittest.TestCase):
    """The H1 fix has two independent layers; each is pinned on its own (either alone stops the swap)."""

    def test_layer_a_visible_target_competes_with_the_newcomer(self):
        """Evidence layer: while the target's own box is observed beside a same-kit newcomer, the newcomer's
        reacquisition contrast is computed AGAINST the target's tracklet (its logged best competitor is the target's
        visible detection), so the target never competes with itself but always with the newcomer."""
        ppl = people(("PERSON_A", KIT_A, 0.9))
        frames = []
        for k in range(45):
            boxes = {"TARGET": box_at(400, 400, 40, 100)}
            if k >= 20:
                boxes["PERSON_A"] = box_at(434, 400, 40, 100)
            frames.append(SceneFrame(boxes, order=["TARGET", "PERSON_A"]))
        r = run(frames, ppl)
        checked = 0
        for k in range(20, 45):
            tgt = [i for i, d in enumerate(r.dets[k]) if d.owners == ("TARGET",)]
            for c in r.frames[k].decision.get("candidates", []):
                if c["role"] == T.REACQUISITION and c["detection_index"] not in tgt:
                    self.assertEqual(c["best_competitor"], f"D{tgt[0]}", k)
                    self.assertFalse(c["passed"], k)
                    checked += 1
        self.assertGreater(checked, 0)
        self.assertEqual(r.swaps(), [])

    def test_layer_b_visible_target_box_is_a_rival_when_contrast_is_impossible(self):
        """Rival layer: the target's tracklet has NO population bank (a small object inside its box from the first
        update makes every crop CONTAINS_OTHER), so an identical newcomer cannot be contrasted against it and passes
        every per-frame test. Only the target's own visible box, as a rival, blocks the newcomer's acceptance."""
        ppl = make_people([("TARGET", KIT_A, 1.0), ("TWIN", KIT_A, 1.0), ("BALL", (250, 250, 250), 0.0)])
        frames = []
        for k in range(75):
            tb = box_at(300, 400, 50, 120)
            boxes = {"TARGET": tb}
            if k >= 1:
                boxes["BALL"] = (tb[0] + 6, tb[3] - 30, tb[0] + 50, tb[3] - 2)
            if k >= 15:
                boxes["TWIN"] = box_at(380, 400, 50, 120)
            frames.append(SceneFrame(boxes, order=["TARGET", "BALL", "TWIN"]))
        r = run(frames, ppl, cfg=TrackerConfig.with_overrides({"population_part_box_min_height_ratio": 0.3}))
        twin_passing = [k for k in range(16, 75) for c in r.frames[k].decision.get("candidates", [])
                        if c["role"] == T.REACQUISITION and r.dets[k][c["detection_index"]].owners == ("TWIN",)
                        and c["passed"]]
        self.assertTrue(twin_passing, "the scene must make the twin pass per frame (contrast impossible)")
        self.assertEqual(r.swaps(), [])
        self.assertEqual([k for k in r.tracking_frames() if k >= 15], [])


class DocumentedLimitations(unittest.TestCase):
    """KNOWN LIMIT (reported to the user, not hidden): with no competitor bank to contrast against, a lone
    same-kit look-alike whose appearance passes reacq_appearance_floor can be reacquired as the target. On rendered
    scenes this happens from embedding cosine ~0.78 (composite ~0.86, inside the real same-kit tail: R1-D same-kit
    P99 0.855, max 0.867). These tests pin how such a decision is EXPOSED so downstream can treat it carefully."""

    def test_below_the_floor_a_lone_lookalike_is_rejected(self):
        ppl = people(("PERSON_A", KIT_A, SAME_KIT_COS))
        frames = [SceneFrame({"TARGET": b}) for b in linear(40, (1150, 400), (8, 0))]
        frames += [SceneFrame({"PERSON_A": b}) for b in linear(120, (1250, 400), (-4, 0))]
        r = run(frames, ppl)
        self.assertEqual(r.swaps(), [])

    def test_uncontrasted_reacquisition_is_flagged_and_capped(self):
        for cos in (0.78, 0.8):
            with self.subTest(cos=cos):
                ppl = people(("PERSON_A", KIT_A, cos))
                frames = [SceneFrame({"TARGET": b}) for b in linear(40, (1150, 400), (8, 0))]
                frames += [SceneFrame({"PERSON_A": b}) for b in linear(120, (1250, 400), (-4, 0))]
                r = run(frames, ppl)
                for ev in r.tracker.reacquisition_events:
                    if ev.evidence["best_competitor"] is None:
                        self.assertFalse(ev.evidence["contrast_available"])
                        self.assertIsNone(ev.evidence["contrast"])
                        self.assertLessEqual(r.frames[ev.reacq_accepted_frame].identity_confidence_rb,
                                             1.0 / 2 + 1e-9)


class UncontrastedEpisodeCap(unittest.TestCase):
    """R1-G rule (Nathan, 2026-10-25: 'keep uncontrasted reacquisition capped/flagged, not treated as
    high-confidence identity'; independent review H1: a 0.5 cap equals full trust for the surviving consumers).
    An episode opened by a reacquisition that had nothing to contrast against reports its observed box as TRACKING
    but UNTRUSTED (reason INSUFFICIENT_CONTRAST, confidence <= untrusted ceiling), never updates the target bank and
    never rules out a body as a known other, until a TRACKING frame passes a real contrast against a competitor that
    could be confused with the target (its similarity to the crop >= appearance_floor; review L4)."""

    def _return_alone_then(self, other):
        ppl = people(("PERSON_A",) + other)
        frames = [SceneFrame({"TARGET": box_at(500, 400, 40, 100)}) for _ in range(10)]
        frames += [SceneFrame({}) for _ in range(40)]
        frames += [SceneFrame({"TARGET": box_at(500, 400, 40, 100)}) for _ in range(40)]
        frames += [SceneFrame({"TARGET": box_at(500, 400, 40, 100), "PERSON_A": box_at(580, 400, 40, 100)})
                   for _ in range(40)]
        return run(frames, ppl)

    def _unverified_frames_ok(self, r, frames):
        ceiling = CFG.get("untrusted_identity_confidence_ceiling")
        for k in frames:
            f = r.frames[k]
            if f.rb_state == T.TRACKING:
                self.assertFalse(f.identity_trusted, f"rel {k}")
                self.assertEqual(f.identity_uncertainty_reasons, (T.INSUFFICIENT_CONTRAST,), f"rel {k}")
                self.assertLessEqual(f.identity_confidence_rb, ceiling, f"rel {k}")
                bank = f.decision.get("bank") or {}
                self.assertNotEqual(bank.get("status"), "ACCEPTED", f"rel {k}: bank updated from unverified identity")

    def test_same_kit_contrast_verifies_the_episode(self):
        r = self._return_alone_then((KIT_A, SAME_KIT_COS))
        ev = r.tracker.reacquisition_events[0]
        self.assertFalse(ev.evidence["contrast_available"])
        acc = ev.reacq_accepted_frame
        verified = [k for k in range(acc, len(r.frames)) if r.frames[k].identity_trusted]
        self.assertTrue(verified, "a same-kit competitor the target beats by the margin must verify the episode")
        first = min(verified)
        self.assertTrue(r.frames[first].decision.get("episode_contrast_verified"))
        self._unverified_frames_ok(r, range(acc, first))
        self.assertGreater(r.frames[first].identity_confidence_rb, 1.0 / 2)
        self.assertTrue(all(r.frames[k].identity_trusted for k in range(first, len(r.frames))
                            if r.frames[k].rb_state == T.TRACKING))

    def test_contrast_against_an_obviously_different_person_does_not_verify(self):
        r = self._return_alone_then((KIT_B, 0.0))
        acc = r.tracker.reacquisition_events[0].reacq_accepted_frame
        self.assertEqual([k for k in range(acc, len(r.frames)) if r.frames[k].identity_trusted], [])
        self._unverified_frames_ok(r, range(acc, len(r.frames)))

    def test_wrong_lookalike_is_never_trusted(self):
        for cos in (0.78, 0.8):
            with self.subTest(cos=cos):
                ppl = people(("PERSON_A", KIT_A, cos))
                frames = [SceneFrame({"TARGET": b}) for b in linear(40, (1150, 400), (8, 0))]
                frames += [SceneFrame({"PERSON_A": b}) for b in linear(120, (1250, 400), (-4, 0))]
                r = run(frames, ppl)
                self.assertTrue(r.tracker.reacquisition_events)
                acc = r.tracker.reacquisition_events[0].reacq_accepted_frame
                self.assertEqual([k for k in range(acc, len(r.frames)) if r.frames[k].identity_trusted], [])
                self._unverified_frames_ok(r, range(acc, len(r.frames)))
                self.assertTrue(r.bank_is_pure())

    def test_unverified_episode_rules_nobody_out(self):
        """A body seen beside an unverified identity is not recorded as a known other: if the identity is wrong,
        that body may be the real target and must stay reacquirable."""
        ppl = people(("PERSON_A", KIT_B, 0.0))
        frames = [SceneFrame({"TARGET": box_at(500, 400, 40, 100)}) for _ in range(10)]
        frames += [SceneFrame({}) for _ in range(40)]
        frames += [SceneFrame({"TARGET": box_at(500, 400, 40, 100), "PERSON_A": box_at(620, 400, 40, 100)})
                   for _ in range(60)]
        r = run(frames, ppl)
        acc = r.tracker.reacquisition_events[0].reacq_accepted_frame
        self.assertFalse(any(r.frames[k].identity_trusted for k in range(acc, len(r.frames))))
        self.assertEqual(r.tracker.state_sizes()["target_known_others"], 0)

    def test_seed_episode_is_trusted(self):
        r = run([SceneFrame({"TARGET": b}) for b in linear(20, (300, 400), (3, 0))], people())
        self.assertTrue(all(f.identity_trusted and f.identity_confidence_rb > 1.0 / 2 for f in r.frames[1:]))


def _bank_logs(r):
    return [(f, f.decision.get("bank")) for f in r.frames[1:] if f.decision.get("bank")]


class ProbeBankRules(unittest.TestCase):
    def test_sample_interval_respected(self):
        r = run([SceneFrame({"TARGET": b}) for b in linear(60, (300, 400), (3, 0))], people())
        reached = [f.time_s for f, b in _bank_logs(r) if "SAMPLE_NOT_DUE" not in b["refusals"]]
        self.assertGreater(len(reached), 2)
        for a, b in zip(reached, reached[1:]):
            self.assertGreaterEqual(b - a, CFG.get("target_bank_sample_interval_s") - 1e-9)

    def test_edge_touching_box_refused_by_target_rule(self):
        frames = [SceneFrame({"TARGET": b}) for b in linear(60, (1200, 400), (2, 0))]
        r = run(frames, people())
        edge = [(f, b) for f, b in _bank_logs(r) if f.rb_state == T.TRACKING
                and P.TOUCHES_IMAGE_EDGE in r.pops[f.rel_frame].observations[f.detection_index].flags]
        self.assertTrue(edge)
        for f, b in edge:
            self.assertIn(P.TOUCHES_IMAGE_EDGE, b["refusals"], f.rel_frame)

    def test_unverified_camera_frame_refused(self):
        from player_tracker_rb.camera import UNKNOWN, identity_motion
        cfg = TrackerConfig.with_overrides({"camera_unknown_breaks_continuity": False})
        unk = identity_motion(UNKNOWN, reasons=("TOO_FEW_FEATURES",), continuity_broken=False)
        frames = [SceneFrame({"TARGET": b}, camera=unk if k >= 10 else None)
                  for k, b in enumerate(linear(40, (300, 400), (3, 0)))]
        r = run(frames, people(), cfg=cfg)
        offered = [(f, b) for f, b in _bank_logs(r) if f.rel_frame >= 10]
        self.assertTrue(offered)
        for f, b in offered:
            self.assertIn("CAMERA_UNVERIFIED", b["refusals"], f.rel_frame)


class ProbeRivalContinuation(unittest.TestCase):
    def test_unbanked_identical_neighbour_blocks_continuation(self):
        """An identical-looking person with no bank yet (edge crop) appears 50 px beside the tracked target, inside
        the motion bound but not in contact: the target may not stay TRACKING (ambiguity, D08-5)."""
        ppl = people(("TWIN", KIT_A, 1.0))
        frames = [SceneFrame({"TARGET": box_at(70, 400, 40, 100)}) for _ in range(15)]
        frames += [SceneFrame({"TARGET": box_at(70, 400, 40, 100), "TWIN": box_at(20, 400, 40, 100)})
                   for _ in range(10)]
        r = run(frames, ppl)
        self.assertEqual([k for k in r.tracking_frames() if k >= 15], [])


if __name__ == "__main__":
    unittest.main()

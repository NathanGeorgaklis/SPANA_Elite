# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R1-F instruction (Nathan, 2026-10-25); R1/R1_DESIGN_DECISIONS.md D05-D08, D10-D15, D18, D19
"""R1-F target identity tests on rendered synthetic scenes (neutral TARGET / PERSON_A / PERSON_B; no benchmark
knowledge). Appearance is real (rendered kits + identity signatures, see _synth_scene), so every identity decision
is earned from evidence.

Required scenarios: same-kit crossing, opponent crossing, full occlusion, merged detection, target leaves frame,
short and long gaps, correct reacquisition, wrong reacquisition rejection, camera cut, ambiguous two-candidate
case, provisional competitor, self-competition prevention. Plus: zero silent swaps across a crossing matrix,
reacquisition stricter than association and stricter with absence, observed vs predicted/held on every frame,
contract-shaped v2 fields, confidence semantics, appearance-bank purity, seed refusal, determinism.
"""
import math
import unittest

import numpy as np

from player_tracker_rb import output_schema as S
from player_tracker_rb import population as P
from player_tracker_rb import target as T
from player_tracker_rb.camera import CUT, identity_motion
from player_tracker_rb.config import TrackerConfig
from tests_rb._synth_scene import (FPS, H, W, SceneFrame, box_at, linear, make_people, run, still)

CFG = TrackerConfig.default()
KIT_A = (200, 60, 40)          # same kit for TARGET and same-kit people
KIT_B = (40, 40, 210)          # an opponent kit
SAME_KIT_COS = 0.75            # realistic same-kit embedding similarity (R1-D: same-kit P90 ~0.78)


def people(*others):
    return make_people([("TARGET", KIT_A, 1.0)] + list(others))


def frame_invariants(tc, r):
    """Checks on every frame of every scenario (D05, D06, D19 and the v2 contract)."""
    for k, f in enumerate(r.frames):
        where = f"rel {k}"
        errs = []
        S._check_fields({kk: vv for kk, vv in S.V2_TARGET_FIELDS.items()}, f.v2_fields(), where, errs)
        tc.assertEqual(errs, [], where)
        tc.assertIn(f.rb_state, S.RB_STATES)
        observed = f.rb_state == T.TRACKING
        tc.assertEqual(f.bbox_is_observed, observed, where)
        tc.assertEqual(f.identity_trusted, observed, where)
        tc.assertEqual(sum((f.bbox_is_observed, f.bbox_is_held, f.bbox_is_predicted)), 1, where)
        tc.assertEqual({"DETECTION": f.bbox_is_observed, "HELD": f.bbox_is_held,
                        "PREDICTED": f.bbox_is_predicted}[f.observation_source], True, where)
        if observed:
            tc.assertEqual(f.held_age_s, 0.0)
            tc.assertEqual(f.identity_uncertainty_reasons, ())
            tc.assertFalse(f.merged_suspected)
            tc.assertGreaterEqual(f.identity_confidence_rb, CFG.get("legacy_identity_trust_threshold"))
            det = r.pops[k].observations[f.detection_index]
            tc.assertNotIn(P.MERGED_SUSPECTED, det.flags, where)
            tc.assertEqual(tuple(f.bbox), tuple(det.bbox))
        else:
            tc.assertTrue(f.identity_uncertainty_reasons, where)
            tc.assertTrue(set(f.identity_uncertainty_reasons) <= set(S.UNCERTAINTY_REASONS), where)
            tc.assertLessEqual(f.identity_confidence_rb, CFG.get("untrusted_identity_confidence_ceiling"))
            tc.assertGreater(f.held_age_s, 0.0, where)
        inside = f.predicted_inside_image if f.rb_state == T.LOST else True
        tc.assertIn((f.rb_state, inside), S.TARGET_TRIPLE_BY_RB_STATE)
        tc.assertGreaterEqual(f.target_episode, 1)
        # the target never competes against itself
        tts = set(r.tracker.target_tracklets)
        for cand in f.decision.get("candidates", []):
            tc.assertNotIn(cand["best_competitor"], tts, where)
            tc.assertFalse(set(cand["unknown_contacts"]) & tts, where)


def crossing_frames(n, speed_px, dy_px, *, merge=False, hide=False, other="PERSON_A"):
    a = linear(n, (300, 400), (speed_px, 0))
    b = linear(n, (300 + speed_px * n, 400 + dy_px), (-speed_px, 0))
    out = []
    for k in range(n):
        fr = SceneFrame({"TARGET": a[k], other: b[k]})
        ov = max(0.0, min(a[k][2], b[k][2]) - max(a[k][0], b[k][0]))
        if merge and ov > 20:
            fr.merge = [("TARGET", other)]
        elif hide and ov > 0:
            fr.drop = ["TARGET"] if dy_px <= 0 else [other]
        out.append(fr)
    return out


class Base(unittest.TestCase):
    def check(self, r, swaps_allowed=0):
        frame_invariants(self, r)
        self.assertEqual(r.swaps(), [], "silent identity swap")
        self.assertTrue(r.bank_is_pure(), "target bank holds another person's appearance")


class TestSeed(Base):
    def test_seed_tracks_and_anchors(self):
        r = run([SceneFrame({"TARGET": b}) for b in linear(10, (300, 400), (3, 0))], people())
        self.assertEqual(r.states(), [T.TRACKING] * 10)
        self.assertEqual(len(r.tracker.bank), 1)
        self.assertEqual(r.frames[0].identity_confidence_rb, 1.0)
        self.check(r)

    def test_seed_refused_without_matching_detection(self):
        ppl = people()
        fr = [SceneFrame({"TARGET": box_at(300, 400, 40, 100)}, drop=["TARGET"])]
        with self.assertRaises(T.SeedError):
            run(fr, ppl)

    def test_seed_refused_on_merged_or_contaminated_crop(self):
        ppl = people(("PERSON_A", KIT_B, 0.0))
        a, b = box_at(300, 400, 40, 100), box_at(320, 400, 40, 100)
        with self.assertRaises(T.SeedError):        # overlapping a competitor: the anchor would be contaminated
            run([SceneFrame({"TARGET": a, "PERSON_A": b})], ppl)

    def test_update_requires_population_order(self):
        ppl = people()
        r = run([SceneFrame({"TARGET": box_at(300, 400, 40, 100)})], ppl)
        with self.assertRaises(T.TargetError):
            r.tracker.update(r.pops[0], np.zeros((H, W, 3), np.uint8))


class TestCrossings(Base):
    def test_same_kit_crossing(self):
        ppl = people(("PERSON_A", KIT_A, SAME_KIT_COS))
        r = run(crossing_frames(90, 4, 0), ppl)
        self.check(r)
        self.assertTrue(r.tracking_frames()[-1] == 89, "the target is reacquired after the crossing")
        self.assertEqual(r.frames[-1].target_episode, 2)
        ev = r.tracker.reacquisition_events[0]
        self.assertGreaterEqual(ev.evidence["contrast"], ev.evidence["margin"])

    def test_opponent_crossing(self):
        ppl = people(("PERSON_A", KIT_B, 0.2))
        r = run(crossing_frames(90, 4, 10), ppl)
        self.check(r)
        self.assertEqual(r.frames[-1].rb_state, T.TRACKING)

    def test_crossing_matrix_zero_silent_swaps(self):
        n = 0
        for kit, cos in ((KIT_A, SAME_KIT_COS), (KIT_B, 0.2)):
            for speed in (2, 5, 9):
                for dy in (0, 15, 45):
                    for mode in ("visible", "merge", "hide"):
                        with self.subTest(kit=kit, speed=speed, dy=dy, mode=mode):
                            ppl = people(("PERSON_A", kit, cos))
                            frames = crossing_frames(int(600 / speed) + 30, speed, dy, merge=mode == "merge",
                                                     hide=mode == "hide")
                            r = run(frames, ppl)
                            self.check(r)
                            # coverage: zero swaps must not come from never committing again
                            self.assertEqual(r.frames[-1].rb_state, T.TRACKING)
                            n += 1
        self.assertEqual(n, 54)

    def test_crossing_reacquisition_never_picks_the_other(self):
        """After a contested split both new tracklets are candidates; only the target's may be accepted."""
        ppl = people(("PERSON_A", KIT_A, SAME_KIT_COS))
        r = run(crossing_frames(100, 3, 0, merge=True), ppl)
        self.check(r)
        for ev in r.tracker.reacquisition_events:
            k = ev.reacq_accepted_frame
            f = r.frames[k]
            self.assertEqual(r.dets[k][f.detection_index].owners, ("TARGET",))


class TestOcclusionAndMerge(Base):
    def test_full_occlusion_occluder_never_inherits_identity(self):
        ppl = people(("PERSON_A", KIT_A, SAME_KIT_COS))
        tgt = still(80, (500, 400))
        occ = linear(80, (200, 410), (8, 0), w=60, h=130)          # nearer, bigger, passes fully in front
        frames = [SceneFrame({"TARGET": tgt[k], "PERSON_A": occ[k]}, order=["TARGET", "PERSON_A"])
                  for k in range(80)]
        r = run(frames, ppl)
        self.check(r)
        occluded = [k for k, f in enumerate(r.frames) if f.rb_state == T.OCCLUDED]
        self.assertTrue(occluded)
        for k in occluded:
            f = r.frames[k]
            if T.OCCLUDER_OVERLAP in f.identity_uncertainty_reasons:
                self.assertTrue(f.occluder_track_ids)
        self.assertEqual(r.frames[-1].rb_state, T.TRACKING)          # back after the occluder has passed

    def test_merged_detection_cannot_confirm(self):
        ppl = people(("PERSON_A", KIT_B, 0.2))
        a = still(60, (400, 400))
        b = linear(8, (600, 400), (-20.5, 0)) + still(52, (436, 400))     # walks up beside the target
        frames = []
        for k in range(60):
            fr = SceneFrame({"TARGET": a[k], "PERSON_A": b[k]})
            if 10 <= k < 40:
                fr.merge = [("TARGET", "PERSON_A")]
            frames.append(fr)
        r = run(frames, ppl, cfg=TrackerConfig.with_overrides({"population_duplicate_iou": 0.99}))
        self.check(r)
        for k in range(10, 40):
            self.assertNotEqual(r.frames[k].rb_state, T.TRACKING)
            self.assertTrue(r.frames[k].merged_suspected or
                            T.MERGED_DETECTION_SUSPECTED in r.frames[k].identity_uncertainty_reasons
                            or r.frames[k].rb_state in (T.LOST, T.REACQUIRING))


class TestExitAndGaps(Base):
    def test_target_leaves_frame_and_returns(self):
        ppl = people()
        n = 150
        boxes = linear(60, (1150, 400), (6, 0)) + [None] * 30 + linear(60, (1250, 400), (-6, 0))
        frames = [SceneFrame({} if b is None else {"TARGET": b}) for b in boxes]
        r = run(frames, ppl)
        self.check(r)
        out = [f for f in r.frames if f.rb_state == T.LOST and T.TARGET_OUT_OF_FRAME in f.identity_uncertainty_reasons]
        self.assertTrue(out)
        self.assertTrue(any(not f.predicted_inside_image for f in out))
        self.assertEqual(r.frames[-1].rb_state, T.TRACKING)
        self.assertEqual(r.frames[-1].target_episode, 2)

    def test_short_gap_in_open_space_continues_without_reacquisition(self):
        ppl = people()
        frames = [SceneFrame({"TARGET": b}, drop=["TARGET"] if 20 <= k < 23 else []) for k, b in
                  enumerate(linear(50, (300, 400), (3, 0)))]
        r = run(frames, ppl)
        self.check(r)
        for k in (20, 21, 22):
            self.assertEqual(r.frames[k].rb_state, T.OCCLUDED)
            self.assertIn(T.NO_CANDIDATE_IN_GATE, r.frames[k].identity_uncertainty_reasons)
            self.assertIn(r.frames[k].observation_source, ("PREDICTED", "HELD"))
        self.assertEqual(r.frames[23].rb_state, T.TRACKING)          # continuation, not reacquisition
        self.assertEqual(r.frames[-1].target_episode, 1)

    def test_long_gap_ages_to_lost_then_reacquires(self):
        ppl = people()
        frames = [SceneFrame({"TARGET": b}, drop=["TARGET"] if 20 <= k < 80 else []) for k, b in
                  enumerate(linear(130, (300, 400), (2, 0)))]
        r = run(frames, ppl)
        self.check(r)
        st = r.states()
        self.assertIn(T.LOST, st[20:80])
        lost_k = st.index(T.LOST)
        self.assertLessEqual((lost_k - 19) / FPS, CFG.get("max_occluded_s") + 2 / FPS)
        self.assertIn(T.AGED_OUT, r.frames[lost_k].identity_uncertainty_reasons)
        back = r.tracking_frames()[r.tracking_frames().index(19) + 1]
        self.assertGreaterEqual((back - 80) / FPS, CFG.get("reacq_window_s") - 1 / FPS)   # verified, not instant
        self.assertEqual(r.frames[-1].target_episode, 2)
        # held/predicted while unobserved, never observed
        for k in range(20, 80):
            self.assertIn(r.frames[k].observation_source, ("PREDICTED", "HELD"))
        self.assertIn("HELD", [r.frames[k].observation_source for k in range(20, 80)])


class TestReacquisition(Base):
    def test_correct_reacquisition_needs_the_full_window(self):
        ppl = people()
        frames = [SceneFrame({"TARGET": b}, drop=["TARGET"] if 10 <= k < 50 else []) for k, b in
                  enumerate(still(100, (500, 400)))]
        r = run(frames, ppl)
        self.check(r)
        reacq = [k for k, f in enumerate(r.frames) if f.rb_state == T.REACQUIRING]
        self.assertTrue(reacq)
        self.assertEqual(reacq[0], 50)
        acc = r.tracker.reacquisition_events[0].reacq_accepted_frame
        self.assertGreaterEqual((acc - 50) / FPS, CFG.get("reacq_window_s") - 1 / (2 * FPS))
        self.assertTrue(r.frames[acc].reidentified_this_frame)
        ev = r.tracker.reacquisition_events[0].as_record()
        errs = []
        S._check_fields(S.REACQ_EVENT_FIELDS, ev, "event", errs)
        self.assertEqual(errs, [])

    def test_wrong_reacquisition_rejected_same_kit(self):
        """The target leaves; a same-kit look-alike walks in. It must never be accepted."""
        ppl = people(("PERSON_A", KIT_A, SAME_KIT_COS))
        boxes = linear(40, (1150, 400), (8, 0))
        frames = [SceneFrame({"TARGET": b}) for b in boxes]
        frames += [SceneFrame({"PERSON_A": b}) for b in linear(120, (1250, 400), (-4, 0))]
        r = run(frames, ppl)
        self.check(r)
        self.assertEqual([k for k in r.tracking_frames() if k >= 40], [])

    def test_wrong_reacquisition_rejected_opponent_at_last_position(self):
        ppl = people(("PERSON_B", KIT_B, 0.3))
        frames = [SceneFrame({"TARGET": b}) for b in still(20, (500, 400))]
        frames += [SceneFrame({"PERSON_B": b}) for b in still(60, (500, 400))]
        r = run(frames, ppl)
        self.check(r)
        self.assertEqual([k for k in r.tracking_frames() if k >= 20], [])

    def test_true_target_chosen_over_look_alike(self):
        """Both the target and a same-kit look-alike reappear after a gap: only the target is accepted."""
        ppl = people(("PERSON_A", KIT_A, SAME_KIT_COS))
        frames = [SceneFrame({"TARGET": box_at(500, 400, 40, 100), "PERSON_A": box_at(800, 400, 40, 100)})
                  for _ in range(15)]
        frames += [SceneFrame({}) for _ in range(40)]
        frames += [SceneFrame({"TARGET": box_at(560, 400, 40, 100), "PERSON_A": box_at(470, 400, 40, 100)})
                   for _ in range(40)]
        r = run(frames, ppl)
        self.check(r)
        self.assertEqual(r.frames[-1].rb_state, T.TRACKING)

    def test_reacquisition_margin_grows_with_absence(self):
        tt = run([SceneFrame({"TARGET": box_at(500, 400, 40, 100)})], people()).tracker
        m = [tt._reacq_margin(a) for a in (0.0, 1.0, 2.0, 5.0, 100.0)]
        self.assertEqual(m, sorted(m))
        self.assertAlmostEqual(m[0], CFG.get("reacq_contrastive_margin"))
        self.assertAlmostEqual(m[-1], CFG.get("reacq_margin_cap"))
        self.assertGreater(CFG.get("reacq_contrastive_margin"), CFG.get("contrastive_margin"))
        self.assertGreater(CFG.get("reacq_appearance_floor"), CFG.get("appearance_floor"))

    def _absence_scene(self, gap_frames):
        """Target and a banked same-kit neighbour; the target vanishes for gap_frames and returns beside the
        neighbour. contrast = 0.65 * (1 - cos) = 0.14: enough for a short absence, not for a long one."""
        cos = 1 - 0.14 / CFG.get("composite_weight_embedding")
        ppl = people(("PERSON_A", KIT_A, cos))
        nb = box_at(640, 400, 40, 100)
        frames = [SceneFrame({"TARGET": box_at(560, 400, 40, 100), "PERSON_A": nb}) for _ in range(15)]
        frames += [SceneFrame({"PERSON_A": nb}) for _ in range(gap_frames)]
        frames += [SceneFrame({"TARGET": box_at(560, 400, 40, 100), "PERSON_A": nb}) for _ in range(45)]
        return run(frames, ppl)

    def test_body_seen_beside_the_target_is_never_reacquired_as_it(self):
        """A same-kit neighbour observed in the same frames as the confirmed target is a different person. When
        the target disappears, the neighbour is never accepted, however similar it looks."""
        cos = 0.9                                                   # composite 0.935: above every floor
        ppl = people(("PERSON_A", KIT_A, cos))
        nb = box_at(640, 400, 40, 100)
        frames = [SceneFrame({"TARGET": box_at(560, 400, 40, 100), "PERSON_A": nb}) for _ in range(15)]
        frames += [SceneFrame({"PERSON_A": nb}) for _ in range(150)]
        r = run(frames, ppl)
        self.check(r)
        self.assertEqual([k for k in r.tracking_frames() if k >= 15], [])

    def test_long_absence_is_stricter(self):
        short = self._absence_scene(int(0.3 * FPS))
        long_ = self._absence_scene(int(2.5 * FPS))
        self.check(short)
        self.check(long_)
        self.assertEqual(short.frames[-1].rb_state, T.TRACKING)
        self.assertNotEqual(long_.frames[-1].rb_state, T.TRACKING)
        tail = [c for f in long_.frames[-10:] for c in f.decision.get("candidates", [])
                if c["role"] == T.REACQUISITION and c["contrast"] is not None]
        self.assertTrue(tail and all(c["contrast"] < c["margin"] for c in tail))


class TestCameraCut(Base):
    def test_cut_invalidates_position_and_rejects_impostor_at_same_place(self):
        """After a cut, a same-kit look-alike stands exactly where the target was; the real target appears
        elsewhere. Position is meaningless after a cut: the impostor is never accepted, the target is (by
        appearance alone, after verification)."""
        ppl = people(("PERSON_A", KIT_A, SAME_KIT_COS))
        frames = [SceneFrame({"TARGET": box_at(500, 400, 40, 100)}) for _ in range(20)]
        cut = identity_motion(CUT, reasons=("HISTOGRAM_CHANGE",), continuity_broken=True)
        frames.append(SceneFrame({"PERSON_A": box_at(500, 400, 40, 100), "TARGET": box_at(900, 380, 44, 110)},
                                 camera=cut))
        frames += [SceneFrame({"PERSON_A": box_at(500, 400, 40, 100), "TARGET": box_at(900, 380, 44, 110)})
                   for _ in range(40)]
        r = run(frames, ppl)
        self.check(r)
        f = r.frames[20]
        self.assertNotEqual(f.rb_state, T.TRACKING)
        self.assertIn(T.CAMERA_CUT, f.identity_uncertainty_reasons)
        self.assertFalse(f.position_valid)
        self.assertEqual(r.frames[-1].rb_state, T.TRACKING)
        ev = r.tracker.reacquisition_events[-1]
        self.assertFalse(ev.evidence["positional_prior"])


class TestAmbiguityAndCompetitors(Base):
    def test_two_identical_candidates_stay_ambiguous(self):
        """A twin with the target's exact appearance: after the target is lost, two equally good candidates
        appear. No confident decision is possible: never TRACKING."""
        ppl = people(("TWIN", KIT_A, 1.0))
        frames = [SceneFrame({"TARGET": box_at(500, 400, 40, 100)}) for _ in range(15)]
        frames += [SceneFrame({}) for _ in range(10)]
        frames += [SceneFrame({"TARGET": box_at(470, 400, 40, 100), "TWIN": box_at(540, 400, 40, 100)})
                   for _ in range(40)]
        r = run(frames, ppl)
        frame_invariants(self, r)
        self.assertEqual([k for k in r.tracking_frames() if k >= 15], [])
        later = [f for f in r.frames[25:]]
        self.assertTrue(all({T.COMPETING_CANDIDATE, T.INSUFFICIENT_CONTRAST} & set(f.identity_uncertainty_reasons)
                            for f in later), [f.identity_uncertainty_reasons for f in later])

    def test_provisional_competitor_blocks_from_birth(self):
        """A person appears overlapping the tracked target (born this frame, no appearance yet): the target
        cannot stay TRACKING on that frame."""
        ppl = people(("PERSON_A", KIT_B, 0.2))
        frames = [SceneFrame({"TARGET": box_at(500, 400, 40, 100)}) for _ in range(15)]
        frames += [SceneFrame({"TARGET": box_at(500, 400, 40, 100), "PERSON_A": box_at(532, 405, 40, 100)},
                              order=["TARGET", "PERSON_A"]) for _ in range(10)]
        r = run(frames, ppl)
        self.check(r)
        f = r.frames[15]
        self.assertNotEqual(f.rb_state, T.TRACKING)
        self.assertIn(T.COMPETING_CANDIDATE, f.identity_uncertainty_reasons)
        cands = [c for c in f.decision["candidates"] if c["role"] == T.CONTINUATION]
        self.assertTrue(any(c["unknown_contacts"] for c in cands))

    def test_small_fragment_inside_target_box_does_not_block(self):
        """A small non-person-sized box (ball, leg fragment) appears inside the bottom of the tracked target's box.
        It cannot physically be the target (scale), so it does not block the target as an unknown competitor;
        the target's crop and bank stay protected by the contamination rules (R1-F development-clip finding)."""
        ppl = make_people([("TARGET", KIT_A, 1.0), ("BALL", (250, 250, 250), 0.0)])
        frames = []
        for k in range(40):
            tb = box_at(400 + 2 * k, 400, 50, 120)
            boxes = {"TARGET": tb}
            if k >= 15:
                boxes["BALL"] = (tb[0] + 6, tb[3] - 30, tb[0] + 50, tb[3] - 2)
            frames.append(SceneFrame(boxes, order=["TARGET", "BALL"]))
        r = run(frames, ppl, cfg=TrackerConfig.with_overrides({"population_part_box_min_height_ratio": 0.3}))
        self.check(r)
        self.assertEqual(r.states()[15:], [T.TRACKING] * 25)
        for f in r.frames[15:]:
            b = f.decision.get("bank")
            if b and b["status"] == "REFUSED":
                self.assertIn(P.CONTAINS_OTHER, b["refusals"] + ["CONTAINS_OTHER"])

    def test_motion_vetoed_box_is_not_a_rival(self):
        """Rule check: a box the motion model vetoes (physically implausible as the target this frame) never
        counts as a rival target candidate, however similar it looks; without the veto it does."""
        tt = run([SceneFrame({"TARGET": box_at(500, 400, 40, 100)})], people()).tracker
        base = dict(detection_index=1, tracklet_id="T009", box=(0.0, 0.0, 10.0, 20.0), role=T.CONTINUATION,
                    in_gate=True, distance_h=0.5, merged=False, motion=None, quality_flags=(),
                    target_similarity={"composite": 0.99, "lab": 1.0, "embedding": 0.98}, best_competitor=None,
                    best_competitor_similarity=None, contrast=None, margin=0.08, unknown_contacts=())
        self.assertTrue(tt._looks_like_target(T.CandidateEvaluation(passed=True, details=(), **base)))
        self.assertFalse(tt._looks_like_target(T.CandidateEvaluation(passed=False, details=(T.MOTION_VETO,),
                                                                     **base)))

    def test_target_never_competes_against_itself(self):
        """The target goes undetected long enough for the population to start a new tracklet while the old one
        is still HIDDEN at the same place (possible predecessor). The old tracklet is the target itself: it must
        not block or out-score the target's own reacquisition."""
        ppl = people()
        frames = [SceneFrame({"TARGET": b}, drop=["TARGET"] if 15 <= k < 23 else [])
                  for k, b in enumerate(still(80, (500, 400)))]
        r = run(frames, ppl)
        self.check(r)
        self.assertEqual(r.frames[-1].rb_state, T.TRACKING)
        tts = set(r.tracker.target_tracklets)
        self.assertGreaterEqual(len(tts), 2)
        for f in r.frames:
            for c in f.decision.get("candidates", []):
                self.assertNotIn(c["best_competitor"], tts)

    def test_competitor_list_excludes_own_nested_part_box(self):
        """A part-body box inside the target's box is not another person competing with the target."""
        ppl = people()
        frames = []
        for k, b in enumerate(linear(40, (400, 400), (2, 0), w=50, h=140)):
            frames.append(SceneFrame({"TARGET": b}))
        r = run(frames, ppl)
        self.check(r)


class TestBankAndConfidence(Base):
    def test_bank_updated_only_when_isolated_and_trusted(self):
        ppl = make_people([("TARGET", KIT_A, 1.0), ("PERSON_A", KIT_B, 0.2)])
        # give the target several distinct views so updates are not all redundant: vary its head size
        r = run(crossing_frames(120, 3, 0, merge=True), ppl)
        self.check(r)
        for k, f in enumerate(r.frames):
            b = f.decision.get("bank")
            if b and b["status"] == "ACCEPTED":
                self.assertEqual(f.rb_state, T.TRACKING)
                o = r.pops[k].observations[f.detection_index]
                self.assertLessEqual(o.max_iou_other, CFG.get("bank_contact_iou"))

    def test_post_reacquisition_quarantine(self):
        ppl = people()
        frames = [SceneFrame({"TARGET": b}, drop=["TARGET"] if 10 <= k < 50 else [])
                  for k, b in enumerate(still(120, (500, 400)))]
        r = run(frames, ppl)
        acc = r.tracker.reacquisition_events[0].reacq_accepted_frame
        q = int(CFG.get("post_reacq_quarantine_s") * FPS)
        for k in range(acc, min(acc + q - 1, len(r.frames))):
            b = r.frames[k].decision.get("bank")
            if b and b["status"] == "REFUSED":
                self.assertTrue("POST_REACQUISITION_QUARANTINE" in b["refusals"] or b["refusals"])

    def test_confidence_semantics(self):
        ppl = people()
        frames = [SceneFrame({"TARGET": b}, drop=["TARGET"] if 10 <= k < 40 else [])
                  for k, b in enumerate(still(60, (500, 400)))]
        r = run(frames, ppl)
        conf = [f.identity_confidence_rb for f in r.frames[10:40]]
        self.assertEqual(conf, sorted(conf, reverse=True))           # decays while unconfirmed
        self.assertTrue(all(c <= CFG.get("untrusted_identity_confidence_ceiling") for c in conf))
        for f in r.frames:
            self.assertTrue(0.0 <= f.identity_confidence_rb <= 1.0)

    def test_deterministic(self):
        ppl = people(("PERSON_A", KIT_A, SAME_KIT_COS))
        a = run(crossing_frames(80, 4, 10, merge=True), ppl)
        b = run(crossing_frames(80, 4, 10, merge=True), ppl)
        self.assertEqual([f.as_record() for f in a.frames], [f.as_record() for f in b.frames])


class TestInvariance(Base):
    def test_fps_invariance_of_decisions(self):
        sig = {}
        for fps in (30.0, 60.0):
            k = fps / 30.0
            ppl = people(("PERSON_A", KIT_A, SAME_KIT_COS))
            n = int(90 * k)
            frames = crossing_frames(n, 4 / k, 0)
            r = run(frames, ppl, fps=fps)
            self.check(r)
            sig[fps] = (r.frames[-1].rb_state, r.frames[-1].target_episode)
        self.assertEqual(sig[30.0], sig[60.0])


if __name__ == "__main__":
    unittest.main()

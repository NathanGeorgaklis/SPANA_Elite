# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2 DEV F2 instruction (Nathan, 2026-11-01 17:38): allow safe initialisation when the correct seed
#                 player is detected but the crop is partial, near the image edge, or has nested/alternative
#                 detections; the seed must still fail closed if physical identity is ambiguous; no contaminated seed
#                 bank; test image-edge player, partial body, duplicate/nested detection, genuine second person nearby,
#                 ambiguous seed.
"""F2: deferred seed anchor (synthetic, neutral TARGET / OTHER; no benchmark knowledge).

DEV failure reproduced: the seed body was the only, unambiguous match for the human seed box, but its seed-frame crop
failed the ordinary bank rules for geometry only (partial / image edge / nested part boxes / light overlap), so the
whole run was refused although a clean crop of the same tracklet followed a fraction of a second later.
"""
import unittest

from player_tracker_rb import camera as C
from player_tracker_rb import population as P
from player_tracker_rb.appearance import AppearanceExtractor
from player_tracker_rb.detector import PersonCandidate
from player_tracker_rb import target as T
from player_tracker_rb.config import TrackerConfig
from tests_rb import _synth_scene as SS

CFG = TrackerConfig.default()
KIT_A, KIT_B = (200, 60, 40), (40, 60, 200)
H = 100.0
FPS_SET = (60.0, 30.0, 15.0, 10.0)
ANCHOR_S = CFG.get("seed_anchor_max_s")


def people(*others):
    return SS.make_people([("TARGET", KIT_A, 1.0)] + list(others))


def wrong_trusted(r):
    return [k for k, f in enumerate(r.frames) if f.identity_trusted
            and r.dets[k][f.detection_index].owners != ("TARGET",)]


def bank_only_when_trusted(tc, r):
    for f in r.frames:
        if (f.decision.get("bank") or {}).get("status") == "ACCEPTED":
            tc.assertTrue(f.identity_trusted)


def refused(tc, fn, *needles):
    """fn raises SeedError whose message contains every needle (the reason, not just any refusal)."""
    with tc.assertRaises(T.SeedError) as cm:
        fn()
    for n in needles:
        tc.assertIn(n, str(cm.exception))


def check_anchor(tc, r, k):
    """The anchor frame is on the target, trusted, at the documented confidence (< 1: not the human seed itself)."""
    f = r.frames[k]
    tc.assertEqual(r.dets[k][f.detection_index].owners, ("TARGET",))
    tc.assertTrue(f.identity_trusted)
    s = f.decision["seed_anchor"]["seed_similarity"]["composite"]
    fl = CFG.get("reacq_appearance_floor")
    tc.assertAlmostEqual(f.identity_confidence_rb, 0.5 + 0.5 * min(1.0, max(0.0, (s - fl) / (1 - fl))), places=9)


def anchored_at(r):
    for k, f in enumerate(r.frames):
        if (f.decision.get("seed_anchor") or {}).get("status") == "ANCHORED":
            return k
    return None


def walk_in_from_left(fps, *, clear_after_s=0.4, total_s=3.0):
    """The target starts with its box cut by the left image edge and walks fully into view after clear_after_s."""
    n = int(total_s * fps)
    x0, x1 = 5.0, 5.0 + 60.0
    return [SS.SceneFrame({"TARGET": SS.box_at(x0 + (x1 - x0) * min(1.0, k / (clear_after_s * fps)), 400, 40, H)})
            for k in range(n)]


class TestDeferredAnchor(unittest.TestCase):
    def test_image_edge_player_is_anchored_once_clear(self):
        # FAILS ON b293a28 (SeedError: PARTIAL / TOUCHES_IMAGE_EDGE)
        for fps in FPS_SET:
            with self.subTest(fps=fps):
                fr = walk_in_from_left(fps)
                r = SS.run(fr, people(), fps=fps)
                self.assertFalse(r.frames[0].identity_trusted)
                self.assertEqual(r.frames[0].identity_uncertainty_reasons, (T.LOW_QUALITY_CROP,))
                self.assertEqual(r.frames[0].decision["seed_anchor"]["status"], "PENDING")
                k = anchored_at(r)
                self.assertIsNotNone(k)
                self.assertLessEqual(k / fps, ANCHOR_S + 1e-9)
                self.assertFalse(any(f.identity_trusted for f in r.frames[:k]), "untrusted until anchored")
                self.assertTrue(all(f.identity_trusted for f in r.frames[k:]))
                check_anchor(self, r, k)
                self.assertEqual(wrong_trusted(r), [])
                self.assertTrue(r.bank_is_pure())
                bank_only_when_trusted(self, r)
                self.assertGreater(len(r.tracker.bank.samples), 0)

    def test_partial_body_cut_by_the_bottom_edge_is_anchored_once_whole(self):
        for fps in FPS_SET:
            with self.subTest(fps=fps):
                n, m = int(3 * fps), int(0.3 * fps)
                fr = [SS.SceneFrame({"TARGET": SS.box_at(500, SS.H - 30 - 60 * min(1.0, k / max(1, m)), 40, H)})
                      for k in range(n)]
                r = SS.run(fr, people(), fps=fps)
                k = anchored_at(r)
                self.assertIsNotNone(k)
                self.assertFalse(any(f.identity_trusted for f in r.frames[:k]))
                self.assertTrue(r.frames[-1].identity_trusted)
                self.assertTrue(r.bank_is_pure())

    def test_seed_on_a_nested_part_box_without_its_own_track_is_refused(self):
        # the human seed box matches an upper-body part box better than the full body box: the part box has no track
        # and a part box cannot be told from a second person inside a larger box (review HIGH-1), so it is refused
        for fps in FPS_SET:
            with self.subTest(fps=fps):
                body = SS.box_at(500, 400, 40, H)
                part = (body[0] + 2, body[1] + 2, body[2] - 2, body[1] + 0.7 * H)
                fr = [SS.SceneFrame({"TARGET": body}, parts=[("TARGET", part)]) for _ in range(int(2 * fps))]
                refused(self, lambda: SS.run(fr, people(), fps=fps, seed_box=part), "nested / alternative")

    def test_body_seed_with_its_own_duplicate_part_box_is_seeded_directly(self):
        # the seed matches the full body box; the detector also returns a nested part box of the same person: the
        # part box is in the body's own nested group, so the seed crop is bank-clean and nothing is deferred
        for fps in FPS_SET:
            with self.subTest(fps=fps):
                body = SS.box_at(500, 400, 40, H)
                part = (body[0] + 2, body[1] + 2, body[2] - 2, body[1] + 0.7 * H)
                fr = [SS.SceneFrame({"TARGET": body}, parts=[("TARGET", part)] if k < int(0.2 * fps) + 1 else [])
                      for k in range(int(3 * fps))]
                r = SS.run(fr, people(), fps=fps)
                self.assertTrue(r.frames[0].identity_trusted)
                self.assertNotIn("seed_anchor", r.frames[0].decision)
                self.assertEqual(wrong_trusted(r), [])
                self.assertTrue(r.bank_is_pure())

    def test_anchor_at_exactly_seed_anchor_max_s_is_accepted_and_one_frame_later_refused(self):
        for fps in (30.0, 60.0):
            with self.subTest(fps=fps):
                n = round(ANCHOR_S * fps)
                for clean_at, ok in ((n, True), (n + 1, False)):
                    # partial at the edge until clean_at - 3, then three small steps fully into view at clean_at
                    def xs(k):
                        return 5.0 + 20.0 * min(1.0, max(0.0, (k - (clean_at - 3)) / 3.0))
                    fr = [SS.SceneFrame({"TARGET": SS.box_at(xs(k), 400, 40, H)}) for k in range(clean_at + 20)]
                    if ok:
                        r = SS.run(fr, people(), fps=fps)
                        self.assertEqual(anchored_at(r), clean_at)
                    else:
                        refused(self, lambda: SS.run(fr, people(), fps=fps), "seed_anchor_max_s")

    def test_clean_seed_is_unchanged(self):
        fr = [SS.SceneFrame({"TARGET": SS.box_at(500, 400, 40, H)}) for _ in range(30)]
        r = SS.run(fr, people())
        self.assertTrue(r.frames[0].identity_trusted)
        self.assertNotIn("seed_anchor", r.frames[0].decision)
        self.assertIsNone(anchored_at(r))


def run_with_flag(frames, flag_at, flag, fps=30.0):
    """SS.run, but at frame flag_at the population's observation on the seed body's tracklet also carries `flag`
    (a break the synthetic population rarely produces before it ends the tracklet: tests the second line of defence)."""
    cfg = CFG
    pop = P.Population(cfg, SS.W, SS.H)
    ppl = people()
    tt = T.TargetTracker(cfg, pop, AppearanceExtractor(cfg, SS.SignatureEmbedding(ppl, 0.0)))
    for k, fr in enumerate(frames):
        img = SS.render(fr, ppl, SS.W, SS.H)
        ds = SS.detect(fr, 0.35, SS.W, SS.H)
        cands = [PersonCandidate(d.box, 0.9, d.box[0] <= 2 or d.box[2] >= SS.W - 2, "synthetic") for d in ds]
        cam = SS.identity_motion(C.FIRST_FRAME) if k == 0 else SS.motion_from_matrix([[1, 0, 0], [0, 1, 0]], SS.W, SS.H)
        res = pop.update(k, k / fps, cands, cam)
        if k == flag_at:                                   # in place: the tracker checks it is the population's result
            tid = tt._seed_pending["tracklet"]
            for o in res.observations:
                if o.tracklet_id == tid:
                    object.__setattr__(o, "flags", tuple(o.flags) + (flag,))
        if k == 0:
            tt.seed(res, img, fr.boxes["TARGET"])
        else:
            tt.update(res, img)


class TestFailClosed(unittest.TestCase):
    def test_each_break_flag_while_pending_refuses_the_seed(self):
        # audit: every _SEED_PENDING_BREAK flag (and MERGE_LINEAGE through its own check) refuses the deferred seed
        fr = [SS.SceneFrame({"TARGET": SS.box_at(5, 400, 40, H)}) for _ in range(20)]
        for flag in (P.CONTESTED, P.DUPLICATE_ABSORBED, P.IN_MERGED_REGION, P.MERGED_SUSPECTED, P.SIZE_ANOMALY):
            with self.subTest(flag=flag):
                refused(self, lambda: run_with_flag(fr, 3, flag), "became", flag)

    def test_genuine_second_person_in_contact_refuses_the_seed(self):
        # another player overlaps the target at the seed and stays in contact: no clean crop ever comes, so the seed
        # is refused (fail closed) rather than anchored on a contaminated crop
        for fps in FPS_SET:
            for cos, kit in ((0.75, KIT_A), (0.2, KIT_B)):
                with self.subTest(fps=fps, cos=cos):
                    fr = [SS.SceneFrame({"TARGET": SS.box_at(500, 400, 40, H), "OTHER": SS.box_at(532, 400, 40, H)},
                                        order=["OTHER", "TARGET"]) for _ in range(int(3 * fps))]
                    with self.assertRaises(T.SeedError):         # in contact: refused at the seed or while pending
                        SS.run(fr, people(("OTHER", kit, cos)), fps=fps)

    def test_second_person_overlapping_at_the_seed_then_clear_is_anchored_safely(self):
        for fps in FPS_SET:
            with self.subTest(fps=fps):
                n, m = int(3 * fps), int(0.3 * fps)
                fr = [SS.SceneFrame({"TARGET": SS.box_at(500, 400, 40, H),
                                     "OTHER": SS.box_at(536 + 120 * min(1.0, k / max(1, m)), 400, 40, H)},
                                    order=["TARGET", "OTHER"]) for k in range(n)]
                r = SS.run(fr, people(("OTHER", KIT_A, 0.75)), fps=fps)
                k = anchored_at(r)
                self.assertIsNotNone(k)
                self.assertFalse(any(f.identity_trusted for f in r.frames[:k]))
                check_anchor(self, r, k)
                self.assertEqual(wrong_trusted(r), [])
                self.assertTrue(r.bank_is_pure())
                bank_only_when_trusted(self, r)

    def test_ambiguous_seed_box_holding_two_bodies_is_refused(self):
        # the human box covers the target and another person who is substantially inside it
        fr = [SS.SceneFrame({"TARGET": SS.box_at(5, 400, 40, H), "OTHER": SS.box_at(30, 400, 40, H)},
                            order=["OTHER", "TARGET"]) for _ in range(30)]
        refused(self, lambda: SS.run(fr, people(("OTHER", KIT_A, 0.75)), seed_box=(-15.0, 350.0, 50.0, 450.0)),
                "ambiguous")

    def test_edge_player_that_never_clears_is_refused(self):
        for fps in FPS_SET:
            with self.subTest(fps=fps):
                fr = [SS.SceneFrame({"TARGET": SS.box_at(5, 400, 40, H)}) for _ in range(int((ANCHOR_S + 1) * fps))]
                refused(self, lambda: SS.run(fr, people(), fps=fps), "seed_anchor_max_s")

    def test_cut_while_the_anchor_is_pending_refuses_the_seed(self):
        fps = 30.0
        fr = [SS.SceneFrame({"TARGET": SS.box_at(5, 400, 40, H)}) for _ in range(3)]
        fr += [SS.SceneFrame({"TARGET": SS.box_at(500, 400, 40, H)},
                             camera=SS.identity_motion(C.CUT, continuity_broken=True)) for _ in range(1)]
        fr += [SS.SceneFrame({"TARGET": SS.box_at(500, 400, 40, H)}) for _ in range(20)]
        refused(self, lambda: SS.run(fr, people(), fps=fps), "camera continuity")

    def test_pending_tracklet_that_breaks_refuses_the_seed(self):
        # the edge-cut target is missed for longer than the population gap before it clears: the tracklet ends
        fps = 30.0
        n_gap = int(0.5 * fps)
        fr = [SS.SceneFrame({"TARGET": SS.box_at(5, 400, 40, H)}) for _ in range(3)]
        fr += [SS.SceneFrame({"TARGET": SS.box_at(5, 400, 40, H)}, drop=("TARGET",)) for _ in range(n_gap)]
        fr += [SS.SceneFrame({"TARGET": SS.box_at(200, 400, 40, H)}) for _ in range(20)]
        refused(self, lambda: SS.run(fr, people(), fps=fps), "was not observed")

    def test_host_box_of_another_person_never_carries_the_identity(self):
        # review HIGH-1 (anchored the bank on OTHER at b293a28 + F2 bd1127c): the target's tight box lies nested in a
        # larger box of another person, who then walks away
        for fps in (30.0, 15.0):
            for hw, hh, cos, kit in ((60, 115, 0.75, KIT_A), (64, 120, 0.75, KIT_A), (60, 115, 0.2, KIT_B)):
                with self.subTest(fps=fps, host=(hw, hh), cos=cos):
                    tgt = SS.box_at(500, 400, 40, H)
                    fr = [SS.SceneFrame({"TARGET": tgt, "OTHER": SS.box_at(505 + 4 * k, 400 - (hh - H) / 2 + 2.5,
                                                                           hw, hh)}, order=["OTHER", "TARGET"])
                          for k in range(int(3 * fps))]
                    try:
                        r = SS.run(fr, people(("OTHER", kit, cos)), fps=fps)
                    except T.SeedError:
                        continue
                    self.assertEqual(wrong_trusted(r), [])
                    self.assertTrue(r.bank_is_pure())

    def test_edge_swap_onto_a_same_kit_entrant_is_refused(self):
        # review HIGH-2 (anchored on OTHER at bd1127c): the edge-cut target walks out and a same-kit player enters at
        # the same spot on the next frame; the population carries the track over, the seed-crop check stops it
        for fps in (30.0, 15.0):
            with self.subTest(fps=fps):
                fr = []
                for k in range(int(3 * fps)):
                    boxes = {"TARGET": SS.box_at(5 - 6 * k, 400, 40, H)} if k < 3 else \
                        {"OTHER": SS.box_at(5 + 6 * (k - 3), 400, 40, H)}
                    fr.append(SS.SceneFrame(boxes))
                refused(self, lambda: SS.run(fr, people(("OTHER", KIT_A, 0.75)), fps=fps),
                        "no longer matches the seed crop")

    def test_merge_while_pending_refuses_the_seed(self):
        # the edge-cut target is merged with a newcomer into one detection while the anchor is pending
        fps = 30.0
        fr = [SS.SceneFrame({"TARGET": SS.box_at(5, 400, 40, H)}) for _ in range(3)]
        fr += [SS.SceneFrame({"TARGET": SS.box_at(5, 400, 40, H), "OTHER": SS.box_at(30, 400, 40, H)},
                             merge=[("TARGET", "OTHER")]) for _ in range(10)]
        fr += [SS.SceneFrame({"TARGET": SS.box_at(60, 400, 40, H), "OTHER": SS.box_at(200, 400, 40, H)})
               for _ in range(20)]
        with self.assertRaises(T.SeedError):
            SS.run(fr, people(("OTHER", KIT_A, 0.75)), fps=fps)


if __name__ == "__main__":
    unittest.main()

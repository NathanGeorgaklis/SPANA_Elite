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
from player_tracker_rb import target as T
from player_tracker_rb.config import TrackerConfig
from tests_rb import _synth_scene as SS

CFG = TrackerConfig.default()
KIT_A, KIT_B = (200, 60, 40), (40, 60, 200)
H = 100.0
FPS_SET = (30.0, 15.0, 10.0)
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
                self.assertEqual(wrong_trusted(r), [])
                self.assertTrue(r.bank_is_pure())
                bank_only_when_trusted(self, r)
                self.assertEqual(len(r.tracker.bank.samples) > 0, True)

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

    def test_nested_part_box_seed_is_anchored_on_its_body(self):
        # the detector also returns an upper-body part box of the target; the human seed box matches the part box
        # better than the full body box (as a looser / shorter human box can)
        for fps in FPS_SET:
            with self.subTest(fps=fps):
                body = SS.box_at(500, 400, 40, H)
                part = (body[0] + 2, body[1] + 2, body[2] - 2, body[1] + 0.7 * H)
                n = int(3 * fps)
                fr = [SS.SceneFrame({"TARGET": body}, parts=[("TARGET", part)] if k < int(0.2 * fps) + 1 else [])
                      for k in range(n)]
                r = SS.run(fr, people(), fps=fps, seed_box=part)
                k = anchored_at(r)
                self.assertIsNotNone(k, [f.decision.get("seed_anchor") for f in r.frames[:3]])
                self.assertTrue(r.frames[-1].identity_trusted)
                self.assertEqual(wrong_trusted(r), [])
                self.assertTrue(r.bank_is_pure())

    def test_clean_seed_is_unchanged(self):
        fr = [SS.SceneFrame({"TARGET": SS.box_at(500, 400, 40, H)}) for _ in range(30)]
        r = SS.run(fr, people())
        self.assertTrue(r.frames[0].identity_trusted)
        self.assertNotIn("seed_anchor", r.frames[0].decision)
        self.assertIsNone(anchored_at(r))


class TestFailClosed(unittest.TestCase):
    def test_genuine_second_person_in_contact_refuses_the_seed(self):
        # another player overlaps the target at the seed and stays in contact: no clean crop ever comes, so the seed
        # is refused (fail closed) rather than anchored on a contaminated crop
        for fps in FPS_SET:
            for cos, kit in ((0.75, KIT_A), (0.2, KIT_B)):
                with self.subTest(fps=fps, cos=cos):
                    fr = [SS.SceneFrame({"TARGET": SS.box_at(500, 400, 40, H), "OTHER": SS.box_at(532, 400, 40, H)},
                                        order=["OTHER", "TARGET"]) for _ in range(int(3 * fps))]
                    with self.assertRaises(T.SeedError):
                        SS.run(fr, people(("OTHER", kit, cos)), fps=fps)

    def test_second_person_overlapping_at_the_seed_then_clear_is_anchored_safely(self):
        for fps in FPS_SET:
            with self.subTest(fps=fps):
                n, m = int(3 * fps), int(0.3 * fps)
                fr = [SS.SceneFrame({"TARGET": SS.box_at(500, 400, 40, H),
                                     "OTHER": SS.box_at(536 + 120 * min(1.0, k / max(1, m)), 400, 40, H)},
                                    order=["TARGET", "OTHER"]) for k in range(n)]
                try:
                    r = SS.run(fr, people(("OTHER", KIT_A, 0.75)), fps=fps)
                except T.SeedError:
                    continue                                  # refusing is always safe
                k = anchored_at(r)
                if k is not None:
                    self.assertFalse(any(f.identity_trusted for f in r.frames[:k]))
                self.assertEqual(wrong_trusted(r), [])
                self.assertTrue(r.bank_is_pure())
                bank_only_when_trusted(self, r)

    def test_ambiguous_seed_box_holding_two_bodies_is_refused(self):
        # the human box covers the target and another person who is substantially inside it
        fr = [SS.SceneFrame({"TARGET": SS.box_at(5, 400, 40, H), "OTHER": SS.box_at(30, 400, 40, H)},
                            order=["OTHER", "TARGET"]) for _ in range(30)]
        with self.assertRaises(T.SeedError):
            SS.run(fr, people(("OTHER", KIT_A, 0.75)), seed_box=(-15.0, 350.0, 50.0, 450.0))

    def test_edge_player_that_never_clears_is_refused(self):
        for fps in FPS_SET:
            with self.subTest(fps=fps):
                fr = [SS.SceneFrame({"TARGET": SS.box_at(5, 400, 40, H)}) for _ in range(int((ANCHOR_S + 1) * fps))]
                with self.assertRaises(T.SeedError):
                    SS.run(fr, people(), fps=fps)

    def test_cut_while_the_anchor_is_pending_refuses_the_seed(self):
        fps = 30.0
        fr = [SS.SceneFrame({"TARGET": SS.box_at(5, 400, 40, H)}) for _ in range(3)]
        fr += [SS.SceneFrame({"TARGET": SS.box_at(500, 400, 40, H)},
                             camera=SS.identity_motion(C.CUT)) for _ in range(1)]
        fr += [SS.SceneFrame({"TARGET": SS.box_at(500, 400, 40, H)}) for _ in range(20)]
        with self.assertRaises(T.SeedError):
            SS.run(fr, people(), fps=fps)

    def test_pending_tracklet_that_breaks_refuses_the_seed(self):
        # the edge-cut target is missed for longer than the population gap before it clears: the tracklet ends
        fps = 30.0
        n_gap = int(0.5 * fps)
        fr = [SS.SceneFrame({"TARGET": SS.box_at(5, 400, 40, H)}) for _ in range(3)]
        fr += [SS.SceneFrame({"TARGET": SS.box_at(5, 400, 40, H)}, drop=("TARGET",)) for _ in range(n_gap)]
        fr += [SS.SceneFrame({"TARGET": SS.box_at(200, 400, 40, H)}) for _ in range(20)]
        with self.assertRaises(T.SeedError):
            SS.run(fr, people(), fps=fps)


if __name__ == "__main__":
    unittest.main()

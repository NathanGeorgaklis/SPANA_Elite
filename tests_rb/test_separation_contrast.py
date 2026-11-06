# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2 same-kit contrast instruction (Nathan, 2026-11-07 08:22). Root cause on native 1080p G1_A:
#                 45 INSUFFICIENT_CONTRAST frames (contrast 0.074 < 0.08) with the same-kit teammate observed on its
#                 own tracklet 0.57 body heights away, apart since the last trusted frame, then LOST.
"""PHYSICAL SEPARATION contrast exclusion (target.py). Synthetic scenes with a same-kit look-alike (cos 0.85-0.9).
Cases: teammate alongside (apart, continuous) -> trusted; same-kit crossing -> fail closed; genuine substitution ->
never trusted; two equally plausible candidates -> ambiguous; contact / occlusion -> competes in full; camera pan
with both moving -> trusted; look-alike appearing after the last trusted frame -> competes; flag OFF -> old
behaviour; the exclusion is never applied to a reacquisition.
"""
import unittest

from player_tracker_rb import target as T
from player_tracker_rb.config import TrackerConfig
from tests_rb import _synth_scene as SS

KIT_A = (200, 60, 40)
H = 100.0
FPS = 30.0
ON = TrackerConfig.default()
OFF = TrackerConfig.with_overrides({"separation_exclusion_enabled": False})


def people(*others):
    return SS.make_people([("TARGET", KIT_A, 1.0)] + list(others))


def wrong_trusted(r):
    return [k for k in r.swaps() if r.frames[k].identity_trusted]


def trusted(r):
    return [k for k, f in enumerate(r.frames) if f.identity_trusted]


def reasons(r, k):
    return tuple(r.frames[k].identity_uncertainty_reasons or ())


def alongside(n, gap_h=0.6, look_from=0):
    """TARGET still at x=400; LOOK (same kit) standing apart at gap_h body heights to its right, from frame look_from."""
    out = []
    for k in range(n):
        boxes = {"TARGET": SS.box_at(400, 400, 40, H)}
        if k >= look_from:
            boxes["LOOK"] = SS.box_at(400 + 40 + gap_h * H + 20, 400, 40, H)
        out.append(SS.SceneFrame(boxes))
    return out


class TeammateAlongside(unittest.TestCase):
    def test_true_target_stays_trusted_with_a_same_kit_teammate_apart(self):
        for cos in (0.85, 0.9):
            for fps in (30.0, 60.0):
                with self.subTest(cos=cos, fps=fps):
                    n = int(4 * fps)
                    r = SS.run(alongside(n), people(("LOOK", KIT_A, cos)), cfg=ON, fps=fps)
                    self.assertEqual(wrong_trusted(r), [])
                    self.assertGreaterEqual(len(trusted(r)), int(0.9 * n), r.states()[:40])
                    self.assertNotIn(T.LOST, r.states())
                    ev = [e for e in r.frames[n // 2].decision["candidates"] if e["role"] == T.CONTINUATION][0]
                    self.assertTrue(ev["excluded_competitors"])

    def test_without_the_exclusion_the_same_scene_is_lost(self):
        """Documents the pre-rule behaviour (the real-footage failure): contrast below the margin -> LOST."""
        n = int(4 * FPS)
        r = SS.run(alongside(n), people(("LOOK", KIT_A, 0.9)), cfg=OFF)
        self.assertEqual(wrong_trusted(r), [])
        self.assertIn(T.LOST, r.states())
        self.assertLess(len(trusted(r)), n // 2)

    def test_camera_pan_with_both_bodies_moving_keeps_trust(self):
        n = int(3 * FPS)
        fr = []
        for k in range(n):
            dx = 3.0 * k                                      # the camera pans: both move together
            fr.append(SS.SceneFrame({"TARGET": SS.box_at(300 + dx, 400, 40, H),
                                     "LOOK": SS.box_at(300 + dx + 40 + 0.6 * H + 20, 400, 40, H)}))
        r = SS.run(fr, people(("LOOK", KIT_A, 0.9)), cfg=ON)
        self.assertEqual(wrong_trusted(r), [])
        self.assertGreaterEqual(len(trusted(r)), int(0.85 * n), r.states())


class ExclusionNeverAppliesWhenSeparationIsNotProven(unittest.TestCase):
    def _assert_fail_closed(self, r, after):
        self.assertEqual(wrong_trusted(r), [])
        for k in range(after, len(r.frames)):
            ev = [e for e in r.frames[k].decision.get("candidates", []) if e["role"] == T.CONTINUATION]
            if ev:
                self.assertEqual(ev[0]["excluded_competitors"], [], (k, ev[0]))

    def test_same_kit_crossing_competes_in_full_afterwards(self):
        n1, n2 = int(1.0 * FPS), int(2.5 * FPS)
        fr = alongside(n1, gap_h=0.6)
        # LOOK walks through the target's box and out the other side: overlap frames, then apart again
        for k in range(n2):
            x = 400 + 40 + 0.6 * H + 20 - 4.0 * k
            fr.append(SS.SceneFrame({"TARGET": SS.box_at(400, 400, 40, H), "LOOK": SS.box_at(x, 400, 40, H)}))
        r = SS.run(fr, people(("LOOK", KIT_A, 0.9)), cfg=ON)
        self.assertEqual(wrong_trusted(r), [])
        first_touch = next(k for k in range(n1, len(fr))
                           if SS.iou(fr[k].boxes["TARGET"], fr[k].boxes["LOOK"]) > 0
                           or abs(fr[k].boxes["LOOK"][0] - fr[k].boxes["TARGET"][2]) < 0.25 * H)
        self._assert_fail_closed(r, first_touch)

    def test_genuine_substitution_is_never_trusted(self):
        n1, n2 = int(1.5 * FPS), int(1.5 * FPS)
        fr = [SS.SceneFrame({"TARGET": SS.box_at(400, 400, 40, H)}) for _ in range(n1)]
        fr += [SS.SceneFrame({"LOOK": SS.box_at(402, 400, 40, H)}) for _ in range(n2)]
        for cos in (0.8, 0.9):
            with self.subTest(cos=cos):
                # same-spot substitution is the known production-default limit (caught by the optional handoff
                # rule); the exclusion must not make it any worse, and must not disturb the handoff rule
                on, off = SS.run(fr, people(("LOOK", KIT_A, cos)), cfg=ON), SS.run(fr, people(("LOOK", KIT_A, cos)), cfg=OFF)
                self.assertEqual(wrong_trusted(on), wrong_trusted(off))
                hand = SS.run(fr, people(("LOOK", KIT_A, cos)), cfg=TrackerConfig.with_overrides(
                    {"continuity_step_enabled": True}))
                self.assertEqual(wrong_trusted(hand), [])
                k0 = len(fr) - 1
                ev = [e for e in on.frames[k0].decision.get("candidates", []) if e["role"] == T.CONTINUATION]
                if ev:
                    self.assertEqual(ev[0]["excluded_competitors"], [])

    def test_look_alike_appearing_after_the_last_trusted_frame_competes(self):
        """LOOK is first seen while the target is already untrusted: no continuity certificate -> full contrast."""
        n = int(3 * FPS)
        fr = alongside(n, gap_h=0.6, look_from=int(1.0 * FPS))
        # make the target untrusted just before LOOK appears: drop its detection for a few frames
        for k in range(int(0.8 * FPS), int(1.0 * FPS)):
            fr[k] = SS.SceneFrame(dict(fr[k].boxes), drop=("TARGET",))
        r = SS.run(fr, people(("LOOK", KIT_A, 0.9)), cfg=ON)
        self.assertEqual(wrong_trusted(r), [])
        k0 = int(1.0 * FPS)
        ev = [e for e in r.frames[k0].decision.get("candidates", []) if e["role"] == T.CONTINUATION]
        if ev:
            self.assertEqual(ev[0]["excluded_competitors"], [])

    def test_two_equally_plausible_candidates_stay_ambiguous(self):
        """The target vanishes and two same-kit bodies appear where it was: a reacquisition, never excluded."""
        n1, n2 = int(1.0 * FPS), int(2.0 * FPS)
        fr = [SS.SceneFrame({"TARGET": SS.box_at(400, 400, 40, H)}) for _ in range(n1)]
        fr += [SS.SceneFrame({"L1": SS.box_at(380, 400, 40, H), "L2": SS.box_at(440, 400, 40, H)})
               for _ in range(n2)]
        r = SS.run(fr, people(("L1", KIT_A, 0.9), ("L2", KIT_A, 0.9)), cfg=ON)
        self.assertEqual(wrong_trusted(r), [])
        self.assertEqual([k for k in range(n1, len(r.frames)) if r.frames[k].identity_trusted], [])

    def test_contact_and_occlusion_compete_in_full(self):
        """Apart for 1 s, then LOOK steps into contact (overlapping, drawn in front) and stays: from the first
        contact frame the competitor competes in full and no wrong body is ever trusted."""
        n1, n2 = int(1.0 * FPS), int(2.0 * FPS)
        fr = alongside(n1, gap_h=0.6)
        fr += [SS.SceneFrame({"TARGET": SS.box_at(400, 400, 40, H), "LOOK": SS.box_at(430, 404, 40, H)},
                             order=("TARGET", "LOOK")) for _ in range(n2)]
        r = SS.run(fr, people(("LOOK", KIT_A, 0.9)), cfg=ON)
        self.assertEqual(wrong_trusted(r), [])
        self._assert_fail_closed(r, n1)


if __name__ == "__main__":
    unittest.main()

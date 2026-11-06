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
                for k in range(n1, len(fr)):
                    for e in on.frames[k].decision.get("candidates", []):
                        self.assertEqual(e.get("excluded_competitors", []), [], (k, e))

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
        # the target comes back as a reacquisition with LOOK beside it: two same-kit candidates, never trusted,
        # nothing excluded in any evaluation (a reacquisition never excludes; no trusted anchor exists after k0)
        seen = 0
        for k in range(k0, n):
            self.assertFalse(r.frames[k].identity_trusted, k)
            for e in r.frames[k].decision.get("candidates", []):
                seen += 1
                self.assertEqual(e["excluded_competitors"], [], (k, e["role"]))
        self.assertGreater(seen, 0, "candidates must have been evaluated after LOOK appears")

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


class ReviewCases(unittest.TestCase):
    """Review R2-SEP findings: unverified episodes, the deferred-seed window, a held target frame, the exact gap,
    a rival inside the candidate's nested group, and a rival whose tracklet id changes."""

    def _cont(self, r, k):
        return [e for e in r.frames[k].decision.get("candidates", []) if e["role"] == T.CONTINUATION]

    def test_unverified_episode_never_excludes_and_never_becomes_trusted_on_a_rival(self):
        """Target lost; same-kit L1 reacquired alone (UNVERIFIED episode); same-kit L2 walks in to 0.6 h, never
        within 0.25 h; then a beatable confusable Q appears. With the exclusion the episode must still not verify
        itself on L1 (review finding 1)."""
        n1, n2, n3, n4, n5 = int(1.0 * FPS), int(1.5 * FPS), int(1.0 * FPS), int(1.0 * FPS), int(2.0 * FPS)
        L1 = SS.box_at(400, 400, 40, H)
        Q = SS.box_at(400 - 40 - 0.6 * H - 20, 400, 40, H)

        def L2(gap):
            return SS.box_at(400 + 40 + gap * H + 20, 400, 40, H)
        fr = [SS.SceneFrame({"TARGET": SS.box_at(400, 400, 40, H), "L2": L2(2.2)}) for _ in range(n1)]
        fr += [SS.SceneFrame({"L2": L2(2.2)}) for _ in range(n2)]
        fr += [SS.SceneFrame({"L1": L1, "L2": L2(2.2)}) for _ in range(n3)]
        fr += [SS.SceneFrame({"L1": L1, "L2": L2(2.2 - 1.6 * (k + 1) / n4)}) for k in range(n4)]
        fr += [SS.SceneFrame({"L1": L1, "L2": L2(0.6), "Q": Q}) for _ in range(n5)]
        pp = people(("L1", KIT_A, 0.9), ("L2", KIT_A, 0.9), ("Q", KIT_A, 0.86))
        on, off = SS.run(fr, pp, cfg=ON), SS.run(fr, pp, cfg=OFF)
        self.assertEqual(wrong_trusted(on), [])
        self.assertEqual(wrong_trusted(on), wrong_trusted(off))
        for k in range(n1 + n2, len(fr)):
            for e in self._cont(on, k):
                if not on.frames[k].decision.get("episode_contrast_verified"):
                    self.assertEqual(e["excluded_competitors"], [], k)

    def test_deferred_seed_window_voids_the_certificate(self):
        """Seed deferred (partial box at the image edge); during the pending window the same-kit rival stands
        0.05 h away; after the anchor the rival moves apart. It must compete in full (review finding 2)."""
        n1, n2 = int(0.3 * FPS), int(2.0 * FPS)
        tb = (0.0, 350.0, 36.0, 450.0)                       # touches the left image edge: deferrable seed
        fr = [SS.SceneFrame({"TARGET": tb, "LOOK": SS.box_at(36 + 0.05 * H + 20, 400, 40, H)}) for _ in range(n1)]
        fr += [SS.SceneFrame({"TARGET": SS.box_at(60 + 2 * k, 400, 40, H),
                              "LOOK": SS.box_at(60 + 2 * k + 40 + 0.6 * H + 20, 400, 40, H)}) for k in range(n2)]
        try:
            r = SS.run(fr, people(("LOOK", KIT_A, 0.9)), cfg=ON, seed_box=tb)
        except T.SeedError:
            return                                            # a refused seed is also fail-closed
        self.assertEqual(wrong_trusted(r), [])
        first_anchor = next((k for k, f in enumerate(r.frames) if f.identity_trusted), None)
        if first_anchor is not None:
            k = first_anchor + 1
            for e in self._cont(r, k):
                self.assertEqual(e["excluded_competitors"], [], (k, e))

    def test_one_held_target_frame_voids_the_certificate(self):
        n = int(3 * FPS)
        fr = alongside(n, gap_h=0.6)
        k_hold = int(1.5 * FPS)
        fr[k_hold] = SS.SceneFrame(dict(fr[k_hold].boxes), drop=("TARGET",))
        r = SS.run(fr, people(("LOOK", KIT_A, 0.9)), cfg=ON)
        self.assertEqual(wrong_trusted(r), [])
        for k in range(k_hold + 1, min(k_hold + 4, n)):
            for e in self._cont(r, k):
                self.assertEqual(e["excluded_competitors"], [], k)

    def test_exact_gap_counts_as_contact(self):
        n = int(2 * FPS)
        gap = ON.get("separation_min_gap_h")
        fr = [SS.SceneFrame({"TARGET": SS.box_at(400, 400, 40, H),
                             "LOOK": (420.0 + gap * H, 350.0, 460.0 + gap * H, 450.0)}) for _ in range(n)]
        r = SS.run(fr, people(("LOOK", KIT_A, 0.9)), cfg=ON)
        self.assertEqual(wrong_trusted(r), [])
        for k in range(1, n):
            for e in self._cont(r, k):
                self.assertEqual(e["excluded_competitors"], [], k)

    def test_rival_tracklet_id_change_voids_the_certificate(self):
        """LOOK is undetected for a few frames (its tracklet closes and a new one is born): never excluded after."""
        n = int(4 * FPS)
        fr = alongside(n, gap_h=0.6)
        for k in range(int(1.0 * FPS), int(1.0 * FPS) + int(ON.get("population_closed_memory_s") * FPS) + 5):
            fr[k] = SS.SceneFrame(dict(fr[k].boxes), drop=("LOOK",))
        r = SS.run(fr, people(("LOOK", KIT_A, 0.9)), cfg=ON)
        self.assertEqual(wrong_trusted(r), [])
        k_back = int(1.0 * FPS) + int(ON.get("population_closed_memory_s") * FPS) + 5
        for k in range(k_back, min(k_back + 5, n)):
            for e in self._cont(r, k):
                self.assertEqual(e["excluded_competitors"], [], k)


if __name__ == "__main__":
    unittest.main()

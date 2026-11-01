# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2 DEV F8 instruction (Nathan, 2026-11-01 12:00): fix false camera-scale estimates that distort
#                 target position/size; generic; fail closed on unreliable camera scale; test pan, real zoom, false
#                 zoom, cut, crowded scene.
"""F8: camera scale corroboration against people's measured heights (synthetic textured scenes, no benchmark data).

DEV failure reproduced: under a broadcast pan the similarity fit reported a steady zoom-out (x0.66 over 4 s) while the
people's heights rose; the held target size and gate were scaled by it.
"""
import math
import unittest

import numpy as np

from player_tracker_rb import camera as C
from player_tracker_rb.config import TrackerConfig
from tests_rb.test_camera import H, W, WORLD, canvas, view

CFG = TrackerConfig.default()
WIN = CFG.get("camera_scale_check_window_s")


def people_world(n, seed=3, spread=(700, 1700), y=(600, 800)):
    r = np.random.default_rng(seed)
    return [(float(r.uniform(*spread)), float(r.uniform(*y))) for _ in range(n)]


def boxes_in_view(pts, cx, cy, z, *, h_world=120.0, size_zoom=None):
    """Person boxes (feet at the world point) under view(cx, cy, z). size_zoom: the zoom applied to their SIZE
    (default z: a real zoom; a constant: their image size does not follow the fitted zoom)."""
    sz = z if size_zoom is None else size_zoom
    out = []
    for (px, py) in pts:
        u, v = W / 2 + z * (px - cx), H / 2 + z * (py - cy)
        hh, ww = h_world * sz, 0.4 * h_world * sz
        out.append((u - ww / 2, v - hh, u + ww / 2, v))
    return out


def run(frames):
    est = C.CameraMotionEstimator(CFG)
    return [est.update(img, boxes, dt) for img, boxes, dt in frames]


def flagged(ms):
    return [k for k, m in enumerate(ms) if C.SCALE_UNCORROBORATED in m.reasons]


class TestScaleCorroboration(unittest.TestCase):
    def test_pan_with_people_is_unchanged(self):
        for fps in (30.0, 15.0):
            with self.subTest(fps=fps):
                pts = people_world(5)
                fr = [(view(WORLD, 1000 + 300 * k / fps, 700, 1.0), boxes_in_view(pts, 1000 + 300 * k / fps, 700, 1.0),
                       1 / fps) for k in range(int(2.5 * fps))]
                ms = run(fr)
                self.assertEqual(flagged(ms), [])
                for m in ms[1:]:
                    self.assertEqual(m.status, C.ESTIMATED)
                    self.assertAlmostEqual(m.shift_px[0], -300 / fps, delta=0.6)
                    self.assertAlmostEqual(m.scale, 1.0, delta=0.005)

    def test_real_zoom_agreeing_with_people_is_kept(self):
        for fps in (30.0, 15.0):
            with self.subTest(fps=fps):
                pts = people_world(5)
                n = int(2.5 * fps)
                zs = [1.0 * (1.25 ** (k / n)) for k in range(n)]
                fr = [(view(WORLD, 1200, 700, z), boxes_in_view(pts, 1200, 700, z), 1 / fps) for z in zs]
                ms = run(fr)
                self.assertEqual(flagged(ms), [])
                cum = math.exp(sum(math.log(m.scale) for m in ms[1:]))
                self.assertAlmostEqual(cum, zs[-1] / zs[0], delta=0.03)

    def test_false_zoom_contradicted_by_people_is_rejected(self):
        # the image content zooms out (x0.7 over 2.5 s) but the people's image heights do not change: after the
        # window has filled, the scale is uncorroborated -> translation-only motion, fit kept for audit only
        for fps in (30.0, 15.0):
            with self.subTest(fps=fps):
                pts = people_world(5)
                n = int(2.5 * fps)
                zs = [0.7 ** (k / n) for k in range(n)]
                fr = [(view(WORLD, 1200 + 100 * k / fps, 700, z),
                       boxes_in_view(pts, 1200 + 100 * k / fps, 700, z, size_zoom=1.0), 1 / fps)
                      for k, z in enumerate(zs)]
                ms = run(fr)
                f = flagged(ms)
                self.assertTrue(f)
                self.assertGreaterEqual(f[0] / fps, WIN * CFG.get("camera_scale_check_min_span_fraction") - 1.5 / fps)
                self.assertTrue(all(k in f for k in range(f[0], len(ms))), "a sustained false zoom stays rejected")
                for k in f:
                    m = ms[k]
                    self.assertEqual(m.status, C.ESTIMATED)
                    self.assertTrue(m.compensated)
                    self.assertFalse(m.continuity_broken)
                    self.assertEqual(m.scale, 1.0)
                    self.assertEqual(m.rotation_rad, 0.0)
                    self.assertIsNotNone(m.measured_matrix)
                    self.assertLess(math.hypot(*m.measured_matrix[0][:2]), 1.0)   # the rejected fit zoomed out
                    # the pan is still measured: the image moves left by about the pan speed
                    self.assertLess(m.shift_px[0], 0.0)
                    self.assertAlmostEqual(m.shift_px[0], -100 / fps * zs[k], delta=2.0)

    def test_too_few_people_keeps_the_fit(self):
        # no evidence either way: the fitted scale is used as before (two people are not a crowd's median)
        pts = people_world(2)
        n = 75
        zs = [0.7 ** (k / n) for k in range(n)]
        fr = [(view(WORLD, 1200, 700, z), boxes_in_view(pts, 1200, 700, z, size_zoom=1.0), 1 / 30) for z in zs]
        ms = run(fr)
        self.assertEqual(flagged(ms), [])
        self.assertLess(ms[-1].scale, 1.0)

    def test_cut_restarts_the_check(self):
        pts = people_world(5)
        fps = 30.0
        n = int(1.5 * fps)
        zs = [0.7 ** (k / n) for k in range(n)]
        fr = [(view(WORLD, 1200, 700, z), boxes_in_view(pts, 1200, 700, z, size_zoom=1.0), 1 / fps) for z in zs]
        other = canvas(7)
        fr += [(view(other, 1000 + 100 * k / fps, 700, 1.0), boxes_in_view(pts, 1000 + 100 * k / fps, 700, 1.0),
                1 / fps) for k in range(int(1.5 * fps))]
        ms = run(fr)
        self.assertTrue([k for k in flagged(ms) if k < n], "flagged before the cut")
        self.assertTrue(ms[n].continuity_broken)
        self.assertEqual([k for k in flagged(ms) if k > n], [], "a new shot starts a new check")

    def test_crowded_scene_with_own_motion_and_depth_changes_is_not_flagged(self):
        # 14 people, each walking and changing depth on its own (+-1 % height per frame, random), under a real pan
        r = np.random.default_rng(5)
        fps = 30.0
        pts = people_world(14, seed=11)
        hs = [120.0] * len(pts)
        fr = []
        for k in range(int(3 * fps)):
            cx = 1000 + 250 * k / fps
            pts = [(x + r.normal(0, 2), y + r.normal(0, 1)) for (x, y) in pts]
            hs = [h * math.exp(r.normal(0, 0.01)) for h in hs]
            boxes = []
            for (px, py), h in zip(pts, hs):
                u, v = W / 2 + (px - cx), H / 2 + (py - 700)
                boxes.append((u - 0.2 * h, v - h, u + 0.2 * h, v))
            fr.append((view(WORLD, cx, 700, 1.0), boxes, 1 / fps))
        ms = run(fr)
        self.assertEqual(flagged(ms), [])


if __name__ == "__main__":
    unittest.main()

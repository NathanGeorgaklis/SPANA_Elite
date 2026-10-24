# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R1-E instruction (Nathan, 2026-10-24); R1/R1_DESIGN_DECISIONS.md D13, D16
"""R1-E camera-motion tests on synthetic textured scenes: pan, zoom, foreground masking, hard cut, unmeasurable
frames, bounds, resolution change, resolution invariance, determinism, input validation."""
import unittest

import cv2
import numpy as np

from player_tracker_rb import camera as C
from player_tracker_rb.config import TrackerConfig

CFG = TrackerConfig.default()
W, H = 1280, 720
DT = 1 / 30


def canvas(seed, width=2400, height=1400, brightness=1.0):
    r = np.random.default_rng(seed)
    img = (r.random((height // 8, width // 8, 3)) * 255 * brightness).astype(np.uint8)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
    return cv2.GaussianBlur(img, (5, 5), 0)


def view(cv, cx, cy, z, width=W, height=H):
    """The camera looks at world point (cx, cy) with zoom z."""
    m = np.array([[z, 0, width / 2 - z * cx], [0, z, height / 2 - z * cy]], np.float64)
    return cv2.warpAffine(cv, m, (width, height), flags=cv2.INTER_LINEAR)


WORLD = canvas(1)


class TestEstimation(unittest.TestCase):
    def test_first_frame(self):
        est = C.CameraMotionEstimator(CFG)
        m = est.update(view(WORLD, 1200, 700, 1.0), [], DT)
        self.assertEqual(m.status, C.FIRST_FRAME)
        self.assertFalse(m.continuity_broken)
        self.assertFalse(m.compensated)

    def test_pan(self):
        est = C.CameraMotionEstimator(CFG)
        est.update(view(WORLD, 1200, 700, 1.0), [], DT)
        m = est.update(view(WORLD, 1215, 695, 1.0), [], DT)          # camera right 15, up 5 -> image left/down
        self.assertEqual(m.status, C.ESTIMATED)
        self.assertAlmostEqual(m.shift_px[0], -15.0, delta=0.5)
        self.assertAlmostEqual(m.shift_px[1], 5.0, delta=0.5)
        self.assertAlmostEqual(m.scale, 1.0, delta=0.005)
        self.assertFalse(m.continuity_broken)
        self.assertGreater(m.inlier_fraction, CFG.get("cut_flow_inlier_fraction_max"))

    def test_zoom(self):
        est = C.CameraMotionEstimator(CFG)
        est.update(view(WORLD, 1200, 700, 1.0), [], DT)
        z = 1.03                                                     # inside the 3x-per-second zoom bound
        m = est.update(view(WORLD, 1200, 700, z), [], DT)
        self.assertEqual(m.status, C.ESTIMATED)
        self.assertAlmostEqual(m.scale, z, delta=0.003)
        # a world point maps where the zoom puts it
        x, y = m.warp_point(100.0, 100.0)
        self.assertAlmostEqual(x, W / 2 + z * (100 - W / 2), delta=1.5)
        self.assertAlmostEqual(y, H / 2 + z * (100 - H / 2), delta=1.5)

    def test_moving_people_masked(self):
        """A large textured foreground object moving the opposite way must not drag the estimate: its box is
        masked out of the background features."""
        fg = canvas(9, 300, 400)
        boxes = []
        frames = []
        for k, (cx, ox) in enumerate(((1200, 500), (1210, 470))):
            img = view(WORLD, cx, 700, 1.0).copy()
            img[150:550, ox:ox + 300] = fg
            frames.append(img)
            boxes.append([(ox, 150, ox + 300, 550)])
        est = C.CameraMotionEstimator(CFG)
        est.update(frames[0], boxes[0], DT)
        m = est.update(frames[1], boxes[1], DT)
        self.assertEqual(m.status, C.ESTIMATED)
        self.assertAlmostEqual(m.shift_px[0], -10.0, delta=0.5)

    def test_resolution_invariance(self):
        res = {}
        for width, height in ((1280, 720), (1920, 1080)):
            s = width / 1280
            est = C.CameraMotionEstimator(CFG)
            est.update(view(WORLD, 1200, 700, s, width, height), [], DT)
            m = est.update(view(WORLD, 1212, 700, s, width, height), [], DT)
            res[width] = m.shift_px[0] / width
        self.assertAlmostEqual(res[1280], res[1920], delta=0.0005)

    def test_deterministic(self):
        def go():
            est = C.CameraMotionEstimator(CFG)
            return [est.update(view(WORLD, 1200 + 7 * k, 700, 1.0 + 0.01 * k), [], DT).as_record()
                    for k in range(4)]
        self.assertEqual(go(), go())


class TestContinuity(unittest.TestCase):
    def test_hard_cut(self):
        est = C.CameraMotionEstimator(CFG)
        est.update(view(WORLD, 1200, 700, 1.0), [], DT)
        est.update(view(WORLD, 1210, 700, 1.0), [], DT)
        other = canvas(7, brightness=0.35)                           # a different shot, different exposure
        m = est.update(view(other, 1000, 600, 1.0), [], DT)
        self.assertEqual(m.status, C.CUT)
        self.assertIn(C.HISTOGRAM_CHANGE, m.reasons)
        self.assertTrue(m.continuity_broken)
        self.assertFalse(m.compensated)
        self.assertEqual(m.matrix, C.IDENTITY)

    def test_cut_between_similar_shots_breaks_continuity_by_default(self):
        """A cut whose histogram does not change is indistinguishable from a flow failure: UNKNOWN, which breaks
        continuity unless configured otherwise (fail closed)."""
        est = C.CameraMotionEstimator(CFG)
        est.update(view(WORLD, 1200, 700, 1.0), [], DT)
        m = est.update(view(canvas(7), 1000, 600, 1.0), [], DT)
        self.assertIn(m.status, (C.UNKNOWN, C.CUT))
        self.assertTrue(m.continuity_broken)
        cfg = TrackerConfig.with_overrides({"camera_unknown_breaks_continuity": False})
        est = C.CameraMotionEstimator(cfg)
        est.update(view(WORLD, 1200, 700, 1.0), [], DT)
        m = est.update(view(canvas(7), 1000, 600, 1.0), [], DT)
        if m.status == C.UNKNOWN:
            self.assertFalse(m.continuity_broken)

    def test_cut_with_static_broadcast_overlay_breaks_continuity(self):
        """A static score bug / logo is identical in both shots: its corners can dominate the flow features and
        fit a perfect 'no motion' transform across a cut. The photometric residual of the warped previous frame
        exposes the cut (regression for the R1-E code review finding H2)."""
        def shot(seed, base):
            r = np.random.default_rng(seed)
            img = np.zeros((H, W, 3), np.uint8)
            img[:] = base
            noise = cv2.GaussianBlur(r.integers(0, 40, (H, W)).astype(np.uint8), (0, 0), 6)
            img = cv2.add(img, cv2.merge([noise] * 3))
            for _ in range(4):
                cv2.line(img, (int(r.integers(0, W)), 0), (int(r.integers(0, W)), H), (230, 230, 230), 3)
            cv2.rectangle(img, (40, 30), (420, 80), (20, 20, 20), -1)
            cv2.putText(img, "HOME 1 - 0 AWAY 63:12", (50, 68), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(img, "SPORTS HD", (1050, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            return img
        est = C.CameraMotionEstimator(CFG)
        est.update(shot(1, (40, 120, 40)), [], DT)
        m = est.update(shot(2, (60, 60, 140)), [], DT)
        self.assertNotEqual(m.status, C.ESTIMATED)
        self.assertTrue(m.continuity_broken)
        self.assertIn(C.HIGH_RESIDUAL, m.reasons)

    def test_overlay_does_not_hide_a_real_pan(self):
        """The same static overlay on a genuinely panning shot: the estimate must still be the pan (or be
        rejected), never a confident 'no motion'."""
        est = C.CameraMotionEstimator(CFG)
        frames = []
        for cx in (1200, 1220):
            img = view(WORLD, cx, 700, 1.0).copy()
            cv2.rectangle(img, (40, 30), (420, 80), (20, 20, 20), -1)
            cv2.putText(img, "HOME 1 - 0 AWAY 63:12", (50, 68), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            frames.append(img)
        est.update(frames[0], [], DT)
        m = est.update(frames[1], [], DT)
        if m.status == C.ESTIMATED:
            self.assertAlmostEqual(m.shift_px[0], -20.0, delta=1.0)
        else:
            self.assertTrue(m.continuity_broken)

    def test_residual_small_for_true_motion(self):
        est = C.CameraMotionEstimator(CFG)
        est.update(view(WORLD, 1200, 700, 1.0), [], DT)
        m = est.update(view(WORLD, 1212, 703, 1.02), [], DT)
        self.assertEqual(m.status, C.ESTIMATED)
        self.assertLess(m.residual, CFG.get("cut_residual_min") / 4)

    def test_flat_frame_unknown(self):
        est = C.CameraMotionEstimator(CFG)
        est.update(np.full((H, W, 3), 90, np.uint8), [], DT)
        m = est.update(np.full((H, W, 3), 90, np.uint8), [], DT)
        self.assertEqual(m.status, C.UNKNOWN)
        self.assertIn(C.TOO_FEW_FEATURES, m.reasons)
        self.assertTrue(m.continuity_broken)
        self.assertEqual(m.matrix, C.IDENTITY)

    def test_implausible_motion_rejected(self):
        cfg = TrackerConfig.with_overrides({"camera_max_shift_frac_per_s": 0.1})   # 0.1 W/s: 4 px per frame
        est = C.CameraMotionEstimator(cfg)
        est.update(view(WORLD, 1200, 700, 1.0), [], DT)
        m = est.update(view(WORLD, 1230, 700, 1.0), [], DT)
        self.assertNotEqual(m.status, C.ESTIMATED)
        self.assertIn(C.IMPLAUSIBLE_SHIFT, m.reasons)
        cfg = TrackerConfig.with_overrides({"camera_max_scale_ratio_per_s": 1.1})
        est = C.CameraMotionEstimator(cfg)
        est.update(view(WORLD, 1200, 700, 1.0), [], DT)
        m = est.update(view(WORLD, 1200, 700, 1.05), [], DT)
        self.assertIn(C.IMPLAUSIBLE_SCALE, m.reasons)
        self.assertEqual(m.scale, 1.0)                                # rejected: not applied ...
        self.assertAlmostEqual(np.hypot(m.measured_matrix[0][0], m.measured_matrix[1][0]), 1.05, delta=0.005)

    def test_resolution_change_is_a_cut(self):
        est = C.CameraMotionEstimator(CFG)
        est.update(view(WORLD, 1200, 700, 1.0), [], DT)
        m = est.update(view(WORLD, 1200, 700, 1.0, 960, 540), [], DT)
        self.assertEqual(m.status, C.CUT)
        self.assertIn(C.RESOLUTION_CHANGED, m.reasons)


class TestRecordsAndValidation(unittest.TestCase):
    def test_matrix_helpers(self):
        m = C.motion_from_matrix([[1, 0, 10], [0, 1, -4]], W, H)
        self.assertEqual(m.status, C.ESTIMATED)
        self.assertEqual(m.shift_px, (10.0, -4.0))
        self.assertEqual(m.warp_box((0, 0, 10, 20)), (10.0, -4.0, 20.0, 16.0))
        with self.assertRaises(C.CameraError):
            C.identity_motion(C.ESTIMATED)
        with self.assertRaises(C.CameraError):
            C.motion_from_matrix([[1, 0, float("nan")], [0, 1, 0]], W, H)

    def test_invalid_input(self):
        est = C.CameraMotionEstimator(CFG)
        good = view(WORLD, 1200, 700, 1.0)
        for img in (None, np.zeros((H, W), np.uint8), good.astype(np.float32), np.zeros((0, W, 3), np.uint8)):
            with self.assertRaises(C.CameraError):
                est.update(img, [], DT)
        with self.assertRaises(C.CameraError):
            est.update(good, [(0, 0, 1)], DT)
        with self.assertRaises(C.CameraError):
            est.update(good, [(0, 0, float("inf"), 1)], DT)
        for dt in (0.0, -1.0, float("nan")):
            with self.assertRaises(C.CameraError):
                est.update(good, [], dt)


if __name__ == "__main__":
    unittest.main()

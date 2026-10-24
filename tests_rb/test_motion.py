# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R1-E instruction (Nathan, 2026-10-24); R1/R1_DESIGN_DECISIONS.md D06, D09, D13, D16
"""R1-E motion-model unit tests: observed vs predicted vs held, bounded velocity, the plausibility veto, partial
observations, camera compensation, fps and resolution invariance, input validation. Motion is a veto only: no
function here returns an identity score."""
import math
import unittest

from player_tracker_rb.camera import CUT, FIRST_FRAME, UNKNOWN, identity_motion, motion_from_matrix
from player_tracker_rb.config import TrackerConfig
from player_tracker_rb.motion import DETECTION, HELD, PREDICTED, SCALE, SPEED, MotionError, MotionModel

CFG = TrackerConfig.default()
H_PX = 80.0


def box(cx, cy, w=32.0, h=H_PX):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def walk(mm, fps, seconds, v_h_per_s, h=H_PX, scale=1.0):
    st = mm.start(box(300 * scale, 360 * scale, 32 * scale, h * scale), 0.0)
    n = int(round(seconds * fps))
    for k in range(1, n + 1):
        t = k / fps
        st = mm.update(st, box((300 + v_h_per_s * h * t) * scale, 360 * scale, 32 * scale, h * scale), t)
    return st


class TestPrediction(unittest.TestCase):
    def setUp(self):
        self.mm = MotionModel(CFG)

    def test_predicted_then_held_never_detection(self):
        st = walk(self.mm, 30.0, 1.0, 2.0)
        horizon = CFG.get("motion_max_extrapolation_s")
        p1 = self.mm.predict(st, st.t_obs + horizon / 2)
        p2 = self.mm.predict(st, st.t_obs + horizon * 3)
        self.assertEqual(p1.source, PREDICTED)
        self.assertEqual(p2.source, HELD)
        for p in (p1, p2):
            self.assertNotEqual(p.source, DETECTION)
        self.assertAlmostEqual(p2.extrapolated_s, horizon)
        # extrapolation stops at the horizon: a held box no longer moves
        p3 = self.mm.predict(st, st.t_obs + horizon * 10)
        self.assertEqual(p2.box, p3.box)

    def test_prediction_follows_velocity(self):
        st = walk(self.mm, 30.0, 1.0, 2.0)
        p = self.mm.predict(st, st.t_obs + 0.1)
        dx = (p.box[0] + p.box[2]) / 2 - (st.ref_box[0] + st.ref_box[2]) / 2
        self.assertAlmostEqual(dx, 2.0 * H_PX * 0.1, delta=2.0 * H_PX * 0.1 * 0.1)

    def test_velocity_bounded(self):
        st = self.mm.start(box(300, 360), 0.0)
        st = self.mm.update(st, box(3000, 360), 1 / 30)         # teleport: not a physical speed
        vmax = CFG.get("motion_veto_max_speed_h_per_s") * H_PX
        self.assertLessEqual(math.hypot(st.vx, st.vy), vmax * (1 + 1e-9))
        rmax = math.log(CFG.get("motion_veto_max_scale_ratio_per_s"))
        st2 = self.mm.update(self.mm.start(box(300, 360), 0.0), box(300, 360, 32, 400), 1 / 30)
        self.assertLessEqual(abs(st2.log_h_rate), rmax + 1e-12)

    def test_fps_invariance_of_velocity(self):
        a = walk(self.mm, 30.0, 1.0, 2.0)
        b = walk(self.mm, 60.0, 1.0, 2.0)
        self.assertAlmostEqual(a.vx, b.vx, delta=abs(a.vx) * 0.02)


class TestVeto(unittest.TestCase):
    def setUp(self):
        self.mm = MotionModel(CFG)
        self.st = walk(self.mm, 30.0, 0.5, 0.0)

    def test_small_move_plausible(self):
        chk = self.mm.check(self.st, box(305, 360), self.st.t_obs + 1 / 30)
        self.assertTrue(chk.plausible)
        self.assertEqual(chk.reasons, ())

    def test_large_jump_vetoed(self):
        chk = self.mm.check(self.st, box(300 + 3 * H_PX, 360), self.st.t_obs + 1 / 30)
        self.assertFalse(chk.plausible)
        self.assertIn(SPEED, chk.reasons)
        self.assertGreater(chk.displacement_h, chk.allowed_displacement_h)

    def test_allowance_grows_with_time(self):
        far = box(300 + 3 * H_PX, 360)
        self.assertFalse(self.mm.check(self.st, far, self.st.t_obs + 1 / 30).plausible)
        self.assertTrue(self.mm.check(self.st, far, self.st.t_obs + 1.0).plausible)

    def test_scale_veto_and_partial_exemption(self):
        tall = box(300, 360, 32, H_PX * 2.5)
        self.assertIn(SCALE, self.mm.check(self.st, tall, self.st.t_obs + 1 / 30).reasons)
        short = box(300, 340, 32, H_PX * 0.4)
        self.assertIn(SCALE, self.mm.check(self.st, short, self.st.t_obs + 1 / 30).reasons)
        self.assertNotIn(SCALE, self.mm.check(self.st, short, self.st.t_obs + 1 / 30, partial=True).reasons)

    def test_veto_is_scale_invariant(self):
        st2 = walk(self.mm, 30.0, 0.5, 0.0, scale=2.0)
        for dx_h in (0.1, 0.5, 1.0, 3.0):
            a = self.mm.check(self.st, box(300 + dx_h * H_PX, 360), self.st.t_obs + 1 / 30)
            b = self.mm.check(st2, box(600 + dx_h * 2 * H_PX, 720, 64, 2 * H_PX), st2.t_obs + 1 / 30)
            self.assertEqual(a.plausible, b.plausible)
            self.assertAlmostEqual(a.displacement_h, b.displacement_h)

    def test_no_identity_output(self):
        chk = self.mm.check(self.st, box(305, 360), self.st.t_obs + 1 / 30)
        for k in chk.as_record():
            self.assertNotIn("identity", k)
            self.assertNotIn("similar", k)


class TestPartial(unittest.TestCase):
    def test_partial_does_not_update_velocity_or_size_rate(self):
        mm = MotionModel(CFG)
        st = walk(mm, 30.0, 0.5, 1.0)
        v0, r0 = (st.vx, st.vy), st.log_h_rate
        t = st.t_obs + 1 / 30
        st = mm.update(st, (300, 320, 332, 360), t, partial=True)       # top half only
        self.assertEqual((st.vx, st.vy), v0)
        self.assertEqual(st.log_h_rate, r0)
        self.assertTrue(st.ref_partial)
        # the first full observation after a partial one is not motion evidence either
        st = mm.update(st, box(320, 360), t + 1 / 30)
        self.assertEqual((st.vx, st.vy), v0)
        self.assertFalse(st.ref_partial)


class TestCamera(unittest.TestCase):
    def setUp(self):
        self.mm = MotionModel(CFG)
        self.st = walk(self.mm, 30.0, 0.5, 1.0)

    def test_pan_moves_reference(self):
        cam = motion_from_matrix([[1, 0, -40], [0, 1, 5]], 1280, 720)
        st = self.mm.apply_camera(self.st, cam)
        self.assertAlmostEqual(st.ref_box[0], self.st.ref_box[0] - 40)
        self.assertAlmostEqual(st.ref_box[1], self.st.ref_box[1] + 5)
        self.assertEqual((st.vx, st.vy), (self.st.vx, self.st.vy))     # a pure pan does not change velocity
        self.assertTrue(st.camera_compensated)

    def test_zoom_scales_box_and_velocity(self):
        s = 1.1
        cam = motion_from_matrix([[s, 0, 640 - s * 640], [0, s, 360 - s * 360]], 1280, 720)
        st = self.mm.apply_camera(self.st, cam)
        self.assertAlmostEqual(st.ref_box[3] - st.ref_box[1], (self.st.ref_box[3] - self.st.ref_box[1]) * s)
        self.assertAlmostEqual(st.vx, self.st.vx * s)

    def test_unknown_marks_uncompensated(self):
        st = self.mm.apply_camera(self.st, identity_motion(UNKNOWN, continuity_broken=False))
        self.assertFalse(st.camera_compensated)
        self.assertFalse(self.mm.predict(st, st.t_obs + 0.1).camera_compensated)
        self.assertEqual(st.ref_box, self.st.ref_box)

    def test_first_frame_is_a_no_op(self):
        self.assertEqual(self.mm.apply_camera(self.st, identity_motion(FIRST_FRAME)), self.st)

    def test_cut_refuses_compensation(self):
        with self.assertRaises(MotionError):
            self.mm.apply_camera(self.st, identity_motion(CUT, continuity_broken=True))
        with self.assertRaises(MotionError):
            self.mm.apply_camera(self.st, identity_motion(UNKNOWN, continuity_broken=True))


class TestValidation(unittest.TestCase):
    def test_bad_inputs(self):
        mm = MotionModel(CFG)
        st = mm.start(box(300, 360), 1.0)
        for bad in ((0, 0, 0, 10), (0, 0, 10, float("nan")), (0, 0, 1), (5, 5, 1, 10)):
            with self.subTest(bad=bad):
                with self.assertRaises(MotionError):
                    mm.start(bad, 0.0)
        for t in (1.0, 0.5, float("nan")):
            with self.subTest(t=t):
                with self.assertRaises(MotionError):
                    mm.predict(st, t)
                with self.assertRaises(MotionError):
                    mm.update(st, box(300, 360), t)


if __name__ == "__main__":
    unittest.main()

# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2 DEV F9 instruction (Nathan, 2026-11-01 13:18): let the tracker find the target after larger
#                 real movement while LOST without weakening identity trust; expand candidate search, not trust;
#                 test large movement during loss, camera pan, same-kit neighbour, wrong distant candidate, leaving
#                 and re-entering the frame.
"""F9: the LOST search gate grows at a running speed with no cap until the positional prior is dropped (synthetic,
neutral TARGET / OTHER; no benchmark knowledge).

DEV failure reproduced: a LOST target moved ~2.6 h/s (tracker height units); the gate was capped at 3 h after about a
second, so its box was never even evaluated until lost_long_s dropped the positional prior.
"""
import unittest

from player_tracker_rb import target as T
from player_tracker_rb.config import TrackerConfig
from tests_rb import _synth_scene as SS

KIT_A = (200, 60, 40)
SAME_KIT_COS = 0.75
CFG = TrackerConfig.default()
LOST_LONG = CFG.get("lost_long_s")
FPS_SET = (30.0, 15.0, 10.0)


def people(*others):
    return SS.make_people([("TARGET", KIT_A, 1.0)] + list(others))


def trusted_without_target(r):
    return [k for k, f in enumerate(r.frames) if f.identity_trusted
            and "TARGET" not in r.dets[k][f.detection_index].owners]


def bank_only_when_trusted(tc, r):
    for f in r.frames:
        if (f.decision.get("bank") or {}).get("status") == "ACCEPTED":
            tc.assertTrue(f.identity_trusted)


def first_target_candidate(r):
    """(frame, absence_s) of the first REACQUISITION evaluation of the target's own box, or None."""
    for k, f in enumerate(r.frames):
        for cnd in f.decision.get("candidates", []):
            if cnd["role"] == T.REACQUISITION and r.dets[k][cnd["detection_index"]].owners == ("TARGET",):
                return k, f.decision["absence_s"]
    return None


def moved_while_lost(fps, *, gone_s=1.6, speed_h_per_s=2.5, pan_px_per_s=0.0, neighbour=None):
    """The target is seen 1 s, then not detected at all for gone_s while it runs on at speed_h_per_s, then seen again
    (more than the old 3 h cap from where it was last seen). Optional camera pan (compensated) and a neighbour."""
    h = 100.0
    n_seen, n_gone, n_back = int(fps), int(gone_s * fps), int(2 * fps)
    t_back = n_seen / fps + gone_s
    fr = []
    for k in range(n_seen + n_gone + n_back):
        s = k / fps
        # runs at speed_h_per_s while not detected, then jogs at 1 h/s (at 10 fps a sustained 2.5 h/s run breaks
        # the population's frame-to-frame association, a separate, known population limit)
        run_s = min(max(0.0, s - 1.0), t_back - 1.0)
        x = 300 + speed_h_per_s * h * run_s + 1.0 * h * max(0.0, s - t_back) - pan_px_per_s * s
        boxes = {"TARGET": SS.box_at(x, 400, 40, h)}
        if neighbour is not None and k >= n_seen + n_gone:
            boxes["OTHER"] = SS.box_at(x + neighbour, 400, 40, h)
        cam = None
        if pan_px_per_s and k > 0:
            cam = SS.motion_from_matrix([[1, 0, -pan_px_per_s / fps], [0, 1, 0]], SS.W, SS.H)
        fr.append(SS.SceneFrame(boxes, drop=("TARGET",) if n_seen <= k < n_seen + n_gone else (), camera=cam))
    return fr, n_seen + n_gone


class TestLostSearch(unittest.TestCase):
    def test_large_movement_during_loss_is_searched_before_the_prior_is_dropped(self):
        for fps in FPS_SET:
            with self.subTest(fps=fps):
                fr, back = moved_while_lost(fps)
                r = SS.run(fr, people(), fps=fps)
                self.assertEqual(r.frames[back - 1].rb_state, T.LOST)
                got = first_target_candidate(r)
                self.assertIsNotNone(got)
                k, absence = got
                self.assertLess(absence, LOST_LONG, "evaluated while the positional prior still holds")
                self.assertLessEqual(k, back + 1)
                self.assertEqual(r.frames[-1].rb_state, T.TRACKING)
                self.assertEqual(trusted_without_target(r), [])
                self.assertTrue(r.bank_is_pure())
                bank_only_when_trusted(self, r)

    def test_found_is_not_trusted_the_identity_rules_are_unchanged(self):
        # the return is a reacquisition: REACQUIRING (untrusted) for the verification window first, then a new
        # episode that is unverified (no confusable competitor): finding the box never made it trusted
        fps = 30.0
        fr, back = moved_while_lost(fps)
        r = SS.run(fr, people(), fps=fps)
        k, _ = first_target_candidate(r)
        self.assertEqual(r.frames[k].rb_state, T.REACQUIRING)
        self.assertFalse(r.frames[k].identity_trusted)
        win = int(CFG.get("reacq_window_s") * fps)
        self.assertTrue(all(not f.identity_trusted for f in r.frames[back:k + win - 1]))
        self.assertGreaterEqual(r.frames[-1].target_episode, 2)

    def test_camera_pan_during_loss(self):
        for fps in FPS_SET:
            with self.subTest(fps=fps):
                fr, back = moved_while_lost(fps, pan_px_per_s=150.0)
                r = SS.run(fr, people(), fps=fps)
                got = first_target_candidate(r)
                self.assertIsNotNone(got)
                self.assertLess(got[1], LOST_LONG)
                self.assertEqual(trusted_without_target(r), [])
                self.assertTrue(r.bank_is_pure())

    def test_same_kit_neighbour_at_the_return_is_never_trusted(self):
        for fps in FPS_SET:
            with self.subTest(fps=fps):
                fr, back = moved_while_lost(fps, neighbour=60.0)
                r = SS.run(fr, people(("OTHER", KIT_A, SAME_KIT_COS)), fps=fps)
                self.assertEqual(trusted_without_target(r), [])
                self.assertTrue(r.bank_is_pure())
                bank_only_when_trusted(self, r)

    def test_wrong_distant_candidate_is_never_trusted(self):
        # the target never comes back; a same-kit look-alike appears far away (beyond the running-speed gate) while
        # the prior holds, and stays: it is never trusted, and it is not even searched until the gate could reach it
        for fps in FPS_SET:
            with self.subTest(fps=fps):
                n_seen, n_rest = int(fps), int(4 * fps)
                fr = [SS.SceneFrame({"TARGET": SS.box_at(300, 400, 40, 100)}) for _ in range(n_seen)]
                fr += [SS.SceneFrame({"OTHER": SS.box_at(1150, 400, 40, 100)} if k >= int(0.2 * fps) else {})
                       for k in range(n_rest)]
                r = SS.run(fr, people(("OTHER", KIT_A, SAME_KIT_COS + 0.15)), fps=fps)
                self.assertEqual(trusted_without_target(r), [])
                self.assertTrue(r.bank_is_pure())
                for k, f in enumerate(r.frames[n_seen:], start=n_seen):
                    for cnd in f.decision.get("candidates", []):
                        if cnd["role"] == T.REACQUISITION and f.decision.get("gate_h") is not None:
                            self.assertLessEqual(cnd["distance_h"], f.decision["gate_h"] + 1e-6)

    def test_target_leaving_and_re_entering_the_frame(self):
        for fps in FPS_SET:
            with self.subTest(fps=fps):
                fr = [SS.SceneFrame({"TARGET": SS.box_at(1180 + 200 * k / fps, 400, 40, 100)})
                      for k in range(int(0.6 * fps))]
                fr += [SS.SceneFrame({}) for _ in range(int(1.5 * fps))]
                fr += [SS.SceneFrame({"TARGET": SS.box_at(20 + 150 * k / fps, 400, 40, 100)})
                       for k in range(int(3 * fps))]
                r = SS.run(fr, people(), fps=fps)
                self.assertEqual(trusted_without_target(r), [])
                self.assertTrue(r.bank_is_pure())
                self.assertEqual(r.frames[-1].rb_state, T.TRACKING)


if __name__ == "__main__":
    unittest.main()

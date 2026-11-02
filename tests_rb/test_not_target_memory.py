# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2 DEV NOT_TARGET memory instruction (Nathan, 2026-11-02 16:12): remember previously confirmed
#                 non-target people so they can be excluded during target reacquisition; target-centric; NOT_TARGET may
#                 only veto a candidate, never prove TARGET; never learn from ambiguous/untrusted identity; reset or
#                 invalidate across unsafe cuts/merges/identity breaks; preserve F3c size-anomaly safety; test known
#                 non-target returns, same-kit non-target, look-alike, camera cut, merge/anomaly, true target never
#                 excluded.
"""NOT_TARGET memory (target.py, not_target_memory_enabled) on synthetic scenes (neutral TARGET / OTHER)."""
import unittest

from player_tracker_rb import camera as C
from player_tracker_rb import target as T
from player_tracker_rb.config import TrackerConfig
from tests_rb import _synth_scene as SS

KIT_A, KIT_B = (200, 60, 40), (40, 60, 200)
H = 100.0
ON = TrackerConfig.with_overrides({"not_target_memory_enabled": True})
OFF = TrackerConfig.default()
FPS_SET = (30.0, 15.0)


def people(*others):
    return SS.make_people([("TARGET", KIT_A, 1.0)] + list(others))


def wrong_tracking(r):
    """Frames TRACKING (trusted or not) on a box that is not exactly the target."""
    return [k for k, f in enumerate(r.frames) if f.rb_state == T.TRACKING
            and r.dets[k][f.detection_index].owners != ("TARGET",)]


def wrong_trusted(r):
    return [k for k in wrong_tracking(r) if r.frames[k].identity_trusted]


def vetoed(r, owner):
    """Frames where a reacquisition candidate of `owner` was vetoed by the memory."""
    out = []
    for k, f in enumerate(r.frames):
        for nt in (f.decision.get("reacquisition") or {}).get("not_target_memory", []):
            if nt["veto"]:
                for o in r.pops[k].observations:
                    if o.tracklet_id == nt["tracklet"] and r.dets[k][o.index].owners == (owner,):
                        out.append(k)
    return out


def target_vetoed(r):
    return vetoed(r, "TARGET")


def two_losses(fps, *, other=("OTHER", KIT_B, 0.2), walk_to_target=True, second_gap_s=1.0, extra=None):
    """The target is seen beside OTHER (who becomes a known other), is missed long enough to be reacquired (a new
    episode: the known others are reset), then missed again while OTHER walks to where the target stands; the target
    then returns at its place beside OTHER's new position."""
    n1, g1, n2, g2, n3 = int(1.5 * fps), int(0.6 * fps), int(1.5 * fps), int(second_gap_s * fps), int(2.0 * fps)
    fr = []
    for k in range(n1 + g1 + n2 + g2 + n3):
        s = k / fps
        tgt = SS.box_at(400, 400, 40, H)
        if walk_to_target and k >= n1 + g1 + n2:
            ox = max(470.0, 700.0 - 400.0 * (s - (n1 + g1 + n2) / fps))
        else:
            ox = 700.0
        boxes = {"TARGET": tgt, "OTHER": SS.box_at(ox, 400, 40, H)}
        drop = ("TARGET",) if (n1 <= k < n1 + g1) or (n1 + g1 + n2 <= k < n1 + g1 + n2 + g2) else ()
        cam = extra(k) if extra is not None else None
        fr.append(SS.SceneFrame(boxes, drop=drop, camera=cam))
    return fr, (n1, g1, n2, g2, n3)


class TestNotTargetMemory(unittest.TestCase):
    def test_off_by_default_and_identical_to_the_tracker_without_it(self):
        self.assertFalse(OFF.get("not_target_memory_enabled"))
        fr, _ = two_losses(30.0)
        a = SS.run(fr, people(("OTHER", KIT_B, 0.2)), fps=30.0)
        b = SS.run(fr, people(("OTHER", KIT_B, 0.2)), fps=30.0, cfg=TrackerConfig.default())
        self.assertEqual([f.as_record() for f in a.frames], [f.as_record() for f in b.frames])
        self.assertEqual(a.tracker.state_sizes()["target_not_target_memory"], 0)

    def test_known_non_target_returning_after_a_reacquisition_is_still_vetoed(self):
        for fps in FPS_SET:
            with self.subTest(fps=fps):
                fr, (n1, g1, n2, g2, n3) = two_losses(fps)
                r = SS.run(fr, people(("OTHER", KIT_B, 0.2)), fps=fps, cfg=ON)
                self.assertGreaterEqual(max(f.target_episode for f in r.frames), 2, "the first loss was reacquired")
                self.assertTrue([k for k in vetoed(r, "OTHER") if k >= n1 + g1 + n2],
                                "OTHER is vetoed in the second loss, after the known others were reset")
                self.assertEqual(target_vetoed(r), [])
                self.assertEqual(wrong_trusted(r), [])
                self.assertTrue(r.bank_is_pure())

    def test_same_kit_and_look_alike_non_targets_are_vetoed_and_never_taken(self):
        # a same-kit player and a near look-alike walk to the target's place while it is missed: the baseline can lock
        # onto them (untrusted); with the memory they are vetoed and the true target is never vetoed
        for fps in FPS_SET:
            for cos in (0.75, 0.9, 0.97):
                with self.subTest(fps=fps, cos=cos):
                    fr, (n1, g1, n2, g2, n3) = two_losses(fps, other=("OTHER", KIT_A, cos))
                    off = SS.run(fr, people(("OTHER", KIT_A, cos)), fps=fps, cfg=OFF)
                    on = SS.run(fr, people(("OTHER", KIT_A, cos)), fps=fps, cfg=ON)
                    self.assertEqual(target_vetoed(on), [])
                    self.assertEqual(wrong_trusted(on), [])
                    self.assertLessEqual(len(wrong_tracking(on)), len(wrong_tracking(off)))
                    if fps == 30.0 and cos >= 0.9:            # the baseline locks onto the look-alike (untrusted)
                        self.assertGreater(len(wrong_tracking(off)), 0)
                        self.assertEqual(wrong_tracking(on), [])
                    self.assertEqual([f.identity_trusted for f in on.frames], [f.identity_trusted for f in off.frames],
                                     "a veto never changes what is trusted here")
                    self.assertTrue(on.bank_is_pure())

    def test_camera_cut_clears_the_memory(self):
        fps = 30.0
        fr, (n1, g1, n2, g2, n3) = two_losses(fps)
        cut_at = n1 + g1 + n2 // 2

        def cam(k):
            return SS.identity_motion(C.CUT, continuity_broken=True) if k == cut_at else None
        fr, _ = two_losses(fps, extra=cam)
        r = SS.run(fr, people(("OTHER", KIT_B, 0.2)), fps=fps, cfg=ON)
        self.assertEqual([k for k in vetoed(r, "OTHER") if k > cut_at], [], "nothing remembered across a cut")
        self.assertEqual(target_vetoed(r), [])
        self.assertEqual(wrong_trusted(r), [])

    def test_a_continuity_break_clears_every_entry_even_for_live_tracklets(self):
        # defence in depth: the population closes tracklets on a cut, but the memory is cleared regardless
        from player_tracker_rb import population as P
        from player_tracker_rb.appearance import AppearanceExtractor
        from player_tracker_rb.detector import PersonCandidate
        pop = P.Population(ON, SS.W, SS.H)
        tt = T.TargetTracker(ON, pop, AppearanceExtractor(ON, SS.SignatureEmbedding(people(), 0.0)))
        cands = [PersonCandidate(SS.box_at(400, 400, 40, H), 0.9, False, "synthetic")]
        pop.update(0, 0.0, cands, SS.identity_motion(C.FIRST_FRAME))
        res = pop.update(1, 1 / 30.0, cands, SS.identity_motion(C.CUT, continuity_broken=True))
        tid = res.tracklets[0].tracklet_id
        tt._nt_memory = {tid: []}
        tt._nt_prune(res)
        self.assertEqual(tt._nt_memory, {})

    def test_merge_or_size_anomaly_on_the_remembered_tracklet_drops_it(self):
        # OTHER's remembered tracklet is merged with a newcomer into one wide box (F3c: the box may now hold anyone):
        # its entry is dropped and never vetoes again
        fps = 30.0
        fr, (n1, g1, n2, g2, n3) = two_losses(fps, walk_to_target=False)
        m0 = n1 + g1 + n2 // 3
        for k in range(m0, m0 + int(0.5 * fps)):
            b = dict(fr[k].boxes)
            b["THIRD"] = SS.box_at(725, 400, 40, H)
            fr[k] = SS.SceneFrame(b, drop=fr[k].drop, merge=[("OTHER", "THIRD")])
        r = SS.run(fr, people(("OTHER", KIT_B, 0.2), ("THIRD", KIT_A, 0.9)), fps=fps, cfg=ON)
        self.assertEqual(r.tracker.state_sizes()["target_not_target_memory"], 0)
        self.assertEqual([k for k in vetoed(r, "OTHER") if k >= m0], [])
        self.assertEqual(target_vetoed(r), [])
        self.assertEqual(wrong_trusted(r), [])

    def test_true_target_carried_onto_a_remembered_tracklet_is_never_vetoed(self):
        # OTHER (remembered) leaves at the image edge and the target appears at the very same spot on the next frame:
        # the population carries OTHER's tracklet onto the target; the appearance guard withdraws the veto
        for fps in FPS_SET:
            with self.subTest(fps=fps):
                n1, g1 = int(1.5 * fps), int(0.6 * fps)
                fr = []
                for k in range(n1):
                    fr.append(SS.SceneFrame({"TARGET": SS.box_at(600, 400, 40, H), "OTHER": SS.box_at(25, 400, 40, H)}))
                for k in range(g1):                       # target missed -> reacquired later (episode 2)
                    fr.append(SS.SceneFrame({"TARGET": SS.box_at(600, 400, 40, H), "OTHER": SS.box_at(25, 400, 40, H)},
                                            drop=("TARGET",)))
                for k in range(int(1.0 * fps)):
                    fr.append(SS.SceneFrame({"TARGET": SS.box_at(600, 400, 40, H), "OTHER": SS.box_at(25, 400, 40, H)}))
                # the target walks off at the edge where OTHER stood and OTHER leaves; then the swap at the edge
                for k in range(3):
                    fr.append(SS.SceneFrame({"OTHER": SS.box_at(25 - 8 * k, 400, 40, H)}))
                for k in range(int(2.5 * fps)):
                    fr.append(SS.SceneFrame({"TARGET": SS.box_at(5 + 4 * k, 400, 40, H)}))
                r = SS.run(fr, people(("OTHER", KIT_B, 0.2)), fps=fps, cfg=ON)
                self.assertEqual(target_vetoed(r), [], "the true target is never vetoed")
                self.assertEqual(wrong_trusted(r), [])
                self.assertTrue(r.bank_is_pure())

    def test_nothing_is_learned_while_the_target_is_untrusted(self):
        # OTHER only ever appears while the target is not trusted (it is missed): never remembered
        fps = 30.0
        n1 = int(1.0 * fps)
        fr = [SS.SceneFrame({"TARGET": SS.box_at(400, 400, 40, H)}) for _ in range(n1)]
        fr += [SS.SceneFrame({"TARGET": SS.box_at(400, 400, 40, H), "OTHER": SS.box_at(800, 400, 40, H)},
                             drop=("TARGET",)) for _ in range(int(0.5 * fps))]
        fr += [SS.SceneFrame({"TARGET": SS.box_at(400, 400, 40, H)}) for _ in range(int(1.0 * fps))]
        r = SS.run(fr, people(("OTHER", KIT_B, 0.2)), fps=fps, cfg=ON)
        self.assertEqual(r.tracker._nt_memory, {})

    def test_memory_is_bounded_by_live_tracklets(self):
        fps = 15.0
        fr = []
        for k in range(int(30 * fps)):
            b = {"TARGET": SS.box_at(400, 400, 40, H)}
            if (k // int(fps)) % 2 == 0:
                b["OTHER"] = SS.box_at(900, 400, 40, H)
            fr.append(SS.SceneFrame(b))
        r = SS.run(fr, people(("OTHER", KIT_B, 0.2)), fps=fps, cfg=ON)
        self.assertLessEqual(r.tracker.state_sizes()["target_not_target_memory"], 1)


if __name__ == "__main__":
    unittest.main()

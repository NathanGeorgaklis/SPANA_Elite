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
ON_F5 = TrackerConfig.with_overrides({"not_target_memory_enabled": True, "target_reverify_enabled": True})
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


def edge_carry(fps):
    """OTHER (remembered) leaves at the image edge and the target appears at the very same spot next: the population
    can carry OTHER's tracklet onto the target."""
    n1, g1 = int(1.5 * fps), int(0.6 * fps)
    fr = []
    for k in range(n1):
        fr.append(SS.SceneFrame({"TARGET": SS.box_at(600, 400, 40, H), "OTHER": SS.box_at(25, 400, 40, H)}))
    for k in range(g1):                                   # target missed -> reacquired later (episode 2)
        fr.append(SS.SceneFrame({"TARGET": SS.box_at(600, 400, 40, H), "OTHER": SS.box_at(25, 400, 40, H)},
                                drop=("TARGET",)))
    for k in range(int(1.0 * fps)):
        fr.append(SS.SceneFrame({"TARGET": SS.box_at(600, 400, 40, H), "OTHER": SS.box_at(25, 400, 40, H)}))
    for k in range(3):                                    # the target walks off; OTHER leaves at the edge
        fr.append(SS.SceneFrame({"OTHER": SS.box_at(25 - 8 * k, 400, 40, H)}))
    for k in range(int(2.5 * fps)):                       # the target appears where OTHER left
        fr.append(SS.SceneFrame({"TARGET": SS.box_at(5 + 4 * k, 400, 40, H)}))
    return fr


class TestNotTargetMemory(unittest.TestCase):
    def test_off_by_default_records_nothing_and_vetoes_nothing(self):
        self.assertFalse(OFF.get("not_target_memory_enabled"))
        fr, _ = two_losses(30.0)
        off = SS.run(fr, people(("OTHER", KIT_B, 0.2)), fps=30.0, cfg=OFF)
        on = SS.run(fr, people(("OTHER", KIT_B, 0.2)), fps=30.0, cfg=ON)
        self.assertTrue(vetoed(on, "OTHER"), "the same scene vetoes with the memory on")
        self.assertEqual(vetoed(off, "OTHER"), [])
        self.assertTrue(all(not (f.decision.get("reacquisition") or {}).get("not_target_memory")
                            for f in off.frames), "off: the memory is never consulted")
        self.assertEqual(off.tracker._nt_memory, {})

    def test_before_the_first_reset_the_memory_changes_no_decision(self):
        # memory is a subset of the known others until a reacquisition resets them: identical decisions
        fps = 30.0
        n1, g1, n3 = int(1.5 * fps), int(0.6 * fps), int(1.5 * fps)
        fr = [SS.SceneFrame({"TARGET": SS.box_at(400, 400, 40, H), "OTHER": SS.box_at(700, 400, 40, H)},
                            drop=("TARGET",) if n1 <= k < n1 + g1 else ()) for k in range(n1 + g1 + n3)]
        off = SS.run(fr, people(("OTHER", KIT_A, 0.9)), fps=fps, cfg=OFF)
        on = SS.run(fr, people(("OTHER", KIT_A, 0.9)), fps=fps, cfg=ON)

        def strip(f):
            d = f.as_record()
            (d.get("decision", {}).get("reacquisition") or {}).get("log", {}).pop("not_target_memory", None)
            (d.get("decision", {}).get("reacquisition") or {}).pop("not_target_memory", None)
            return d
        self.assertEqual([strip(f) for f in on.frames], [strip(f) for f in off.frames])

    def test_known_non_target_returning_after_a_reacquisition_is_still_vetoed(self):
        for fps in FPS_SET:
          for cfg in (ON, ON_F5):
            with self.subTest(fps=fps, f5=cfg.get("target_reverify_enabled")):
                fr, (n1, g1, n2, g2, n3) = two_losses(fps)
                r = SS.run(fr, people(("OTHER", KIT_B, 0.2)), fps=fps, cfg=cfg)
                self.assertGreaterEqual(max(f.target_episode for f in r.frames), 2, "the first loss was reacquired")
                self.assertTrue([k for k in vetoed(r, "OTHER") if k >= n1 + g1 + n2],
                                "OTHER is vetoed in the second loss, after the known others were reset")
                self.assertEqual(target_vetoed(r), [])
                self.assertEqual(wrong_trusted(r), [])
                self.assertTrue(r.bank_is_pure())

    def test_same_kit_and_look_alike_non_targets_are_vetoed_and_never_taken(self):
        # a same-kit player and a near look-alike walk to the target's place while it is missed: the baseline can lock
        # onto them (untrusted). A separable one is vetoed; a near twin (0.97) is NOT (no positive evidence: the veto
        # gives way to safety) and is no worse than without the memory; the true target is never vetoed
        for fps in FPS_SET:
            for cos in (0.75, 0.9, 0.97):
                with self.subTest(fps=fps, cos=cos):
                    fr, (n1, g1, n2, g2, n3) = two_losses(fps, other=("OTHER", KIT_A, cos))
                    off = SS.run(fr, people(("OTHER", KIT_A, cos)), fps=fps, cfg=OFF)
                    on = SS.run(fr, people(("OTHER", KIT_A, cos)), fps=fps, cfg=ON)
                    self.assertEqual(target_vetoed(on), [])
                    self.assertEqual(wrong_trusted(on), [])
                    self.assertLessEqual(len(wrong_tracking(on)), len(wrong_tracking(off)))
                    if fps == 30.0 and cos == 0.9:            # the baseline locks onto the look-alike (untrusted)
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
        # (review HIGH-1: a same-kit near twin carried onto the target scored 1.0 vs 0.98 and stayed vetoed)
        for fps in FPS_SET:
          for other in (("OTHER", KIT_B, 0.2), ("OTHER", KIT_A, 0.97), ("OTHER", KIT_A, 0.9)):
            with self.subTest(fps=fps, other=other):
                fr = edge_carry(fps)
                r = SS.run(fr, people(other), fps=fps, cfg=ON)
                self.assertEqual(target_vetoed(r), [], "the true target is never vetoed")
                off = SS.run(fr, people(other), fps=fps, cfg=OFF)
                self.assertEqual([f.rb_state for f in r.frames], [f.rb_state for f in off.frames])
                self.assertEqual(wrong_trusted(r), [])
                self.assertTrue(r.bank_is_pure())

    def test_a_vetoed_box_remains_a_rival(self):
        # review HIGH-2: even a WRONG veto of the true target (forced here) must not let a worse look-alike Z be
        # reacquired: the vetoed box can never be accepted, but it still blocks Z as a rival. (Without the memory the
        # target itself is reacquired here.)
        for fps in FPS_SET:
            with self.subTest(fps=fps):
                n1, g1, n3 = int(1.5 * fps), int(1.0 * fps), int(2.0 * fps)
                fr = []
                for k in range(n1 + g1 + n3):
                    b = {"TARGET": SS.box_at(400, 400, 40, H)}
                    if k >= n1 + g1:
                        b["Z"] = SS.box_at(650, 400, 40, H)    # appears with the target's return: never known
                    fr.append(SS.SceneFrame(b, drop=("TARGET",) if n1 <= k < n1 + g1 else ()))
                pp = people(("Z", KIT_A, 0.8))
                off = SS.run(fr, pp, fps=fps, cfg=OFF)
                self.assertTrue([k for k in off.tracking_frames() if k >= n1 + g1], "control: reacquired when off")
                real = T.TargetTracker._nt_veto

                def wrong_veto(self_, cache, res, image, o):
                    if abs((o.bbox[0] + o.bbox[2]) / 2 - 400) <= 2:   # the true target's box
                        return {"tracklet": o.tracklet_id, "target_similarity": None,
                                "own_snapshot_similarity": None, "veto": True}
                    return real(self_, cache, res, image, o)
                T.TargetTracker._nt_veto = wrong_veto
                try:
                    r = SS.run(fr, pp, fps=fps, cfg=ON)
                finally:
                    T.TargetTracker._nt_veto = real
                self.assertTrue(target_vetoed(r), "the forced veto applied")
                self.assertEqual(wrong_tracking(r), [], "Z is never taken")
                self.assertEqual(wrong_trusted(r), [])
                self.assertTrue(r.bank_is_pure())

    def test_a_look_alike_of_both_is_never_vetoed(self):
        # positive evidence only: a crop matching its snapshot no better than the target bank is not vetoed
        fr, _ = two_losses(30.0)
        pp = people(("OTHER", KIT_B, 0.2))
        r = SS.run(fr, pp, fps=30.0, cfg=ON)
        k = len(fr) - 1
        res, img = r.pops[k], SS.render(fr[k], pp)
        o = next(o for o in res.observations if o.tracklet_id and r.dets[k][o.index].owners == ("TARGET",))
        tt = r.tracker
        tt._nt_memory = {o.tracklet_id: list(tt.bank.samples)}   # a snapshot exactly as similar as the target bank
        nt = tt._nt_veto({}, res, img, o)
        self.assertFalse(nt["veto"], nt)

    def test_partial_edge_and_reassociated_boxes_drop_the_entry_and_are_not_recorded(self):
        import dataclasses
        from player_tracker_rb import population as P
        fr, _ = two_losses(30.0)
        pp = people(("OTHER", KIT_B, 0.2))
        r = SS.run(fr, pp, fps=30.0, cfg=ON)
        k = len(fr) - 1
        res = r.pops[k]
        o = next(o for o in res.observations if o.tracklet_id and r.dets[k][o.index].owners == ("OTHER",))
        tt = r.tracker
        tt._target_tracklets = set()
        self.assertTrue(tt._nt_recordable(res, o), "control: a clean whole body is recordable")
        for flag in (P.PARTIAL, P.TOUCHES_IMAGE_EDGE, P.REASSOCIATED_AFTER_GAP):
            with self.subTest(flag=flag):
                ob = dataclasses.replace(o, flags=o.flags + (flag,))
                obs = tuple(ob if x.index == o.index else x for x in res.observations)
                fres = dataclasses.replace(res, observations=obs)
                tt._nt_memory = {o.tracklet_id: []}
                tt._nt_prune(fres)
                self.assertEqual(tt._nt_memory, {})
                self.assertFalse(tt._nt_recordable(fres, ob))
        v = res.tracklet(o.tracklet_id)
        lin = dataclasses.replace(res, tracklets=tuple(dataclasses.replace(x, flags=x.flags + (P.MERGE_LINEAGE,))
                                                       if x.tracklet_id == o.tracklet_id else x
                                                       for x in res.tracklets))
        self.assertIsNotNone(v)
        self.assertFalse(tt._nt_recordable(lin, o), "merge lineage: never recorded")

    def test_an_empty_snapshot_never_vetoes(self):
        fr, _ = two_losses(30.0)
        pp = people(("OTHER", KIT_B, 0.2))
        r = SS.run(fr, pp, fps=30.0, cfg=ON)
        k = len(fr) - 1
        res, img = r.pops[k], SS.render(fr[k], pp)
        o = next(o for o in res.observations if o.tracklet_id and r.dets[k][o.index].owners == ("OTHER",))
        tt = r.tracker
        tt._nt_memory = {o.tracklet_id: list(tt._pop.bank_of(o.tracklet_id).samples)}
        self.assertTrue(tt._nt_veto({}, res, img, o)["veto"], "control: its own snapshot vetoes it")
        tt._nt_memory = {o.tracklet_id: []}
        self.assertFalse(tt._nt_veto({}, res, img, o)["veto"])

    def test_prune_drops_merge_lineage_and_target_tracklets_and_keeps_clean_entries(self):
        import dataclasses
        from player_tracker_rb import population as P
        from player_tracker_rb.appearance import AppearanceExtractor
        from player_tracker_rb.detector import PersonCandidate
        pop = P.Population(ON, SS.W, SS.H)
        tt = T.TargetTracker(ON, pop, AppearanceExtractor(ON, SS.SignatureEmbedding(people(), 0.0)))
        cands = [PersonCandidate(SS.box_at(400, 400, 40, H), 0.9, False, "synthetic")]
        pop.update(0, 0.0, cands, SS.identity_motion(C.FIRST_FRAME))
        res = pop.update(1, 1 / 30.0, cands, SS.identity_motion(C.FIRST_FRAME))
        tid = res.tracklets[0].tracklet_id
        tt._nt_memory = {tid: []}
        tt._nt_prune(res)
        self.assertEqual(list(tt._nt_memory), [tid], "control: a clean live entry is kept")
        lin = dataclasses.replace(res, tracklets=(dataclasses.replace(res.tracklets[0],
                                                                      flags=res.tracklets[0].flags + (P.MERGE_LINEAGE,)),))
        tt._nt_prune(lin)
        self.assertEqual(tt._nt_memory, {})
        tt._nt_memory = {tid: []}
        tt._target_tracklets = {tid}
        tt._nt_prune(res)
        self.assertEqual(tt._nt_memory, {}, "a tracklet the target was confirmed on is never another person")

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

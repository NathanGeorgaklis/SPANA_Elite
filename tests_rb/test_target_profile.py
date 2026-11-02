# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2 DEV positive target evidence instruction (Nathan, 2026-11-02 17:46): a target-reference layer from
#                 the user-selected player and further clean trusted views; diverse anchors instead of near-duplicates;
#                 candidates compared against the full trusted profile (best match AND consistency across anchors);
#                 kit / number support only; temporal consistency; tracker unchanged; test pose/view change, same-kit
#                 look-alike, occlusion and return, image edge, appearance drift, a wrong candidate matching only one
#                 anchor, the true target matching several anchors consistently.
"""Target reference profile (player_tracker_rb.target_profile) on synthetic scenes and synthetic features."""
import unittest

import numpy as np

from player_tracker_rb import camera as C
from player_tracker_rb import target as T
from player_tracker_rb import target_profile as TP
from player_tracker_rb.appearance import AppearanceFeature, CropQuality, l2_normalise
from player_tracker_rb.config import TrackerConfig
from tests_rb import _synth_scene as SS

KIT_A, KIT_B = (200, 60, 40), (40, 60, 200)
H = 100.0
CFG = TrackerConfig.default()
FPS_SET = (30.0, 15.0)


def people(*others):
    return SS.make_people([("TARGET", KIT_A, 1.0)] + list(others))


def run(frames, pp, fps, *, pose_gain=0.0, reader=None, cfg=CFG):
    out = {"frames": [], "prof": None}

    def obs(k, res, img, tf, tt, ext):
        if out["prof"] is None:
            out["prof"] = TP.TargetProfile(cfg, ext, number_reader=reader)
        out["frames"].append(out["prof"].update(res, img, tf, tt.bank))
    r = SS.run(frames, pp, fps=fps, pose_gain=pose_gain, observer=obs, cfg=cfg)
    return r, out["prof"], out["frames"]


def owner(r, k, index):
    return r.dets[k][index].owners


def supported(r, pf, name):
    """Frames where the profile verdict is SUPPORTED on `name`'s box."""
    return [k for k, f in enumerate(pf) if f["verdict"]["status"] == TP.SUPPORTED
            and owner(r, k, f["verdict"]["index"]) == (name,)]


def anchors_only_from_trusted_bank_accepts(tc, r, prof):
    for a in prof.anchors:
        tf = r.frames[a.rel_frame]
        tc.assertTrue(tf.identity_trusted, f"anchor from untrusted frame {a.rel_frame}")
        tc.assertEqual((tf.decision.get("bank") or {}).get("status"), "ACCEPTED", a.rel_frame)
        tc.assertEqual(owner(r, a.rel_frame, tf.detection_index), ("TARGET",))


def feat(emb, lab):
    q = CropQuality(H, 1.0, 0.0, 1.0, True, ())
    return AppearanceFeature(lab=lab, embedding=l2_normalise(np.asarray(emb, float)), quality=q)


def lab_of(seed):
    rng = np.random.default_rng(seed)
    x = rng.random((2, 64))
    return x / x.sum(axis=1, keepdims=True)


class TestTargetProfile(unittest.TestCase):
    def test_tracker_output_is_identical_with_the_profile(self):
        fr = [SS.SceneFrame({"TARGET": SS.box_at(400 + 2 * k, 400, 40, H), "OTHER": SS.box_at(700, 400, 40, H)},
                            drop=("TARGET",) if 40 <= k < 60 else ()) for k in range(120)]
        pp = people(("OTHER", KIT_A, 0.9))
        a = SS.run(fr, pp, fps=30.0)
        b, prof, _ = run(fr, pp, 30.0)
        self.assertEqual([f.as_record() for f in a.frames], [f.as_record() for f in b.frames])
        self.assertEqual(len(a.tracker.bank.samples), len(b.tracker.bank.samples))
        anchors_only_from_trusted_bank_accepts(self, b, prof)

    def test_pose_view_change_collects_diverse_anchors_and_keeps_matching(self):
        for fps in FPS_SET:
            with self.subTest(fps=fps):
                n = int(4.0 * fps)
                fr = [SS.SceneFrame({"TARGET": SS.box_at(400, 400, 30 + 40 * k / n, H)}) for k in range(n)]
                r, prof, pf = run(fr, people(), fps, pose_gain=0.6)
                anchors_only_from_trusted_bank_accepts(self, r, prof)
                self.assertGreater(len(prof.anchors), 1, "several views collected")
                self.assertLessEqual(len(prof.anchors), CFG.get("profile_max_anchors"))
                asp = [a.aspect for a in prof.anchors]
                self.assertGreater(max(asp) - min(asp), 0.1, "the anchors span the pose range")
                self.assertEqual(prof.anchors[0].source, "SEED")
                self.assertTrue(supported(r, pf, "TARGET"))

    def test_near_duplicate_views_are_not_kept(self):
        fps = 15.0
        fr = [SS.SceneFrame({"TARGET": SS.box_at(400, 400, 40, H)}) for _ in range(int(6 * fps))]
        r, prof, _ = run(fr, people(), fps)
        self.assertEqual(len(prof.anchors), 1)
        self.assertEqual(prof.decisions.get(TP.ADDED), 1)

    def test_anchor_set_is_bounded_and_keeps_the_seed(self):
        cfg = TrackerConfig.with_overrides({"profile_max_anchors": 2})
        fps = 15.0
        n = int(6.0 * fps)
        fr = [SS.SceneFrame({"TARGET": SS.box_at(400, 400, 25 + 50 * k / n, H)}) for k in range(n)]
        r, prof, _ = run(fr, people(), fps, pose_gain=0.8, cfg=cfg)
        self.assertEqual(len(prof.anchors), 2)
        self.assertEqual(prof.anchors[0].rel_frame, 0)

    def test_same_kit_look_alike_is_never_supported_beside_the_target(self):
        for fps in FPS_SET:
            for cos in (0.8, 0.9, 0.97):
                with self.subTest(fps=fps, cos=cos):
                    n1, g, n3 = int(1.5 * fps), int(1.0 * fps), int(2.0 * fps)
                    fr = []
                    for k in range(n1 + g + n3):
                        b = {"TARGET": SS.box_at(400, 400, 40, H)}
                        if k >= n1:
                            b["LOOK"] = SS.box_at(700, 400, 40, H)
                        fr.append(SS.SceneFrame(b, drop=("TARGET",) if n1 <= k < n1 + g else ()))
                    r, prof, pf = run(fr, people(("LOOK", KIT_A, cos)), fps)
                    seen = [k for k in range(len(fr)) if any(d.owners == ("TARGET",) for d in r.dets[k])]
                    self.assertEqual([k for k in supported(r, pf, "LOOK") if k in seen], [])
                    anchors_only_from_trusted_bank_accepts(self, r, prof)
                    if cos == 0.97:                      # a near twin: the target is not supported beside it either
                        self.assertEqual([k for k in supported(r, pf, "TARGET") if k >= n1], [])

    @unittest.expectedFailure
    def test_residual_a_never_seen_look_alike_alone_in_view_is_not_supported(self):
        # open world: while the target is missed, a same-kit player (cos 0.8, composite above the reacquisition floor)
        # alone in view has no rival; appearance evidence alone cannot reject it (documented residual)
        fps = 30.0
        n1, g = int(1.5 * fps), int(1.0 * fps)
        fr = [SS.SceneFrame({"TARGET": SS.box_at(400, 400, 40, H), **({"LOOK": SS.box_at(700, 400, 40, H)}
                             if k >= n1 else {})}, drop=("TARGET",) if k >= n1 else ()) for k in range(n1 + g)]
        r, prof, pf = run(fr, people(("LOOK", KIT_A, 0.8)), fps)
        self.assertEqual(supported(r, pf, "LOOK"), [])

    def test_a_tracklet_carried_onto_another_body_is_not_supported_by_its_past(self):
        # a different player steps onto the target's spot as the target vanishes: the population may continue the
        # target's tracklet on the new box; its past matches must not support it
        for fps in FPS_SET:
            with self.subTest(fps=fps):
                n1, n2 = int(1.5 * fps), int(1.5 * fps)
                fr = [SS.SceneFrame({"TARGET": SS.box_at(400, 400, 40, H)}) for _ in range(n1)]
                fr += [SS.SceneFrame({"OTHER": SS.box_at(402, 400, 40, H)}) for _ in range(n2)]
                r, prof, pf = run(fr, people(("OTHER", KIT_B, 0.2)), fps)
                self.assertEqual(supported(r, pf, "OTHER"), [])

    def test_occlusion_and_return_needs_fresh_temporal_evidence(self):
        for fps in FPS_SET:
            with self.subTest(fps=fps):
                n1, g, n3 = int(1.5 * fps), int(1.0 * fps), int(1.5 * fps)
                fr = [SS.SceneFrame({"TARGET": SS.box_at(400, 400, 40, H), "OCC": SS.box_at(400, 405, 44, H)},
                                    order=("TARGET", "OCC"), drop=("OCC",) if not (n1 <= k < n1 + g) else ())
                      if n1 <= k < n1 + g else SS.SceneFrame({"TARGET": SS.box_at(400, 400, 40, H)})
                      for k in range(n1 + g + n3)]
                r, prof, pf = run(fr, people(("OCC", KIT_B, 0.2)), fps)
                anchors_only_from_trusted_bank_accepts(self, r, prof)
                self.assertTrue(all(a.rel_frame < n1 or a.rel_frame >= n1 + g for a in prof.anchors),
                                "no anchor while occluded")
                self.assertEqual([k for k in supported(r, pf, "OCC")], [])
                back = [k for k in supported(r, pf, "TARGET") if k >= n1 + g]
                self.assertTrue(back, "supported again after the return")

    def test_target_at_the_image_edge_adds_no_anchor(self):
        fps = 30.0
        fr = [SS.SceneFrame({"TARGET": SS.box_at(400, 400, 40, H)}) for _ in range(int(0.5 * fps))]
        fr += [SS.SceneFrame({"TARGET": SS.box_at(max(10.0, 400 - 30 * k), 400, 40, H)}) for k in range(int(3 * fps))]
        r, prof, _ = run(fr, people(), fps, pose_gain=0.6)
        anchors_only_from_trusted_bank_accepts(self, r, prof)
        for a in prof.anchors:
            box = r.frames[a.rel_frame].bbox
            self.assertGreater(box[0], 2, "no anchor from an edge crop")

    def test_appearance_drift_while_untrusted_never_enters_the_profile(self):
        # the target is contested (overlapping another body) while its pose drifts: those views are untrusted
        fps = 30.0
        n1, n2, n3 = int(1.0 * fps), int(1.5 * fps), int(1.0 * fps)
        fr = [SS.SceneFrame({"TARGET": SS.box_at(400, 400, 40, H)}) for _ in range(n1)]
        fr += [SS.SceneFrame({"TARGET": SS.box_at(400, 400, 40 + 30 * k / n2, H), "OTHER": SS.box_at(420, 400, 40, H)})
               for k in range(n2)]
        fr += [SS.SceneFrame({"TARGET": SS.box_at(400, 400, 70, H)}) for _ in range(n3)]
        r, prof, _ = run(fr, people(("OTHER", KIT_A, 0.85)), fps, pose_gain=0.6)
        anchors_only_from_trusted_bank_accepts(self, r, prof)
        for a in prof.anchors:
            self.assertFalse(n1 <= a.rel_frame < n1 + n2 and not r.frames[a.rel_frame].identity_trusted)

    def test_a_wrong_candidate_matching_only_one_anchor_is_not_consistent(self):
        # four trusted views of the target (shared identity, different poses); Z is another player (cos 0.8) seen in
        # the pose of anchor 1 only: Z's BEST match beats the true target's, its consistency does not
        lab = lab_of(1)
        g = 0.5
        poses = np.eye(12)[4:9]
        ident_t = np.eye(12)[0]
        ident_z = 0.8 * np.eye(12)[0] + np.sqrt(1 - 0.64) * np.eye(12)[1]
        prof = TP.TargetProfile(CFG, None)
        for i in range(4):
            self.assertEqual(prof.offer(feat(ident_t + g * poses[i], lab), (0, 0, 40, 100), i, i, "TRUSTED_BANK"),
                             TP.ADDED)
        true_new_pose = prof.match(feat(ident_t + g * poses[4], lab))
        z = prof.match(feat(ident_z + g * poses[0], lab))
        self.assertGreater(z.best, true_new_pose.best, "best match alone prefers the wrong player")
        self.assertGreaterEqual(true_new_pose.topk, z.topk + CFG.get("ambiguity_margin"))
        self.assertEqual(true_new_pose.support, 1.0, "the true target matches every anchor")
        self.assertLess(z.support, CFG.get("profile_support_min_fraction"))

    def _rows(self, entries):
        return [{"index": i, "tracklet": f"T{i}", "match": e.get("now", {"topk": e["topk"], "support": e["support"]}),
                 "temporal": e, "number": e.get("n", TP.UNKNOWN)} for i, e in enumerate(entries)]

    def test_verdict_needs_every_cue_and_a_margin(self):
        prof = TP.TargetProfile(CFG, None)
        prof.offer(feat(np.eye(12)[0], lab_of(1)), (0, 0, 40, 100), 0, 0.0, "SEED")
        good = {"topk": 0.95, "support": 1.0, "kit": 0.9}
        self.assertEqual(prof._verdict(self._rows([good]))["status"], TP.SUPPORTED)
        cases = {"INCONSISTENT_ACROSS_ANCHORS": dict(good, support=0.25),
                 "KIT_INCOMPATIBLE": dict(good, kit=0.1),
                 "TOPK_BELOW_REACQ_FLOOR": dict(good, topk=0.82),
                 "NUMBER_DIFFERENT": dict(good, n="DIFFERENT")}
        for why, e in cases.items():
            with self.subTest(why=why):
                v = prof._verdict(self._rows([e]))
                self.assertNotEqual(v["status"], TP.SUPPORTED)
                self.assertIn(why, v["reasons"])
        v = prof._verdict(self._rows([dict(good, now={"topk": 0.6, "support": 0.0})]))
        self.assertNotEqual(v["status"], TP.SUPPORTED, "good past evidence never outvotes a poor current box")
        v = prof._verdict(self._rows([good, dict(good, topk=0.92)]))
        self.assertEqual(v["status"], TP.AMBIGUOUS)
        self.assertIn("RIVAL_WITHIN_MARGIN", v["reasons"])
        self.assertEqual(prof._verdict(self._rows([dict(good, n="SAME", topk=0.82)]))["status"], TP.AMBIGUOUS,
                         "an equal number never proves the target")
        self.assertEqual(TP.TargetProfile(CFG, None)._verdict(self._rows([good]))["status"], TP.UNKNOWN,
                         "no anchors: UNKNOWN")

    def test_kit_is_support_only(self):
        prof = TP.TargetProfile(CFG, None)
        prof.offer(feat(np.eye(12)[0], lab_of(1)), (0, 0, 40, 100), 0, 0.0, "SEED")
        same_emb_other_kit = prof.match(feat(np.eye(12)[0], lab_of(99)))
        same_kit_other_player = prof.match(feat(np.eye(12)[2], lab_of(1)))
        self.assertLess(same_kit_other_player.topk, CFG.get("appearance_floor"), "kit alone never supports")
        self.assertIsNotNone(same_emb_other_kit.kit)

    def test_number_reader_vetoes_a_confirmed_different_number(self):
        fps = 15.0
        n = int(3.0 * fps)
        fr = [SS.SceneFrame({"TARGET": SS.box_at(400, 400, 30 + 30 * k / n, H),
                             "LOOK": SS.box_at(700, 400, 30 + 30 * k / n, H)}) for k in range(n)]

        def reader(image, box):
            return "7" if box[0] < 550 else "9"
        r, prof, pf = run(fr, people(("LOOK", KIT_A, 0.99)), fps, pose_gain=0.6, reader=reader)
        look = [c["number"] for f in pf for c in f["candidates"] if owner(r, f["rel_frame"], c["index"]) == ("LOOK",)]
        self.assertIn("DIFFERENT", look)
        self.assertEqual(supported(r, pf, "LOOK"), [])

    def test_cut_clears_histories_and_memory_is_bounded(self):
        fps = 15.0
        fr = []
        for k in range(int(20 * fps)):
            b = {"TARGET": SS.box_at(400, 400, 40, H)}
            if (k // int(fps)) % 2 == 0:
                b["OTHER"] = SS.box_at(900, 400, 40, H)
            cam = SS.identity_motion(C.CUT, continuity_broken=True) if k == int(10 * fps) else None
            fr.append(SS.SceneFrame(b, camera=cam))
        sizes = []

        def obs(k, res, img, tf, tt, ext, state={}):
            if "p" not in state:
                state["p"] = TP.TargetProfile(CFG, ext)
            f = state["p"].update(res, img, tf, tt.bank)
            sizes.append((k, state["p"].state_sizes(), f))
        SS.run(fr, people(("OTHER", KIT_B, 0.2)), fps=fps, observer=obs)
        cut = int(10 * fps)
        self.assertTrue(all(c["temporal"] is None for c in sizes[cut][2]["candidates"]), "nothing carried over a cut")
        win = CFG.get("profile_temporal_window_s")
        self.assertTrue(all(s["profile_histories"] <= 2 and s["profile_history_samples"] <= 2 * (win * fps + 2)
                            for _, s, _ in sizes))


if __name__ == "__main__":
    unittest.main()

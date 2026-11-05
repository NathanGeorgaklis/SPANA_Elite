# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2 DEV real-user reacquisition protocol (Nathan, 2026-11-06 00:39): the user taps the player; snap to
#                 exactly one existing detection; normal re-seed validation; refuse multiple / merged / nested; "I can't
#                 identify" continues automatically; tests for centre, shoulder / edge, nearby same-kit, crowd / overlap,
#                 wrong tap, target not visible.
"""Tap-to-detection snapping (checkpoint.snap_tap) and tap-driven re-seeds on synthetic scenes."""
import unittest
from types import SimpleNamespace as NS

from player_tracker_rb import checkpoint as CK
from player_tracker_rb import population as P
from tests_rb import _synth_scene as SS
from tests_rb.test_identity_checkpoint import (FPS, H, KIT_A, Oracle, people, persistent_look_change, run, strip)


def obs(boxes, flags=None, nested=None):
    out = []
    for k, b in enumerate(boxes):
        out.append(NS(index=k, is_body=True, bbox=b, tracklet_id=f"T{k}", flags=set((flags or {}).get(k, ())),
                      nested_in=(nested or {}).get(k)))
    return out


def group_fn(res, i):
    o = res.observations
    root = i
    while o[root].nested_in is not None:
        root = o[root].nested_in
    return {root} | {x.index for x in o if x.nested_in == root}


class TestSnap(unittest.TestCase):
    def snap(self, boxes, x, y, **kw):
        res = NS(observations=obs(boxes, **kw))
        o, why = CK.snap_tap(res, x, y, group_fn)
        return (None if o is None else o.index), why

    def test_centre_and_edge_taps_snap_to_the_one_player(self):
        a = SS.box_at(400, 400, 40, H)
        self.assertEqual(self.snap([a], 400, 400)[0], 0)
        self.assertEqual(self.snap([a], a[0], a[1])[0], 0)              # exact corner still inside
        self.assertEqual(self.snap([a], a[0] + 3, a[1] + 20)[0], 0)      # shoulder
        k, why = self.snap([a], a[2] + 1, 400)
        self.assertIsNone(k); self.assertTrue(why.startswith("TAP_OFF_PLAYER"))

    def test_overlap_of_two_players_is_ambiguous(self):
        a, b = SS.box_at(400, 400, 40, H), SS.box_at(425, 400, 40, H)
        k, why = self.snap([a, b], 412, 400)
        self.assertIsNone(k); self.assertTrue(why.startswith("TAP_AMBIGUOUS"))
        self.assertEqual(self.snap([a, b], 385, 400)[0], 0)              # tap on the non-overlapping part

    def test_nested_part_box_snaps_to_the_host(self):
        host, part = SS.box_at(400, 400, 40, H), SS.box_at(400, 380, 24, 40)
        k, why = self.snap([host, part], 400, 380, nested={1: 0})
        self.assertEqual(k, 0, why)

    def test_merged_detection_is_refused(self):
        a = SS.box_at(400, 400, 70, H)
        k, why = self.snap([a], 400, 400, flags={0: (P.MERGED_SUSPECTED,)})
        self.assertIsNone(k); self.assertTrue(why.startswith("TAP_MERGED"))
        k, why = self.snap([a], 400, 400, flags={0: (P.IN_MERGED_REGION,)})
        self.assertIsNone(k); self.assertTrue(why.startswith("TAP_MERGED"))


def tap_user(pick):
    """pick(scene_frame, detections) -> (x, y) | 'NOT_VISIBLE' | None"""
    class U(Oracle):
        def __call__(self, view, res):
            self.calls.append(view)
            a = self.pick(*self.scene)
            if a is None:
                return None
            if a == CK.NOT_VISIBLE:
                return {"decision": CK.NOT_VISIBLE}
            return {"decision": CK.TAP, "point": list(a)}
    return U(pick)


def centre_of(names, dx=0.0, dy=0.0):
    def pick(fr, ds):
        for n in names:
            if n in fr.boxes:
                b = fr.boxes[n]
                return ((b[0] + b[2]) / 2 + dx * (b[2] - b[0]), (b[1] + b[3]) / 2 + dy * (b[3] - b[1]))
        return CK.NOT_VISIBLE
    return pick


class TestTapReseed(unittest.TestCase):
    def test_centre_and_shoulder_taps_recover_a_false_suspension(self):
        for dx, dy, label in ((0.0, 0.0, "centre"), (-0.42, -0.30, "shoulder"), (0.48, 0.0, "edge")):
            with self.subTest(tap=label):
                fr, pp, n1 = persistent_look_change()
                r = run(fr, pp, tap_user(centre_of(("TARGET_ALT", "TARGET"), dx, dy)))
                cp = r.tracker.checkpoints[0]
                self.assertEqual(cp["status"], CK.RESEEDED, cp["answers"][:2])
                acc = [a for a in cp["answers"] if a["outcome"] == "ACCEPTED"][0]
                self.assertEqual(acc["decision"], CK.TAP)
                self.assertIn("snapped", acc)
                self.assertEqual(r.wrong_trusted(("TARGET", "TARGET_ALT")), [])
                self.assertTrue(r.bank_of(("TARGET_ALT",)))
                self.assertTrue(r.episodes_consistent())

    def test_tap_on_nearby_same_kit_player_is_followed_as_the_users_decision(self):
        fr, pp, n1 = persistent_look_change()
        for f in fr:
            f.boxes["MATE"] = SS.box_at(560, 400, 40, H)
        pp = people(("TARGET_ALT", KIT_A, 0.7), ("MATE", KIT_A, 0.8))
        r = run(fr, pp, tap_user(centre_of(("MATE",))))
        cp = r.tracker.checkpoints[0]
        self.assertEqual(cp["status"], CK.RESEEDED)
        k = cp["closed_frame"]
        self.assertTrue(all(r.owners(j) == ("MATE",) for j in range(k, len(fr)) if r.out[j].identity_trusted))
        self.assertTrue(r.bank_of(("MATE",)), "a clean bank of the (wrong) confirmed player only")

    def test_tap_in_a_crowd_overlap_is_refused_until_clear(self):
        n1, n2, n3 = int(1.5 * FPS), int(3.0 * FPS), int(3.0 * FPS)
        fr = [SS.SceneFrame({"TARGET": SS.box_at(400, 400, 40, H)}) for _ in range(n1)]
        fr += [SS.SceneFrame({"TARGET_ALT": SS.box_at(401, 400, 40, H), "MATE": SS.box_at(425, 402, 40, H)},
                             order=("MATE", "TARGET_ALT")) for _ in range(n2)]            # overlapping
        fr += [SS.SceneFrame({"TARGET_ALT": SS.box_at(401, 400, 40, H), "MATE": SS.box_at(600, 402, 40, H)})
               for _ in range(n3)]                                                          # clear
        pp = people(("TARGET_ALT", KIT_A, 0.7), ("MATE", KIT_A, 0.8))
        r = run(fr, pp, tap_user(centre_of(("TARGET_ALT",), dx=0.3)))     # taps on the overlapped side
        ans = [a for c in r.tracker.checkpoints for a in c["answers"]]
        refused = [a for a in ans if a["outcome"] == "REFUSED"]
        self.assertTrue(refused)
        bad = [a for a in refused if not a["reason"].startswith(("TAP_AMBIGUOUS", "TAP_MERGED", "ambiguous",
                                                                  "seed validation", "the selected"))]
        self.assertEqual(bad, [], bad[:3])
        self.assertTrue(any(a["reason"].startswith("TAP_AMBIGUOUS") for a in refused))
        self.assertEqual(r.wrong_trusted(("TARGET", "TARGET_ALT")), [])
        accepted = [a for a in ans if a["outcome"] == "ACCEPTED" and a["decision"] == CK.TAP]
        if accepted:
            self.assertGreaterEqual(accepted[0]["frame"], n1 + n2, "accepted only once the players separated")

    def test_wrong_tap_on_empty_pitch_and_cannot_identify(self):
        fr, pp, n1 = persistent_look_change()
        r = run(fr, pp, tap_user(lambda f, d: (1100.0, 150.0)))
        ans = [a for c in r.tracker.checkpoints for a in c["answers"]]
        self.assertTrue(ans and all(a["outcome"] == "REFUSED" and a["reason"].startswith("TAP_OFF_PLAYER") for a in ans))
        base = run(fr, pp)
        self.assertEqual([strip(a) for a in base.out], [strip(b) for b in r.out])
        r2 = run(fr, pp, tap_user(lambda f, d: CK.NOT_VISIBLE))
        self.assertEqual([strip(a) for a in base.out], [strip(b) for b in r2.out], "cannot identify: automatic continues")
        self.assertTrue(all(c["status"] == CK.DECLINED for c in r2.tracker.checkpoints))

    def test_malformed_tap_is_refused(self):
        fr, pp, n1 = persistent_look_change()
        it = iter([{"decision": CK.TAP, "point": [1]}, {"decision": CK.TAP, "point": ["a", 2]}, {"decision": CK.TAP}])

        class R:
            scene = None
            def __call__(self, view, res):
                return next(it, None)
        r = run(fr, pp, R())
        ans = r.tracker.checkpoints[0]["answers"]
        self.assertEqual(len(ans), 3)
        self.assertTrue(all(a["outcome"] == "REFUSED" for a in ans))


if __name__ == "__main__":
    unittest.main()


class TestSimulatedTapUser(unittest.TestCase):
    def test_protocol_taps_from_truth(self):
        from tools_rb import r2_tap_dev as TD
        v1 = {"keyframes": {"0": {"visibility": "VISIBLE", "human_uncertain": False, "box": [100, 100, 140, 200]},
                            "30": {"visibility": "HIDDEN", "human_uncertain": False, "box": None},
                            "60": {"visibility": "VISIBLE", "human_uncertain": True, "box": [100, 100, 140, 200]}}}
        v2 = {"competitors": {"C1": {"relation": "SAME_TEAM", "keyframes": {"0": {"visibility": "VISIBLE", "box": [300, 100, 340, 200],
                                                                                    "human_uncertain": False}}}}}
        view = {"frame": 0, "candidates": [{"box": [100, 100, 140, 200]}, {"box": [500, 100, 540, 200]}]}
        self.assertEqual(TD.make_user("CENTRE", "c", v1, v2, 0)(view, None), {"decision": "TAP", "point": [120.0, 150.0]})
        self.assertEqual(TD.make_user("SHOULDER", "c", v1, v2, 0)(view, None), {"decision": "TAP", "point": [110.0, 122.0]})
        j = TD.make_user("JITTER", "c", v1, v2, 0)(view, None)
        self.assertEqual(j, TD.make_user("JITTER", "c", v1, v2, 0)(view, None), "deterministic")
        self.assertEqual(TD.make_user("WRONG", "c", v1, v2, 0)(view, None), {"decision": "TAP", "point": [320.0, 150.0]})
        self.assertEqual(TD.make_user("WRONG", "c", v1, {}, 0)(view, None), {"decision": "TAP", "point": [520.0, 150.0]})
        self.assertIsNone(TD.make_user("SILENT", "c", v1, v2, 0)(view, None))
        self.assertIsNone(TD.make_user("CENTRE", "c", v1, v2, 0)({"frame": 7, "candidates": []}, None), "not a keyframe")
        self.assertEqual(TD.make_user("CENTRE", "c", v1, v2, 0)({"frame": 30, "candidates": []}, None), {"decision": "NOT_VISIBLE"})
        self.assertEqual(TD.make_user("CENTRE", "c", v1, v2, 0)({"frame": 60, "candidates": []}, None), {"decision": "NOT_VISIBLE"},
                         "an uncertain human cannot identify")
        wh = TD.make_user("WRONG_HIDDEN", "c", v1, v2, 0)
        self.assertEqual(wh(view, None), {"decision": "TAP", "point": [120.0, 150.0]}, "correct while visible")
        self.assertEqual(wh({"frame": 30, "candidates": [{"box": [500, 100, 540, 200]}]}, None),
                         {"decision": "TAP", "point": [520.0, 150.0]}, "taps the nearest other player while hidden")
        self.assertEqual(TD._reason_class("TAP_AMBIGUOUS: x"), "TAP_AMBIGUOUS")
        self.assertEqual(TD._reason_class("ambiguous selection: y"), "PRECHECK_AMBIGUOUS")

    def test_match_rule_containment_aware(self):
        from tools_rb import r2_tap_dev as TD
        human = [794, 307, 859, 413]
        tight = [809, 318, 845, 397]                 # J02 f150: same body, IoU 0.40
        self.assertFalse(TD.match(tight, human, strict=True))
        self.assertTrue(TD.match(tight, human))
        self.assertFalse(TD.match([700, 307, 760, 413], human), "a neighbour outside the box never matches")
        self.assertFalse(TD.match([800, 380, 850, 413], human), "a small part box (< half height) is not the body")

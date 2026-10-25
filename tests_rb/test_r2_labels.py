# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2 instruction (Nathan, 2026-10-26); tools_rb/r2_labels.py
"""r2_labels_v1 validation and freezing. Synthetic label documents only (no footage, no real labels)."""
import copy
import json
import os
import shutil
import tempfile
import unittest

from tools_rb import r2_labels as L

N, FPS, W, H = 30, 10.0, 640, 360
BOX = [100.0, 100.0, 140.0, 200.0]
OTHER = [150.0, 100.0, 190.0, 200.0]


def frame(vis="VISIBLE", res="YES", occ="NONE", merged=False, contact=(), cut=False, unc=False):
    return {"visibility": vis, "resolvable": res, "occlusion": occ, "merged": merged, "contact_with": list(contact),
            "camera_cut": cut, "human_uncertain": unc, "note": ""}


def base():
    """A valid clip: visible throughout, seed at frame 2, every frame boxed."""
    lab = {"schema": L.SCHEMA, "clip_id": "R2_TEST", "clip_sha256": "a" * 64, "fps": FPS, "frames": N, "width": W,
           "height": H, "annotator": "Nathan", "tool": "test", "saved_utc": "2026-10-26T00:00:00Z",
           "clip_rejected": None, "seed": {"frame": 2, "box": list(BOX)},
           "per_frame": [frame() for _ in range(N)],
           "target_boxes": {str(i): list(BOX) for i in range(N)},
           "people": {"P1": {"description": "opponent #4", "relation": "OPPONENT", "boxes": {}}},
           "first_clean_reappearance": []}
    return lab


CLIP = {"clip_id": "R2_TEST", "clip_sha256": "a" * 64, "frames": N, "fps": FPS, "split": "DEV"}


class Validate(unittest.TestCase):
    def test_base_is_valid(self):
        self.assertEqual(L.validate(base(), CLIP), [])

    def test_box_grid_is_5hz_from_the_seed_plus_event_padding(self):
        lab = base()
        self.assertEqual(L.box_frames(lab), list(range(2, N, 2)))            # 10 fps / 5 Hz = every 2nd frame
        lab["per_frame"][15] = frame(contact=["P1"])
        lab["people"]["P1"]["boxes"] = {str(i): list(OTHER) for i in range(N)}
        req = L.box_frames(lab)
        self.assertTrue(set(range(10, 21)) <= set(req))                      # +-0.5 s = +-5 frames
        self.assertNotIn(9, req)

    def test_missing_target_box_on_a_required_frame(self):
        lab = base()
        del lab["target_boxes"]["4"]
        self.assertIn("frame 4: target box required", L.validate(lab))
        del lab["target_boxes"]["5"]                                          # off the grid: allowed
        self.assertEqual(len(L.validate(lab)), 1)

    def test_uncertain_or_unresolvable_frames_need_no_box(self):
        lab = base()
        del lab["target_boxes"]["4"], lab["target_boxes"]["6"]
        lab["per_frame"][4] = frame(unc=True)
        lab["per_frame"][6] = frame(res="NO")
        self.assertEqual([e for e in L.validate(lab) if "target box required" in e], [])

    def test_hidden_or_out_of_frame_forbids_a_target_box(self):
        lab = base()
        lab["per_frame"][10] = frame("HIDDEN", occ="FULL_BY_PERSON")
        self.assertTrue(any("target box given while target is HIDDEN" in e for e in L.validate(lab)))
        lab = base()
        lab["per_frame"][10] = frame("OUT_OF_FRAME")
        self.assertTrue(any("OUT_OF_FRAME" in e for e in L.validate(lab)))

    def test_contact_person_needs_boxes_on_required_frames(self):
        lab = base()
        lab["per_frame"][15] = frame(contact=["P1"])
        self.assertIn("frame 15: box required for contact person P1", L.validate(lab))
        lab["per_frame"][15] = frame(contact=["NOBODY"])
        self.assertIn("frame 15: contact_with must list known people", L.validate(lab))

    def test_state_consistency(self):
        for f, msg in ((frame("HIDDEN", occ="PARTIAL_BY_PERSON"), "HIDDEN needs a FULL_* occlusion"),
                       (frame("OUT_OF_FRAME", merged=True), "OUT_OF_FRAME cannot"),
                       (frame("BLURRY"), "visibility"), (frame(res="MAYBE"), "resolvable"),
                       (frame(occ="SOME"), "occlusion")):
            with self.subTest(msg=msg):
                lab = base()
                lab["target_boxes"].pop("11", None)
                lab["per_frame"][11] = f
                self.assertTrue(any(msg in e for e in L.validate(lab)), L.validate(lab))

    def test_seed_rules(self):
        lab = base()
        lab["seed"]["box"] = [1.0, 1.0, 2.0, 2.0]
        self.assertIn("seed box must equal the target box on the seed frame", L.validate(lab))
        lab = base()
        lab["per_frame"][2] = frame(merged=True)
        self.assertTrue(any("seed frame must be" in e for e in L.validate(lab)))
        lab = base()
        lab["seed"]["frame"] = N - 1
        self.assertTrue(any("seed must have" in e for e in L.validate(lab)))

    def test_first_clean_reappearance_rules(self):
        lab = base()
        lab["first_clean_reappearance"] = [12]
        self.assertTrue(any("previous frame already clean" in e for e in L.validate(lab)))
        lab = base()
        lab["per_frame"][11] = frame("HIDDEN", occ="FULL_BY_PERSON")
        del lab["target_boxes"]["11"]
        lab["first_clean_reappearance"] = [12]
        self.assertEqual(L.validate(lab), [])

    def test_boxes_must_lie_inside_the_image(self):
        lab = base()
        lab["target_boxes"]["8"] = [600.0, 100.0, 700.0, 200.0]
        self.assertTrue(any("target box at 8 invalid" in e for e in L.validate(lab)))

    def test_clip_record_mismatch_and_wrong_frame_count(self):
        self.assertTrue(L.validate(base(), dict(CLIP, clip_sha256="b" * 64)))
        lab = base()
        lab["per_frame"].pop()
        self.assertTrue(any("per_frame has" in e for e in L.validate(lab)))

    def test_rejected_clip_needs_no_frame_labels(self):
        lab = base()
        lab["clip_rejected"] = "target never identifiable"
        lab["per_frame"] = []
        self.assertEqual(L.validate(lab, CLIP), [])

    def test_malformed_input_reports_instead_of_crashing(self):
        lab = base()
        lab["per_frame"][3] = None
        self.assertTrue(L.validate(lab))


class Freeze(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="r2_labels_")
        self.lab = os.path.join(self.tmp, "in.json")
        self.clip = os.path.join(self.tmp, "clip.json")
        with open(self.lab, "w") as f:
            json.dump(base(), f)
        with open(self.clip, "w") as f:
            json.dump(CLIP, f)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_freeze_writes_copy_and_record_once(self):
        out = os.path.join(self.tmp, "frozen")
        rec = L.freeze(self.lab, self.clip, out)
        self.assertEqual(rec["status"], "FROZEN")
        self.assertTrue(rec["valid"])
        self.assertEqual(rec["labels_file"], "R2_TEST.labels_v1.json")
        with open(os.path.join(out, rec["labels_file"]), "rb") as f:
            import hashlib
            self.assertEqual(hashlib.sha256(f.read()).hexdigest(), rec["sha256"])
        self.assertEqual((rec["split"], rec["seed_frame"], rec["fps"]), ("DEV", 2, FPS))
        with open(os.path.join(out, "R2_TEST.fixture.json")) as f:
            fx = json.load(f)
        self.assertEqual((fx["video_sha256"], fx["seed_frame"], fx["seed_bbox"], fx["fps"]), ("a" * 64, 2, BOX, FPS))
        with self.assertRaises(L.LabelError):
            L.freeze(self.lab, self.clip, out)                                # never replaced

    def test_invalid_labels_are_not_frozen(self):
        bad = base()
        del bad["target_boxes"]["4"]
        with open(self.lab, "w") as f:
            json.dump(bad, f)
        out = os.path.join(self.tmp, "frozen")
        with self.assertRaises(L.LabelError):
            L.freeze(self.lab, self.clip, out)
        self.assertFalse(os.path.exists(out) and os.listdir(out))


if __name__ == "__main__":
    unittest.main()


class R2Adapter(unittest.TestCase):
    """tracker_eval's pre-registered r2_v1 mapping (box grid, uncertain frames, seed offset)."""

    def _doc(self, boxes, trusted=True):
        recs = []
        for b in boxes:
            recs.append({"selected_player": True, "observed_position_box": list(b),
                         "v2": {"identity_trusted": trusted, "rb_state": "TRACKING", "bbox_is_observed": True,
                                "observation_source": "DETECTION", "bbox_is_held": False, "bbox_is_predicted": False}})
        return {"summary": {"fps": FPS}, "frame_log": [{"frame": i + 1, "tracks": [r]} for i, r in enumerate(recs)]}

    def _truth(self, lab):
        from tools_rb import tracker_eval as E
        return E, E.parse_truth(json.dumps(lab).encode(), "r2_v1", lab["seed"]["frame"], FPS)

    def test_rel_frames_start_after_the_seed_and_grid_marks_box_frames(self):
        lab = base()
        lab["target_boxes"] = {str(i): list(BOX) for i in L.box_frames(lab)}
        E, truth = self._truth(lab)
        self.assertEqual([t.rel_frame for t in truth], list(range(1, N - 2)))
        self.assertEqual([t.rel_frame for t in truth if t.box_annotated], [2 * k for k in range(1, 14)])
        r = E.evaluate(self._doc([BOX] * (N - 3)), truth)
        fci = r["priority_1_false_confident_identity"]
        self.assertEqual((fci["count"], fci["status"]), (0, "COMPLETE"))       # off-grid frames are not "unverifiable"
        self.assertEqual(fci["confident_frames_between_box_samples"], 14)

    def test_wrong_box_on_a_grid_frame_is_caught_and_off_grid_is_not_judged(self):
        lab = base()
        lab["target_boxes"] = {str(i): list(BOX) for i in L.box_frames(lab)}
        E, truth = self._truth(lab)
        boxes = [BOX] * (N - 3)
        boxes[1] = OTHER                       # rel 2 = clip frame 4, on the grid
        boxes[2] = OTHER                       # rel 3 = clip frame 5, off the grid
        r = E.evaluate(self._doc(boxes), truth)
        self.assertEqual(r["priority_1_false_confident_identity"]["frames"], [2])

    def test_contact_person_boxes_make_swaps_measurable(self):
        lab = base()
        lab["per_frame"][10] = frame(contact=["P1"])
        lab["people"]["P1"]["boxes"] = {str(i): list(OTHER) for i in L.box_frames(lab)}
        E, truth = self._truth(lab)
        boxes = [BOX] * (N - 3)
        boxes[10 - 2 - 1] = OTHER              # clip frame 10 -> rel 8
        r = E.evaluate(self._doc(boxes), truth)
        self.assertEqual(r["priority_2_identity_swaps"]["count"], 1)
        self.assertEqual(r["b5_contact_integrity"]["status"], "FAIL")

    def test_uncertain_frames_are_excluded_and_reported(self):
        lab = base()
        lab["per_frame"][8] = frame("HIDDEN", occ="FULL_BY_PERSON", unc=True)
        del lab["target_boxes"]["8"]
        E, truth = self._truth(lab)
        r = E.evaluate(self._doc([BOX] * (N - 3)), truth)
        self.assertEqual(r["priority_4_trusted_on_hidden_merged_unresolvable"]["count"], 0)
        self.assertEqual(r["human_uncertain"], {"frames": 1, "confident_on_uncertain": [6]})
        self.assertEqual(r["coverage_and_abstention"]["available_frames"], N - 3 - 1)

    def test_rejected_or_invalid_labels_are_refused(self):
        from tools_rb import tracker_eval as E
        lab = base()
        lab["clip_rejected"] = "no identifiable target"
        with self.assertRaises(E.EvalRefused):
            self._truth(lab)
        bad = base()
        del bad["target_boxes"]["4"]
        with self.assertRaises(E.EvalRefused):
            self._truth(bad)
        with self.assertRaises(E.EvalRefused):
            E.parse_truth(json.dumps(base()).encode(), "r2_v1", 3, FPS)      # seed mismatch

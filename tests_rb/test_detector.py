# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R1-C instruction (Nathan, 2026-10-24, required detector tests); R1/R1_GENERIC_TRACKER_REQUIREMENT.md
#                 (neutral identities: TARGET, PERSON_A, PERSON_B, PERSON_C)
"""Tests for the generic person-detector interface and the YOLOX ONNX backend.

Geometry, filtering, NMS, clipping and tile merging are tested with neutral synthetic boxes (TARGET, PERSON_A..C).
The YOLOX backend tests need models/yolox_s.onnx (git-ignored, pinned by sha256); they are skipped if it is absent.
Real-footage checks (partial people at the image border) run only when SPANA_GOLD_VIDEO points to a clip; they use
it purely as generic footage with people in it.
"""
import ast
import dataclasses
import hashlib
import os
import shutil
import tempfile
import unittest

import numpy as np

from player_tracker_rb import detector as D
from player_tracker_rb.config import TrackerConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "models", "yolox_s.onnx")
HAVE_MODEL = os.path.isfile(MODEL)


def cfg(**over):
    return TrackerConfig.with_overrides(over)


IMG = np.zeros((720, 1280, 3), np.uint8)
TARGET = (400.0, 200.0, 460.0, 330.0)
PERSON_A = (470.0, 205.0, 528.0, 332.0)       # beside the target, no overlap
PERSON_B = (420.0, 210.0, 470.0, 320.0)       # overlapping the target (IoU ~0.4): must survive NMS
PERSON_C = (1250.0, 600.0, 1300.0, 740.0)     # partly outside the image (right and bottom)


class TestGeometry(unittest.TestCase):
    def test_letterbox_full_frame(self):
        lb = D.letterbox_for(1280, 720, 640)
        self.assertAlmostEqual(lb.scale, 0.5)
        self.assertEqual(D.resized_extent(lb), (640, 360))

    def test_letterbox_round_trip(self):
        for (w, h, x0, y0) in ((1280, 720, 0, 0), (704, 400, 576, 320), (333, 777, 10, 20)):
            with self.subTest(region=(w, h, x0, y0)):
                lb = D.letterbox_for(w, h, 640, x0, y0)
                boxes = np.array([[x0 + 1.5, y0 + 2.25, x0 + w - 3.0, y0 + h - 4.5], [x0, y0, x0 + 10, y0 + 20]])
                back = lb.to_image(lb.to_network(boxes))
                np.testing.assert_allclose(back, boxes, atol=1e-9)

    def test_letterbox_reversal_known_values(self):
        lb = D.letterbox_for(704, 400, 640, origin_x=576, origin_y=320)   # a tile at (576, 320)
        scale = min(640 / 400, 640 / 704)
        net = np.array([[0.0, 0.0, 64.0, 128.0]])
        np.testing.assert_allclose(lb.to_image(net), [[576, 320, 576 + 64 / scale, 320 + 128 / scale]])

    def test_clip(self):
        out = D.clip_boxes(np.array([[-5, -5, 50, 50], [1270, 700, 1300, 740]]), 1280, 720)
        np.testing.assert_allclose(out, [[0, 0, 50, 50], [1270, 700, 1280, 720]])

    def test_tile_layout_covers_image(self):
        for cols, rows in ((1, 1), (2, 2), (3, 2), (4, 3)):
            with self.subTest(grid=(cols, rows)):
                tiles = D.tile_layout(1280, 720, cols, rows, 0.2)
                self.assertEqual(len(tiles), cols * rows)
                cover = np.zeros((720, 1280), bool)
                for t in tiles:
                    self.assertTrue(0 <= t.x0 < t.x1 <= 1280 and 0 <= t.y0 < t.y1 <= 720)
                    cover[t.y0:t.y1, t.x0:t.x1] = True
                self.assertTrue(cover.all())
                self.assertEqual(len({(t.x1 - t.x0, t.y1 - t.y0) for t in tiles}), 1)   # equal-size tiles


class TestScriptedPipeline(unittest.TestCase):
    def run_script(self, rows, **over):
        return D.ScriptedDetector([rows], cfg(**over)).detect(IMG)

    def test_empty(self):
        self.assertEqual(self.run_script([]), ())

    def test_multiple_people_generic_output(self):
        out = self.run_script([TARGET + (0.9,), PERSON_A + (0.8,), PERSON_B + (0.7,)])
        self.assertEqual(len(out), 3)
        self.assertEqual([c.score for c in out], [0.9, 0.8, 0.7])          # highest first
        for c in out:
            self.assertIsInstance(c, D.PersonCandidate)
            self.assertEqual(c.kind, "PERSON_CANDIDATE")
            for f in dataclasses.fields(c):
                v = getattr(c, f.name)
                self.assertIn(type(v), (tuple, float, bool, str), f.name)
            self.assertTrue(all(type(v) is float for v in c.bbox))
            self.assertEqual(set(c.as_record()), {"kind", "bbox", "score", "touches_image_edge", "detector_pass"})

    def test_no_identity_or_role_fields(self):
        names = {f.name for f in dataclasses.fields(D.PersonCandidate)}
        for banned in ("team", "role", "colour", "color", "number", "jersey", "identity", "track_id", "name"):
            self.assertFalse(any(banned in n for n in names), banned)

    def test_confidence_filter(self):
        out = self.run_script([TARGET + (0.09,), PERSON_A + (0.10,), PERSON_B + (0.5,)])
        self.assertEqual(sorted(c.score for c in out), [0.10, 0.5])         # >= detector_min_score (0.10)
        out = self.run_script([TARGET + (0.3,)], detector_min_score=0.4)
        self.assertEqual(out, ())

    def test_nms_suppresses_duplicates_keeps_overlapping_people(self):
        dup = (402.0, 201.0, 461.0, 331.0)                                  # same person, IoU ~0.95
        out = self.run_script([TARGET + (0.9,), dup + (0.8,), PERSON_B + (0.7,)])
        boxes = [c.bbox for c in out]
        self.assertIn(TARGET, boxes)
        self.assertNotIn(dup, boxes)
        self.assertIn(PERSON_B, boxes)                                      # a different, overlapping person

    def test_nms_threshold_from_config(self):
        out = self.run_script([TARGET + (0.9,), PERSON_B + (0.7,)], detector_nms_iou=0.3)
        self.assertEqual(len(out), 1)

    def test_clipping_and_edge_flag(self):
        out = self.run_script([PERSON_C + (0.8,), TARGET + (0.9,)])
        c = {x.bbox: x for x in out}
        self.assertIn((1250.0, 600.0, 1280.0, 720.0), c)
        self.assertTrue(c[(1250.0, 600.0, 1280.0, 720.0)].touches_image_edge)
        self.assertFalse(c[TARGET].touches_image_edge)

    def test_fully_outside_box_dropped(self):
        self.assertEqual(self.run_script([(1300.0, 10.0, 1400.0, 100.0, 0.9)]), ())

    def test_deterministic_tie_order(self):
        rows = [PERSON_A + (0.8,), TARGET + (0.8,)]
        self.assertEqual(self.run_script(rows), self.run_script(list(rows)))


class TestTileMerging(unittest.TestCase):
    """Tile passes: a cut piece of a person must not beat or duplicate the whole person, but a partly hidden
    second person inside someone's box must survive."""

    def merge(self, rows, sides, **over):
        arr = np.asarray(rows, dtype=float)
        keep = D.merge_passes(arr[:, :4], arr[:, 4], ["p"] * len(rows), np.asarray(sides), cfg(**over))
        return [tuple(arr[k, :4]) for k in keep]

    def test_cut_fragment_of_whole_box_dropped_even_with_higher_score(self):
        whole = TARGET + (0.75,)
        frag = (401.0, 260.0, 459.0, 330.0, 0.90)          # lower part of TARGET, cut at its top tile edge
        kept = self.merge([whole, frag], [0, D.CUT_TOP])
        self.assertEqual(kept, [TARGET])

    def test_partly_hidden_second_person_kept(self):
        whole = TARGET + (0.75,)
        hidden = (425.0, 200.0, 450.0, 270.0, 0.88)          # PERSON_B mostly hidden behind TARGET, cut at bottom
        kept = self.merge([whole, hidden], [0, D.CUT_BOTTOM])
        self.assertEqual(len(kept), 2)

    def test_two_cut_pieces_of_one_person(self):
        top = (400.0, 200.0, 460.0, 300.0, 0.8)
        bottom_piece = (401.0, 290.0, 459.0, 299.0, 0.7)    # sliver fully inside the other, spanning its width
        kept = self.merge([top, bottom_piece], [D.CUT_BOTTOM, D.CUT_TOP])
        self.assertEqual(len(kept), 1)

    def test_cut_box_seen_uncut_by_neighbour_tile(self):
        # 2x2 tiles on 1280x720 with 0.2 overlap: rows span y 0-400 and 320-720. A box cut at the top edge of a
        # lower tile (y=320) that ends above y=400 lies wholly inside the upper tile, which saw it without the cut.
        tiles = D.tile_layout(1280, 720, 2, 2, 0.2)
        lower_left = [t for t in tiles if t.name == "tile_c0_r1"][0]
        sliver = (100.0, 320.0, 130.0, 345.0)                         # e.g. the legs of PERSON_A
        self.assertTrue(D.seen_uncut_elsewhere(sliver, D.CUT_TOP, lower_left, tiles))
        tall = (100.0, 320.0, 130.0, 460.0)                           # extends past the upper tile: keep
        self.assertFalse(D.seen_uncut_elsewhere(tall, D.CUT_TOP, lower_left, tiles))
        uncut_side = (100.0, 320.0, 130.0, 345.0)                     # the upper tile does not extend left of x=0
        self.assertFalse(D.seen_uncut_elsewhere(uncut_side, D.CUT_LEFT, lower_left, tiles))

    def test_no_tiling_is_plain_nms(self):
        rows = [TARGET + (0.9,), PERSON_A + (0.8,), PERSON_B + (0.7,)]
        arr = np.asarray(rows)
        self.assertEqual(D.merge_passes(arr[:, :4], arr[:, 4], ["f"] * 3, np.zeros(3, int), cfg()),
                         D.nms(arr[:, :4], arr[:, 4], cfg().get("detector_nms_iou")))


class TestNoBackendLeakage(unittest.TestCase):
    def test_only_backend_module_imports_onnxruntime(self):
        pkg = os.path.join(ROOT, "player_tracker_rb")
        for f in sorted(os.listdir(pkg)):
            if not f.endswith(".py"):
                continue
            tree = ast.parse(open(os.path.join(pkg, f), encoding="utf-8").read())
            mods = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
            mods |= {(n.module or "").split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
            with self.subTest(f):
                if f != "detector_yolox.py":
                    self.assertNotIn("onnxruntime", mods)

    def test_generic_interface_names_no_backend(self):
        src = open(os.path.join(ROOT, "player_tracker_rb", "detector.py"), encoding="utf-8").read().lower()
        self.assertNotIn("yolox", src)


@unittest.skipUnless(HAVE_MODEL, "models/yolox_s.onnx not present")
class TestYoloxBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from player_tracker_rb import detector_yolox as Y
        cls.Y = Y
        cls.tmp = tempfile.mkdtemp(prefix="rb_det_")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_model_hash_pinned(self):
        h = hashlib.sha256(open(MODEL, "rb").read()).hexdigest()
        self.assertEqual(h, cfg().get("detector_model_sha256"))

    def test_hash_mismatch_refused(self):
        with self.assertRaises(D.DetectorModelError):
            self.Y.YoloxOnnxDetector(cfg(detector_model_sha256="0" * 64))

    def test_missing_model_refused(self):
        with self.assertRaises(D.DetectorModelError):
            self.Y.YoloxOnnxDetector(cfg(detector_model_file=os.path.join(self.tmp, "absent.onnx")))

    def test_invalid_model_refused(self):
        bad = os.path.join(self.tmp, "bad.onnx")
        with open(bad, "wb") as f:
            f.write(b"not an onnx model" * 100)
        h = hashlib.sha256(open(bad, "rb").read()).hexdigest()
        with self.assertRaises(D.DetectorModelError):
            self.Y.YoloxOnnxDetector(cfg(detector_model_file=bad, detector_model_sha256=h))

    def test_input_size_mismatch_refused(self):
        with self.assertRaises(D.DetectorModelError):
            self.Y.YoloxOnnxDetector(cfg(detector_input_size=960))

    def test_preprocessing_geometry(self):
        det = self.Y.YoloxOnnxDetector(cfg())
        img = np.full((720, 1280, 3), 200, np.uint8)
        tensor, lb = det._preprocess(img, 0, 0)
        self.assertEqual(tensor.shape, (1, 3, 640, 640))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(np.all(tensor[0, :, :360, :] == 200))          # resized image at the top-left
        self.assertTrue(np.all(tensor[0, :, 360:, :] == 114))          # pad value below, no normalisation
        self.assertAlmostEqual(lb.scale, 0.5)

    def test_blank_image_gives_no_people(self):
        det = self.Y.YoloxOnnxDetector(cfg())
        self.assertEqual(det.detect(np.full((720, 1280, 3), 90, np.uint8)), ())

    def test_bad_image_refused(self):
        det = self.Y.YoloxOnnxDetector(cfg())
        for bad in (np.zeros((10, 10), np.uint8), np.zeros((10, 10, 3), np.float32), np.zeros((10, 10, 4), np.uint8)):
            with self.assertRaises(D.DetectorError):
                det.detect(bad)

    def test_provenance_and_model_record(self):
        det = self.Y.YoloxOnnxDetector(cfg())
        p = det.provenance()
        self.assertEqual(p["model_sha256"], cfg().get("detector_model_sha256"))
        self.assertEqual(p["model_status"], "LICENCE_UNCLEAR")
        self.assertEqual(p["input_size"], 640)
        m = det.model_record()
        self.assertEqual(set(m), {"role", "filename", "sha256", "status"})


GOLD_VIDEO = os.environ.get("SPANA_GOLD_VIDEO", "")


@unittest.skipUnless(HAVE_MODEL and os.path.isfile(GOLD_VIDEO), "needs models/yolox_s.onnx and SPANA_GOLD_VIDEO")
class TestYoloxOnRealFootage(unittest.TestCase):
    """Real footage used only as generic people-in-a-frame input. No identities, no events."""

    @classmethod
    def setUpClass(cls):
        import cv2
        from player_tracker_rb import detector_yolox as Y
        cap = cv2.VideoCapture(GOLD_VIDEO)
        frames = []
        for _ in range(3):
            ok, img = cap.read()
            frames.append(img)
        cap.release()
        cls.frames = frames
        cls.Y = Y

    def test_people_found_and_output_is_generic(self):
        det = self.Y.YoloxOnnxDetector(cfg())
        out = det.detect(self.frames[0])
        self.assertGreater(len(out), 0)
        for c in out:
            self.assertEqual(type(c), D.PersonCandidate)
            self.assertTrue(all(type(v) is float for v in c.bbox) and type(c.score) is float)

    def test_deterministic_repeated_inference(self):
        for mode in ("none", "grid"):
            with self.subTest(tiling=mode):
                a = self.Y.YoloxOnnxDetector(cfg(detector_tiling=mode)).detect(self.frames[1])
                b = self.Y.YoloxOnnxDetector(cfg(detector_tiling=mode)).detect(self.frames[1])
                self.assertEqual(a, b)

    def test_partial_people_at_image_border(self):
        # Crop so that the people found in the full frame are cut by the crop border: the cut ones must be found
        # (as partial people) and flagged touches_image_edge.
        det = self.Y.YoloxOnnxDetector(cfg())
        img = self.frames[2]
        full = [c for c in det.detect(img) if c.score >= 0.5 and c.height >= 60]
        self.assertTrue(full)
        c0 = full[0]
        cut_x = int((c0.bbox[0] + c0.bbox[2]) / 2)                        # vertical cut through that person
        crop = np.ascontiguousarray(img[:, cut_x:])
        part = det.detect(crop)
        near = [p for p in part if p.bbox[0] <= cfg().get("detector_edge_touch_margin_px")
                and p.bbox[1] < c0.bbox[3] and p.bbox[3] > c0.bbox[1]]
        self.assertTrue(near, "the half-visible person should still be detected")
        self.assertTrue(all(p.touches_image_edge for p in near))


if __name__ == "__main__":
    unittest.main()

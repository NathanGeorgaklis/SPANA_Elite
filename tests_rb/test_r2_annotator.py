# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2 instruction (Nathan, 2026-10-26); tools_rb/r2_annotator.html, tools_rb/r2_labels.py
"""Browser test of the R2 annotation page on a synthetic pack: draw target boxes, set the seed, add a contact person
with boxes, download, and validate the file with r2_labels.validate. Skipped when Playwright is not installed."""
import json
import os
import shutil
import tempfile
import unittest

import cv2
import numpy as np

from tools_rb import r2_labels as L

try:
    from playwright.sync_api import sync_playwright
except ImportError:                                     # pragma: no cover
    sync_playwright = None

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N, FPS, W, H = 20, 10.0, 640, 360


@unittest.skipIf(sync_playwright is None, "playwright not installed")
class AnnotatorPage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="r2_annot_")
        fr = os.path.join(self.tmp, "frames", "R2_FAKE")
        os.makedirs(fr)
        for i in range(N):
            im = np.full((H, W, 3), (40, 120, 40), np.uint8)
            cv2.rectangle(im, (100 + i * 5, 100), (140 + i * 5, 200), (200, 60, 40), -1)
            cv2.imwrite(os.path.join(fr, f"f_{i:06d}.jpg"), im)
        shutil.copyfile(os.path.join(HERE, "tools_rb", "r2_annotator.html"), os.path.join(self.tmp, "index.html"))
        m = [{"clip_id": "R2_FAKE", "split": "DEV", "source": "SYO_MAR", "fps": FPS, "frames": N, "width": W,
              "height": H, "clip_sha256": "a" * 64, "proposed_target": "red box", "proposed_seed_s": 0.2,
              "frames_ready": True}]
        with open(os.path.join(self.tmp, "r2_clips_manifest.js"), "w") as f:
            f.write("window.R2_CLIPS = " + json.dumps(m) + ";\n")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_label_a_clip_and_validate_the_download(self):
        errs = []
        with sync_playwright() as pw:
            b = pw.chromium.launch()
            ctx = b.new_context(accept_downloads=True, viewport={"width": 1500, "height": 900})
            p = ctx.new_page()
            p.on("pageerror", lambda e: errs.append(str(e)))
            p.goto("file://" + os.path.join(self.tmp, "index.html"))
            p.wait_for_timeout(300)
            p.fill("#annotator", "Tester")
            bb = p.query_selector("#cv").bounding_box()
            sx, sy = bb["width"] / W, bb["height"] / H

            def drag(x1, y1, x2, y2):
                p.mouse.move(bb["x"] + x1 * sx, bb["y"] + y1 * sy)
                p.mouse.down()
                p.mouse.move(bb["x"] + x2 * sx, bb["y"] + y2 * sy, steps=3)
                p.mouse.up()

            drag(110, 100, 150, 200)                      # frame 2 (proposed seed)
            p.click("#bSeed")
            p.fill("#pDesc", "opponent #4")
            p.click("#bAddP")
            p.click('#layerSeg button:has-text("TARGET")')
            for i in range(3, N):
                p.keyboard.press("ArrowRight")
                drag(100 + i * 5, 100, 140 + i * 5, 200)
            for _ in range(N - 1 - 10):
                p.keyboard.press("ArrowLeft")            # frame 10
            p.click("#contactBox input")
            p.click('#layerSeg button:has-text("P1")')
            for _ in range(5):
                p.keyboard.press("ArrowLeft")            # frame 5
            for _ in range(5, 16):
                drag(400, 100, 440, 200)
                p.keyboard.press("ArrowRight")
            p.click("#bCheck")
            self.assertIn("No problems", p.text_content("#msgs"))
            with p.expect_download() as d:
                p.click("#btnSave")
            out = os.path.join(self.tmp, "out.json")
            d.value.save_as(out)
            b.close()
        self.assertEqual(errs, [])
        with open(out) as f:
            lab = json.load(f)
        self.assertEqual(lab["seed"]["frame"], 2)
        self.assertEqual(lab["per_frame"][10]["contact_with"], ["P1"])
        self.assertEqual(L.validate(lab, {"clip_id": "R2_FAKE", "clip_sha256": "a" * 64, "frames": N, "fps": FPS}), [])


if __name__ == "__main__":
    unittest.main()

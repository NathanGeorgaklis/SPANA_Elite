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


class _Pack:
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


@unittest.skipIf(sync_playwright is None, "playwright not installed")
class AnnotatorPage(_Pack, unittest.TestCase):
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

            # label every frame VISIBLE / YES / NONE with the range tool (frames start UNLABELLED)
            p.click("#bRangeStart")                      # range start at frame 2
            for _ in range(N - 1 - 2):
                p.keyboard.press("ArrowRight")           # frame 19: set its state, apply to 2..19
            p.click('#visSeg button:has-text("VISIBLE")')
            p.click('#resSeg button:has-text("YES")')
            p.select_option("#occSel", "NONE")
            p.click("#bRangeApply")
            for _ in range(N - 1 - 2):
                p.keyboard.press("ArrowLeft")
            # frames 0-1 (before the seed) stay UNLABELLED: allowed
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
            for _ in range(5, 16):                        # partner boxed across the +-0.5 s contact window
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
        self.assertEqual(lab["per_frame"][0]["visibility"], "UNLABELLED")
        self.assertEqual(lab["per_frame"][10]["contact_with"], ["P1"])
        self.assertEqual(L.validate(lab, {"clip_id": "R2_FAKE", "clip_sha256": "a" * 64, "frames": N, "fps": FPS}), [])


def _random_label(rng, n, fps):
    """A random label document (valid or not) exercising every rule the page re-implements."""
    vis = ["VISIBLE", "PARTIAL", "HIDDEN", "OUT_OF_FRAME", "UNLABELLED"]
    occ = ["NONE", "PARTIAL_BY_PERSON", "FULL_BY_PERSON", "PARTIAL_BY_OTHER", "FULL_BY_OTHER", "UNLABELLED"]
    seed = int(rng.integers(0, n - 1))
    pf, cur = [], None
    for i in range(n):
        if cur is None or rng.random() < 0.15:
            v = vis[int(rng.integers(0, 4 if rng.random() < 0.95 else 5))]
            o = ("NONE" if v in ("VISIBLE", "PARTIAL") and rng.random() < 0.7 else occ[int(rng.integers(0, 5))])
            if v == "OUT_OF_FRAME" and rng.random() < 0.8:
                o = "NONE"
            cur = {"visibility": v, "resolvable": "YES" if rng.random() < 0.8 else "NO", "occlusion": o,
                   "merged": bool(rng.random() < 0.1), "contact_with": ["P1"] if rng.random() < 0.2 else [],
                   "camera_cut": False, "human_uncertain": bool(rng.random() < 0.05), "note": ""}
        f = dict(cur, contact_with=list(cur["contact_with"]), camera_cut=bool(rng.random() < 0.02))
        pf.append(f)
    pf[seed] = {"visibility": "VISIBLE", "resolvable": "YES", "occlusion": "NONE", "merged": False,
                "contact_with": [], "camera_cut": False, "human_uncertain": False, "note": ""}
    tb = {str(i): [100.0, 100.0, 140.0, 200.0] for i in range(n)
          if pf[i]["visibility"] in ("VISIBLE", "PARTIAL") and rng.random() < 0.97}
    tb[str(seed)] = [100.0, 100.0, 140.0, 200.0]
    absent = [i for i in range(n) if rng.random() < 0.05]
    people = {"P1": {"description": "x", "relation": "OPPONENT", "absent_frames": absent,
                     "boxes": {str(i): [300.0, 100.0, 340.0, 200.0] for i in range(n)
                               if i not in absent and rng.random() < 0.95}}}
    clean = lambda x: (x["visibility"] in ("VISIBLE", "PARTIAL") and x["resolvable"] == "YES"  # noqa: E731
                       and not x["merged"] and not x["human_uncertain"])
    fcr = [i for i in range(seed + 1, n) if clean(pf[i]) and not clean(pf[i - 1]) and rng.random() < 0.9]
    return {"schema": L.SCHEMA, "clip_id": "R2_FAKE", "clip_sha256": "a" * 64, "fps": fps, "frames": n,
            "width": W, "height": H, "annotator": "T", "tool": "t", "saved_utc": "", "clip_rejected": None,
            "seed": {"frame": seed, "box": [100.0, 100.0, 140.0, 200.0]}, "per_frame": pf, "target_boxes": tb,
            "people": people, "first_clean_reappearance": fcr, "box_copied": {}}


@unittest.skipIf(sync_playwright is None, "playwright not installed")
class RangeApply(_Pack, unittest.TestCase):
    def test_range_apply_keeps_existing_camera_cuts_and_notes(self):
        """Review finding M6: applying a state over a range must not erase a cut already marked inside it."""
        with sync_playwright() as pw:
            b = pw.chromium.launch()
            p = b.new_page()
            p.goto("file://" + os.path.join(self.tmp, "index.html"))
            p.wait_for_timeout(200)                       # opens at frame 2
            for _ in range(8):
                p.keyboard.press("ArrowRight")            # frame 10: mark a cut and a note
            p.check("#cut")
            p.fill("#note", "cut here")
            p.press("#note", "Enter")
            p.dispatch_event("#note", "change")
            p.evaluate("document.activeElement.blur()")
            for _ in range(8):
                p.keyboard.press("ArrowLeft")
            p.click("#bRangeStart")
            for _ in range(15):
                p.keyboard.press("ArrowRight")            # frame 17
            p.click('#visSeg button:has-text("VISIBLE")')
            p.click('#resSeg button:has-text("YES")')
            p.select_option("#occSel", "NONE")
            p.click("#bRangeApply")
            pf = p.evaluate("lab.per_frame")
            b.close()
        self.assertTrue(pf[10]["camera_cut"])
        self.assertEqual(pf[10]["note"], "cut here")
        self.assertEqual([x["visibility"] for x in pf[2:18]], ["VISIBLE"] * 16)
        self.assertEqual(sum(x["camera_cut"] for x in pf), 1)


@unittest.skipIf(sync_playwright is None, "playwright not installed")
class UiBehaviours(_Pack, unittest.TestCase):
    """Test-audit gaps: redrawing the seed-frame box updates the seed; HIDDEN removes the target box; boxes are
    clamped to the image; a missing contact-partner box is reported by the page."""

    def test_ui_behaviours(self):
        with sync_playwright() as pw:
            b = pw.chromium.launch()
            p = b.new_page(viewport={"width": 1500, "height": 900})
            p.goto("file://" + os.path.join(self.tmp, "index.html"))
            p.wait_for_timeout(200)                                     # frame 2
            bb = p.query_selector("#cv").bounding_box()
            sx, sy = bb["width"] / W, bb["height"] / H

            def drag(x1, y1, x2, y2):
                p.mouse.move(bb["x"] + x1 * sx, bb["y"] + y1 * sy)
                p.mouse.down()
                p.mouse.move(bb["x"] + x2 * sx, bb["y"] + y2 * sy, steps=3)
                p.mouse.up()

            drag(110, 100, 150, 200)
            p.click("#bSeed")
            drag(120, 110, 160, 210)                                    # redraw on the seed frame
            seed, tb = p.evaluate("[lab.seed.box, lab.target_boxes[2]]")
            self.assertEqual(seed, tb)
            self.assertNotEqual(seed[0], 110.0)
            p.keyboard.press("ArrowRight")                              # frame 3: box past the right edge
            drag(600, 100, 700, 200)
            self.assertEqual(p.evaluate("lab.target_boxes[3][2]"), float(W))
            p.click('#visSeg button:has-text("HIDDEN")')
            self.assertIsNone(p.evaluate("lab.target_boxes[3] || null"))
            occ = p.evaluate("lab.per_frame[3].occlusion")
            self.assertEqual(occ, "UNLABELLED")                         # never auto-filled
            p.fill("#annotator", "Tester")
            p.evaluate("""() => { lab.people.P1 = {description: 'x', relation: 'OPPONENT', boxes: {}};
                          for (let i = 0; i < lab.frames; i++) { lab.per_frame[i] = {visibility: 'VISIBLE',
                          resolvable: 'YES', occlusion: 'NONE', merged: false, contact_with: [], camera_cut: false,
                          human_uncertain: false, note: ''}; lab.target_boxes[i] = lab.seed.box; }
                          lab.per_frame[10].contact_with = ['P1']; }""")
            probs = p.evaluate("pageProblems()")
            b.close()
        self.assertTrue(any("missing box: P1" in x for x in probs), probs[:3])


@unittest.skipIf(sync_playwright is None, "playwright not installed")
class JsPythonParity(_Pack, unittest.TestCase):
    """The page's box grid, partner windows and completeness check agree with r2_labels on random documents
    (review finding L9), at 25, 30 and 60 fps (25 fps exercises half-to-even rounding: 0.5 * 25 = 12.5)."""

    def test_page_rules_match_python_on_random_labels(self):
        rng = np.random.default_rng(20261026)
        mismatches = []
        with sync_playwright() as pw:
            b = pw.chromium.launch()
            p = b.new_page()
            p.goto("file://" + os.path.join(self.tmp, "index.html"))
            p.wait_for_timeout(200)
            for k in range(120):
                fps = [25.0, 30.0, 60.0][k % 3]
                n = int(rng.integers(20, 90))
                lab = _random_label(rng, n, fps)
                js = p.evaluate("""(d) => { lab = d; const pfm = partnerFrames(); const out = {};
                    for (const [k, v] of Object.entries(pfm)) out[k] = [...v].sort((a, b) => a - b);
                    return {grid: boxFrames(), partners: out, problems: pageProblems()}; }""", lab)
                py_grid = L.box_frames(lab)
                py_partners = {k: sorted(v) for k, v in L.partner_frames(lab).items()}
                py_valid = L.validate(lab) == []
                if js["grid"] != py_grid or js["partners"] != py_partners or (js["problems"] == []) != py_valid:
                    mismatches.append((k, fps, js["grid"] == py_grid, js["partners"] == py_partners,
                                       js["problems"][:2], L.validate(lab)[:2]))
            b.close()
        self.assertEqual(mismatches, [])


if __name__ == "__main__":
    unittest.main()

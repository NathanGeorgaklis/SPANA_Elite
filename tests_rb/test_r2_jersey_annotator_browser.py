# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2 DEV jersey-number validation dataset instruction (Nathan, 2026-11-04 08:24);
#                 tools_rb/r2_jersey_annotator.html must write files that tools_rb/r2_jersey_labels.py accepts
"""End-to-end: drive the jersey annotation page in headless Chromium on a synthetic pack, download the labels and
validate them with the Python schema. Skipped when node / playwright / ffmpeg are unavailable."""
import json
import os
import shutil
import subprocess
import tempfile
import unittest

from tools_rb import r2_jersey_labels as L

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _have():
    if not (shutil.which("node") and shutil.which("ffmpeg")):
        return False
    return subprocess.run(["node", "-e", "require('playwright')"], capture_output=True).returncode == 0


@unittest.skipUnless(_have(), "node + playwright + ffmpeg required")
class AnnotatorBrowser(unittest.TestCase):
    def test_page_output_validates(self):
        t = tempfile.mkdtemp(prefix="r2_jersey_page_")
        try:
            grid = L.keyframe_grid(60.0, 480)
            os.makedirs(os.path.join(t, "frames", "CLIPX"))
            shutil.copyfile(os.path.join(ROOT, "tools_rb", "r2_jersey_annotator.html"), os.path.join(t, "index.html"))
            man = [{"clip_id": "CLIPX", "clip_sha256": "a" * 64, "fps": 60.0, "frames": 480, "source": "SYN",
                    "split": "DEV", "proposed_target": "synthetic", "classes": [], "width": 1280, "height": 720,
                    "keyframes": grid}]
            with open(os.path.join(t, "jersey_clips_manifest.js"), "w") as f:
                f.write("window.JERSEY_CLIPS=" + json.dumps(man) + ";")
            png = os.path.join(t, "frames", "CLIPX", "f_000000.png")
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=green:s=1280x720", "-frames:v", "1",
                            png], check=True)
            for g in grid[1:]:
                shutil.copyfile(png, os.path.join(t, "frames", "CLIPX", f"f_{g:06d}.png"))
            out = os.path.join(t, "out.json")
            r = subprocess.run(["node", os.path.join(ROOT, "tests_rb", "jersey_annotator_smoke.js"), t, out],
                               capture_output=True, text=True, timeout=120, cwd=ROOT)
            self.assertEqual(r.returncode, 0, r.stderr)
            res = json.loads(r.stdout.strip().splitlines()[-1])
            self.assertEqual(res["errs"], [])
            self.assertIn("No problems found", res["check"])
            lab = json.load(open(out))
            self.assertEqual(L.validate(lab), [])
            s = L.summary(lab)
            self.assertEqual((s["expected_number"], s["readable"], s["unreadable"], s["not_visible"]), (7, 14, 1, 1))
            self.assertEqual(lab["target"]["number_visibility"], "READABLE_IN_SOME_FRAMES")
            self.assertEqual(s["same_team_competitor_numbers"], [8])
        finally:
            shutil.rmtree(t)


if __name__ == "__main__":
    unittest.main()

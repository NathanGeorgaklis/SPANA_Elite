# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: Nathan, 2026-11-04 18:44; tools_rb/r2_jersey_region_annotator.html must write layers that
#                 tools_rb/r2_jersey_regions.py accepts, and never alter the frozen v1
"""End-to-end: build a synthetic frozen v1 + pack, drive the region page in headless Chromium, validate the download.
Skipped when node / playwright / ffmpeg are unavailable."""
import json
import os
import shutil
import subprocess
import tempfile
import unittest

from tools_rb import r2_jersey_labels as L1
from tools_rb import r2_jersey_region_pack as RP
from tools_rb import r2_jersey_regions as R
from tests_rb.test_r2_jersey_labels import GRID, base, reg
from tests_rb.test_r2_jersey_annotator_browser import _have

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CID = "R2_JERSEY_MCFC_MIT_J04"


@unittest.skipUnless(_have(), "node + playwright + ffmpeg required")
class RegionBrowser(unittest.TestCase):
    def test_page_output_validates(self):
        t = tempfile.mkdtemp(prefix="r2_jreg_page_")
        try:
            rp = os.path.join(t, "reg.json")
            with open(rp, "w") as f:
                json.dump(reg([CID]), f)
            src = os.path.join(t, "v1.json")
            with open(src, "w") as f:
                json.dump(base(CID), f)
            v1dir, pack = os.path.join(t, "v1"), os.path.join(t, "pack")
            L1.freeze(src, rp, v1dir)
            v1_sha = L1._sha(L1.frozen_path(v1dir, CID, 1))
            os.makedirs(os.path.join(pack, "frames", CID))
            png = os.path.join(pack, "frames", CID, f"f_{GRID[0]:06d}.png")
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=green:s=1280x720", "-frames:v", "1",
                            png], check=True)
            for g in GRID[1:]:
                shutil.copyfile(png, os.path.join(pack, "frames", CID, f"f_{g:06d}.png"))
            RP.build(rp, v1dir, pack)
            with self.assertRaises(SystemExit):
                RP.build(rp, v1dir, pack)                     # never overwrites
            out = os.path.join(t, "out.json")
            r = subprocess.run(["node", os.path.join(ROOT, "tests_rb", "jersey_region_smoke.js"), pack, out],
                               capture_output=True, text=True, timeout=180, cwd=ROOT)
            self.assertEqual(r.returncode, 0, r.stderr)
            res = json.loads(r.stdout.strip().splitlines()[-1])
            self.assertEqual(res["errs"], [])
            self.assertIn("No problems found", res["check"])
            lab = json.load(open(out))
            v1 = json.load(open(L1.frozen_path(v1dir, CID, 1)))
            self.assertEqual(R.validate(lab, v1, v1_sha, L1._load_registry(rp)[CID]), [])
            s = R.summary(lab, v1)
            self.assertEqual(s["target_shorts_number"], 7)
            self.assertEqual(s["n_competitors"], 1)
            self.assertEqual(s["distractors"], 1)
            self.assertEqual(s["target"]["shorts_only"], 1)        # the v1-UNREADABLE keyframe has readable shorts
            self.assertEqual(L1._sha(L1.frozen_path(v1dir, CID, 1)), v1_sha)
        finally:
            for root, _, files in os.walk(t):
                for n in files:
                    os.chmod(os.path.join(root, n), 0o644)
            shutil.rmtree(t)


if __name__ == "__main__":
    unittest.main()

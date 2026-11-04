# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: Nathan, 2026-11-04 18:44 (v2 region layer over frozen v1)
"""Add the v2 region-annotation page to an existing jersey annotation pack (built by r2_jersey_pack.py).

    python3 tools_rb/r2_jersey_region_pack.py REGISTRY_JSON V1_FROZEN_DIR PACK_DIR

Writes PACK_DIR/regions.html (copy of tools_rb/r2_jersey_region_annotator.html) and
PACK_DIR/jersey_regions_manifest.js, which embeds each clip's FROZEN v1 label (verified against its freeze record).
Fails closed when a keyframe image is missing; refuses to overwrite.
"""
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools_rb import r2_jersey_labels as L1  # noqa: E402
from tools_rb import r2_jersey_regions as R  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def build(registry_json, v1_dir, pack_dir):
    out_page, out_man = os.path.join(pack_dir, "regions.html"), os.path.join(pack_dir, "jersey_regions_manifest.js")
    if os.path.exists(out_page) or os.path.exists(out_man):
        raise SystemExit(f"{pack_dir} already has the region page; refusing to overwrite")
    reg = L1._load_registry(registry_json)
    man = []
    for cid, c in reg.items():
        v1, sha, _ = R._v1(v1_dir, reg, cid)
        grid = L1.keyframe_grid(c["fps"], c["frames"])
        for g in grid:
            if not os.path.exists(os.path.join(pack_dir, "frames", cid, f"f_{g:06d}.png")):
                raise SystemExit(f"{cid}: keyframe image {g} missing from the pack (fail closed)")
        man.append({"clip_id": cid, "clip_sha256": c["clip_sha256"], "fps": c["fps"], "frames": c["frames"],
                    "width": v1["width"], "height": v1["height"], "source": c["source"], "keyframes": grid,
                    "v1_file": os.path.basename(L1.frozen_path(v1_dir, cid, 1)), "v1_sha256": sha, "v1": v1})
    with open(out_man, "x") as f:
        f.write("window.JERSEY_REGION_CLIPS = " + json.dumps(man) + ";\n")
    shutil.copyfile(os.path.join(HERE, "r2_jersey_region_annotator.html"), out_page)
    print("REGION PAGE", pack_dir, len(man), "clips")


if __name__ == "__main__":
    build(*sys.argv[1:4])

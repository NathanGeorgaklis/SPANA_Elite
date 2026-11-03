# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2 DEV jersey-number validation dataset instruction (Nathan, 2026-11-04 08:24)
"""Build the offline jersey annotation pack. Evaluation tooling only: shows only the clips' own keyframes.

    python3 tools_rb/r2_jersey_pack.py REGISTRY_JSON CLIPS_DIR PACK_DIR

Writes PACK_DIR/index.html (copy of tools_rb/r2_jersey_annotator.html), PACK_DIR/jersey_clips_manifest.js and
PACK_DIR/frames/<clip_id>/f_<frame>.png for every keyframe (r2_jersey_labels.keyframe_grid; lossless PNG decoded from
the lossless clip). Verifies each clip's sha256 against the registry first (fail closed) and refuses to overwrite.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools_rb import r2_jersey_labels as L  # noqa: E402
from tools_rb import r2_protocol as P  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def build(registry_json, clips_dir, pack_dir):
    with open(registry_json) as f:
        reg = json.load(f)
    if os.path.exists(os.path.join(pack_dir, "index.html")):
        raise SystemExit(f"{pack_dir} already holds a pack; refusing to overwrite")
    os.makedirs(pack_dir, exist_ok=True)
    man = []
    for c in reg["clips"]:
        if c["split"] != "DEV":
            raise SystemExit(f"{c['clip_id']}: split {c['split']} (the jersey dataset is DEV only; fail closed)")
        src = os.path.join(clips_dir, c["clip_id"] + ".mp4")
        if _sha(src) != c["clip_sha256"]:
            raise SystemExit(f"{src}: sha256 differs from the registry (fail closed)")
        grid = L.keyframe_grid(c["fps"], c["frames"])
        out = os.path.join(pack_dir, "frames", c["clip_id"])
        os.makedirs(out)
        sel = "+".join(f"eq(n\\,{g})" for g in grid)
        subprocess.run(["ffmpeg", "-v", "error", "-i", src, "-vf", f"select='{sel}'", "-vsync", "vfr",
                        os.path.join(out, "k_%03d.png")], check=True)
        for i, g in enumerate(grid):
            os.rename(os.path.join(out, f"k_{i + 1:03d}.png"), os.path.join(out, f"f_{g:06d}.png"))
        if sorted(os.listdir(out)) != [f"f_{g:06d}.png" for g in grid]:
            raise SystemExit(f"{c['clip_id']}: keyframe extraction incomplete")
        man.append({k: c[k] for k in ("clip_id", "clip_sha256", "fps", "frames", "source", "split", "proposed_target",
                                      "classes")} | {"width": P.SOURCES[c["source"]]["width"],
                                                       "height": P.SOURCES[c["source"]]["height"], "keyframes": grid})
    with open(os.path.join(pack_dir, "jersey_clips_manifest.js"), "x") as f:
        f.write("window.JERSEY_CLIPS = " + json.dumps(man, indent=1) + ";\n")
    shutil.copyfile(os.path.join(HERE, "r2_jersey_annotator.html"), os.path.join(pack_dir, "index.html"))
    print("PACK", pack_dir, len(man), "clips", sum(len(m["keyframes"]) for m in man), "keyframes")


if __name__ == "__main__":
    build(*sys.argv[1:4])

# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2/R2_PREREGISTRATION.md §3 (clip selection without any tracker or detector)
"""R2 candidate scan: draw candidate windows with a fixed seed from the allowed regions of each R2 source and render a
1-frame-per-second strip of each (ffmpeg only; no tracker, no detector, no model). Selection is then done by eye.

    python3 tools_rb/r2_scan.py candidates SOURCE N SEED > candidates.json
    python3 tools_rb/r2_scan.py strips SOURCE_DIR candidates.json OUT_DIR [FIRST LAST]
    python3 tools_rb/r2_scan.py detail SOURCE_DIR candidates.json OUT_DIR ID [ID ...]
        2 frames per second, 400 px wide, timestamp burned in (seconds from the window start)
"""
import json
import os
import random
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools_rb import r2_protocol as P  # noqa: E402

WINDOW_S = 10.0             # strip length (a candidate; final clips may be trimmed to 5-15 s)
STRIP_FPS = 1               # frames per second in a strip
STRIP_W = 320               # thumbnail width (px)
EDGE_SKIP_S = 300.0         # skip the first / last 5 min of each source (pre-match / post-match material)


def candidates(source, n, seed):
    rng = random.Random(f"{seed}:{source}")
    dur = P.SOURCES[source]["duration_s"]
    out, tries = [], 0
    while len(out) < n and tries < 100000:
        tries += 1
        t = round(rng.uniform(EDGE_SKIP_S, dur - EDGE_SKIP_S - WINDOW_S), 1)
        try:
            split = P.check_window(source, t, t + WINDOW_S)
        except ValueError:
            continue
        if any(abs(t - c["start_s"]) < WINDOW_S * 2 for c in out):
            continue
        out.append({"id": f"{source}_C{len(out):03d}", "source": source, "start_s": t, "end_s": t + WINDOW_S,
                    "split": split})
    return {"source": source, "seed": seed, "n": n, "window_s": WINDOW_S, "protocol": P.PROTOCOL_VERSION,
            "candidates": sorted(out, key=lambda c: c["start_s"])}


def strips(source_dir, cand_json, out_dir, first=0, last=None):
    with open(cand_json) as f:
        c = json.load(f)
    src = os.path.join(source_dir, P.SOURCES[c["source"]]["file"])
    os.makedirs(out_dir, exist_ok=True)
    for item in c["candidates"][first:last]:
        out = os.path.join(out_dir, item["id"] + ".jpg")
        if os.path.exists(out):
            continue
        cols = int(WINDOW_S * STRIP_FPS) // 2
        subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{item['start_s']:.3f}", "-i", src, "-t", f"{WINDOW_S}",
                        "-vf", f"fps={STRIP_FPS},scale={STRIP_W}:-2,tile={cols}x2", "-frames:v", "1", "-q:v", "4",
                        out], check=True)
        print(out)


DETAIL_FPS = 2
DETAIL_W = 400


def detail(source_dir, cand_json, out_dir, ids):
    with open(cand_json) as f:
        c = json.load(f)
    src = os.path.join(source_dir, P.SOURCES[c["source"]]["file"])
    os.makedirs(out_dir, exist_ok=True)
    by_id = {x["id"]: x for x in c["candidates"]}
    for i in ids:
        item = by_id[i]
        out = os.path.join(out_dir, i + "_detail.jpg")
        n = int(WINDOW_S * DETAIL_FPS)
        cols = 5
        draw = ("drawtext=text='%{pts\\:hms}':x=6:y=6:fontsize=22:fontcolor=white:box=1:boxcolor=black")
        subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{item['start_s']:.3f}", "-i", src, "-t", f"{WINDOW_S}",
                        "-vf", f"fps={DETAIL_FPS},scale={DETAIL_W}:-2,{draw},tile={cols}x{n // cols}",
                        "-frames:v", "1", "-q:v", "3", out], check=True)
        print(out)


if __name__ == "__main__":
    if sys.argv[1] == "candidates":
        json.dump(candidates(sys.argv[2], int(sys.argv[3]), int(sys.argv[4])), sys.stdout, indent=1)
    elif sys.argv[1] == "strips":
        a = sys.argv[2:]
        strips(a[0], a[1], a[2], *(int(x) for x in a[3:5]))
    elif sys.argv[1] == "detail":
        detail(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:])
    else:
        raise SystemExit(__doc__)

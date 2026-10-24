# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R0_TRACKER_ACCEPTANCE_TARGETS.md (Part B truth model, B5 GOLD collision check),
#                 R0_TESTS_AND_MODELS.md (fixtures X1-X3, X6), R0_TRACKER_FAILURE_CATALOGUE.md (F1, F7, F15)
"""
make_gold_label_sheet.py — builds the HUMAN identity-labelling sheets for GOLD_DUEL_01.

Scope:
  - Covers extract frames 250-290, the window around the #23 / MAR #4 collision.
  - NEW tool. It is not part of the tracker, and it runs no detector, tracker or ReID model.

What the sheets show:
  - Each frame at full width (downscaled for layout).
  - A 1.29x zoom of one FIXED window.
    - The window's coordinates are CLI parameters.
    - The default was chosen by eye from raw frames 250, 268 and 290, NOT from any tracker output.
  - A thin grey outline marking the zoom window on the full frame.
  - The frame number and video time.

What the sheets never show:
  - Tracker boxes, detector boxes, states, track ids, or any earlier Claude visual label.
  - Those labels are DIAGNOSTIC_ONLY and must not become truth (R1 prompt §42).

Reference page:
  - Frame 180 with the FIXTURE seed box: GOLD_DUEL_01_BENCHMARK_FIXTURE.json sha256 7929bb28...
  - It is for orientation only (who #23 is). Nathan signed off the target identity in GOLD_DUEL_01_HUMAN_SIGNOFF.json.

How frames are read (F15):
  - The source is verified by SHA-256 before anything is read.
  - Decoding is SEQUENTIAL from frame 0, with no seek, so frame indices are exact.

Outputs (in OUT_DIR):
  - sheet_00_reference_seed_frame180.jpg
  - sheet_NN_frames_XXX-YYY.jpg
  - GOLD_HUMAN_LABELS_TEMPLATE.csv: one row per frame, to be filled in by Nathan.
  - LABELLING_INSTRUCTIONS.md
  - LABEL_SHEET_MANIFEST.json: inputs, parameters, and SHA-256 of every output.

Usage:
  python3 make_gold_label_sheet.py VIDEO FIXTURE_JSON OUT_DIR [--first 250] [--last 290]
      [--zoom x1 y1 x2 y2] [--rows 4]
"""
import argparse
import csv
import hashlib
import json
import os
import sys

import cv2
import numpy as np

EXPECTED_VIDEO_SHA256 = "164541108ca719f574b54533686cee1f662607930a35bdd5a58df9e5257fd3e9"
EXPECTED_FIXTURE_SHA256 = "7929bb2803eecd7a856b64700095fcc447eecfce6020c7dd8d38915b90bd0a79"
DEFAULT_ZOOM = (640, 220, 1200, 500)   # full-res px; chosen by eye from raw frames 250/268/290 (see docstring)
FULL_W, FULL_H = 640, 360              # layout size of the full-frame panel
ZOOM_H = 360                           # layout height of the zoom panel
CAPTION_H = 34
LABEL_COLUMNS = [
    "frame", "video_time_s",
    "t23_visibility",            # VISIBLE / PARTIAL / HIDDEN / OUT_OF_FRAME / HUMAN_UNCERTAIN
    "t23_identity_resolvable",   # YES / NO
    "t23_location_note",         # free text, e.g. "behind #4, right side" (optional)
    "mar4_visibility",           # VISIBLE / PARTIAL / HIDDEN / OUT_OF_FRAME / HUMAN_UNCERTAIN
    "overlap_23_4",              # YES / NO / HUMAN_UNCERTAIN
    "merged_or_hidden",          # NONE / MERGED (one body-shape for both) / T23_HIDDEN / HUMAN_UNCERTAIN
    "first_clean_reappearance",  # Y on the single frame where #23 is first clearly visible again after hiding; else blank
    "notes",
]


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def caption(img, text):
    bar = np.full((CAPTION_H, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    return np.vstack([bar, img])


def frame_row(frame, idx, fps, zoom):
    x1, y1, x2, y2 = zoom
    full = frame.copy()
    cv2.rectangle(full, (x1, y1), (x2, y2), (160, 160, 160), 2)   # zoom-window outline only (not a detection)
    full = cv2.resize(full, (FULL_W, FULL_H), interpolation=cv2.INTER_AREA)
    crop = frame[y1:y2, x1:x2]
    zw = int(round(crop.shape[1] * ZOOM_H / crop.shape[0]))
    crop = cv2.resize(crop, (zw, ZOOM_H), interpolation=cv2.INTER_CUBIC)
    gap = np.full((FULL_H, 8, 3), 255, np.uint8)
    row = np.hstack([full, gap, crop])
    return caption(row, f"extract frame {idx}   t = {idx / fps:.3f} s   (left: full frame, grey = zoom window; right: zoom)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("fixture_json")
    ap.add_argument("out_dir")
    ap.add_argument("--first", type=int, default=250)
    ap.add_argument("--last", type=int, default=290)
    ap.add_argument("--zoom", type=int, nargs=4, default=list(DEFAULT_ZOOM))
    ap.add_argument("--rows", type=int, default=4)
    a = ap.parse_args(argv)

    vsha = sha256_file(a.video)
    if vsha != EXPECTED_VIDEO_SHA256:
        sys.exit(f"REFUSED: video sha256 {vsha} != expected GOLD extract {EXPECTED_VIDEO_SHA256}")
    fsha = sha256_file(a.fixture_json)
    if fsha != EXPECTED_FIXTURE_SHA256:
        sys.exit(f"REFUSED: fixture sha256 {fsha} != expected {EXPECTED_FIXTURE_SHA256}")
    fx = json.load(open(a.fixture_json))
    seed_frame, seed_bbox = int(fx["seed_frame"]), [float(v) for v in fx["seed_bbox"]]

    os.makedirs(a.out_dir, exist_ok=True)
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    wanted = set(range(a.first, a.last + 1)) | {seed_frame}
    frames, idx = {}, 0
    while idx <= max(wanted):                       # sequential decode from frame 0, no seek (F15)
        ok, fr = cap.read()
        if not ok:
            break
        if idx in wanted:
            frames[idx] = fr
        idx += 1
    cap.release()
    missing = sorted(wanted - set(frames))
    if missing:
        sys.exit(f"REFUSED: could not decode frames {missing}")

    outputs = []
    # reference page: seed frame with the FIXTURE seed box (orientation only)
    ref = frames[seed_frame].copy()
    bx = [int(round(v)) for v in seed_bbox]
    cv2.rectangle(ref, (bx[0], bx[1]), (bx[2], bx[3]), (0, 255, 255), 3)
    ref = caption(cv2.resize(ref, (1280, 720)),
                  f"REFERENCE ONLY: extract frame {seed_frame} (t={seed_frame / fps:.2f}s), yellow = fixture seed box "
                  f"= SOL #23 (royal blue). MAR #4 = sky-blue/white defender.")
    p = os.path.join(a.out_dir, f"sheet_00_reference_seed_frame{seed_frame}.jpg")
    cv2.imwrite(p, ref, [cv2.IMWRITE_JPEG_QUALITY, 92])
    outputs.append(p)

    idxs = list(range(a.first, a.last + 1))
    for s, i0 in enumerate(range(0, len(idxs), a.rows), start=1):
        chunk = idxs[i0:i0 + a.rows]
        rows = [frame_row(frames[i], i, fps, tuple(a.zoom)) for i in chunk]
        w = max(r.shape[1] for r in rows)
        rows = [np.hstack([r, np.full((r.shape[0], w - r.shape[1], 3), 255, np.uint8)]) for r in rows]
        sheet = np.vstack(rows)
        p = os.path.join(a.out_dir, f"sheet_{s:02d}_frames_{chunk[0]}-{chunk[-1]}.jpg")
        cv2.imwrite(p, sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
        outputs.append(p)

    tpl = os.path.join(a.out_dir, "GOLD_HUMAN_LABELS_TEMPLATE.csv")
    with open(tpl, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(LABEL_COLUMNS)
        for i in idxs:
            w.writerow([i, f"{i / fps:.3f}"] + [""] * (len(LABEL_COLUMNS) - 2))
    outputs.append(tpl)

    instr = os.path.join(a.out_dir, "LABELLING_INSTRUCTIONS.md")
    with open(instr, "w") as f:
        f.write(INSTRUCTIONS.format(first=a.first, last=a.last, seed=seed_frame))
    outputs.append(instr)

    manifest = dict(
        tool="tools_rb/make_gold_label_sheet.py (SPANA engine v2-rebuild, NEW)",
        video=os.path.basename(a.video), video_sha256=vsha, fixture_sha256=fsha,
        fps=fps, first=a.first, last=a.last, zoom_window_px=list(a.zoom), rows_per_sheet=a.rows,
        decode="sequential from frame 0, no seek",
        model_outputs_shown="NONE (no detector/tracker/ReID run; no prior labels shown)",
        outputs={os.path.basename(p): sha256_file(p) for p in outputs},
    )
    with open(os.path.join(a.out_dir, "LABEL_SHEET_MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


INSTRUCTIONS = """# GOLD_DUEL_01: human identity labels, extract frames {first}–{last}

**Status: AWAITING NATHAN.** These labels become R1/R2 truth only after you fill them in and approve them. The sheets show **no tracker or model output**.

## Who's who

`sheet_00_reference_seed_frame{seed}.jpg` shows frame {seed}. The yellow box is the fixture seed:

- **SOL #23** wears royal blue. He is the target.
- **MAR #4** wears sky-blue and white. He is the defender.

## How to fill in `GOLD_HUMAN_LABELS_TEMPLATE.csv`

Fill in one row per frame. Leave a cell blank only where these instructions allow it.

| Column | Values | Meaning |
|---|---|---|
| t23_visibility | VISIBLE / PARTIAL / HIDDEN / OUT_OF_FRAME / HUMAN_UNCERTAIN | How much of #23 can be seen. |
| t23_identity_resolvable | YES / NO | Could a careful person tell, from this frame and its neighbours, which body is #23? |
| t23_location_note | free text (optional) | For example, "behind #4", "left of #4", "upper body only". |
| mar4_visibility | VISIBLE / PARTIAL / HIDDEN / OUT_OF_FRAME / HUMAN_UNCERTAIN | How much of #4 can be seen. |
| overlap_23_4 | YES / NO / HUMAN_UNCERTAIN | Do the two players overlap in the image? |
| merged_or_hidden | NONE / MERGED / T23_HIDDEN / HUMAN_UNCERTAIN | MERGED means both bodies look like one shape. T23_HIDDEN means #23 is fully behind #4. |
| first_clean_reappearance | Y or blank | Put **Y** on exactly one frame: the first frame where #23 is clearly visible again after being hidden or overlapped. |
| notes | free text | Anything else. |

**If you can't tell, write HUMAN_UNCERTAIN.** Don't guess. HUMAN_UNCERTAIN frames are excluded from hard scoring.

## When you're done

1. Save the file as `GOLD_HUMAN_LABELS_v1.csv` in the same folder.
2. Tell Claude it's approved.

Claude will then:
- hash and freeze the file before any tracker run;
- record your sign-off.
"""

if __name__ == "__main__":
    main()

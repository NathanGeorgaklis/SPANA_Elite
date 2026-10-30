# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2 DEV run (2026-10-30): RUN_REFUSED "no detection overlaps the seed box by IoU >= 0.5" on
#                 DEV clips; diagnosis must find the first failing layer (rule 2) without changing anything
"""Read-only diagnosis of a seed refusal: compares the human seed box with the RAW detector output on the seed frame.

    python3 tools_rb/r2_seed_diag.py REPO CLIPS_DIR OUT_JSON CLIP_ID [CLIP_ID ...]

DEV fixtures only (refuses EVAL). Uses the configured production detector exactly as the run does. Changes no
tracker logic or setting. Reports, per clip: the seed box (size, aspect), every raw detection with IoU > 0 against
it (box, score, IoU, the fraction of the seed box it covers, and the fraction of the detection inside the seed box),
so the refusal can be classified (human box looser / tighter / offset than the detector's body box, or no body
detected).
"""
import json
import os
import sys

import cv2
import numpy as np


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0, inter


def main(repo, clips_dir, out_json, clip_ids):
    sys.path.insert(0, repo)
    from player_tracker_rb.config import TrackerConfig
    from player_tracker_rb.detector_yolox import YoloxOnnxDetector
    det = YoloxOnnxDetector(TrackerConfig.default())
    out = {}
    for cid in clip_ids:
        with open(os.path.join(repo, "R2", "labels", cid + ".fixture.json")) as f:
            fx = json.load(f)
        if fx["split"] != "DEV":
            raise SystemExit("DEV only (fail closed)")
        cap = cv2.VideoCapture(os.path.join(clips_dir, fx["video_file"]))
        for _ in range(fx["seed_frame"]):
            cap.grab()
        ok, img = cap.read()
        cap.release()
        if not ok:
            raise SystemExit(f"{cid}: cannot read seed frame")
        sb = [float(v) for v in fx["seed_bbox"]]
        sa = (sb[2] - sb[0]) * (sb[3] - sb[1])
        rows = []
        for c in det.detect(img):
            b = [float(v) for v in c.bbox]
            iou, inter = _iou(sb, b)
            if iou <= 0:
                continue
            da = (b[2] - b[0]) * (b[3] - b[1])
            rows.append({"bbox": [round(v, 1) for v in b], "score": round(float(c.score), 3), "iou": round(iou, 3),
                         "covers_seed": round(inter / sa, 3), "inside_seed": round(inter / da, 3),
                         "h_ratio_det_over_seed": round((b[3] - b[1]) / (sb[3] - sb[1]), 3),
                         "w_ratio_det_over_seed": round((b[2] - b[0]) / (sb[2] - sb[0]), 3),
                         "pass": c.detector_pass})
        rows.sort(key=lambda r: -r["iou"])
        out[cid] = {"seed_frame": fx["seed_frame"], "seed_bbox": sb,
                    "seed_w_h": [round(sb[2] - sb[0], 1), round(sb[3] - sb[1], 1)], "detections_overlapping": rows}
    with open(out_json, "x") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:])

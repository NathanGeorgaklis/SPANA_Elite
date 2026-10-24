"""Task 3M Part 10/14 -- DIAGNOSTIC measurement only (nothing here is wired into any pipeline).

For each clip, re-reads the Pass-1 selected-track box for every frame and measures two independent
post-tracker identity signals for it:
  kit_sim  : football_analysis.colour KitDescriptor similarity of the box's torso to the target's
             own clean reference kit (median of the first 15 frames' descriptors, with no other
             track's box overlapping the selected box)
  reid_cos : ReID cosine (separate AppearanceEncoder, no OCR) of the box to the target's clean
             reference embedding (the same 15 frames)
Then it counts which frames a candidate rule would flag:
  SWAP_SUSPECT = kit_sim < MIN_CONFIDENT_SIM (0.55, existing constant) AND descriptor not
                 low-evidence AND reid_cos < (clean-reference median reid_cos - 0.25)
                 for >= 3 consecutive frames.
The 0.25 / 3-frame values are illustrative, not tuned and not proposed as final constants.

usage: python3 selected_box_integrity_signals.py OUT_JSON
"""
import json
import os
import sys

import cv2
import numpy as np

from football_analysis.colour import shirt_color_hist, color_similarity
from football_analysis.possession_detector import MIN_CONFIDENT_SIM
from player_tracker.appearance import AppearanceEncoder
from player_tracker.geometry import iou

EE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UP = "/root/.claude/uploads/98d69dca-3149-5a43-9670-3c417f2af6d4/"
CLIPS = {
    "GOLD_DUEL_01": (os.path.join(EE, "task3k/GOLD_DUEL_01_syo_mar_3036_3048.mp4"),
                     os.path.join(EE, "task3k/mo_tracking/GOLD_DUEL_01_syo_mar_3036_3048_mo_tracking_results.json")),
    "pass_completed_01": (UP + "f5bd8ce0-pass_completed_01.mov",
                          os.path.join(EE, "pass_completed_01/mo_tracking/f5bd8ce0-pass_completed_01_mo_tracking_results.json")),
    "duel_lost_01": (UP + "8a710009-dual_lost_01.mov",
                     os.path.join(EE, "duel_lost_01/mo_tracking/8a710009-dual_lost_01_mo_tracking_results.json")),
}
enc = AppearanceEncoder(use_ocr=False)
out = {}
for clip, (video, jpath) in CLIPS.items():
    d = json.load(open(jpath))
    start = d["summary"]["start_frame"]
    fl = d["frame_log"]
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    cap.read()  # frame 0 (seed)
    recs = []
    for rec in fl:
        ok, frame = cap.read()
        if not ok:
            break
        s = next((t for t in rec["tracks"] if t["selected_player"]), None)
        if s is None:
            recs.append({"frame": rec["frame"], "state": None})
            continue
        box = s["bbox"]
        desc = shirt_color_hist(frame, box)
        emb = enc.embed(frame, box)
        others = [t["bbox"] for t in rec["tracks"] if not t["selected_player"]
                  and t["state"] not in ("TEMPORARILY_LOST", "OUT_OF_FRAME", "TERMINATED")]
        max_iou = max([iou(box, b) for b in others] or [0.0])
        recs.append({"frame": rec["frame"], "state": s["state"], "idc": s["identity_confidence"],
                     "desc": desc, "emb": emb, "max_iou_other": max_iou})
    cap.release()
    clean = [r for r in recs[:40] if r.get("state") == "ACTIVE" and r["max_iou_other"] < 0.05
             and r["desc"] is not None and not r["desc"].low_evidence and r["emb"] is not None][:15]
    kit_ref = np.median(np.array([r["desc"].dominant_lab for r in clean]), axis=0)
    emb_ref = np.mean([r["emb"] for r in clean], axis=0)
    emb_ref /= np.linalg.norm(emb_ref)
    series = []
    for r in recs:
        if r.get("state") is None:
            continue
        ks = round(color_similarity(r["desc"], tuple(kit_ref)), 3) if r["desc"] is not None else None
        rc = round(float(np.dot(r["emb"], emb_ref)), 3) if r["emb"] is not None else None
        series.append({"frame": r["frame"], "state": r["state"], "idc": r["idc"], "kit_sim": ks,
                       "kit_low_evidence": bool(r["desc"].low_evidence) if r["desc"] is not None else True,
                       "reid_cos": rc, "max_iou_other": round(r["max_iou_other"], 3)})
    clean_frames = {r["frame"] for r in clean}
    reid_med = float(np.median([x["reid_cos"] for x in series if x["frame"] in clean_frames]))
    raw = [x["kit_sim"] is not None and x["kit_sim"] < MIN_CONFIDENT_SIM and not x["kit_low_evidence"]
           and x["reid_cos"] is not None and x["reid_cos"] < reid_med - 0.25 for x in series]
    flagged = set()
    i = 0
    while i < len(raw):
        if raw[i]:
            j = i
            while j < len(raw) and raw[j]:
                j += 1
            if j - i >= 3:
                flagged.update(series[k]["frame"] for k in range(i, j))
            i = j
        else:
            i += 1
    for x in series:
        x["raw_suspect"] = bool(raw[series.index(x)])
        x["SWAP_SUSPECT"] = x["frame"] in flagged
    out[clip] = {"start_frame": start, "clean_ref_frames": sorted(clean_frames), "reid_clean_median": round(reid_med, 3),
                 "n_frames": len(series), "flagged_frames": sorted(flagged),
                 "raw_suspect_frames": [x["frame"] for x in series if x["raw_suspect"]], "series": series}
    print(clip, "clean ref", len(clean), "reid median", round(reid_med, 3), "flagged", sorted(flagged),
          "raw", out[clip]["raw_suspect_frames"])
json.dump(out, open(sys.argv[1], "w"), indent=0)

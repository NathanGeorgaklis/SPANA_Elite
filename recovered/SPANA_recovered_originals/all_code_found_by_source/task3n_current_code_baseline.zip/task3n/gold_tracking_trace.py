"""Task 3N Part 4/6: selected-target and Marconi #4 timeline from a Pass-1 tracking JSON (read-only).

Physical identity of the selected box is measured, not assumed:
  * kit evidence: football_analysis.colour LAB KitDescriptor similarity of the selected box to the clean
    #23 kit and the clean #4 kit (references from Task 3M, extract 240-257, work/references.json)
  * #4's reference detections per frame are the visually labelled #4 boxes from Task 3M's trace
    (the detector is deterministic, so the same boxes appear in any fresh run). The #4 track is the
    track whose box best overlaps that detection.
Labels: '#23' if sim23 - sim4 > 0.12, '#4' if sim4 - sim23 > 0.12, else 'MIXED/LOW-EVIDENCE'.
0.12 is the existing football_analysis MARGIN_THRESH. Visual verification is done separately
(overlay sheet), and the report uses the visual verdict where they disagree.

usage: python3 gold_tracking_trace.py MO_JSON OUT_CSV
"""
import csv
import json
import os
import sys

import cv2

from football_analysis.colour import shirt_color_hist, color_similarity
from player_tracker.geometry import iou

EE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = json.load(open(os.path.join(EE, "task3m", "work", "references.json")))
K23, K4 = tuple(REF["kit23_lab"]), tuple(REF["kit4_lab"])
TRACE3M = {int(r["extract_frame"]): r for r in csv.DictReader(open(os.path.join(EE, "task3m", "TASK3M_TARGET_IDENTITY_FRAME_TRACE.csv")))}
FRAMES = os.path.join(EE, "task3j", "gold_frames", "g{:03d}.png")
DORM = ("TEMPORARILY_LOST", "OUT_OF_FRAME", "TERMINATED")

mo_json, out_csv = sys.argv[1], sys.argv[2]
d = json.load(open(mo_json))
start = d["summary"]["start_frame"]
rows = []
for rec in d["frame_log"]:
    ef = start + rec["frame"]
    if not 250 <= ef <= 285:
        continue
    img = cv2.imread(FRAMES.format(ef))
    s = next((t for t in rec["tracks"] if t["selected_player"]), None)
    desc = shirt_color_hist(img, s["bbox"]) if s else None
    s23 = color_similarity(desc, K23) if desc is not None else None
    s4 = color_similarity(desc, K4) if desc is not None else None
    if s23 is None or desc.low_evidence:
        phys = "LOW-EVIDENCE"
    elif s23 - s4 > 0.12:
        phys = "#23"
    elif s4 - s23 > 0.12:
        phys = "#4"
    else:
        phys = "MIXED"
    b4 = json.loads(TRACE3M[ef]["c4_bbox"]) if TRACE3M[ef]["c4_bbox"] else None
    four = None
    if b4:
        cands = [t for t in rec["tracks"] if not t["selected_player"] and t["state"] not in DORM]
        if cands:
            best = max(cands, key=lambda t: iou(t["bbox"], b4))
            if iou(best["bbox"], b4) > 0.3:
                four = best
    rows.append({
        "extract_frame": ef, "engine_frame": rec["frame"], "video_time_s": round(ef / 30.0, 3),
        "sel_track_id": s["track_id"] if s else None, "sel_online_id": s["online_track_id"] if s else None,
        "sel_state": s["state"] if s else "NONE", "sel_bbox": [round(v, 1) for v in s["bbox"]] if s else None,
        "sel_position_source": s.get("position_source") if s else None,
        "sel_detection_backed": (s.get("position_source") == "CONFIRMED_DETECTION") if s else False,
        "sel_identity_confidence": s.get("identity_confidence") if s else None,
        "sel_occlusion_status": s.get("occlusion_status") if s else None,
        "sel_ambiguous": (s["state"] == "AMBIGUOUS") if s else None,
        "sel_kit_sim_23": round(s23, 3) if s23 is not None else None,
        "sel_kit_sim_4": round(s4, 3) if s4 is not None else None,
        "sel_physical_by_kit": phys,
        "overlap_contests": rec["overlap_contests"],
        "n4_track_id": four["track_id"] if four else None, "n4_state": four["state"] if four else None,
        "n4_team": four["team"] if four else None,
        "n4_team_confidence": four["team_confidence"] if four else None,
        "n4_bbox": [round(v, 1) for v in four["bbox"]] if four else None,
        "n4_in_scorable_population": (four["state"] not in ("PROVISIONAL",) + DORM) if four else None,
    })
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow({k: json.dumps(v) if isinstance(v, (list, bool)) or v is None else v for k, v in r.items()})
print("wrote", out_csv, len(rows))

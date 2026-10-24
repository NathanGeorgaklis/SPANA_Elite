"""Task 3M Part 10/14 -- DIAGNOSTIC measurement only (not wired into anything).

Contrastive "occluder takeover" signal, computed from the Pass-1 frame_log plus the video:
  1. Track every non-selected track box that comes into contact with the selected box (IoU >= 0.10).
     That track is an OCCLUDER CANDIDATE. Its kit reference is the median KitDescriptor of its own box
     over its frames with IoU < 0.05 to the selected box, taken BEFORE first contact (at least 3 frames).
  2. Kits are only treated as distinguishable when sim(occluder_ref, target_ref) < MIN_CONFIDENT_SIM
     (0.55, an existing constant). Same-kit contacts are reported as UNTESTABLE, never as clean.
  3. For up to 30 frames after first contact, the selected box is OCCLUDER_KIT if
     sim(selected, occluder_ref) - sim(selected, target_ref) > MARGIN_THRESH (0.12, existing constant)
     and the descriptor is not low-evidence.
The target reference is the median descriptor of the selected box over its first 15 contact-free frames.
usage: python3 occluder_contrast_signal.py OUT_JSON
"""
import json
import os
import sys

import cv2
import numpy as np

from football_analysis.colour import shirt_color_hist, color_similarity
from football_analysis.possession_detector import MIN_CONFIDENT_SIM, MARGIN_THRESH
from player_tracker.geometry import iou

EE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UP = "/root/.claude/uploads/98d69dca-3149-5a43-9670-3c417f2af6d4/"
CLIPS = {  # same table as selected_box_integrity_signals.py
    "GOLD_DUEL_01": (os.path.join(EE, "task3k/GOLD_DUEL_01_syo_mar_3036_3048.mp4"),
                     os.path.join(EE, "task3k/mo_tracking/GOLD_DUEL_01_syo_mar_3036_3048_mo_tracking_results.json")),
    "pass_completed_01": (UP + "f5bd8ce0-pass_completed_01.mov",
                          os.path.join(EE, "pass_completed_01/mo_tracking/f5bd8ce0-pass_completed_01_mo_tracking_results.json")),
    "duel_lost_01": (UP + "8a710009-dual_lost_01.mov",
                     os.path.join(EE, "duel_lost_01/mo_tracking/8a710009-dual_lost_01_mo_tracking_results.json")),
}

DORM = ("TEMPORARILY_LOST", "OUT_OF_FRAME", "TERMINATED")
res = {}
for clip, (video, jpath) in CLIPS.items():
    d = json.load(open(jpath))
    fl = d["frame_log"]
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, d["summary"]["start_frame"])
    cap.read()
    per = []
    for rec in fl:
        ok, frame = cap.read()
        if not ok:
            break
        s = next((t for t in rec["tracks"] if t["selected_player"]), None)
        if s is None:
            per.append(None)
            continue
        others = {}
        for t in rec["tracks"]:
            if t["selected_player"] or t["state"] in DORM:
                continue
            ov = iou(s["bbox"], t["bbox"])
            if ov >= 0.10 or ov < 0.05:
                others[t["track_id"]] = (ov, shirt_color_hist(frame, t["bbox"]) if ov < 0.05 else None)
        per.append({"frame": rec["frame"], "sel": shirt_color_hist(frame, s["bbox"]), "others": others})
    cap.release()
    clean = [p for p in per[:60] if p and p["sel"] is not None and not p["sel"].low_evidence
             and all(ov < 0.05 for ov, _ in p["others"].values())][:15]
    target_ref = tuple(np.median([p["sel"].dominant_lab for p in clean], axis=0))
    contacts, flagged = [], {}
    first_contact = {}
    for i, p in enumerate(per):
        if not p:
            continue
        for tid, (ov, _) in p["others"].items():
            if ov >= 0.10 and tid not in first_contact:
                first_contact[tid] = i
    for tid, i0 in sorted(first_contact.items(), key=lambda kv: kv[1]):
        pre = [p["others"][tid][1] for p in per[max(0, i0 - 30):i0]
               if p and tid in p["others"] and p["others"][tid][0] < 0.05 and p["others"][tid][1] is not None
               and not p["others"][tid][1].low_evidence]
        entry = {"occluder_track": tid, "first_contact_frame": per[i0]["frame"], "n_pre_samples": len(pre)}
        if len(pre) < 3:
            entry["result"] = "UNTESTABLE_NO_PRE_CONTACT_KIT"
            contacts.append(entry)
            continue
        occ_ref = tuple(np.median([x.dominant_lab for x in pre], axis=0))
        kit_sep = color_similarity(occ_ref, target_ref)
        entry["occluder_vs_target_kit_sim"] = round(kit_sep, 3)
        if kit_sep >= MIN_CONFIDENT_SIM:
            entry["result"] = "UNTESTABLE_SAME_KIT"
            contacts.append(entry)
            continue
        hits = []
        for p in per[i0:i0 + 30]:
            if not p or p["sel"] is None or p["sel"].low_evidence:
                continue
            so = color_similarity(p["sel"], occ_ref)
            st = color_similarity(p["sel"], target_ref)
            if so - st > MARGIN_THRESH:
                hits.append(p["frame"])
                flagged[p["frame"]] = {"occluder": tid, "sim_occluder": round(so, 3), "sim_target": round(st, 3)}
        entry["occluder_kit_frames"] = hits
        entry["result"] = "OCCLUDER_KIT_IN_SELECTED_BOX" if hits else "NO_TAKEOVER"
        contacts.append(entry)
    res[clip] = {"target_ref_lab": target_ref, "n_contacts": len(contacts), "contacts": contacts,
                 "flagged_frames": sorted(flagged), "flag_detail": flagged}
    print(clip, "contacts", len(contacts),
          {r: sum(c["result"] == r for c in contacts) for r in {c["result"] for c in contacts}},
          "flagged", sorted(flagged))
json.dump(res, open(sys.argv[1], "w"), indent=1, default=float)

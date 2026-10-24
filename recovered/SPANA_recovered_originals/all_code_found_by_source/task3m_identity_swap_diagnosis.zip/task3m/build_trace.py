"""Task 3M -- build TASK3M_TARGET_IDENTITY_FRAME_TRACE.csv and the numeric evidence for the audits.

Inputs (all read-only):
  * scored Pass-1 frame_log (Task 3K JSON, sha d8926d...)
  * instrumented replay capture (association inputs/outputs, raw detections + confidences)
  * diagnostic ReID embeddings (separate encoder instance) and the extract frames

Physical identity labels for the candidates (#23 / #4) come from visual inspection of the
detection overlays (TASK3M_COLLISION_DETECTION_OVERLAYS). The PHYSICAL_LABELS table below records
them. Kit-pixel share and ReID similarity to clean references are reported as independent
numeric checks, never as the source of a label.
"""
import csv
import json
import os

import cv2
import numpy as np

from football_analysis.colour import shirt_color_hist, color_similarity
from player_tracker.geometry import iou, center, dist

ROOT = os.path.dirname(os.path.abspath(__file__))
EE = os.path.dirname(ROOT)
FRAMES = os.path.join(EE, "task3j", "gold_frames", "g{:03d}.png")
FL = json.load(open(os.path.join(EE, "task3k", "mo_tracking",
                                 "GOLD_DUEL_01_syo_mar_3036_3048_mo_tracking_results.json")))["frame_log"]
FL = {r["frame"] + 180: r for r in FL}
CAP = {r["extract_frame"]: r for r in json.load(open(os.path.join(ROOT, "replay", "association_capture.json")))}
EMB = np.load(os.path.join(ROOT, "replay", "embeddings.npz"))

# Visual labels (detection index per extract frame), verified on dets_261_276.png / overlay sheets.
# None = not independently detected. "MERGED" = one box covering both players.
PHYSICAL_LABELS = {
    266: {"23": None, "4": 0, "note": "single detection covers #4 (in front) and the occluded #23"},
    267: {"23": None, "4": 0, "note": "#23 almost fully occluded behind #4; single detection is #4"},
    268: {"23": None, "4": 0, "note": "#23 occluded behind #4; single detection is #4"},
    269: {"23": 4, "4": 2, "note": "#23 re-emerges on the left (upper body only)"},
    270: {"23": 4, "4": 2, "note": ""},
    271: {"23": 5, "4": 2, "note": ""},
    272: {"23": 5, "4": 3, "note": ""},
    273: {"23": 5, "4": 4, "note": "d6 is a second, partial box on #23"},
}


def sel(rec):
    return next(t for t in rec["tracks"] if t["selected_player"])


def track_box(rec, tid):
    for t in rec["tracks"]:
        if t["track_id"] == tid:
            return t["bbox"]
    return None


def best_iou_det(dets, box):
    if box is None:
        return None
    b = max(dets, key=lambda d: iou(d["box"], box))
    return b["j"] if iou(b["box"], box) > 0.5 else None


def candidates(ef):
    """(#23 det index, #4 det index, note)."""
    if ef in PHYSICAL_LABELS:
        L = PHYSICAL_LABELS[ef]
        return L["23"], L["4"], L["note"]
    cap = CAP[ef]
    post = {t["track_id"]: t for t in cap["post_step_tracks"]}
    a = next(x for x in cap["assignments"] if x["track_id"] == "T001")
    j23 = a["det"]                      # outside 266-273 T001's own confirmed det is #23 (visually verified)
    four_tid = "T003" if ef < 266 else "T016"
    fb = post.get(four_tid, {}).get("current_box")
    j4 = best_iou_det(cap["detections"], fb) if fb else None
    return j23, j4, ""


# ---------------- clean references (before any overlap: extract 240-257)
def ref_emb(label):
    acc = []
    for ef in range(240, 258):
        j23, j4, _ = candidates(ef)
        j = j23 if label == "23" else j4
        k = f"f{ef-180}_d{j}"
        if j is not None and k in EMB:
            acc.append(EMB[k])
    m = np.mean(acc, axis=0)
    return m / np.linalg.norm(m), len(acc)


REF23, N23 = ref_emb("23")
REF4, N4 = ref_emb("4")


def lab_ref(label):
    acc = []
    for ef in range(240, 258):
        img = cv2.imread(FRAMES.format(ef))
        j23, j4, _ = candidates(ef)
        j = j23 if label == "23" else j4
        if j is None:
            continue
        d = shirt_color_hist(img, CAP[ef]["detections"][j]["box"])
        if d is not None and not d.low_evidence:
            acc.append(d.dominant_lab)
    return tuple(np.mean(acc, axis=0))


KIT23 = lab_ref("23")
KIT4 = lab_ref("4")


def kit_share(img, box):
    """Share of non-grass pixels in the upper 60% of the box nearer (LAB) to #23's kit than to #4's,
    among pixels within 30 LAB units of either kit centroid. Diagnostic only."""
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    h = y2 - y1
    crop = img[max(0, y1):max(0, y1 + int(0.6 * h)), max(0, x1):max(0, x2)]
    if crop.size == 0:
        return None, None
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float64)
    # same OpenCV 8-bit LAB space as football_analysis.colour's KitDescriptor.dominant_lab
    d23 = np.linalg.norm(lab - np.array(KIT23), axis=1)
    d4 = np.linalg.norm(lab - np.array(KIT4), axis=1)
    near = (np.minimum(d23, d4) < 30)
    if near.sum() < 30:
        return None, int(near.sum())
    return round(float((d23[near] < d4[near]).mean()), 3), int(near.sum())


def cos(a, b):
    return round(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))), 3)


def score_of(cap, j):
    if j is None:
        return None
    return cap["score_matrix"]["T001"][j]


def r1(b):
    return [round(v, 1) for v in b] if b is not None else None


rows = []
for ef in range(250, 286):
    rec, cap = FL[ef], CAP[ef]
    s = sel(rec)
    a = next(x for x in cap["assignments"] if x["track_id"] == "T001")
    t1 = next(t for t in cap["scorable"] if t["track_id"] == "T001")
    j23, j4, note = candidates(ef)
    img = cv2.imread(FRAMES.format(ef))
    dets = {d["j"]: d for d in cap["detections"]}
    sc_a = score_of(cap, a["det"])
    sc23, sc4 = score_of(cap, j23), score_of(cap, j4)
    b23 = dets[j23]["box"] if j23 is not None else None
    b4 = dets[j4]["box"] if j4 is not None else None
    # physically selected box = the box written to frame_log (ambiguous frames keep the previous box)
    sbox = s["bbox"]
    ks, kn = kit_share(img, sbox)
    selected_emb_key = f"f{ef-180}_d{a['det']}" if a["det"] is not None else None
    if a["ambiguous"]:
        phys = "#4 (box not updated: previous #4 box retained)" if 266 <= ef <= 273 else "#23 (box not updated)"
    elif a["det"] is None:
        phys = "none"
    elif a["det"] == j23:
        phys = "#23"
    elif a["det"] == j4:
        phys = "#4" if ef != 266 else "MERGED (#4 in front, #23 occluded)"
    else:
        phys = "other"
    gate_applied = cap["gate_matrix"]["T001"][a["det"]] if a["det"] is not None else None
    rows.append({
        "extract_frame": ef, "engine_frame": ef - 180, "video_time_s": round(ef / 30.0, 3),
        "selected_bbox": r1(sbox), "selected_detection_id": a["det"],
        "selected_detection_conf": dets[a["det"]]["det_conf"] if a["det"] is not None else None,
        "online_track_id": s["online_track_id"], "offline_identity_state": "AUTO_RESOLVED_IDENTITY",
        "tracking_state": s["state"], "identity_confidence": s["identity_confidence"],
        "reid_confidence": s["reid_confidence"], "association_score": a["score"],
        "association_ambiguous": a["ambiguous"],
        "motion_score": sc_a["motion_similarity"] if sc_a else None,
        "iou_score": sc_a["iou"] if sc_a else None,
        "appearance_score": sc_a["appearance_similarity"] if sc_a else None,
        "colour_score": sc_a["colour_similarity"] if sc_a else None,
        "continuity_score": s["track_continuity_confidence"], "occlusion_flag": s["occlusion_status"],
        "position_source": s["position_source"],
        "overlap_contest_flag": bool(cap["overlap_contests_scorable"]),
        "overlap_contest_if_provisional_counted_DIAG": any("T001" in p for p in cap["overlap_contests_incl_provisional_DIAG"]),
        "identity_gate_applied": gate_applied, "spot_check_frame": cap["spot_check_frame"],
        "gate_radius": t1["gate_radius"], "predicted_center": t1["pred_center"],
        "n_detections": len(cap["detections"]),
        "c23_bbox": r1(b23), "c23_detection_id": j23,
        "c23_det_conf": dets[j23]["det_conf"] if j23 is not None else None,
        "c23_appearance_score": sc23["appearance_similarity"] if sc23 else None,
        "c23_association_score": sc23["final_score"] if sc23 and sc23["result"] == "ACCEPTED" else (sc23["result"] if sc23 else None),
        "c4_bbox": r1(b4), "c4_detection_id": j4,
        "c4_det_conf": dets[j4]["det_conf"] if j4 is not None else None,
        "c4_appearance_score": sc4["appearance_similarity"] if sc4 else None,
        "c4_association_score": sc4["final_score"] if sc4 and sc4["result"] == "ACCEPTED" else (sc4["result"] if sc4 else None),
        "candidate_center_distance_px": round(dist(center(b23), center(b4)), 1) if b23 and b4 else None,
        "iou_23_4": round(iou(b23, b4), 3) if b23 and b4 else None,
        "selected_candidate": phys,
        "c23_reid_vs_clean23": cos(EMB[f"f{ef-180}_d{j23}"], REF23) if j23 is not None and f"f{ef-180}_d{j23}" in EMB else None,
        "c23_reid_vs_clean4": cos(EMB[f"f{ef-180}_d{j23}"], REF4) if j23 is not None and f"f{ef-180}_d{j23}" in EMB else None,
        "c4_reid_vs_clean23": cos(EMB[f"f{ef-180}_d{j4}"], REF23) if j4 is not None and f"f{ef-180}_d{j4}" in EMB else None,
        "c4_reid_vs_clean4": cos(EMB[f"f{ef-180}_d{j4}"], REF4) if j4 is not None and f"f{ef-180}_d{j4}" in EMB else None,
        "selected_box_kit23_pixel_share": ks,
        "c23_kit_lab_sim_to_23": round(color_similarity(shirt_color_hist(img, b23), KIT23), 3) if b23 else None,
        "c4_kit_lab_sim_to_23": round(color_similarity(shirt_color_hist(img, b4), KIT23), 3) if b4 else None,
        "selected_box_kit_lab_sim_to_23": round(color_similarity(shirt_color_hist(img, sbox), KIT23), 3),
        "selected_box_kit_lab_sim_to_4": round(color_similarity(shirt_color_hist(img, sbox), KIT4), 3),
        "note": note,
    })

out = os.path.join(ROOT, "TASK3M_TARGET_IDENTITY_FRAME_TRACE.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow({k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()})
json.dump({"ref23_n": N23, "ref4_n": N4, "kit23_lab": KIT23, "kit4_lab": KIT4,
           "ref23_vs_ref4_cos": cos(REF23, REF4),
           "kit23_vs_kit4_lab_sim": round(color_similarity(KIT23, KIT4), 3)},
          open(os.path.join(ROOT, "work", "references.json"), "w"), indent=1)
print("wrote", out, len(rows))

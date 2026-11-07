# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2 small-body appearance usability stage (Nathan, 2026-11-07 17:17), reviewer R2-SB H3
#                 re-measurement. Evaluation tooling only; no tracker logic.
"""Diagnosis only (review R2-SB H3 re-measurement): LAB similarity, same tracklet (8-48 frames apart) vs other
tracklets of similar height on the same frame, binned by the LOWER band pixel count (the limiting band), crops at or
above crop_min_height_px only, reported PER CLIP, with the veto rate (< lab_floor) and p95 of the other-tracklet
distribution as primary, and a SAME-KIT subset (other tracklet whose lifetime-median LAB descriptor matches this
tracklet's at >= 0.8). lab_min_band_pixels overridden to 1 so sparse bands are measured, not refused.
Usage: lab_probe2.py REPO OUT_JSON CLIP:LOG:LABEL [...]"""
import json, sys, os
import numpy as np, cv2
sys.path.insert(0, sys.argv[1])
from player_tracker_rb.config import TrackerConfig
from player_tracker_rb.appearance import crop_image, lab_descriptor, lab_similarity

repo, out = sys.argv[1], sys.argv[2]
cfg = TrackerConfig.default(); os.chdir(repo)
cfg_lab = TrackerConfig.with_overrides({"lab_min_band_pixels": 1})
HMIN = cfg.get("crop_min_height_px"); FLOOR = cfg.get("lab_floor")
BINS = [(28, 50), (50, 100), (100, 150), (150, 250), (250, 500), (500, 100000)]


def lower_band_pixels(crop):
    h, w = crop.shape[:2]
    keep = cfg.get("lab_center_width_fraction")
    cx1 = int(round(w * (1 - keep) / 2)); cx2 = max(cx1 + 1, int(round(w * (1 + keep) / 2)))
    lo, hi = cfg.get("lab_lower_band")
    y1, y2 = int(round(h * lo)), int(round(h * hi))
    return max(1, y2 - y1) * (cx2 - cx1)


def b_of(n):
    for lo, hi in BINS:
        if lo <= n < hi: return f"{lo}-{hi}"
    return None


results = {}
for arg in sys.argv[3:]:
    clip, log, label = arg.split(":")
    frames = {}
    for l in open(log):
        d = json.loads(l)
        frames[d["frame"]] = [(o["tracklet_id"], o["bbox"]) for o in d["population"]["observations"]
                              if o["tracklet_id"] and not o["flags"] and o["bbox"][3] - o["bbox"][1] >= HMIN]
    # pass 1: lifetime-median LAB per tracklet (kit), from frames every 4
    cap = cv2.VideoCapture(clip); k = 0; per_tid = {}; feats = {}
    while True:
        ok, img = cap.read()
        if not ok: break
        if k % 4 == 0 and k in frames:
            for tid, bb in frames[k]:
                crop = crop_image(img, bb)
                if crop is None: continue
                lab = lab_descriptor(crop, cfg_lab)
                if lab is None: continue
                feats[(k, tid)] = (lab, lower_band_pixels(crop))
                per_tid.setdefault(tid, []).append(lab)
        k += 1
    kit = {tid: np.median(np.stack(v), axis=0) for tid, v in per_tid.items() if len(v) >= 3}
    for tid in kit:
        kit[tid] = kit[tid] / kit[tid].sum(axis=1, keepdims=True)
    same = {}; diff = {}; diff_kit = {}
    keys = sorted(feats)
    by_frame = {}
    for (k, tid), (lab, n) in feats.items():
        by_frame.setdefault(k, []).append((tid, lab, n))
    for k in sorted(by_frame):
        for tid, lab, n in by_frame[k]:
            bn = b_of(n)
            if bn is None: continue
            for dk in (8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48):
                prev = feats.get((k - dk, tid))
                if prev is not None and b_of(prev[1]) == bn:
                    same.setdefault(bn, []).append(lab_similarity(lab, prev[0]))
            for tid2, lab2, n2 in by_frame[k]:
                if tid2 == tid: continue
                for dk in (8, 16, 24, 32, 40, 48):
                    prev = feats.get((k - dk, tid2))
                    if prev is None: continue
                    s = lab_similarity(lab, prev[0])
                    diff.setdefault(bn, []).append(s)
                    if tid in kit and tid2 in kit and lab_similarity(kit[tid], kit[tid2]) >= 0.8:
                        diff_kit.setdefault(bn, []).append(s)
    rows = {}
    for bn in [f"{a}-{b}" for a, b in BINS]:
        s = np.asarray(same.get(bn, [])); d_ = np.asarray(diff.get(bn, [])); dk_ = np.asarray(diff_kit.get(bn, []))
        if len(s) < 20 or len(d_) < 20: continue
        rows[bn] = dict(n_same=int(len(s)), n_diff=int(len(d_)), n_diff_samekit=int(len(dk_)),
                        same_p5=float(np.percentile(s, 5)), same_med=float(np.median(s)), same_veto=float(np.mean(s < FLOOR)),
                        diff_med=float(np.median(d_)), diff_p95=float(np.percentile(d_, 95)), diff_veto=float(np.mean(d_ < FLOOR)),
                        diff_ge_08=float(np.mean(d_ >= 0.8)),
                        samekit_med=float(np.median(dk_)) if len(dk_) else None, samekit_p5=float(np.percentile(dk_, 5)) if len(dk_) else None,
                        samekit_veto=float(np.mean(dk_ < FLOOR)) if len(dk_) else None)
        r = rows[bn]
        print(f"{label:14s} lower band {bn:>11s}: same n={r['n_same']:4d} p5={r['same_p5']:.3f} veto={r['same_veto']:.3f} | other n={r['n_diff']:5d} med={r['diff_med']:.3f} p95={r['diff_p95']:.3f} veto={r['diff_veto']:.3f} >=0.8:{r['diff_ge_08']:.2f} | same-kit n={r['n_diff_samekit']:4d} med={r['samekit_med']} p5={r['samekit_p5']} veto={r['samekit_veto']}")
    results[label] = rows
json.dump({"crop_min_height_px": HMIN, "lab_floor": FLOOR, "bins_lower_band_px": BINS, "frame_stride": 4,
           "same_pair_gaps_frames": [8, 48], "clips": results}, open(out, "w"), indent=1)

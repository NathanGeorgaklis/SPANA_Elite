# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2 small-body appearance usability stage (Nathan, 2026-11-07 14:53 / 17:17). Evaluation tooling
#                 only (measurements from decision logs and clip pixels); no tracker logic.
"""Diagnosis only: DINOv2 same-tracklet vs different-tracklet similarity binned by box height (px), over several
native clips, with the production preprocessing. Same = the same unflagged tracklet 8-48 frames apart; different =
other unflagged tracklets of similar height on the same frame. Boxes from decision logs; no tracker logic.
Usage: height_probe.py REPO OUT_JSON CLIP:LOG [CLIP:LOG ...]"""
import json, sys, os
import numpy as np, cv2
sys.path.insert(0, sys.argv[1])
from player_tracker_rb.config import TrackerConfig
from player_tracker_rb import embedding_dinov2 as E
from player_tracker_rb.appearance import crop_image, lab_descriptor, lab_similarity

repo, out = sys.argv[1], sys.argv[2]
cfg = TrackerConfig.default(); os.chdir(repo)
backend = E.Dinov2OnnxEmbedding(cfg)
BINS = [(24, 40), (40, 55), (55, 70), (70, 90), (90, 120), (120, 160), (160, 400)]


def b_of(h):
    for lo, hi in BINS:
        if lo <= h < hi: return (lo, hi)
    return None


same = {b: [] for b in BINS}; diff = {b: [] for b in BINS}; lsame = {b: [] for b in BINS}; ldiff = {b: [] for b in BINS}
for arg in sys.argv[3:]:
    clip, log = arg.split(":")
    frames = {}
    for l in open(log):
        d = json.loads(l)
        obs = d["population"]["observations"]
        frames[d["frame"]] = [(o["tracklet_id"], o["bbox"]) for o in obs if o["tracklet_id"] and not o["flags"]]
    cap = cv2.VideoCapture(clip); k = 0; hist = {}
    while True:
        ok, img = cap.read()
        if not ok: break
        if k % 4 == 0 and k in frames:
            cur = []
            for tid, bb in frames[k]:
                h = bb[3] - bb[1]
                if b_of(h) is None: continue
                crop = crop_image(img, bb)
                if crop is None: continue
                e = backend.embed(crop); lab = lab_descriptor(crop, cfg)
                cur.append((tid, bb, h, e, lab))
            for tid, bb, h, e, lab in cur:
                bn = b_of(h)
                for kk, hh2, e2, lab2 in hist.get(tid, []):
                    if 8 <= k - kk <= 48 and b_of(hh2) == bn:
                        same[bn].append(float(np.dot(e, e2)))
                        ls = lab_similarity(lab, lab2)
                        if ls is not None: lsame[bn].append(ls)
                for tid2, bb2, h2, e2, lab2 in cur:
                    if tid2 == tid or not (0.75 * h <= h2 <= 1.33 * h): continue
                    for kk, hh3, e3, lab3 in hist.get(tid2, []):
                        if 8 <= k - kk <= 48:
                            diff[bn].append(float(np.dot(e, e3)))
                            ls = lab_similarity(lab, lab3)
                            if ls is not None: ldiff[bn].append(ls)
            for tid, bb, h, e, lab in cur:
                hist.setdefault(tid, []).append((k, h, e, lab))
                hist[tid] = [x for x in hist[tid] if k - x[0] <= 48]
        k += 1
    print("done", clip, flush=True)
res = {}
for b in BINS:
    s, d_ = np.asarray(same[b]), np.asarray(diff[b])
    if len(s) >= 20 and len(d_) >= 20:
        row = dict(n_same=int(len(s)), n_diff=int(len(d_)), same_p5=float(np.percentile(s, 5)), same_p25=float(np.percentile(s, 25)),
                   same_med=float(np.median(s)), diff_med=float(np.median(d_)), diff_p95=float(np.percentile(d_, 95)),
                   same_ge_floor=float(np.mean(s >= cfg.get("embedding_floor"))), diff_ge_floor=float(np.mean(d_ >= cfg.get("embedding_floor"))),
                   lab_same_med=float(np.median(lsame[b])) if lsame[b] else None, lab_diff_med=float(np.median(ldiff[b])) if ldiff[b] else None)
        res[f"{b[0]}-{b[1]}"] = row
        print(f"h {b[0]:3d}-{b[1]:3d}: same n={row['n_same']:5d} p5={row['same_p5']:.3f} p25={row['same_p25']:.3f} med={row['same_med']:.3f} | "
              f"diff n={row['n_diff']:5d} med={row['diff_med']:.3f} p95={row['diff_p95']:.3f} | same>=floor {row['same_ge_floor']:.2f} diff>=floor {row['diff_ge_floor']:.2f} | "
              f"LAB same {row['lab_same_med']} diff {row['lab_diff_med']}")
json.dump(res, open(out, "w"), indent=1)

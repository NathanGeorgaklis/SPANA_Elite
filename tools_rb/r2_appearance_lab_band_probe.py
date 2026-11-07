# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2 small-body appearance usability stage (Nathan, 2026-11-07 14:53 / 17:17). Evaluation tooling
#                 only (measurements from decision logs and clip pixels); no tracker logic.
"""Diagnosis only: LAB part-colour similarity (same tracklet 8-48 frames apart vs other tracklets of similar
height on the same frame) binned by the UPPER BAND pixel count, computed with lab_min_band_pixels overridden to 1 so
that sparse bands are measured rather than refused. Also the embedding for the same pairs. No tracker logic.
Usage: lab_probe.py REPO OUT_JSON CLIP:LOG [...]"""
import json, sys, os
import numpy as np, cv2
sys.path.insert(0, sys.argv[1])
from player_tracker_rb.config import TrackerConfig
from player_tracker_rb import embedding_dinov2 as E
from player_tracker_rb.appearance import crop_image, lab_descriptor, lab_similarity

repo, out = sys.argv[1], sys.argv[2]
cfg = TrackerConfig.default(); os.chdir(repo)
cfg_lab = TrackerConfig.with_overrides({"lab_min_band_pixels": 1})
backend = E.Dinov2OnnxEmbedding(cfg)
BINS = [(20, 50), (50, 100), (100, 150), (150, 250), (250, 500), (500, 1000), (1000, 100000)]


def band_pixels(crop):
    h, w = crop.shape[:2]
    keep = cfg.get("lab_center_width_fraction")
    cx1 = int(round(w * (1 - keep) / 2)); cx2 = max(cx1 + 1, int(round(w * (1 + keep) / 2)))
    b = cfg.get("lab_upper_band")
    y1, y2 = int(round(h * b[0])), int(round(h * b[1]))
    return max(1, y2 - y1) * (cx2 - cx1)


def b_of(n):
    for lo, hi in BINS:
        if lo <= n < hi: return (lo, hi)
    return None


same = {b: [] for b in BINS}; diff = {b: [] for b in BINS}; esame = {b: [] for b in BINS}; ediff = {b: [] for b in BINS}
hs = {b: [] for b in BINS}
for arg in sys.argv[3:]:
    clip, log = arg.split(":")
    frames = {}
    for l in open(log):
        d = json.loads(l)
        frames[d["frame"]] = [(o["tracklet_id"], o["bbox"]) for o in d["population"]["observations"] if o["tracklet_id"] and not o["flags"]]
    cap = cv2.VideoCapture(clip); k = 0; hist = {}
    while True:
        ok, img = cap.read()
        if not ok: break
        if k % 4 == 0 and k in frames:
            cur = []
            for tid, bb in frames[k]:
                crop = crop_image(img, bb)
                if crop is None or crop.shape[0] < 12: continue
                n = band_pixels(crop); bn = b_of(n)
                if bn is None: continue
                lab = lab_descriptor(crop, cfg_lab); e = backend.embed(crop)
                cur.append((tid, bb, bb[3] - bb[1], n, bn, lab, e))
            for tid, bb, h, n, bn, lab, e in cur:
                for kk, bn2, lab2, e2, h2 in hist.get(tid, []):
                    if 8 <= k - kk <= 48 and bn2 == bn:
                        s = lab_similarity(lab, lab2)
                        if s is not None: same[bn].append(s); esame[bn].append(float(np.dot(e, e2))); hs[bn].append(h)
                for tid2, bb2, h2, n2, bn2, lab2, e2 in cur:
                    if tid2 == tid or not (0.75 * h <= h2 <= 1.33 * h): continue
                    for kk, bn3, lab3, e3, h3 in hist.get(tid2, []):
                        if 8 <= k - kk <= 48:
                            s = lab_similarity(lab, lab3)
                            if s is not None: diff[bn].append(s); ediff[bn].append(float(np.dot(e, e3)))
            for tid, bb, h, n, bn, lab, e in cur:
                hist.setdefault(tid, []).append((k, bn, lab, e, h))
                hist[tid] = [x for x in hist[tid] if k - x[0] <= 48]
        k += 1
    print("done", clip, flush=True)
res = {}
for b in BINS:
    s, d_ = np.asarray(same[b]), np.asarray(diff[b])
    if len(s) >= 20 and len(d_) >= 20:
        es, ed = np.asarray(esame[b]), np.asarray(ediff[b])
        row = dict(n_same=int(len(s)), n_diff=int(len(d_)), body_h_med=float(np.median(hs[b])),
                   lab_same_p5=float(np.percentile(s, 5)), lab_same_med=float(np.median(s)), lab_diff_med=float(np.median(d_)),
                   lab_diff_p95=float(np.percentile(d_, 95)), lab_same_below_floor=float(np.mean(s < cfg.get("lab_floor"))),
                   lab_diff_below_floor=float(np.mean(d_ < cfg.get("lab_floor"))),
                   emb_same_med=float(np.median(es)), emb_diff_med=float(np.median(ed)),
                   emb_same_ge_floor=float(np.mean(es >= cfg.get("embedding_floor"))), emb_diff_ge_floor=float(np.mean(ed >= cfg.get("embedding_floor"))))
        res[f"{b[0]}-{b[1]}"] = row
        print(f"band px {b[0]:5d}-{b[1]:6d} (body h~{row['body_h_med']:.0f}): LAB same p5={row['lab_same_p5']:.3f} med={row['lab_same_med']:.3f} | diff med={row['lab_diff_med']:.3f} p95={row['lab_diff_p95']:.3f} | "
              f"same<floor {row['lab_same_below_floor']:.3f} diff<floor {row['lab_diff_below_floor']:.2f} || emb same {row['emb_same_med']:.3f} diff {row['emb_diff_med']:.3f} same>=fl {row['emb_same_ge_floor']:.2f} diff>=fl {row['emb_diff_ge_floor']:.2f}  n={row['n_same']}/{row['n_diff']}")
json.dump(res, open(out, "w"), indent=1)

# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2 small-body appearance usability stage (Nathan, 2026-11-07 14:53 / 17:17). Evaluation tooling
#                 only (measurements from decision logs and clip pixels); no tracker logic.
"""Diagnosis only: same-person vs different-person DINOv2 similarity on a native clip's small target under the
production preprocessing and alternatives. Boxes from the decision log (target's own tracklet, and other tracked
bodies of similar height), pixels from the clip. No tracker logic.
Usage: probe.py REPO CLIP LOG TID FRAMES_MAX"""
import json, sys, os
import numpy as np, cv2
sys.path.insert(0, sys.argv[1])
from player_tracker_rb.config import TrackerConfig
from player_tracker_rb import embedding_dinov2 as E
from player_tracker_rb.appearance import crop_image

repo, clip, log, tid, fmax = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
cfg = TrackerConfig.default()
os.chdir(repo)
backend = E.Dinov2OnnxEmbedding(cfg)
MEAN = np.asarray(cfg.get("embedding_mean_rgb")); STD = np.asarray(cfg.get("embedding_std_rgb"))
HH, WW = (int(v) for v in cfg.get("embedding_input_hw"))


def pre(crop, interp):
    h, w = crop.shape[:2]
    s = min(HH / h, WW / w)
    nh, nw = max(1, min(HH, int(round(h * s)))), max(1, min(WW, int(round(w * s))))
    rgb = cv2.cvtColor(np.ascontiguousarray(crop), cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (nw, nh), interpolation=(cv2.INTER_AREA if s < 1.0 else interp))
    canvas = np.empty((HH, WW, 3), dtype=np.uint8); canvas[:] = np.round(MEAN * 255).astype(np.uint8)
    y0, x0 = (HH - nh) // 2, (WW - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    x = ((canvas.astype(np.float64) / 255 - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
    return np.ascontiguousarray(x)


def emb(img, box, interp, pad):
    x1, y1, x2, y2 = box
    if pad:
        w, h = x2 - x1, y2 - y1
        box = (x1 - pad * w, y1 - pad * h, x2 + pad * w, y2 + pad * h)
    crop = crop_image(img, box)
    cls, pm = backend._sess.run(list(E.OUTPUT_NAMES), {backend._in: pre(crop, interp)})
    return E.pool(np.asarray(cls[0], float), np.asarray(pm[0], float), cfg.get("embedding_pooling"))


frames = {}
for l in open(log):
    d = json.loads(l); fr = d["frame"]
    if fr > fmax: break
    obs = d["population"]["observations"]
    tb = next((o["bbox"] for o in obs if o["tracklet_id"] == tid and not o["flags"]), None)
    hh = tb[3] - tb[1] if tb else 60
    others = [(o["tracklet_id"], o["bbox"]) for o in obs if o["tracklet_id"] and o["tracklet_id"] != tid
              and not o["flags"] and 0.75 * hh <= o["bbox"][3] - o["bbox"][1] <= 1.3 * hh]
    frames[fr] = (tb, others)

variants = {"linear": (cv2.INTER_LINEAR, 0.0), "cubic": (cv2.INTER_CUBIC, 0.0),
            "lanczos": (cv2.INTER_LANCZOS4, 0.0), "linear_pad10": (cv2.INTER_LINEAR, 0.10)}
same = {k: [] for k in variants}; diff = {k: [] for k in variants}
cap = cv2.VideoCapture(clip); k = 0; prev = {}
sizes = []
while k <= fmax:
    ok, img = cap.read()
    if not ok: break
    tb, others = frames.get(k, (None, []))
    if tb is not None and k % 2 == 0:
        sizes.append((round(tb[2] - tb[0]), round(tb[3] - tb[1])))
        for name, (interp, pad) in variants.items():
            e = emb(img, tb, interp, pad)
            # same person: against the target's own embeddings from 8-48 frames earlier (the bank's spacing)
            for kk, ee in prev.get(name, []):
                if 8 <= k - kk <= 48:
                    same[name].append(float(np.dot(e, ee)))
            # different person: other bodies of similar height on this frame vs the target's earlier embeddings
            for _, ob in others[:3]:
                eo = emb(img, ob, interp, pad)
                for kk, ee in prev.get(name, []):
                    if 8 <= k - kk <= 48:
                        diff[name].append(float(np.dot(eo, ee)))
            prev.setdefault(name, []).append((k, e))
    k += 1
print("target sizes (w,h) sampled:", sizes[:6], "...", sizes[-3:], "n=", len(sizes))
for name in variants:
    s = np.asarray(same[name]); d_ = np.asarray(diff[name])
    if len(s) and len(d_):
        print(f"{name:13s} same n={len(s)} p5={np.percentile(s,5):.3f} p25={np.percentile(s,25):.3f} med={np.median(s):.3f} | "
              f"diff n={len(d_)} med={np.median(d_):.3f} p95={np.percentile(d_,95):.3f} | "
              f"same>=0.70: {np.mean(s>=0.70):.2f}  diff>=0.70: {np.mean(d_>=0.70):.2f}")

# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2 DEV ReID evaluation instruction (Nathan, 2026-11-01 18:57): determine offline whether a stronger
#                 person-ReID model can safely improve reacquisition and ambiguous identity cases; DEV only; do not
#                 replace DINOv2; compare true target vs same-kit / nearby competitors; identity separation, not raw
#                 similarity; no tracker change.
"""Offline appearance-backend comparison on frozen DEV truth (read-only; refuses EVAL fixtures; no tracker change).

    python tools_rb/r2_reid_eval.py REPO CLIPS_DIR RUN_OUT_DIR MODELS_DIR OUT_JSON CLIP_ID [CLIP_ID ...]

Run in a separate environment with torch (the production environment is not modified). For every frozen-truth frame
where the target is available (VISIBLE/PARTIAL, resolvable, not merged, not human-uncertain) and boxed:
  * TRUE crop: the population observation (from the DEV run's decision log, RUN_OUT_DIR) matching the human box
    (IoU >= 0.5, else the largest detection >= 0.9 inside it with >= 0.5 of its height);
  * COMPETITOR crops: every other body observation (not a duplicate, not nested in / containing the true box) whose
    centre is within COMP_RADIUS_H true-box heights of the true box centre;
  * GALLERY ("bank"): TRUE crops from the first GALLERY_S seconds of the clip (at most GALLERY_N); queries are the
    TRUE / COMPETITOR crops of every later frame. A crop's similarity = max cosine to the gallery (bank best_match).
Backends (each L2-normalised, cosine): LAB part-colour (production), DINOv2 ViT-S/14 (production ONNX), COMPOSITE
(production weights LAB 0.35 + DINOv2 0.65), OSNet-AIN x1.0 MSMT17, Youtu ReID (OpenCV Zoo), OSNet x1.0 (a
Market-1501-trained upload, diagnostic only).
Metrics per backend (pooled over clips and per clip), all scale-free so models with different similarity ranges
compare fairly:
  * true-target similarity (median); hardest nearby competitor similarity per frame (median); raw margin (median);
  * RANK-1: share of query frames with >= 1 competitor where TRUE beats the hardest competitor;
  * AUC: probability a TRUE crop scores above a COMPETITOR crop (pooled);
  * SAME-KIT FMR @ TPR90: one global threshold per backend = the 10th percentile of all TRUE similarities (90 % of
    true-target crops pass); FMR = share of SAME-KIT competitor crops at or above it. SAME-KIT = a competitor whose
    LAB similarity to the gallery is >= the 10th percentile of the true target's own LAB similarities (as close in
    colour as the target is to itself): fixed before any ReID result was seen;
  * POST-GAP (recovery after pose / view change): the same metrics on queries in the first POST_GAP_S seconds after
    the target was unavailable or unboxed for >= GAP_S.
"""
import json
import math
import os
import sys

COMP_RADIUS_H = 3.0
GALLERY_S = 1.0
GALLERY_N = 5
GAP_S = 1.0
POST_GAP_S = 1.5
SAME_KIT_PCT = 10
TPR_PCT = 10
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def _inside(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1])
    return ix * iy / area if area > 0 else 0.0


def match_true(obs, hb):
    bodies = [o for o in obs if o.get("duplicate_of") is None]
    best = max(bodies, key=lambda o: _iou(o["bbox"], hb), default=None)
    if best is not None and _iou(best["bbox"], hb) >= 0.5:
        return best
    cont = [o for o in bodies if _inside(o["bbox"], hb) >= 0.9 and (o["bbox"][3] - o["bbox"][1]) >= 0.5 * (hb[3] - hb[1])]
    return max(cont, key=lambda o: o["bbox"][3] - o["bbox"][1]) if cont else None


def competitors(obs, tb):
    h = tb[3] - tb[1]
    cx, cy = (tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2
    out = []
    for o in obs:
        b = o["bbox"]
        if o.get("duplicate_of") is not None or b == tb:
            continue
        if _inside(b, tb) >= 0.75 or _inside(tb, b) >= 0.75:     # a part box of the target, or a box around it
            continue
        if math.hypot((b[0] + b[2]) / 2 - cx, (b[1] + b[3]) / 2 - cy) <= COMP_RADIUS_H * h:
            out.append(o)
    return out


class ReidOnnx:
    def __init__(self, path):
        import onnxruntime as ort
        self.s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.inp = self.s.get_inputs()[0].name

    def __call__(self, crops):
        import numpy as np
        x = np.stack([_prep(c) for c in crops]).astype(np.float32)
        f = self.s.run(None, {self.inp: x})[0].reshape(len(crops), -1)
        return f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-12)


class ReidTorch:
    def __init__(self, module_path, factory, num_classes, weights, extra_safe=False):
        import importlib.util
        import torch
        spec = importlib.util.spec_from_file_location("osnet_mod_" + factory, module_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.torch = torch
        self.m = getattr(mod, factory)(num_classes=num_classes, pretrained=False, loss="softmax")
        safe = []
        if extra_safe:
            import numpy as np
            from numpy._core.multiarray import scalar
            safe = [scalar, np.dtype] + [getattr(np.dtypes, n) for n in dir(np.dtypes) if n.endswith("DType")]
        with torch.serialization.safe_globals(safe):
            d = torch.load(weights, map_location="cpu", weights_only=True)
        sd = d.get("state_dict", d)
        sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
        self.m.load_state_dict(sd, strict=True)
        self.m.eval()

    def __call__(self, crops):
        import numpy as np
        x = self.torch.from_numpy(np.stack([_prep(c) for c in crops]).astype(np.float32))
        with self.torch.no_grad():
            f = self.m(x).numpy().reshape(len(crops), -1)
        return f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-12)


def _prep(crop_bgr):
    import cv2
    import numpy as np
    im = cv2.resize(crop_bgr, (128, 256), interpolation=cv2.INTER_LINEAR)[:, :, ::-1] / 255.0
    im = (im - np.array(MEAN)) / np.array(STD)
    return im.transpose(2, 0, 1)


def _crop(image, b):
    H, W = image.shape[:2]
    x0, y0 = max(0, int(math.floor(b[0]))), max(0, int(math.floor(b[1])))
    x1, y1 = min(W, int(math.ceil(b[2]))), min(H, int(math.ceil(b[3])))
    if x1 - x0 < 4 or y1 - y0 < 8:
        return None
    return image[y0:y1, x0:x1]


def _pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * p / 100.0
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _auc(pos, neg):
    if not pos or not neg:
        return None
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    rank, r = 0.0, 1
    i = 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        avg = (r + r + (j - i) - 1) / 2.0
        rank += avg * sum(1 for k in range(i, j) if allv[k][1] == 1)
        r += j - i
        i = j
    n1, n0 = len(pos), len(neg)
    return (rank - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def main(repo, clips_dir, run_out, models_dir, out_json, cids):
    import numpy as np
    sys.path.insert(0, repo)
    from player_tracker_rb import run_rb as R
    from player_tracker_rb.appearance import AppearanceExtractor, lab_similarity
    from player_tracker_rb.config import TrackerConfig
    from player_tracker_rb.video_io import SequentialVideoReader
    cfg = TrackerConfig.default()
    _det, emb = R.build_backends(cfg)
    ext = AppearanceExtractor(cfg, emb)
    tr_dir = os.path.join(os.path.dirname(__import__("torch").__file__), "..", "torchreid", "reid", "models")
    reid = {
        "OSNET_AIN_MSMT17": ReidTorch(os.path.join(tr_dir, "osnet_ain.py"), "osnet_ain_x1_0", 4101,
                                      os.path.join(models_dir, "osnet_ain_x1_0_msmt17.pth")),
        "YOUTU_REID": ReidOnnx(os.path.join(models_dir, "person_reid_youtu_2021nov.onnx")),
        "OSNET_MARKET_UPLOAD": ReidTorch(os.path.join(tr_dir, "osnet.py"), "osnet_x1_0", 751,
                                         os.path.join(models_dir, "football_osnet_x1_0.pth.tar"), extra_safe=True),
    }
    wl, we = cfg.get("composite_weight_lab"), cfg.get("composite_weight_embedding")
    backends = ["LAB", "DINOV2", "COMPOSITE"] + list(reid)
    rows = []                                   # (clip, rel, kind, post_gap, {backend: sim}, lab_sim)
    for cid in cids:
        with open(os.path.join(repo, "R2", "labels", cid + ".fixture.json")) as f:
            fx = json.load(f)
        if fx["split"] != "DEV":
            raise SystemExit(f"{cid} is not a DEV clip: refused")
        with open(os.path.join(repo, "R2", "labels", fx["labels_file"])) as f:
            lab = json.load(f)
        s = lab["seed"]["frame"]
        logs = {}
        with open(os.path.join(run_out, cid, cid + "_rb_decision_log.jsonl")) as f:
            for line in f:
                r = json.loads(line)
                logs[r["frame"]] = r["population"]["observations"]
        avail = {}
        for k, hb in lab["target_boxes"].items():
            pf = lab["per_frame"][int(k)]
            if pf["visibility"] in ("VISIBLE", "PARTIAL") and pf["resolvable"] == "YES" and not pf["merged"] \
                    and not pf["human_uncertain"]:
                avail[int(k) - s] = hb
        with SequentialVideoReader(os.path.join(clips_dir, fx["video_file"]), fx["seed_frame"], cfg,
                                   expected_sha256=fx["video_sha256"]) as reader:
            fps = reader.info.fps
            gallery = {b: [] for b in backends}
            seg = {"prev_t": None, "start": None, "after_gap": False}

            def handle(rel, image):
                hb = avail.get(rel)
                obs = logs.get(rel)
                if hb is None or obs is None:
                    return
                t = rel / fps
                # segments of availability: a new segment starts after >= GAP_S without an available boxed frame
                if seg["prev_t"] is None:
                    seg.update(start=t, after_gap=False)
                elif t - seg["prev_t"] >= GAP_S:
                    seg.update(start=t, after_gap=True)
                seg["prev_t"] = t
                post_gap = bool(seg["after_gap"] and t - seg["start"] <= POST_GAP_S)
                to = match_true(obs, hb)
                if to is None:
                    return
                comps = competitors(obs, to["bbox"])
                items = [("TRUE", to)] + [("COMP", o) for o in comps]
                crops, feats, keep = [], [], []
                for kind, o in items:
                    c = _crop(image, o["bbox"])
                    if c is None:
                        continue
                    ft = ext.extract(image, o["bbox"], others=[])
                    if not ft.usable:
                        continue
                    crops.append(c)
                    feats.append(ft)
                    keep.append(kind)
                if not keep or keep[0] != "TRUE":
                    return
                vec = {"DINOV2": [f.embedding for f in feats]}
                for name, model in reid.items():
                    vec[name] = list(model(crops))
                in_gallery = t <= GALLERY_S and len(gallery["DINOV2"]) < GALLERY_N
                if in_gallery:
                    gallery["LAB"].append(feats[0].lab)
                    for b in ["DINOV2"] + list(reid):
                        gallery[b].append(np.asarray(vec[b][0], float))
                    return
                if not gallery["DINOV2"]:
                    return
                for j, kind in enumerate(keep):
                    sims = {}
                    labs = max(lab_similarity(g, feats[j].lab) or 0.0 for g in gallery["LAB"])
                    sims["LAB"] = labs
                    for b in ["DINOV2"] + list(reid):
                        v = np.asarray(vec[b][j], float)
                        sims[b] = float(max(float(g @ v) for g in gallery[b]))
                    sims["COMPOSITE"] = wl * sims["LAB"] + we * sims["DINOV2"]
                    rows.append({"clip": cid, "rel": rel, "kind": kind, "post_gap": post_gap, "sims": sims,
                                 "lab": labs})

            handle(0, reader.seed.image)
            for fr in reader:
                handle(fr.rel_frame, fr.image)
        print(cid, "rows", sum(1 for r in rows if r["clip"] == cid), flush=True)

    def summarise(sel):
        out = {}
        true_lab = [r["lab"] for r in sel if r["kind"] == "TRUE"]
        kit_thr = _pct(true_lab, SAME_KIT_PCT)
        all_true_lab = [r["lab"] for r in rows if r["kind"] == "TRUE"]
        kit_thr_global = _pct(all_true_lab, SAME_KIT_PCT)
        for b in backends:
            t_all = [r["sims"][b] for r in rows if r["kind"] == "TRUE"]
            thr = _pct(t_all, TPR_PCT)                        # global per-backend threshold (90 % TPR over all clips)
            tv = [r["sims"][b] for r in sel if r["kind"] == "TRUE"]
            cv = [r["sims"][b] for r in sel if r["kind"] == "COMP"]
            sk = [r["sims"][b] for r in sel if r["kind"] == "COMP" and r["lab"] >= kit_thr_global]
            by_frame = {}
            for r in sel:
                by_frame.setdefault((r["clip"], r["rel"]), []).append(r)
            margins, wins = [], 0
            hard = []
            for _k, rs in by_frame.items():
                tt = [x["sims"][b] for x in rs if x["kind"] == "TRUE"]
                cc = [x["sims"][b] for x in rs if x["kind"] == "COMP"]
                if tt and cc:
                    margins.append(tt[0] - max(cc))
                    hard.append(max(cc))
                    wins += tt[0] > max(cc)
            out[b] = {"n_true": len(tv), "n_comp": len(cv), "n_same_kit": len(sk),
                      "true_median": _pct(tv, 50), "hardest_comp_median": _pct(hard, 50),
                      "margin_median": _pct(margins, 50),
                      "rank1": wins / len(margins) if margins else None, "n_frames_with_comp": len(margins),
                      "auc": _auc(tv, cv),
                      "threshold_tpr90": thr,
                      "tpr_at_thr": (sum(v >= thr for v in tv) / len(tv)) if tv else None,
                      "same_kit_fmr_at_tpr90": (sum(v >= thr for v in sk) / len(sk)) if sk else None,
                      "comp_fmr_at_tpr90": (sum(v >= thr for v in cv) / len(cv)) if cv else None}
        return out

    report = {"params": {"COMP_RADIUS_H": COMP_RADIUS_H, "GALLERY_S": GALLERY_S, "GALLERY_N": GALLERY_N,
                         "GAP_S": GAP_S, "POST_GAP_S": POST_GAP_S, "SAME_KIT_PCT": SAME_KIT_PCT, "TPR_PCT": TPR_PCT},
              "pooled": summarise(rows), "post_gap": summarise([r for r in rows if r["post_gap"]]),
              "per_clip": {c: summarise([r for r in rows if r["clip"] == c]) for c in cids}}
    with open(out_json, "x") as f:
        json.dump(report, f, indent=1)
    with open(out_json + ".rows.jsonl", "x") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    for scope in ("pooled", "post_gap"):
        print("==", scope)
        for b, m in report[scope].items():
            print(f"  {b:20s} true {m['true_median']:.3f} hardest {m['hardest_comp_median'] if m['hardest_comp_median'] is None else round(m['hardest_comp_median'],3)} "
                  f"margin {m['margin_median'] if m['margin_median'] is None else round(m['margin_median'],3)} rank1 {m['rank1'] if m['rank1'] is None else round(m['rank1'],3)} "
                  f"auc {m['auc'] if m['auc'] is None else round(m['auc'],3)} sameKitFMR@TPR90 {m['same_kit_fmr_at_tpr90'] if m['same_kit_fmr_at_tpr90'] is None else round(m['same_kit_fmr_at_tpr90'],3)} "
                  f"(n_true {m['n_true']}, n_comp {m['n_comp']}, n_samekit {m['n_same_kit']})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6:])

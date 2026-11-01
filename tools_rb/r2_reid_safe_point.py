# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2 DEV ReID evaluation instruction (Nathan, 2026-11-01 18:57): prioritise identity separation, not raw
#                 similarity; measure same-kit false-match rate and true-target recovery.
"""Safe operating points from r2_reid_eval rows (read-only; no model, no video).

    python3 tools_rb/r2_reid_safe_point.py ROWS_DIR CLIP_ID [CLIP_ID ...]

For each backend (and the candidate composites LAB 0.35 + REID 0.65 that would replace DINOv2 in the production
weighting), with SAME-KIT competitors defined as in r2_reid_eval (competitor LAB similarity >= the 10th percentile of
the true target's own LAB similarities, over all clips):
  * TPR at same-kit FMR <= 5 % and <= 1 % (one global threshold per backend: the smallest value that keeps the
    same-kit false-match rate at or below the budget): the share of true-target crops a safe threshold still keeps;
  * per clip TPR at that same global threshold (does the gain hold on every clip, not only on average);
  * CONTRASTIVE rank-1 on same-kit frames: share of frames with >= 1 same-kit competitor where the true target beats
    the hardest same-kit competitor (what the tracker's contrast rule needs), with the median margin.
"""
import json
import math
import os
import sys

W_LAB, W_EMB = 0.35, 0.65
SAME_KIT_PCT = 10


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * p / 100.0
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def main(rows_dir, cids):
    rows = []
    for cid in cids:
        with open(os.path.join(rows_dir, cid + ".rows.jsonl")) as f:
            rows += [json.loads(line) for line in f]
    backends = list(rows[0]["sims"])
    reid = [b for b in backends if b not in ("LAB", "DINOV2", "COMPOSITE")]
    for r in rows:
        for b in reid:
            r["sims"]["LAB+" + b] = W_LAB * r["sims"]["LAB"] + W_EMB * r["sims"][b]
    backends += ["LAB+" + b for b in reid]
    kit = pct([r["lab"] for r in rows if r["kind"] == "TRUE"], SAME_KIT_PCT)
    out = {}
    for b in backends:
        tv = [r["sims"][b] for r in rows if r["kind"] == "TRUE"]
        sk = sorted((r["sims"][b] for r in rows if r["kind"] == "COMP" and r["lab"] >= kit), reverse=True)
        res = {"n_true": len(tv), "n_same_kit": len(sk)}
        for budget in (0.05, 0.01):
            k = int(math.floor(budget * len(sk)))               # at most k same-kit crops may pass
            thr = (sk[k] + 1e-12) if k < len(sk) else -1.0      # strictly above the (k+1)-th highest
            res[f"thr_fmr{budget}"] = thr
            res[f"tpr_at_fmr{budget}"] = sum(v >= thr for v in tv) / len(tv)
            res[f"per_clip_tpr_at_fmr{budget}"] = {
                c: (lambda xs: sum(v >= thr for v in xs) / len(xs) if xs else None)(
                    [r["sims"][b] for r in rows if r["kind"] == "TRUE" and r["clip"] == c]) for c in cids}
        frames = {}
        for r in rows:
            frames.setdefault((r["clip"], r["rel"]), []).append(r)
        wins, margins = 0, []
        for rs in frames.values():
            t = [x["sims"][b] for x in rs if x["kind"] == "TRUE"]
            c = [x["sims"][b] for x in rs if x["kind"] == "COMP" and x["lab"] >= kit]
            if t and c:
                margins.append(t[0] - max(c))
                wins += t[0] > max(c)
        res["same_kit_frames"] = len(margins)
        res["same_kit_rank1"] = wins / len(margins) if margins else None
        res["same_kit_margin_median"] = pct(margins, 50)
        out[b] = res
    print(json.dumps({"same_kit_lab_threshold": kit, "backends": out}, indent=1))
    for b, m in out.items():
        print(f"{b:26s} TPR@FMR5% {m['tpr_at_fmr0.05']:.3f}  TPR@FMR1% {m['tpr_at_fmr0.01']:.3f}  "
              f"same-kit rank1 {m['same_kit_rank1']:.3f} (n {m['same_kit_frames']}) margin {m['same_kit_margin_median']:.3f}  "
              f"per-clip TPR@5% {json.dumps({k[-4:]: None if v is None else round(v, 2) for k, v in m['per_clip_tpr_at_fmr0.05'].items()})}",
              file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])

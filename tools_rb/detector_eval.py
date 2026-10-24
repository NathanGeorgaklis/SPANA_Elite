# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R1 approval §3 (detector input size chosen on person-detection quality only); R1/R1_DESIGN_DECISIONS.md
#                 D17, D18; R1-C instruction (GOLD only as a generic real-footage detection fixture)
"""Evaluate person-detector settings on a real clip, using detection quality only.

No identity, tracking or event result is read or produced. From the fixture, only video_sha256, seed_frame and
seed_bbox are read: the seed box is the one human-confirmed person box available, so it is used to check that the
detector can see the person a user would select (the D18 seed check). Everything else is generic statistics over
all person candidates in every frame from the seed frame to the end of the clip.

Settings compared (each in its own subprocess, so runtime and peak memory are its own):
  s_full     YOLOX-s, one full-frame 640 pass                (fixed 640x640 model input)
  s_grid2x2  YOLOX-s, 2x2 overlapping tiles + full-frame pass (more network pixels per small person)
  s_grid3x2  YOLOX-s, 3x2 overlapping tiles + full-frame pass
  m_full     YOLOX-m, one full-frame 640 pass                 (approved fallback model)

Metrics per setting:
  seed_iou / seed_score     best IoU with the seed box at the seed frame, and that candidate's score
  per-frame counts          all candidates; candidates with score >= 0.5
  height bins               candidates by box height (px): <40, 40-80, 80-120, >=120 ("small" = <80)
  pool_recall               PROXY, NOT GROUND TRUTH: the "pool" is every candidate with score >= 0.5 from ANY
                            setting, clustered across settings at IoU >= 0.5. A setting's pool_recall is the share of
                            pool people it also finds (IoU >= 0.5, any kept score). pool_recall_small: the same for
                            pool people shorter than 80 px.
  edge_touch / wide_boxes   partial-at-border candidates; boxes wider than tall*0.9 (merged-pair proxy)
  runtime_ms (mean, p95), peak_rss_mb

Selection rule (declared before running; R1 approval §3: "the smallest setting that preserves required
player-detection quality"). Cost order: s_full < s_grid2x2 < s_grid3x2; m_full only if no YOLOX-s setting passes.
A setting PASSES if: seed_iou >= seed_min_iou (config, 0.5) AND pool_recall >= 0.95 AND pool_recall_small >= 0.90.
The first passing setting in cost order is selected. If none passes, nothing is selected and the report says so.

Usage:
  all settings in one process tree:   python3 tools_rb/detector_eval.py VIDEO FIXTURE_JSON OUT_DIR [M_MODEL M_SHA]
  one setting, one chunk of frames:    python3 tools_rb/detector_eval.py --chunk NAME REL_START REL_COUNT VIDEO
                                           FIXTURE_JSON OUT_DIR [M_MODEL M_SHA]
  summarise existing chunk files:      python3 tools_rb/detector_eval.py --summarise FIXTURE_JSON OUT_DIR
Chunks exist only because a remote shell limits each call to a few minutes. Every chunk still decodes the video
sequentially from frame 0 (no seek, no resume); --summarise refuses to report unless every setting covers every
frame from the seed to the end exactly once.
"""
import json
import os
import resource
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from player_tracker_rb import detector as D  # noqa: E402
from player_tracker_rb import output_schema as S  # noqa: E402
from player_tracker_rb.config import TrackerConfig  # noqa: E402
from player_tracker_rb.video_io import SequentialVideoReader  # noqa: E402

SETTINGS = {
    "s_full": {},
    "s_grid2x2": {"detector_tiling": "grid", "detector_tile_grid": [2, 2]},
    "s_grid3x2": {"detector_tiling": "grid", "detector_tile_grid": [3, 2]},
    "m_full": {"_model": "m"},
}
COST_ORDER = ["s_full", "s_grid2x2", "s_grid3x2"]
FALLBACK = ["m_full"]
POOL_SCORE = 0.5
MATCH_IOU = 0.5
SMALL_PX = 80
PASS_RECALL = 0.95
PASS_RECALL_SMALL = 0.90


def run_setting(name, video, fx, out_dir, m_model=None, m_sha=None, rel_start=0, rel_count=None):
    from player_tracker_rb.detector_yolox import YoloxOnnxDetector
    over = {k: v for k, v in SETTINGS[name].items() if not k.startswith("_")}
    if SETTINGS[name].get("_model") == "m":
        over.update(detector_model_file=m_model, detector_model_sha256=m_sha)
    cfg = TrackerConfig.with_overrides(over)
    det = YoloxOnnxDetector(cfg)
    frames, times = [], []
    rel_end = None if rel_count is None else rel_start + rel_count
    with SequentialVideoReader(video, fx["seed_frame"], cfg, expected_sha256=fx["video_sha256"]) as r:
        for fr in [r.seed] + list(r):          # seed frame (rel 0) and every frame after it, decoded in order
            if fr.rel_frame < rel_start or (rel_end is not None and fr.rel_frame >= rel_end):
                continue
            t = time.perf_counter()
            cands = det.detect(fr.image)
            times.append(time.perf_counter() - t)
            frames.append({"rel": fr.rel_frame, "video_frame_index": fr.video_frame_index,
                           "cands": [[round(v, 2) for v in c.bbox] + [round(c.score, 4), int(c.touches_image_edge)]
                                     for c in cands]})
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = rss_mb / 1024.0 if sys.platform != "darwin" else rss_mb / (1024.0 * 1024.0)
    out = {"setting": name, "overrides": over if "detector_model_sha256" not in over else
           dict(over, detector_model_file=os.path.basename(m_model)), "provenance": det.provenance(),
           "config_hash": cfg.config_hash(), "frames": frames, "frame_times_s": times,
           "peak_rss_mb": rss_mb}
    path = os.path.join(out_dir, f"raw_{name}__{rel_start:04d}.json")
    if os.path.exists(path):
        raise SystemExit(f"refusing to overwrite {path}")
    with open(path, "w") as f:
        f.write(S.dumps_canonical(out))


def load_setting(out_dir, name):
    """Join a setting's chunk files; None if absent. Fails unless frames are contiguous from 0 with no overlap."""
    files = sorted(f for f in os.listdir(out_dir) if f.startswith(f"raw_{name}__") and f.endswith(".json"))
    if not files:
        return None
    parts = [json.load(open(os.path.join(out_dir, f))) for f in files]
    frames = [fr for p in parts for fr in p["frames"]]
    rels = [fr["rel"] for fr in frames]
    if rels != list(range(len(rels))):
        raise SystemExit(f"{name}: chunks do not cover frames 0..N-1 exactly once: {rels[:5]}...{rels[-5:]}")
    if len({p["config_hash"] for p in parts}) != 1:
        raise SystemExit(f"{name}: chunks were run with different configs")
    times = [t for p in parts for t in p["frame_times_s"]]
    return {"setting": name, "overrides": parts[0]["overrides"], "provenance": parts[0]["provenance"],
            "config_hash": parts[0]["config_hash"], "frames": frames, "chunks": files,
            "runtime_ms_mean": 1000 * float(np.mean(times)), "runtime_ms_p95": 1000 * float(np.percentile(times, 95)),
            "peak_rss_mb": max(p["peak_rss_mb"] for p in parts)}


def summarise(out_dir, fx):
    raw = {n: load_setting(out_dir, n) for n in SETTINGS}
    raw = {n: v for n, v in raw.items() if v is not None}
    names = list(raw)
    n_frames = len(raw[names[0]]["frames"])
    if any(len(raw[n]["frames"]) != n_frames for n in names):
        raise SystemExit("settings cover different numbers of frames")
    seed_box = np.array([fx["seed_bbox"]], float)
    rep = {}
    # Build the agreement pool frame by frame.
    pool_hits = {n: 0 for n in names}
    pool_hits_small = {n: 0 for n in names}
    pool_total = pool_small_total = 0
    for i in range(n_frames):
        per = {n: np.array(raw[n]["frames"][i]["cands"], float).reshape(-1, 6) for n in names}
        strong = np.concatenate([per[n][per[n][:, 4] >= POOL_SCORE, :5] for n in names], 0)
        if strong.shape[0]:
            keep = D.nms(strong[:, :4], strong[:, 4], MATCH_IOU)      # cluster across settings
            pool = strong[keep, :4]
            for p in pool:
                small = (p[3] - p[1]) < SMALL_PX
                pool_total += 1
                pool_small_total += int(small)
                for n in names:
                    if per[n].shape[0] and D.box_iou_matrix(p[None], per[n][:, :4]).max() >= MATCH_IOU:
                        pool_hits[n] += 1
                        pool_hits_small[n] += int(small)
    for n in names:
        fr = raw[n]["frames"]
        allc = [np.array(f["cands"], float).reshape(-1, 6) for f in fr]
        cat = np.concatenate(allc, 0) if allc else np.zeros((0, 6))
        h = cat[:, 3] - cat[:, 1] if cat.shape[0] else np.zeros(0)
        w = cat[:, 2] - cat[:, 0] if cat.shape[0] else np.zeros(0)
        seed_c = allc[0]
        seed_iou = float(D.box_iou_matrix(seed_box, seed_c[:, :4]).max()) if seed_c.shape[0] else 0.0
        seed_score = float(seed_c[int(D.box_iou_matrix(seed_box, seed_c[:, :4]).argmax()), 4]) if seed_c.shape[0] else 0.0
        rep[n] = {
            "provenance": raw[n]["provenance"], "config_hash": raw[n]["config_hash"],
            "seed_iou": round(seed_iou, 4), "seed_score": round(seed_score, 4),
            "frames": n_frames,
            "candidates_per_frame_mean": round(float(np.mean([a.shape[0] for a in allc])), 2),
            "strong_per_frame_mean": round(float(np.mean([(a[:, 4] >= POOL_SCORE).sum() for a in allc])), 2),
            "height_bins": {"lt40": int((h < 40).sum()), "40_80": int(((h >= 40) & (h < 80)).sum()),
                            "80_120": int(((h >= 80) & (h < 120)).sum()), "ge120": int((h >= 120).sum())},
            "small_lt80_per_frame_mean": round(float((h < SMALL_PX).sum()) / n_frames, 2),
            "edge_touch_total": int(cat[:, 5].sum()) if cat.shape[0] else 0,
            "wide_boxes_total": int((w > 0.9 * h).sum()) if cat.shape[0] else 0,
            "pool_recall": round(pool_hits[n] / pool_total, 4) if pool_total else None,
            "pool_recall_small": round(pool_hits_small[n] / pool_small_total, 4) if pool_small_total else None,
            "runtime_ms_mean": round(raw[n]["runtime_ms_mean"], 1), "runtime_ms_p95": round(raw[n]["runtime_ms_p95"], 1),
            "peak_rss_mb": round(raw[n]["peak_rss_mb"], 1),
        }
    seed_min = TrackerConfig.default().get("seed_min_iou")
    for n in names:
        r = rep[n]
        r["passes"] = bool(r["seed_iou"] >= seed_min and (r["pool_recall"] or 0) >= PASS_RECALL
                           and (r["pool_recall_small"] or 0) >= PASS_RECALL_SMALL)
    selected = next((n for n in COST_ORDER + FALLBACK if n in rep and rep[n]["passes"]), None)
    return {"tool": "tools_rb/detector_eval.py", "fixture_fields_used": ["video_sha256", "seed_frame", "seed_bbox"],
            "identity_or_event_results_used": False, "frames_evaluated": n_frames,
            "pool": {"definition": f"score>={POOL_SCORE} in any setting, clustered at IoU>={MATCH_IOU}; PROXY, not ground truth",
                     "people_frames": pool_total, "small_lt80_people_frames": pool_small_total},
            "selection_rule": {"cost_order": COST_ORDER, "fallback": FALLBACK, "seed_min_iou": seed_min,
                               "min_pool_recall": PASS_RECALL, "min_pool_recall_small": PASS_RECALL_SMALL},
            "settings": rep, "selected": selected,
            "chunks": {n: raw[n]["chunks"] for n in names}}


def _fx(fx_json):
    return {k: json.load(open(fx_json))[k] for k in ("video_sha256", "seed_frame", "seed_bbox")}


def _write_summary(fx_json, out_dir):
    rep = summarise(out_dir, _fx(fx_json))
    with open(os.path.join(out_dir, "DETECTOR_EVAL_SUMMARY.json"), "w") as f:
        f.write(S.dumps_canonical(rep))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("settings", "chunks")}, indent=1))
    for n, r in rep["settings"].items():
        print(n, {k: v for k, v in r.items() if k not in ("provenance", "config_hash")})


def main(argv):
    if argv[0] == "--chunk":
        _, name, start, count, video, fx_json, out_dir, *m = argv
        os.makedirs(out_dir, exist_ok=True)
        run_setting(name, video, _fx(fx_json), out_dir, *(m or [None, None]), rel_start=int(start),
                    rel_count=int(count))
        return 0
    if argv[0] == "--summarise":
        _write_summary(argv[1], argv[2])
        return 0
    video, fx_json, out_dir, *m = argv
    os.makedirs(out_dir, exist_ok=True)
    for name in SETTINGS:
        if SETTINGS[name].get("_model") == "m" and not m:
            continue
        subprocess.run([sys.executable, os.path.abspath(__file__), "--chunk", name, "0", "1000000", video, fx_json,
                        out_dir, *m], check=True)
    _write_summary(fx_json, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

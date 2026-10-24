# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R1-F instruction (Nathan, 2026-10-25: GOLD only as development footage after synthetic tests pass;
#                 no event evaluation); R1/R1C/detector_eval (saved person candidates); R1/gold_label_sheet (frozen
#                 human tracking labels v1, identity/visibility only)
"""Target identity behaviour on a development clip (no events).

Two steps, so a run fits within a short process limit:
  embed  VIDEO FIXTURE_JSON DETECTOR_EVAL_DIR SETTING CACHE_DIR FROM_REL TO_REL
         DINOv2 embeddings (pinned model) for every saved candidate box crop of frames [FROM_REL, TO_REL), keyed
         by the SHA-1 of the exact crop bytes and shape. Written to CACHE_DIR/emb_<from>_<to>.npz.
  run    VIDEO FIXTURE_JSON DETECTOR_EVAL_DIR SETTING CACHE_DIR LABELS_CSV OUT_JSON
         Camera + population + target tracker from the seed with a cached embedding backend (a cache miss stops
         the run: fail closed). The tracker sees only images, candidates and the seed box.

From the fixture only video_sha256, seed_frame and seed_bbox are read (the seed is a tracker input; no event
field is read). The frozen human labels are used only AFTER the run, to classify the tracker's states per labelled
frame (visibility / identity resolvability / merged-or-hidden); they never influence the tracker. Reported:
  * state counts and transitions, reacquisition events, uncertainty reasons, decision logs for the label window;
  * per label class: frames where the tracker claimed TRACKING (a claim on a frame labelled hidden or
    not-resolvable contradicts the human truth and must be zero), and coverage on frames labelled visible;
  * a contact sheet of the label window (target box coloured by state) so a human can check which person each
    TRACKING box is on.
"""
import csv
import hashlib
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from player_tracker_rb import population as P  # noqa: E402
from player_tracker_rb import target as T  # noqa: E402
from player_tracker_rb.appearance import AppearanceExtractor, EmbeddingBackend, crop_image  # noqa: E402
from player_tracker_rb.camera import CameraMotionEstimator  # noqa: E402
from player_tracker_rb.config import TrackerConfig  # noqa: E402
from player_tracker_rb.detector import PersonCandidate  # noqa: E402
from player_tracker_rb.video_io import SequentialVideoReader  # noqa: E402
from tools_rb.population_eval import candidate_files, file_sha256, load_candidates  # noqa: E402

FIXTURE_FIELDS = ("video_sha256", "seed_frame", "seed_bbox")
SHEET_FROM_BEFORE_LABELS = 10
SHEET_STEP = 2
SHEET_COLS = 4
SHEET_TILE_W = 480
SHEET_JPEG_QUALITY = 80
STATE_COLOUR = {"TRACKING": (0, 255, 0), "OCCLUDED": (0, 165, 255), "AMBIGUOUS": (0, 255, 255),
                "LOST": (0, 0, 255), "REACQUIRING": (255, 0, 255)}


def crop_key(crop):
    h = hashlib.sha1()
    h.update(str(crop.shape).encode())
    h.update(np.ascontiguousarray(crop).tobytes())
    return h.hexdigest()


class CachedEmbedding(EmbeddingBackend):
    def __init__(self, table, provenance):
        self.table = table
        self._prov = provenance

    def embed(self, crop):
        k = crop_key(crop)
        if k not in self.table:
            raise SystemExit(f"embedding cache miss for a crop of shape {crop.shape} (fail closed)")
        return self.table[k]

    def provenance(self):
        return dict(self._prov, cache=True)


def read_fixture(path):
    d = json.load(open(path))
    return {k: d[k] for k in FIXTURE_FIELDS}


def embed(video, fx_json, det_dir, setting, cache_dir, a, b):
    from player_tracker_rb.embedding_dinov2 import Dinov2OnnxEmbedding
    fx = read_fixture(fx_json)
    cfg = TrackerConfig.default()
    emb = Dinov2OnnxEmbedding(cfg)
    cands = load_candidates(det_dir, setting)
    keys, vecs = [], []
    t0 = time.time()
    with SequentialVideoReader(video, fx["seed_frame"], cfg, expected_sha256=fx["video_sha256"]) as reader:
        for fr in [reader.seed] + list(reader):
            if fr.rel_frame < a:
                continue
            if fr.rel_frame >= b:
                break
            for row in cands[fr.rel_frame]:
                crop = crop_image(fr.image, row[:4])
                if crop is None:
                    continue
                k = crop_key(crop)
                if k in keys:
                    continue
                keys.append(k)
                vecs.append(emb.embed(crop))
    os.makedirs(cache_dir, exist_ok=True)
    out = os.path.join(cache_dir, f"emb_{a:04d}_{b:04d}.npz")
    np.savez(out, keys=np.asarray(keys), vecs=np.asarray(vecs), provenance=json.dumps(emb.provenance()))
    print(json.dumps({"frames": [a, b], "crops": len(keys), "seconds": round(time.time() - t0, 1), "out": out}))


def load_cache(cache_dir):
    table, prov, files = {}, None, {}
    for f in sorted(os.listdir(cache_dir)):
        if f.startswith("emb_") and f.endswith(".npz"):
            z = np.load(os.path.join(cache_dir, f))
            for k, v in zip(z["keys"], z["vecs"]):
                table[str(k)] = v
            prov = json.loads(str(z["provenance"]))
            files[f] = file_sha256(os.path.join(cache_dir, f))
    if not table:
        raise SystemExit("empty embedding cache (fail closed)")
    return table, prov, files


def read_labels(path, seed_frame):
    rows = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows[int(r["frame"]) - seed_frame] = r
    return rows


def label_class(r):
    """Generic per-frame class from the human label columns (identity/visibility only)."""
    vis = r["t23_visibility"].strip().lower()
    res = r["t23_identity_resolvable"].strip().lower()
    mh = r["merged_or_hidden"].strip().lower()
    if res == "no" or vis == "hidden":
        return "HIDDEN_OR_NOT_RESOLVABLE"
    if mh in ("merged", "t23_hidden"):
        return "MERGED_WITH_OPPONENT"
    if vis == "partial":
        return "PARTIAL"
    return "VISIBLE"


def draw(img, tf, res):
    out = img.copy()
    for o in res.observations:
        x1, y1, x2, y2 = (int(round(v)) for v in o.bbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), (160, 160, 160), 1)
    col = STATE_COLOUR[tf.rb_state]
    x1, y1, x2, y2 = (int(round(v)) for v in tf.bbox)
    cv2.rectangle(out, (x1, y1), (x2, y2), col, 3 if tf.rb_state == "TRACKING" else 1)
    if tf.observed_or_contested_box is not None and tf.rb_state != "TRACKING":
        x1, y1, x2, y2 = (int(round(v)) for v in tf.observed_or_contested_box)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
    txt = f"rel {tf.rel_frame} {tf.rb_state} ep{tf.target_episode} c{tf.identity_confidence_rb:.2f}"
    cv2.putText(out, txt, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, ",".join(tf.identity_uncertainty_reasons)[:60], (8, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)
    return out


def run(video, fx_json, det_dir, setting, cache_dir, labels_csv, out_json):
    fx = read_fixture(fx_json)
    cfg = TrackerConfig.default()
    table, prov, cache_files = load_cache(cache_dir)
    cands = load_candidates(det_dir, setting)
    labels = read_labels(labels_csv, fx["seed_frame"])
    lo, hi = min(labels), max(labels)
    frames, sheet, timings = [], [], []
    with SequentialVideoReader(video, fx["seed_frame"], cfg, expected_sha256=fx["video_sha256"]) as reader:
        fps = reader.info.fps
        w, h = reader.info.frame_width, reader.info.frame_height
        est = CameraMotionEstimator(cfg)
        pop = P.Population(cfg, w, h)
        ext = AppearanceExtractor(cfg, CachedEmbedding(table, prov))
        tt = T.TargetTracker(cfg, pop, ext)
        for fr in [reader.seed] + list(reader):
            r_ = fr.rel_frame
            if r_ not in cands:
                raise SystemExit(f"no saved candidates for rel frame {r_} (fail closed)")
            dets = [PersonCandidate(tuple(float(v) for v in row[:4]), float(row[4]), bool(row[5]), "saved")
                    for row in cands[r_]]
            t0 = time.perf_counter()
            cam = est.update(fr.image, [d.bbox for d in dets], 1.0 / fps)
            res = pop.update(r_, r_ / fps, dets, cam)
            tf = tt.seed(res, fr.image, fx["seed_bbox"]) if r_ == 0 else tt.update(res, fr.image)
            timings.append(time.perf_counter() - t0)
            frames.append(tf)
            if lo - SHEET_FROM_BEFORE_LABELS <= r_ <= hi and (r_ - lo) % SHEET_STEP == 0:
                sheet.append(draw(fr.image, tf, res))
    states = {}
    for f in frames:
        states[f.rb_state] = states.get(f.rb_state, 0) + 1
    transitions = [{"rel": f.rel_frame, "state": f.rb_state, "reasons": list(f.identity_uncertainty_reasons),
                    "episode": f.target_episode} for k, f in enumerate(frames)
                   if k == 0 or f.rb_state != frames[k - 1].rb_state
                   or f.identity_uncertainty_reasons != frames[k - 1].identity_uncertainty_reasons]
    by_class = {}
    for rel, row in sorted(labels.items()):
        cl = label_class(row)
        d = by_class.setdefault(cl, {"frames": 0, "tracking": 0, "states": {}, "tracking_rel": []})
        d["frames"] += 1
        st = frames[rel].rb_state
        d["states"][st] = d["states"].get(st, 0) + 1
        if st == "TRACKING":
            d["tracking"] += 1
            d["tracking_rel"].append(rel)
    window = [frames[rel].as_record() for rel in sorted(labels)]
    report = {
        "tool": "tools_rb/target_eval.py", "fixture_fields_used": list(FIXTURE_FIELDS),
        "events_evaluated": False, "labels_used_by_tracker": False,
        "labels_file_sha256": file_sha256(labels_csv), "video_sha256": fx["video_sha256"],
        "candidate_files_sha256": {f: file_sha256(os.path.join(det_dir, f)) for f in candidate_files(det_dir, setting)},
        "embedding_cache_files_sha256": cache_files, "embedding_provenance": prov,
        "config_registry_version": cfg.as_record()["registry_version"], "config_hash": cfg.config_hash(),
        "frames": len(frames), "fps": fps, "state_counts": states, "transitions": transitions,
        "reacquisition_events": [e.as_record() for e in tt.reacquisition_events],
        "target_bank_samples": len(tt.bank), "target_tracklets": list(tt.target_tracklets),
        "label_window_rel": [lo, hi], "by_label_class": by_class,
        "runtime_ms_per_frame_median": float(np.median(timings) * 1000.0),
        "label_window_frames": window,
    }
    with open(out_json, "w") as f:
        json.dump(report, f, indent=1, sort_keys=True, default=str)
    if sheet:
        tiles = [cv2.resize(im, (SHEET_TILE_W, int(round(im.shape[0] * SHEET_TILE_W / im.shape[1]))),
                            interpolation=cv2.INTER_AREA) for im in sheet]
        while len(tiles) % SHEET_COLS:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.hstack(tiles[k:k + SHEET_COLS]) for k in range(0, len(tiles), SHEET_COLS)]
        cv2.imwrite(os.path.splitext(out_json)[0] + "_contact_sheet.jpg", np.vstack(rows),
                    [cv2.IMWRITE_JPEG_QUALITY, SHEET_JPEG_QUALITY])
    print(json.dumps({k: report[k] for k in ("frames", "state_counts", "by_label_class", "reacquisition_events",
                                             "target_bank_samples", "runtime_ms_per_frame_median")},
                     indent=1, default=str)[:6000])
    print(json.dumps(transitions, indent=0)[:4000])


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "embed" and len(sys.argv) == 9:
        embed(*sys.argv[2:7], int(sys.argv[7]), int(sys.argv[8]))
    elif len(sys.argv) >= 2 and sys.argv[1] == "run" and len(sys.argv) == 9:
        run(*sys.argv[2:9])
    else:
        raise SystemExit(__doc__)

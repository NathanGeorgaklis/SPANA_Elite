# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R1-E instruction (Nathan, 2026-10-24: population + motion, no identity logic); R1/R1C/detector_eval
#                 (saved person candidates); R1/R1_GENERIC_TRACKER_REQUIREMENT.md (GOLD is development footage only)
"""Population + camera-motion behaviour on a development clip (no identity, no labels, no events).

Runs CameraMotionEstimator and Population over a clip, frame by frame, with the person candidates saved by the R1-C
detector evaluation (default setting s_grid2x2), so no detection is re-run. From the fixture only video_sha256 and
seed_frame are read (to open the same frames). No human label, tracker identity, target seed box or event is used, and
nothing here says which tracklet is which person.

Reported (behaviour, not accuracy; there is no ground truth here):
  * camera: status counts, continuity breaks, image shift and zoom per frame, rejected estimates;
  * population: tracklets started, lifetimes (s), closures by reason, contact events by kind, merged detections by
    reason, duplicates, partial detections, hidden-tracklet frames by source (PREDICTED / HELD), bank-eligible share
    of observed tracklet boxes, provisional share, competitors per observation;
  * invariant checks on every frame (observed vs inferred, one detection one tracklet, no merged box tracked);
  * runtime per frame.
A contact sheet (every frame with a contact event, up to SHEET_FRAMES) is written next to the report so a human can
check that MERGED and CONTACT decisions are made where people really touch.

Usage: python3 tools_rb/population_eval.py VIDEO FIXTURE_JSON DETECTOR_EVAL_DIR SETTING OUT_JSON
"""
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from player_tracker_rb import population as P  # noqa: E402
from player_tracker_rb.camera import CameraMotionEstimator  # noqa: E402
from player_tracker_rb.config import TrackerConfig  # noqa: E402
from player_tracker_rb.detector import PersonCandidate  # noqa: E402
from player_tracker_rb.motion import DETECTION  # noqa: E402
from player_tracker_rb.video_io import SequentialVideoReader  # noqa: E402

SHEET_FRAMES = 12
SHEET_TILE_W = 640
SHEET_JPEG_QUALITY = 80
CODE_FILES = ("player_tracker_rb/population.py", "player_tracker_rb/motion.py", "player_tracker_rb/camera.py",
              "player_tracker_rb/config.py", "tools_rb/population_eval.py")
COMPETITOR_PROBE_H = "median observed box height of the frame"


def code_hashes():
    import hashlib
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = {}
    for rel in CODE_FILES:
        with open(os.path.join(root, rel), "rb") as f:
            out[rel] = hashlib.sha256(f.read()).hexdigest()
    return out


def load_candidates(det_dir, setting):
    files = sorted(f for f in os.listdir(det_dir) if f.startswith(f"raw_{setting}__") and f.endswith(".json"))
    frames = [fr for f in files for fr in json.load(open(os.path.join(det_dir, f)))["frames"]]
    if [f["rel"] for f in frames] != list(range(len(frames))):
        raise SystemExit("detector candidates do not cover frames 0..N-1 exactly once")
    return {f["rel"]: np.array(f["cands"], float).reshape(-1, 6) for f in frames}


def check_invariants(r):
    errs = []
    seen = {}
    for v in r.tracklets:
        if v.observed:
            if v.observed_box is None or v.detection_index is None or v.observation_source != DETECTION:
                errs.append(f"{v.tracklet_id}: observed without a detection")
            elif P.MERGED_SUSPECTED in r.observations[v.detection_index].flags:
                errs.append(f"{v.tracklet_id}: tracked on a merged detection")
            if v.detection_index in seen:
                errs.append(f"D{v.detection_index}: two tracklets")
            seen[v.detection_index] = v.tracklet_id
        elif v.observed_box is not None or v.bank_eligible:
            errs.append(f"{v.tracklet_id}: inferred box treated as observed")
    return errs


def draw(img, r):
    out = img.copy()
    for o in r.observations:
        x1, y1, x2, y2 = (int(round(v)) for v in o.bbox)
        if not o.is_body:
            col = (128, 128, 128)
        elif P.MERGED_SUSPECTED in o.flags:
            col = (0, 0, 255)
        elif P.CONTESTED in o.flags:
            col = (0, 165, 255)
        else:
            col = (0, 255, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        label = o.tracklet_id or ("MERGED" if P.MERGED_SUSPECTED in o.flags else o.ref)
        cv2.putText(out, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    for v in r.tracklets:
        if not v.observed:
            x1, y1, x2, y2 = (int(round(c)) for c in v.box)
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 0), 1)
            cv2.putText(out, v.tracklet_id + "?", (x1, y2 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    cv2.putText(out, f"rel {r.rel_frame}  camera {r.camera.status}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    return out


def stats(x):
    if not x:
        return None
    a = np.asarray(x, float)
    return {"n": int(a.size), "min": float(a.min()), "p10": float(np.percentile(a, 10)),
            "median": float(np.median(a)), "p90": float(np.percentile(a, 90)), "max": float(a.max())}


def count(d, k):
    d[k] = d.get(k, 0) + 1


def main(video, fx_json, det_dir, setting, out_json):
    fxall = json.load(open(fx_json))
    fx = {k: fxall[k] for k in ("video_sha256", "seed_frame")}
    cfg = TrackerConfig.default()
    cands = load_candidates(det_dir, setting)
    cam_status, cam_reasons, closures, events, merge_reasons, flags = {}, {}, {}, {}, {}, {}
    hidden_src, shifts, scales, n_comp, timings, inv_errors = {}, [], [], [], [], []
    births, first_seen, last_seen = 0, {}, {}
    n_obs_tracked = n_bank_eligible = n_provisional = n_dup = 0
    sheet = []
    with SequentialVideoReader(video, fx["seed_frame"], cfg, expected_sha256=fx["video_sha256"]) as reader:
        fps = reader.info.fps
        width, height = reader.info.frame_width, reader.info.frame_height
        est = CameraMotionEstimator(cfg)
        pop = P.Population(cfg, width, height)
        for fr in [reader.seed] + list(reader):
            r_ = fr.rel_frame
            c = cands.get(r_, np.zeros((0, 6)))
            dets = [PersonCandidate(tuple(float(v) for v in row[:4]), float(row[4]), bool(row[5]), "saved")
                    for row in c]
            t0 = time.perf_counter()
            cam = est.update(fr.image, [d.bbox for d in dets], 1.0 / fps)
            res = pop.update(r_, r_ / fps, dets, cam)
            timings.append(time.perf_counter() - t0)
            count(cam_status, cam.status)
            for x in cam.reasons:
                count(cam_reasons, x)
            if cam.compensated:
                shifts.append(float(np.hypot(*cam.shift_px)) * fps / width)
                scales.append(float(cam.scale) ** fps)
            for cl in res.closed:
                count(closures, cl.reason)
            for e in res.contact_events:
                count(events, e.kind)
            for o in res.observations:
                for f in o.flags:
                    count(flags, f)
                for m in o.merge_reasons:
                    count(merge_reasons, m)
                if not o.is_body:
                    n_dup += 1
            hs = [o.bbox[3] - o.bbox[1] for o in res.observations if o.is_body]
            for v in res.tracklets:
                first_seen.setdefault(v.tracklet_id, r_)
                last_seen[v.tracklet_id] = r_
                if v.status == P.NEW:
                    births += 1
                if v.observed:
                    n_obs_tracked += 1
                    n_bank_eligible += int(v.bank_eligible)
                    n_provisional += int(v.provisional)
                    if hs:
                        n_comp.append(len(pop.competitors(v.box, float(np.median(hs)),
                                                          exclude_tracklet_ids=[v.tracklet_id])))
                else:
                    count(hidden_src, v.observation_source)
            inv_errors += [f"rel {r_}: {e}" for e in check_invariants(res)]
            if res.contact_events and len(sheet) < SHEET_FRAMES:
                sheet.append(draw(fr.image, res))
    lifetimes = [(last_seen[t] - first_seen[t] + 1) / fps for t in first_seen]
    report = {
        "tool": "tools_rb/population_eval.py", "fixture_fields_used": sorted(fx), "detector_setting": setting,
        "labels_identity_or_event_results_used": False, "fps": fps, "frames": len(timings),
        "config_registry_version": cfg.as_record()["registry_version"] if hasattr(cfg, "as_record") else None,
        "config_hash": cfg.config_hash() if hasattr(cfg, "config_hash") else None,
        "code_sha256": code_hashes(),
        "camera": {"status_counts": cam_status, "reason_counts": cam_reasons,
                   "shift_frac_width_per_s": stats(shifts), "zoom_factor_per_s": stats(scales)},
        "population": {"tracklets_started": births, "tracklets_seen": len(first_seen),
                       "tracklet_lifetime_s": stats(lifetimes), "closures_by_reason": closures,
                       "contact_events_by_kind": events, "observation_flags": flags,
                       "merge_reasons": merge_reasons, "duplicates_suppressed": n_dup,
                       "hidden_frames_by_source": hidden_src,
                       "observed_tracklet_boxes": n_obs_tracked,
                       "bank_eligible_share": (n_bank_eligible / n_obs_tracked) if n_obs_tracked else None,
                       "provisional_share": (n_provisional / n_obs_tracked) if n_obs_tracked else None,
                       "competitors_per_observed_tracklet": stats(n_comp),
                       "competitor_probe_height": COMPETITOR_PROBE_H},
        "invariant_errors": inv_errors,
        "runtime_ms_per_frame": stats([t * 1000.0 for t in timings]),
    }
    with open(out_json, "w") as f:
        json.dump(report, f, indent=1, sort_keys=True)
    if sheet:
        tiles = [cv2.resize(im, (SHEET_TILE_W, int(round(im.shape[0] * SHEET_TILE_W / im.shape[1]))),
                            interpolation=cv2.INTER_AREA) for im in sheet]
        while len(tiles) % 3:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.hstack(tiles[k:k + 3]) for k in range(0, len(tiles), 3)]
        cv2.imwrite(os.path.splitext(out_json)[0] + "_contact_sheet.jpg", np.vstack(rows),
                    [cv2.IMWRITE_JPEG_QUALITY, SHEET_JPEG_QUALITY])
    print(json.dumps({k: report[k] for k in ("frames", "camera", "invariant_errors")}, indent=1)[:4000])
    print(json.dumps(report["population"], indent=1)[:4000])
    return report


if __name__ == "__main__":
    if len(sys.argv) != 6:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])

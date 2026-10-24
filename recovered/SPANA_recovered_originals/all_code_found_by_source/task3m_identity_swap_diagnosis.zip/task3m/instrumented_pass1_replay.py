"""Task 3M -- READ-ONLY instrumented replay of Pass 1 (player_tracker_mo) for GOLD_DUEL_01.

No tracker source file is edited. Runtime wrappers record the inputs and outputs of the
association functions that track_manager.step() calls, plus raw detector output (with
confidences) and the identity-bank admission decisions. The replay loop copies run_mo.run()
(seed frame 180, same seed box, default config, no jersey hint). It writes no debug video and
no checkpoint, because neither feeds back into tracking.

Determinism check: the replay's debug_log must match the scored Pass-1 frame_log for every
replayed frame. verify_replay.py checks this.

Diagnostic embeddings come from a SEPARATE AppearanceEncoder instance (no OCR), so the tracker's
own encoder sees exactly the same call sequence as in the scored run.

usage: python3 -m event_engine_v1.task3m.instrumented_pass1_replay VIDEO OUT_DIR MAX_FRAMES LOG_FROM
"""
import json
import os
import sys

import cv2
import numpy as np

import football_analysis.possession_detector as _pd
# The scored Pass-1 run (Task 3K) predates Task 3L. player_tracker_mo's team classifier imports
# football_analysis.possession_detector.TeamClassifier, so replay with the 3L flag OFF to reproduce
# the tracker exactly as it ran when the scored JSON was produced.
_pd.TASK3L_OTHER_PLAYER_CONTEXT = os.environ.get('TASK3M_3L_FLAG', '0') == '1'   # default OFF = scored-run reproduction
import player_tracker_mo.track_manager as tm
from player_tracker_mo.track_manager import MultiPlayerTrackManager, TrackManagerConfig
from player_tracker.appearance import AppearanceEncoder
from player_tracker.geometry import center

SEED_FRAME = 180
SEED_BOX = [635.3, 304.1, 757.8, 438.3]

VIDEO, OUT_DIR, MAX_FRAMES, LOG_FROM = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
os.makedirs(OUT_DIR, exist_ok=True)

rec = {}            # per-frame capture for the current step
records = []
emb_store = {}      # key "f{frame}_d{j}" -> diag embedding; "f{frame}_ref_{tid}" -> representative emb
diag_enc = AppearanceEncoder(use_ocr=False)
mgr = None


def r3(x):
    return None if x is None else round(float(x), 3)


def track_snapshot(t):
    return {
        "track_id": t.track_id, "state": str(t.state),
        "pred_center": [r3(v) for v in t.motion.position],
        "gate_radius": r3(getattr(t, "_gate_radius", None)),
        "frames_since_confirmed": t.identity.frames_since_confirmed,
        "last_confirmed_center": [r3(v) for v in t.identity.last_confirmed_center] if t.identity.last_confirmed_center is not None else None,
        "avg_size": [r3(v) for v in t.identity.average_size()],
        "current_box": [r3(v) for v in t.current_box] if t.current_box is not None else None,
        "bank_n": t.identity_bank_v2.n_trusted_samples if t.identity_bank_v2 is not None else None,
    }


# ---- detector wrapper: identical predict() call, also records confidences
_orig_detect = None


def detect_wrapper(frame):
    d = mgr.detector
    res = d.model.predict(frame, classes=[0], conf=d.conf, imgsz=d.imgsz, verbose=False)[0]
    boxes = [tuple(b) for b in res.boxes.xyxy.tolist()]
    if rec.get("log"):
        rec["detections"] = [{"j": j, "box": [r3(v) for v in b], "det_conf": r3(c)}
                             for j, (b, c) in enumerate(zip(boxes, res.boxes.conf.tolist()))]
        rec["_frame"] = frame
    return boxes


# ---- association wrappers
_orig_gate = tm.compute_identity_gate_matrix
_orig_build = tm.build_score_matrix
_orig_assign = tm.assign
_orig_overlap = tm.detect_overlap_contests


def gate_wrapper(tracks, detections, *a, **k):
    g = _orig_gate(tracks, detections, *a, **k)
    if rec.get("log"):
        rec["gate_matrix"] = {tracks[i].track_id: [bool(x) for x in row] for i, row in enumerate(g)}
        rec["spot_check_frame"] = (mgr.cfg.spot_check_interval > 0 and mgr.frame_idx % mgr.cfg.spot_check_interval == 0)
    return g


def build_wrapper(tracks, detections, *a, **k):
    scores, scored = _orig_build(tracks, detections, *a, **k)
    if rec.get("log"):
        f = mgr.frame_idx
        rec["scorable"] = [track_snapshot(t) for t in tracks]
        rec["all_live"] = [track_snapshot(t) for t in mgr.tracks.values() if str(t.state) != str(tm.TrackState.TERMINATED)]
        rec["score_matrix"] = {tracks[i].track_id: [sc.as_log_dict() for sc in row] for i, row in enumerate(scored)}
        rec["tracker_embedding_present"] = [d.get("embedding") is not None for d in detections]
        # representative embeddings (pre-update) for every live track, and diag embeddings for all dets
        for t in mgr.tracks.values():
            ref = mgr._representative_embedding(t)
            if ref is not None:
                emb_store[f"f{f}_ref_{t.track_id}"] = np.asarray(ref, dtype=np.float64)
        frame = rec["_frame"]
        for j, d in enumerate(detections):
            e = diag_enc.embed(frame, d["box"])
            if e is not None:
                emb_store[f"f{f}_d{j}"] = e
            if d.get("embedding") is not None:
                emb_store[f"f{f}_trk_d{j}"] = np.asarray(d["embedding"], dtype=np.float64)
    return scores, scored


def assign_wrapper(tracks, detections, scores):
    res = _orig_assign(tracks, detections, scores)
    if rec.get("log"):
        rec["assignments"] = [{"track_id": a.track_id, "det": a.detection_index, "score": r3(a.score),
                               "ambiguous": a.ambiguous, "contested_with": a.contested_with,
                               "reject_reason": a.reject_reason} for a in res]
    return res


def overlap_wrapper(tracks, *a, **k):
    out = _orig_overlap(tracks, *a, **k)
    if rec.get("log"):
        rec["overlap_contests_scorable"] = [list(p) for p in out]
        # counterfactual diagnostic only (not fed back): same test over ALL live non-dormant tracks
        live_nd = [t for t in mgr.tracks.values()
                   if t.state not in tm.TrackState.DORMANT and t.state != tm.TrackState.TERMINATED]
        rec["overlap_contests_incl_provisional_DIAG"] = [list(p) for p in _orig_overlap(live_nd, *a, **k)]
    return out


tm.compute_identity_gate_matrix = gate_wrapper
tm.build_score_matrix = build_wrapper
tm.assign = assign_wrapper
tm.detect_overlap_contests = overlap_wrapper

# ---- bank admission wrapper (per track, installed lazily)
import player_tracker_mo.identity_bank_mo as ibm
_orig_consider = ibm.IdentityBankV2.consider


def consider_wrapper(self, embedding, quality, appearance_sim, is_ambiguous_this_frame):
    ok = _orig_consider(self, embedding, quality, appearance_sim, is_ambiguous_this_frame)
    if rec.get("log"):
        owner = getattr(self, "_diag_owner", None)
        rec.setdefault("bank_consider", []).append({"track_id": owner, "quality": r3(quality),
                                                    "appearance_sim": r3(appearance_sim), "admitted": ok})
    return ok


ibm.IdentityBankV2.consider = consider_wrapper

# ---- replay loop (mirrors run_mo.run without debug video / checkpointing)
mgr = MultiPlayerTrackManager(config=TrackManagerConfig())
mgr.detector.detect_persons = detect_wrapper
cap = cv2.VideoCapture(VIDEO)
cap.set(cv2.CAP_PROP_POS_FRAMES, SEED_FRAME)
ok, frame0 = cap.read()
mgr.initialize(frame0, SEED_BOX, jersey_number=None)
n = 0
while n < MAX_FRAMES:
    ok, frame = cap.read()
    if not ok:
        break
    n += 1
    for t in mgr.tracks.values():
        if t.identity_bank_v2 is not None:
            t.identity_bank_v2._diag_owner = t.track_id
    rec.clear()
    rec["log"] = n >= LOG_FROM
    snap = mgr.step(frame)
    if rec["log"]:
        rec.pop("_frame", None)
        rec.pop("log", None)
        rec["frame"] = n
        rec["extract_frame"] = SEED_FRAME + n
        rec["post_step_tracks"] = [track_snapshot(t) for t in mgr.tracks.values()
                                   if str(t.state) != str(tm.TrackState.TERMINATED)]
        records.append(dict(rec))
    if n % 10 == 0:
        print(f"frame {n}", flush=True)

cap.release()
with open(os.path.join(OUT_DIR, "replay_debug_log.json"), "w") as f:
    json.dump(mgr.debug_log, f, default=float)
with open(os.path.join(OUT_DIR, "association_capture.json"), "w") as f:
    json.dump(records, f, default=float)
np.savez_compressed(os.path.join(OUT_DIR, "embeddings.npz"), **emb_store)
print("done", n)

"""
touch_detector.py -- MILESTONE 2 TASK 1: ball<->player touch detection.

Consumes ONLY the frozen Milestone 1 outputs: per-frame player-track records
(observed_position_box/position_source/position_confidence/
frames_since_observed_position/team/state -- never current_box) from
clip_mo_tracking_results.json's frame_log, and a fresh BallTracker pass via
the existing, unmodified attribute_actors.run_ball_pass(). Nothing in
player_tracker_mo/, attribute_actors.py, release_gate.py, or
football_analysis/ball_tracker.py is modified or re-implemented here.

See MILESTONE_2_TASK1_TOUCH_SPEC.md for the full design rationale. This
module is pure, stateless, read-only with respect to all frozen code --
it only reads clip_mo_tracking_results.json and calls run_ball_pass().
"""
import json
import math
import sys
from pathlib import Path

MILESTONE2_DIR = Path(__file__).resolve().parent
MILESTONE1_DIR = MILESTONE2_DIR.parent / "milestone1"
sys.path.insert(0, str(MILESTONE1_DIR))

from attribute_actors import run_ball_pass  # noqa: E402  (frozen, reused, not modified)

# ---------------------------------------------------------------------------
# Tunable constants -- all named here, none buried inline (Section 16: keep
# component scores visible for debugging).
INTERACTION_RADIUS_BOX_HEIGHTS = 0.45     # Section 8
TEMPORAL_WINDOW_S = 0.33                  # Section 9, same scale as Stage A's FRESH_MAX_STALENESS_S
MIN_SUSTAINED_FRAMES = 2                  # Section 9
PREDICTED_GRACE_FRAMES = 3                # Section 12, ~0.1s @30fps
COMPETITION_MARGIN = 0.15                 # Section 13
DEDUP_GAP_FRAMES = 5                      # Section 18, ~0.17s @30fps

BALL_STATE_WEIGHT = {"CONFIRMED": 1.0, "PREDICTED": 0.5, "REACQUIRING": 0.3, "OCCLUDED": 0.0, "LOST": 0.0}

W_BALL_QUALITY = 0.30
W_PLAYER_QUALITY = 0.30
W_PROXIMITY = 0.20
W_TEMPORAL = 0.15
W_TRAJECTORY = 0.05

CONFIRMED_THRESH = 0.75
PROBABLE_THRESH = 0.5
AMBIGUOUS_WEAK_THRESH = 0.3


# ---------------------------------------------------------------------------
# Geometry (Section 7-8)

def feet_box(box):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return (x1 - 0.15 * w, y1 + 0.65 * h, x2 + 0.15 * w, y2 + 0.10 * h)


def point_to_box_dist(px, py, box):
    x1, y1, x2, y2 = box
    dx = max(x1 - px, 0.0, px - x2)
    dy = max(y1 - py, 0.0, py - y2)
    return math.hypot(dx, dy)


def normalized_distance(ball_xy, track_box):
    x1, y1, x2, y2 = track_box
    h = max(1e-6, y2 - y1)
    fb = feet_box(track_box)
    d = point_to_box_dist(ball_xy[0], ball_xy[1], fb)
    return d / h


# ---------------------------------------------------------------------------
# Player position quality (Section 12)

def player_position_quality(track_rec):
    src = track_rec.get("position_source")
    conf = track_rec.get("position_confidence") or 0.0
    fsop = track_rec.get("frames_since_observed_position", 0)
    if src == "CONFIRMED_DETECTION":
        return 1.0 * conf
    if src == "AMBIGUOUS_DETECTION" and fsop == 0:
        return 0.6 * conf
    if src == "WEAK_SPATIAL_RECOVERY":
        return 0.5 * conf
    if src == "PREDICTED" and fsop <= PREDICTED_GRACE_FRAMES:
        return 0.3 * conf
    return 0.0


# ---------------------------------------------------------------------------
# Ball quality

def ball_quality(ball_rec):
    if ball_rec is None:
        return 0.0, None
    state = ball_rec.get("state")
    weight = BALL_STATE_WEIGHT.get(state, 0.0)
    conf = ball_rec.get("confidence") or 0.0
    return weight * conf, state


# ---------------------------------------------------------------------------
# Trajectory change (Section 15) -- optional booster from ball velocity
# history around a candidate frame.

def trajectory_change_score(ball_by_frame, frame, lookback=3, lookahead=3):
    before = [ball_by_frame[f] for f in range(frame - lookback, frame)
              if f in ball_by_frame and ball_by_frame[f].get("state") in ("CONFIRMED", "PREDICTED")]
    after = [ball_by_frame[f] for f in range(frame + 1, frame + 1 + lookahead)
             if f in ball_by_frame and ball_by_frame[f].get("state") in ("CONFIRMED", "PREDICTED")]
    if not before or not after:
        return 0.0
    vxb = sum(b["velocity_x"] for b in before) / len(before)
    vyb = sum(b["velocity_y"] for b in before) / len(before)
    vxa = sum(a["velocity_x"] for a in after) / len(after)
    vya = sum(a["velocity_y"] for a in after) / len(after)
    speed_b = math.hypot(vxb, vyb)
    speed_a = math.hypot(vxa, vya)
    speed_change = abs(speed_a - speed_b) / max(1.0, speed_b, speed_a)
    if speed_b > 1e-3 and speed_a > 1e-3:
        cos_angle = (vxb * vxa + vyb * vya) / (speed_b * speed_a)
        cos_angle = max(-1.0, min(1.0, cos_angle))
        direction_change = math.acos(cos_angle) / math.pi  # 0..1
    else:
        direction_change = 0.0
    return max(0.0, min(1.0, 0.5 * speed_change + 0.5 * direction_change))


# ---------------------------------------------------------------------------
# Candidate generation (Section 6, 9)

def generate_frame_candidates(frame_log_by_frame, ball_by_frame, fps):
    """Returns {frame: [{"track_id", "normalized_distance", "ball_quality",
    "player_quality", "ball_state"}]} for every frame with >=1 physically
    plausible candidate (within INTERACTION_RADIUS_BOX_HEIGHTS, nonzero
    ball/player quality). No temporal or competition logic yet -- raw
    per-frame geometry only."""
    out = {}
    for frame, rec in frame_log_by_frame.items():
        ball_rec = ball_by_frame.get(frame)
        bq, bstate = ball_quality(ball_rec)
        if bq <= 0.0 or ball_rec.get("centre_x") is None:
            continue
        ball_xy = (ball_rec["centre_x"], ball_rec["centre_y"])
        cands = []
        for tr in rec.get("tracks", []):
            box = tr.get("observed_position_box")
            if not box:
                continue
            pq = player_position_quality(tr)
            if pq <= 0.0:
                continue
            nd = normalized_distance(ball_xy, box)
            if nd <= INTERACTION_RADIUS_BOX_HEIGHTS:
                cands.append({"track_id": tr["track_id"], "normalized_distance": nd,
                              "ball_quality": bq, "player_quality": pq, "ball_state": bstate})
        if cands:
            out[frame] = cands
    return out


def temporal_proximity_score(track_id, frame, frame_candidates, fps):
    window = max(1, round(TEMPORAL_WINDOW_S * fps))
    this_nd = None
    for c in frame_candidates.get(frame, []):
        if c["track_id"] == track_id:
            this_nd = c["normalized_distance"]
    if this_nd is None:
        return 0.0
    neighborhood = []
    for f in range(frame - window, frame + window + 1):
        for c in frame_candidates.get(f, []):
            if c["track_id"] == track_id:
                neighborhood.append((f, c["normalized_distance"]))
    neighborhood.sort()
    is_local_min = all(this_nd <= nd + 1e-9 for _, nd in neighborhood)
    sustained = sum(1 for f, nd in neighborhood if abs(f - frame) <= window)
    if is_local_min and sustained >= MIN_SUSTAINED_FRAMES:
        return 1.0
    if sustained >= MIN_SUSTAINED_FRAMES:
        return 0.6
    return 0.3


def score_candidates(frame_candidates, ball_by_frame, fps):
    """Attach full component scores + touch_confidence + competition_penalty
    to every (frame, track) candidate. Returns {frame: [{...full record...}]}."""
    scored = {}
    for frame, cands in frame_candidates.items():
        rows = []
        for c in cands:
            tp = temporal_proximity_score(c["track_id"], frame, frame_candidates, fps)
            tc = trajectory_change_score(ball_by_frame, frame)
            prox = max(0.0, 1.0 - c["normalized_distance"] / INTERACTION_RADIUS_BOX_HEIGHTS)
            raw = (W_BALL_QUALITY * c["ball_quality"] + W_PLAYER_QUALITY * c["player_quality"]
                   + W_PROXIMITY * prox + W_TEMPORAL * tp + W_TRAJECTORY * tc)
            rows.append({**c, "normalized_proximity": prox, "temporal_proximity": tp,
                         "trajectory_change": tc, "raw_score": raw})
        rows.sort(key=lambda r: -r["raw_score"])
        competition_penalty = 0.0
        if len(rows) >= 2 and (rows[0]["raw_score"] - rows[1]["raw_score"]) < COMPETITION_MARGIN:
            competition_penalty = 0.35
        for r in rows:
            r["touch_confidence"] = round(r["raw_score"] * (1 - competition_penalty), 4)
            r["competition_penalty"] = competition_penalty
        scored[frame] = rows
    return scored


# Section 9: one-frame proximity is NOT enough for a confirmed touch, no
# matter how strong the other components are. This is a hard gate, not just
# a weighted component -- temporal_proximity must show at least sustained
# (>=2-frame) evidence before CONFIRMED_TOUCH is reachable at this frame.
MIN_TEMPORAL_FOR_CONFIRMED = 0.6


def classify_frame(rows):
    """rows: scored candidates for one frame, sorted best-first. Returns
    (touch_state, leader_or_None, candidate_list_for_output)."""
    if not rows:
        return "NO_TOUCH", None, []
    leader = rows[0]
    penalty = leader["competition_penalty"]
    if penalty > 0:
        return "AMBIGUOUS_TOUCH", None, rows
    conf = leader["touch_confidence"]
    if conf >= CONFIRMED_THRESH:
        if leader["temporal_proximity"] >= MIN_TEMPORAL_FOR_CONFIRMED:
            return "CONFIRMED_TOUCH", leader, rows
        return "PROBABLE_TOUCH", leader, rows  # strong single-frame evidence, but isolated -- capped
    if conf >= PROBABLE_THRESH:
        return "PROBABLE_TOUCH", leader, rows
    if conf >= AMBIGUOUS_WEAK_THRESH:
        return "AMBIGUOUS_TOUCH", leader, rows
    return "NO_TOUCH", None, rows


# ---------------------------------------------------------------------------
# Temporal deduplication (Section 18-19)

def cluster_events(scored_candidates, fps, play_id, source_note):
    """Walk frames in order, build clusters of consecutive (gap <=
    DEDUP_GAP_FRAMES) frames that share the same leading track_id (or are
    all AMBIGUOUS with overlapping candidate sets), emit one touch event per
    cluster."""
    frames_sorted = sorted(scored_candidates.keys())
    events = []
    cluster = []  # list of (frame, touch_state, leader, rows)
    cluster_key = None  # track_id of leader, or "__AMBIGUOUS__"

    def flush():
        nonlocal cluster
        if not cluster:
            return
        states = [c[1] for c in cluster]
        # pick representative frame = peak confidence among non-NO_TOUCH entries
        scored_entries = [c for c in cluster if c[1] != "NO_TOUCH"]
        if not scored_entries:
            cluster = []
            return
        if cluster_key == "__AMBIGUOUS__":
            peak = max(scored_entries, key=lambda c: (c[3][0]["touch_confidence"] if c[3] else 0))
            touch_frame, touch_state, leader, rows = peak
            candidate_tracks = sorted({(r["track_id"], round(r["touch_confidence"], 2))
                                        for _, _, _, rws in cluster for r in (rws or [])[:3]},
                                       key=lambda x: -x[1])
            events.append({
                "touch_id": f"TOUCH_{len(events)+1:03d}", "play_id": play_id,
                "start_frame": cluster[0][0], "end_frame": cluster[-1][0],
                "touch_frame": touch_frame, "timestamp": round(touch_frame / fps, 3),
                "actor_track_id": None,
                "candidate_tracks": [{"track_id": t, "touch_confidence": s} for t, s in candidate_tracks],
                "touch_state": "AMBIGUOUS_TOUCH", "touch_confidence": candidate_tracks[0][1] if candidate_tracks else 0.0,
                "ball_confidence": round(rows[0]["ball_quality"], 3) if rows else None,
                "player_position_confidence": None,
                "normalized_contact_distance": round(min(r["normalized_distance"] for r in rows), 3) if rows else None,
                "trajectory_change_score": round(rows[0]["trajectory_change"], 3) if rows else None,
                "review_required": True,
                "provenance": {"fps": fps, "source": source_note, "n_candidate_frames": len(cluster)},
            })
        else:
            peak = max(scored_entries, key=lambda c: c[2]["touch_confidence"] if c[2] else 0)
            touch_frame, touch_state, leader, rows = peak
            events.append({
                "touch_id": f"TOUCH_{len(events)+1:03d}", "play_id": play_id,
                "start_frame": cluster[0][0], "end_frame": cluster[-1][0],
                "touch_frame": touch_frame, "timestamp": round(touch_frame / fps, 3),
                "actor_track_id": cluster_key,
                "candidate_tracks": [{"track_id": cluster_key, "touch_confidence": leader["touch_confidence"]}],
                "touch_state": touch_state, "touch_confidence": leader["touch_confidence"],
                "ball_confidence": round(leader["ball_quality"], 3),
                "player_position_confidence": round(leader["player_quality"], 3),
                "normalized_contact_distance": round(leader["normalized_distance"], 3),
                "trajectory_change_score": round(leader["trajectory_change"], 3),
                "review_required": touch_state != "CONFIRMED_TOUCH",
                "provenance": {"fps": fps, "source": source_note, "n_candidate_frames": len(cluster),
                               "component_scores": {
                                   "ball_quality": round(leader["ball_quality"], 3),
                                   "player_position_quality": round(leader["player_quality"], 3),
                                   "normalized_proximity": round(leader["normalized_proximity"], 3),
                                   "temporal_proximity": round(leader["temporal_proximity"], 3),
                                   "trajectory_change": round(leader["trajectory_change"], 3),
                                   "competition_penalty": leader["competition_penalty"],
                               }},
            })
        cluster = []

    prev_frame = None
    for frame in frames_sorted:
        rows = scored_candidates[frame]
        touch_state, leader, out_rows = classify_frame(rows)
        if touch_state == "NO_TOUCH":
            continue
        this_key = "__AMBIGUOUS__" if leader is None else leader["track_id"]
        if cluster and (this_key != cluster_key or (prev_frame is not None and frame - prev_frame > DEDUP_GAP_FRAMES)):
            flush()
        cluster_key = this_key
        cluster.append((frame, touch_state, leader, out_rows))
        prev_frame = frame
    flush()
    return events


# ---------------------------------------------------------------------------
# Top-level entry point

def detect_touches(clip_path, tracking_result_dir, play_id, max_frames=None, ball_history=None):
    """ball_history: optional pre-computed list of per-frame ball dicts
    (frame->{centre_x, centre_y, state, confidence, velocity_x, velocity_y})
    to avoid re-running the expensive ball pass when called repeatedly in
    tests/regression; production path (None) runs it fresh via the frozen
    run_ball_pass()."""
    with open(f"{tracking_result_dir}/clip_mo_tracking_results.json") as f:
        tracking = json.load(f)
    frame_log_by_frame = {rec["frame"]: rec for rec in tracking["frame_log"]}
    fps = tracking.get("fps", 29.97)
    n_frames = max(frame_log_by_frame.keys()) + 1 if frame_log_by_frame else 0
    if max_frames:
        n_frames = min(n_frames, max_frames)

    if ball_history is None:
        tracker, ball_states = run_ball_pass(clip_path, n_frames, fps)
        ball_by_frame = {}
        for f, bs in ball_states.items():
            ball_by_frame[f] = {"centre_x": bs.centre_x, "centre_y": bs.centre_y, "state": bs.state,
                                 "confidence": bs.confidence, "velocity_x": bs.velocity_x, "velocity_y": bs.velocity_y}
        source_note = f"{clip_path} + {tracking_result_dir}/clip_mo_tracking_results.json (fresh ball pass)"
    else:
        ball_by_frame = ball_history
        source_note = f"{tracking_result_dir}/clip_mo_tracking_results.json (ball history supplied)"

    frame_candidates = generate_frame_candidates(frame_log_by_frame, ball_by_frame, fps)
    scored = score_candidates(frame_candidates, ball_by_frame, fps)
    events = cluster_events(scored, fps, play_id, source_note)
    return events, {"fps": fps, "n_frames": n_frames, "n_candidate_frames": len(frame_candidates)}


if __name__ == "__main__":
    clip_path, result_dir, play_id, out_path = sys.argv[1:5]
    events, meta = detect_touches(clip_path, result_dir, play_id)
    with open(out_path, "w") as f:
        json.dump({"play_id": play_id, "meta": meta, "events": events}, f, indent=2, default=str)
    print(json.dumps({"play_id": play_id, "meta": meta, "n_events": len(events),
                       "events_summary": [{"touch_id": e["touch_id"], "touch_frame": e["touch_frame"],
                                            "actor_track_id": e["actor_track_id"], "touch_state": e["touch_state"],
                                            "touch_confidence": e["touch_confidence"]} for e in events]},
                      indent=2, default=str))

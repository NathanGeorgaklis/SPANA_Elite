"""
test_touch_detector.py -- MILESTONE 2 TASK 1 regression tests.

Pure synthetic inputs -- no video, no YOLO, no BallTracker, no tracking
pipeline. Exercises touch_detector.py's pure functions directly against the
shapes clip_mo_tracking_results.json's frame_log and a ball-state dict
already provide. Run with: python3 -m pytest milestone2/test_touch_detector.py -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from touch_detector import (  # noqa: E402
    normalized_distance, player_position_quality, ball_quality,
    generate_frame_candidates, score_candidates, classify_frame, cluster_events,
    trajectory_change_score,
)

FPS = 29.97


def track(tid, box, src, pos_conf, fsop=0):
    return {"track_id": tid, "observed_position_box": list(box), "position_source": src,
            "position_confidence": pos_conf, "frames_since_observed_position": fsop}


def ball(cx, cy, state, conf, vx=0.0, vy=0.0):
    return {"centre_x": cx, "centre_y": cy, "state": state, "confidence": conf,
            "velocity_x": vx, "velocity_y": vy}


# --- A. scale-aware distance ------------------------------------------------

def test_A_scale_aware_distance():
    """A 20px-tall and a 150px-tall player at the same absolute pixel offset
    from the ball must NOT get the same normalized distance -- the smaller
    player's normalized distance must be much larger (farther, in
    body-relative terms) for the same raw pixel gap."""
    small_box = (100, 100, 110, 120)   # 20px tall
    big_box = (100, 100, 160, 250)     # 150px tall
    ball_xy = (130, 160)  # same absolute point for both
    nd_small = normalized_distance(ball_xy, small_box)
    nd_big = normalized_distance(ball_xy, big_box)
    ok = nd_small > nd_big
    assert ok, f"nd_small={nd_small} nd_big={nd_big}"


# --- B. player position quality gating -------------------------------------

def test_B_confirmed_detection_full_quality():
    t = track("T001", (0, 0, 10, 10), "CONFIRMED_DETECTION", 1.0)
    q = player_position_quality(t)
    assert q == 1.0, q


def test_C_stale_predicted_excluded():
    t = track("T002", (0, 0, 10, 10), "PREDICTED", 0.3, fsop=50)
    q = player_position_quality(t)
    assert q == 0.0, q


def test_D_brief_predicted_partial_quality():
    t = track("T003", (0, 0, 10, 10), "PREDICTED", 0.3, fsop=1)
    q = player_position_quality(t)
    assert 0.0 < q < 0.5, q


def test_E_fresh_ambiguous_partial_quality():
    t = track("T004", (0, 0, 10, 10), "AMBIGUOUS_DETECTION", 0.53, fsop=0)
    q = player_position_quality(t)
    assert 0.0 < q < 1.0, q


# --- F. ball quality gating --------------------------------------------------

def test_F_lost_ball_excluded():
    q, state = ball_quality(ball(100, 100, "LOST", 0.0))
    assert q == 0.0, q


def test_G_confirmed_ball_full_weight():
    q, state = ball_quality(ball(100, 100, "CONFIRMED", 0.9))
    assert abs(q - 0.9) < 1e-6, q


# --- H. temporal evidence required (Section 9: one frame is not enough) ----

def test_H_isolated_single_frame_proximity_is_weak_not_confirmed():
    """A track that is near the ball for exactly ONE frame, with no
    sustained approach, must not reach CONFIRMED_TOUCH."""
    frame_log = {
        10: {"tracks": [track("T010", (95, 95, 105, 145), "CONFIRMED_DETECTION", 1.0)]},
    }
    ball_by_frame = {10: ball(100, 140, "CONFIRMED", 1.0)}
    cands = generate_frame_candidates(frame_log, ball_by_frame, FPS)
    scored = score_candidates(cands, ball_by_frame, FPS)
    state, leader, rows = classify_frame(scored[10])
    ok = state != "CONFIRMED_TOUCH"
    assert ok, f"state={state} conf={leader['touch_confidence'] if leader else None}"


def test_I_sustained_approach_reaches_confirmed():
    """A track with the ball approaching, staying close for several frames
    (local minimum + sustained), with CONFIRMED ball and CONFIRMED player
    detection throughout, should reach CONFIRMED_TOUCH."""
    box = (90, 90, 110, 150)  # 60px tall
    frame_log = {}
    ball_by_frame = {}
    # ball approaches from frame 8 to 12, closest at 10-11, leaves by 14
    xs = {8: 60, 9: 75, 10: 100, 11: 102, 12: 115, 13: 140, 14: 170}
    for f, x in xs.items():
        frame_log[f] = {"tracks": [track("T020", box, "CONFIRMED_DETECTION", 1.0)]}
        ball_by_frame[f] = ball(x, 145, "CONFIRMED", 1.0, vx=15.0)
    cands = generate_frame_candidates(frame_log, ball_by_frame, FPS)
    scored = score_candidates(cands, ball_by_frame, FPS)
    # frame 10 or 11 should be near the local minimum
    best_frame = min((f for f in scored if scored[f]), key=lambda f: scored[f][0]["normalized_distance"])
    state, leader, rows = classify_frame(scored[best_frame])
    assert state == "CONFIRMED_TOUCH", f"frame={best_frame} state={state} rows={scored[best_frame]}"


# --- J. nearest-player-alone is forbidden / competition handling -----------

def test_J_two_equally_close_tracks_produce_ambiguous_not_forced_winner():
    """Two tracks with near-identical evidence contesting the same ball
    position must NOT produce a single confident winner -- this is the
    explicit 'nearest player = touch' prohibition (Section 14)."""
    box_a = (90, 90, 110, 150)
    box_b = (92, 90, 112, 150)
    frame_log = {10: {"tracks": [
        track("T020", box_a, "CONFIRMED_DETECTION", 1.0),
        track("T026", box_b, "CONFIRMED_DETECTION", 1.0),
    ]}}
    ball_by_frame = {10: ball(100, 145, "CONFIRMED", 1.0)}
    cands = generate_frame_candidates(frame_log, ball_by_frame, FPS)
    scored = score_candidates(cands, ball_by_frame, FPS)
    state, leader, rows = classify_frame(scored[10])
    ok = state == "AMBIGUOUS_TOUCH" and leader is None
    assert ok, f"state={state} leader={leader}"


def test_K_clearly_closer_track_wins_over_farther_one():
    """Sanity check in the other direction: when one track is clearly
    better-positioned than another (not a near-tie), it should win outright,
    not be forced into AMBIGUOUS."""
    box_close = (95, 95, 105, 145)
    box_far = (300, 90, 320, 150)
    frame_log = {10: {"tracks": [
        track("T020", box_close, "CONFIRMED_DETECTION", 1.0),
        track("T099", box_far, "CONFIRMED_DETECTION", 1.0),
    ]}}
    ball_by_frame = {10: ball(100, 140, "CONFIRMED", 1.0)}
    cands = generate_frame_candidates(frame_log, ball_by_frame, FPS)
    scored = score_candidates(cands, ball_by_frame, FPS)
    assert 10 in scored
    rows = scored[10]
    track_ids = [r["track_id"] for r in rows]
    assert track_ids[0] == "T020", track_ids
    # the far track shouldn't even be a candidate at normalized distance this large
    assert "T099" not in track_ids, track_ids


# --- L. stale track must not out-compete a fresh farther one ---------------

def test_L_stale_predicted_track_does_not_win_over_fresh_one():
    stale_box = (95, 95, 105, 145)   # geometrically closer
    fresh_box = (130, 90, 150, 150)  # farther but fresh
    frame_log = {10: {"tracks": [
        track("T_STALE", stale_box, "PREDICTED", 0.2, fsop=50),
        track("T_FRESH", fresh_box, "CONFIRMED_DETECTION", 1.0),
    ]}}
    ball_by_frame = {10: ball(100, 140, "CONFIRMED", 1.0)}
    cands = generate_frame_candidates(frame_log, ball_by_frame, FPS)
    scored = score_candidates(cands, ball_by_frame, FPS)
    track_ids = [r["track_id"] for r in scored.get(10, [])]
    assert "T_STALE" not in track_ids, track_ids


# --- M. temporal deduplication ----------------------------------------------

def test_M_adjacent_frames_merge_into_one_event():
    box = (90, 90, 110, 150)
    frame_log, ball_by_frame = {}, {}
    for f, x in {196: 95, 197: 98, 198: 100, 199: 103}.items():
        frame_log[f] = {"tracks": [track("T015", box, "CONFIRMED_DETECTION", 1.0)]}
        ball_by_frame[f] = ball(x, 145, "CONFIRMED", 1.0)
    cands = generate_frame_candidates(frame_log, ball_by_frame, FPS)
    scored = score_candidates(cands, ball_by_frame, FPS)
    events = cluster_events(scored, FPS, "TEST_PLAY", "synthetic")
    assert len(events) == 1, f"expected 1 merged event, got {len(events)}: {events}"
    assert events[0]["start_frame"] == 196 and events[0]["end_frame"] == 199


def test_N_distant_frames_do_not_merge():
    box = (90, 90, 110, 150)
    frame_log, ball_by_frame = {}, {}
    for f, x in {50: 100, 250: 100}.items():
        frame_log[f] = {"tracks": [track("T015", box, "CONFIRMED_DETECTION", 1.0)]}
        ball_by_frame[f] = ball(x, 145, "CONFIRMED", 1.0)
    cands = generate_frame_candidates(frame_log, ball_by_frame, FPS)
    scored = score_candidates(cands, ball_by_frame, FPS)
    events = cluster_events(scored, FPS, "TEST_PLAY", "synthetic")
    assert len(events) == 2, f"expected 2 separate events, got {len(events)}: {events}"


if __name__ == "__main__":
    import inspect
    fns = [obj for name, obj in list(globals().items()) if name.startswith("test_") and inspect.isfunction(obj)]
    n_pass = 0
    for fn in fns:
        try:
            fn()
            print(f"[PASS] {fn.__name__}")
            n_pass += 1
        except AssertionError as e:
            print(f"[FAIL] {fn.__name__} -- {e}")
    print(f"\n{n_pass}/{len(fns)} touch-detector regression tests passed")
    if n_pass != len(fns):
        sys.exit(1)

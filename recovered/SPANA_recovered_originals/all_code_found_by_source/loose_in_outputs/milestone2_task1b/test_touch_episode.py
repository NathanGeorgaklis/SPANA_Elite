"""
test_touch_episode.py -- MILESTONE 2 TASK 1B regression tests (Section 28).

Pure synthetic inputs built directly against touch_detector's own frozen
pipeline (generate_frame_candidates/score_candidates/cluster_events), then
fed into touch_episode.build_episodes. No video, no ball tracker.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from touch_detector import generate_frame_candidates, score_candidates, cluster_events  # noqa: E402
from touch_episode import build_episodes  # noqa: E402

FPS = 29.97


def track(tid, box, src="CONFIRMED_DETECTION", conf=1.0, fsop=0):
    return {"track_id": tid, "observed_position_box": list(box), "position_source": src,
            "position_confidence": conf, "frames_since_observed_position": fsop}


def ball(cx, cy, state="CONFIRMED", conf=1.0, vx=0.0, vy=0.0):
    return {"centre_x": cx, "centre_y": cy, "state": state, "confidence": conf,
            "velocity_x": vx, "velocity_y": vy}


def run(frame_log, ball_by_frame):
    fc = generate_frame_candidates(frame_log, ball_by_frame, FPS)
    scored = score_candidates(fc, ball_by_frame, FPS)
    events = cluster_events(scored, FPS, "TEST", "synthetic")
    episodes = build_episodes(events, scored, FPS, "TEST", "synthetic")
    return events, episodes


BOX_A = (90, 90, 110, 150)
BOX_B = (115, 90, 135, 150)


# A. stable single actor across many frames -> SINGLE_ACTOR_TOUCH
def test_A_stable_single_actor():
    frame_log, bbf = {}, {}
    for f, x in {10: 95, 11: 98, 12: 100, 13: 103, 14: 106}.items():
        frame_log[f] = {"tracks": [track("T020", BOX_A)]}
        bbf[f] = ball(x, 145, vx=5.0)
    events, episodes = run(frame_log, bbf)
    assert len(episodes) == 1, episodes
    assert episodes[0]["episode_state"] == "SINGLE_ACTOR_TOUCH", episodes[0]
    assert episodes[0]["primary_actor_track_id"] == "T020"


# B. two actors alternate every frame with overlapping candidate sets ->
#    AMBIGUOUS_CONTESTED_TOUCH
def test_B_alternating_contest_stays_ambiguous():
    frame_log, bbf = {}, {}
    # ball position itself alternates which box it is closer to, so the
    # per-frame geometric leader genuinely flips, not just a confidence knob
    xs = {10: 100, 11: 125, 12: 100, 13: 125, 14: 100, 15: 125, 16: 100}
    for f, x in xs.items():
        frame_log[f] = {"tracks": [track("T001", BOX_A), track("T002", BOX_B)]}
        bbf[f] = ball(x, 145)
    events, episodes = run(frame_log, bbf)
    states = {e["episode_state"] for e in episodes}
    assert "AMBIGUOUS_CONTESTED_TOUCH" in states, episodes
    # no episode should confidently claim a single actor across the whole
    # alternating stretch
    confirmed_single = [e for e in episodes if e["episode_state"] == "SINGLE_ACTOR_TOUCH"]
    assert not confirmed_single, episodes


# C. A contacts, ball clearly travels, then B contacts -> kept as two
#    genuinely separate touches (no shared candidate set -> SEQUENTIAL link)
def test_C_sequential_touches_preserved():
    frame_log, bbf = {}, {}
    box_far = (400, 90, 420, 150)
    for f, x in {10: 95, 11: 98, 12: 100}.items():
        frame_log[f] = {"tracks": [track("T010", BOX_A)]}
        bbf[f] = ball(x, 145, vx=5.0)
    for f, x in {15: 395, 16: 400, 17: 405}.items():
        frame_log[f] = {"tracks": [track("T099", box_far)]}
        bbf[f] = ball(x, 145, vx=5.0)
    events, episodes = run(frame_log, bbf)
    assert len(events) == 2, events  # Task 1's own raw detector keeps them separate
    states = [e["episode_state"] for e in episodes]
    actors = [e["primary_actor_track_id"] for e in episodes if e["primary_actor_track_id"]]
    # Section 9: do not collapse into one ambiguous blob; both actors must
    # still be individually recoverable (either as two episodes, or as one
    # SEQUENTIAL_TOUCHES container listing both).
    assert "AMBIGUOUS_CONTESTED_TOUCH" not in states, episodes
    assert {"T010", "T099"} <= (set(actors) | set().union(*[set(e["candidate_tracks"]) for e in episodes])), episodes


# D. one dominant actor + weak occasional competitor -> SINGLE_ACTOR_TOUCH
def test_D_dominant_with_weak_competitor():
    frame_log, bbf = {}, {}
    for f, x in {10: 95, 11: 98, 12: 100, 13: 103}.items():
        tracks = [track("T020", BOX_A, conf=1.0)]
        if f == 11:
            tracks.append(track("T021", (300, 90, 320, 150), conf=0.3))  # far, barely eligible
        frame_log[f] = {"tracks": tracks}
        bbf[f] = ball(x, 145, vx=5.0)
    events, episodes = run(frame_log, bbf)
    single = [e for e in episodes if e["episode_state"] == "SINGLE_ACTOR_TOUCH" and e["primary_actor_track_id"] == "T020"]
    assert single, episodes


# E. three-player crowded contest with no dominant actor ->
#    AMBIGUOUS_CONTESTED_TOUCH
def test_E_three_way_crowd_stays_ambiguous():
    frame_log, bbf = {}, {}
    box_c = (100, 90, 120, 150)
    xs = {10: 105, 11: 107, 12: 105, 13: 107, 14: 105, 15: 107}
    leaders = ["T001", "T002", "T003", "T001", "T002", "T003"]
    for (f, x), lead in zip(xs.items(), leaders):
        confs = {"T001": 0.5, "T002": 0.5, "T003": 0.5}
        confs[lead] = 0.55
        frame_log[f] = {"tracks": [track("T001", BOX_A, conf=confs["T001"]),
                                    track("T002", BOX_B, conf=confs["T002"]),
                                    track("T003", box_c, conf=confs["T003"])]}
        bbf[f] = ball(x, 145)
    events, episodes = run(frame_log, bbf)
    single = [e for e in episodes if e["episode_state"] == "SINGLE_ACTOR_TOUCH"]
    assert not single, episodes


# F. two separate interactions well apart in time -> separate episodes
def test_F_distant_interactions_stay_separate():
    frame_log, bbf = {}, {}
    for f, x in {10: 95, 11: 98}.items():
        frame_log[f] = {"tracks": [track("T010", BOX_A)]}
        bbf[f] = ball(x, 145, vx=5.0)
    for f, x in {200: 95, 201: 98}.items():
        frame_log[f] = {"tracks": [track("T010", BOX_A)]}
        bbf[f] = ball(x, 145, vx=5.0)
    events, episodes = run(frame_log, bbf)
    assert len(episodes) == 2, episodes
    assert episodes[0]["episode_state"] == episodes[1]["episode_state"] == "SINGLE_ACTOR_TOUCH"


# G. the same real-time interaction at 30fps vs 60fps classifies the same
def test_G_fps_equivalence():
    frame_log30, bbf30 = {}, {}
    for f, x in {10: 95, 11: 98, 12: 100, 13: 103, 14: 106}.items():
        frame_log30[f] = {"tracks": [track("T020", BOX_A)]}
        bbf30[f] = ball(x, 145, vx=5.0)
    fc30 = generate_frame_candidates(frame_log30, bbf30, 29.97)
    scored30 = score_candidates(fc30, bbf30, 29.97)
    events30 = cluster_events(scored30, 29.97, "TEST", "synthetic")
    eps30 = build_episodes(events30, scored30, 29.97, "TEST", "synthetic")

    frame_log60, bbf60 = {}, {}
    for f, x in {20: 95, 22: 98, 24: 100, 26: 103, 28: 106}.items():
        frame_log60[f] = {"tracks": [track("T020", BOX_A)]}
        bbf60[f] = ball(x, 145, vx=2.5)
    fc60 = generate_frame_candidates(frame_log60, bbf60, 59.94)
    scored60 = score_candidates(fc60, bbf60, 59.94)
    events60 = cluster_events(scored60, 59.94, "TEST", "synthetic")
    eps60 = build_episodes(events60, scored60, 59.94, "TEST", "synthetic")

    assert eps30 and eps60
    assert eps30[0]["episode_state"] == eps60[0]["episode_state"] == "SINGLE_ACTOR_TOUCH"


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
        except Exception as e:
            print(f"[ERROR] {fn.__name__} -- {type(e).__name__}: {e}")
    print(f"\n{n_pass}/{len(fns)} touch-episode regression tests passed")
    if n_pass != len(fns):
        sys.exit(1)

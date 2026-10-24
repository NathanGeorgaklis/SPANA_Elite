"""
test_actor_attribution.py -- TASK 3F required unit tests (Section 14, 1-8)
plus extra safety cases. Synthetic frame_log records and raw frozen-engine
event dicts only -- no video, no model. Runs as a script (same harness style
as test_ball_static_decoy.py).
"""
import sys
sys.path.insert(0, "/tmp/claude-0/-home-claude/98d69dca-3149-5a43-9670-3c417f2af6d4/scratchpad")

from event_engine_v1.actor_attribution import assess_actor, build_frame_index
from event_engine_v1.analyze_target_player import assign_ids_and_merge
from event_engine_v1.player_stats import compute_player_stats, compute_stat_confidence

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, cond))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


def sel(online_id="T001", state="ACTIVE", idc=0.70, pos="CONFIRMED_DETECTION", occ="NONE", identity_state=None):
    return {"track_id": "T001", "online_track_id": online_id, "selected_player": True, "state": state,
            "identity_confidence": idc, "position_source": pos, "occlusion_status": occ,
            "identity_state": identity_state, "reid_confidence": 0.6, "overall_track_confidence": 0.75}


def frame_log(n, contest_frames=(), **kw):
    log = []
    for f in range(1, n + 1):
        contests = [["T001", "T099"]] if f in contest_frames else []
        log.append({"frame": f, "tracks": [sel(**kw)], "overlap_contests": contests})
    return log


def raw(t, s, e, conf, band, review, success=None):
    return {"type": t, "start_frame": s, "end_frame": e, "start_time": s / 25.0, "end_time": e / 25.0,
            "confidence": conf, "confidence_band": band, "review_required": review, "success": success,
            "qualifiers": []}


def run(raw_events, log, status="RESOLVED", conflict=False):
    idx = build_frame_index(log)
    assessments = [assess_actor(e, raw_events, idx, status, conflict) for e in raw_events]
    return assign_ids_and_merge(raw_events, "t", 0, 25.0, assessments)


# 1. high actor confidence + high event confidence -> TARGET_PLAYER
def t1():
    ev = run([raw("CARRY", 10, 20, 0.90, "HIGH", False), raw("PASS", 20, 30, 0.85, "HIGH", False, True)], frame_log(40))
    p = [e for e in ev if e["type"] == "PASS"][0]
    check("1: high actor + high event -> TARGET_PLAYER", p["actor"] == "TARGET_PLAYER" and p["actor_confidence_band"] == "HIGH",
          f"actor={p['actor']} actor_conf={p['actor_confidence']}")


# 2. high actor confidence + review-required event -> TARGET_PLAYER, review stays true
def t2():
    ev = run([raw("RECEIVE", 10, 12, 0.95, "HIGH", False), raw("DUEL", 12, 15, 0.677, "MEDIUM", True, False)], frame_log(40))
    d = [e for e in ev if e["type"] == "DUEL"][0]
    check("2: high actor + review-required DUEL -> TARGET_PLAYER, review_required stays true",
          d["actor"] == "TARGET_PLAYER" and d["review_required"] is True and d["event_confidence"] == 0.677,
          f"actor={d['actor']} review={d['review_required']}")


# 3. low actor confidence + high event confidence -> UNKNOWN
def t3():
    log = frame_log(40)
    for rec in log:
        if 10 <= rec["frame"] <= 20:
            rec["tracks"][0]["identity_confidence"] = 0.3      # appearance contradicts identity
    ev = run([raw("CARRY", 10, 20, 0.92, "HIGH", False), raw("PASS", 20, 30, 0.88, "HIGH", False, True)], log)
    p = [e for e in ev if e["type"] == "PASS"][0]
    check("3: low actor + high event -> UNKNOWN", p["actor"] == "UNKNOWN" and p["actor_confidence_band"] == "LOW",
          f"actor={p['actor']} actor_conf={p['actor_confidence']}")


# 4. unresolved resolver -> UNKNOWN
def t4():
    ev = run([raw("CARRY", 10, 20, 0.92, "HIGH", False)], frame_log(40), status="UNRESOLVED")
    check("4: UNRESOLVED resolver -> UNKNOWN", ev[0]["actor"] == "UNKNOWN", f"actor_conf={ev[0]['actor_confidence']}")
    ev = run([raw("CARRY", 10, 20, 0.92, "HIGH", False)], frame_log(40), status="REVIEW_REQUIRED")
    check("4b: REVIEW_REQUIRED resolver -> UNKNOWN (0.5, MEDIUM, not HIGH)", ev[0]["actor"] == "UNKNOWN",
          f"actor_conf={ev[0]['actor_confidence']}")


# 5. ambiguous competing identity -> UNKNOWN
def t5():
    ev = run([raw("CARRY", 10, 20, 0.92, "HIGH", False)], frame_log(40, contest_frames=range(10, 21)))
    check("5: selected track in overlap contest across window -> UNKNOWN", ev[0]["actor"] == "UNKNOWN",
          f"actor_conf={ev[0]['actor_confidence']}")
    log = frame_log(40)
    for rec in log:
        if rec["frame"] >= 15:
            rec["tracks"][0]["online_track_id"] = "T077"       # silent online id switch mid-window
    ev = run([raw("CARRY", 10, 20, 0.92, "HIGH", False)], log)
    check("5b: online track id switch inside window -> UNKNOWN", ev[0]["actor"] == "UNKNOWN",
          f"ids={ev[0]['actor_evidence']['online_track_ids']}")
    log = frame_log(40, identity_state="REVIEW_REQUIRED_IDENTITY")
    ev = run([raw("CARRY", 10, 20, 0.92, "HIGH", False)], log)
    check("5c: offline identity_state REVIEW_REQUIRED_IDENTITY -> UNKNOWN", ev[0]["actor"] == "UNKNOWN")


# 6. TARGET_PLAYER review-required DUEL LOST -> duels += 1, duels_lost += 1
def t6():
    ev = run([raw("RECEIVE", 10, 12, 0.95, "HIGH", False), raw("DUEL", 12, 15, 0.677, "MEDIUM", True, False)], frame_log(40))
    st = compute_player_stats(ev); sc = compute_stat_confidence(ev)
    check("6: credited review-required DUEL LOST counts", st["duels"] == 1 and st["duels_lost"] == 1 and st["duels_won"] == 0,
          f"duels={st['duels']} lost={st['duels_lost']}")
    check("6b: stat_confidence flags the duel as a review-required contribution",
          sc["review_required_contributions"].get("duels") == 1 and sc["review_required_contributions"].get("duels_lost") == 1,
          str(sc))


# 7. UNKNOWN DUEL LOST -> no target duel stat
def t7():
    ev = run([raw("RECEIVE", 10, 12, 0.95, "HIGH", False), raw("DUEL", 12, 15, 0.677, "MEDIUM", True, False)],
             frame_log(40), status="UNRESOLVED")
    st = compute_player_stats(ev)
    check("7: UNKNOWN DUEL LOST -> no duel stat", st["duels"] == 0 and st["duels_lost"] == 0)


# 8. PASS COMPLETE Task 3A semantics unchanged (HIGH CARRY->PASS COMPLETE, LOW CROSS not counted)
def t8():
    evs = [raw("CARRY", 32, 66, 0.926, "HIGH", False), raw("PASS", 66, 161, 0.804, "HIGH", False, True),
           raw("TOUCH", 245, 250, 0.75, "HIGH", False), raw("CROSS", 250, 310, 0.457, "LOW", True, False)]
    ev = run(evs, frame_log(320))
    st = compute_player_stats(ev)
    expected = {"touches": 1, "receives": 0, "carries": 1, "passes_attempted": 1, "passes_completed": 1,
                "passes_incomplete": 0, "pass_completion_pct": 100.0, "shots": 0, "shots_on_target": 0,
                "duels": 0, "duels_won": 0, "duels_lost": 0, "possession_won": 2, "possession_lost": 1,
                "events_review_required": 1}
    check("8: Task 3A PASS semantics -> identical stat line", st == expected, str(st))
    p = [e for e in ev if e["type"] == "PASS"][0]
    check("8b: PASS COMPLETE actor TARGET_PLAYER, outcome COMPLETE, conf 0.804",
          p["actor"] == "TARGET_PLAYER" and p["outcome"] == "COMPLETE" and p["confidence"] == 0.804)
    c = [e for e in ev if e["type"] == "CROSS"][0]
    check("8c: LOW-band CROSS is not stat-eligible even with a strong actor", c["stat_eligible"] is False)


# extra: release actor window is the PARENT possession, not ball flight afterwards
def t9():
    log = frame_log(60)
    for rec in log:
        if rec["frame"] > 21:
            rec["tracks"][0]["state"] = "AMBIGUOUS"     # identity degrades AFTER the release
    ev = run([raw("RECEIVE", 15, 21, 0.9, "HIGH", False), raw("CROSS", 21, 40, 0.6, "MEDIUM", True, False)], log)
    c = [e for e in ev if e["type"] == "CROSS"][0]
    check("9: post-release identity loss does not change who released the ball",
          c["actor"] == "TARGET_PLAYER" and c["actor_evidence"]["actor_window_frames"] == [15, 21])
    ev = run([raw("RECEIVE", 15, 21, 0.9, "HIGH", False), raw("DUEL", 21, 40, 0.6, "MEDIUM", True, False)], log)
    d = [e for e in ev if e["type"] == "DUEL"][0]
    check("9b: but a DUEL needs identity through the contest window too -> UNKNOWN here", d["actor"] == "UNKNOWN",
          f"actor_conf={d['actor_confidence']}")


# extra: missing assessment fails closed
def t10():
    ev = assign_ids_and_merge([raw("CARRY", 1, 5, 0.9, "HIGH", False)], "t", 0, 25.0, None)
    check("10: no actor assessment supplied -> UNKNOWN, not counted", ev[0]["actor"] == "UNKNOWN" and
          compute_player_stats(ev)["carries"] == 0)


for fn in (t1, t2, t3, t4, t5, t6, t7, t8, t9, t10):
    fn()
fails = [n for n, ok in RESULTS if not ok]
print(f"\n{len(RESULTS) - len(fails)}/{len(RESULTS)} checks passed")
if fails:
    raise SystemExit(f"FAILED: {fails}")

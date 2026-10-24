"""Task 3F audit (duel_lost_01 only -- the only clip with a per-frame ball-position
trace, from Task 3E's read-only instrumented rerun). For each actor-window frame with a
ball position, compare ball->target-box distance against ball->every-other-track-box
distance. Answers 'could another player plausibly be the one in possession?'.
Read-only; audit evidence, not used by the production rule (the frozen engine does not
expose per-frame ball positions through its public return value)."""
import sys, json
mo = json.load(open("event_engine_v1/duel_lost_01/mo_tracking/8a710009-dual_lost_01_mo_tracking_results.json"))
fl = {r["frame"]: r for r in mo["frame_log"]}
trace = {r["frame"]: r for r in json.load(open("event_engine_v1/duel_lost_01/_diag_ball_vs_player_trace_postfix.json"))}
audit = json.load(open("event_engine_v1/task3f/duel_lost_01_actor_audit.json"))

def box_dist(b, x, y):
    dx = max(b[0] - x, 0, x - b[2]); dy = max(b[1] - y, 0, y - b[3])
    return (dx * dx + dy * dy) ** 0.5

out = {}
for ev in audit:
    w0, w1 = ev["actor_evidence"]["actor_window_frames"]
    rows = []
    for f in range(w0, w1 + 1):
        t = trace.get(f)
        if not t or t["ball_cx"] is None:
            continue
        bx, by = t["ball_cx"], t["ball_cy"]
        sel = [tr for tr in fl[f]["tracks"] if tr["selected_player"]][0]
        dt = box_dist(sel["bbox"], bx, by)
        others = sorted((box_dist(tr["bbox"], bx, by), tr["track_id"], tr.get("team"))
                        for tr in fl[f]["tracks"] if not tr["selected_player"] and tr["state"] in ("ACTIVE", "PROVISIONAL", "AMBIGUOUS"))
        nearest = others[0] if others else (None, None, None)
        rows.append(dict(frame=f, ball_state=t["ball_state"], ball=(round(bx), round(by)),
                         target_box_dist=round(dt, 1), nearest_other=(round(nearest[0], 1) if nearest[0] is not None else None, nearest[1], nearest[2]),
                         target_closest=(nearest[0] is None or dt <= nearest[0])))
    key = f"{ev['type']}@{ev['frames'][0]}"
    out[key] = rows
    print(key)
    for r in rows:
        print("   ", r)
json.dump(out, open("event_engine_v1/task3f/duel_lost_01_competing_proximity.json", "w"), indent=2)

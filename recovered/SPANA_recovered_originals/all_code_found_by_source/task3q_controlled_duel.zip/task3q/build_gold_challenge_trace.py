"""Task 3Q Part 3: GOLD_DUEL_01 frame-by-frame challenge trace, 7.0-10.0 s (+ context).

Measurement only. Inputs (both frozen, both produced by current code on the frozen Task 3N tracking):
  * ../task3o/gold_capture/capture.json  -- per-frame Pass-2 values captured in Task 3O: target snapshot
    (bbox, tracking state), ball observation (state, centre), PossessionFrame, and the team label of every
    other-player box. Task 3P changed none of these (it only removed UNCERTAIN boxes from nearest_* fields).
  * ../task3n/gold/pass1/..._mo_tracking_results.json -- Pass-1 frame log, used ONLY for the audit columns
    (pass1 target state, frames since confirmed identity, Pass-1 T003 box). The engine never sees T003.

No label is read. The challenger is not looked up by shirt number or track id: it is the OPPONENT-labelled
box chained by continuity from the OPPONENT box nearest the target at the possession interruption.
All geometry is in units of body height (h).
"""
import csv
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = json.load(open(os.path.join(HERE, "../task3o/gold_capture/capture.json")))["frames"]
P1 = {x["frame"]: x for x in json.load(open(os.path.join(
    HERE, "../task3n/gold/pass1/GOLD_DUEL_01_syo_mar_3036_3048_mo_tracking_results.json")))["frame_log"]}
FPS = 30.0
OFFSET = 180          # engine frame f == gold extract frame 180 + f
F_LO, F_HI = 24, 125  # engine frames written (6.8-10.2 s; the requested 7.0-10.0 s plus margin)
RELEASE_FRAME = 82    # the possession interruption the engine evaluates (segment end of CARRY 16-82),
                      # read from the Task 3P event log, not from the truth label

VALID_BALL = ("CONFIRMED", "REACQUIRING")
REACH_H = 0.5          # challenger foot within half a body height of the ball (leg reach)
CONTACT_CENTRE_H = 0.75
CHAIN_STEP_H = 0.5     # max challenger centre step per frame when chaining
ENGAGE_H = 1.5
OPP_CONTROL_H = 0.35


def centre(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def foot(b):
    return ((b[0] + b[2]) / 2.0, b[3])


def height(b):
    return b[3] - b[1]


def overlap(a, b):
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def frame(f):
    return CAP[str(f)]


# ---- held body scale: the target's own height at its last TRACKING frame
held_h = {}
last_h = None
for f in range(0, 180):
    det = frame(f)["detector"]
    if det["snapshot_state"] == "TRACKING":
        last_h = height(det["snapshot_bbox"])
    held_h[f] = last_h

# ---- challenger chain (OPPONENT boxes only), seeded at the release frame
def opp_boxes(f):
    return [t["box"] for t in frame(f)["team"] if t["team"] == "OPPONENT"]


seed_box = min(opp_boxes(RELEASE_FRAME),
               key=lambda b: math.dist(centre(b), centre(frame(RELEASE_FRAME)["detector"]["snapshot_bbox"])))
chain = {RELEASE_FRAME: seed_box}
for direction in (-1, 1):
    last_f, last_b = RELEASE_FRAME, seed_box
    f = RELEASE_FRAME + direction
    while F_LO - 10 <= f <= F_HI + 10:
        h = held_h[f]
        cands = [b for b in opp_boxes(f) if math.dist(centre(b), centre(last_b)) <= CHAIN_STEP_H * h * abs(f - last_f)]
        if cands:
            b = min(cands, key=lambda b: math.dist(centre(b), centre(last_b)))
            chain[f] = b
            last_f, last_b = f, b
        elif abs(f - last_f) > int(round(0.5 * FPS)):
            break
        f += direction


def store_id(f, box):
    for r in frame(f)["merge"]["records"]:
        if [round(v, 1) for v in r["box"]] == [round(v, 1) for v in box]:
            return r["track_id"]
    return None


rows = []
prev_ball = None
carry_run = 0
for f in range(F_LO, F_HI + 1):
    fr = frame(f)
    det = fr["detector"]
    pf = det["possession"]
    tb = det["snapshot_bbox"]
    state = det["snapshot_state"]
    h = held_h[f]
    p1t = [t for t in P1.get(f, {}).get("tracks", []) if t["track_id"] == "T001"]
    p1o = [t for t in P1.get(f, {}).get("tracks", []) if t["track_id"] == "T003"]
    bc = det["ball_centre"]
    bstate = det["ball_state"]
    ball_valid = bc is not None and bstate in VALID_BALL
    speed = None
    if ball_valid:
        if prev_ball is not None:
            speed = math.dist(bc, prev_ball[1]) / (f - prev_ball[0]) * FPS / h
        prev_ball = (f, bc)
    carry_run = carry_run + 1 if pf["in_possession"] else 0
    target_obs = state == "TRACKING"
    btd = math.dist(foot(tb), bc) / h if (bc is not None and ball_valid) else None

    cb = chain.get(f)
    c_centre_d = c_foot_d = c_iou = c_reach = c_overlap = None
    if cb is not None:
        c_centre_d = math.dist(centre(cb), centre(tb)) / h
        c_foot_d = math.dist(foot(cb), foot(tb)) / h
        c_overlap = overlap(cb, tb)
        c_iou = iou(cb, tb)
        if ball_valid:
            c_reach = math.dist(foot(cb), bc) / height(cb)
    # closing rate: change of challenger-target centre distance over +-2 frames (h per s), observed frames only
    rate = None
    if cb is not None and target_obs and (f - 2) in chain and (f + 2) in chain:
        a, b2 = frame(f - 2)["detector"], frame(f + 2)["detector"]
        if a["snapshot_state"] == "TRACKING" and b2["snapshot_state"] == "TRACKING":
            da = math.dist(centre(chain[f - 2]), centre(a["snapshot_bbox"])) / h
            db = math.dist(centre(chain[f + 2]), centre(b2["snapshot_bbox"])) / h
            rate = (db - da) / (4.0 / FPS)

    # reach / contact both need the target's own position -> only on frames where the target is observed
    reach_frame = bool(target_obs and c_reach is not None and c_reach <= REACH_H and
                       c_centre_d is not None and c_centre_d <= 1.0)
    contact_frame = bool(target_obs and c_overlap and c_centre_d is not None and c_centre_d <= CONTACT_CENTRE_H)
    if not target_obs:
        contest = "UNOBSERVED"
    else:
        contest = "YES" if (reach_frame or contact_frame) else "NO"

    # opponent control: any OPPONENT foot within 0.35 of its own height of a valid ball, box not overlapping target
    opp_ctrl = False
    nearest_opp_ball = None
    for ob in opp_boxes(f):
        if ball_valid:
            d = math.dist(foot(ob), bc) / height(ob)
            nearest_opp_ball = d if nearest_opp_ball is None else min(nearest_opp_ball, d)
            if d <= OPP_CONTROL_H and not overlap(ob, tb):
                opp_ctrl = True
    nearest_opp_ball_h = None
    if ball_valid and opp_boxes(f):
        nearest_opp_ball_h = min(math.dist(foot(ob), bc) / h for ob in opp_boxes(f))

    rows.append({
        "engine_frame": f, "extract_frame": f + OFFSET, "time_s": round((f + OFFSET) / FPS, 3),
        "target_tracking_state": state,
        "pass1_target_state": p1t[0]["state"] if p1t else None,
        "pass1_frames_since_confirmed_identity": p1t[0].get("frames_since_confirmed_identity") if p1t else None,
        "target_bbox": [round(v) for v in tb], "body_height_px": round(h, 1),
        "target_box_is_held": not target_obs,
        "target_in_possession": pf["in_possession"], "carry_run_frames": carry_run,
        "ball_state": bstate, "ball_valid": ball_valid,
        "ball_x": None if bc is None else round(bc[0], 1), "ball_y": None if bc is None else round(bc[1], 1),
        "ball_target_dist_px": pf["ball_distance"] if ball_valid else None,
        "ball_target_dist_h": None if btd is None else round(btd, 2),
        "ball_speed_h_per_s": None if speed is None else round(speed, 2),
        "challenger_box": None if cb is None else [round(v) for v in cb],
        "challenger_store_track_id": None if cb is None else store_id(f, cb),
        "pass1_T003_bbox_audit_only": [round(v) for v in p1o[0]["bbox"]] if p1o else None,
        "pass1_T003_team_audit_only": p1o[0]["team"] if p1o else None,
        "challenger_target_centre_dist_h": None if c_centre_d is None else round(c_centre_d, 2),
        "challenger_target_foot_dist_h": None if c_foot_d is None else round(c_foot_d, 2),
        "boxes_overlap": c_overlap, "bbox_iou": None if c_iou is None else round(c_iou, 3),
        "challenger_ball_reach_own_h": None if c_reach is None else round(c_reach, 2),
        "separation_rate_h_per_s": None if rate is None else round(rate, 2),
        "opponent_closing": None if rate is None else rate < 0,
        "contest_active": contest,
        "target_controls_ball": bool(pf["in_possession"] and target_obs),
        "opponent_controls_ball": opp_ctrl,
        "nearest_opponent_ball_dist_h": None if nearest_opp_ball_h is None else round(nearest_opp_ball_h, 2),
        "target_closer_to_ball_than_opponents": (None if (btd is None or nearest_opp_ball_h is None or not target_obs)
                                                 else btd < nearest_opp_ball_h),
    })

# ---- phase identification from the measured columns (rules stated in TASK3Q_GOLD_CHALLENGE_TRACE notes)
by_f = {r["engine_frame"]: r for r in rows}
contest_frames = [r["engine_frame"] for r in rows if r["contest_active"] == "YES"]
near_release = [f for f in contest_frames if abs(f - RELEASE_FRAME) <= int(round(0.3 * FPS))]
entry = min(near_release)
# contest end: last YES frame before an observed NO-run longer than 0.2 s
gap_max = int(round(0.2 * FPS))
end = entry
no_run = 0
for f in range(entry, F_HI + 1):
    c = by_f[f]["contest_active"]
    if c == "YES":
        end, no_run = f, 0
    elif c == "NO":
        no_run += 1
        if no_run > gap_max:
            break
# challenge start: walk back from entry while challenger chained and within 1.5 h (observed frames), <= 2.0 s
start = entry
for f in range(entry - 1, max(F_LO, entry - int(round(2.0 * FPS))) - 1, -1):
    r = by_f[f]
    d = r["challenger_target_centre_dist_h"]
    if d is None or d > ENGAGE_H:
        break
    start = f
# peak: smallest challenger ball reach (valid ball) inside the contest
peak = min((f for f in range(entry, end + 1) if by_f[f]["challenger_ball_reach_own_h"] is not None),
           key=lambda f: by_f[f]["challenger_ball_reach_own_h"])
# outcome resolution: first frame after contest end where the target controls the ball for >= 2 frames
resolution = None
for f in range(end + 1, F_HI):
    if by_f[f]["target_controls_ball"] and by_f[f + 1]["target_controls_ball"]:
        resolution = f
        break
opp_ctrl_after_entry = [f for f in range(entry, (resolution or F_HI) + 1) if by_f[f]["opponent_controls_ball"]]

marks = {start: "CHALLENGE_START", entry: "CONTEST_ENTRY", peak: "PEAK_CONTEST", end: "CHALLENGE_END",
         resolution: "OUTCOME_RESOLUTION", RELEASE_FRAME: "POSSESSION_INTERRUPTION"}
for r in rows:
    f = r["engine_frame"]
    tags = [v for k, v in marks.items() if k == f]
    if f < start:
        ph = "PRE_CHALLENGE"
    elif f < entry:
        ph = "APPROACH"
    elif f <= end:
        ph = "CONTEST"
    elif resolution is None or f < resolution:
        ph = "POST_CONTEST"
    else:
        ph = "RESOLVED"
    r["phase"] = ph
    r["marker"] = "+".join(tags)

cols = ["engine_frame", "extract_frame", "time_s", "phase", "marker"] + \
       [k for k in rows[0] if k not in ("engine_frame", "extract_frame", "time_s", "phase", "marker")]
out = os.path.join(HERE, "TASK3Q_GOLD_CHALLENGE_TRACE.csv")
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader()
    for r in rows:
        w.writerow(r)

summary = {
    "CHALLENGE_START": start, "CONTEST_ENTRY": entry, "PEAK_CONTEST": peak, "CHALLENGE_END": end,
    "OUTCOME_RESOLUTION": resolution, "POSSESSION_INTERRUPTION": RELEASE_FRAME,
    "times_s": {k: (None if v is None else round((v + OFFSET) / FPS, 2)) for k, v in
                {"start": start, "entry": entry, "peak": peak, "end": end, "resolution": resolution}.items()},
    "closing_h": round(by_f[start]["challenger_target_centre_dist_h"] - by_f[entry]["challenger_target_centre_dist_h"], 2),
    "contest_yes_frames": sum(1 for f in range(entry, end + 1) if by_f[f]["contest_active"] == "YES"),
    "contest_unobserved_frames": sum(1 for f in range(entry, end + 1) if by_f[f]["contest_active"] == "UNOBSERVED"),
    "ball_reach_frames": [f for f in range(entry, end + 1)
                          if by_f[f]["contest_active"] == "YES" and (by_f[f]["challenger_ball_reach_own_h"] or 9) <= REACH_H
                          and (by_f[f]["challenger_target_centre_dist_h"] or 9) <= 1.0],
    "opponent_control_frames_entry_to_resolution": opp_ctrl_after_entry,
}
json.dump(summary, open(os.path.join(HERE, "work_gold_trace_summary.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))

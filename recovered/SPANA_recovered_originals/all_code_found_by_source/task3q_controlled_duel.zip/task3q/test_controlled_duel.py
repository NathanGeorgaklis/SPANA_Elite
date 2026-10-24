"""
test_controlled_duel.py -- TASK 3Q: DUEL path B (controlled-possession challenge).

Synthetic, model-free scenarios built from generic football situations. No clip, frame number,
shirt number, colour or GOLD coordinate appears here. Exercises the real controlled_duel.evaluate(),
the real event_arbitration.resolve_release() (which calls the frozen shot_candidate.evaluate_release),
the real ConfidenceEngine, possession_events and player_stats.

Required cases (Part 18) are 1-10; 11-18 cover identity, scale/fps invariance, the flag, the event
window, confidence, stats and possession semantics.
"""
import sys
from types import SimpleNamespace as NS

sys.path.insert(0, "/tmp/claude-0/-home-claude/98d69dca-3149-5a43-9670-3c417f2af6d4/scratchpad")

from football_analysis import controlled_duel as cd
from football_analysis.ball_track_state import BallTrackState
from football_analysis.possession_detector import PossessionFrame
from football_analysis.event_arbitration import resolve_release
from football_analysis.event_classifier import Event
from football_analysis.confidence import ConfidenceEngine
from football_analysis import event_candidates
from event_engine_v1.possession_events import derive_possession_events
from event_engine_v1.player_stats import compute_player_stats

RESULTS = []


def check(label, cond, detail=""):
    RESULTS.append(bool(cond))
    print(f"[{'PASS' if cond else 'FAIL'}] {label}" + ("" if cond else f"  -- {detail}"))


class FakeBallTracker:
    def __init__(self, history):
        self.history = history

    def velocity_at(self, idx, window=3):          # verbatim from ball_tracker.BallTracker.velocity_at
        pts = []
        lo, hi = max(0, idx - window), min(len(self.history), idx + window + 1)
        for k in range(lo, hi):
            o = self.history[k]
            if o.state not in ("LOST", "OCCLUDED") and o.centre_x is not None:
                pts.append((o.frame_number, (o.centre_x, o.centre_y)))
        if len(pts) < 2:
            return None
        (f0, c0), (f1, c1) = pts[0], pts[-1]
        if f1 == f0:
            return None
        return ((c1[0] - c0[0]) / (f1 - f0), (c1[1] - c0[1]) / (f1 - f0))


# ---------------------------------------------------------------- scene builder
class Scene:
    """Everything in body heights (H) and seconds, converted to pixels/frames by `scale` and `fps`.
    Target box: width 0.45 H, foot at ground_y. Ball sits on the ground line."""
    def __init__(self, duration_s=4.0, fps=25.0, scale=100.0):
        self.fps, self.H = fps, scale
        self.n = int(round(duration_s * fps))
        self.ground_y = 4.0 * scale
        self.target_x = [None] * self.n
        self.target_state = ["TRACKING"] * self.n
        self.ball = [None] * self.n          # (x_H, state)
        self.poss = [False] * self.n
        self.others = [[] for _ in range(self.n)]   # [(x_H, team)]

    def f(self, t):
        return int(round(t * self.fps))

    def box(self, x_h):
        H = self.H
        cx = x_h * H
        return (cx - 0.225 * H, self.ground_y - H, cx + 0.225 * H, self.ground_y)

    def set_range(self, attr, t0, t1, fn):
        for k in range(self.f(t0), min(self.n, self.f(t1) + 1)):
            t = k / self.fps
            getattr(self, attr)[k] = fn(t)

    def add_other(self, t0, t1, fn_x, team="OPPONENT"):
        for k in range(self.f(t0), min(self.n, self.f(t1) + 1)):
            self.others[k].append((fn_x(k / self.fps), team))

    def build(self):
        H = self.H
        snaps, hist, pfs, boxes = [], [], [], []
        for k in range(self.n):
            tb = self.box(self.target_x[k])
            snaps.append(NS(bbox=list(tb), tracking_state=self.target_state[k], frame_number=k))
            b = self.ball[k]
            if b is None:
                hist.append(BallTrackState(frame_number=k, centre_x=None, centre_y=None, state="LOST"))
                bd, bs = None, "LOST"
            else:
                bx, bs = b
                hist.append(BallTrackState(frame_number=k, centre_x=bx * H, centre_y=self.ground_y - 0.02 * H,
                                           state=bs, confidence=0.9))
                bd = abs(bx * H - (tb[0] + tb[2]) / 2.0)
            pfs.append(PossessionFrame(frame=k, ball_distance=bd, ball_state=bs, in_possession=self.poss[k],
                                       tracking_state=self.target_state[k]))
            boxes.append([(self.box(x), team) for x, team in self.others[k]])
        ctx = cd.ControlledDuelContext(snaps, hist, boxes, fps=self.fps)
        return ctx, pfs, hist


def challenge_scene(fps=25.0, scale=100.0, outcome="WON", challenger_team="OPPONENT", passive=False,
                    approach_only=False, long_occlusion=False, ball_escape=False):
    """Target dribbles right from x=2.0 H. Possession 0.0-1.6 s (i_end at 1.6 s). Unless passive /
    approach_only, the opponent closes from 1.8 H ahead-left to body contact by ~1.5 s, his foot reaches the
    ball 1.5-1.8 s, the ball is knocked locally (<= 0.8 H) and the target's control is interrupted."""
    s = Scene(fps=fps, scale=scale)
    s.set_range("target_x", 0.0, 4.0, lambda t: 2.0 + 0.3 * t)
    s.set_range("ball", 0.0, 1.6, lambda t: (2.0 + 0.3 * t + 0.15, "CONFIRMED"))
    s.set_range("poss", 0.0, 1.6, lambda t: True)
    if passive:
        # stands 0.8 H BEHIND the dribbler (ball on the far side), never closes, never reaches
        s.add_other(0.0, 4.0, lambda t: 2.0 + 0.3 * t - 0.8, challenger_team)
    elif approach_only:
        # closes from 1.8 H to 0.9 H behind the dribbler, then shadows; never reaches the ball
        s.add_other(0.0, 4.0, lambda t: 2.0 + 0.3 * t - max(0.9, 1.8 - 0.9 * t), challenger_team)
    else:
        def opp_x(t):
            tx = 2.0 + 0.3 * t
            if t <= 1.5:
                return tx + 0.15 + max(0.25, 1.8 - 1.0 * t)           # closes onto the ball side
            if t <= 2.0:
                return tx + 0.4                                        # engaged, foot at the ball
            if outcome == "UNKNOWN":
                return tx + 0.4 - 2.0 * (t - 2.0)                      # breaks off behind the target
            return tx + 0.4 + 2.0 * (t - 2.0) if outcome != "LOST" else None
        s.add_other(0.0, 2.0, opp_x, challenger_team)
        if outcome != "LOST":
            s.add_other(2.04, 4.0, opp_x, challenger_team)
    # interruption: ball knocked locally 1.64-2.0 s
    if not passive and not approach_only:
        if ball_escape:
            s.set_range("ball", 1.64, 2.4, lambda t: (2.0 + 0.3 * 1.6 + 0.15 + 6.0 * (t - 1.6), "CONFIRMED"))
        else:
            s.set_range("ball", 1.64, 2.0, lambda t: (2.0 + 0.3 * 1.6 + 0.15 + 0.5 * (t - 1.6) / 0.4, "CONFIRMED"))
    else:
        s.set_range("ball", 1.64, 2.0, lambda t: (2.0 + 0.3 * t + 0.6, "CONFIRMED"))
    if long_occlusion:
        s.set_range("target_state", 1.7, 2.9, lambda t: "OCCLUDED")
    if passive or approach_only or outcome == "WON":
        # target reaches the ball first and re-establishes control
        s.set_range("ball", 2.04, 4.0, lambda t: (2.0 + 0.3 * t + 0.15, "CONFIRMED"))
        s.set_range("poss", 2.3, 4.0, lambda t: True)
    elif outcome == "LOST":
        # opponent takes the ball away, separating from the target
        s.add_other(2.04, 4.0, lambda t: 2.0 + 0.3 * 2.0 + 0.4 + 1.5 * (t - 2.0), challenger_team)
        s.set_range("ball", 2.04, 4.0, lambda t: (2.0 + 0.3 * 2.0 + 0.4 + 1.5 * (t - 2.0) + 0.1, "CONFIRMED"))
    else:  # UNKNOWN: ball runs loose ahead, away from both players; nobody controls it within the window
        s.set_range("ball", 2.04, 4.0, lambda t: (2.0 + 0.3 * 1.6 + 0.15 + 0.5 + 0.6 * (t - 2.0), "CONFIRMED"))
    return s


def assess(scene):
    ctx, pfs, _ = scene.build()
    return cd.evaluate(ctx, pfs, scene.f(1.6), 0)


# ---------------------------------------------------------------- Part 18 cases
print("=" * 16, "required cases (Part 18)", "=" * 16)

a = assess(challenge_scene(passive=True))
check("1 controlled dribbler + passive nearby opponent -> NO DUEL", not a.valid, a.failed_gates)

a = assess(challenge_scene(approach_only=True))
check("2 opponent approaches but never contests the ball -> NO DUEL", not a.valid, a.failed_gates)

a_won = assess(challenge_scene(outcome="WON"))
check("3 genuine challenge, attacker retains -> DUEL WON",
      a_won.valid and a_won.outcome == "WON", (a_won.failed_gates, a_won.outcome))

a_lost = assess(challenge_scene(outcome="LOST"))
check("4 genuine challenge, opponent wins the ball -> DUEL LOST",
      a_lost.valid and a_lost.outcome == "LOST", (a_lost.failed_gates, a_lost.outcome))

a_unk = assess(challenge_scene(outcome="UNKNOWN"))
check("5 genuine challenge, ambiguous resolution -> DUEL UNKNOWN",
      a_unk.valid and a_unk.outcome == "UNKNOWN", (a_unk.failed_gates, a_unk.outcome))


# 6 / 7 / 8 go through the real resolve_release (frozen shot_candidate inside)
def release_case(touch_len, release_path, nearest_by_rel, incoming=2.0, fps=25.0):
    hist, pfs = [], []
    x, y = 500.0, 400.0
    for f in range(touch_len):
        hist.append(BallTrackState(frame_number=f, centre_x=x, centre_y=y, state="CONFIRMED", confidence=0.9))
        pfs.append(PossessionFrame(frame=f, ball_distance=10.0, ball_state="CONFIRMED", in_possession=True,
                                   tracking_state="TRACKING"))
    for i, (dx, dy) in enumerate(release_path):
        f = touch_len + i
        x, y = x + dx, y + dy
        team, dv = nearest_by_rel.get(i, (None, None))
        hist.append(BallTrackState(frame_number=f, centre_x=x, centre_y=y, state="CONFIRMED", confidence=0.9))
        pfs.append(PossessionFrame(frame=f, ball_distance=None, ball_state="CONFIRMED", in_possession=False,
                                   tracking_state="TRACKING", nearest_other_team=team, nearest_other_distance=dv,
                                   nearest_teammate_distance=dv if team == "TEAMMATE" else None))
    while len(hist) < touch_len + 140:
        f = len(hist)
        hist.append(BallTrackState(frame_number=f, centre_x=None, centre_y=None, state="LOST"))
        pfs.append(PossessionFrame(frame=f, ball_distance=None, ball_state="LOST", in_possession=False,
                                   tracking_state="TRACKING"))
    return FakeBallTracker(hist), pfs, touch_len - 1, {"incoming_speed_px_per_frame": incoming,
                                                       "segment_frames": touch_len}


def pressure_context(bt, i_end, fps=25.0, H=100.0):
    """An opponent who closes in and engages the target around the release, with the target box parked
    at the ball's pre-release position. Returns a ControlledDuelContext over bt.history."""
    n = len(bt.history)
    tx = bt.history[0].centre_x - 0.15 * H
    gy = bt.history[0].centre_y + 0.02 * H
    snaps, boxes = [], []
    for k in range(n):
        snaps.append(NS(bbox=[tx - 0.225 * H, gy - H, tx + 0.225 * H, gy], tracking_state="TRACKING"))
        dt = (k - i_end) / fps
        ox = tx + 0.55 * H + max(0.0, -dt) * 1.2 * H if dt <= 0.4 else None
        boxes.append([((ox - 0.225 * H, gy - H, ox + 0.225 * H, gy), "OPPONENT")] if ox is not None else [])
    return cd.ControlledDuelContext(snaps, bt.history, boxes, fps=fps)


# 6: completed pass under pressure (confirmed reception) -> PASS; path B never even evaluated
bt, pfs, i_end, prior = release_case(30, [(20.0, 0.0)] * 3 + [(1.0, 0.0)] * 6,
                                     {k: ("TEAMMATE", 12.0) for k in range(2, 12)})
base = resolve_release(bt, pfs, i_end, fps=25.0, seg_start=0, prior_segment_evidence=prior)
with_ctx = resolve_release(bt, pfs, i_end, fps=25.0, seg_start=0, prior_segment_evidence=prior,
                           controlled_duel_context=pressure_context(bt, i_end))
check("6 ordinary completed pass under mild pressure -> PASS, not DUEL",
      base.type == "PASS" and with_ctx.type == "PASS" and with_ctx.success is True and
      with_ctx.evidence.get("arbitration", {}).get("short_circuited") is True,
      (base.type, with_ctx.type, with_ctx.evidence.get("resolution_reason")))

# 7: shot under pressure (fast, straight, escapes the contest) -> not DUEL; identical to no-context result
bt, pfs, i_end, prior = release_case(30, [(45.0, 0.0)] * 12, {})
base = resolve_release(bt, pfs, i_end, fps=25.0, seg_start=0, prior_segment_evidence=prior)
with_ctx = resolve_release(bt, pfs, i_end, fps=25.0, seg_start=0, prior_segment_evidence=prior,
                           controlled_duel_context=pressure_context(bt, i_end))
cda = with_ctx.evidence.get("arbitration", {}).get("controlled_duel_assessment", {})
check("7 shot under pressure -> not automatically DUEL (ball escaped the contest; result unchanged)",
      with_ctx.type == base.type and with_ctx.type != "DUEL" and "ball_escaped_contest" in cda.get("failed_gates", []),
      (base.type, with_ctx.type, cda.get("failed_gates")))

# 8: existing contested-arrival DUEL (path A) unchanged: same type, success, path-A score with/without ctx
bt, pfs, i_end, prior = release_case(2, [(6.0, 3.0)] * 2 + [(1.0, 0.0)] * 10,
                                     {0: ("OPPONENT", 20.0), 1: ("OPPONENT", 15.0)}, incoming=180.0)
for k in range(i_end - 3, i_end + 4):
    if 0 <= k < len(pfs):
        pfs[k].nearest_other_team, pfs[k].nearest_other_distance = "OPPONENT", 20.0
base = resolve_release(bt, pfs, i_end, fps=25.0, seg_start=0, prior_segment_evidence=prior)
empty_ctx = cd.ControlledDuelContext([NS(bbox=[0, 0, 45, 100], tracking_state="TRACKING")] * len(bt.history),
                                     bt.history, [[] for _ in bt.history], fps=25.0)
with_ctx = resolve_release(bt, pfs, i_end, fps=25.0, seg_start=0, prior_segment_evidence=prior,
                           controlled_duel_context=empty_ctx)
pa = base.evidence["arbitration"]["candidates"]["DUEL"]
pb = with_ctx.evidence["arbitration"]["candidates"]["DUEL"]
check("8 existing contested-arrival DUEL -> unchanged (type, outcome, path-A candidate identical)",
      base.type == "DUEL" and with_ctx.type == base.type and with_ctx.success == base.success and pa == pb and
      cd.CANDIDATE_TYPE not in with_ctx.evidence["arbitration"]["candidates"],
      (base.type, with_ctx.type, base.success, with_ctx.success))

a = assess(challenge_scene(outcome="WON", challenger_team="TEAMMATE"))
check("9 teammate collision -> NO DUEL", not a.valid and "no_opponent_challenger" in a.failed_gates, a.failed_gates)

a = assess(challenge_scene(outcome="WON", challenger_team="UNKNOWN"))
check("10 UNKNOWN-team person nearby -> no confident DUEL", not a.valid, a.failed_gates)

# ---------------------------------------------------------------- additional coverage
print("=" * 16, "additional cases", "=" * 16)

a = assess(challenge_scene(outcome="WON", long_occlusion=True))
check("11 target unobserved for > 0.5 s around the contest -> NO DUEL (identity not grounded)",
      not a.valid and "identity_gap_too_long" in a.failed_gates, a.failed_gates)

a2 = assess(challenge_scene(outcome="WON", scale=230.0))
same_scale = (a2.valid == a_won.valid and a2.outcome == a_won.outcome and a2.window() == a_won.window() and
              all(abs(a2.components[k] - a_won.components[k]) < 1e-6 for k in a_won.components))
check("12 resolution / zoom invariance: same scene at 2.3x pixel scale -> identical verdict and components",
      same_scale, (a2.components, a_won.components))

a3 = assess(challenge_scene(outcome="WON", fps=50.0))
check("13 frame-rate invariance: same scene at 50 fps -> same verdict and outcome",
      a3.valid and a3.outcome == "WON" and abs(a3.score - a_won.score) < 0.05, (a3.failed_gates, a3.score, a_won.score))

# 14: flag off -> resolve_release identical to no context
scene = challenge_scene(outcome="WON")
ctx, spfs, shist = scene.build()
sbt = FakeBallTracker(shist)
i_end = scene.f(1.6)
prior = {"incoming_speed_px_per_frame": 2.0, "segment_frames": i_end + 1}
no_ctx = resolve_release(sbt, spfs, i_end, fps=25.0, seg_start=0, prior_segment_evidence=prior)
cd.TASK3Q_CONTROLLED_DUEL = False
off = resolve_release(sbt, spfs, i_end, fps=25.0, seg_start=0, prior_segment_evidence=prior,
                      controlled_duel_context=ctx)
cd.TASK3Q_CONTROLLED_DUEL = True
on = resolve_release(sbt, spfs, i_end, fps=25.0, seg_start=0, prior_segment_evidence=prior,
                     controlled_duel_context=ctx)
check("14 flag off -> identical to pre-3Q (no path-B candidate); flag on -> DUEL WON wins arbitration",
      off.type == no_ctx.type and off.evidence.get("arbitration", {}).get("candidates") ==
      no_ctx.evidence.get("arbitration", {}).get("candidates") and on.type == "DUEL" and on.success is True,
      (no_ctx.type, off.type, on.type, on.evidence.get("resolution_reason")))

check("15 DUEL window spans the contest (challenge start -> contest end), not the parent carry",
      on.type == "DUEL" and 0 < on.start_frame < i_end < on.end_frame and
      (on.start_frame, on.end_frame) == a_won.window() and "CONTROLLED_POSSESSION_DUEL" in on.qualifiers,
      (on.start_frame, on.end_frame, i_end))

# 16: confidence is duel-specific for path B, unchanged for path A
on.evidence["team_sim"] = None
raw_b, band_b, rr_b = ConfidenceEngine().score(on)
pa_event = Event("DUEL", 10, 12, success=True, evidence={"tracking_states": {"TRACKING": 3},
                                                       "ball_states": {"CONFIRMED": 3}, "team_sim": None})
raw_a, _, _ = ConfidenceEngine().score(pa_event)
check("16 path-B confidence = its own duel_confidence; path-A confidence formula unchanged",
      abs(raw_b - on.evidence["controlled_duel"]["duel_confidence"]) < 1e-9 and
      abs(raw_a - (0.35 * 1.0 + 0.45 * 1.0 + 0.20 * 0.4)) < 1e-9, (raw_b, raw_a))


# 17 / 18: canonical stats + possession semantics
def norm(t, start, end, success=None, tags=(), band="HIGH"):
    return {"type": t, "start_time": start, "end_time": end, "start_frame": int(start * 25), "end_frame": int(end * 25),
            "success": success, "actor": "TARGET_PLAYER", "confidence": 0.9, "review_required": success is None,
            "stat_eligible": band in ("HIGH", "MEDIUM"), "qualifiers": {"tags": list(tags), "confidence_band": band}}


won_b = [norm("CARRY", 0.0, 1.6, tags=["CARRY"]), norm("DUEL", 1.0, 2.0, True, ["DUEL_WON", "CONTROLLED_POSSESSION_DUEL"]),
         norm("CARRY", 2.3, 4.0, tags=["CARRY"])]
won_a = [norm("CARRY", 0.0, 1.6, tags=["CARRY"]), norm("DUEL", 1.6, 1.8, True, ["DUEL_WON"]),
         norm("CARRY", 2.3, 4.0, tags=["CARRY"])]
pe_b = [e["type"] for e in derive_possession_events(won_b)]
pe_a = [e["type"] for e in derive_possession_events(won_a)]
check("17 possession: path-B DUEL WON keeps the episode open; path-A DUEL keeps its documented LOST-then-WON",
      pe_b == ["POSSESSION_WON"] and pe_a == ["POSSESSION_WON", "POSSESSION_LOST", "POSSESSION_WON"], (pe_b, pe_a))

st_won = compute_player_stats(won_b)
st_unk = compute_player_stats([norm("DUEL", 1.0, 2.0, None, ["DUEL_UNKNOWN", "CONTROLLED_POSSESSION_DUEL"])])
st_lost = compute_player_stats([norm("DUEL", 1.0, 2.0, False, ["DUEL_LOST", "CONTROLLED_POSSESSION_DUEL"])])
check("18 stats: WON -> duels 1 / won 1 / passes 0; LOST -> lost 1; UNKNOWN -> duels 1, neither won nor lost",
      (st_won["duels"], st_won["duels_won"], st_won["duels_lost"], st_won["passes_attempted"]) == (1, 1, 0, 0) and
      (st_lost["duels"], st_lost["duels_lost"]) == (1, 1) and
      (st_unk["duels"], st_unk["duels_won"], st_unk["duels_lost"]) == (1, 0, 0),
      (st_won, st_lost, st_unk))

# path-A source untouched (byte-level check against the Task 3P frozen copy of event_candidates.py)
import hashlib, os
here = os.path.dirname(os.path.abspath(__file__))
cur = hashlib.sha256(open(os.path.join(here, "football_analysis/event_candidates.py"), "rb").read()).hexdigest()
man = open(os.path.join(here, "event_engine_v1/task3n/CODE_BASELINE_SHA256.txt")).read()
check("19 event_candidates.py (path-A DUEL candidate, PASS/SHOT/CROSS candidates) byte-identical to the 3N baseline",
      cur in man, cur)

print(f"\nTOTAL: {sum(RESULTS)}/{len(RESULTS)} passed")
sys.exit(0 if all(RESULTS) else 1)

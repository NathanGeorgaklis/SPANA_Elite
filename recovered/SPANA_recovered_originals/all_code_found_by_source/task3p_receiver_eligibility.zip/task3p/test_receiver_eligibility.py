"""
test_receiver_eligibility.py -- TASK 3P: target-state-aware receiver independence.

Tests 1-10 are synthetic (pure helper, a lightweight FootballEventDetector harness, and
shot_candidate / event_arbitration / player_stats with fakes). Tests 11-12 check the real
post-fix Pass-2 artifacts of pass_completed_01 and GOLD_DUEL_01 (event_engine_v1/task3p/),
so they run after those Pass-2 runs exist.

Run: python3 test_receiver_eligibility.py
"""
import json
import os
import sys
from dataclasses import dataclass
from typing import Optional

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from football_analysis import receiver_eligibility as RE
from football_analysis.receiver_eligibility import (assess_candidate, ReceiverIndependenceTracker,
                                                     RECEIVER_ELIGIBLE, RECEIVER_IDENTITY_UNCERTAIN,
                                                     RECEIVER_INELIGIBLE)
from football_analysis.possession_detector import PossessionFrame, PossessionDetector
from football_analysis.interaction_detector import InteractionFrame
from football_analysis.shot_candidate import evaluate_release
from football_analysis.event_arbitration import resolve_release
from football_analysis.event_detector import FootballEventDetector
from event_engine_v1.player_stats import compute_player_stats

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" -- {detail}" if detail and not cond else ""))


# Target box ~ 100 px tall, centred at (500, 350)
TARGET = (475.0, 300.0, 525.0, 400.0)
H = 100.0


def box_at(cx, cy, w=50.0, h=100.0):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


# ---------------------------------------------------------------- 1
v = assess_candidate(box_at(900, 350), "TEAMMATE", TARGET, "TRACKING")
check("1 target TRACKING + independent teammate -> ELIGIBLE", v.status == RECEIVER_ELIGIBLE, v)

# ---------------------------------------------------------------- 2
occluder = (430.0, 290.0, 540.0, 405.0)                      # opponent box covering the target
cand = (440.0, 300.0, 480.0, 370.0)                          # partial same-team box inside the occluder
v = assess_candidate(cand, "TEAMMATE", TARGET, "OCCLUDED", [occluder])
check("2 target OCCLUDED + same-team candidate inside occluder near last target -> UNCERTAIN",
      v.status == RECEIVER_IDENTITY_UNCERTAIN, v)

# ---------------------------------------------------------------- 3
v = assess_candidate(box_at(570, 345), "TEAMMATE", TARGET, "AMBIGUOUS")
check("3 target AMBIGUOUS + same-team candidate consistent with re-emergence (0.7 h) -> UNCERTAIN",
      v.status == RECEIVER_IDENTITY_UNCERTAIN, v)

# ---------------------------------------------------------------- 4
v = assess_candidate(box_at(800, 350), "TEAMMATE", TARGET, "OCCLUDED", [occluder])
check("4 target OCCLUDED + clearly independent teammate far away (3 h) -> ELIGIBLE",
      v.status == RECEIVER_ELIGIBLE, v)

# ---------------------------------------------------------------- 5
v = assess_candidate(box_at(520, 350), "OPPONENT", TARGET, "OCCLUDED", [occluder])
check("5 opponent candidate -> INELIGIBLE (never a receiver)", v.status == RECEIVER_INELIGIBLE, v)


# ---------------------------------------------------------------- 6
@dataclass
class FakeBallState:
    frame_number: int
    state: str
    centre_x: Optional[float]
    centre_y: Optional[float]
    confidence: float = 1.0


class FakeBallTracker:
    def __init__(self, history):
        self.history = history

    def velocity_at(self, idx, window=3):
        pts = [(o.frame_number, (o.centre_x, o.centre_y)) for o in
               self.history[max(0, idx - window):min(len(self.history), idx + window + 1)]
               if o.state not in ("LOST", "OCCLUDED") and o.centre_x is not None]
        if len(pts) < 2 or pts[-1][0] == pts[0][0]:
            return None
        (f0, c0), (f1, c1) = pts[0], pts[-1]
        return ((c1[0] - c0[0]) / (f1 - f0), (c1[1] - c0[1]) / (f1 - f0))


def pass_like_history(n=40):
    """A ball released at frame 0, travelling then stopping at frame 8 and staying put."""
    hist = []
    for k in range(n):
        x = 100 + 12 * min(k, 8)
        hist.append(FakeBallState(k, "CONFIRMED", float(x), 300.0, 0.9))
    return hist


def frames(n, team_at=None, dist_at=None, uncertain_at=None):
    out = []
    for k in range(n):
        pf = PossessionFrame(frame=k, ball_distance=200.0, ball_state="CONFIRMED", in_possession=(k == 0),
                             tracking_state="TRACKING", nearest_other_team="OPPONENT", nearest_other_distance=300.0)
        out.append(pf)
    for k, (team, d) in (team_at or {}).items():
        out[k].nearest_other_team, out[k].nearest_other_distance = team, d
        if team == "TEAMMATE":
            out[k].nearest_teammate_distance = d
        elif team == "UNKNOWN":
            out[k].nearest_unknown_distance = d
    for k, d in (uncertain_at or {}).items():
        out[k].receiver_identity_uncertain_distance = d
    return out


bt = FakeBallTracker(pass_like_history())
pfs_unknown = frames(40, team_at={k: ("UNKNOWN", 20.0) for k in range(8, 14)})
ev = evaluate_release(bt, pfs_unknown, 0, fps=25.0)
rstat = (ev.evidence.get("receiver_evidence") or {}).get("status")
check("6 ambiguous (UNKNOWN) team label close to ball -> not RECEIVER_CONFIRMED",
      rstat != "RECEIVER_CONFIRMED" and ev.evidence.get("resolution_reason") != "PASS_CONFIRMED: teammate_receiver",
      f"{rstat} {ev.evidence.get('resolution_reason')}")

# ---------------------------------------------------------------- 7
partial = box_at(620, 330, w=40, h=60)          # 1.2 h from target centre, partial body (0.6 h tall)
v = assess_candidate(partial, "TEAMMATE", TARGET, "OCCLUDED", [])
full_same_dist = box_at(620, 350, w=50, h=100)  # same distance, full body, no occluder
v_full = assess_candidate(full_same_dist, "TEAMMATE", TARGET, "OCCLUDED", [])
check("7 partial same-team box near stale held target (1.2 h) -> UNCERTAIN; full body at 1.2 h with no "
      "occluder cue -> ELIGIBLE", v.status == RECEIVER_IDENTITY_UNCERTAIN and v_full.status == RECEIVER_ELIGIBLE,
      f"{v} / {v_full}")

# ---------------------------------------------------------------- 8
tr = ReceiverIndependenceTracker(fps=25.0)
for k in range(30):     # 1.2 s of a teammate standing 2 h away while the target is TRACKING
    tr.assess_frame(TARGET, "TRACKING", [(box_at(700, 350), "TEAMMATE")])
for k in range(3):      # target hidden; the teammate drifts to 0.8 h
    verdict = tr.assess_frame(TARGET, "OCCLUDED", [(box_at(700 - 40 * (k + 1), 350), "TEAMMATE")])[0]
tr2 = ReceiverIndependenceTracker(fps=25.0)
for k in range(30):
    tr2.assess_frame(TARGET, "TRACKING", [])
verdict_new = tr2.assess_frame(TARGET, "OCCLUDED", [(box_at(580, 350), "TEAMMATE")])[0]
check("8 receiver with persistent independent history -> ELIGIBLE even next to the occluded target; "
      "same position with no history -> UNCERTAIN",
      verdict.status == RECEIVER_ELIGIBLE and verdict_new.status == RECEIVER_IDENTITY_UNCERTAIN,
      f"{verdict} / {verdict_new}")


# ---------------------------------------------------------------- 9 (detector exclusion + no short-circuit)
class _Obs:
    def __init__(self, c):
        self.center = c


class _Ball:
    def __init__(self, c):
        self.c = c

    def step(self, *a, **k):
        return _Obs(self.c)


class _Inter:
    def step(self, snapshot, ball_obs):
        return InteractionFrame(snapshot.frame_number, 120.0, "CONFIRMED", snapshot.tracking_state)


class _Team:
    def __init__(self, labels):
        self.labels = labels

    def observe_tracked_player(self, *a, **k):
        pass

    def classify(self, frame, box, frame_number=None):
        return self.labels[tuple(box)], 0.8


@dataclass
class _Snap:
    frame_number: int
    bbox: tuple
    tracking_state: str


def make_detector(labels, ball_c):
    d = FootballEventDetector.__new__(FootballEventDetector)
    d.ball_tracker, d.interaction, d.team_classifier = _Ball(ball_c), _Inter(), _Team(labels)
    d.possession, d.receiver_independence = PossessionDetector(), ReceiverIndependenceTracker(fps=25.0)
    d._other_positions, d._possession_frames, d._snapshots = [], [], []
    d._enable_visual_calibration, d._visual_calibration_interval = False, 5
    d._visual_calibration_frames, d._visual_calibration_errors = [], 0
    return d


self_box = (440.0, 300.0, 480.0, 370.0)          # target re-emerging inside the occluder
labels = {self_box: "TEAMMATE", occluder: "OPPONENT"}
det = make_detector(labels, (430.0, 380.0))
pf_occ = det.process_frame(None, _Snap(1, TARGET, "OCCLUDED"), [], [self_box, occluder])
det2 = make_detector(labels, (430.0, 380.0))
pf_trk = det2.process_frame(None, _Snap(1, TARGET, "TRACKING"), [], [self_box, occluder])
saved = RE.TASK3P_RECEIVER_INDEPENDENCE
RE.TASK3P_RECEIVER_INDEPENDENCE = False
det3 = make_detector(labels, (430.0, 380.0))
pf_off = det3.process_frame(None, _Snap(1, TARGET, "OCCLUDED"), [], [self_box, occluder])
RE.TASK3P_RECEIVER_INDEPENDENCE = saved
check("9a detector: OCCLUDED target -> self box kept out of nearest-teammate/other and recorded as "
      "identity-uncertain; TRACKING and flag-off -> unchanged behaviour",
      pf_occ.nearest_teammate_distance is None and pf_occ.nearest_other_team == "OPPONENT"
      and pf_occ.receiver_identity_uncertain_distance is not None
      and pf_trk.nearest_other_team == "TEAMMATE" and pf_trk.receiver_identity_uncertain_distance is None
      and pf_off.nearest_other_team == "TEAMMATE" and pf_off.nearest_teammate_distance == pf_trk.nearest_teammate_distance,
      f"occ={pf_occ} trk={pf_trk} off={pf_off}")

# same release, before (TEAMMATE nearest -> confirmed) vs after (identity uncertain, opponent nearest)
pfs_before = frames(40, team_at={8: ("TEAMMATE", 30.0)})
ev_before = evaluate_release(bt, pfs_before, 0, fps=25.0)
pfs_after = frames(40, team_at={8: ("OPPONENT", 65.0)}, uncertain_at={8: 30.0})
ev_after = evaluate_release(bt, pfs_after, 0, fps=25.0)
res_after = resolve_release(bt, pfs_after, 0, fps=25.0)
check("9b receiver identity uncertain -> no RECEIVER_CONFIRMED, no PASS_COMPLETE short-circuit, "
      "RECEIVER_IDENTITY_UNCERTAIN recorded",
      ev_before.evidence.get("resolution_reason") == "PASS_CONFIRMED: teammate_receiver"
      and ev_after.evidence.get("resolution_reason") != "PASS_CONFIRMED: teammate_receiver"
      and (ev_after.evidence.get("receiver_identity") or {}).get("status") == RECEIVER_IDENTITY_UNCERTAIN
      and not (res_after.evidence.get("arbitration") or {}).get("short_circuited", False)
      and not (res_after.type == "PASS" and res_after.success is True),
      f"before={ev_before.evidence.get('resolution_reason')} after={ev_after.evidence.get('resolution_reason')} "
      f"final={res_after.type}/{res_after.success}")

# ---------------------------------------------------------------- 10
base = {"actor": "TARGET_PLAYER", "confidence_band": "MEDIUM", "event_confidence_band": "MEDIUM",
        "stat_eligible": True, "review_required": True}
unknown_release = dict(base, type="UNKNOWN_RELEASE", success=None, outcome=None)
false_pass = dict(base, type="PASS", success=True, outcome="COMPLETE")
s_unknown = compute_player_stats([unknown_release])
s_pass = compute_player_stats([false_pass])
check("10 receiver identity uncertain -> release is UNKNOWN_RELEASE -> no pass stats",
      s_unknown["passes_attempted"] == 0 and s_unknown["passes_completed"] == 0 and s_pass["passes_completed"] == 1,
      f"{s_unknown} / {s_pass}")

# ---------------------------------------------------------------- 11 / 12 (real post-fix artifacts)
T3P = os.path.join(ROOT, "event_engine_v1", "task3p")
T3N = os.path.join(ROOT, "event_engine_v1", "task3n")


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


pc_new = load(os.path.join(T3P, "pc_postfix", "pass_completed_01_player_analysis.json"))
pc_raw_new = load(os.path.join(T3P, "pc_postfix", "pass_completed_01_events_raw.json"))
pc_raw_old = load(os.path.join(T3N, "pass_completed_01", "pass2", "pass_completed_01_events_raw.json"))
if pc_new and pc_raw_new and pc_raw_old:
    p_new = [e for e in pc_raw_new["events"] if e["type"] == "PASS"]
    p_old = [e for e in pc_raw_old["events"] if e["type"] == "PASS"]
    same_receiver = bool(p_new and p_old and p_new[0]["evidence"].get("receiver_evidence") ==
                         p_old[0]["evidence"].get("receiver_evidence"))
    ok = (pc_new["stats"]["passes_attempted"] == 1 and pc_new["stats"]["passes_completed"] == 1 and p_new
          and p_new[0]["success"] is True and abs(p_new[0]["confidence"] - 0.804) < 0.005 and same_receiver)
    check("11 pass_completed_01 unchanged (PASS COMPLETE ~0.804, passes 1/1, identical receiver evidence)", ok,
          f"{pc_new['stats']} {p_new[0]['confidence'] if p_new else None} same_receiver={same_receiver}")
else:
    check("11 pass_completed_01 unchanged (artifacts missing -- run the post-fix Pass 2 first)", False)

g_new = load(os.path.join(T3P, "gold_postfix", "gold_duel_01_player_analysis.json"))
g_raw = load(os.path.join(T3P, "gold_postfix", "gold_duel_01_events_raw.json"))
if g_new and g_raw:
    rel = [e for e in g_raw["events"] if e["start_frame"] == 82]
    no_pass = not any(e["type"] == "PASS" and e.get("success") is True for e in g_new["events"])
    uncertain = bool(rel and (rel[0]["evidence"].get("receiver_identity") or {}).get("status") ==
                     RECEIVER_IDENTITY_UNCERTAIN)
    ok = no_pass and uncertain and g_new["stats"]["passes_attempted"] == 0 and rel and rel[0]["type"] != "PASS"
    check("12 GOLD false self-receiver blocked (no PASS COMPLETE, receiver identity uncertain, passes 0)", ok,
          f"release={rel[0]['type'] if rel else None} uncertain={uncertain} stats={g_new['stats']}")
else:
    check("12 GOLD false self-receiver blocked (artifacts missing -- run the post-fix Pass 2 first)", False)

print()
print(f"TOTAL: {sum(ok for _, ok in RESULTS)}/{len(RESULTS)} passed")
sys.exit(0 if all(ok for _, ok in RESULTS) else 1)

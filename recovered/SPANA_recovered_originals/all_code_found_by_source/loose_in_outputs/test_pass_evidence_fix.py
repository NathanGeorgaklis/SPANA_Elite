"""
test_pass_evidence_fix.py -- targeted synthetic regression tests for THIS
task's specific fix (PASS-candidate teammate/receiver evidence in
event_arbitration.py / event_candidates.py / event_detector.py /
possession_detector.py). Complements the existing, still-fully-passing
test_event_arbitration.py (24/24) and test_receiver_reception.py (11/11)
suites -- this file exercises only the NEW behaviour those pre-existing
suites' hand-built fixtures never populated: the additive per-team
nearest_teammate_distance/nearest_unknown_distance fields, and the wider
PASS-only receiver search window.

Uses the real, unmodified football_analysis.event_arbitration.resolve_release
against small synthetic BallTrackState/PossessionFrame timelines -- same
methodology as test_event_arbitration.py, no video, no models.
"""
import sys
sys.path.insert(0, "/tmp/claude-0/-home-claude/98d69dca-3149-5a43-9670-3c417f2af6d4/scratchpad")

from football_analysis.ball_track_state import BallTrackState
from football_analysis.possession_detector import PossessionFrame
from football_analysis.event_arbitration import resolve_release, PASS_RECEIVER_SEARCH_FLOOR_S

PASS_COUNT = 0
FAIL_COUNT = 0
FPS = 25.0


def check(label, cond, detail=""):
    global PASS_COUNT, FAIL_COUNT
    if cond:
        PASS_COUNT += 1
        print(f"  PASS: {label}")
    else:
        FAIL_COUNT += 1
        print(f"  FAIL: {label}  {detail}")


class FakeBallTracker:
    def __init__(self, history):
        self.history = history

    def velocity_at(self, idx, window=3):
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


def make_ball_state(frame, cx, cy, state="CONFIRMED", confidence=0.9):
    return BallTrackState(frame_number=frame, centre_x=cx, centre_y=cy, state=state, confidence=confidence)


def resolve(bt, pfs, i_end, prior_evidence):
    return resolve_release(bt, pfs, i_end, fps=FPS, log=None, pitch_context=None,
                            keeper_candidate=None, seg_start=0, prior_segment_evidence=prior_evidence)


def build_case_v2(touch_len, touch_incoming_speed_px_frame, release_path, per_frame_overrides, tail_len=140):
    """Like test_event_arbitration.build_case, but per_frame_overrides is a
    dict {relative_frame_index: dict(**PossessionFrame kwargs)} so this file
    can populate the NEW nearest_teammate_distance/nearest_unknown_distance
    fields directly (test_event_arbitration.py's own build_case/make_pf only
    ever set the legacy nearest_other_team/nearest_other_distance pair)."""
    i_end = touch_len - 1
    n_total = touch_len + tail_len
    history, pfs = [], []
    x, y = 500.0, 500.0
    for f in range(touch_len):
        history.append(make_ball_state(f, x, y, "CONFIRMED", 0.9))
        pfs.append(PossessionFrame(frame=f, ball_distance=10.0, ball_state="CONFIRMED",
                                    in_possession=True, tracking_state="TRACKING"))
    for i, (dx, dy, state, conf) in enumerate(release_path):
        f = touch_len + i
        x, y = x + dx, y + dy
        cx = x if state not in ("OCCLUDED", "LOST") else None
        cy = y if state not in ("OCCLUDED", "LOST") else None
        history.append(make_ball_state(f, cx, cy, state, conf))
        kwargs = dict(nearest_other_team=None, nearest_other_distance=None,
                      nearest_teammate_distance=None, nearest_unknown_distance=None)
        kwargs.update(per_frame_overrides.get(i, {}))
        pfs.append(PossessionFrame(frame=f, ball_distance=None, ball_state=state, in_possession=False,
                                    tracking_state="TRACKING", **kwargs))
    while len(history) < n_total:
        f = len(history)
        history.append(make_ball_state(f, None, None, "LOST", 0.0))
        pfs.append(PossessionFrame(frame=f, ball_distance=None, ball_state="LOST",
                                    in_possession=False, tracking_state="TRACKING"))
    bt = FakeBallTracker(history)
    prior_evidence = {"incoming_speed_px_per_frame": touch_incoming_speed_px_frame, "segment_frames": touch_len}
    return bt, pfs, i_end, prior_evidence


def straight_path(n, speed):
    return [(speed, 0.0, "CONFIRMED", 0.9) for _ in range(n)]


print("=" * 20, "PASS-evidence-fix scenarios", "=" * 20)

# 1. UNKNOWN-team proximity (item #17): the actual receiver is right at the
# ball but TeamClassifier could not confidently label them TEAMMATE this
# frame (UNKNOWN) -- a real, confirmed pattern from this task's own
# instrumented trace of pass_completed_04. A genuine TEAMMATE also exists,
# but far away (700px, irrelevant). Old code: PASS sees nearest_teammate_
# distance=None ("no teammate detected... at all") since nearest_other_team
# never carries UNKNOWN through to anything PASS could use. New code: PASS
# should get real, reduced-confidence credit from the close UNKNOWN proximity.
path = straight_path(30, 15.0)
overrides = {k: {"nearest_other_team": "UNKNOWN", "nearest_other_distance": 14.0,
                  "nearest_unknown_distance": 14.0,
                  "nearest_teammate_distance": None} for k in range(10, 20)}
for k in range(0, 30):
    overrides.setdefault(k, {})
    overrides[k].setdefault("nearest_other_team", "TEAMMATE")
    overrides[k].setdefault("nearest_other_distance", 700.0)
    overrides[k].setdefault("nearest_teammate_distance", 700.0)
bt, pfs, i_end, pe = build_case_v2(5, 3.0, path, overrides)
ev = resolve(bt, pfs, i_end, pe)
arb = ev.evidence.get("arbitration", {})
pass_cand = arb.get("candidates", {}).get("PASS", {})
check("1. UNKNOWN-team-near-ball gives PASS real (not degenerate) credit",
      pass_cand.get("score", 0) > 0.15,
      f"type={ev.type} pass_score={pass_cand.get('score')} contradictory={pass_cand.get('contradictory')}")
check("1b. evidence text reflects the unclassified-team signal, not a blanket 'no teammate' claim",
      any("unclassified" in s.lower() for s in pass_cand.get("supporting", [])),
      str(pass_cand.get("supporting")))

# 2. Extended PASS-only receiver search window (item #9/#13, lobbed pass):
# a genuine teammate closes to reception range only ~2.2s after release --
# well past a short SHOT-style candidate_timeout_frames window, but inside
# this fix's PASS_RECEIVER_SEARCH_FLOOR_S=3.0s floor. Confirmed real-world
# pattern from this task's own instrumented trace of pass_completed_03
# (real reception ~1.0s past the previous window edge).
late_frame_rel = int(FPS * 2.2)   # ~55 frames after release at 25fps
path = straight_path(max(late_frame_rel + 5, 70), 10.0)
# a SUSTAINED, multi-frame close approach (matching real pass_completed_03 evidence: 8
# consecutive close frames) -- not a single-frame blip, which the fix's own regression-check
# safeguard (see event_arbitration._sustained_pass_receiver_search) deliberately excludes.
overrides = {late_frame_rel + j: {"nearest_other_team": "TEAMMATE", "nearest_other_distance": 12.0,
                                    "nearest_teammate_distance": 12.0} for j in range(4)}
bt, pfs, i_end, pe = build_case_v2(3, 2.0, path, overrides, tail_len=140)
ev = resolve(bt, pfs, i_end, pe)
arb = ev.evidence.get("arbitration", {})
pass_cand = arb.get("candidates", {}).get("PASS", {})
check("2. a teammate reaching the ball ~2.2s post-release is visible to the PASS candidate",
      pass_cand.get("evidence", {}).get("nearest_teammate_distance") == 12.0,
      f"pass_evidence={pass_cand.get('evidence')}")
check("2b. that late-arriving teammate gives PASS a real, non-degenerate score",
      pass_cand.get("score", 0) > 0.3,
      f"type={ev.type} pass_score={pass_cand.get('score')}")

# 3. Regression: genuinely no teammate/unclassified-team evidence anywhere,
# even in the extended window -- PASS must still read as the honest,
# unsupported floor case (never fabricated evidence).
path = straight_path(60, 12.0)
bt, pfs, i_end, pe = build_case_v2(3, 2.0, path, {})
ev = resolve(bt, pfs, i_end, pe)
arb = ev.evidence.get("arbitration", {})
pass_cand = arb.get("candidates", {}).get("PASS", {})
check("3. genuinely no receiver evidence anywhere -> PASS stays at the honest floor",
      pass_cand.get("score", 1.0) <= 0.06,
      f"pass_score={pass_cand.get('score')}")

# 4. A teammate within the OLD window (not requiring the extension) still
# scores exactly as it did before -- the extension is additive, not a
# wholesale rescoring of already-working cases.
path = straight_path(3, 20.0) + straight_path(6, 1.0)
overrides = {0: {"nearest_other_team": "TEAMMATE", "nearest_other_distance": 200.0, "nearest_teammate_distance": 200.0},
             1: {"nearest_other_team": "TEAMMATE", "nearest_other_distance": 80.0, "nearest_teammate_distance": 80.0},
             2: {"nearest_other_team": "TEAMMATE", "nearest_other_distance": 20.0, "nearest_teammate_distance": 20.0}}
for k in range(3, 9):
    overrides[k] = {"nearest_other_team": "TEAMMATE", "nearest_other_distance": 10.0, "nearest_teammate_distance": 10.0}
bt, pfs, i_end, pe = build_case_v2(5, 3.0, path, overrides)
ev = resolve(bt, pfs, i_end, pe)
check("4. a clean, already-working completed-pass scenario still resolves PASS/success=True",
      ev.type == "PASS" and ev.success is True, str((ev.type, ev.success)))

print()
print(f"TOTAL: {PASS_COUNT} passed, {FAIL_COUNT} failed")
sys.exit(1 if FAIL_COUNT else 0)

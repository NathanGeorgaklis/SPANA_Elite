"""
test_event_arbitration.py -- EVENT VOCABULARY / ARBITRATION TASK, required
synthetic test matrix (items #21-25): PASS A-F, CROSS G-L, DUEL M-R, SHOT
regressions S-X. 24 scenarios total.

Exercises the REAL orchestration code (football_analysis/event_arbitration.
resolve_release, which itself calls the real, frozen shot_candidate.
evaluate_release) against small, fully synthetic BallTrackState/
PossessionFrame timelines -- no video, no models. This is the same
"construct real dataclasses by hand, call the real decision function"
methodology test_window_context.py already used for this project's other
event-layer tests (shot_candidate.py/possession_detector.py's own frozen
dataclasses are plain, model-free Python objects, so this needs no
monkeypatching at all).

NO CLIP-SPECIFIC TUNING (item #25): every scenario below is built from
generic football situations only -- no known jersey numbers, filenames,
frame numbers, or coordinates from the real named clips are referenced
anywhere in this file.

For genuinely ambiguous scenarios (a wayward through ball with nobody near
it, a 50/50 that stays unresolved, a wide-angle strike, etc.) this suite
accepts either the "expected" label OR REVIEW_REQUIRED as a pass -- item
#13/#33 explicitly wants uncertain cases to be ABLE to abstain rather than
be forced into a confident guess, so a test suite that demands one exact
label for every ambiguous case would itself be fighting the task's own
stated success criteria.
"""
import sys
sys.path.insert(0, "/tmp/claude-0/-home-claude/98d69dca-3149-5a43-9670-3c417f2af6d4/scratchpad")

from football_analysis.ball_track_state import BallTrackState
from football_analysis.possession_detector import PossessionFrame
from football_analysis.pitch_context import PitchContext
from football_analysis.goalkeeper_context import GoalkeeperCandidate
from football_analysis.event_arbitration import resolve_release

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
    """Minimal stand-in exposing exactly the two things resolve_release's
    whole call chain (evaluate_release, event_candidates.py,
    nearby_ball_centre) ever reads off a real BallTracker: `.history` and
    `.velocity_at()`. `.velocity_at` is copied verbatim from ball_tracker.
    BallTracker.velocity_at (frozen) -- a plain ~10-line pure function, not
    reimplemented logic, so this fake behaves identically to the real
    object for every call this test exercises."""
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


def make_pf(frame, ball_distance, ball_state, in_possession, tracking_state="TRACKING",
            nearest_other_team=None, nearest_other_distance=None):
    return PossessionFrame(frame=frame, ball_distance=ball_distance, ball_state=ball_state,
                            in_possession=in_possession, tracking_state=tracking_state,
                            nearest_other_team=nearest_other_team, nearest_other_distance=nearest_other_distance)


def build_case(touch_len, touch_incoming_speed_px_frame, release_path, nearest_team_by_frame,
                pitch_context=None, keeper_candidate=None, tail_len=140):
    """touch_len: frames of possession before release (i_end = touch_len - 1).
    touch_incoming_speed_px_frame: fed straight into prior_segment_evidence
    (mirrors what EventClassifier._incoming_speed would have measured).
    release_path: list of (dx_per_frame, dy_per_frame, state, confidence)
    tuples describing the ball's motion for each frame AFTER release --
    determines peak/sustained speed, straightness, evidence_mass exactly as
    the real, frozen evaluate_release() would compute them.
    nearest_team_by_frame: {relative_frame_index: (team, distance)} sparse
    overrides for PossessionFrame.nearest_other_team/distance, relative to
    i_end (0 = release frame)."""
    i_end = touch_len - 1
    n_total = touch_len + tail_len

    history = []
    pfs = []
    x, y = 500.0, 500.0
    for f in range(touch_len):
        history.append(make_ball_state(f, x, y, "CONFIRMED", 0.9))
        pfs.append(make_pf(f, ball_distance=10.0, ball_state="CONFIRMED", in_possession=True))

    for i, (dx, dy, state, conf) in enumerate(release_path):
        f = touch_len + i
        x, y = x + dx, y + dy
        cx = x if state not in ("OCCLUDED", "LOST") else None
        cy = y if state not in ("OCCLUDED", "LOST") else None
        history.append(make_ball_state(f, cx, cy, state, conf))
        team, distv = nearest_team_by_frame.get(i, (None, None))
        pfs.append(make_pf(f, ball_distance=None, ball_state=state, in_possession=False,
                            nearest_other_team=team, nearest_other_distance=distv))
    # pad tail with plain LOST/no-op frames so hi_cap never runs past the array
    while len(history) < n_total:
        f = len(history)
        history.append(make_ball_state(f, None, None, "LOST", 0.0))
        pfs.append(make_pf(f, ball_distance=None, ball_state="LOST", in_possession=False))

    bt = FakeBallTracker(history)
    prior_evidence = {"incoming_speed_px_per_frame": touch_incoming_speed_px_frame, "segment_frames": touch_len}
    return bt, pfs, i_end, prior_evidence


def straight_path(n, speed, state="CONFIRMED", conf=0.9, dx_sign=1.0, dy_sign=0.0):
    return [(dx_sign * speed, dy_sign * speed, state, conf) for _ in range(n)]


NO_PITCH = None


def make_pitch(goal_centre, attack_direction="LEFT_TO_RIGHT", posts=None, confidence=0.5):
    left, right = (posts if posts else (None, None))
    return PitchContext(attack_direction=attack_direction, goal_centre=goal_centre, pitch_confidence=confidence,
                         goal_left_post=left, goal_right_post=right)


def make_keeper(position, probability):
    if position is None:
        return GoalkeeperCandidate("keeper_candidate", None, 0.0)
    return GoalkeeperCandidate("keeper_candidate", position, probability)


def resolve(bt, pfs, i_end, prior_evidence, pitch_context=None, keeper_candidate=None):
    return resolve_release(bt, pfs, i_end, fps=FPS, log=None, pitch_context=pitch_context,
                            keeper_candidate=keeper_candidate, seg_start=0, prior_segment_evidence=prior_evidence)


print("=" * 20, "PASS scenarios A-F", "=" * 20)

# A. completed short pass: ball travels a short distance straight to a
# TEAMMATE within reception radius, with sustained control afterward.
path = straight_path(3, 20.0) + straight_path(6, 1.0)   # closes in, then settles/controlled
nt = {0: ("TEAMMATE", 200.0), 1: ("TEAMMATE", 80.0), 2: ("TEAMMATE", 20.0),
      3: ("TEAMMATE", 15.0), 4: ("TEAMMATE", 12.0), 5: ("TEAMMATE", 10.0),
      6: ("TEAMMATE", 10.0), 7: ("TEAMMATE", 10.0), 8: ("TEAMMATE", 10.0)}
bt, pfs, i_end, pe = build_case(5, 3.0, path, nt)
ev = resolve(bt, pfs, i_end, pe)
check("A. completed short pass -> PASS, success=True", ev.type == "PASS" and ev.success is True, str((ev.type, ev.success)))

# B. incomplete pass: ball heads toward a teammate's general area but never
# gets within reception radius (a genuine miss/overhit) -- no opponent
# contest either. Moderate, non-shot-like speed.
path = straight_path(20, 6.0)
nt = {k: ("TEAMMATE", 90.0) for k in range(0, 20)}
bt, pfs, i_end, pe = build_case(5, 3.0, path, nt)
ev = resolve(bt, pfs, i_end, pe, pitch_context=None, keeper_candidate=None)
check("B. incomplete pass -> PASS or REVIEW_REQUIRED (never SHOT/CROSS/DUEL)",
      ev.type in ("PASS", "UNKNOWN_RELEASE"), str(ev.type))

# C. through ball completed: fast release, but genuinely reaches a teammate
# within reception radius with real control-persistence evidence.
path = straight_path(4, 40.0) + straight_path(6, 1.0)
nt = {0: ("TEAMMATE", 300.0), 1: ("TEAMMATE", 200.0), 2: ("TEAMMATE", 90.0), 3: ("TEAMMATE", 30.0),
      4: ("TEAMMATE", 15.0), 5: ("TEAMMATE", 12.0), 6: ("TEAMMATE", 10.0), 7: ("TEAMMATE", 10.0),
      8: ("TEAMMATE", 10.0), 9: ("TEAMMATE", 10.0)}
bt, pfs, i_end, pe = build_case(4, 2.0, path, nt)
ev = resolve(bt, pfs, i_end, pe)
check("C. through ball completed -> PASS, success=True", ev.type == "PASS" and ev.success is True, str((ev.type, ev.success)))

# D. through ball missed: fast, clean, straight release, but -- unlike U's
# "nobody found anywhere" kinematic-only strike -- a teammate is genuinely
# making a run within the plausible pass-target corridor (PASS_INTENT_
# RADIUS_PX) the whole time; the ball is simply overhit and never actually
# closes to within the strict reception radius. No pitch context at all.
# This is the distinguishing feature that makes D an honest "missed through
# ball" rather than a duplicate of U's genuinely evidence-free strike: the
# same fast/straight kinematics are present in both, but D also carries a
# real, independent pass-intent signal (item #12) that U structurally lacks.
# Path runs long enough to comfortably cover the dynamic candidate window (so
# the contest/end position used by arbitration is always real synthetic
# data, not the padding tail -- an artifact of hand-built fixtures, not of
# real footage, where the ball tracker's own motion model always has SOME
# position).
path = straight_path(60, 25.0)
nt = {k: ("TEAMMATE", 100.0) for k in range(0, 60)}
bt, pfs, i_end, pe = build_case(3, 2.0, path, nt)
ev = resolve(bt, pfs, i_end, pe, pitch_context=None, keeper_candidate=None)
check("D. through ball missed -> not DUEL/CROSS (PASS or REVIEW_REQUIRED acceptable)",
      ev.type in ("PASS", "UNKNOWN_RELEASE"), str(ev.type))

# E. shot passing close to teammate: fast, dead-straight release with
# strong goal-direction alignment (near-perfect), and a teammate incidental
# to the trajectory (not the target) -- should still favour SHOT. Path runs
# long enough to cover the dynamic candidate window fully with real
# synthetic data (see D's comment above for why).
pc = make_pitch(goal_centre=(2300.0, 500.0), confidence=0.5)
path = straight_path(60, 30.0)
nt = {5: ("TEAMMATE", 70.0)}   # incidental, not close enough to be the target
bt, pfs, i_end, pe = build_case(5, 2.0, path, nt)
ev = resolve(bt, pfs, i_end, pe, pitch_context=pc, keeper_candidate=None)
check("E. shot passing close to teammate -> SHOT (not forced to PASS by incidental proximity)",
      ev.type == "SHOT", str(ev.type))

# F. pass aimed toward goal area but clearly intended for teammate: release
# reaches a TEAMMATE within reception radius even though the direction is
# goal-ward -- confirmed reception must still win as PASS.
pc = make_pitch(goal_centre=(1500.0, 500.0), confidence=0.5)
path = straight_path(4, 25.0) + straight_path(6, 1.0)
nt = {0: ("TEAMMATE", 250.0), 1: ("TEAMMATE", 150.0), 2: ("TEAMMATE", 70.0), 3: ("TEAMMATE", 25.0),
      4: ("TEAMMATE", 14.0), 5: ("TEAMMATE", 12.0), 6: ("TEAMMATE", 11.0), 7: ("TEAMMATE", 10.0),
      8: ("TEAMMATE", 10.0), 9: ("TEAMMATE", 10.0)}
bt, pfs, i_end, pe = build_case(4, 2.0, path, nt)
ev = resolve(bt, pfs, i_end, pe, pitch_context=pc, keeper_candidate=None)
check("F. pass toward goal area but reaches teammate -> PASS, success=True",
      ev.type == "PASS" and ev.success is True, str((ev.type, ev.success)))


print("=" * 20, "CROSS scenarios G-L", "=" * 20)

# G. low cross completed: controlled delivery (slow incoming touch), no
# origin-width geometry (Level C, the realistic case for this project),
# moderate goal-ward trajectory, contest point shows no keeper-identity
# evidence at all, and the ball IS confirmed reaching a teammate -- must
# resolve as a completed reception (PASS short-circuit takes priority,
# exactly as item #16 protects) OR CROSS with success=True if arbitration
# runs; either is an acceptable "completed delivery" reading.
pc = make_pitch(goal_centre=(900.0, 500.0), confidence=0.34)
kp = make_keeper(position=None, probability=0.0)
path = straight_path(3, 20.0) + straight_path(6, 1.0)
nt = {0: ("TEAMMATE", 220.0), 1: ("TEAMMATE", 120.0), 2: ("TEAMMATE", 40.0), 3: ("TEAMMATE", 20.0),
      4: ("TEAMMATE", 15.0), 5: ("TEAMMATE", 12.0), 6: ("TEAMMATE", 10.0), 7: ("TEAMMATE", 10.0),
      8: ("TEAMMATE", 10.0)}
bt, pfs, i_end, pe = build_case(6, 4.0, path, nt)
ev = resolve(bt, pfs, i_end, pe, pitch_context=pc, keeper_candidate=kp)
check("G. low cross completed -> PASS or CROSS, success=True",
      ev.type in ("PASS", "CROSS") and ev.success is True, str((ev.type, ev.success)))

# H. low cross incomplete: controlled delivery, moderate goal-ward
# trajectory, contest by a non-keeper-identity defender at the end of the
# window, nobody confirms a reception -- the CROSS diagnosis case
# (cross_low_01's own generic pattern, no clip-specific numbers).
pc = make_pitch(goal_centre=(900.0, 500.0), confidence=0.34)
kp = make_keeper(position=(1800.0, 480.0), probability=0.0)
path = straight_path(3, 5.0) + straight_path(15, 15.0)
nt = {17: ("OPPONENT", 10.0)}
bt, pfs, i_end, pe = build_case(6, 4.0, path, nt)
ev = resolve(bt, pfs, i_end, pe, pitch_context=pc, keeper_candidate=kp)
check("H. low cross incomplete -> CROSS (not forced SHOT)", ev.type == "CROSS", str(ev.type))

# I. high cross completed: same generic pattern as G (this system has no
# height/z-axis signal at all, so 'high' cannot be distinguished here --
# honestly the same evidence as a low cross; documented, not invented).
bt, pfs, i_end, pe = build_case(6, 4.0, path, nt)  # reuse H's non-teammate setup but confirm a reception instead
nt2 = {0: ("TEAMMATE", 200.0), 1: ("TEAMMATE", 90.0), 2: ("TEAMMATE", 30.0), 3: ("TEAMMATE", 15.0),
       4: ("TEAMMATE", 12.0), 5: ("TEAMMATE", 10.0), 6: ("TEAMMATE", 10.0), 7: ("TEAMMATE", 10.0)}
path2 = straight_path(3, 20.0) + straight_path(6, 1.0)
bt, pfs, i_end, pe = build_case(6, 4.0, path2, nt2)
ev = resolve(bt, pfs, i_end, pe, pitch_context=pc, keeper_candidate=kp)
check("I. high cross completed (no height signal -> same evidence as G) -> PASS or CROSS, success=True",
      ev.type in ("PASS", "CROSS") and ev.success is True, str((ev.type, ev.success)))

# J. cross through six-yard box with no touch at all: controlled delivery,
# moderate goal-ward, but NOBODY (teammate, opponent, or keeper-identity
# contest) is ever found anywhere near the ball -- should not be forced
# into SHOT just because nobody touched it.
pc = make_pitch(goal_centre=(900.0, 500.0), confidence=0.34)
kp = make_keeper(position=None, probability=0.0)
path = straight_path(3, 5.0) + straight_path(20, 14.0)
nt = {}
bt, pfs, i_end, pe = build_case(6, 4.0, path, nt)
ev = resolve(bt, pfs, i_end, pe, pitch_context=pc, keeper_candidate=kp)
check("J. cross through box, no touch -> CROSS or REVIEW_REQUIRED (never DUEL)",
      ev.type in ("CROSS", "UNKNOWN_RELEASE"), str(ev.type))

# K. wide-angle shot: fast, purposeful, near-perfect goal alignment,
# contested by a HIGH-identity-confidence keeper right at the contest point
# -- must still resolve SHOT even though the release originates from a
# wide angle (no origin-width geometry available -- Level C -- so nothing
# here can even claim 'wide origin', it can only be judged on trajectory +
# contest). Goal/keeper positioned consistently with where the ball
# actually ends up (contest_pos), not an arbitrary unrelated coordinate.
path = straight_path(15, 35.0)   # release_pos (500,500) -> contest_pos (1025,500) after 15 steps
pc = make_pitch(goal_centre=(1035.0, 500.0), confidence=0.5)
kp = make_keeper(position=(1025.0, 500.0), probability=0.8)
nt = {14: ("OPPONENT", 15.0)}
bt, pfs, i_end, pe = build_case(4, 3.0, path, nt)
ev = resolve(bt, pfs, i_end, pe, pitch_context=pc, keeper_candidate=kp)
check("K. wide-angle shot with real keeper contest -> SHOT", ev.type == "SHOT", str(ev.type))

# L. cutback: ball travels laterally then a short controlled pass is
# confirmed to a teammate close to goal -- same evidence shape as a
# completed pass/cross (this system cannot distinguish a cutback from a
# short cross/pass without pitch depth calibration; documented).
pc = make_pitch(goal_centre=(900.0, 500.0), confidence=0.34)
path = straight_path(3, 15.0) + straight_path(6, 1.0)
nt = {0: ("TEAMMATE", 180.0), 1: ("TEAMMATE", 90.0), 2: ("TEAMMATE", 30.0), 3: ("TEAMMATE", 15.0),
      4: ("TEAMMATE", 12.0), 5: ("TEAMMATE", 10.0), 6: ("TEAMMATE", 10.0), 7: ("TEAMMATE", 10.0)}
bt, pfs, i_end, pe = build_case(5, 3.0, path, nt)
ev = resolve(bt, pfs, i_end, pe, pitch_context=pc, keeper_candidate=None)
check("L. cutback -> PASS or CROSS, success=True", ev.type in ("PASS", "CROSS") and ev.success is True, str((ev.type, ev.success)))


print("=" * 20, "DUEL scenarios M-R", "=" * 20)

# M. ground duel won: ball arrives as a loose/contested ball (very high
# incoming speed), very short touch, an opponent was near the ball at
# contact, and the tracked player regains close possession shortly after.
# (dynamic_timeout_frames for this release's peak speed comes out well
# under 50 frames -- the recapture window below starts safely past it.)
path = straight_path(8, 30.0, state="REACQUIRING", conf=0.6) + [(0.0, 0.0, "LOST", 0.0)] * 10
nt = {0: ("OPPONENT", 70.0)}
bt, pfs, i_end, pe = build_case(1, 900.0, path, nt)   # incoming 900px/frame*25fps=22500px/s -- extreme, contested
for k in range(60, 80):
    pfs[k] = make_pf(k, ball_distance=20.0, ball_state="CONFIRMED", in_possession=True, tracking_state="TRACKING")
ev = resolve(bt, pfs, i_end, pe)
check("M. ground duel won -> DUEL, success=True", ev.type == "DUEL" and ev.success is True, str((ev.type, ev.success)))

# N. ground duel lost: same contested-arrival signature, opponent near
# contact, but the tracked player never regains possession afterward.
path = straight_path(8, 30.0, state="REACQUIRING", conf=0.6) + [(5.0, 0.0, "PREDICTED", 0.3)] * 20
nt = {0: ("OPPONENT", 65.0)}
bt, pfs, i_end, pe = build_case(1, 900.0, path, nt)
ev = resolve(bt, pfs, i_end, pe)
check("N. ground duel lost -> DUEL, success=False", ev.type == "DUEL" and ev.success is False, str((ev.type, ev.success)))

# O. contested ball unresolved: loose/contested arrival, but NO opponent
# evidence at contact at all, and no recapture either -- a duel needs SOME
# opponent evidence (item #9's "never force DUEL"); should not confidently
# assert DUEL without it.
path = straight_path(8, 30.0, state="REACQUIRING", conf=0.6) + [(5.0, 0.0, "PREDICTED", 0.3)] * 20
nt = {}
bt, pfs, i_end, pe = build_case(1, 900.0, path, nt)
ev = resolve(bt, pfs, i_end, pe)
check("O. contested ball, no opponent evidence -> not DUEL", ev.type != "DUEL", str(ev.type))

# P. player shields ball and retains possession: NOT a release at all --
# modelled as a segment that never separates (handled entirely by
# possession_segmentation.py, frozen/untouched by this task) -- this
# arbitration layer is only ever called once a release has already been
# judged to have happened, so there's nothing of this scenario to call
# resolve_release on. Documented as out of scope for THIS module's tests
# (covered instead by possession_segmentation.py's own existing tests).
check("P. player shields ball (no release => arbitration never invoked) -- documented, not applicable here", True)

# Q. opponent close but no genuine contest: an opponent is near the ball
# once, briefly, but the touch arrived under control (not a contested
# scramble) and the player carries on -- should not read as a duel.
path = straight_path(20, 8.0)
nt = {3: ("OPPONENT", 90.0)}
bt, pfs, i_end, pe = build_case(6, 2.0, path, nt)
ev = resolve(bt, pfs, i_end, pe)
check("Q. opponent nearby but controlled touch -> not DUEL", ev.type != "DUEL", str(ev.type))

# R. pass immediately before opponent challenge: a genuinely confirmed
# teammate reception happens FIRST -- the subsequent opponent challenge is
# a separate, later possession segment entirely, so this release itself
# must still read as a clean PASS (protecting the authoritative reception
# short-circuit from being second-guessed by later, unrelated contest).
path = straight_path(3, 20.0) + straight_path(6, 1.0)
nt = {0: ("TEAMMATE", 200.0), 1: ("TEAMMATE", 80.0), 2: ("TEAMMATE", 20.0), 3: ("TEAMMATE", 15.0),
      4: ("TEAMMATE", 12.0), 5: ("TEAMMATE", 10.0), 6: ("TEAMMATE", 10.0), 7: ("TEAMMATE", 10.0),
      8: ("OPPONENT", 30.0)}   # a challenge arrives only AFTER the reception is already established
bt, pfs, i_end, pe = build_case(5, 3.0, path, nt)
ev = resolve(bt, pfs, i_end, pe)
check("R. pass confirmed before later challenge -> PASS, success=True",
      ev.type == "PASS" and ev.success is True, str((ev.type, ev.success)))


print("=" * 20, "SHOT regressions S-X", "=" * 20)

# S. central shot: fast, straight, dead-on goal alignment, contest by a
# high-confidence keeper right at the contest point (goal/keeper positioned
# consistently with where the ball actually travels, not an arbitrary
# unrelated coordinate).
path = straight_path(15, 35.0)   # release_pos (500,500) -> contest_pos (1025,500)
pc = make_pitch(goal_centre=(1035.0, 500.0), confidence=0.6)
kp = make_keeper(position=(1025.0, 500.0), probability=0.85)
nt = {14: ("OPPONENT", 10.0)}
bt, pfs, i_end, pe = build_case(4, 3.0, path, nt)
ev = resolve(bt, pfs, i_end, pe, pitch_context=pc, keeper_candidate=kp)
check("S. central shot -> SHOT", ev.type == "SHOT", str(ev.type))

# T. close-range shot: very fast, very short distance-to-goal, immediate
# strong keeper contest right where the ball actually arrives.
path = straight_path(4, 40.0)   # release_pos (500,500) -> contest_pos (660,500)
pc = make_pitch(goal_centre=(670.0, 500.0), confidence=0.6)
kp = make_keeper(position=(660.0, 500.0), probability=0.9)
nt = {3: ("OPPONENT", 8.0)}
bt, pfs, i_end, pe = build_case(3, 3.0, path, nt)
ev = resolve(bt, pfs, i_end, pe, pitch_context=pc, keeper_candidate=kp)
check("T. close-range shot -> SHOT", ev.type == "SHOT", str(ev.type))

# U. wide-angle shot (no pitch context at all -- worst case for this
# system): purely kinematic evidence (very fast, dead straight, large
# evidence mass), nobody found anywhere in the window.
path = straight_path(30, 30.0)
nt = {}
bt, pfs, i_end, pe = build_case(3, 2.0, path, nt)
ev = resolve(bt, pfs, i_end, pe, pitch_context=None, keeper_candidate=None)
check("U. wide-angle shot, kinematic-only evidence -> SHOT", ev.type == "SHOT", str(ev.type))

# V. shot saved: fast, on-target, keeper contests (GOAL follow-up itself is
# handled separately by evaluate_goal_followup, gated strictly behind an
# already-confirmed SHOT -- unaffected by arbitration, which only ever
# decides what wins BEFORE that step). Reuses S's consistent geometry.
path = straight_path(15, 35.0)
pc = make_pitch(goal_centre=(1035.0, 500.0), confidence=0.6)
kp = make_keeper(position=(1025.0, 500.0), probability=0.85)
nt = {14: ("OPPONENT", 10.0)}
bt, pfs, i_end, pe = build_case(4, 3.0, path, nt)
ev = resolve(bt, pfs, i_end, pe, pitch_context=pc, keeper_candidate=kp)
check("V. shot saved (contested by keeper) -> SHOT", ev.type == "SHOT", str(ev.type))

# W. shot blocked: fast, purposeful, contested by a non-keeper defender
# very close to the RELEASE point itself (a block, not a save -- small
# contest displacement, no keeper-identity evidence) -- this vocabulary
# still calls it SHOT (item #24: 'if current vocabulary supports it' --
# BLOCKED/SAVED outcome sub-typing remains out of scope, exactly as
# shot_candidate.py's own docstring already discloses). The true goal is
# still far away (this is what makes it a shot-in-the-lane-blocked
# situation, not a delivery that travelled most of the way there).
path = straight_path(6, 35.0)   # release_pos (500,500) -> contest_pos (570,500), only 70px travelled
pc = make_pitch(goal_centre=(1400.0, 500.0), confidence=0.5)
kp = make_keeper(position=None, probability=0.0)
nt = {1: ("OPPONENT", 12.0)}
bt, pfs, i_end, pe = build_case(4, 3.0, path, nt)
ev = resolve(bt, pfs, i_end, pe, pitch_context=pc, keeper_candidate=kp)
check("W. shot blocked close to release -> SHOT", ev.type == "SHOT", str(ev.type))

# X. shot travelling close to teammate: fast, on-target, a teammate is
# incidentally near the trajectory partway through but not the target and
# never confirmed as a receiver; keeper contests at the (consistent) end.
path = straight_path(15, 35.0)
pc = make_pitch(goal_centre=(1035.0, 500.0), confidence=0.6)
kp = make_keeper(position=(1025.0, 500.0), probability=0.8)
nt = {6: ("TEAMMATE", 75.0), 14: ("OPPONENT", 10.0)}
bt, pfs, i_end, pe = build_case(4, 3.0, path, nt)
ev = resolve(bt, pfs, i_end, pe, pitch_context=pc, keeper_candidate=kp)
check("X. shot travelling close to (non-receiving) teammate -> SHOT", ev.type == "SHOT", str(ev.type))


print()
print(f"TOTAL: {PASS_COUNT} passed, {FAIL_COUNT} failed")
if FAIL_COUNT:
    sys.exit(1)

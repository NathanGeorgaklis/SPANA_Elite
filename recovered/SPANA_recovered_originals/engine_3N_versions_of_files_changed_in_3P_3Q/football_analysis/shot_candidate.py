"""
shot_candidate.py -- ShotCandidate evaluation: replaces EventQualifier's old
fixed RELEASE_LOOKAHEAD=12 window with a persistent, temporal-evidence-
accumulating SHOT state machine.

FREEZE NOTE: this module, like every other file in football_analysis/, does
not import player_tracker, and it does not import or alter any of the
ball-tracking files that are frozen for this task (ball_track_state.py,
ball_motion.py, ball_gating.py, ball_roi_search.py, ball_confidence.py,
ball_tracker.py, ball_debug_overlay.py). It only ever READS
BallTracker.history (a list of already-produced BallTrackState snapshots)
and PossessionFrame objects that event_detector.py already built. Nothing
here calls BallTracker.step()/_advance()/_confirm() or touches ball state at
all -- it is a pure, read-only, after-the-fact interpreter of ball evidence
that already exists by the time a RELEASE is evaluated.

WHY A "STATE MACHINE" WALK OVER STORED HISTORY, NOT A NEW LIVE HOOK:
This pipeline is a two-pass, whole-clip batch design (see engine.py) --
FootballEventDetector already buffers every frame's PossessionFrame and
every ball-tracker frame's BallTrackState during PASS 2's per-frame loop,
and only turns that buffered history into discrete events afterward, in
finalize(). qualify_release() was always called post-hoc that way, even
before this rewrite -- the bug was never "runs after the fact", it was
"only ever looks 12 frames ahead once, at segment-end". So this module
still runs post-hoc (via finalize()), but it walks the buffered per-frame
history ONE FRAME AT A TIME, in order, carrying forward state (evidence
mass, best-reception-so-far, speed samples) exactly as a genuinely live,
frame-by-frame state machine would if it ran during PASS 2 itself -- the
only difference is *when* the walk happens, not *how* it evaluates each
frame. This keeps the change entirely inside event-analysis code (no new
coupling to BallTracker, no changes to when/how ball state is computed) and
is disclosed here explicitly so it is not mistaken for a truly-online
mid-clip hook. The evidence a real-time system would have at each frame k
(everything up to and including k) is exactly the evidence this walk uses
at each step k -- it never looks backward past a point in a way a live
system couldn't.

RELEASE -> SHOT_CANDIDATE lifecycle:
  RELEASE (i_end, a possession segment just ended)
      -> ACTIVE candidate opens with a dynamic (fps + release-speed aware,
         seconds-based) timeout -- NOT a fixed frame count (rule: replace
         RELEASE_LOOKAHEAD=12).
      -> walked forward frame-by-frame. At each frame:
           * a genuine RECEPTION (ball settles near another player, or
             reaches a same-kit teammate at any speed) resolves the
             candidate immediately as PASS, success True/False/None.
           * a fast ball merely passing near a defender/keeper WITHOUT
             settling is explicitly NOT treated as a reception (this is the
             fix for the real bug this task surfaced -- see
             _find_reception's docstring) -- the walk continues.
      -> if no reception resolves it before the dynamic timeout: classified
         from ACCUMULATED evidence (weighted by ball-state trustworthiness,
         never a single frame) into SHOT or UNKNOWN_RELEASE.
      -> every resolution (confirmed or rejected) is logged with a short
         REJECTED_SHOT:/PASS_CONFIRMED:/etc. reason string, exactly the
         style the task asked for, into `evidence["resolution_reason"]` and
         into the optional `log` list passed in (rule #19-style transition
         log, this time for event-analysis, not ball-tracking).

GOAL is handled separately, in evaluate_goal_followup(), and is gated
strictly behind an already-produced SHOT event -- see that function's
docstring for why it can only ever be a low-confidence, always-
review-required hypothesis in this system (no pitch/goal-line calibration
exists, and adding one is explicitly out of scope for this task).
"""
import math
from dataclasses import dataclass, asdict
from typing import Optional

from .event_classifier import Event
from .geometry import dist
from .pitch_context import goal_direction_score, distance_to_goal, lateral_wide_release_score
from .goalkeeper_context import goalkeeper_probability_at, keeper_evidence_at
from .release_geometry import estimate_release_width

RELEASE_FAST_SPEED = 16.0       # px/frame -- a release at/above this reads as a purposeful strike
RECEPTION_RADIUS = 55.0         # px -- ball within this of another player counts as "reached them"
EARLY_SPEED_HORIZON_S = 0.6     # how far forward to look for the TRUE peak release speed, rather than
                                 # trusting the instantaneous speed at the exact release frame -- a real
                                 # kick's speed can still be ramping up right at the moment a possession
                                 # segment is judged to have ended (foot-to-ball contact isn't instant)

# Distinguishing "the ball reached a defender/keeper because it was a
# purposeful strike that got contested/blocked/saved" (-> SHOT) from "the
# ball reached them because it was a genuinely slow, short, misplaced
# touch/pass" (-> PASS, failed) cannot rely on absolute px/frame speed
# alone: this project has no pitch/camera calibration, so the same real-
# world ball speed can read very differently depending on clip zoom/
# resolution, and a close-range effort can legitimately look "slow" in
# pixels while still being a shot. Two independent signals are combined
# generously (either one is enough to call it purposeful) rather than
# picking one arbitrary threshold; when NEITHER a clearly slow speed NOR a
# clearly short displacement holds, the outcome is genuinely ambiguous
# under the signals this task allows (no goalkeeper/role identification, no
# pitch-zone context) and is reported as UNKNOWN_RELEASE / review_required
# rather than forced into a confident guess either way.
PURPOSEFUL_SPEED_MIN = RELEASE_FAST_SPEED               # 16.0 px/frame
PURPOSEFUL_DISPLACEMENT_MIN = 1.5 * RECEPTION_RADIUS    # 82.5px net travel from release to contest point
CLEARLY_INCIDENTAL_SPEED = 6.0                          # px/frame -- below this, reads as a genuinely slow touch
CLEARLY_INCIDENTAL_DISPLACEMENT = RECEPTION_RADIUS      # 55px -- ball barely moved from the release point

MIN_TIMEOUT_S = 0.6
MAX_TIMEOUT_S = 2.5
BASE_TIMEOUT_S = 1.0
SPEED_TIMEOUT_BONUS_CAP_S = 0.9   # a faster release plausibly needs longer, real-world, to resolve

MIN_EVIDENCE_MASS = 1.5          # below this: not enough trustworthy ball evidence to say anything
CONFIRM_EVIDENCE_MASS = 3.0      # at/above this (+ fast + straight-ish): confident enough to call SHOT
STRAIGHTNESS_MIN = 0.45          # net-displacement / path-length -- allows real trajectory curvature
                                  # (perspective, dip) while still rejecting sharply deflected/broken paths

# Weights per ball-state trustworthiness (rule: configurable weights per ball
# state, CONFIRMED strongest ... LOST none) -- mirrors confidence.py's
# _ball_quality weighting philosophy, kept local so this module never needs
# to import or depend on anything from the frozen ball-tracking files beyond
# the plain BallTrackState objects it's handed.
STATE_WEIGHT = {"CONFIRMED": 1.0, "REACQUIRING": 0.85, "PREDICTED": 0.35, "OCCLUDED": 0.0, "LOST": 0.0}

# --- Receiver/reception evidence-gating (this task, rules #3-#15) ----------
#
# NEAR BALL != RECEIVED BALL. Before this task, the TEAMMATE branch of
# _find_reception (below) accepted the FIRST frame anywhere in the
# (potentially multi-second) candidate window where a same-kit teammate was
# merely within RECEPTION_RADIUS of the ball's reported position, with no
# regard at all for how reliable that ball position was at that instant.
# Once TeamClassifier correctly distinguishes TEAMMATE from OPPONENT (see
# colour.py / possession_detector.py -- frozen, correct, and NOT touched by
# this task), that plain proximity check became dangerous: when the ball
# tracker (also frozen -- rule #16 -- and NOT fixed here) enters an extended
# LOST / PREDICTED / REACQUIRING / OCCLUDED stretch, its reacquired or
# motion-model-predicted position can coincidentally land near a genuinely,
# correctly-classified TEAMMATE purely by chance, and that single low-
# confidence coincidence was enough to convert a real, contested SHOT into a
# false PASS (see this task's real-clip trace: known_match_shot_195_223 and
# second_match_carry_0_40 both false-PASS on exactly this pattern).
#
# This section replaces that single proximity check with temporally-gated
# reception evidence (_teammate_reception_evidence / _score_reception_at
# below): a candidate frame must show a genuine closing-distance approach,
# ball-state/ball-confidence reliability, a trajectory/velocity change
# consistent with actual contact, and (unless one-touch evidence is
# exceptionally strong -- rule #10) some persistence of control afterward,
# before TEAMMATE reception is RECEIVER_CONFIRMED rather than left as
# RECEIVER_REJECTED or a still-open RECEIVER_CANDIDATE (rule #4). Team
# identity and reception identity remain strictly separate (rule #13): this
# machinery never runs for, or is influenced by, TeamClassifier's own
# confidence -- only by what the ball itself did around the candidate frame.
BALL_STATE_RECEPTION_WEIGHT = {"CONFIRMED": 1.0, "REACQUIRING": 0.55, "PREDICTED": 0.25,
                                "OCCLUDED": 0.15, "LOST": 0.0}
# Rule #8, made a hard floor rather than a soft weight: a ball whose overall
# confidence has crashed below this -- even if the discrete state label
# still says REACQUIRING rather than LOST -- cannot, by itself, independently
# confirm a reception (this project's own trace data shows REACQUIRING
# frames with confidence as low as 0.2-0.3 immediately preceding total
# tracking collapse; the discrete label alone is not trustworthy enough,
# rule #7's own caution).
RECEPTION_LOST_CONFIDENCE_FLOOR = 0.15
RECEPTION_TREND_WINDOW = 4          # frames of prior distance history consulted for the "closing" signal
RECEPTION_PERSISTENCE_WINDOW = 4    # frames after the candidate consulted for the control-persistence signal
RECEPTION_CONFIRM_SCORE = 0.55      # multi-signal confirmation bar -- expects some persistence too (rule #9)
RECEPTION_ONE_TOUCH_SCORE = 0.72    # single-frame, no-persistence-required confirmation bar (rule #10)
RECEPTION_REJECT_SCORE = 0.30       # at/below this: RECEIVER_REJECTED outright, not merely a weak candidate
RECEPTION_MIN_PERSISTENCE_FOR_MULTI_SIGNAL = 0.35
RECEPTION_SCORE_WEIGHTS = {"ball_state": 0.30, "distance_trend": 0.25,
                            "trajectory_change": 0.20, "persistence": 0.25}

GOAL_LOOKAHEAD_S = 1.5

# Composite spatial+kinematic SHOT-evidence scoring (rule #11: no one signal
# is decisive -- every available signal below contributes an equal-weighted
# 0-1 vote, and a signal that isn't available (no pitch context, no keeper
# candidate) simply doesn't vote rather than being treated as evidence
# against). CROSS_PENALTY_WEIGHT subtracts a wide-release/lateral-arrival
# signal from the composite -- this project still cannot positively emit a
# CROSS label with confidence (see evaluate_release's docstring), but this
# lets a cross-shaped release's SHOT confidence be suppressed rather than
# defaulting to SHOT purely because it was fast and unreceived.
COMPOSITE_SHOT_THRESHOLD = 0.55
COMPOSITE_INCIDENTAL_THRESHOLD = 0.25
CROSS_PENALTY_WEIGHT = 0.3

# --- confidence-weighted evidence aggregation (evidence-fusion correction) ---
#
# SIGNAL SEMANTICS: speed / displacement / goal_direction / keeper_proximity
# are all POSITIVE evidence FOR shot (higher value = more shot-like); the
# wide-origin/CROSS score is kept separate as NEGATIVE evidence (evidence
# FOR a cross-style delivery), subtracted as its own penalty rather than
# blended into the positive average -- unchanged in spirit from before this
# task (rule: keep the wide-release penalty separate, use its existing
# score+confidence, do not redesign it).
#
# Every positive slot is (value, confidence, weight). A slot with NOTHING
# computable at all carries confidence=0.0 -- this is NOT "evidence against
# shot" (that would be value=0 at a REAL, non-zero confidence, e.g. a
# clearly-off-target trajectory) -- it is "this slot has nothing to say".
# Because confidence multiplies both the numerator and that slot's own
# contribution to the denominator below, confidence=0 makes a slot drop out
# of the composite untouched by its absence (rule #6/#7): missing evidence
# no longer behaves like negative evidence, and removing an unreliable slot
# does not inflate the remaining average, it just means the average is
# computed over less evidence -- which `evidence_coverage` (a SEPARATE
# denominator: every slot's weight, whether available or not) makes visible.
WEIGHT_SPEED = 1.0
WEIGHT_DISPLACEMENT = 1.0
WEIGHT_GOAL_DIRECTION = 1.0
WEIGHT_KEEPER = 1.0


def _log(log, frame_idx, reason, **extra):
    if log is None:
        return
    entry = {"frame": frame_idx, "reason": reason}
    entry.update(extra)
    log.append(entry)


def dynamic_timeout_frames(fps, release_speed):
    """Seconds-based, fps- and speed-aware timeout in place of a fixed frame
    count. A ball travels roughly the same amount of REAL TIME to reach a
    teammate, a defender, a keeper, or the net regardless of clip fps -- so
    the window is sized in seconds first, then converted to frames. A faster
    release gets a modestly longer allowance (it may need to travel farther,
    e.g. a long-range strike) but the bonus is capped so a candidate can
    never run away and swallow unrelated later play."""
    speed_bonus_s = 0.0
    if release_speed is not None and release_speed > 0:
        speed_bonus_s = min(SPEED_TIMEOUT_BONUS_CAP_S, release_speed / 40.0)
    total_s = min(MAX_TIMEOUT_S, max(MIN_TIMEOUT_S, BASE_TIMEOUT_S + speed_bonus_s))
    return max(6, int(round(total_s * fps)))


def _weighted_motion_evidence(ball_tracker, i_start, i_stop):
    """Walk frames i_start..i_stop (inclusive) and collect (frame, x, y,
    weight) for every sample that carries a real or motion-model position,
    weighted by how trustworthy that ball state is. OCCLUDED/LOST samples
    (no usable position at all, or a position we've explicitly decided not
    to trust) contribute nothing -- they're skipped, never synthesized."""
    samples = []
    hi = min(i_stop, len(ball_tracker.history) - 1)
    for idx in range(i_start, hi + 1):
        o = ball_tracker.history[idx]
        w = STATE_WEIGHT.get(o.state, 0.0)
        if w <= 0.0 or o.centre_x is None:
            continue
        samples.append((o.frame_number, o.centre_x, o.centre_y, w))
    return samples


def _straightness(samples):
    """net displacement / total path length over the samples -- 1.0 is a
    perfectly straight line, lower values mean the ball wandered, curved
    sharply, or reversed direction (consistent with a deflection/block
    changing its course, not necessarily inconsistent with a real shot's
    natural dip -- STRAIGHTNESS_MIN is deliberately forgiving)."""
    if len(samples) < 2:
        return None
    path = 0.0
    for (f0, x0, y0, w0), (f1, x1, y1, w1) in zip(samples, samples[1:]):
        path += ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
    net = ((samples[-1][1] - samples[0][1]) ** 2 + (samples[-1][2] - samples[0][2]) ** 2) ** 0.5
    if path <= 0:
        return None
    return min(1.0, net / path)


def _weighted_avg_speed(samples):
    """Weighted average px/frame speed across consecutive sample pairs --
    the SUSTAINED speed over the whole window, not just the instantaneous
    release-frame velocity, so a single noisy fast frame can't manufacture a
    shot and a single noisy slow frame can't erase one."""
    if len(samples) < 2:
        return None
    total_w, total_speed = 0.0, 0.0
    for (f0, x0, y0, w0), (f1, x1, y1, w1) in zip(samples, samples[1:]):
        if f1 == f0:
            continue
        d = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 / (f1 - f0)
        w = min(w0, w1)
        total_speed += d * w
        total_w += w
    return (total_speed / total_w) if total_w > 0 else None


def _speed_at(ball_tracker, k, window=3):
    vel = ball_tracker.velocity_at(k, window=window)
    if vel is None:
        return None
    return (vel[0] ** 2 + vel[1] ** 2) ** 0.5


def _peak_speed_in_window(ball_tracker, i_start, i_stop, vwindow=3):
    """Max instantaneous speed anywhere in [i_start, i_stop] -- used instead
    of the single instantaneous speed AT the release frame, since a real
    strike's peak speed can arrive a few frames after a possession segment
    is judged to have ended, not at that exact frame."""
    best = None
    hi = min(i_stop, len(ball_tracker.history) - 1)
    for k in range(i_start, hi + 1):
        s = _speed_at(ball_tracker, k, window=vwindow)
        if s is not None and (best is None or s > best):
            best = s
    return best


def _nearby_centre(ball_tracker, idx, max_back=6):
    """The ball's centre at frame idx, or -- if that exact frame has no
    position (OCCLUDED/LOST) -- the nearest earlier frame that does, within
    max_back frames. Used only to measure how far the ball actually
    travelled between two points in the window, never to fabricate a
    position where none exists."""
    for k in range(idx, max(-1, idx - max_back - 1), -1):
        if 0 <= k < len(ball_tracker.history):
            o = ball_tracker.history[k]
            if o.centre_x is not None:
                return (o.centre_x, o.centre_y)
    return None


def _ball_state_counts_window(ball_tracker, i0, i1):
    counts = {}
    hi = min(i1, len(ball_tracker.history) - 1)
    for o in ball_tracker.history[i0:hi + 1]:
        counts[o.state] = counts.get(o.state, 0) + 1
    return counts


def _composite_shot_evidence(peak_speed, net_displacement, gscore, goal_direction_confidence,
                              keeper_evidence, lateral_value, lateral_confidence):
    """Confidence-weighted aggregation replacing the old naive equal
    average (see the module-level comment above this function for the full
    semantics). Returns (composite, evidence_coverage, components):

      composite         -- 0-1 (or None if literally nothing was usable),
                            the SHOT-evidence score, compared against the
                            SAME COMPOSITE_SHOT_THRESHOLD/
                            COMPOSITE_INCIDENTAL_THRESHOLD as before this
                            task (those thresholds are unchanged).
      evidence_coverage  -- 0-1, how much of the FOUR possible signal slots
                            actually carried usable confidence, independent
                            of whether that evidence was favourable or not
                            (rule #7/#8) -- a diagnostic, not a new
                            classification gate (the PASS/SHOT/
                            UNKNOWN_RELEASE decision structure itself is
                            unchanged).
      components         -- full per-slot value/confidence/weight/
                            contribution breakdown, for evidence reporting
                            (rule #14).

    peak_speed / net_displacement / gscore: these three are treated exactly
        as they were BEFORE this task -- full weight whenever computable,
        excluded (not zeroed) when not. This is a deliberate, disclosed
        scope decision (rule #3 explicitly allows NOT applying the same
        confidence-weighting formula to every signal when semantics
        justify it): trialling pitch_context.pitch_confidence as
        goal_direction's aggregation confidence was tested and rejected --
        it systematically UNDER-weighted a clearly contradicting trajectory
        (gscore=0.0, i.e. the ball travelled exactly away from the
        estimated goal) whenever pitch_confidence was only moderate,
        letting confident-but-unrelated speed/displacement evidence
        outvote it and push borderline cases toward SHOT. That is exactly
        what rule #9 forbids ("genuine contradictory evidence... should
        still suppress SHOT strongly. Confidence weighting is NOT
        permission to ignore inconvenient evidence"), and it is not what
        this task's motivating bug was about -- the demonstrated,
        reported defect (rules #1/#5/#6/#12) was specifically the
        goalkeeper slot conflating identity confidence with distance, not
        goal_direction. So goal_direction's reliability is still reported
        (see goal_direction_confidence below, kept purely informational)
        but no longer used to scale its vote -- it participates in the
        weighted average exactly as speed/displacement do: fully, when a
        real trajectory-vs-goal measurement exists at all.
    goal_direction_confidence: pitch_context.pitch_confidence, reported in
        the evidence breakdown for transparency (how reliable the
        underlying goal-location estimate itself is) but NOT fed into the
        weighted average, for the reason above.
    keeper_evidence: a goalkeeper_context.KeeperEvidence(proximity_value,
        identity_confidence), or None if there is no keeper candidate at
        all. proximity_value is the VALUE, identity_confidence is the
        CONFIDENCE -- kept apart per rule #5, so a candidate we're barely
        sure is the keeper can no longer drag the composite around just
        because it happened to be near (or far from) the contest point.
    lateral_value / lateral_confidence: the wide-origin/CROSS raw value and
        its own confidence, straight from release_geometry.py (unchanged --
        rule #10), applied as a separate subtracted penalty exactly as
        before this task (CROSS_PENALTY_WEIGHT * value * confidence is
        arithmetically identical to the old CROSS_PENALTY_WEIGHT *
        lateral_wide_release_score(...), since that scalar already IS
        value*confidence)."""
    slots = {
        "speed": (min(1.0, peak_speed / RELEASE_FAST_SPEED) if peak_speed is not None else 0.0,
                  1.0 if peak_speed is not None else 0.0, WEIGHT_SPEED),
        "displacement": (min(1.0, net_displacement / PURPOSEFUL_DISPLACEMENT_MIN)
                          if net_displacement is not None else 0.0,
                          1.0 if net_displacement is not None else 0.0, WEIGHT_DISPLACEMENT),
        # confidence here is presence-based (1.0/0.0), matching pre-task
        # behaviour exactly -- see the docstring above for why
        # goal_direction_confidence (pitch_confidence) is reported but not
        # used to scale this vote.
        "goal_direction": (gscore if gscore is not None else 0.0,
                            1.0 if gscore is not None else 0.0,
                            WEIGHT_GOAL_DIRECTION),
        "keeper_proximity": (keeper_evidence.proximity_value if keeper_evidence is not None else 0.0,
                              keeper_evidence.identity_confidence if keeper_evidence is not None else 0.0,
                              WEIGHT_KEEPER),
    }

    weighted_sum = sum(value * conf * weight for value, conf, weight in slots.values())
    normalisation = sum(conf * weight for _, conf, weight in slots.values())
    total_weight = sum(weight for _, _, weight in slots.values())

    components = {}
    for name, (value, conf, weight) in slots.items():
        contribution = (value * conf * weight / normalisation) if normalisation > 0 else 0.0
        components[name] = {"value": round(value, 3), "confidence": round(conf, 3), "weight": weight,
                             "contribution": round(contribution, 3)}

    evidence_coverage = round(normalisation / total_weight, 3) if total_weight > 0 else 0.0

    if normalisation <= 0:
        # every slot carries zero confidence -- genuinely nothing usable to
        # aggregate (distinct from "aggregated evidence says no", which
        # would show up as a low, not absent, composite)
        components["wide_origin_penalty"] = {
            "value": round(lateral_value, 3) if lateral_value is not None else None,
            "confidence": round(lateral_confidence, 3) if lateral_value is not None else 0.0,
            "weight": CROSS_PENALTY_WEIGHT, "penalty_contribution": 0.0}
        return None, evidence_coverage, components

    positive_raw = weighted_sum / normalisation
    penalty = CROSS_PENALTY_WEIGHT * (lateral_value * lateral_confidence) if lateral_value is not None else 0.0
    composite = max(0.0, min(1.0, positive_raw - penalty))
    components["wide_origin_penalty"] = {
        "value": round(lateral_value, 3) if lateral_value is not None else None,
        "confidence": round(lateral_confidence, 3) if lateral_value is not None else 0.0,
        "weight": CROSS_PENALTY_WEIGHT, "penalty_contribution": round(-penalty, 3)}
    return round(composite, 3), evidence_coverage, components


@dataclass
class ReceiverEvidence:
    """Rule #15's structured reception-evidence object -- makes a PASS/
    non-PASS decision on a TEAMMATE-proximity candidate explainable, rather
    than a single opaque boolean.

    `candidate_player_id` and `team_confidence` are honestly reported as
    None: this pipeline carries no persistent per-box track identity across
    frames (a pre-existing, disclosed limitation -- see event_detector.py's
    `other_boxes`), and PossessionFrame carries only the discrete
    TEAMMATE/OPPONENT/UNKNOWN label, not a numeric TeamClassifier
    confidence score. Per rule #13, this is not a gap in the fix: team
    identity and reception identity are deliberately separate concerns, and
    `status` below is decided purely from the reception/control signals
    (distance trend, ball-state reliability, trajectory change,
    persistence) -- never from how confident TeamClassifier was."""
    candidate_player_id: Optional[str]
    team: Optional[str]
    team_confidence: Optional[float]
    minimum_ball_distance: Optional[float]
    closest_approach_frame: Optional[int]
    ball_state_at_contact: Optional[str]
    ball_localisation_confidence: Optional[float]
    distance_trend_score: Optional[float]
    trajectory_change_score: Optional[float]
    control_persistence_score: Optional[float]
    reception_score: float
    status: str   # "RECEIVER_CANDIDATE" | "RECEIVER_CONFIRMED" | "RECEIVER_REJECTED"

    def to_dict(self):
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def _distance_trend_score(possession_frames, i_end, k, window=RECEPTION_TREND_WINDOW):
    """0-1: how much the ball-to-nearest-other-player distance CLOSED in
    the frames immediately before candidate frame k, relative to the
    largest distance seen in that short lookback (rule #6: "distance
    closing -> closest approach" is the real-reception pattern, not merely
    "ball happens to be near a player"). No persistent per-box identity
    exists in this pipeline (see ReceiverEvidence's docstring), so this
    necessarily uses the reported nearest-other-player distance itself as
    the closing signal -- an approximation, but one that reads directly off
    real per-frame ball-to-player geometry rather than fabricated identity
    continuity."""
    d_k = possession_frames[k].nearest_other_distance if k < len(possession_frames) else None
    if d_k is None:
        return 0.0
    lo = max(i_end, k - window)
    prior = [possession_frames[idx].nearest_other_distance
             for idx in range(lo, k)
             if idx < len(possession_frames) and possession_frames[idx].nearest_other_distance is not None]
    if not prior:
        # No prior distance history to compare against (e.g. the candidate
        # is the very first frame of the window -- a genuine one-touch
        # reception right at release). Fall back to a weak, capped signal
        # from raw proximity alone -- close counts for something, but never
        # as much as an observed, multi-frame closing approach would.
        return max(0.0, min(1.0, 1.0 - d_k / RECEPTION_RADIUS)) * 0.5
    max_prior = max(prior)
    if max_prior <= 0:
        return 0.0
    return max(0.0, min(1.0, (max_prior - d_k) / max_prior))


def _nearby_centre_forward(ball_tracker, idx, max_fwd=6):
    """Mirror of _nearby_centre, searching FORWARD instead of backward --
    the nearest frame at/after idx with a usable ball position. Kept as a
    separate, additive helper rather than adding a direction parameter to
    the existing _nearby_centre (used elsewhere in this module and left
    untouched) to keep this task's diff minimal and unambiguous."""
    hi = len(ball_tracker.history)
    for k in range(idx, min(hi, idx + max_fwd + 1)):
        o = ball_tracker.history[k]
        if o.centre_x is not None:
            return k, (o.centre_x, o.centre_y)
    return None


def _trajectory_change_score(ball_tracker, k, i_end, hi_cap, half_window=3):
    """0-1: how much the ball's velocity (speed and/or direction) changed
    around candidate frame k -- rule #6's "ball trajectory/control changes"
    signal, and rule #10's one-touch signal ("sharp ball-direction/velocity
    change at teammate"). A ball that merely continues past/through a
    player's position unchanged in speed and heading is not exhibiting a
    contact/control event; one that slows and/or bends sharply right at the
    candidate frame is."""
    # NOTE: before_idx is deliberately NOT floored at i_end. The ball's own
    # velocity immediately before the release frame already happened and is
    # legitimate evidence (this module's release_speed itself is computed
    # the same way, via ball_tracker.velocity_at(i_end, window=4) reaching
    # back before i_end) -- it is not "looking into the future" the way
    # reading frames after k would be (rule #9's real-time-system framing
    # only forbids the latter). Without this, a genuine one-touch reception
    # that happens to land exactly ON i_end itself (the release frame) --
    # a real pattern in this project's own labelled clips -- could never
    # show a trajectory-change score at all, since there would be no
    # in-window frame before it to compare against.
    before_idx = max(0, k - half_window)
    after_idx = min(hi_cap, k + half_window)
    if before_idx >= k or after_idx <= k:
        return 0.0
    before_pt = _nearby_centre(ball_tracker, before_idx)
    at_pt = _nearby_centre(ball_tracker, k)
    after = _nearby_centre_forward(ball_tracker, after_idx)
    if before_pt is None or at_pt is None or after is None:
        return 0.0
    _after_idx, after_pt = after
    v1 = (at_pt[0] - before_pt[0], at_pt[1] - before_pt[1])
    v2 = (after_pt[0] - at_pt[0], after_pt[1] - at_pt[1])
    s1, s2 = math.hypot(*v1), math.hypot(*v2)
    if s1 <= 1e-6 and s2 <= 1e-6:
        return 0.0
    speed_drop = max(0.0, min(1.0, (s1 - s2) / s1)) if s1 > 1e-6 else 0.0
    if s1 > 1e-6 and s2 > 1e-6:
        cos_ang = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (s1 * s2)))
        direction_change = math.acos(cos_ang) / math.pi
    else:
        direction_change = 0.0
    return max(0.0, min(1.0, 0.6 * speed_drop + 0.4 * direction_change))


def _control_persistence_score(ball_tracker, k, hi_cap, window=RECEPTION_PERSISTENCE_WINDOW):
    """0-1: average ball-state-reliability weight (scaled by that frame's
    own confidence, where the ball-state object models one -- fakes/tests
    that don't model `confidence` fall back to trusting the discrete state
    label alone, rule #7) over the frames immediately AFTER candidate frame
    k. Rule #8's core case: a ball that is reacquired near a teammate and
    then immediately collapses back into LOST/near-zero-confidence (this
    task's own real-clip trace: both false-PASS frames are followed within
    1-2 frames by total tracking collapse) shows NO control persistence at
    all, and must not read as a confirmed reception on proximity alone."""
    hi = min(hi_cap, k + window, len(ball_tracker.history) - 1)
    if hi <= k:
        return 0.0
    weights = []
    for idx in range(k + 1, hi + 1):
        o = ball_tracker.history[idx]
        conf = getattr(o, "confidence", None)
        conf_factor = 1.0 if conf is None else max(0.0, min(1.0, conf))
        weights.append(BALL_STATE_RECEPTION_WEIGHT.get(o.state, 0.0) * conf_factor)
    return max(0.0, min(1.0, sum(weights) / len(weights))) if weights else 0.0


def _score_reception_at(ball_tracker, possession_frames, i_end, hi_cap, k):
    """Combine the four rule-#6 signals (ball-state/confidence reliability,
    distance-closing trend, trajectory/velocity change, post-approach
    control persistence) into a single 0-1 reception_score for candidate
    frame k, with rule #8's hard floor applied last: a functionally-LOST
    ball (LOST state, or confidence below RECEPTION_LOST_CONFIDENCE_FLOOR)
    cannot independently confirm a reception no matter how favourably the
    other signals happen to read."""
    pf = possession_frames[k] if k < len(possession_frames) else None
    bts = ball_tracker.history[k] if k < len(ball_tracker.history) else None
    if pf is None or bts is None:
        return None

    ball_state_label = bts.state
    state_w = BALL_STATE_RECEPTION_WEIGHT.get(ball_state_label, 0.0)
    ball_conf = getattr(bts, "confidence", None)
    if ball_conf is None:
        # No modeled confidence (e.g. a lightweight test fake) -- fall back
        # to trusting the discrete state label alone rather than crashing
        # or silently zeroing this slot out.
        ball_reliability = state_w
        effectively_lost = (ball_state_label == "LOST")
    else:
        ball_conf = max(0.0, min(1.0, ball_conf))
        ball_reliability = 0.5 * state_w + 0.5 * ball_conf
        effectively_lost = (ball_state_label == "LOST") or (ball_conf < RECEPTION_LOST_CONFIDENCE_FLOOR)

    distance_trend = _distance_trend_score(possession_frames, i_end, k)
    trajectory_change = _trajectory_change_score(ball_tracker, k, i_end, hi_cap)
    persistence = _control_persistence_score(ball_tracker, k, hi_cap)

    score = (RECEPTION_SCORE_WEIGHTS["ball_state"] * ball_reliability +
             RECEPTION_SCORE_WEIGHTS["distance_trend"] * distance_trend +
             RECEPTION_SCORE_WEIGHTS["trajectory_change"] * trajectory_change +
             RECEPTION_SCORE_WEIGHTS["persistence"] * persistence)

    if effectively_lost:
        # Rule #8, hard floor: "If ball lost -> ball reacquired near
        # teammate, do NOT assume teammate received ball." No combination
        # of the other three signals may push a functionally-lost-ball
        # frame into confirmation territory.
        score = min(score, 0.2)

    score = max(0.0, min(1.0, score))
    return {
        "frame": k,
        "team": pf.nearest_other_team,
        "minimum_ball_distance": pf.nearest_other_distance,
        "ball_state_at_contact": ball_state_label,
        "ball_localisation_confidence": ball_conf,
        "distance_trend_score": distance_trend,
        "trajectory_change_score": trajectory_change,
        "control_persistence_score": persistence,
        "reception_score": score,
        "effectively_lost_ball": effectively_lost,
    }


def _teammate_reception_evidence(ball_tracker, possession_frames, i_end, hi_cap):
    """Replaces the old "first TEAMMATE proximity anywhere in window =
    reception" check (rules #3-#10). Scores EVERY frame in the window where
    a TEAMMATE is within RECEPTION_RADIUS of the ball (not just the first),
    keeps the best-evidenced one, and classifies it RECEIVER_CONFIRMED /
    RECEIVER_CANDIDATE / RECEIVER_REJECTED from temporal control evidence --
    never from proximity alone. Returns None when no TEAMMATE was ever
    within range anywhere in the window (nothing to evaluate -- the caller
    falls straight through to the unchanged non-teammate path)."""
    best = None
    for k in range(i_end, hi_cap + 1):
        pf = possession_frames[k] if k < len(possession_frames) else None
        if pf is None or pf.nearest_other_distance is None or pf.nearest_other_distance > RECEPTION_RADIUS:
            continue
        if pf.nearest_other_team != "TEAMMATE":
            continue
        scored = _score_reception_at(ball_tracker, possession_frames, i_end, hi_cap, k)
        if scored is None:
            continue
        if best is None or scored["reception_score"] > best["reception_score"]:
            best = scored

    if best is None:
        return None

    score = best["reception_score"]
    one_touch_ok = score >= RECEPTION_ONE_TOUCH_SCORE                    # rule #10
    multi_signal_ok = (score >= RECEPTION_CONFIRM_SCORE and
                       best["control_persistence_score"] >= RECEPTION_MIN_PERSISTENCE_FOR_MULTI_SIGNAL)  # rule #9
    if one_touch_ok or multi_signal_ok:
        status = "RECEIVER_CONFIRMED"
    elif score <= RECEPTION_REJECT_SCORE:
        status = "RECEIVER_REJECTED"
    else:
        status = "RECEIVER_CANDIDATE"   # rule #4/#11: genuinely ambiguous -- not confirmed, not outright rejected

    return ReceiverEvidence(
        candidate_player_id=None,
        team="TEAMMATE",
        team_confidence=None,
        minimum_ball_distance=best["minimum_ball_distance"],
        closest_approach_frame=best["frame"],
        ball_state_at_contact=best["ball_state_at_contact"],
        ball_localisation_confidence=best["ball_localisation_confidence"],
        distance_trend_score=best["distance_trend_score"],
        trajectory_change_score=best["trajectory_change_score"],
        control_persistence_score=best["control_persistence_score"],
        reception_score=score,
        status=status,
    )


def _find_nonteammate_closest_approach(possession_frames, i_end, hi_cap):
    """The GLOBAL closest approach within the window among OPPONENT/colour-
    UNKNOWN candidates ONLY. TEAMMATE-labeled frames are deliberately
    excluded here (rule #8/#21 safeguard, this task): TEAMMATE reception is
    now decided exclusively by _teammate_reception_evidence's temporal-
    control scoring, called separately in evaluate_release. If a TEAMMATE
    proximity frame were still allowed to leak into this fallback as "the
    closest approach", a REJECTED/CANDIDATE teammate reception could still
    be reported back as team_at_reception == "TEAMMATE" via this path and
    silently reopen the exact bug this task fixes -- so this function must
    never return "TEAMMATE".

    For everyone else, this is otherwise UNCHANGED from before this task:
    the GLOBAL closest approach within the window (the frame where the
    nearest-other-player distance bottoms out), not the first crossing --
    what to make of that arrival (a genuine misplaced pass vs. a contested/
    blocked/saved shot) is still decided separately in evaluate_release
    using speed AND displacement evidence (rule #21: SHOT thresholds/
    evidence weights are frozen, untouched by this task)."""
    best_d, best_k, best_team = None, None, None
    for k in range(i_end, hi_cap + 1):
        pf = possession_frames[k] if k < len(possession_frames) else None
        if pf is None or pf.nearest_other_distance is None or pf.nearest_other_distance > RECEPTION_RADIUS:
            continue
        if pf.nearest_other_team == "TEAMMATE":
            continue
        if best_d is None or pf.nearest_other_distance < best_d:
            best_d, best_k, best_team = pf.nearest_other_distance, k, pf.nearest_other_team
    return best_team, best_d, best_k


def evaluate_release(ball_tracker, possession_frames, i_end, fps=25.0, log=None,
                      pitch_context=None, keeper_candidate=None):
    """i_end = index of the last in-possession frame of a segment. Returns
    an Event of type PASS, SHOT, or UNKNOWN_RELEASE.

    `pitch_context`/`keeper_candidate` (both optional, from pitch_context.py
    / goalkeeper_context.py): read-only spatial context computed once per
    clip, AFTER the per-frame loop, from data that loop already produced.
    IMPORTANT (explicit instruction: do not loosen thresholds as a
    workaround): the original speed/displacement gates below (`purposeful`,
    `clearly_incidental`) are UNCHANGED from before spatial context existed
    -- a case that was confidently SHOT or confidently PASS before still is.
    Spatial context is used ONLY to attempt to resolve the cases that were,
    and without it still are, genuinely ambiguous, and to apply an
    honestly-motivated CROSS/wide-release penalty that can turn a marginal
    SHOT into UNKNOWN_RELEASE when the release looks more like a delivery
    into the box than a strike at goal. When no spatial context is
    available (pitch_context is None, or its own confidence is 0 because
    attack direction or the goalkeeper candidate couldn't be estimated),
    behaviour is IDENTICAL to before this task -- ambiguous stays
    UNKNOWN_RELEASE, exactly as it did.

    KNOWN, DISCLOSED LIMITATION (CROSS): CROSS is reserved in the type
    vocabulary but never emitted as its own label. A hard, flat, driven
    cross with no immediate receiver can still look like a shot on every
    signal available here; the lateral/wide-release penalty above reduces
    (does not eliminate) this risk. This system has no true pitch homography
    or detected goal mouth (see pitch_context.py), only a coarse positional
    proxy -- so this remains a best-effort mitigation, not a guarantee."""
    n = len(ball_tracker.history)
    vel = ball_tracker.velocity_at(i_end, window=4)
    release_speed = (vel[0] ** 2 + vel[1] ** 2) ** 0.5 if vel else None

    early_horizon = max(6, int(round(EARLY_SPEED_HORIZON_S * fps)))
    peak_speed = _peak_speed_in_window(ball_tracker, i_end, min(n - 1, i_end + early_horizon))
    release_is_fast = peak_speed is not None and peak_speed >= RELEASE_FAST_SPEED

    timeout_frames = dynamic_timeout_frames(fps, peak_speed if peak_speed is not None else release_speed)
    hi_cap = min(n - 1, i_end + timeout_frames)

    evidence = {
        "release_speed_px_per_frame": round(release_speed, 1) if release_speed is not None else None,
        "peak_release_speed_px_per_frame": round(peak_speed, 1) if peak_speed is not None else None,
        "candidate_timeout_frames": timeout_frames,
        "candidate_timeout_s": round(timeout_frames / fps, 2),
    }

    evidence["ball_states_in_window"] = _ball_state_counts_window(ball_tracker, i_end, hi_cap)

    # rules #3-#15: TEAMMATE reception is no longer "first proximity in
    # window wins" -- it requires temporal control evidence. This is
    # evaluated FIRST and separately from the (unchanged) non-teammate path
    # below, so a rejected/candidate teammate proximity can never leak back
    # in as a false reception via that path's fallback (see
    # _find_nonteammate_closest_approach's docstring).
    receiver_evidence = _teammate_reception_evidence(ball_tracker, possession_frames, i_end, hi_cap)
    if receiver_evidence is not None:
        evidence["receiver_evidence"] = receiver_evidence.to_dict()

    if receiver_evidence is not None and receiver_evidence.status == "RECEIVER_CONFIRMED":
        evidence["reception_distance_px"] = (round(receiver_evidence.minimum_ball_distance, 1)
                                              if receiver_evidence.minimum_ball_distance is not None else None)
        evidence["reception_team"] = "TEAMMATE"
        evidence["note"] = ("ball reached a same-kit player within the dynamic candidate window, with temporal "
                             "control evidence (closing distance, ball-state reliability, trajectory change, "
                             "and/or post-approach persistence -- not mere proximity) confirming a genuine "
                             f"reception; reception_score={receiver_evidence.reception_score}")
        evidence["resolution_reason"] = "PASS_CONFIRMED: teammate_receiver"
        _log(log, i_end, "PASS_CONFIRMED: teammate_receiver", release_frame=i_end,
             reception_score=receiver_evidence.reception_score)
        return Event("PASS", i_end, receiver_evidence.closest_approach_frame, success=True, evidence=evidence)

    # TEAMMATE proximity existed but did NOT confirm as a genuine reception
    # (RECEIVER_CANDIDATE or RECEIVER_REJECTED) -- rule #3: NEAR BALL !=
    # RECEIVED BALL. Fall through to the unchanged non-teammate resolution
    # logic, which never reconsiders TEAMMATE frames, so a rejected/
    # candidate teammate proximity is free to correctly resolve as SHOT,
    # UNKNOWN_RELEASE, or an unresolved release exactly as if no nearby
    # player existed at all (rules #11/#12: the SHOT path stays fully
    # alive and is never biased against by an unconfirmed receiver
    # candidate).
    team_at_reception, dist_at_reception, frame_at_reception = _find_nonteammate_closest_approach(
        possession_frames, i_end, hi_cap)

    evidence["reception_distance_px"] = round(dist_at_reception, 1) if dist_at_reception is not None else None
    evidence["reception_team"] = team_at_reception

    if team_at_reception is not None:   # OPPONENT or colour-UNKNOWN closest approach
        release_pos = _nearby_centre(ball_tracker, i_end)
        contest_pos = _nearby_centre(ball_tracker, frame_at_reception)
        net_displacement = (dist(release_pos, contest_pos) if (release_pos is not None and contest_pos is not None)
                             else None)
        evidence["release_to_contest_displacement_px"] = round(net_displacement, 1) if net_displacement is not None else None

        purposeful = (peak_speed is not None and peak_speed >= PURPOSEFUL_SPEED_MIN) or \
                     (net_displacement is not None and net_displacement >= PURPOSEFUL_DISPLACEMENT_MIN)
        clearly_incidental = (peak_speed is not None and peak_speed < CLEARLY_INCIDENTAL_SPEED) and \
                              (net_displacement is None or net_displacement < CLEARLY_INCIDENTAL_DISPLACEMENT)

        if purposeful:
            evidence["note"] = ("a non-teammate (defender/keeper) was the closest approach within the window, "
                                 "but the release itself reads as purposeful (fast peak speed and/or "
                                 "substantial displacement toward them) -- read as a contested/blocked/saved "
                                 "SHOT, not a failed pass. On-target/saved/blocked/scored outcome remains out "
                                 "of scope without pitch/goal-line calibration")
            evidence["resolution_reason"] = "SHOT_CONFIRMED: contested_by_opponent_during_purposeful_release"
            _log(log, i_end, "SHOT_CONFIRMED: contested_by_opponent_during_purposeful_release", release_frame=i_end)
            return Event("SHOT", i_end, frame_at_reception, success=None, evidence=evidence)

        if clearly_incidental:
            outcome_success = False if team_at_reception == "OPPONENT" else None
            evidence["note"] = ("ball reached a non-teammate at low speed/short displacement -- reads as a "
                                 "genuinely slow, short misplaced touch/pass, not a shot")
            evidence["resolution_reason"] = "PASS_CONFIRMED: opponent_settled_reception" if outcome_success is False \
                else "PASS_UNDETERMINED: unclassified_settled_reception"
            _log(log, i_end, evidence["resolution_reason"], release_frame=i_end)
            return Event("PASS", i_end, frame_at_reception, success=outcome_success, evidence=evidence)

        # neither clearly a purposeful strike NOR clearly a slow, short
        # touch on speed/displacement alone -- this is exactly the zone
        # spatial context (when available) now gets a chance to resolve,
        # rather than defaulting straight to UNKNOWN_RELEASE
        gscore = goal_direction_score(release_pos, [contest_pos], pitch_context)
        # kp: unchanged legacy blended value, kept ONLY for the
        # goalkeeper_probability_at_contest evidence field (backward-compat
        # display, matches the debug-overlay's "GK? p=..." convention) --
        # no longer what the composite itself consumes.
        kp = goalkeeper_probability_at(contest_pos, keeper_candidate) if keeper_candidate is not None else 0.0
        keeper_ev = keeper_evidence_at(contest_pos, keeper_candidate) if keeper_candidate is not None else None
        lateral_score = lateral_wide_release_score(release_pos, pitch_context)
        wide_est = estimate_release_width(release_pos, pitch_context)   # raw (value, confidence) pair for aggregation
        goal_direction_confidence = pitch_context.pitch_confidence if pitch_context is not None else 0.0
        # gate on GENUINE spatial evidence, not just "speed/displacement
        # produced a component" -- speed and displacement are already fully
        # accounted for by `purposeful`/`clearly_incidental` above (rule:
        # do not loosen those thresholds as a workaround). Composite
        # resolution below only ever runs when real spatial context (a
        # usable goal-direction score, or a keeper candidate with SOME
        # identity confidence -- not just "a candidate object exists")
        # actually exists; otherwise this falls straight through to the
        # same UNKNOWN_RELEASE this case always got. This is the same gate
        # as before this task (kp > 0 was already, in effect, "identity
        # confidence > 0", since kp = identity_confidence * a strictly-
        # positive exponential decay) -- unchanged decision boundary, only
        # the underlying value it reads is now split apart.
        has_spatial_evidence = gscore is not None or (keeper_ev is not None and keeper_ev.identity_confidence > 0)
        composite, evidence_coverage, components = (
            _composite_shot_evidence(peak_speed, net_displacement, gscore, goal_direction_confidence,
                                      keeper_ev, wide_est.wide_score, wide_est.confidence)
            if has_spatial_evidence else (None, 0.0, {}))
        # composite can still legitimately come back None even when
        # has_spatial_evidence gated True (e.g. gscore is not None but
        # pitch_confidence is exactly 0.0 AND there is no keeper evidence
        # either -- every slot's confidence is 0, so there is truly nothing
        # to aggregate) -- guard the threshold comparisons on the actual
        # returned value, not just the earlier gate, so a None never reaches
        # a numeric comparison.
        composite_available = composite is not None

        evidence["goal_direction_score"] = gscore
        evidence["goal_direction_confidence"] = goal_direction_confidence
        evidence["goalkeeper_probability_at_contest"] = kp
        evidence["keeper_identity_confidence"] = keeper_ev.identity_confidence if keeper_ev is not None else None
        evidence["keeper_proximity_value"] = keeper_ev.proximity_value if keeper_ev is not None else None
        evidence["lateral_wide_release_score"] = lateral_score
        evidence["distance_to_goal_px"] = distance_to_goal(release_pos, pitch_context)
        evidence["attack_direction"] = pitch_context.attack_direction if pitch_context is not None else "UNKNOWN"
        evidence["composite_shot_evidence"] = composite
        evidence["evidence_coverage"] = evidence_coverage
        evidence["composite_components"] = components

        if composite_available and composite >= COMPOSITE_SHOT_THRESHOLD:
            evidence["note"] = ("a non-teammate was the closest approach at a pace/displacement that speed and "
                                 "displacement alone couldn't resolve, but goal-direction/goalkeeper-proximity "
                                 "spatial context pushes this toward a contested/blocked/saved SHOT")
            evidence["resolution_reason"] = "SHOT_CONFIRMED: resolved_by_pitch_context"
            _log(log, i_end, "SHOT_CONFIRMED: resolved_by_pitch_context", release_frame=i_end,
                 composite=round(composite, 3))
            return Event("SHOT", i_end, frame_at_reception, success=None, evidence=evidence)

        if composite_available and composite <= COMPOSITE_INCIDENTAL_THRESHOLD:
            outcome_success = False if team_at_reception == "OPPONENT" else None
            evidence["note"] = ("a non-teammate was the closest approach, and spatial context (weak/negative "
                                 "goal-direction, no goalkeeper-proximity, and/or a wide-release signal) points "
                                 "away from a shot -- read as a genuinely incidental touch/pass")
            evidence["resolution_reason"] = "PASS_CONFIRMED: resolved_by_pitch_context" if outcome_success is False \
                else "PASS_UNDETERMINED: resolved_by_pitch_context"
            _log(log, i_end, evidence["resolution_reason"], release_frame=i_end, composite=round(composite, 3))
            return Event("PASS", i_end, frame_at_reception, success=outcome_success, evidence=evidence)

        # spatial context either unavailable or itself inconclusive -- this
        # project has no goalkeeper/role IDENTITY, no true pitch calibration,
        # so there is no honest basis to force this either way
        evidence["note"] = ("a non-teammate was the closest approach within the window, at a pace/displacement "
                             "that is genuinely ambiguous between a contested/challenged pass and a "
                             "blocked/saved shot, and available spatial context (if any) did not resolve it "
                             "either -- distinguishing the two with certainty would need true pitch calibration "
                             "and real goalkeeper identity, both out of scope for this system")
        evidence["resolution_reason"] = "UNKNOWN_RELEASE: ambiguous_opponent_contest"
        _log(log, i_end, "UNKNOWN_RELEASE: ambiguous_opponent_contest", release_frame=i_end,
             composite=composite if composite_available else None)
        return Event("UNKNOWN_RELEASE", i_end, frame_at_reception, success=None, evidence=evidence)

    # no reception resolved the candidate -- accumulate temporal evidence
    # over the FULL dynamic window rather than judging the release instant
    samples = _weighted_motion_evidence(ball_tracker, i_end, hi_cap)
    evidence_mass = sum(w for (_, _, _, w) in samples)
    straightness = _straightness(samples)
    sustained_speed = _weighted_avg_speed(samples)

    evidence["evidence_mass"] = round(evidence_mass, 2)
    evidence["straightness"] = round(straightness, 2) if straightness is not None else None
    evidence["sustained_speed_px_per_frame"] = round(sustained_speed, 1) if sustained_speed is not None else None

    is_fast = release_is_fast or (sustained_speed is not None and sustained_speed >= RELEASE_FAST_SPEED)

    if not is_fast:
        evidence["note"] = "no reception found and release was not fast -- outcome not observed"
        evidence["resolution_reason"] = "PASS_UNDETERMINED: slow_release_no_reception"
        _log(log, i_end, "PASS_UNDETERMINED: slow_release_no_reception", release_frame=i_end)
        return Event("PASS", i_end, hi_cap, success=None, evidence=evidence)

    if evidence_mass < MIN_EVIDENCE_MASS:
        evidence["note"] = ("fast release but too little trustworthy ball evidence (CONFIRMED/REACQUIRING/"
                             "PREDICTED samples) accumulated during the candidate window to confirm a shot")
        evidence["resolution_reason"] = "REJECTED_SHOT: insufficient_release_velocity"
        _log(log, i_end, "REJECTED_SHOT: insufficient_release_velocity", release_frame=i_end)
        return Event("UNKNOWN_RELEASE", i_end, hi_cap, success=None, evidence=evidence)

    if straightness is not None and straightness < STRAIGHTNESS_MIN:
        evidence["note"] = ("fast release but the trajectory changed direction sharply during the window -- "
                             "reads as deflected/broken up rather than a clean strike; not confident enough "
                             "to call SHOT")
        evidence["resolution_reason"] = "REJECTED_SHOT: trajectory_not_goal_directed"
        _log(log, i_end, "REJECTED_SHOT: trajectory_not_goal_directed", release_frame=i_end)
        return Event("UNKNOWN_RELEASE", i_end, hi_cap, success=None, evidence=evidence)

    release_pos_nr = _nearby_centre(ball_tracker, i_end)
    end_pos_nr = samples[-1][1:3] if samples else None
    gscore_nr = goal_direction_score(release_pos_nr, [end_pos_nr] if end_pos_nr else [], pitch_context)
    lateral_score_nr = lateral_wide_release_score(release_pos_nr, pitch_context)
    evidence["goal_direction_score"] = gscore_nr
    evidence["lateral_wide_release_score"] = lateral_score_nr
    evidence["distance_to_goal_px"] = distance_to_goal(release_pos_nr, pitch_context)
    evidence["attack_direction"] = pitch_context.attack_direction if pitch_context is not None else "UNKNOWN"

    if evidence_mass < CONFIRM_EVIDENCE_MASS:
        # borderline evidence mass -- speed/straightness alone couldn't
        # confirm it within the dynamic window. If spatial context is
        # available and clearly supports (or clearly contradicts) a shot,
        # let it resolve this rather than defaulting straight to
        # UNKNOWN_RELEASE (the original evidence_mass/straightness gates
        # above are UNCHANGED -- this only ever applies to what already
        # fell through them).
        if gscore_nr is not None:
            if gscore_nr >= 0.7 and (lateral_score_nr is None or lateral_score_nr < 0.5):
                evidence["note"] = ("insufficient evidence mass alone, but the ball's travel direction strongly "
                                     "and consistently points at the estimated goal -- resolved as SHOT by "
                                     "spatial context")
                evidence["resolution_reason"] = "SHOT_CONFIRMED: resolved_by_pitch_context"
                _log(log, i_end, "SHOT_CONFIRMED: resolved_by_pitch_context", release_frame=i_end,
                     goal_direction_score=gscore_nr)
                return Event("SHOT", i_end, hi_cap, success=None, evidence=evidence)
            if gscore_nr <= 0.3:
                evidence["note"] = ("insufficient evidence mass, and the ball's travel direction does not point "
                                     "toward the estimated goal -- reads as a release that was not aimed at goal")
                evidence["resolution_reason"] = "UNKNOWN_RELEASE: not_goal_directed_per_pitch_context"
                _log(log, i_end, "UNKNOWN_RELEASE: not_goal_directed_per_pitch_context", release_frame=i_end,
                     goal_direction_score=gscore_nr)
                return Event("UNKNOWN_RELEASE", i_end, hi_cap, success=None, evidence=evidence)
        evidence["note"] = ("fast, reasonably straight release, but the candidate window closed (dynamic "
                             "timeout) before enough trustworthy evidence accumulated to confirm a shot with "
                             "the required margin, and available spatial context (if any) did not resolve it "
                             "either")
        evidence["resolution_reason"] = "REJECTED_SHOT: candidate_timeout"
        _log(log, i_end, "REJECTED_SHOT: candidate_timeout", release_frame=i_end)
        return Event("UNKNOWN_RELEASE", i_end, hi_cap, success=None, evidence=evidence)

    # would otherwise be a confirmed SHOT on speed/straightness/evidence-mass
    # alone -- apply the disclosed, non-decisive CROSS/wide-release penalty
    # (section 11/17: reduce, don't eliminate, cross-into-SHOT misreads)
    # before finalizing it. A release with no lateral signal available (no
    # pitch context) is UNCHANGED from before this task.
    if lateral_score_nr is not None and lateral_score_nr >= 0.65 and (gscore_nr is None or gscore_nr < 0.6):
        evidence["note"] = ("fast, sustained, direction-consistent release with no receiver found -- but the "
                             "release origin reads as wide/lateral relative to the estimated goal and the "
                             "trajectory doesn't clearly point at goal either, which is more consistent with a "
                             "driven cross/delivery into the box than a strike at goal -- downgraded from SHOT")
        evidence["resolution_reason"] = "UNKNOWN_RELEASE: cross_shaped_release_suppressed_shot"
        _log(log, i_end, "UNKNOWN_RELEASE: cross_shaped_release_suppressed_shot", release_frame=i_end,
             lateral_wide_release_score=lateral_score_nr)
        return Event("UNKNOWN_RELEASE", i_end, hi_cap, success=None, evidence=evidence)

    evidence["note"] = ("fast, sustained, direction-consistent release with no receiving player identified "
                         "anywhere in the dynamic candidate window -- read as a shot. On-target/saved/blocked/"
                         "scored outcome is out of scope without pitch/goal-line calibration, except for the "
                         "always-review-required GOAL follow-up signal computed separately")
    evidence["resolution_reason"] = "SHOT_CONFIRMED"
    _log(log, i_end, "SHOT_CONFIRMED", release_frame=i_end, evidence_mass=round(evidence_mass, 2))
    return Event("SHOT", i_end, hi_cap, success=None, evidence=evidence)


def evaluate_goal_followup(shot_event, ball_tracker, possession_frames, fps=25.0, log=None, pitch_context=None):
    """Basic GOAL follow-up, gated strictly behind an already-confirmed SHOT
    event (never evaluated for PASS/UNKNOWN_RELEASE).

    HONESTLY LIMITED BY DESIGN: this project has no pitch/goal-line
    calibration, and this task explicitly forbids adding camera/homography
    work. There is therefore no way to actually detect a ball crossing a
    goal line. This function NEVER asserts a positive, confident GOAL. It
    either:
      - returns None (no follow-up event at all) when there's no
        circumstantial signal worth flagging -- silence, not a guessed
        "NO_GOAL", exactly like the rest of this project's "insufficient
        evidence -> say nothing / mark not-determinable" pattern, or
      - returns a GOAL Event with success=None (always), built only from
        circumstantial motion evidence (ball stayed fast/straight through
        the end of the SHOT window, then became untrackable (LOST/OCCLUDED)
        without ever reappearing back in ordinary play afterward). Because
        success is always None, ConfidenceEngine's existing OUTCOME_DRIVEN
        rule (review_required = band != HIGH or success is None) makes this
        ALWAYS review_required, by construction -- there is no code path
        that lets a GOAL come out looking confidently confirmed.
      - a shot that is blocked/saved and stays in visible play (ball
        reappears CONFIRMED/REACQUIRING again afterward) correctly produces
        no GOAL follow-up at all, so a blocked shot remains just a SHOT
        (rule: blocked shot still classifies SHOT).
    """
    if shot_event.type != "SHOT":
        return None

    i_end = shot_event.end_frame
    n = len(ball_tracker.history)
    hi = min(n - 1, i_end + int(round(GOAL_LOOKAHEAD_S * fps)))
    if hi <= i_end:
        return None

    samples = _weighted_motion_evidence(ball_tracker, shot_event.start_frame, i_end)
    straightness = _straightness(samples)

    reappeared_in_play = False
    for k in range(i_end + 1, hi + 1):
        o = ball_tracker.history[k]
        if o.state in ("CONFIRMED", "REACQUIRING") and o.centre_x is not None:
            reappeared_in_play = True
            break

    tail = ball_tracker.history[max(shot_event.start_frame, i_end - 3):i_end + 1]
    disappeared_at_end = bool(tail) and tail[-1].state in ("LOST", "OCCLUDED")

    if reappeared_in_play:
        _log(log, i_end, "GOAL_FOLLOWUP: ball_returned_to_play_no_goal_hypothesis", release_frame=i_end)
        return None
    if not disappeared_at_end:
        return None

    release_pos = _nearby_centre(ball_tracker, shot_event.start_frame)
    end_pos = _nearby_centre(ball_tracker, i_end)
    gscore = goal_direction_score(release_pos, [end_pos] if end_pos else [], pitch_context)

    evidence = {
        "goal_followup_window_frames": hi - i_end,
        "trajectory_straightness_during_shot": round(straightness, 2) if straightness is not None else None,
        "ball_reappeared_in_play_after_shot": reappeared_in_play,
        "ball_untracked_at_end_of_shot_window": disappeared_at_end,
        "goal_direction_score": gscore,
        "distance_to_goal_px": distance_to_goal(end_pos, pitch_context),
        "resolution_reason": "GOAL_FOLLOWUP: circumstantial_only_review_required",
        "note": ("heuristic-only signal: no pitch/goal-line calibration exists in this system, so an actual "
                 "goal-line crossing cannot be detected. This only reflects that the ball stayed fast/straight "
                 "through the shot and then became untrackable without visibly returning to play (plus, where "
                 "available, that its travel direction pointed at the estimated goal) -- consistent with, but "
                 "nowhere near proof of, a goal. success is always left undetermined and this event is always "
                 "review_required."),
    }
    _log(log, i_end, "GOAL_FOLLOWUP: circumstantial_only_review_required", release_frame=i_end)
    return Event("GOAL", i_end, hi, success=None, evidence=evidence)

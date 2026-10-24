"""
possession_detector.py -- TeamClassifier + PossessionDetector.

SCOPE LIMIT, stated plainly: only the ONE tracked player has a persistent
identity (that comes from player_tracker, via TrackingSnapshot). Every
other person box is just "a detection this frame" with no identity carried
across frames. So this module can say "a same-kit player is near the ball"
but never "the same specific teammate who was near it three seconds ago" --
that would need multi-player tracking this project deliberately does not
build (rule #6: never re-detect or re-associate players independently).

PossessionDetector implements rule #7 directly: while the tracked player's
state is anything other than TRACKING, this detector PAUSES -- it freezes
whatever in_possession value it last held rather than evaluating (and
possibly wrongly resetting) it against a stale/predicted box. It resumes
ordinary evaluation the moment TRACKING returns. A missing player frame
never gets treated as "possession lost" or "possession gained" -- it is
simply not evaluated.

EVENT-SEGMENTATION FIX (this task): OCCLUDED ball frames are now treated
as CLOSE, not as "unmeasurable". Per ball_tracker.py's own state
assignment (`occluded = self._overlaps_player(cx, cy, player_bbox)`),
OCCLUDED specifically means the ball tracker's own predicted position
overlaps the TRACKED PLAYER's box -- i.e. it is POSITIVE evidence the
ball is still with the player, not an absence of evidence. Treating it
the same as LOST (as this file did before) threw that evidence away and
was directly responsible for one of the two false PASS events diagnosed
in the known-match carry clip: a 6-frame OCCLUDED stretch (caused by a
nearby defender/teammate crowding the ball out of clean detection, not by
the ball actually leaving the player's feet) reset the possession streak
to zero and forced the segment closed. This is the ONLY behavioural
change in this file; the genuinely-unmeasurable LOST case, and the
TRACKING-state pause above it, are untouched. See possession_segmentation.py
for the second, complementary fix (a POSSIBLE_RELEASE hysteresis window
for genuine distance-threshold crossings, which this file's raw
per-frame signal still feeds).
"""
from dataclasses import dataclass
from typing import Optional
from collections import deque

import numpy as np

from .colour import shirt_color_hist, color_similarity, KitDescriptor
from .geometry import iou, center, dist

TOUCH_RADIUS = 40.0             # px -- ball within this of the player's foot point counts as "at his feet"
MIN_POSSESSION_FRAMES = 2       # consecutive close+usable+TRACKING frames required before calling it possession
SAME_TEAM_COLOUR_THRESH = 0.5   # RETAINED for backward compatibility / external reference only -- the rewritten
                                  # TeamClassifier below no longer uses a single flat threshold (see rule #12).

# --- TEAM-CLASSIFICATION TASK constants (rules #2-16) -----------------------
MIN_CONFIDENT_SIM = 0.55          # a similarity must clear this to confidently call TEAMMATE or OPPONENT at all
MARGIN_THRESH = 0.12              # rule #12: the GAP between teammate_sim and opponent_sim must exceed this,
                                    # not just one of them clearing a floor -- close calls fall back to UNKNOWN
TEAM_B_MIN_TRUSTED = 3            # need at least this many trusted opponent samples before a genuine two-sided
                                    # (teammate-vs-opponent) comparison is used; below this, only a stricter,
                                    # one-sided TEAM_A-only comparison is possible (rule #10: no forced TEAMMATE)
OPPONENT_BOOTSTRAP_MAX_SIM = 0.30 # one-sided path (no trusted TEAM_B yet): a candidate clearly UNLIKE the only
                                    # known team (football has exactly two sides) is real bootstrap evidence for
                                    # OPPONENT, not a guess -- this is what lets team_b_trusted ever get its first
                                    # sample. The genuinely ambiguous middle (between this and MIN_CONFIDENT_SIM+
                                    # MARGIN_THRESH) still returns UNKNOWN rather than being forced either way.
TEAM_A_MAX_SAMPLES = 40
TEAM_B_MAX_SAMPLES = 40
CONTAMINATION_IOU_THRESH = 0.15   # rule #14: candidate-box IoU vs the tracked player's own box at/above this
                                    # triggers a re-crop to the far side of the torso + a confidence penalty
CONTAMINATION_SIM_PENALTY = 0.85  # multiplicative damping applied to both similarities when contamination fires
CONTAMINATION_SEVERE_IOU_THRESH = 0.5  # rule #14's "UNKNOWN if severe": at/above this IoU, the candidate box
                                    # shares more than half its area with the tracked player's own box -- for two
                                    # DISTINCT people's boxes that is already extreme (a real physical-challenge
                                    # overlap is normally well below this); a box this close to the tracked
                                    # player's own is much more consistent with an artefact re-detection of the
                                    # SAME person (a duplicate/echo box the upstream detector didn't dedupe) than
                                    # a second, genuinely distinct player. Below this line, re-cropping to the
                                    # farther half of the torso (avoid_side_of) still leaves a usable, if noisier,
                                    # sample of a real second person; at/above it, even that half-crop is still
                                    # overwhelmingly the tracked player's own body, so ANY colour evidence read
                                    # from it is self-referential, not evidence about who this box actually is.
                                    # Discovered during rule #22 real-clip regression: a box at IoU=0.83 against
                                    # the tracked player's own bbox (second_match_carry_0_40, frame 32) still read
                                    # back as a confident TEAMMATE match under the old flat-penalty-only handling,
                                    # for the unsurprising reason that its "torso" crop mostly just resampled the
                                    # tracked player's own kit -- and because that box also happened to be the
                                    # nearest detection to the ball, shot_candidate.py's frozen reception logic
                                    # (any TEAMMATE within RECEPTION_RADIUS resolves the release as a completed
                                    # PASS on first touch) turned two known-correct real SHOT events into PASS.
                                    # This is a bug in THIS class's own overlap handling, not a downstream issue:
                                    # rule #14 already calls for "UNKNOWN if severe", this only adds the missing
                                    # severe branch rather than tuning the flat damping factor to try to fit one
                                    # clip's numbers.
TEMPORAL_HISTORY_LEN = 5          # rule #13: short per-position history, frames
TEMPORAL_MATCH_DIST = 45.0        # px -- position-proximity match for temporal smoothing (no real track id
                                    # is available at this call site; position proximity across the short,
                                    # regular per-frame cadence this is called at is a reasonable, self-contained
                                    # proxy -- see the module docstring below)
TEMPORAL_HISTORY_MAX_AGE = 6      # frames -- a position bucket not refreshed within this many frames is dropped,
                                    # so a genuinely new/different player reusing a nearby position never inherits
                                    # a stale verdict

# --- TASK 3L (other-player context) -------------------------------------------
# Diagnosis (event_engine_v1/task3l/TASK3L_TEAM_EVIDENCE_COMPARISON.md): on
# GOLD_DUEL_01 the trusted OPPONENT reference was bootstrapped in frames 0-7 from
# people cut off by the bottom edge of the frame (bench/stand, dark clothing),
# and the reference was the plain MEAN of every trusted sample. From then on a
# genuine light-kit opponent was dissimilar to BOTH references and stayed
# UNKNOWN (183/185 frames). Two general repairs, both about evidence quality, no
# threshold on the decision itself is changed:
#  1. A box truncated by the frame border is a partial person: its proportional
#     torso ROI no longer lands on a torso. It is classified UNKNOWN and can never
#     enter the trusted pool (task rule: partial crop -> UNKNOWN; spectators/staff
#     must not become trusted participants merely from colour).
#  2. The OPPONENT reference is the dominant kit WORN BY SEVERAL PEOPLE AT ONCE,
#     not the mean of every non-target-coloured sample. A team kit is worn by many
#     players simultaneously; a goalkeeper, referee, spectator or on-screen
#     graphic is a singleton. Until some non-target colour cluster has been seen on
#     >= 2 different boxes in the same frame, there is no two-sided reference and
#     the existing one-sided path continues (UNKNOWN stays possible).
TASK3L_OTHER_PLAYER_CONTEXT = True     # False reproduces pre-Task-3L behaviour exactly (diagnostic replays)
FRAME_BORDER_MARGIN_PX = 2.0           # box edge within this many px of the image edge = truncated by the frame
# Same-kit radius for clustering trusted OPPONENT samples. Not a new tuned value:
# it is the LAB distance at which color_similarity() equals MIN_CONFIDENT_SIM,
# i.e. exactly what this classifier already treats as "confidently the same kit"
# (SCALE=35 in colour.py: sim = 1/(1+d/35)  ->  d = 35*(1/0.55 - 1) ~= 28.6).
TEAM_B_CLUSTER_RADIUS = 35.0 * (1.0 / MIN_CONFIDENT_SIM - 1.0)
TEAM_B_MIN_SIMULTANEOUS_WEARERS = 2


def box_truncated_by_frame(box, frame_shape, margin=FRAME_BORDER_MARGIN_PX):
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    return x1 <= margin or y1 <= margin or x2 >= (w - margin) or y2 >= (h - margin)


def dominant_multi_wearer_cluster(samples, radius=TEAM_B_CLUSTER_RADIUS,
                                  min_simultaneous=TEAM_B_MIN_SIMULTANEOUS_WEARERS):
    """samples: [(lab_tuple, frame_no), ...]. Returns the centroid (L,a,b) of
    the colour cluster seen on the most different boxes within a single frame
    (ties: most samples overall), or None if no cluster has ever been seen on
    `min_simultaneous` boxes in the same frame."""
    if not samples:
        return None
    labs = np.array([s[0] for s in samples], dtype=np.float64)
    frames = [s[1] for s in samples]
    best_key, best_members = None, None
    for i in range(len(samples)):
        d = np.linalg.norm(labs - labs[i], axis=1)
        members = np.where(d <= radius)[0]
        per_frame = {}
        for j in members:
            per_frame[frames[j]] = per_frame.get(frames[j], 0) + 1
        key = (max(per_frame.values()), len(members))
        if best_key is None or key > best_key:
            best_key, best_members = key, members
    if best_key[0] < min_simultaneous:
        return None
    return tuple(float(v) for v in labs[best_members].mean(axis=0))


class TeamClassifier:
    """rules #2-16: torso/kit-evidence-based TEAMMATE/OPPONENT/UNKNOWN
    classification, replacing the old whole-top-half HSV-histogram +
    flat-threshold design (see colour.py's module docstring for the
    rule-#1 diagnosis of why that failed on a small, overlap-adjacent
    defender). This class owns all of the DECISION logic (references,
    margins, contamination handling, temporal smoothing, trusted-sample
    bookkeeping); colour.py only extracts a robust colour descriptor of a
    single box.

    Construction stays backward compatible: `tracked_color_hist` is
    whatever `colour.shirt_color_hist(frame0, initial_bbox)` already
    returns (every external caller -- engine.py's own callers, and every
    validation/diagnostic driver script -- keeps calling it exactly the
    same way). `classify(frame, box)` still returns a plain (team, sim)
    2-tuple, so event_detector.py's one call site does not need to change
    its unpacking at all.

    Two small, additive, backward-compatible extensions are used by
    event_detector.py (rules #9/#13 -- see that module's docstring for
    why these are judged in-scope as "minimal compatibility code"):
      - `observe_tracked_player(frame, tracked_bbox)`: called once per
        frame with the tracked player's OWN (frozen, read-only) box, to
        refine the TEAM_A reference from multiple clean frames instead of
        trusting a single frame-0 sample (rule #9), and to know the
        current frame's tracked bbox for rule #14's overlap-contamination
        check. This NEVER reads back into player_tracker or influences
        the tracked bbox in any way -- it only ever appends a colour
        sample to this object's own internal list.
      - `classify(frame, box, frame_number=None)`: frame_number is
        optional (old 2-positional-arg calls keep working) and is used
        only for rule #13's temporal-consistency bookkeeping.
    """

    def __init__(self, tracked_color_hist):
        self.tracked_hist = tracked_color_hist
        self._team_a_samples = [tracked_color_hist.dominant_lab] if isinstance(tracked_color_hist, KitDescriptor) else []
        self._team_b_trusted = []          # rule #7/#8: only ever appended to from HIGH-CONFIDENCE, low-contamination
                                              # OPPONENT calls -- a bad/ambiguous sample never gets in
        self._team_b_frames = []           # TASK 3L: frame number of each trusted sample (parallel list), so the
                                              # reference can require a kit seen on several boxes in one frame
        self._last_tracked_bbox = None
        self._temporal_history = {}         # {(round(cx/BUCKET), round(cy/BUCKET)): deque of (frame_no, team, sim)}
        self._call_counter = 0              # fallback recency clock when frame_number isn't supplied

    # ------------------------------------------------------------ rule #9
    def observe_tracked_player(self, frame, tracked_bbox):
        """Read-only: refine the TEAM_A reference from one more genuine
        frame of the tracked player's own torso, and remember this frame's
        tracked bbox for rule #14's contamination check. Never mutates or
        reads back `tracked_bbox` beyond this plain tuple -- it is never
        the source of a decision about the tracked player itself."""
        self._last_tracked_bbox = tuple(tracked_bbox)
        desc = shirt_color_hist(frame, tracked_bbox)
        if desc is not None and not desc.low_evidence:
            self._team_a_samples.append(desc.dominant_lab)
            if len(self._team_a_samples) > TEAM_A_MAX_SAMPLES:
                self._team_a_samples.pop(0)

    def _team_a_centroid(self):
        if not self._team_a_samples:
            return None
        return tuple(float(v) for v in np.mean(np.array(self._team_a_samples), axis=0))

    def _team_b_centroid(self):
        if len(self._team_b_trusted) < TEAM_B_MIN_TRUSTED:
            return None
        if TASK3L_OTHER_PLAYER_CONTEXT:
            return dominant_multi_wearer_cluster(list(zip(self._team_b_trusted, self._team_b_frames)))
        return tuple(float(v) for v in np.mean(np.array(self._team_b_trusted), axis=0))

    # -------------------------------------------------------- rule #13
    def _temporal_bucket(self, box):
        cx, cy = center(box)
        BUCKET = TEMPORAL_MATCH_DIST
        return (round(cx / BUCKET), round(cy / BUCKET))

    def _temporal_lookup(self, box, frame_number):
        key = self._temporal_bucket(box)
        hist = self._temporal_history.get(key)
        if hist is None:
            return None
        last_frame_no = hist[-1][0]
        age = (frame_number - last_frame_no) if frame_number is not None else 1
        if age > TEMPORAL_HISTORY_MAX_AGE:
            return None
        return hist

    def _temporal_update(self, box, frame_number, team, sim):
        key = self._temporal_bucket(box)
        fno = frame_number if frame_number is not None else self._call_counter
        hist = self._temporal_history.setdefault(key, deque(maxlen=TEMPORAL_HISTORY_LEN))
        hist.append((fno, team, sim))

    # ------------------------------------------------------------- main
    def classify(self, frame, box, frame_number=None):
        """Returns ('TEAMMATE'|'OPPONENT'|'UNKNOWN', confidence).
        'UNKNOWN' covers both "too little evidence" (rule #10/#19.H) and
        "ambiguous margin between two real candidates" (rule #12) --
        never a guessed label either way."""
        self._call_counter += 1
        if TASK3L_OTHER_PLAYER_CONTEXT and box_truncated_by_frame(box, frame.shape):
            # TASK 3L: partial person (cut by the image edge) -- no trustworthy torso
            # evidence, never a team label, never a trusted reference sample
            self._temporal_update(box, frame_number, "UNKNOWN", 0.0)
            return "UNKNOWN", 0.0
        desc = shirt_color_hist(frame, box)
        if desc is None:
            return "UNKNOWN", 0.0

        # rule #14: overlap-contamination handling
        contamination = 0.0
        if self._last_tracked_bbox is not None:
            contamination = iou(box, self._last_tracked_bbox)
            if contamination >= CONTAMINATION_SEVERE_IOU_THRESH:
                # too contaminated to say anything at all (see
                # CONTAMINATION_SEVERE_IOU_THRESH's comment) -- this candidate
                # shares more than half its area with the tracked player's own
                # box, so even a re-cropped "far half" sample would still be
                # overwhelmingly his own body. Rather than confidently
                # mislabel a self-referential match, refuse to guess.
                self._temporal_update(box, frame_number, "UNKNOWN", 0.0)
                return "UNKNOWN", 0.0
            if contamination >= CONTAMINATION_IOU_THRESH:
                recropped = shirt_color_hist(frame, box, avoid_side_of=self._last_tracked_bbox)
                if recropped is not None:
                    desc = recropped

        if desc.low_evidence:
            self._temporal_update(box, frame_number, "UNKNOWN", 0.0)
            return "UNKNOWN", 0.0

        team_a = self._team_a_centroid()
        team_b = self._team_b_centroid()
        teammate_sim = color_similarity(desc, team_a) if team_a is not None else 0.0
        opponent_sim = color_similarity(desc, team_b) if team_b is not None else None

        team, confidence = self._decide(teammate_sim, opponent_sim)

        # rule #14: contamination damps the FINAL confidence, applied after
        # the decision is made -- never the raw similarities feeding into
        # it. Damping an input similarity before _decide() would distort
        # the one-sided bootstrap path (OPPONENT confidence there is
        # 1 - teammate_sim, so shrinking teammate_sim would perversely
        # INCREASE it); damping the output uniformly lowers confidence
        # in either direction instead, exactly rule #14's intent.
        if contamination >= CONTAMINATION_IOU_THRESH:
            confidence *= CONTAMINATION_SIM_PENALTY

        # rule #13: temporal consensus -- ONE DIRECTION ONLY. This softens a
        # frame that flatly contradicts an otherwise-consistent recent run
        # at this screen position down to UNKNOWN (damping flicker), but
        # deliberately NEVER promotes an UNKNOWN verdict up into a confident
        # TEAMMATE/OPPONENT label. There is no real track id at this call
        # site (event_detector.py's other_boxes carry no identity -- see
        # this class's docstring), so "history at this position" is only a
        # proxy and can easily belong to a different physical player a few
        # frames later; letting that proxy invent a label that THIS frame's
        # own colour evidence did not itself support risks exactly the kind
        # of confident-but-wrong call rule #10 exists to prevent (this was
        # caught during rule #22 regression testing -- an earlier, two-
        # directional version of this logic let a single stale/marginal
        # sample silently re-assert TEAMMATE on unrelated later frames and
        # broke both known real SHOT events; removed rather than tuned
        # around, since a bug in this classifier's own logic is this task's
        # responsibility to fix, not something to paper over).
        hist = self._temporal_lookup(box, frame_number)
        if hist and team != "UNKNOWN":
            recent = [t for (_f, t, _s) in hist]
            if len(recent) >= 2 and recent.count(team) == 0 and len(set(recent) - {"UNKNOWN"}) >= 1:
                # this frame flatly contradicts an otherwise-consistent
                # recent run of a DIFFERENT confident label -- treat as a
                # momentary ambiguity, not a flip
                team, confidence = "UNKNOWN", min(confidence, 0.3)

        self._temporal_update(box, frame_number, team, confidence)

        # rule #7/#8: only ever refine the OPPONENT reference from a
        # confident, low-contamination, sufficient-evidence OPPONENT call --
        # a wrong/ambiguous sample can never poison the trusted pool.
        if team == "OPPONENT" and confidence >= MIN_CONFIDENT_SIM and contamination < CONTAMINATION_IOU_THRESH:
            self._team_b_trusted.append(desc.dominant_lab)
            self._team_b_frames.append(frame_number if frame_number is not None else self._call_counter)
            if len(self._team_b_trusted) > TEAM_B_MAX_SAMPLES:
                self._team_b_trusted.pop(0)
                self._team_b_frames.pop(0)

        return team, confidence

    @staticmethod
    def _decide(teammate_sim, opponent_sim):
        """rule #12 (margin-based) + rule #10 (no forced TEAMMATE without
        real opponent evidence) + rule #16 (a third kit matching neither
        prototype well stays UNKNOWN, never defaults to TEAMMATE)."""
        if opponent_sim is None:
            # rule #10: no trusted opponent reference yet -- a strict,
            # ASYMMETRIC one-sided check. Confidently TEAMMATE requires a
            # high absolute similarity to the only known team. A candidate
            # CLEARLY unlike that team is real bootstrap evidence for
            # OPPONENT (football has exactly two sides) -- without this,
            # team_b_trusted could never receive its first sample and this
            # class could never call OPPONENT at all. The genuinely
            # ambiguous middle ground stays UNKNOWN, never forced either way.
            if teammate_sim >= MIN_CONFIDENT_SIM + MARGIN_THRESH:
                return "TEAMMATE", teammate_sim
            if teammate_sim <= OPPONENT_BOOTSTRAP_MAX_SIM:
                return "OPPONENT", 1.0 - teammate_sim
            return "UNKNOWN", teammate_sim

        margin = teammate_sim - opponent_sim
        if teammate_sim >= MIN_CONFIDENT_SIM and margin >= MARGIN_THRESH:
            return "TEAMMATE", teammate_sim
        if opponent_sim >= MIN_CONFIDENT_SIM and margin <= -MARGIN_THRESH:
            return "OPPONENT", opponent_sim
        return "UNKNOWN", max(teammate_sim, opponent_sim) * 0.5


@dataclass
class PossessionFrame:
    frame: int
    ball_distance: Optional[float]
    ball_state: str                             # CONFIRMED | PREDICTED | OCCLUDED | REACQUIRING | LOST
    in_possession: bool
    tracking_state: str                         # copied straight from the TrackingSnapshot
    paused: bool = False                        # True = no fresh possession decision was made this frame; value below is frozen
    pause_reason: Optional[str] = None          # 'TRACKER_NOT_TRACKING' | None -- OCCLUDED is no longer a pause (see module docstring)
    nearest_other_team: Optional[str] = None    # 'TEAMMATE' | 'OPPONENT' | None
    nearest_other_distance: Optional[float] = None
    # PASS-EVIDENCE FIX (additive only -- see event_detector.py's process_frame docstring note
    # at this same call site): nearest_other_team/nearest_other_distance above are computed
    # EXACTLY as before this task (a single "nearest other player of EITHER team" slot, whichever
    # team wins that frame) and are read completely unchanged by shot_candidate.py's frozen
    # confirmed-reception path -- this task never touches those two fields' values. The two
    # fields below are NEW, purely additive, per-team distances computed from the SAME per-frame
    # boxes, independently of which team "wins" the slot above, so a marking opponent who happens
    # to be marginally nearer the ball on a given frame no longer silently erases a genuinely
    # nearby teammate's (or unclassified-team player's) distance from the evidence available to
    # event_candidates.py's PASS-intent candidate. Both default to None and are ignored entirely
    # by any caller that doesn't ask for them (dataclass default -- no existing positional
    # construction anywhere in this codebase breaks).
    nearest_teammate_distance: Optional[float] = None
    nearest_unknown_distance: Optional[float] = None


class PossessionDetector:
    def __init__(self):
        self._close_streak = 0
        self._last_in_possession = False

    def step(self, interaction_frame, nearest_other_team=None, nearest_other_distance=None,
             nearest_teammate_distance=None, nearest_unknown_distance=None):
        """interaction_frame: an InteractionFrame from PlayerBallInteractionDetector
        (already carries distance + ball_state + tracking_state). nearest_other_team/
        nearest_other_distance: precomputed by the caller (event_detector.py) from the
        ball centre and this frame's non-tracked-player detections -- purely
        informational context, independent of the tracked player's own state, so it's
        passed straight through even on a paused frame.

        nearest_teammate_distance/nearest_unknown_distance (PASS-EVIDENCE FIX, both
        optional, default None): also precomputed by event_detector.py, from the exact
        same per-frame boxes, but as independent per-team minimums rather than the single
        any-team-wins slot above -- see PossessionFrame's own docstring note. Passed
        straight through unchanged on every branch, exactly like the two existing
        nearest_other_* parameters this mirrors."""
        if interaction_frame.tracking_state != "TRACKING":
            # RULE #7: pause, don't decide. Freeze the last known value rather
            # than silently ending or continuing possession on a stale/absent box.
            return PossessionFrame(frame=interaction_frame.frame_number, ball_distance=interaction_frame.distance,
                                    ball_state=interaction_frame.ball_state, in_possession=self._last_in_possession,
                                    tracking_state=interaction_frame.tracking_state, paused=True,
                                    pause_reason="TRACKER_NOT_TRACKING",
                                    nearest_other_team=nearest_other_team, nearest_other_distance=nearest_other_distance,
                                    nearest_teammate_distance=nearest_teammate_distance,
                                    nearest_unknown_distance=nearest_unknown_distance)

        if interaction_frame.ball_state == "OCCLUDED":
            # EVENT-SEGMENTATION FIX (see module docstring): OCCLUDED means
            # the ball's predicted position overlaps the tracked player's
            # own box -- treat it as CLOSE (same as a genuine sub-radius
            # detection), not as missing/negative evidence.
            self._close_streak += 1
            in_possession = self._close_streak >= MIN_POSSESSION_FRAMES
            self._last_in_possession = in_possession
            return PossessionFrame(frame=interaction_frame.frame_number, ball_distance=None,
                                    ball_state="OCCLUDED", in_possession=in_possession,
                                    tracking_state=interaction_frame.tracking_state,
                                    nearest_other_team=nearest_other_team, nearest_other_distance=nearest_other_distance,
                                    nearest_teammate_distance=nearest_teammate_distance,
                                    nearest_unknown_distance=nearest_unknown_distance)

        if interaction_frame.distance is None:
            # genuinely LOST (or otherwise no position at all) -- no
            # position to measure against, and no positive evidence either
            self._close_streak = 0
            self._last_in_possession = False
            return PossessionFrame(frame=interaction_frame.frame_number, ball_distance=None,
                                    ball_state=interaction_frame.ball_state,
                                    in_possession=False, tracking_state=interaction_frame.tracking_state,
                                    nearest_other_team=nearest_other_team, nearest_other_distance=nearest_other_distance,
                                    nearest_teammate_distance=nearest_teammate_distance,
                                    nearest_unknown_distance=nearest_unknown_distance)

        close = interaction_frame.distance <= TOUCH_RADIUS
        self._close_streak = self._close_streak + 1 if close else 0
        in_possession = self._close_streak >= MIN_POSSESSION_FRAMES
        self._last_in_possession = in_possession

        return PossessionFrame(frame=interaction_frame.frame_number, ball_distance=interaction_frame.distance,
                                ball_state=interaction_frame.ball_state, in_possession=in_possession,
                                tracking_state=interaction_frame.tracking_state,
                                nearest_other_team=nearest_other_team, nearest_other_distance=nearest_other_distance,
                                nearest_teammate_distance=nearest_teammate_distance,
                                nearest_unknown_distance=nearest_unknown_distance)

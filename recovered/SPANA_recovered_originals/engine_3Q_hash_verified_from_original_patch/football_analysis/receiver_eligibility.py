"""
receiver_eligibility.py -- TASK 3P: target-state-aware receiver independence.

Problem (event_engine_v1/task3o, TASK3O_FINAL_REPORT.md): the pass-reception logic treats
"the nearest non-target box labelled TEAMMATE" as a receiver. While the selected player is
not normally tracked (OCCLUDED / AMBIGUOUS -> the snapshot says anything other than
TRACKING), the target box handed to this engine is a HELD, stale box. The target's own
re-emerging body can then be detected as a separate box next to it, be (correctly) labelled
TEAMMATE, and be confirmed as the "receiver" of a pass to himself. Team colour cannot catch
this: the box really is Team A. What is missing is a check that the candidate is a
physically independent person.

This module supplies that check and nothing else. It never changes a team label, a
detection, a threshold or any tracker output. Per frame, every non-OPPONENT other-player
box gets one of three states:

  RECEIVER_ELIGIBLE            -- may be used as a receiver / pass-intent target as before
  RECEIVER_IDENTITY_UNCERTAIN  -- might be the selected player himself; must not be used as
                                  receiver evidence (event_detector excludes it from the
                                  nearest-teammate / nearest-other / nearest-unknown fields)
  RECEIVER_INELIGIBLE          -- an OPPONENT box: never a pass receiver (unchanged behaviour)

Rule:
  * target TRACKING -> every non-OPPONENT box is RECEIVER_ELIGIBLE (behaviour unchanged).
  * target not TRACKING -> a box is RECEIVER_IDENTITY_UNCERTAIN when it lies in the target's
    re-emergence region, measured in the TARGET'S OWN BODY HEIGHT (h = held box height):
        core:      centre distance to the held/last-confirmed target box <= 1.0 h
        extended:  centre distance <= 1.5 h AND (the box is >= 50% inside an OPPONENT box that
                   overlaps or adjoins the target box -- the likely occluder -- OR the box is a
                   partial body, height <= FRAGMENT_MAX_HEIGHT_RATIO (0.8, the Task 3L constant)
                   of the target's)
    UNLESS the box has an independent history: over the previous INDEPENDENT_HISTORY_S it can
    be chained back frame-to-frame (<= 0.5 h per step) through at least half of those frames,
    and in every chained frame where the target was TRACKING (at least one) it stood >= 1.0 h
    from the target box. A real teammate who was already standing apart from the target before the
    occlusion therefore stays eligible.

Why these values (none fitted to a clip): one body height is the scale at which "next to the
target" is meaningful for any camera distance; 1.5 h only applies with a second, independent
cue (inside the occluder, or a partial body); 0.5 h per frame is far above real running speed
at broadcast frame rates; 1 s of history is the order of the shortest real off-the-ball run.
On GOLD_DUEL_01 the self-receiver sits at 0.49 h, so the outcome does not hinge on the exact
radius. TASK3P_RECEIVER_INDEPENDENCE = False reproduces pre-Task-3P behaviour exactly.
"""
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .geometry import center, dist
from .opponent_detection import FRAGMENT_MAX_HEIGHT_RATIO

TASK3P_RECEIVER_INDEPENDENCE = True   # False reproduces pre-Task-3P behaviour exactly (diagnostic replays)

RECEIVER_ELIGIBLE = "RECEIVER_ELIGIBLE"
RECEIVER_IDENTITY_UNCERTAIN = "RECEIVER_IDENTITY_UNCERTAIN"
RECEIVER_INELIGIBLE = "RECEIVER_INELIGIBLE"

REEMERGENCE_CORE_BODY_HEIGHTS = 1.0
REEMERGENCE_EXTENDED_BODY_HEIGHTS = 1.5
OCCLUDER_CONTAINMENT_MIN = 0.5
OCCLUDER_ADJACENCY_BODY_HEIGHTS = 1.0
INDEPENDENT_HISTORY_S = 1.0
HISTORY_STEP_BODY_HEIGHTS = 0.5
HISTORY_MIN_FRACTION = 0.5
INDEPENDENT_SEPARATION_BODY_HEIGHTS = 1.0


@dataclass
class EligibilityVerdict:
    status: str
    reason: str
    distance_body_heights: Optional[float] = None


def _height(box):
    return max(0.0, box[3] - box[1])


def _containment(inner, outer):
    ia = max(0.0, inner[2] - inner[0]) * max(0.0, inner[3] - inner[1])
    if ia <= 0:
        return 0.0
    ix = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    iy = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    return ix * iy / ia


def _overlaps_or_adjoins(box, target_box, h):
    ix = min(box[2], target_box[2]) - max(box[0], target_box[0])
    iy = min(box[3], target_box[3]) - max(box[1], target_box[1])
    if ix > 0 and iy > 0:
        return True
    return dist(center(box), center(target_box)) <= OCCLUDER_ADJACENCY_BODY_HEIGHTS * h


def assess_candidate(candidate_box, candidate_team, target_box, target_tracking_state,
                     opponent_boxes: Sequence = (), has_independent_history: bool = False) -> EligibilityVerdict:
    """Pure function: the eligibility of one other-player box in one frame."""
    if candidate_team == "OPPONENT":
        return EligibilityVerdict(RECEIVER_INELIGIBLE, "opponent box: never a pass receiver")
    if target_tracking_state == "TRACKING":
        return EligibilityVerdict(RECEIVER_ELIGIBLE, "target normally tracked")
    h = _height(target_box)
    if h <= 0:
        return EligibilityVerdict(RECEIVER_IDENTITY_UNCERTAIN, "target not tracked and target scale unknown")
    d = dist(center(candidate_box), center(target_box)) / h
    if d > REEMERGENCE_EXTENDED_BODY_HEIGHTS:
        return EligibilityVerdict(RECEIVER_ELIGIBLE, "outside the target re-emergence region", round(d, 3))
    in_occluder = any(_containment(candidate_box, ob) >= OCCLUDER_CONTAINMENT_MIN and
                      _overlaps_or_adjoins(ob, target_box, h) for ob in opponent_boxes)
    partial = _height(candidate_box) <= FRAGMENT_MAX_HEIGHT_RATIO * h
    if d <= REEMERGENCE_CORE_BODY_HEIGHTS:
        why = f"within {REEMERGENCE_CORE_BODY_HEIGHTS} target body-height of the held target box"
    elif in_occluder or partial:
        why = ("inside the occluding opponent's box" if in_occluder else "partial body") + \
              f" within {REEMERGENCE_EXTENDED_BODY_HEIGHTS} target body-heights of the held target box"
    else:
        return EligibilityVerdict(RECEIVER_ELIGIBLE, "no re-emergence cue beyond 1 body-height", round(d, 3))
    if has_independent_history:
        return EligibilityVerdict(RECEIVER_ELIGIBLE, "near the occluded target but with independent history "
                                                     "apart from the target", round(d, 3))
    return EligibilityVerdict(RECEIVER_IDENTITY_UNCERTAIN, "target not tracked; candidate " + why, round(d, 3))


class ReceiverIndependenceTracker:
    """Per-clip helper owned by FootballEventDetector. assess_frame() returns one verdict per
    other-player box (same order as given) and then records the frame for history checks."""

    def __init__(self, fps=25.0):
        self.window = max(2, int(round(INDEPENDENT_HISTORY_S * fps)))
        self._history = deque(maxlen=self.window)   # (target_box, tracking_state, [(centre, team)])

    def _independent_history(self, candidate_box, h):
        point = center(candidate_box)
        chained, separated_tracking_frames = 0, 0
        for target_box, state, boxes in reversed(self._history):
            near = [c for c, team in boxes if team != "OPPONENT" and dist(c, point) <= HISTORY_STEP_BODY_HEIGHTS * h]
            if not near:
                break
            point = min(near, key=lambda c: dist(c, point))
            chained += 1
            if state == "TRACKING":   # separation is only meaningful where the target position is real
                th = _height(target_box) or h
                if dist(point, center(target_box)) < INDEPENDENT_SEPARATION_BODY_HEIGHTS * th:
                    return False   # was on top of the real target at some point: not independent
                separated_tracking_frames += 1
        return chained >= HISTORY_MIN_FRACTION * self.window and separated_tracking_frames >= 1

    def assess_frame(self, target_box, target_tracking_state, classified_boxes) -> List[EligibilityVerdict]:
        """classified_boxes: [(xyxy, team_label), ...] for this frame's other players."""
        opponents = [b for b, team in classified_boxes if team == "OPPONENT"]
        h = _height(target_box)
        verdicts = []
        for b, team in classified_boxes:
            hist = (target_tracking_state != "TRACKING" and team != "OPPONENT" and h > 0
                    and self._independent_history(b, h))
            verdicts.append(assess_candidate(b, team, target_box, target_tracking_state, opponents, hist))
        self._history.append((tuple(target_box), target_tracking_state,
                              [(center(b), team) for b, team in classified_boxes]))
        return verdicts

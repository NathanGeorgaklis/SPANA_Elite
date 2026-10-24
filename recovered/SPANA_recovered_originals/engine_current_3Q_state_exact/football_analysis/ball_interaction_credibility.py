"""
ball_interaction_credibility.py -- TASK 3G: physical-validity gate between ball
tracking and possession.

Failure class (diagnosed on duel_lost_01, Task 3F/3G): after the ball track is
lost, a low-confidence candidate found by the staged ROI search INSIDE the
tracked player's own lower-body region (his shadow, his boot) is accepted with
no trustworthy prediction to gate it against, confirmed after a few frames on
the same spot, and -- because PossessionDetector only checks "ball state has a
position + within TOUCH_RADIUS of the foot point" -- immediately becomes a
touch, a possession segment, and then RECEIVE / CARRY / PASS / DUEL / CROSS
events. The pixels never contained a football.

This module does not change how the ball is tracked. It runs once, offline,
after the clip's frame loop (from FootballEventDetector.finalize, before
possession segmentation), and labels every frame's ball evidence as

  BALL_CREDIBLE          -- may support a touch / possession
  BALL_PLAUSIBLE_REVIEW  -- physically plausible but unverified; recorded, no touch
  BALL_NOT_CREDIBLE      -- may not support a touch / possession

using only evidence the tracker already produces.

UNIT OF DECISION -- the track segment. A segment starts at a COLD acquisition:
an accepted candidate when the motion prediction was not trustworthy (track
start, or after more than max_predicted_frames without a detection -- the same
condition under which ball_tracker.py itself stops gating by distance) that is
not simply the next frame of an anchored reacquire streak. It continues
through every following frame while the prediction stays trustworthy or the
reacquire anchor holds (CONFIRMED / REACQUIRING / PREDICTED / OCCLUDED), and
ends when the track is LOST. Frames inside one segment are linked by gated continuity; frames
in different segments are not.

A segment is CREDIBLE if ANY of (each measured on real clips before use; see
TRUE_VS_FALSE_BALL_EVIDENCE_COMPARISON.md):

  S1 ROOT AWAY FROM THE TARGET -- its cold acquisition is outside the tracked
     player's lower-body zone. The player-adjacent failure class cannot apply;
     screen overlays / boards at distance are StaticDecoyTracker's job.
  S2 STRONG DIRECT DETECTION AT PHYSICAL SCALE -- at least one observation
     came from the full-frame detector at the engine's normal threshold
     (tier A, i.e. not a low-confidence ROI search hit) AND its box is the size
     a football at the player's depth can be (see SCALE BAND below).
  S3 INDEPENDENT TRAJECTORY -- the segment has direct observations both inside
     and outside the lower-body zone (the ball entered from, or left to, open
     space while the track stayed continuous). A shadow or a boot never leaves
     the player.

A segment that fails S1-S3 is BALL_PLAUSIBLE_REVIEW if every direct
observation in it is physically sized and the observations are consistent
over >= MIN_CONSISTENT_OBSERVATIONS frames, else BALL_NOT_CREDIBLE.

Signals measured and NOT used as standalone credibility:
  - multi-frame consistency alone: the false shadow and boot candidates were
    themselves consistent for 3 frames (they are attached to the player);
  - detector confidence alone: true balls in pass_completed_01 were confirmed
    at 0.05-0.07 confidence via continuity.

SCALE BAND (physics, not fitted): a size-5 ball is 0.22 m; adult players are
1.65-1.95 m, so ball/player-height at the same depth is 0.11-0.13. Detector
boxes are padded and motion blur stretches one axis, so the band on the box's
LONGER side is [0.06, 0.20] x player bbox height. Only applied to candidates
in the lower-body zone, where "same depth as the player" holds.

LOWER-BODY ZONE: x in [x1 - 0.25h, x2 + 0.25h], y in [y1 + 0.5h, y2 + 0.25h]
of the tracked player's box (h = box height) -- legs, feet and the ground
shadow next to them.

What this deliberately does not do: reject candidates near the feet (a
continuous ball arriving at the feet is CREDIBLE through S1/S3), require a
detector box every frame (PREDICTED/OCCLUDED frames inherit their segment's
label), or use any clip-specific coordinate, colour or threshold.
"""
from dataclasses import dataclass, field
from typing import List, Optional

GATE_ENABLED = True               # single switch; False reproduces pre-Task-3G behaviour exactly

CREDIBLE = "BALL_CREDIBLE"
REVIEW = "BALL_PLAUSIBLE_REVIEW"
NOT_CREDIBLE = "BALL_NOT_CREDIBLE"

SCALE_BAND = (0.06, 0.20)          # box longer side / player bbox height, see module docstring
ZONE_SIDE_PAD = 0.25               # x-pad, fraction of player height
ZONE_TOP = 0.5                     # zone starts at half the player's height (hips down)
ZONE_BOTTOM_PAD = 0.25             # below the feet, for the ground shadow
MIN_CONSISTENT_OBSERVATIONS = 3
TIER_A_STAGE = "tier_a_full_frame"


@dataclass
class FrameBallEvidence:
    """One frame's ball evidence, as recorded by BallTracker (accept_log) plus
    the tracked player's box for that frame."""
    frame: int
    state: str                         # online state: CONFIRMED/REACQUIRING/PREDICTED/OCCLUDED/LOST
    center: Optional[tuple]
    box: Optional[list]                # accepted candidate box (direct observations only)
    stage: Optional[str]               # search stage that produced the accepted candidate
    det_conf: Optional[float]
    prediction_trusted: bool           # was the motion prediction trustworthy when this frame was processed
    player_bbox: Optional[list]


@dataclass
class SegmentVerdict:
    segment_id: int
    first_frame: int
    last_frame: int
    label: str
    reasons: List[str] = field(default_factory=list)
    root_frame: Optional[int] = None
    root_in_zone: Optional[bool] = None


def in_lower_body_zone(cx, cy, player_bbox) -> bool:
    if player_bbox is None or cx is None:
        return False
    x1, y1, x2, y2 = player_bbox
    h = max(1.0, y2 - y1)
    return (x1 - ZONE_SIDE_PAD * h) <= cx <= (x2 + ZONE_SIDE_PAD * h) and \
           (y1 + ZONE_TOP * h) <= cy <= (y2 + ZONE_BOTTOM_PAD * h)


def physically_sized(box, player_bbox) -> Optional[bool]:
    if box is None or player_bbox is None:
        return None
    h = max(1.0, player_bbox[3] - player_bbox[1])
    longer = max(box[2] - box[0], box[3] - box[1])
    r = longer / h
    return SCALE_BAND[0] <= r <= SCALE_BAND[1]


def _is_direct(ev: FrameBallEvidence) -> bool:
    return ev.state in ("CONFIRMED", "REACQUIRING") and ev.box is not None and ev.center is not None


REACQUIRE_ANCHOR_TOL = 45.0       # == BallTracker.reacquire_anchor_tol default: the tracker's own
                                   # definition of "the same candidate as last frame" during a reacquire streak


def _anchored_to_previous(evidence, i) -> bool:
    """A cold-acquired REACQUIRING streak links consecutive direct observations
    by the tracker's own anchor rule (centre within reacquire_anchor_tol of the
    previous frame's candidate), not by prediction -- treat those as one segment."""
    if i == 0:
        return False
    prev, cur = evidence[i - 1], evidence[i]
    if not (_is_direct(prev) and _is_direct(cur)):
        return False
    dx, dy = cur.center[0] - prev.center[0], cur.center[1] - prev.center[1]
    return (dx * dx + dy * dy) ** 0.5 < REACQUIRE_ANCHOR_TOL


def split_segments(evidence: List[FrameBallEvidence]) -> List[List[int]]:
    """Indices grouped into continuity segments (see module docstring)."""
    segments, cur = [], None
    for i, ev in enumerate(evidence):
        if ev.state == "LOST":
            cur = None
            continue
        cold_start = _is_direct(ev) and not ev.prediction_trusted and not _anchored_to_previous(evidence, i)
        if cur is None or cold_start:
            cur = []
            segments.append(cur)
        cur.append(i)
    return segments


def judge_segment(evidence: List[FrameBallEvidence], idxs: List[int], seg_id: int) -> SegmentVerdict:
    direct = [evidence[i] for i in idxs if _is_direct(evidence[i])]
    first, last = evidence[idxs[0]].frame, evidence[idxs[-1]].frame
    if not direct:
        # a run of prediction-only frames with no observation of its own (e.g. a
        # clip that starts mid-extrapolation) -- nothing to verify it with
        return SegmentVerdict(seg_id, first, last, NOT_CREDIBLE, ["no direct observation in segment"])
    root = direct[0]
    root_in_zone = in_lower_body_zone(*root.center, root.player_bbox)
    zones = [in_lower_body_zone(*d.center, d.player_bbox) for d in direct]
    reasons = []
    if not root_in_zone:
        reasons.append("S1: acquired outside the tracked player's lower-body zone")
    for d in direct:
        if d.stage == TIER_A_STAGE and physically_sized(d.box, d.player_bbox) is not False:
            reasons.append(f"S2: tier-A detection at physical scale (frame {d.frame}, conf {d.det_conf:.2f})")
            break
    if any(zones) and not all(zones):
        reasons.append("S3: observed both inside and outside the lower-body zone within one continuous segment")
    if reasons:
        return SegmentVerdict(seg_id, first, last, CREDIBLE, reasons, root.frame, root_in_zone)

    sized = [physically_sized(d.box, d.player_bbox) for d in direct]
    if len(direct) >= MIN_CONSISTENT_OBSERVATIONS and all(s is not False for s in sized):
        return SegmentVerdict(seg_id, first, last, REVIEW,
                              ["acquired cold inside the lower-body zone; physically sized and consistent, "
                               "but no tier-A detection and no independent trajectory"], root.frame, root_in_zone)
    why = ["acquired cold inside the lower-body zone with no tier-A detection at physical scale "
           "and no trajectory outside the zone"]
    if any(s is False for s in sized):
        why.append("one or more observations outside the physical ball-size band")
    if len(direct) < MIN_CONSISTENT_OBSERVATIONS:
        why.append(f"only {len(direct)} direct observation(s)")
    return SegmentVerdict(seg_id, first, last, NOT_CREDIBLE, why, root.frame, root_in_zone)


def assess(evidence: List[FrameBallEvidence]):
    """Returns (per-frame labels list, segment verdicts). LOST frames get None
    (no ball position -- nothing to gate)."""
    labels: List[Optional[str]] = [None] * len(evidence)
    verdicts = []
    for k, idxs in enumerate(split_segments(evidence)):
        v = judge_segment(evidence, idxs, k)
        verdicts.append(v)
        for i in idxs:
            labels[i] = v.label
    return labels, verdicts


def gate_possession_frames(possession_frames, labels):
    """Replays PossessionDetector over the same per-frame inputs with every
    non-CREDIBLE ball frame treated as carrying no ball position (exactly what
    the detector does on a LOST frame). Credible frames, paused (non-TRACKING)
    frames and every nearest_* context field are passed through unchanged, so a
    clip whose ball evidence is all credible reproduces the original frames
    exactly. Returns a new list; the input is not mutated."""
    from dataclasses import replace
    from .possession_detector import PossessionDetector
    from .interaction_detector import InteractionFrame

    det = PossessionDetector()
    out = []
    for pf, label in zip(possession_frames, labels):
        credible = label == CREDIBLE
        state = pf.ball_state if credible else ("LOST" if pf.ball_state != "LOST" else pf.ball_state)
        distance = pf.ball_distance if credible else None
        inter = InteractionFrame(frame_number=pf.frame, distance=distance, ball_state=state,
                                 tracking_state=pf.tracking_state)
        new = det.step(inter, nearest_other_team=pf.nearest_other_team,
                       nearest_other_distance=pf.nearest_other_distance,
                       nearest_teammate_distance=pf.nearest_teammate_distance,
                       nearest_unknown_distance=pf.nearest_unknown_distance)
        # keep the ORIGINAL ball_state/distance for every downstream reader that
        # describes the ball itself; only the possession decision is gated
        out.append(replace(new, ball_state=pf.ball_state, ball_distance=pf.ball_distance))
    return out


def evidence_from_tracker(ball_tracker) -> List[FrameBallEvidence]:
    """Build per-frame evidence from BallTracker.history (online states, as the
    possession detector saw them -- the offline bridge only relabels LOST
    frames, which carry no position either way) and BallTracker.accept_log."""
    out = []
    log = getattr(ball_tracker, "accept_log", [])
    for i, st in enumerate(ball_tracker.history):
        rec = log[i] if i < len(log) else {}
        state = st.state
        if getattr(st, "bridge_info", None) is not None:
            state = "LOST"          # bridged = reconstructed afterwards; it was LOST online
        center = (st.centre_x, st.centre_y) if st.centre_x is not None else None
        direct = state in ("CONFIRMED", "REACQUIRING")
        out.append(FrameBallEvidence(frame=st.frame_number, state=state, center=center,
                                     box=rec.get("box") if direct else None,
                                     stage=rec.get("stage") if direct else None,
                                     det_conf=rec.get("conf") if direct else None,
                                     prediction_trusted=bool(rec.get("prediction_trusted", False)),
                                     player_bbox=rec.get("player_bbox")))
    return out

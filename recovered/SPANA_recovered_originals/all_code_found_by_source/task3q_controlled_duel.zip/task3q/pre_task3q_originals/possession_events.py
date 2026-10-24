"""
event_engine_v1/possession_events.py -- Task 3A, Step 4.

Derives POSSESSION_WON / POSSESSION_LOST for the target player, purely as
edge-detection over the EXISTING, already-classified event stream the
frozen football_analysis engine emits (TOUCH/RECEIVE/CARRY as "control"
events, PASS/SHOT/CROSS/DUEL/UNKNOWN_RELEASE/GOAL as "release" events).
This does NOT build a new possession detector and introduces NO new
thresholds: it reads boundaries that already exist in events the frozen
engine itself produced and arbitrated.

DESIGN NOTE (flagged honestly, not hidden). The frozen engine also computes
a lower-level, per-frame boolean possession timeline internally
(football_analysis/event_detector.py's FootballEventDetector.
_possession_frames, fed by possession_detector.py's PossessionDetector.
step()) -- this is literally the "possession timeline" phrase in the task
document most directly evokes. But that timeline is a private
implementation detail: FootballAnalysisEngine.run_analysis_pass() builds
the detector, consumes _possession_frames internally during finalize(),
and discards the detector object -- it is never part of the function's
public return value. Reaching it would require either editing the frozen
engine.py to return it (forbidden -- "do not change frozen files") or
duplicating engine.py's ~80-line per-frame orchestration loop in this new
module just to capture one private attribute off a detector object this
module would have to build a second time (which risks re-implementing
event-engine logic, the opposite of "do not duplicate event logic already
present"). The target player's own emitted event stream -- TOUCH/RECEIVE/
CARRY/PASS/SHOT/CROSS/DUEL/GOAL/UNKNOWN_RELEASE, each already fully
arbitrated and confidence-scored -- is the only possession-relevant signal
FootballAnalysisEngine.run_analysis_pass() actually exposes through its
real public contract (`result["events"]`), so that is what this module
reduces over. If Step 10/12's hand-check finds this too coarse, that is a
reportable finding about this design choice, not something to quietly
patch by reaching into private engine state.

Known simplification, stated plainly for the human-comparison step: a
DUEL event ends the target's current possession episode here regardless
of its own win/lose outcome. If the target wins a duel and immediately
regains the ball, the engine's own next TOUCH/CARRY event emits a fresh
POSSESSION_WON right after -- so a won duel shows as LOST-then-WON-again,
not one continuous possession. This is a known coarseness of edge-
detection over discrete events, not a bug to silently paper over.
"""
from typing import Dict, List

CONTROL_TYPES = {"TOUCH", "RECEIVE", "CARRY"}
RELEASE_TYPES = {"PASS", "SHOT", "CROSS", "DUEL", "UNKNOWN_RELEASE", "GOAL"}


def derive_possession_events(target_events: List[Dict]) -> List[Dict]:
    """target_events: the subset of the frozen engine's own emitted events
    that are stat-eligible for the target player (TASK 3F: actor ==
    "TARGET_PLAYER" from identity evidence AND event band not LOW), already
    normalized to the event_engine_v1 schema (has start_time/end_time/type/
    confidence/review_required). Returns new POSSESSION_WON / POSSESSION_LOST
    event dicts in that same normalized schema (event_id left None -- the
    caller assigns ids when merging these into the final event log
    alongside, never in place of, the engine's own events)."""
    ordered = sorted(target_events, key=lambda e: e["start_time"])
    out = []
    has_ball = False
    episode_id = 0
    for e in ordered:
        etype = e["type"]
        if etype in CONTROL_TYPES and not has_ball:
            episode_id += 1
            out.append(_possession_event("POSSESSION_WON", e, episode_id))
            has_ball = True
        elif etype in RELEASE_TYPES and has_ball:
            out.append(_possession_event("POSSESSION_LOST", e, episode_id))
            has_ball = False
    return out


def _possession_event(etype: str, triggering_event: Dict, episode_id: int) -> Dict:
    is_won = etype == "POSSESSION_WON"
    t_frame = triggering_event.get("start_frame") if is_won else triggering_event.get("end_frame")
    t_time = triggering_event["start_time"] if is_won else triggering_event["end_time"]
    return {
        "event_id": None,
        "type": etype,
        "start_frame": t_frame,
        "end_frame": t_frame,
        "start_time": t_time,
        "end_time": t_time,
        "actor": triggering_event.get("actor", "UNKNOWN"),   # TASK 3F: fail closed
        "outcome": None,
        "success": None,
        "confidence": triggering_event.get("confidence"),
        "event_confidence": triggering_event.get("event_confidence", triggering_event.get("confidence")),
        "event_confidence_band": triggering_event.get("event_confidence_band"),
        "actor_confidence": triggering_event.get("actor_confidence"),
        "actor_confidence_band": triggering_event.get("actor_confidence_band"),
        "outcome_confidence": None,
        "outcome_confidence_source": None,
        "review_required": triggering_event.get("review_required", False),
        "stat_eligible": triggering_event.get("stat_eligible", False),
        "qualifiers": {
            "possession_episode_id": episode_id,
            "triggering_event_type": triggering_event["type"],
        },
        "source_event": "event_engine_v1.possession_events.derive_possession_events "
                         "(edge-detection over the frozen engine's own emitted event stream)",
    }

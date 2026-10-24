"""
event_engine_v1/player_stats.py -- Task 3A, Step 6.

Pure, deterministic reducer: event_log -> the approved
`player_analysis.json` schema's `stats` block (MILESTONE3_TASK3A_TARGET_
PLAYER_EVENT_ENGINE_SPEC.md Section 6). No model, no inference, no
independent prediction -- every number here is a direct count/sum over
the already-finalized, normalized event_log this module receives. Running
this twice on the same event_log must produce byte-identical output.

Exact schema fields produced (Section 6, verbatim):
  touches, receives, carries, passes_attempted, passes_completed,
  passes_incomplete, pass_completion_pct, shots, shots_on_target,
  duels, duels_won, duels_lost, possession_won, possession_lost.

RULE NOTES (flagged, not hidden -- specific claims for Step 11 to verify
against the actual visible clip, not buried assumptions):

  - The frozen engine's raw event schema (football_analysis/clip_tagger.py)
    has no separate "outcome" enum -- PASS/CROSS/SHOT/DUEL all carry one
    boolean `success`. The event-normalization layer in
    analyze_target_player.py maps that boolean onto this module's inputs;
    this module reads success/type as already normalized.
  - PASS: passes_attempted += 1 per PASS event; passes_completed += 1 when
    success is True, else passes_incomplete += 1.
  - CROSS: the approved schema has no dedicated cross counters. A CROSS
    event (CROSS_EMISSION_FINDING.md: confirmed actively emitted) is
    functionally a pass-to-teammate with the same COMPLETE/INCOMPLETE
    outcome vocabulary as PASS (confidence.py), so it is folded into the
    passes_* counters here rather than silently dropped from stats. This
    folding is a flagged design choice, not a schema requirement -- if
    Step 11's hand-check or the user wants CROSS broken out separately,
    that is a one-line schema addition, not a rework.
  - SHOT: shots += 1 per SHOT event AND per GOAL event (ClipTagger's own
    meta.phase_1_scope lists SHOT and GOAL as sibling types -- a given
    shot attempt is emitted as one or the other, never both, per
    shot_candidate.py's goal-follow-up upgrade path). shots_on_target is
    only incremented for GOAL events in V1: a GOAL is definitionally on
    target, but the frozen engine does not currently expose a distinct
    signal for "on target but saved" vs. "off target" on a plain SHOT
    event, so shots_on_target is a known UNDERCOUNT of the true on-target
    rate until that signal exists -- reported as a limitation, not patched
    with an invented heuristic.
  - DUEL: duels += 1 per DUEL event; duels_won += 1 when success is True
    (per confidence.py, DUEL's success boolean IS its WON/LOST outcome),
    else duels_lost += 1.
  - CARRY: carries += 1 per CARRY event. No distance/duration stat is
    computed (no pitch-calibration coordinates exist yet --
    CURRENT_SYSTEM_GAP_ANALYSIS.md Section 1), matching the approved
    schema, which also has no carry-distance field.
  - possession_won / possession_lost: counted directly from the
    POSSESSION_WON / POSSESSION_LOST events possession_events.py produced
    and that the caller already merged into event_log.
  - Any event with review_required=True is counted separately
    (events_review_required, additive -- not in the approved schema but
    needed for an honest confidence_summary/review_required_events
    cross-check) and excluded from every count above: an unresolved-
    identity event contributes to no player's confident stat line.
"""
from typing import Dict, List


def compute_player_stats(event_log: List[Dict]) -> Dict:
    stats = {
        "touches": 0, "receives": 0, "carries": 0,
        "passes_attempted": 0, "passes_completed": 0, "passes_incomplete": 0,
        "pass_completion_pct": 0.0,
        "shots": 0, "shots_on_target": 0,
        "duels": 0, "duels_won": 0, "duels_lost": 0,
        "possession_won": 0, "possession_lost": 0,
        "events_review_required": 0,
    }
    for e in event_log:
        if e.get("review_required"):
            stats["events_review_required"] += 1
            continue
        if e.get("actor") not in ("TARGET_PLAYER", None):
            continue
        t = e["type"]
        if t == "TOUCH":
            stats["touches"] += 1
        elif t == "RECEIVE":
            stats["receives"] += 1
        elif t == "CARRY":
            stats["carries"] += 1
        elif t in ("PASS", "CROSS"):
            stats["passes_attempted"] += 1
            if e.get("success") is True:
                stats["passes_completed"] += 1
            else:
                stats["passes_incomplete"] += 1
        elif t == "SHOT":
            stats["shots"] += 1
        elif t == "GOAL":
            stats["shots"] += 1
            stats["shots_on_target"] += 1
        elif t == "DUEL":
            stats["duels"] += 1
            if e.get("success") is True:
                stats["duels_won"] += 1
            else:
                stats["duels_lost"] += 1
        elif t == "POSSESSION_WON":
            stats["possession_won"] += 1
        elif t == "POSSESSION_LOST":
            stats["possession_lost"] += 1

    if stats["passes_attempted"] > 0:
        stats["pass_completion_pct"] = round(
            100.0 * stats["passes_completed"] / stats["passes_attempted"], 1)
    return stats

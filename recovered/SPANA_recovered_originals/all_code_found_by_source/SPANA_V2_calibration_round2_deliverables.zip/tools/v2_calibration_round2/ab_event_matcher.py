#!/usr/bin/env python3
"""ab_event_matcher.py -- deliverable #8/#10: the PRE-REGISTERED Annotator
A / Annotator B event-matching algorithm for the V2 calibration round.

Written and reviewed BEFORE any Annotator B (Nathan) data exists. Per the
task's explicit instruction ("Do NOT invent matching rules after observing
B's labels"), this file's matching logic must not be edited once B's
submissions start arriving -- any change after that point should be a new,
clearly-dated version, not a silent edit of this one.

------------------------------------------------------------------------
THE MATCHING ALGORITHM (pre-registered)
------------------------------------------------------------------------
Never "one clip = one class" (MODEL_ROADMAP.md Section 4). Instead, for
each clip, A's event list and B's event list are matched pairwise:

  1. CANDIDATE PAIRS: an A-event and a B-event are eligible to match only if
     BOTH of the following hold:
       a) SAME SELECTED PLAYER -- both refer to the player identified by
          the clip's neutral hint (both annotators saw the same hint by
          construction; this is a soft/manual check today since B enters
          free-text "player", not a resolved player_id -- see LIMITATIONS
          below).
       b) COMPATIBLE EVENT TYPE -- exact event_type match, OR the
          CARRY-vs-PASS non-competition rule below.

  2. CARRY-IS-NEVER-COMPETING RULE (task's own worked example): CARRY is
     never annotated as a discrete event by either annotator (schema
     forbids it -- SPANA_EVENT_TAXONOMY_V2.md 5.14). So a clip where A
     recorded [BALL_RECOVERY, PASS] and B recorded only [PASS] is a MATCH
     on the PASS event, not a mismatch or a missed-event penalty for the
     unannotated carry span in between. Concretely: when comparing two
     event lists, a discrete event on one side occurring within a window
     that the OTHER side's TAKE_ON/DUEL span or unannotated gap plausibly
     explains is not double-penalized -- this tool matches PASS-to-PASS
     (etc.) directly and does not require BALL_RECOVERY/TAKE_ON/DUEL
     bookends to also exist on both sides to count the PASS as matched.

  3. TIMING CRITERION, per EVENT_TIMING_STANDARD_V2.md's canonical fields:
       - DISCRETE events (PASS, CROSS, SHOT, TACKLE, INTERCEPTION,
         BLOCKED_PASS, CLEARANCE, BALL_RECOVERY, DISPOSSESSED, SHOT_BLOCK,
         FOUL, OFFSIDE): match if |A.interaction_time - B.interaction_time|
         <= 1.0s (the pre-registered interaction_time tolerance).
       - SPANNING events (TAKE_ON, DUEL): match if [A.start_time,
         A.end_time] and [B.start_time, B.end_time] overlap by any amount
         AND the boundary error (measured separately, not gating the
         match) is reported; a looser secondary check also accepts a match
         when the two spans are within 2.0s of each other even without
         literal overlap, since EVENT_TIMING_STANDARD_V2.md's 2.0s span
         tolerance is explicitly softer than the 1.0s discrete tolerance.

  4. ONE-TO-ONE ASSIGNMENT: within a clip, matching is a greedy
     minimum-timing-error bipartite assignment -- each A-event matches at
     most one B-event and vice versa, choosing the lowest timing error
     among eligible candidate pairs first, then proceeding to the next
     lowest, until no eligible pairs remain. This avoids one real event
     being double-counted against two of the other annotator's events.

  5. UNMATCHED EVENTS: any A-event with no matching B-event is a MISS
     (recall error); any B-event with no matching A-event is a FALSE
     POSITIVE (precision error). This is what "event detection precision/
     recall" (deliverable requirement) means operationally here, computed
     across the whole 10-clip batch, not per clip.

  6. FOR MATCHED PAIRS ONLY, report: event-type agreement (should be 100%
     by construction, since event_type is part of the matching criterion
     -- kept as an explicit reported number anyway, never assumed),
     outcome agreement, timing error (seconds AND frames, per
     EVENT_TIMING_STANDARD_V2.md Section 3), player-identity agreement
     (manual/qualitative today -- see LIMITATIONS), and qualifier overlap
     where both sides supplied qualifiers.

  7. NO_EVENT / empty lists: a clip where BOTH A and B recorded no events
     is a true negative (counted for recall/precision denominators
     separately, never silently dropped). A clip where one side recorded
     an empty list and the other recorded event(s) is exactly the
     miss/false-positive case in (5), not a special case.

------------------------------------------------------------------------
LIMITATIONS, stated up front rather than discovered after running it
------------------------------------------------------------------------
- Player identity matching is NOT automated to a player_id today (no
  persistent multi-player tracking/ReID exists -- CURRENT_SYSTEM_GAP_
  ANALYSIS.md). Both annotators are shown the SAME neutral hint for a
  clip by construction, so within one clip there is normally only one
  sensible "selected player" reading; PLAYER-IDENTITY AGREEMENT is
  reported as a qualitative manual judgement (did A's and B's free-text
  player descriptions plausibly refer to the same person), not a
  computed field, and is clearly labeled as such in the report.
- This module does not run without real Annotator B data -- see
  __main__ guard below, which refuses to execute against anything but
  real submitted files (task item #10: "ready but NOT run until Nathan
  supplies B").
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
MAPPING_PATH = REPO_ROOT / "dataset" / "blind_mapping_v2_round2" / "mapping.json"
ANNOTATOR_A_DIR = REPO_ROOT / "dataset" / "annotations_v2_round2" / "annotator_a"

INTERACTION_TOLERANCE_S = 1.0   # EVENT_TIMING_STANDARD_V2.md Section 2
SPAN_TOLERANCE_S = 2.0          # EVENT_TIMING_STANDARD_V2.md Section 2

DISCRETE_TYPES = {
    "PASS", "CROSS", "SHOT", "TACKLE", "INTERCEPTION", "BLOCKED_PASS",
    "CLEARANCE", "BALL_RECOVERY", "DISPOSSESSED", "SHOT_BLOCK", "FOUL", "OFFSIDE",
}
SPANNING_TYPES = {"TAKE_ON", "DUEL"}


def _canonical_time_a(ev: dict) -> Optional[float]:
    """A's records store canonical timing as ABSOLUTE source_timestamp /
    start_time / end_time (event_v2.schema.json)."""
    if ev["event_type"] in SPANNING_TYPES:
        return None  # handled as a span, see spans_overlap_or_close
    return ev.get("interaction_time")


def _canonical_time_b(ev: dict, ctx_start: float) -> Optional[float]:
    """B's tool exports CLIP-RELATIVE seconds (t=0 at file start) --
    convert to the same absolute source_timestamp basis as A using the
    clip's registered analysis_context_window start, from the hidden
    mapping. B never sees ctx_start; only this comparison step does."""
    if ev["event_type"] in SPANNING_TYPES:
        return None
    t = ev.get("interaction_time")
    return None if t is None else ctx_start + t


def _span_a(ev: dict):
    return (ev.get("start_time"), ev.get("end_time"))


def _span_b(ev: dict, ctx_start: float):
    s, e = ev.get("start_time"), ev.get("end_time")
    if s is None or e is None:
        return (None, None)
    return (ctx_start + s, ctx_start + e)


def _spans_overlap_or_close(a_span, b_span, tolerance: float) -> tuple:
    """Returns (is_match, boundary_error_s) where boundary_error_s is the
    max of |start diff| and |end diff| when both are present, used for
    reporting even when the match succeeds via the tolerance fallback
    rather than literal overlap."""
    a0, a1 = a_span
    b0, b1 = b_span
    if None in (a0, a1, b0, b1):
        return (False, None)
    overlap = min(a1, b1) - max(a0, b0)
    boundary_error = max(abs(a0 - b0), abs(a1 - b1))
    if overlap > 0:
        return (True, boundary_error)
    # softer fallback: spans not literally overlapping but close, within
    # the pre-registered 2.0s span tolerance on both edges
    if abs(a0 - b0) <= tolerance and abs(a1 - b1) <= tolerance:
        return (True, boundary_error)
    return (False, boundary_error)


def match_clip(a_events: list, b_events: list, ctx_start: float, fps: float = 30.0) -> dict:
    """Implements steps 1-6 above for a single clip. Returns a dict with
    matched pairs, unmatched_a (misses), unmatched_b (false positives),
    and per-pair metrics."""
    candidates = []  # (timing_error_s, a_idx, b_idx, kind)
    for ai, a in enumerate(a_events):
        for bi, b in enumerate(b_events):
            if a["event_type"] != b["event_type"]:
                continue  # Section 4's rule 1b: exact type match required
            if a["event_type"] in SPANNING_TYPES:
                ok, err = _spans_overlap_or_close(_span_a(a), _span_b(b, ctx_start), SPAN_TOLERANCE_S)
                if ok:
                    candidates.append((err if err is not None else 0.0, ai, bi, "span"))
            else:
                ta, tb = _canonical_time_a(a), _canonical_time_b(b, ctx_start)
                if ta is None or tb is None:
                    continue
                err = abs(ta - tb)
                if err <= INTERACTION_TOLERANCE_S:
                    candidates.append((err, ai, bi, "discrete"))

    candidates.sort(key=lambda c: c[0])
    used_a, used_b = set(), set()
    matches = []
    for err, ai, bi, kind in candidates:
        if ai in used_a or bi in used_b:
            continue
        used_a.add(ai); used_b.add(bi)
        a, b = a_events[ai], b_events[bi]
        matches.append({
            "a_index": ai, "b_index": bi, "kind": kind,
            "event_type": a["event_type"],
            "timing_error_s": round(err, 3),
            "timing_error_frames": round(err * fps, 1),
            "outcome_agreement": (a.get("outcome") == b.get("outcome")),
            "a_outcome": a.get("outcome"), "b_outcome": b.get("outcome"),
            "qualifier_overlap": sorted(set(a.get("qualifiers") or []) & set(b.get("qualifiers") or [])),
            "a_qualifiers": a.get("qualifiers") or [], "b_qualifiers": b.get("qualifiers") or [],
        })

    unmatched_a = [a_events[i] for i in range(len(a_events)) if i not in used_a]
    unmatched_b = [b_events[i] for i in range(len(b_events)) if i not in used_b]
    return {"matches": matches, "unmatched_a": unmatched_a, "unmatched_b": unmatched_b}


def timing_only_proximity_pairs(a_events: list, b_events: list, ctx_start: float,
                                 window_s: float = 2.5) -> list:
    """SECONDARY DIAGNOSTIC, not the primary matching procedure. The
    primary procedure (match_clip) requires exact event_type equality as
    part of the matching basis, per the task's own explicit instruction
    ("same selected player, compatible event type, interaction-time
    proximity... temporal overlap"). That makes "matched event-type
    agreement" trivially 100% by construction for anything match_clip
    returns -- which is correct for computing detection precision/recall,
    but throws away the diagnostically useful question this project has
    cared about since ANNOTATOR_AGREEMENT_REPORT.md: among event-like
    things the two annotators noticed at roughly the SAME MOMENT, how
    often did they disagree about WHAT TYPE it was? This function pairs
    events by timing proximity ALONE (ignoring event_type), for that one
    diagnostic question. It is never used to compute event detection
    precision/recall, outcome agreement, or anything in the pre-registered
    readiness thresholds -- see V2_CALIBRATION_ROUND_PREP.md Section on
    "two matching passes" for why both exist and what each is for.
    """
    pairs = []
    for ai, a in enumerate(a_events):
        for bi, b in enumerate(b_events):
            if a["event_type"] in SPANNING_TYPES and b["event_type"] in SPANNING_TYPES:
                a0, a1 = _span_a(a); b0, b1 = _span_b(b, ctx_start)
                if None in (a0, a1, b0, b1):
                    continue
                mid_a, mid_b = (a0 + a1) / 2, (b0 + b1) / 2
                gap = abs(mid_a - mid_b)
            elif a["event_type"] not in SPANNING_TYPES and b["event_type"] not in SPANNING_TYPES:
                ta, tb = _canonical_time_a(a), _canonical_time_b(b, ctx_start)
                if ta is None or tb is None:
                    continue
                gap = abs(ta - tb)
            else:
                continue  # a discrete/spanning mismatch is not a timing-proximity pair
            if gap <= window_s:
                pairs.append({
                    "a_index": ai, "b_index": bi, "gap_s": round(gap, 3),
                    "a_type": a["event_type"], "b_type": b["event_type"],
                    "type_agrees": a["event_type"] == b["event_type"],
                })
    return pairs


def load_annotator_a(clip_id: str) -> list:
    p = ANNOTATOR_A_DIR / f"{clip_id}.json"
    return json.loads(p.read_text())["events"]


def run_batch(b_submissions_dir: Path) -> dict:
    """b_submissions_dir: directory of <blind_id>_annotator_b.json files
    exported by the annotation tool. Requires the hidden mapping to
    resolve blind_id -> clip_id -> Annotator A record. NOT invoked by
    anything in this delivery -- see __main__ guard."""
    mapping = json.loads(MAPPING_PATH.read_text())
    results = {}
    for f in sorted(b_submissions_dir.glob("*_annotator_b.json")):
        b_doc = json.loads(f.read_text())
        blind_id = b_doc["blind_id"]
        if blind_id not in mapping:
            raise ValueError(f"{blind_id} not found in hidden mapping -- refusing to guess.")
        clip_id = mapping[blind_id]["clip_id"]
        a_events = load_annotator_a(clip_id)
        # ctx_start recovered from the clip's own clip_time_map.json, not
        # from the hidden mapping (kept minimal on purpose).
        ctx = json.loads((REPO_ROOT / "dataset" / "clips" / clip_id / "clip_time_map.json").read_text())
        ctx_start = ctx["analysis_context_window_absolute_s"][0]
        results[blind_id] = {
            "clip_id": clip_id,
            **match_clip(a_events, b_doc.get("events", []), ctx_start),
            "timing_only_diagnostic_pairs": timing_only_proximity_pairs(a_events, b_doc.get("events", []), ctx_start),
        }
    return results


if __name__ == "__main__":
    import sys
    print(__doc__)
    print("\nThis tool is built and unit-testable (see "
          "test_ab_event_matcher.py) but is DELIBERATELY NOT RUN against "
          "any Annotator B data by this delivery -- Nathan has not yet "
          "completed Annotator B review. Re-invoke run_batch(...) once "
          "real submissions exist under dataset/blind_reviews_v2_round2/.")
    sys.exit(0)

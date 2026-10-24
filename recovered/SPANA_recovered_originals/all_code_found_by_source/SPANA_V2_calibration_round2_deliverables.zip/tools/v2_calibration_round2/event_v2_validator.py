#!/usr/bin/env python3
"""event_v2_validator.py -- schema + cross-field validator for event_v2
records (deliverable #3: "schema validator passing on generated records").

Two layers of checking, because JSON Schema alone cannot express
everything SPANA_EVENT_TAXONOMY_V2.md / EVENT_QUALIFIERS_V2.md /
EVENT_TIMING_STANDARD_V2.md require:

  1. STRUCTURAL: every record validates against
     schema/event_v2.schema.json (required fields present, enums
     respected, additionalProperties=false honoured).

  2. CROSS-FIELD / TAXONOMY RULES (schema/event_v2.schema.json deliberately
     leaves these as free-form strings/nulls rather than re-encoding a full
     per-type whitelist in JSON Schema -- see its own docstring):
       - outcome vocabulary is valid for the record's event_type
         (SPANA_EVENT_TAXONOMY_V2.md Section 5's per-type outcome list)
       - canonical timing field is populated for the event_type
         (EVENT_TIMING_STANDARD_V2.md Section 1): interaction_time for
         discrete events, start_time+end_time for TAKE_ON/DUEL, neither
         required (but neither forbidden) for NO_EVENT
       - qualifiers, where present, are drawn from a group applicable to
         this event_type (EVENT_QUALIFIERS_V2.md)
       - coordinates are only present when spatial_confidence is HIGH or
         MEDIUM -- a LOW/UNKNOWN spatial_confidence with a non-null
         coordinate would be exactly the "fabricated precision" the master
         specification and PASS_PROGRESSION_AND_LINE_BREAKING_SPEC.md both
         forbid
       - CARRY never appears as event_type (it is derived-only, never an
         annotated discrete class -- SPANA_EVENT_TAXONOMY_V2.md 5.14)
       - annotator-facing confidence fields are never left as the Python
         None/JSON null that a machine-only field is allowed to be -- a
         HUMAN-sourced record must state a confidence (even if that stated
         confidence is itself "UNKNOWN"), because "the annotator didn't say"
         and "the annotator said UNKNOWN" are different, auditable facts

Usage:
    python3 event_v2_validator.py <path-to-json-file-or-directory> [...]

A directory is scanned recursively for *.json files. Each file may contain
either a single event record (a dict with "event_type") or a list of event
records, or a clip-record wrapper produced by
build_annotator_a_records.py ({"clip_id": ..., "events": [...]});
the validator unwraps whichever shape it finds.

Exit code 0 iff every record in every file passes both layers.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import jsonschema

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "schema" / "event_v2.schema.json"

DISCRETE_INTERACTION_TYPES = {
    "PASS", "CROSS", "SHOT", "CLEARANCE", "SHOT_BLOCK", "INTERCEPTION",
    "BLOCKED_PASS", "BALL_RECOVERY", "DISPOSSESSED", "TACKLE", "FOUL", "OFFSIDE",
}
SPANNING_TYPES = {"TAKE_ON", "DUEL"}

# SPANA_EVENT_TAXONOMY_V2.md Section 5's per-type outcome vocabulary.
# `null` and `UNKNOWN` are always additionally allowed -- Section 7 makes
# uncertainty a first-class outcome, never a forced guess.
OUTCOME_VOCAB_BY_TYPE = {
    "PASS": {"COMPLETE", "INCOMPLETE"},
    "CROSS": {"COMPLETE", "INCOMPLETE"},
    "SHOT": {"GOAL", "SAVED", "BLOCKED", "OFF_TARGET", "WOODWORK", "GOAL_LINE_BLOCK"},
    "TACKLE": {"WON", "LOST"},
    "TAKE_ON": {"SUCCESSFUL", "UNSUCCESSFUL"},
    "DUEL": {"WON", "LOST"},
    "INTERCEPTION": set(),      # no defined outcome concept -- expect null
    "BLOCKED_PASS": set(),
    "CLEARANCE": set(),
    "BALL_RECOVERY": set(),
    "DISPOSSESSED": set(),
    "SHOT_BLOCK": set(),
    "FOUL": set(),
    "OFFSIDE": set(),
    "NO_EVENT": set(),
}

# EVENT_QUALIFIERS_V2.md groups -> the event_types each group applies to.
QUALIFIER_GROUPS = {
    frozenset({"LEFT_FOOT", "RIGHT_FOOT", "HEADER", "OTHER_BODY_PART"}): {"SHOT", "PASS", "CROSS"},
    frozenset({"HIGH", "LOW", "GROUND"}): {"CROSS", "PASS", "SHOT"},
    frozenset({"FORWARD", "BACKWARD", "LATERAL"}): {"PASS"},
    frozenset({"SHORT", "MEDIUM", "LONG"}): {"PASS"},
    frozenset({"CORNER", "FREE_KICK", "THROW_IN", "GOAL_KICK", "KICK_OFF", "PENALTY"}): {"PASS", "CROSS", "SHOT"},
    frozenset({"DEFENSIVE_THIRD", "MIDDLE_THIRD", "FINAL_THIRD", "BOX"}): None,  # any spatially-located event
    frozenset({"UNDER_PRESSURE", "NO_PRESSURE"}): None,  # any event
    frozenset({"THROUGH_BALL", "CHIPPED", "LAUNCH", "FLICK_ON", "PULL_BACK", "LAY_OFF", "TAP_PASS"}): {"PASS"},
    frozenset({"PROGRESSIVE", "LINE_BREAKING", "SWITCH_OF_PLAY", "INTO_FINAL_THIRD", "INTO_BOX"}): {"PASS"},
    frozenset({"VOLLEY", "HALF_VOLLEY", "FIRST_TIME", "FREE_KICK", "PENALTY", "OPEN_PLAY"}): {"SHOT"},
    frozenset({"CUTBACK", "EARLY_CROSS", "INSWINGING", "OUTSWINGING"}): {"CROSS"},
    frozenset({"GROUND", "AERIAL"}): {"DUEL"},
}


def _qualifier_owner_types(value: str):
    """Return the set of event_types a qualifier value is valid for, or
    None if it's a universally-applicable group (LOCATION_ZONE, PRESSURE),
    or "UNRECOGNISED" if no group contains it. A value like "GROUND"
    legitimately appears in more than one group (HEIGHT for CROSS/PASS/SHOT,
    DUEL_TYPE for DUEL) -- this checks every matching group and unions the
    allowed event_types rather than stopping at the first match, so a value
    is rejected only when NO group sanctions it for that event_type."""
    matches = [types for group, types in QUALIFIER_GROUPS.items() if value in group]
    if not matches:
        return "UNRECOGNISED"
    if any(types is None for types in matches):
        return None  # at least one matching group is universally applicable
    union = set()
    for types in matches:
        union |= types
    return union


HUMAN_REQUIRED_CONFIDENCE_FIELDS = [
    "player_identity_confidence", "event_type_confidence",
    "outcome_confidence", "timing_confidence",
]
MACHINE_ONLY_FIELDS = [
    "tracking_confidence", "ball_confidence", "control_confidence",
    "body_orientation_confidence", "tactical_context_confidence",
]


def cross_field_errors(rec: dict) -> list:
    errors = []
    et = rec.get("event_type")

    if et == "CARRY":
        errors.append("event_type=CARRY must never appear on an annotated record -- CARRY is derived-only (SPANA_EVENT_TAXONOMY_V2.md 5.14).")

    # --- outcome vocabulary ---
    outcome = rec.get("outcome")
    if et in OUTCOME_VOCAB_BY_TYPE:
        allowed = OUTCOME_VOCAB_BY_TYPE[et] | {None, "UNKNOWN"}
        if outcome not in allowed:
            errors.append(f"outcome={outcome!r} is not valid for event_type={et!r} (allowed: {sorted(x for x in allowed if x)} or null).")

    # --- canonical timing per EVENT_TIMING_STANDARD_V2.md Section 1 ---
    if et in DISCRETE_INTERACTION_TYPES:
        if rec.get("interaction_time") is None:
            errors.append(f"event_type={et!r} is a discrete/ball-contact type -- interaction_time must be set (canonical timing field).")
    elif et in SPANNING_TYPES:
        if rec.get("start_time") is None or rec.get("end_time") is None:
            errors.append(f"event_type={et!r} is a spanning type -- both start_time and end_time must be set (canonical timing fields).")
    elif et == "NO_EVENT":
        pass  # not applicable -- no event to time

    # --- qualifier validity ---
    for q in rec.get("qualifiers") or []:
        owner = _qualifier_owner_types(q)
        if owner == "UNRECOGNISED":
            errors.append(f"qualifier {q!r} is not in any EVENT_QUALIFIERS_V2.md group.")
        elif owner is not None and et not in owner:
            errors.append(f"qualifier {q!r} is not valid for event_type={et!r} (valid for: {sorted(owner)}).")

    # --- never fabricate precision: coordinates require real confidence ---
    coord_fields = ["start_x", "start_y", "end_x", "end_y"]
    has_coord = any(rec.get(f) is not None for f in coord_fields)
    spatial_conf = rec.get("spatial_confidence")
    if has_coord and spatial_conf in (None, "LOW", "UNKNOWN"):
        errors.append(f"a non-null coordinate is set but spatial_confidence={spatial_conf!r} -- coordinates must be null unless spatial_confidence is HIGH or MEDIUM (never fabricate precision).")

    # --- human-facing confidence fields must be explicitly stated ---
    if rec.get("annotation_source") == "HUMAN":
        for f in HUMAN_REQUIRED_CONFIDENCE_FIELDS:
            if rec.get(f) is None:
                errors.append(f"annotation_source=HUMAN but {f} is null -- annotator-facing confidence must be explicitly stated (UNKNOWN is fine, missing is not).")
        for f in MACHINE_ONLY_FIELDS:
            if rec.get(f) is not None:
                errors.append(f"{f}={rec.get(f)!r} is set on a HUMAN record -- machine-only confidence fields must stay null unless actual tooling produced the evidence (task instruction: 'must not be guessed by human annotators').")

    return errors


def _iter_records(obj):
    if isinstance(obj, dict) and "event_type" in obj:
        yield obj
    elif isinstance(obj, dict) and "events" in obj:
        yield from obj["events"]
    elif isinstance(obj, list):
        for item in obj:
            yield from _iter_records(item)


def validate_file(path: Path, schema: dict) -> tuple[int, list]:
    obj = json.loads(path.read_text())
    n_ok = 0
    problems = []
    for i, rec in enumerate(_iter_records(obj)):
        try:
            jsonschema.validate(rec, schema)
        except jsonschema.ValidationError as e:
            problems.append(f"{path}[{i}] event_id={rec.get('event_id')}: SCHEMA ERROR: {e.message}")
            continue
        errs = cross_field_errors(rec)
        if errs:
            for e in errs:
                problems.append(f"{path}[{i}] event_id={rec.get('event_id')}: RULE ERROR: {e}")
        else:
            n_ok += 1
    return n_ok, problems


def main(argv: list) -> int:
    if not argv:
        print(__doc__)
        return 2
    schema = json.loads(SCHEMA_PATH.read_text())
    files = []
    for a in argv:
        p = Path(a)
        if p.is_dir():
            files.extend(sorted(p.rglob("*.json")))
        else:
            files.append(p)

    total_ok = 0
    total_problems = []
    for f in files:
        # skip non-event metadata files that happen to be *.json in these
        # directories (e.g. a summary file with no "events"/"event_type" key)
        try:
            obj = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        if not (isinstance(obj, dict) and ("events" in obj or "event_type" in obj)) and not isinstance(obj, list):
            continue
        n_ok, problems = validate_file(f, schema)
        total_ok += n_ok
        total_problems.extend(problems)

    print(f"Validated files: {len(files)}")
    print(f"Records passed:  {total_ok}")
    print(f"Records failed:  {len(total_problems)}")
    if total_problems:
        print("\n--- FAILURES ---")
        for p in total_problems:
            print(p)
        return 1
    print("\nSCHEMA VALIDATION: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

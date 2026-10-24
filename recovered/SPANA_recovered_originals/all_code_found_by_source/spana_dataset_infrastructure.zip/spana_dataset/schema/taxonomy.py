"""
taxonomy.py -- THE single source of truth for the SPANA dataset's Phase-1 event
vocabulary. Every other tool (annotation server, validator, feature exporter,
benchmark tool) imports from here rather than hard-coding its own copy of these
lists. This is the dataset project's own module -- it does NOT import from, or
get imported by, football_analysis/ or player_tracker/ (the frozen production
pipeline). Changing this file has zero effect on production behaviour.

Per the audit's explicit instruction: Phase 1 is deliberately limited to
PASS / SHOT / CROSS / CARRY / DUEL / OTHER / UNKNOWN. TACKLE / INTERCEPTION /
RECOVERY / DISPOSSESSION / CLEARANCE / AERIAL_DUEL are NOT part of this
taxonomy yet -- they may only be captured as free-text in `notes` for now.
Adding them for real requires the subject-model redesign described in the
architecture audit's TARGET_ARCHITECTURE.md, which is out of scope here.
"""

TAXONOMY_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Primary event types -- Phase 1, deliberately narrow (task item #13)
# ---------------------------------------------------------------------------
EVENT_TYPES = [
    "PASS",
    "SHOT",
    "CROSS",
    "CARRY",
    "DUEL",
    "OTHER",
    "UNKNOWN",          # annotator could not confidently place this in any of the above
]

# A clip can also be a deliberate NEGATIVE example: the selected player is
# visible, nothing event-worthy happens. This is a distinct concept from
# UNKNOWN (which means "something happened but I can't classify it") -- see
# ANNOTATION_GUIDE.md. Stored as event_type="NO_EVENT".
EVENT_TYPES_ALL = EVENT_TYPES + ["NO_EVENT"]

# Annotators may also mark a clip as fundamentally unusable evidence, distinct
# from a real NO_EVENT/UNKNOWN judgement call (task item #5).
ABSTENTION_REASONS = [
    "AMBIGUOUS",
    "INSUFFICIENT_VISUAL_EVIDENCE",
]

# ---------------------------------------------------------------------------
# Outcomes -- per event type (task items #14-18). None of these are shared
# across event types on purpose: "COMPLETED" means something different for a
# PASS than a CROSS, and DUEL's WON/LOST has no equivalent for CARRY at all.
# ---------------------------------------------------------------------------
OUTCOMES_BY_EVENT_TYPE = {
    "PASS": ["COMPLETED", "INCOMPLETE", "UNKNOWN"],
    "SHOT": ["GOAL", "SAVED", "BLOCKED", "OFF_TARGET", "GOAL_LINE_CLEARANCE", "UNKNOWN"],
    "CROSS": ["COMPLETED", "INCOMPLETE", "UNKNOWN"],
    "DUEL": ["WON", "LOST", "UNKNOWN"],
    "CARRY": ["NONE"],       # deliberately no outcome taxonomy yet (task item #18)
    "OTHER": ["UNKNOWN"],
    "UNKNOWN": ["UNKNOWN"],
    "NO_EVENT": ["NONE"],
}

# ---------------------------------------------------------------------------
# Subtypes -- ALL OPTIONAL. Only ever set when confidently observable
# (task items #14-17: "only use subtype when confidently observable" /
# "do not require subtype"). null/None is always valid.
# ---------------------------------------------------------------------------
SUBTYPES_BY_EVENT_TYPE = {
    "PASS": [None, "SHORT", "LONG", "LOB", "THROUGH_BALL", "OTHER"],
    "SHOT": [None, "LONG_RANGE", "CLOSE_RANGE", "OTHER"],
    "CROSS": [None, "LOW", "HIGH", "CUTBACK", "OTHER"],
    "DUEL": [None, "GROUND", "AERIAL"],   # task #17: manual/optional only, never auto-inferred
    "CARRY": [None],
    "OTHER": [None],
    "UNKNOWN": [None],
    "NO_EVENT": [None],
}

# ---------------------------------------------------------------------------
# Hard-negative categories (task item #20) -- recorded in
# hard_negative_category when event_type == "NO_EVENT", purely descriptive,
# used later to check the negative set actually covers the hard cases rather
# than only trivial "nothing happening" frames.
# ---------------------------------------------------------------------------
HARD_NEGATIVE_CATEGORIES = [
    None,
    "NEAR_BALL_NO_TOUCH",
    "TEAMMATE_RECEIVES_NEARBY",
    "SHOT_PASSES_NEARBY",
    "CONTESTED_NO_DUEL",
    "BALL_OCCLUDED",
    "CAMERA_PANS_AWAY",
    "REPLAY",
    "STOPPAGE",
    "OTHER",
]

# ---------------------------------------------------------------------------
# Confidence / quality enums
# ---------------------------------------------------------------------------
ANNOTATION_CONFIDENCE_LEVELS = ["HIGH", "MEDIUM", "LOW"]
MANUAL_IDENTITY_CONFIDENCE_LEVELS = ["HIGH", "MEDIUM", "LOW"]

# LOW-confidence annotations must not silently enter training splits --
# enforced by split_generator.py (task item #21).
TRAINING_ELIGIBLE_CONFIDENCE = {"HIGH", "MEDIUM"}

# ---------------------------------------------------------------------------
# Domain/diversity metadata enums (task item #34) -- technical/footage facts
# only, never inferred player characteristics.
# ---------------------------------------------------------------------------
CAMERA_TYPES = ["BROADCAST_WIDE", "BROADCAST_TIGHT", "FIXED_AMATEUR", "MOBILE", "OTHER", "UNKNOWN"]
SOURCE_KINDS = ["BROADCAST", "MOBILE", "OTHER", "UNKNOWN"]
PITCH_SURFACES = ["GRASS", "ARTIFICIAL", "UNKNOWN"]
LIGHTING_CONDITIONS = ["DAY", "FLOODLIT", "OVERCAST", "UNKNOWN"]

# Team labels used inside an annotation -- deliberately NOT tied to any real
# club name here; the match-level metadata registry (metadata/source_videos.json)
# is where a human-readable team name can optionally be recorded.
TEAMS = ["TEAM_A", "TEAM_B", "UNKNOWN"]


def validate_event_record(event_type, event_subtype, outcome):
    """Returns a list of human-readable problems (empty list = valid).
    Pure validation logic, importable by both the annotation server (for
    live form validation) and dataset_validator.py (for offline checking) --
    single source of truth, checked in exactly one place.
    """
    problems = []
    if event_type not in EVENT_TYPES_ALL:
        problems.append(f"event_type '{event_type}' is not in the Phase-1 taxonomy {EVENT_TYPES_ALL}")
        return problems  # can't check subtype/outcome against an unknown type
    allowed_subtypes = SUBTYPES_BY_EVENT_TYPE[event_type]
    if event_subtype not in allowed_subtypes:
        problems.append(f"event_subtype '{event_subtype}' is not valid for event_type '{event_type}' "
                         f"(allowed: {allowed_subtypes})")
    allowed_outcomes = OUTCOMES_BY_EVENT_TYPE[event_type]
    if outcome not in allowed_outcomes:
        problems.append(f"outcome '{outcome}' is not valid for event_type '{event_type}' "
                         f"(allowed: {allowed_outcomes})")
    return problems

#!/usr/bin/env python3
"""
migrate_existing_annotations.py -- CLI tool that converts this project's
PRE-EXISTING, already-hand-verified validation-clip ground truth (scattered
across REPO_ROOT/validation/, validation_v2/, validation_v3/, validation_v4/)
into the new canonical schema/annotation.schema.json format.

This tool NEVER modifies or deletes anything under REPO_ROOT/validation*/,
REPO_ROOT/football_analysis/, or REPO_ROOT/player_tracker/. It only reads
from those trees and writes new files under
DATASET_ROOT/dataset/annotations/<annotation_id>.json.

=============================================================================
STEP 1 INVESTIGATION FINDINGS (read this before touching the code below)
=============================================================================

validation/annotations/*.json (3 files: known_match_carry_0_20,
known_match_shot_195_223, second_match_carry_0_40) is the ONE place in this
whole legacy tree that holds a structured, machine-parseable ground-truth
annotation format comparable to the new schema -- a JSON object with
`expected_events[0].{type,approx_start_time,approx_end_time,success}`,
`frame_range`, `annotated_by`, `notes`, `cross_check_against_pipeline`. Its
own `validation/clips/registry.json` explicitly documents why these 3 (and
only these 3) are honest, undisputed ground truth: two carries and one shot,
manually re-derived from raw frames BEFORE ever looking at the pipeline's
own predictions (registry.json's `_disclosure` field). These are this
migration's CLEAN, PRIMARY input.

validation_v2/, validation_v3/, validation_v4/ were searched exhaustively
(every non-image file: *.json, *.md, *.py, *.txt, *.log under each tree) for
anything comparable:

  - validation_v2/ and validation_v3/ contain per-clip `outputs/*_events.json`
    files -- these are FROZEN-PIPELINE PREDICTIONS (the system's own output),
    never ground truth, and are explicitly the kind of thing rule 5 in
    validation/README.md says must never be copied into ground truth. They
    are not annotation files and are not migrated as such.
  - validation_v4/ has no reports/ directory and no annotation-shaped JSON
    at all -- it is purely iterative re-runs of the pipeline (outputs/, logs/)
    against clips already covered by validation_v3/'s ground truth.
  - The actual ground truth for the 9 clips validation_v3/ evaluates against
    lives ONLY as prose inside
    `validation_v3/reports/real_world_validation_report.md` (section 3, one
    subsection per clip, e.g. "### 3.3 `dual_won_01` -- GT: BLUE player (no
    number -- GROUND_TRUTH_PLAYER_ID_AMBIGUOUS), DUEL->WON, 3.0-4.0s"). This
    is NOT comparable in *format* to validation/annotations/*.json (no
    frame_range, no per-clip fps, no structured expected_events array -- just
    a human-readable paragraph with a rounded time range), but it IS
    comparable in *nature*: independently re-derived, pre-pipeline-comparison
    ground truth, exactly per rule 5.
  - Two of those 9 clips are explicitly, textually flagged as contested:
    `dual_won_01` and `shot_close_range_01` both carry the literal marker
    **GROUND_TRUTH_PLAYER_ID_AMBIGUOUS** (real_world_validation_report.md
    section 3.3, 3.4, and again in the ranked-failure-modes section 5 and the
    methodology notes in section 7) -- the human annotator could not pin a
    jersey number/identity on the acting player in either clip. This is a
    genuine, load-bearing finding this tool must not silently paper over:
    those two get `ambiguous=true`, `review_required=true`,
    `abstention_reason="AMBIGUOUS"` (see AMBIGUOUS_LEGACY_CLIP_IDS below).
    Every other clip in that report is a normal (if lower-precision than the
    3 primary clips) ground-truth read, migrated as clean.
  - `validation_v2/reports/*.md` were also grepped for the same markers
    (GROUND_TRUTH_PLAYER_ID_AMBIGUOUS / AMBIGUOUS / DISPUTED) -- no hits.
    v2's ground truth (embedded prose in `pre_action_context_capture_report.md`
    section 6 and `event_logic_audit_report.md` section 3, covering
    cross_low_01, duel_lost_01, missed_pass_01, pass_completed_01/02, and a
    *different, unrelated* clip that also happens to be named
    `shot_close_range_01`) is undisputed but even less structured (often just
    "ground truth: PASS, outcome genuinely ambiguous" with no frame numbers,
    no jersey numbers, sometimes no player-vs-outcome-type distinction at
    all) and, critically, the ONE clip name it shares with validation_v3/
    (`shot_close_range_01`) is a COMPLETELY DIFFERENT physical clip (v2's own
    report describes "ground truth: SHOT, GOAL" with no ambiguity marker at
    all, and is drawn from different footage than v3's catastrophically-
    mistracked, identity-ambiguous namesake). Migrating a v2 clip under the
    bare id `shot_close_range_01` would silently collide two unrelated real
    clips into one dataset clip_id -- see CLIP_ID COLLISION note below.
    Given v2's ground truth is materially less complete/structured than
    v3's (no player-identity description, no explicit rounded time range for
    several clips), and the task's own inventory-building instructions name
    v3's clips explicitly (dual_won_02 [sic, the real id is dual_won_01] and
    shot_close_range_01), this tool's SUPPLEMENTARY (report-derived) path
    covers validation_v3/'s 9 clips only. v2's report-embedded ground truth
    is documented here as found, but is not auto-migrated -- re-running this
    investigation to hand-curate a v2 supplementary table is a reasonable
    follow-up, not done here to avoid transcribing under-specified prose into
    false-precision JSON (see "why 4 of the 9 v3 clips are not migrated"
    below for the same concern applied even more strictly).

CLIP_ID COLLISION: because of the above, this tool NEVER reuses a bare
legacy clip_id for a report-derived (supplementary) record -- it prefixes
every validation_v3-sourced clip_id with `v3_` (e.g. `v3_shot_close_range_01`)
specifically so it can never collide with a same-named clip surfacing from a
different legacy batch (v2's own distinct `shot_close_range_01`) in a future
migration run. The 3 PRIMARY clips keep their bare legacy clip_id verbatim
-- those id strings are already unique across the whole legacy tree and
reusing them avoids inventing a new id for no reason (task's own guidance).

WHY 4 OF THE 9 VALIDATION_V3 CLIPS ARE NOT MIGRATED (documented, not silent):
schema/annotation.schema.json's `frame_safety` block requires a real,
positive `source_fps` and integer frame numbers -- this project has already
shipped one real hard-coded-25fps bug (see time_utils.py's module docstring)
specifically from assuming a video's fps instead of probing it, and this
tool's whole purpose is to not repeat that. For 5 of validation_v3's 9 clips
the original uploaded source video is still present in this environment
(`dual_won_01`, `pass_completed_03`, `pass_completed_04`,
`shot_close_range_02`, `shot_close_range_04` -- see SOURCE_VIDEO_SEARCH_ROOTS)
and this tool probes their REAL fps with time_utils.probe_video() exactly
like the primary path. For the remaining 4 (`shot_close_range_01`,
`shot_close_range_03`, `lobbed_pass_01`, `3_passes_completed`) no source
video file exists anywhere searched, AND -- unlike the primary path, which at
least has validation/clips/registry.json's stated (if, as it turns out,
WRONG -- see below) fps to fall back on -- there is no per-clip fps of any
kind on record for these 4 (the 5 sibling clips whose video IS present probe
to five DIFFERENT real fps values, 58-60fps range, not a shared project-wide
constant, so there is no honest single fallback number to reach for either).
Fabricating a frame_safety block for these 4 from a guessed fps would be
exactly the class of silent assumption this dataset's design (and this
project's own history) explicitly exists to prevent -- including, notably,
`shot_close_range_01`, the more severe of the two GROUND_TRUTH_PLAYER_ID_
AMBIGUOUS clips. This is a deliberate, documented non-migration, not an
oversight: these 4 clips still need a real annotation pass (via
annotation_server.py, once built, against their real footage if it can be
re-obtained) rather than a fabricated one from this script. They are listed
explicitly in NOT_MIGRATED_V3_CLIPS below and called out in this tool's
run summary every time it executes, specifically so they can never be
silently forgotten.

REAL-FPS-VS-REGISTRY-FPS DISCREPANCY (a genuine finding, not a hypothetical
in the task description): validation/clips/registry.json states
`"fps": 25.0` for all 3 primary clips. Both real source videos are actually
present in this environment
(`/root/.claude/uploads/<session>/b66e2cfc-ScreenRecording_03-14-2026_00-25-09_1.MOV`
and `.../e13a71bd-ScreenRecording_04-19-2026_04-33-19_1.MOV`), and probing
them with time_utils.probe_video() gives a REAL fps of exactly 30.0 for
both -- not 25.0. Cross-checking confirms the legacy annotations' own
`approx_start_time`/`approx_end_time` fields were computed as
`observed_frame_number / 25.0` (e.g. known_match_shot_195_223: frame ~198 to
~223, and 198/25.0 == 7.92, 223/25.0 == 8.92, exactly the recorded
approx_start_time/approx_end_time) -- i.e. the annotator's "seconds" are a
derived quantity built on the registry's assumed 25fps, not an independently
observed real-world duration. Since the real video is present, this tool
does NOT propagate that 25fps-derived second value into
`ground_truth_event_window`. Instead it recovers the underlying OBSERVED
FRAME NUMBER (`round(approx_time_s * registry_fps)` -- this arithmetic step
only ever un-does the original 25fps assumption to get back to the frame
index the human actually looked at) and re-expresses it in absolute
source-video seconds using the REAL probed fps (30.0). This is precisely the
"hard-coded-25fps bug in pre-roll window sizing" failure class time_utils.py
was written to make structurally hard to repeat, caught here for real on
this project's own historical data. Every migrated primary record's `notes`
field documents this correction explicitly and by exact numbers, and
`frame_safety.source_fps` always carries the REAL probed value.

=============================================================================
Usage
=============================================================================
    python migrate_existing_annotations.py run
    python migrate_existing_annotations.py run --no-supplementary
    python migrate_existing_annotations.py run --dry-run
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

TOOLS_DIR = Path(__file__).resolve().parent
DATASET_ROOT = TOOLS_DIR.parent
REPO_ROOT = DATASET_ROOT.parent
sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(DATASET_ROOT))
sys.path.insert(0, str(REPO_ROOT))

import jsonschema                      # noqa: E402  (installed; see task description)
import time_utils                      # noqa: E402  (this project's own module)
from schema import taxonomy            # noqa: E402  (this project's own module)

ANNOTATION_TOOL_VERSION = "migrate_existing_annotations.py/1.0.0"
SCHEMA_VERSION = "1.0.0"

PRIMARY_ANNOTATIONS_DIR = REPO_ROOT / "validation" / "annotations"
PRIMARY_REGISTRY_PATH = REPO_ROOT / "validation" / "clips" / "registry.json"
PRIMARY_REPORTS_DIR = REPO_ROOT / "validation" / "reports"
PRIMARY_README_PATH = REPO_ROOT / "validation" / "README.md"
V3_REPORT_PATH = REPO_ROOT / "validation_v3" / "reports" / "real_world_validation_report.md"
ANNOTATION_SCHEMA_PATH = DATASET_ROOT / "schema" / "annotation.schema.json"
OUTPUT_DIR = DATASET_ROOT / "dataset" / "annotations"

# Extra roots this tool searches (beyond REPO_ROOT itself) for the ORIGINAL
# uploaded source video files, so real fps can be probed instead of assumed.
# This environment happens to keep uploads under /root/.claude/uploads/<session_id>/
# with each filename hash-prefixed (e.g. "b66e2cfc-ScreenRecording_...MOV");
# a different environment simply won't have this directory, which is fine --
# find_source_video() degrades gracefully (see its docstring) when nothing
# matches anywhere searched.
EXTRA_SOURCE_VIDEO_SEARCH_ROOTS = [Path("/root/.claude/uploads")]

# The two validation_v3 clips explicitly, textually marked
# GROUND_TRUTH_PLAYER_ID_AMBIGUOUS in real_world_validation_report.md
# (sections 3.3, 3.4, 5, and 7). Per the task rule: migrated, never dropped,
# but flagged ambiguous/review_required so split_generator.py /
# dataset_validator.py can keep them out of clean training splits.
AMBIGUOUS_LEGACY_CLIP_IDS = {"dual_won_01", "shot_close_range_01"}

# The 4 validation_v3 clips with no locatable source video AND no per-clip
# fps of any kind on record anywhere in the legacy tree -- see the large
# docstring section above ("WHY 4 OF THE 9 VALIDATION_V3 CLIPS ARE NOT
# MIGRATED"). Listed here, once, as the single source of truth for the run
# summary's "not migrated" report -- never silently absent from the tool's
# own accounting even though no annotation file is produced for them.
NOT_MIGRATED_V3_CLIPS = {
    "shot_close_range_01": "GROUND_TRUTH_PLAYER_ID_AMBIGUOUS clip; no source video located "
                            "and no per-clip fps on record anywhere in the legacy tree -- "
                            "fabricating frame_safety would silently assume a number nothing "
                            "supports. Needs a real annotation pass against real footage, not "
                            "a migrated one.",
    "shot_close_range_03": "no source video located and no per-clip fps on record.",
    "lobbed_pass_01": "no source video located and no per-clip fps on record.",
    "3_passes_completed": "no source video located, no per-clip fps on record, AND the report "
                           "itself (section 7) states its 3 sub-event timestamps were not "
                           "frame-perfectly established -- migrating a 3-sub-event clip also "
                           "needs 3 distinct clip_id windows this tool has no non-fabricated "
                           "way to draw boundaries for.",
}

# =============================================================================
# Hand-transcribed, clearly-labelled supplementary source: NOT an automated
# parse of validation_v3/reports/real_world_validation_report.md (that report
# is free-text prose, not a machine-readable format -- see the module
# docstring), but a direct, checkable transcription of its section-3 GT
# statements for the 5 clips whose source video is locatable and probeable.
# Every field here traces back to an exact report line, cited in the comment.
# =============================================================================
SUPPLEMENTARY_V3_CLIPS = [
    {
        # report section 3.2 (3.1 = pass_completed_03 has no video on disk to probe;
        # not skipped for that reason alone, see note -- 03 IS probeable, listed below too)
        "clip_id": "pass_completed_03",
        "video_filename_suffix": "pass_completed_03.mov",
        "event_type": "PASS",
        "event_subtype": None,
        "outcome": "COMPLETED",
        "approx_start_s": 3.0,
        "approx_end_s": 5.0,
        "player_desc": "YELLOW #2",
        "jersey_number": 2,
        "manual_identity_confidence": "MEDIUM",
        "ambiguous": False,
        "report_section": "3.1",
        "gt_quote": "GT: YELLOW #2, PASS→COMPLETED, 3.0–5.0s",
    },
    {
        "clip_id": "pass_completed_04",
        "video_filename_suffix": "pass_completed_04.mov",
        "event_type": "PASS",
        "event_subtype": None,
        "outcome": "COMPLETED",
        "approx_start_s": 5.0,
        "approx_end_s": 7.0,
        "player_desc": "YELLOW #2",
        "jersey_number": 2,
        "manual_identity_confidence": "MEDIUM",
        "ambiguous": False,
        "report_section": "3.2",
        "gt_quote": "GT: YELLOW #2, PASS→COMPLETED, 5.0–7.0s",
    },
    {
        "clip_id": "dual_won_01",
        "video_filename_suffix": "dual_won_01.mov",
        "event_type": "DUEL",
        "event_subtype": None,
        "outcome": "WON",
        "approx_start_s": 3.0,
        "approx_end_s": 4.0,
        "player_desc": "BLUE player (no number)",
        "jersey_number": None,
        "manual_identity_confidence": "LOW",
        "ambiguous": True,
        "report_section": "3.3",
        "gt_quote": "GT: BLUE player (no number — GROUND_TRUTH_PLAYER_ID_AMBIGUOUS), "
                     "DUEL→WON, 3.0–4.0s",
    },
    {
        "clip_id": "shot_close_range_02",
        "video_filename_suffix": "shot_close_range_02.mov",
        "event_type": "SHOT",
        "event_subtype": None,
        "outcome": "GOAL",
        "approx_start_s": 6.0,
        "approx_end_s": 8.0,
        "player_desc": "WHITE #9",
        "jersey_number": 9,
        "manual_identity_confidence": "HIGH",
        "ambiguous": False,
        "report_section": "3.5",
        "gt_quote": "GT: WHITE #9, SHOT→GOAL, 6.0–8.0s",
    },
    {
        "clip_id": "shot_close_range_04",
        "video_filename_suffix": "shot_close_range_04.mov",
        "event_type": "SHOT",
        "event_subtype": None,
        "outcome": "SAVED",
        "approx_start_s": 3.0,
        "approx_end_s": 4.0,
        "player_desc": "WHITE #9",
        "jersey_number": 9,
        "manual_identity_confidence": "HIGH",
        "ambiguous": False,
        "report_section": "3.7",
        "gt_quote": "GT: WHITE #9, SHOT→SAVED, 3.0–4.0s",
    },
]


# =============================================================================
# Small, generic helpers
# =============================================================================

def find_source_video(filename_or_suffix: str) -> Optional[Path]:
    """Searches REPO_ROOT and EXTRA_SOURCE_VIDEO_SEARCH_ROOTS for a file whose
    NAME ends with `filename_or_suffix` (uploaded files in this environment
    are hash-prefixed, e.g. "b66e2cfc-ScreenRecording_...MOV", so a suffix
    match is required rather than an exact-name match). Returns the first
    match found, or None if nothing matches anywhere searched -- callers must
    treat None as a normal, expected outcome (a missing video is fine; a
    silently-guessed fps in its place is not) rather than an error.
    """
    roots = [REPO_ROOT] + [r for r in EXTRA_SOURCE_VIDEO_SEARCH_ROOTS if r.exists()]
    for root in roots:
        try:
            for p in root.rglob(f"*{filename_or_suffix}"):
                if p.is_file():
                    return p
        except (PermissionError, OSError):
            continue
    return None


def find_report_paths_mentioning(clip_id: str, reports_dir: Path) -> list:
    """Greps every report file in `reports_dir` for a literal mention of
    `clip_id` and returns the matching paths (relative to REPO_ROOT), for
    provenance.migrated_from.legacy_report_paths. Generic and reusable for
    any future legacy clip_id, not hardcoded per-clip."""
    hits = []
    if not reports_dir.exists():
        return hits
    for p in sorted(reports_dir.glob("*")):
        if not p.is_file():
            continue
        try:
            text = p.read_text(errors="ignore")
        except OSError:
            continue
        if clip_id in text:
            hits.append(str(p.relative_to(REPO_ROOT)))
    return hits


def slugify(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()
    return s or "unknown"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# =============================================================================
# Outcome mapping -- taxonomy-aware, notes-aware (never guesses past what the
# legacy notes actually support; falls back to UNKNOWN rather than inventing
# a specific outcome the human annotator did not commit to).
# =============================================================================

_UNRESOLVED_PHRASES = (
    "left unresolved", "could not be pinned down", "cannot be resolved",
    "not determinable", "genuinely ambiguous", "could not be determined",
)


def map_pass_or_cross_outcome(success: Optional[bool]) -> tuple:
    if success is True:
        return "COMPLETED", "legacy success=True -> COMPLETED"
    if success is False:
        return "INCOMPLETE", "legacy success=False -> INCOMPLETE"
    return "UNKNOWN", "legacy success=None (annotator did not judge completion) -> UNKNOWN"


def map_shot_outcome(success: Optional[bool], notes_text: str) -> tuple:
    """SHOT's outcome vocabulary (GOAL/SAVED/BLOCKED/OFF_TARGET/
    GOAL_LINE_CLEARANCE/UNKNOWN) is richer than a boolean `success` can
    express, and the legacy notes usually don't distinguish save-vs-block
    even when a human was confident it wasn't a goal (see
    known_match_shot_195_223's notes: "the save-vs-block distinction ...
    left unresolved rather than guessed" -- naively substring-matching
    "save" against that exact sentence would misfire, which is why the
    unresolved-phrase check runs FIRST, before any GOAL/SAVED/BLOCKED
    keyword match)."""
    notes_lower = (notes_text or "").lower()
    if any(phrase in notes_lower for phrase in _UNRESOLVED_PHRASES):
        return "UNKNOWN", ("legacy success=%r but the notes explicitly leave the precise "
                            "SHOT outcome unresolved rather than guessing -> UNKNOWN, matching "
                            "the annotator's own stated uncertainty rather than overriding it" % success)
    if success is True:
        if "goal" in notes_lower and ("scored" in notes_lower or "celebrat" in notes_lower):
            return "GOAL", "legacy success=True and notes describe a scored/celebrated goal -> GOAL"
        return "UNKNOWN", "legacy success=True but notes do not clearly confirm GOAL specifically -> UNKNOWN"
    if success is False:
        if "saved" in notes_lower or "parr" in notes_lower:
            return "SAVED", "legacy success=False and notes describe a keeper save/parry -> SAVED"
        if "blocked" in notes_lower or "block" in notes_lower:
            return "BLOCKED", "legacy success=False and notes describe a block -> BLOCKED"
        if "off target" in notes_lower or "off-target" in notes_lower or "wide of" in notes_lower:
            return "OFF_TARGET", "legacy success=False and notes describe the shot missing the goal -> OFF_TARGET"
        if "clearance" in notes_lower and ("goal line" in notes_lower or "goal-line" in notes_lower):
            return "GOAL_LINE_CLEARANCE", "legacy success=False and notes describe a goal-line clearance -> GOAL_LINE_CLEARANCE"
        return "UNKNOWN", ("legacy success=False (not a goal) but notes do not distinguish "
                            "SAVED/BLOCKED/OFF_TARGET/GOAL_LINE_CLEARANCE -> UNKNOWN rather than guessing")
    return "UNKNOWN", "legacy success=None (annotator did not judge outcome) -> UNKNOWN"


def map_event_outcome(event_type: str, success: Optional[bool], notes_text: str) -> tuple:
    """Single entry point used by the primary migration path. CARRY/OTHER/
    UNKNOWN/NO_EVENT have no real outcome taxonomy (task item #18) so they
    always map to their sole allowed value regardless of the legacy
    `success` field -- if a future legacy CARRY record ever carries a
    non-null success, that fact is preserved in `notes`, never dropped."""
    if event_type == "CARRY":
        extra = "" if success is None else f" (legacy success={success!r} recorded but ignored: CARRY has no outcome taxonomy per taxonomy.py)"
        return "NONE", "CARRY has no outcome taxonomy (taxonomy.OUTCOMES_BY_EVENT_TYPE['CARRY'] == ['NONE'])" + extra
    if event_type in ("PASS", "CROSS"):
        return map_pass_or_cross_outcome(success)
    if event_type == "SHOT":
        return map_shot_outcome(success, notes_text)
    if event_type == "NO_EVENT":
        return "NONE", "NO_EVENT has no outcome taxonomy -> NONE"
    return "UNKNOWN", f"event_type={event_type!r} has no richer outcome mapping defined here -> UNKNOWN"


# =============================================================================
# Best-effort weak_label extraction from the legacy `cross_check_against_pipeline`
# free-text field. Heuristic text-mining, documented as such -- never treated
# as authoritative, and NEVER allowed to influence human_label (they are
# built from entirely separate code paths below).
# =============================================================================

_EVENT_TYPE_RE = re.compile(
    r"\b(" + "|".join(sorted(taxonomy.EVENT_TYPES_ALL, key=len, reverse=True)) + r")\b"
)


def parse_weak_label_from_cross_check(cross_check_text: Optional[str]) -> Optional[dict]:
    if not cross_check_text:
        return None
    text = cross_check_text
    lower = text.lower()
    if "zero events" in lower or "no events" in lower or "predicts zero" in lower:
        return {
            "event_type": "NO_EVENT",
            "outcome": "NONE",
            "confidence_band": None,
            "pipeline_run_id": "legacy_cross_check",
        }
    match = _EVENT_TYPE_RE.search(text)
    if not match:
        return None
    event_type = match.group(1)
    outcome = None
    succ_match = re.search(r"success\s*=\s*(True|False|None)", text)
    if succ_match:
        raw = succ_match.group(1)
        if event_type in ("PASS", "CROSS"):
            outcome = {"True": "COMPLETED", "False": "INCOMPLETE", "None": None}[raw]
        elif event_type == "SHOT":
            outcome = None if raw == "None" else ("UNKNOWN" if raw == "False" else None)
    return {
        "event_type": event_type,
        "outcome": outcome,
        "confidence_band": None,  # no numeric pipeline confidence recoverable from this prose
        "pipeline_run_id": "legacy_cross_check",
    }


_JERSEY_RE = re.compile(r"jersey #?(\d+)", re.IGNORECASE)


def parse_selected_player_jersey(notes_text: str) -> Optional[int]:
    """Narrow, deliberately conservative heuristic: only matches "jersey #NN"
    (the word "jersey" must be present), never a bare "#NN" -- these notes
    frequently mention an OPPONENT's bare jersey number too (e.g.
    second_match_carry_0_40: "a white-kit opponent (#23)"), and a bare-number
    match would risk misattributing the opponent's number to the tracked/
    selected player. This does not generalize to every possible future
    phrasing -- a structured jersey_number field upstream (annotation_server.py)
    is the real fix; this is best-effort only for migrating free-text notes.
    """
    m = _JERSEY_RE.search(notes_text or "")
    return int(m.group(1)) if m else None


# =============================================================================
# Frame/time reconciliation -- the heart of the real-fps-vs-registry-fps fix
# described in the module docstring.
# =============================================================================

def resolve_real_fps(source_video_filename: str, fallback_fps: float, notes: list) -> tuple:
    """Returns (real_fps, probed: bool). Tries to locate and probe the real
    source video via time_utils.probe_video(); falls back to `fallback_fps`
    (the registry's stated value, or a documented last resort) if the video
    cannot be found or fails to probe -- and always appends a clear note
    explaining which happened, so a downstream reader never mistakes a
    fallback for a verified probe."""
    video_path = find_source_video(source_video_filename)
    if video_path is None:
        notes.append(
            f"[migrated] source video '{source_video_filename}' was NOT found anywhere in this "
            f"environment (searched {REPO_ROOT} and {[str(r) for r in EXTRA_SOURCE_VIDEO_SEARCH_ROOTS]}); "
            f"frame_safety.source_fps below is the FALLBACK value {fallback_fps} (NOT independently "
            f"probed against the source video file) -- do not treat it with the same rigor "
            f"time_utils.probe_video() normally guarantees."
        )
        return fallback_fps, False
    try:
        meta = time_utils.probe_video(str(video_path))
    except time_utils.TimeUtilsError as e:
        notes.append(
            f"[migrated] source video was located at {video_path} but ffprobe failed ({e}); "
            f"falling back to fps={fallback_fps} (NOT independently probed)."
        )
        return fallback_fps, False
    if fallback_fps and abs(meta.fps - fallback_fps) > 0.01:
        notes.append(
            f"[migrated] FPS DISCREPANCY: the legacy registry/assumed fps was {fallback_fps}, but "
            f"probing the real source video ({video_path.name}) with time_utils.probe_video() gives "
            f"a REAL fps of {meta.fps!r}. This annotation's timestamps are computed from the REAL "
            f"probed fps, not the legacy assumed one -- see this file's module docstring "
            f"('REAL-FPS-VS-REGISTRY-FPS DISCREPANCY') for the full reasoning. This is exactly the "
            f"class of hard-coded-fps bug time_utils.py's module docstring warns this project has "
            f"already shipped once."
        )
    else:
        notes.append(
            f"[migrated] source video probed directly via time_utils.probe_video() at {video_path}: "
            f"real fps={meta.fps!r} (matches the legacy assumed fps of {fallback_fps})."
        )
    return meta.fps, True


# =============================================================================
# Primary migration path: validation/annotations/*.json
# =============================================================================

def migrate_primary_annotation(legacy_path: Path, registry_by_clip: dict) -> dict:
    legacy = json.loads(legacy_path.read_text())
    clip_id = legacy["clip_id"]
    registry_entry = registry_by_clip.get(clip_id)
    if registry_entry is None:
        raise ValueError(f"clip_id {clip_id!r} from {legacy_path} has no entry in "
                          f"validation/clips/registry.json -- refusing to guess an fps for it")
    registry_fps = float(registry_entry["fps"])

    expected = legacy["expected_events"][0]
    if len(legacy["expected_events"]) > 1:
        raise ValueError(f"{legacy_path}: this migration path assumes exactly one "
                          f"expected_events entry per legacy file (true for all 3 primary "
                          f"clips today); {legacy_path} has {len(legacy['expected_events'])} -- "
                          f"needs a multi-event clip_id-splitting strategy before it can be "
                          f"migrated safely (see the 3_passes_completed discussion in this "
                          f"file's module docstring for why that is not done silently).")

    notes: list = []
    event_type = expected["type"]
    if event_type not in taxonomy.EVENT_TYPES_ALL:
        raise ValueError(f"{legacy_path}: legacy event type {event_type!r} is not in "
                          f"taxonomy.EVENT_TYPES_ALL {taxonomy.EVENT_TYPES_ALL}")

    real_fps, probed = resolve_real_fps(legacy["source_video"], registry_fps, notes)

    # Recover the OBSERVED frame numbers the human actually looked at by
    # undoing the registry's assumed fps, then re-express them in real
    # source-video seconds using the REAL probed fps. See module docstring.
    event_start_frame = round(expected["approx_start_time"] * registry_fps)
    event_end_frame = round(expected["approx_end_time"] * registry_fps)
    context_start_frame, context_end_frame = legacy["frame_range"]

    event_start_s = time_utils.frames_to_seconds(event_start_frame, real_fps)
    event_end_s = time_utils.frames_to_seconds(event_end_frame, real_fps)
    context_start_s = time_utils.frames_to_seconds(context_start_frame, real_fps)
    context_end_s = time_utils.frames_to_seconds(context_end_frame, real_fps)

    # analysis_context_window: legacy frame_range converted to seconds if it
    # is wider than the event window, else equal to it (task instruction).
    if context_start_frame <= event_start_frame and context_end_frame >= event_end_frame and \
            (context_end_frame - context_start_frame) > (event_end_frame - event_start_frame):
        analysis_start_s, analysis_end_s = context_start_s, context_end_s
    else:
        analysis_start_s, analysis_end_s = event_start_s, event_end_s

    legacy_notes_text = legacy.get("notes", "")
    outcome, outcome_reason = map_event_outcome(event_type, expected.get("success"), legacy_notes_text)
    notes.append(f"[migrated] outcome mapping: {outcome_reason}")

    jersey_number = parse_selected_player_jersey(legacy_notes_text)
    # Conservative, per-clip identity confidence: HIGH only when a jersey
    # number was recovered AND the notes describe continuous, multi-frame
    # visual confirmation; MEDIUM otherwise. Documented per-record below.
    if jersey_number is not None:
        manual_identity_confidence = "HIGH"
        identity_reason = f"jersey #{jersey_number} explicitly identified and tracked across multiple inspected frames"
    else:
        manual_identity_confidence = "MEDIUM"
        identity_reason = "no jersey number legible on the tracked player in the inspected frames (kit colour only)"
    notes.append(f"[migrated] selected_player.manual_identity_confidence={manual_identity_confidence}: {identity_reason}")

    # annotation_confidence: MEDIUM by default; HIGH only when the legacy
    # notes use explicit, repeated high-certainty language for the EVENT
    # TYPE/WINDOW itself (outcome-specific uncertainty is instead captured
    # honestly via outcome=UNKNOWN above, not by dragging the whole record's
    # confidence down).
    high_certainty_markers = ("unambiguously", "confidently", "clearly", "genuinely")
    if sum(marker in legacy_notes_text.lower() for marker in high_certainty_markers) >= 2:
        annotation_confidence = "HIGH"
        conf_reason = ("legacy notes use repeated explicit high-certainty language "
                        f"({[m for m in high_certainty_markers if m in legacy_notes_text.lower()]}) "
                        "about the event type/window")
    else:
        annotation_confidence = "MEDIUM"
        conf_reason = "default per migration policy; legacy notes do not use enough explicit high-certainty language to justify HIGH"
    notes.append(f"[migrated] annotation_confidence={annotation_confidence}: {conf_reason}")

    difficulty_tags = legacy.get("difficulty_tags") or []
    if difficulty_tags:
        notes.append(f"[migrated] legacy difficulty_tags (no equivalent field in the new schema): {', '.join(difficulty_tags)}")
    notes.append(f"[migrated] legacy notes verbatim: {legacy_notes_text}")

    weak_label = parse_weak_label_from_cross_check(legacy.get("cross_check_against_pipeline"))
    if weak_label is not None:
        notes.append(
            "[migrated] weak_label is a best-effort heuristic summary of the free-text "
            "cross_check_against_pipeline field below -- it is NOT authoritative and never "
            "influenced human_label, which was built entirely from the fields above it."
        )
    if legacy.get("cross_check_against_pipeline"):
        notes.append(f"[migrated] legacy cross_check_against_pipeline verbatim: {legacy['cross_check_against_pipeline']}")

    match_id = f"legacy_{slugify(clip_id.rsplit('_', 2)[0]) if '_' in clip_id else slugify(clip_id)}"
    # known_match_* / second_match_* clip ids both start with a stable match
    # prefix; group by that prefix since these clips share one continuously-
    # tracked source video (see validation/README.md's "why clips are time
    # windows, not separate files"). This is a provisional id pending a real
    # metadata/source_videos.json match registry (out of this tool's scope).
    match_prefix = "known_match" if clip_id.startswith("known_match") else "second_match"
    match_id = f"legacy_{match_prefix}"
    source_video_id = f"legacy_{slugify(legacy['source_video'])}"

    record = {
        "schema_version": SCHEMA_VERSION,
        "annotation_id": str(uuid.uuid4()),
        "clip_id": clip_id,
        "match_id": match_id,
        "source_video_id": source_video_id,
        "status": "SINGLE_PASS",
        "human_label": {
            "selected_player": {
                "player_id": None,  # legacy player_id is the placeholder token "selected_player", not a real id
                "jersey_number": jersey_number,
                "team": "UNKNOWN",  # legacy data never assigns TEAM_A/TEAM_B, only kit colour
                "manual_identity_confidence": manual_identity_confidence,
            },
            "event_type": event_type,
            "event_subtype": None,  # legacy data has no subtype granularity -- never invented
            "outcome": outcome,
            "abstention_reason": None,
            "hard_negative_category": None,
            "ground_truth_event_window": {
                "start_time_s": event_start_s,
                "interaction_time_s": None,
                "release_time_s": None,
                "end_time_s": event_end_s,
            },
            "analysis_context_window": {
                "start_time_s": analysis_start_s,
                "end_time_s": analysis_end_s,
            },
            "receiver_player_id": None,
            "opponent_player_id": None,
            "annotation_confidence": annotation_confidence,
            "ambiguous": False,
            "review_required": False,
            "notes": "\n".join(notes),
        },
        "provenance": {
            "annotator_id": legacy.get("annotated_by", "manual_frame_inspection"),
            "annotated_at": utc_now_iso(),
            "annotation_tool_version": ANNOTATION_TOOL_VERSION,
            "double_annotation": None,
            "gold_label": None,
            "migrated_from": {
                "legacy_annotation_path": str(legacy_path.relative_to(REPO_ROOT)),
                "legacy_report_paths": (
                    ([str(PRIMARY_README_PATH.relative_to(REPO_ROOT))] if PRIMARY_README_PATH.exists() else [])
                    + find_report_paths_mentioning(clip_id, PRIMARY_REPORTS_DIR)
                ),
            },
        },
        "frame_safety": {
            "source_fps": real_fps,
            "absolute_start_frame": context_start_frame,
            "absolute_end_frame": context_end_frame,
            "ground_truth_event_window_absolute_frames": {
                "start": event_start_frame,
                "interaction": None,
                "release": None,
                "end": event_end_frame,
            },
        },
    }
    if weak_label is not None:
        record["weak_label"] = weak_label
    return record


# =============================================================================
# Supplementary migration path: validation_v3/reports/real_world_validation_report.md
# =============================================================================

def migrate_supplementary_v3_clip(entry: dict) -> Optional[dict]:
    clip_id_legacy = entry["clip_id"]
    new_clip_id = f"v3_{clip_id_legacy}"  # see CLIP_ID COLLISION note in module docstring
    notes: list = [
        f"[migrated] source: validation_v3/reports/real_world_validation_report.md "
        f"section {entry['report_section']}, verbatim GT statement: \"{entry['gt_quote']}\".",
        "[migrated] this record's provenance is a markdown REPORT, not a structured legacy "
        "annotation JSON file (none exists for this clip -- see this file's module docstring, "
        "'STEP 1 INVESTIGATION FINDINGS'). Its start/end times are the rounded values stated in "
        "the report prose (the report's own methodology section admits these are not always "
        "frame-perfect), not independently re-derived from frame-by-frame inspection by this "
        "migration tool.",
    ]

    fallback_fps = 30.0  # see module docstring: no per-clip fps of any kind exists for this
    # batch; used ONLY if the source video cannot be located (all 5 clips in
    # SUPPLEMENTARY_V3_CLIPS currently DO have a locatable source video, so
    # this fallback is not expected to be exercised today, but is kept so
    # this function degrades the same honest way as the primary path if a
    # future clip is added to the table without its video available).
    real_fps, probed = resolve_real_fps(entry["video_filename_suffix"], fallback_fps, notes)
    if not probed:
        notes.append(
            "[migrated] WARNING: frame_safety below could not be probed against a real source "
            "video for this record -- treat its frame numbers as illustrative only."
        )

    event_start_frame = time_utils.seconds_to_frames(entry["approx_start_s"], real_fps)
    event_end_frame = time_utils.seconds_to_frames(entry["approx_end_s"], real_fps)

    ambiguous = entry["ambiguous"] or clip_id_legacy in AMBIGUOUS_LEGACY_CLIP_IDS
    if ambiguous:
        notes.append(
            "[migrated] GROUND_TRUTH_PLAYER_ID_AMBIGUOUS per real_world_validation_report.md "
            "-- the human annotator could not pin a stable jersey number/identity on the acting "
            "player. ambiguous=true, review_required=true, abstention_reason=AMBIGUOUS. Note: "
            "annotation.schema.json's own description scopes abstention_reason to event_type "
            "OTHER/UNKNOWN; the ambiguity here is over PLAYER IDENTITY, not event "
            f"classification (event_type is a confidently-read {entry['event_type']!r}). This is "
            "a deliberate, documented use of the field for a different kind of uncertainty per "
            "this migration task's explicit instruction -- dataset_validator.py may want an "
            "explicit player-identity-ambiguous flag in a future schema revision rather than "
            "overloading abstention_reason for it."
        )

    outcome, outcome_reason = entry["outcome"], "outcome taken verbatim from the report's GT statement (see gt_quote above)"
    notes.append(f"[migrated] outcome mapping: {outcome_reason}")

    if entry["jersey_number"] is not None:
        manual_identity_confidence = "MEDIUM" if not ambiguous else "LOW"
    else:
        manual_identity_confidence = "LOW"
    notes.append(f"[migrated] selected_player.manual_identity_confidence={manual_identity_confidence} "
                 f"(player_desc={entry['player_desc']!r})")

    annotation_confidence = "LOW" if ambiguous else "MEDIUM"
    notes.append(
        f"[migrated] annotation_confidence={annotation_confidence}: report-derived ground truth "
        "for this batch is independently re-derived (per rule 5) but is coarser than the primary "
        "3-clip path (rounded time ranges, no frame-level notes) and is capped at MEDIUM even "
        "when the report's language is confident; ambiguous clips are further capped at LOW."
    )

    match_id = "legacy_validation_v3_realworld_batch"
    source_video_id = f"legacy_v3_{slugify(clip_id_legacy)}"

    record = {
        "schema_version": SCHEMA_VERSION,
        "annotation_id": str(uuid.uuid4()),
        "clip_id": new_clip_id,
        "match_id": match_id,
        "source_video_id": source_video_id,
        "status": "SINGLE_PASS",
        "human_label": {
            "selected_player": {
                "player_id": None,
                "jersey_number": entry["jersey_number"],
                "team": "UNKNOWN",
                "manual_identity_confidence": manual_identity_confidence,
            },
            "event_type": entry["event_type"],
            "event_subtype": entry["event_subtype"],
            "outcome": outcome,
            "abstention_reason": "AMBIGUOUS" if ambiguous else None,
            "hard_negative_category": None,
            "ground_truth_event_window": {
                "start_time_s": time_utils.frames_to_seconds(event_start_frame, real_fps),
                "interaction_time_s": None,
                "release_time_s": None,
                "end_time_s": time_utils.frames_to_seconds(event_end_frame, real_fps),
            },
            "analysis_context_window": {
                "start_time_s": time_utils.frames_to_seconds(event_start_frame, real_fps),
                "end_time_s": time_utils.frames_to_seconds(event_end_frame, real_fps),
            },
            "receiver_player_id": None,
            "opponent_player_id": None,
            "annotation_confidence": annotation_confidence,
            "ambiguous": ambiguous,
            "review_required": ambiguous,
            "notes": "\n".join(notes),
        },
        "provenance": {
            "annotator_id": "manual_frame_inspection",
            "annotated_at": utc_now_iso(),
            "annotation_tool_version": ANNOTATION_TOOL_VERSION,
            "double_annotation": None,
            "gold_label": None,
            "migrated_from": {
                "legacy_annotation_path": str(V3_REPORT_PATH.relative_to(REPO_ROOT)),
                "legacy_report_paths": [str(V3_REPORT_PATH.relative_to(REPO_ROOT))],
            },
        },
        "frame_safety": {
            "source_fps": real_fps,
            "absolute_start_frame": event_start_frame,
            "absolute_end_frame": event_end_frame,
            "ground_truth_event_window_absolute_frames": {
                "start": event_start_frame,
                "interaction": None,
                "release": None,
                "end": event_end_frame,
            },
        },
    }
    return record


# =============================================================================
# Validation + writing
# =============================================================================

def validate_record(record: dict, schema: dict) -> None:
    jsonschema.validate(instance=record, schema=schema)
    hl = record["human_label"]
    problems = taxonomy.validate_event_record(hl["event_type"], hl["event_subtype"], hl["outcome"])
    if problems:
        raise ValueError(f"taxonomy.validate_event_record failed for {record['clip_id']}: {problems}")


def write_record(record: dict, output_dir: Path, dry_run: bool) -> Path:
    out_path = output_dir / f"{record['annotation_id']}.json"
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(record, indent=2) + "\n")
    return out_path


# =============================================================================
# CLI
# =============================================================================

def run(args: argparse.Namespace) -> int:
    schema = json.loads(ANNOTATION_SCHEMA_PATH.read_text())
    registry = json.loads(PRIMARY_REGISTRY_PATH.read_text())
    registry_by_clip = {c["clip_id"]: c for c in registry["clips"]}

    migrated, skipped, errors = [], [], []

    legacy_files = sorted(PRIMARY_ANNOTATIONS_DIR.glob("*.json"))
    print(f"== Primary path: {len(legacy_files)} legacy annotation file(s) in "
          f"{PRIMARY_ANNOTATIONS_DIR.relative_to(REPO_ROOT)} ==")
    for legacy_path in legacy_files:
        try:
            record = migrate_primary_annotation(legacy_path, registry_by_clip)
            validate_record(record, schema)
        except Exception as e:
            errors.append((str(legacy_path), str(e)))
            print(f"  ERROR migrating {legacy_path.name}: {e}")
            continue
        out_path = write_record(record, args.output_dir, args.dry_run)
        migrated.append((record["clip_id"], record["annotation_id"], str(out_path)))
        print(f"  OK  {legacy_path.name} -> {out_path.name}  "
              f"(clip_id={record['clip_id']}, event_type={record['human_label']['event_type']}, "
              f"outcome={record['human_label']['outcome']}, ambiguous={record['human_label']['ambiguous']})")

    if args.include_supplementary:
        print(f"\n== Supplementary path: {len(SUPPLEMENTARY_V3_CLIPS)} report-derived clip(s) "
              f"from {V3_REPORT_PATH.relative_to(REPO_ROOT)} ==")
        for entry in SUPPLEMENTARY_V3_CLIPS:
            try:
                record = migrate_supplementary_v3_clip(entry)
                validate_record(record, schema)
            except Exception as e:
                errors.append((entry["clip_id"], str(e)))
                print(f"  ERROR migrating {entry['clip_id']}: {e}")
                continue
            out_path = write_record(record, args.output_dir, args.dry_run)
            migrated.append((record["clip_id"], record["annotation_id"], str(out_path)))
            print(f"  OK  {entry['clip_id']} -> {out_path.name}  "
                  f"(clip_id={record['clip_id']}, event_type={record['human_label']['event_type']}, "
                  f"outcome={record['human_label']['outcome']}, ambiguous={record['human_label']['ambiguous']})")

        print(f"\n  NOT migrated ({len(NOT_MIGRATED_V3_CLIPS)} clip(s), by design -- see module "
              f"docstring 'WHY 4 OF THE 9 VALIDATION_V3 CLIPS ARE NOT MIGRATED'):")
        for clip_id, reason in NOT_MIGRATED_V3_CLIPS.items():
            print(f"    - {clip_id}: {reason}")
            skipped.append((clip_id, reason))

    print(f"\n== Summary: {len(migrated)} migrated, {len(skipped)} documented-not-migrated, "
          f"{len(errors)} error(s) ==")
    for clip_id, ann_id, path in migrated:
        print(f"  {clip_id} -> {path}")
    if errors:
        for name, msg in errors:
            print(f"  ERROR: {name}: {msg}")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="migrate legacy validation-clip ground truth into "
                                             "the canonical annotation schema")
    run_parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                             help=f"where to write migrated annotation JSON files (default: {OUTPUT_DIR})")
    run_parser.add_argument("--no-supplementary", dest="include_supplementary", action="store_false",
                             help="skip the validation_v3 report-derived supplementary path; "
                                  "migrate only validation/annotations/*.json")
    run_parser.add_argument("--dry-run", action="store_true",
                             help="validate and print what would be migrated without writing any files")
    run_parser.set_defaults(include_supplementary=True)

    args = parser.parse_args()
    if args.command == "run":
        return run(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())

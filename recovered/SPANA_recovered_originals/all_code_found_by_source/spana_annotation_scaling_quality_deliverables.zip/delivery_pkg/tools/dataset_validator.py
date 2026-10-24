#!/usr/bin/env python3
"""
dataset_validator.py -- validates every dataset/annotations/*.json record and
produces a comprehensive health-check report. This is the single tool that
ties together the schema (schema/annotation.schema.json), the vocabulary
(schema/taxonomy.py), the frame/fps safety machinery (tools/time_utils.py),
and the split-leakage auditor (tools/leakage_checker.py) into one report, so
"is the dataset healthy right now" is always a single command away rather
than something you have to reassemble from four different tools by hand.

Checks performed, and how each is scored (HARD error vs WARNING vs INFO):

  (a) Schema validity against schema/annotation.schema.json.        HARD
  (b) schema.taxonomy.validate_event_record(...) passes.            HARD
      (an invalid event_type/event_subtype/outcome combination is exactly
      as much a correctness bug as a schema violation -- see taxonomy.py's
      own docstring on why this check exists as a single source of truth.)
  (c) ground_truth_event_window timestamp ordering (start <= interaction
      [if set] <= release [if set] <= end) and containment inside
      analysis_context_window.                                       HARD
      (audit item #6/#7: the true event timing is never widened/narrowed by
      context, and a corrupted window is exactly the class of silent-frame-
      math bug time_utils.py exists to make structurally hard to write --
      an inconsistent window here is exactly as serious as a bad frame
      number.)
  (d) selected_player.manual_identity_confidence == HIGH but NEITHER
      player_id NOR jersey_number is set.                            WARNING
      SEMANTICS, CORRECTED during the infrastructure-acceptance task: this
      check originally warned on player_id alone being null, which produced
      two false-positive warnings against real, correct data (the two
      known_match_* annotations -- jersey #20 was directly, visibly
      identified across many inspected frames and IS recorded in
      selected_player.jersey_number, but player_id was correctly left null).
      Root cause: `player_id` in this schema means a STABLE, CROSS-CLIP
      DATABASE IDENTITY (a persistent player-roster ID), which THIS PROJECT
      DOES NOT HAVE -- there is no player database anywhere in
      football_analysis/ or player_tracker/, and none is in scope for this
      dataset-infrastructure work. `manual_identity_confidence == HIGH`
      correctly and only ever means "I am highly confident which bounding
      box/player is being tracked in THIS clip" -- a purely visual,
      single-clip judgement -- and `jersey_number` (or, absent a legible
      number, a clear kit/role description in `notes`) is the right and
      sufficient evidence for that claim, not `player_id`. `player_id`
      stays in the schema unchanged (a future stable roster/re-ID system
      could still populate it), but this validator no longer conflates "no
      persistent database ID" with "no visual identity evidence." A record
      now only warns here if it claims HIGH confidence with NO identifying
      information at all (no player_id AND no jersey_number) -- that
      combination genuinely is worth a human's attention.
  (e) Duplicate annotation_id across files.                          HARD
      Duplicate clip_id with byte-identical human_label content.     WARNING
      (two files sharing a clip_id is completely normal -- that is what
      double-annotation IS, audit item #22 -- but if the human_label
      content is byte-for-byte identical across the two attempts, that is
      not evidence of an independent second opinion, it's evidence of a
      duplicate submission, and worth a human's attention either way.)
  (f) Frame-range consistency against dataset/clips/<clip_id>/clip_time_map.json.
      HARD if the clip has been extracted and the annotation's frame_safety
      is inconsistent with it (beyond FRAME_TOLERANCE frames); INFO if the
      clip hasn't been extracted yet at all (a legitimate early-dataset
      state per the project brief, not a bug).
  (g) fps consistency (frame_safety.source_fps vs the clip's own recorded
      source_fps).                                                   HARD
      (audit item #7's whole reason for existing: this project has
      previously shipped a real hard-coded-fps bug; a drift between the
      fps an annotation was computed with and the fps the clip was
      actually extracted at is precisely that bug recurring.)
  (h) LOW-confidence-in-a-real-split.                                HARD
      Any exportable-status (SINGLE_PASS/DOUBLE_PASS_AGREED/GOLD_RESOLVED)
      annotation whose annotation_confidence is not in
      taxonomy.TRAINING_ELIGIBLE_CONFIDENCE, but whose match_id is assigned
      to TRAIN/VALIDATION/TEST/GOLD_TEST in dataset/splits/assignment.json.
      (audit item #21: LOW confidence must never silently ride into a real
      split.)
  (i) Class-count summary: by (event_type, outcome), by status, by
      annotation_confidence.                                         (report only)
  (j) Cross-split leakage, delegated to leakage_checker.py and folded into
      this same report (its HARD findings become HARD errors here; its
      informational findings become INFO entries here) -- see
      leakage_checker.py's own module docstring for what it checks and why.

Nothing above ever crashes on an empty or partially-populated dataset tree:
zero annotation files, a missing dataset/clips/<id>/ directory, a missing
dataset/splits/assignment.json, or a missing dataset/metadata/source_videos.json
are all treated as legitimate early states of a dataset being built up by
several tools in parallel, never as reasons to raise an unhandled exception.

Output: writes a JSON report to dataset/reports/validation_report.json and
prints a concise human-readable summary. Exit code is non-zero if and only
if at least one HARD error was found (schema violation, taxonomy violation,
timestamp/frame/fps inconsistency, duplicate annotation_id, LOW-confidence-
in-split, or a leakage finding) -- warnings and info notes alone never fail
the run.

Usage:
    python dataset_validator.py [--annotations-dir DIR] [--clips-dir DIR]
        [--splits-path PATH] [--source-videos PATH] [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import jsonschema

TOOLS_DIR = Path(__file__).resolve().parent
DATASET_ROOT = TOOLS_DIR.parent
for _p in (TOOLS_DIR, DATASET_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import time_utils                       # noqa: E402  (this project's own module)
import leakage_checker                  # noqa: E402
from schema import taxonomy             # noqa: E402

ANNOTATIONS_DIR = DATASET_ROOT / "dataset" / "annotations"
CLIPS_DIR = DATASET_ROOT / "dataset" / "clips"
SPLITS_PATH = DATASET_ROOT / "dataset" / "splits" / "assignment.json"
SOURCE_VIDEOS_PATH = DATASET_ROOT / "dataset" / "metadata" / "source_videos.json"
SCHEMA_PATH = DATASET_ROOT / "schema" / "annotation.schema.json"
REPORTS_DIR = DATASET_ROOT / "dataset" / "reports"

EXPORTABLE_STATUSES = {"SINGLE_PASS", "DOUBLE_PASS_AGREED", "GOLD_RESOLVED"}

# Small, deliberately generous tolerances -- large enough to absorb ordinary
# round-to-nearest-frame slack, small enough that a genuine fps/offset bug
# (this project's own prior incident -- see time_utils.py's module
# docstring) is still caught rather than tolerated away.
FRAME_TOLERANCE = 2
FPS_REL_TOLERANCE = 1e-4
TIME_EPSILON_S = 1e-6


class DatasetValidatorError(Exception):
    pass


# ---------------------------------------------------------------------------
# Loading helpers -- every one of these degrades gracefully on a missing or
# partially-populated file rather than raising, since other tools are
# building out these same directories in parallel.
# ---------------------------------------------------------------------------

def load_schema(schema_path: Path = SCHEMA_PATH) -> dict:
    return json.loads(schema_path.read_text())


def load_assignment(splits_path: Path = SPLITS_PATH) -> dict:
    """Loads dataset/splits/assignment.json for the LOW-confidence-in-split
    cross-reference (check h). Uses leakage_checker's own duplicate-key-safe
    loader so this tool and leakage_checker never disagree about what
    assignment.json actually says (see leakage_checker.load_assignment_file's
    docstring for why a plain json.loads is not trusted here)."""
    try:
        assignment, _dupe_findings = leakage_checker.load_assignment_file(splits_path)
    except leakage_checker.LeakageCheckerError as e:
        raise DatasetValidatorError(str(e)) from e
    return assignment


class ClipTimeMapCache:
    """Loads dataset/clips/<clip_id>/clip_time_map.json at most once per
    clip_id per validator run. Each entry is ("ok", ClipTimeMap),
    ("missing", None) -- the clip hasn't been extracted yet, a normal early
    state -- or ("error", <message>) -- the directory/sidecar exists but
    couldn't be parsed, which IS worth a warning."""

    def __init__(self, clips_dir: Path = CLIPS_DIR):
        self.clips_dir = clips_dir
        self._cache: dict[str, tuple] = {}

    def get(self, clip_id: str) -> tuple:
        if clip_id in self._cache:
            return self._cache[clip_id]
        clip_dir = self.clips_dir / clip_id
        sidecar = clip_dir / "clip_time_map.json"
        if not sidecar.exists():
            result = ("missing", None)
        else:
            try:
                result = ("ok", time_utils.load_clip_time_map(str(clip_dir)))
            except (time_utils.TimeUtilsError, json.JSONDecodeError, KeyError, OSError) as e:
                result = ("error", str(e))
        self._cache[clip_id] = result
        return result


# ---------------------------------------------------------------------------
# Per-record checks
# ---------------------------------------------------------------------------

def _check_timestamp_ordering(human: dict) -> list[str]:
    """Check (c). Returns human-readable problem strings (empty = fine).
    Deliberately tolerant of missing/null fields -- schema validation (check
    a) is what enforces the fields exist at all; this only checks ORDERING
    among whichever of them are present, so a schema-invalid record doesn't
    also spam a wall of misleading ordering complaints."""
    problems = []
    gt = human.get("ground_truth_event_window")
    ctx = human.get("analysis_context_window")
    if not isinstance(gt, dict):
        return problems

    s, i, r, e = gt.get("start_time_s"), gt.get("interaction_time_s"), gt.get("release_time_s"), gt.get("end_time_s")
    named_chain = [("start_time_s", s), ("interaction_time_s", i), ("release_time_s", r), ("end_time_s", e)]
    present = [(name, val) for name, val in named_chain if isinstance(val, (int, float))]
    for (name_a, val_a), (name_b, val_b) in zip(present, present[1:]):
        if val_a > val_b + TIME_EPSILON_S:
            problems.append(f"ground_truth_event_window.{name_a}={val_a} is after "
                             f"{name_b}={val_b} (expected start <= interaction <= release <= end)")

    if isinstance(ctx, dict):
        cs, ce = ctx.get("start_time_s"), ctx.get("end_time_s")
        if isinstance(s, (int, float)) and isinstance(cs, (int, float)) and s < cs - TIME_EPSILON_S:
            problems.append(f"ground_truth_event_window.start_time_s={s} is before "
                             f"analysis_context_window.start_time_s={cs}")
        if isinstance(e, (int, float)) and isinstance(ce, (int, float)) and e > ce + TIME_EPSILON_S:
            problems.append(f"ground_truth_event_window.end_time_s={e} is after "
                             f"analysis_context_window.end_time_s={ce}")
    return problems


def _check_frame_and_fps_consistency(human: dict, frame_safety: dict, ctm: "time_utils.ClipTimeMap") -> tuple[list, list]:
    """Checks (f) and (g). Returns (hard_problems, warnings) as human-readable
    strings. All comparisons use the ANNOTATION's own recorded source_fps
    when available (falling back to the clip's) so that check (g)'s own fps
    mismatch, if present, is reported once clearly rather than cascading
    into a dozen confusing frame-count mismatches computed with the wrong fps."""
    hard_problems: list[str] = []
    warnings: list[str] = []

    ann_fps = frame_safety.get("source_fps")
    if isinstance(ann_fps, (int, float)) and not math.isclose(ann_fps, ctm.source_fps, rel_tol=FPS_REL_TOLERANCE):
        hard_problems.append(
            f"frame_safety.source_fps={ann_fps} does not match clip_time_map.json's "
            f"source_fps={ctm.source_fps} for clip {ctm.clip_id} (audit item #7: fps must always "
            f"be read from the real video, never assumed or allowed to drift downstream)")
    fps_for_checks = ann_fps if isinstance(ann_fps, (int, float)) else ctm.source_fps

    a_start = frame_safety.get("absolute_start_frame")
    a_end = frame_safety.get("absolute_end_frame")
    if isinstance(a_start, int) and isinstance(a_end, int):
        if a_start < ctm.absolute_start_frame - FRAME_TOLERANCE or a_end > ctm.absolute_end_frame + FRAME_TOLERANCE:
            hard_problems.append(
                f"frame_safety absolute frame range [{a_start}, {a_end}] falls outside the "
                f"extracted clip's own range [{ctm.absolute_start_frame}, {ctm.absolute_end_frame}] "
                f"for clip {ctm.clip_id}")

    ctx = human.get("analysis_context_window") if isinstance(human, dict) else None
    if isinstance(ctx, dict):
        cs, ce = ctx.get("start_time_s"), ctx.get("end_time_s")
        if isinstance(cs, (int, float)) and isinstance(a_start, int):
            expected_start = time_utils.seconds_to_frames(cs, fps_for_checks)
            if abs(expected_start - a_start) > FRAME_TOLERANCE:
                hard_problems.append(
                    f"frame_safety.absolute_start_frame={a_start} is inconsistent with "
                    f"analysis_context_window.start_time_s={cs} at source_fps={fps_for_checks} "
                    f"(expected ~{expected_start}, tolerance {FRAME_TOLERANCE} frames)")
        if isinstance(ce, (int, float)) and isinstance(a_end, int):
            expected_end = time_utils.seconds_to_frames(ce, fps_for_checks)
            if abs(expected_end - a_end) > FRAME_TOLERANCE:
                hard_problems.append(
                    f"frame_safety.absolute_end_frame={a_end} is inconsistent with "
                    f"analysis_context_window.end_time_s={ce} at source_fps={fps_for_checks} "
                    f"(expected ~{expected_end}, tolerance {FRAME_TOLERANCE} frames)")

    gtf = frame_safety.get("ground_truth_event_window_absolute_frames")
    gt = human.get("ground_truth_event_window") if isinstance(human, dict) else None
    if isinstance(gtf, dict) and isinstance(gt, dict):
        for key, seconds_key in (("start", "start_time_s"), ("interaction", "interaction_time_s"),
                                  ("release", "release_time_s"), ("end", "end_time_s")):
            frame_val = gtf.get(key)
            seconds_val = gt.get(seconds_key)
            if isinstance(frame_val, int) and isinstance(seconds_val, (int, float)):
                expected = time_utils.seconds_to_frames(seconds_val, fps_for_checks)
                if abs(expected - frame_val) > FRAME_TOLERANCE:
                    hard_problems.append(
                        f"frame_safety.ground_truth_event_window_absolute_frames.{key}={frame_val} "
                        f"is inconsistent with ground_truth_event_window.{seconds_key}={seconds_val} "
                        f"at source_fps={fps_for_checks} (expected ~{expected})")
            elif frame_val is None and seconds_val is not None:
                warnings.append(f"ground_truth_event_window_absolute_frames.{key} is null but "
                                 f"{seconds_key}={seconds_val} is set")
    return hard_problems, warnings


# ---------------------------------------------------------------------------
# Main validation pass
# ---------------------------------------------------------------------------

def validate_all(annotations_dir: Path = ANNOTATIONS_DIR, clips_dir: Path = CLIPS_DIR,
                  splits_path: Path = SPLITS_PATH, source_videos_path: Path = SOURCE_VIDEOS_PATH,
                  schema_path: Path = SCHEMA_PATH) -> dict:
    schema = load_schema(schema_path)
    validator = jsonschema.Draft7Validator(schema)

    hard_errors: list[dict] = []
    warnings: list[dict] = []
    info: list[dict] = []

    annotation_id_to_paths: dict[str, list[Path]] = defaultdict(list)
    clip_id_groups: dict[str, list[tuple[Path, dict]]] = defaultdict(list)
    counts_by_event_outcome: dict[str, Counter] = defaultdict(Counter)
    counts_by_status: Counter = Counter()
    counts_by_confidence: Counter = Counter()

    ctm_cache = ClipTimeMapCache(clips_dir)
    assignment = load_assignment(splits_path)

    files = sorted(annotations_dir.glob("*.json")) if annotations_dir.exists() else []
    n_files = len(files)
    n_parsed = 0
    n_schema_valid = 0

    for path in files:
        try:
            record = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            hard_errors.append({"file": str(path), "check": "json_parse",
                                 "message": f"could not parse as JSON: {e}"})
            continue
        n_parsed += 1

        schema_errors = sorted(validator.iter_errors(record), key=str)
        if schema_errors:
            for err in schema_errors:
                hard_errors.append({
                    "file": str(path), "check": "schema",
                    "annotation_id": record.get("annotation_id") if isinstance(record, dict) else None,
                    "json_path": list(err.absolute_path),
                    "message": err.message,
                })
        else:
            n_schema_valid += 1

        if not isinstance(record, dict):
            continue  # nothing else here is safe to inspect

        annotation_id = record.get("annotation_id")
        clip_id = record.get("clip_id")
        match_id = record.get("match_id")
        status = record.get("status")
        human = record.get("human_label")
        frame_safety = record.get("frame_safety")

        if isinstance(annotation_id, str):
            annotation_id_to_paths[annotation_id].append(path)
        if isinstance(clip_id, str):
            clip_id_groups[clip_id].append((path, record))
        if isinstance(status, str):
            counts_by_status[status] += 1

        if isinstance(human, dict):
            event_type = human.get("event_type")
            event_subtype = human.get("event_subtype")
            outcome = human.get("outcome")
            confidence = human.get("annotation_confidence")

            # (b) taxonomy validity
            problems = taxonomy.validate_event_record(event_type, event_subtype, outcome)
            if problems:
                hard_errors.append({
                    "file": str(path), "check": "taxonomy", "annotation_id": annotation_id,
                    "message": "; ".join(problems),
                })

            # (c) timestamp ordering / window containment
            for problem in _check_timestamp_ordering(human):
                hard_errors.append({"file": str(path), "check": "timestamp_ordering",
                                     "annotation_id": annotation_id, "message": problem})

            # (d) missing-identity sanity -- see the corrected semantics note above:
            # jersey_number is an equally valid identity anchor to player_id (this
            # project has no persistent cross-clip player database), so only warn
            # when HIGH confidence is claimed with NEITHER field set at all.
            sel = human.get("selected_player")
            if isinstance(sel, dict) and sel.get("manual_identity_confidence") == "HIGH" \
                    and sel.get("player_id") is None and sel.get("jersey_number") is None:
                warnings.append({
                    "file": str(path), "check": "missing_identity", "annotation_id": annotation_id,
                    "message": "selected_player.manual_identity_confidence is HIGH but neither "
                               "player_id nor jersey_number is set -- no identifying evidence "
                               "recorded at all for a HIGH-confidence visual identification.",
                })

            # (i) class counts
            if event_type is not None:
                counts_by_event_outcome[event_type][outcome] += 1
            if confidence is not None:
                counts_by_confidence[confidence] += 1

            # (h) LOW-confidence-in-a-real-split
            if isinstance(status, str) and status in EXPORTABLE_STATUSES and isinstance(match_id, str):
                split = assignment.get(match_id)
                if split is not None and confidence not in taxonomy.TRAINING_ELIGIBLE_CONFIDENCE:
                    hard_errors.append({
                        "file": str(path), "check": "low_confidence_in_split",
                        "annotation_id": annotation_id, "match_id": match_id, "split": split,
                        "annotation_confidence": confidence,
                        "message": f"annotation {annotation_id} has status={status} and "
                                   f"annotation_confidence={confidence!r} (not in "
                                   f"{sorted(taxonomy.TRAINING_ELIGIBLE_CONFIDENCE)}) but match "
                                   f"{match_id} is assigned to real split {split} -- LOW confidence "
                                   f"must never silently enter a training split (audit item #21).",
                    })

        # (f)/(g) frame-range and fps consistency
        if isinstance(clip_id, str):
            ctm_status, ctm_value = ctm_cache.get(clip_id)
            if ctm_status == "missing":
                info.append({"file": str(path), "check": "clip_not_extracted", "clip_id": clip_id,
                             "annotation_id": annotation_id,
                             "message": f"clip {clip_id} has not been extracted yet "
                                        f"(no dataset/clips/{clip_id}/clip_time_map.json) -- "
                                        f"skipping frame/fps consistency checks for this record."})
            elif ctm_status == "error":
                warnings.append({"file": str(path), "check": "clip_time_map_unreadable", "clip_id": clip_id,
                                  "annotation_id": annotation_id,
                                  "message": f"dataset/clips/{clip_id}/clip_time_map.json exists but "
                                             f"could not be loaded: {ctm_value}"})
            elif isinstance(frame_safety, dict):
                hard_problems, soft_problems = _check_frame_and_fps_consistency(
                    human if isinstance(human, dict) else {}, frame_safety, ctm_value)
                for problem in hard_problems:
                    hard_errors.append({"file": str(path), "check": "frame_fps_consistency",
                                         "annotation_id": annotation_id, "clip_id": clip_id, "message": problem})
                for problem in soft_problems:
                    warnings.append({"file": str(path), "check": "frame_fps_consistency",
                                      "annotation_id": annotation_id, "clip_id": clip_id, "message": problem})

    # (e) duplicate annotation_id (HARD) and duplicate clip_id + identical content (WARNING)
    duplicate_annotation_ids = []
    for annotation_id, paths in sorted(annotation_id_to_paths.items()):
        if len(paths) > 1:
            duplicate_annotation_ids.append({"annotation_id": annotation_id, "files": [str(p) for p in paths]})
            hard_errors.append({
                "check": "duplicate_annotation_id", "annotation_id": annotation_id,
                "files": [str(p) for p in paths],
                "message": f"annotation_id {annotation_id!r} appears in {len(paths)} files -- "
                           f"annotation_ids must be globally unique across the whole dataset.",
            })

    duplicate_clip_identical_content = []
    for clip_id, group in sorted(clip_id_groups.items()):
        if len(group) < 2:
            continue
        seen: dict[str, Path] = {}
        for path, record in group:
            human = record.get("human_label")
            if not isinstance(human, dict):
                continue
            key = json.dumps(human, sort_keys=True)
            if key in seen:
                pair = {"clip_id": clip_id, "files": [str(seen[key]), str(path)]}
                duplicate_clip_identical_content.append(pair)
                warnings.append({
                    "check": "duplicate_clip_identical_content", "clip_id": clip_id,
                    "files": pair["files"],
                    "message": f"two annotation attempts for clip {clip_id} have byte-identical "
                               f"human_label content -- double-annotation is expected to produce "
                               f"INDEPENDENT judgements, so verify this isn't a duplicate submission.",
                })
            else:
                seen[key] = path

    # (j) leakage, delegated
    leakage_report = leakage_checker.build_report(annotations_dir, splits_path, source_videos_path)
    for item in leakage_report["duplicate_split_keys"]:
        hard_errors.append({"check": "leakage_duplicate_split_key", **item,
                             "message": f"{splits_path} contains duplicate key {item['match_id']!r} "
                                        f"with conflicting values {item['conflicting_values']}"})
    for item in leakage_report["invalid_split_values"]:
        hard_errors.append({"check": "leakage_invalid_split_value", **item,
                             "message": f"match {item['match_id']!r} has invalid split value {item['value']!r}"})
    for item in leakage_report["source_video_split_conflicts"]:
        hard_errors.append({"check": "leakage_source_video_split_conflict", **item,
                             "message": f"source_video_id {item['source_video_id']!r} is shared by "
                                        f"match_ids {item['match_ids']}, which resolve to different "
                                        f"splits: {item['splits_by_match_id']} -- same footage on both "
                                        f"sides of a split boundary."})
    for match_id in leakage_report["unassigned_match_ids"]:
        info.append({"check": "leakage_unassigned_match_id", "match_id": match_id,
                     "message": f"match {match_id!r} has annotations but is not yet present in "
                                f"{splits_path.name} -- run split_generator.py once it's ready to be split."})
    for item in leakage_report["source_video_shared_match_ids_same_split"]:
        info.append({"check": "leakage_source_video_shared_match_ids", **item,
                     "message": f"source_video_id {item['source_video_id']!r} is referenced by "
                                f"multiple match_ids {item['match_ids']} but they don't currently "
                                f"resolve to different splits -- not a leak yet, worth double-"
                                f"checking the match_id registration is correct."})

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "annotations_dir": str(annotations_dir),
        "n_files_scanned": n_files,
        "n_files_valid_json": n_parsed,
        "n_files_schema_valid": n_schema_valid,
        "hard_errors": hard_errors,
        "warnings": warnings,
        "info": info,
        "duplicate_annotation_ids": duplicate_annotation_ids,
        "duplicate_clip_identical_content": duplicate_clip_identical_content,
        "class_counts": {
            "by_event_type_and_outcome": {et: dict(oc) for et, oc in counts_by_event_outcome.items()},
            "by_status": dict(counts_by_status),
            "by_annotation_confidence": dict(counts_by_confidence),
        },
        "leakage": leakage_report,
        "summary": {
            "total_hard_errors": len(hard_errors),
            "total_warnings": len(warnings),
            "total_info": len(info),
            "passed": len(hard_errors) == 0,
        },
    }
    return report


def print_summary(report: dict) -> None:
    s = report["summary"]
    print(f"dataset_validator: scanned {report['n_files_scanned']} file(s) in {report['annotations_dir']}")
    print(f"  valid JSON:          {report['n_files_valid_json']}/{report['n_files_scanned']}")
    print(f"  schema-valid:        {report['n_files_schema_valid']}/{report['n_files_scanned']}")
    print(f"  HARD errors:         {s['total_hard_errors']}")
    print(f"  warnings:            {s['total_warnings']}")
    print(f"  info notes:          {s['total_info']}")
    if report["class_counts"]["by_status"]:
        print(f"  by status:           {report['class_counts']['by_status']}")
    if report["class_counts"]["by_annotation_confidence"]:
        print(f"  by confidence:       {report['class_counts']['by_annotation_confidence']}")
    if report["class_counts"]["by_event_type_and_outcome"]:
        print("  by event_type/outcome:")
        for et, by_outcome in sorted(report["class_counts"]["by_event_type_and_outcome"].items()):
            print(f"    {et}: {dict(by_outcome)}")
    print(f"  leakage: clean={report['leakage']['clean']} "
          f"(known match_ids={report['leakage']['n_known_match_ids']}, "
          f"assigned={report['leakage']['n_assigned_match_ids']})")
    if s["total_hard_errors"]:
        print("\n  first few HARD errors:")
        for e in report["hard_errors"][:10]:
            loc = e.get("annotation_id") or e.get("file") or e.get("match_id") or "?"
            print(f"    [{e.get('check')}] {loc}: {e.get('message')}")
        if s["total_hard_errors"] > 10:
            print(f"    ... and {s['total_hard_errors'] - 10} more (see the full report)")
    print(f"\n  PASSED: {s['passed']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations-dir", type=Path, default=ANNOTATIONS_DIR)
    ap.add_argument("--clips-dir", type=Path, default=CLIPS_DIR)
    ap.add_argument("--splits-path", type=Path, default=SPLITS_PATH)
    ap.add_argument("--source-videos", type=Path, default=SOURCE_VIDEOS_PATH)
    ap.add_argument("--schema-path", type=Path, default=SCHEMA_PATH)
    ap.add_argument("--out", type=Path, default=REPORTS_DIR / "validation_report.json")
    args = ap.parse_args()

    report = validate_all(args.annotations_dir, args.clips_dir, args.splits_path,
                           args.source_videos, args.schema_path)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    print_summary(report)
    print(f"\nfull report written to {args.out}")

    sys.exit(0 if report["summary"]["passed"] else 1)


if __name__ == "__main__":
    main()

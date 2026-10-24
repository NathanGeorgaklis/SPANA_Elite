#!/usr/bin/env python3
"""
collection_tracker.py -- tracks progress toward the 300-500 labelled-action
collection target (task items #39-40).

The audit is explicit that DIVERSITY across matches/sources matters at least
as much as raw volume -- 500 actions from one match is a worse dataset than
300 spread across a dozen matches, because a model trained on one match's
pitch, camera angle and teams will not generalise. This tool therefore
reports two independent things every run, never conflating them:

  1. Progress toward a per-(event_type, outcome) and per-NO_EVENT-hard-
     negative-category TARGET (dataset/metadata/collection_targets.json),
     e.g. "PASS_COMPLETED 18 / 75" -- so collection effort can be steered
     toward whatever is furthest behind, not just toward the easiest clips
     to find.
  2. The number of DISTINCT match_ids / source_video_ids actually
     represented in the annotated set so far, reported alongside (1) rather
     than folded into it, so "we hit 400 actions" and "we hit 400 actions
     from 2 matches" are never confused with each other.

Only EXPORTABLE annotation statuses count toward the target tally --
SINGLE_PASS, DOUBLE_PASS_AGREED, GOLD_RESOLVED. An unresolved
DOUBLE_PASS_DISAGREEMENT is real annotator effort but not yet a usable
labelled example (see DATASET_SCHEMA.md sec 2 / annotation.schema.json's
`status` enum) -- it doesn't count until a GOLD_RESOLVED record supersedes
it, same rule pipeline_benchmark.py and feature_exporter.py both apply.

dataset/metadata/collection_targets.json is deliberately plain, flat,
heavily-commented (via `_notes`, since JSON has no real comments) JSON --
it is meant to be hand-edited by whoever is steering collection, not
treated as a generated artifact to overwrite blindly. This tool only
creates it if it doesn't already exist (or on `--init`, which OVERWRITES it
back to the shipped defaults -- used deliberately to reset, not run
casually).

Usage:
    python tools/collection_tracker.py                 # progress report
    python tools/collection_tracker.py --init           # (re)write default targets, then report
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
DATASET_ROOT = TOOLS_DIR.parent
sys.path.insert(0, str(DATASET_ROOT))

from schema import taxonomy  # noqa: E402

ANNOTATIONS_DIR = DATASET_ROOT / "dataset" / "annotations"
SOURCE_VIDEOS_PATH = DATASET_ROOT / "dataset" / "metadata" / "source_videos.json"
TARGETS_PATH = DATASET_ROOT / "dataset" / "metadata" / "collection_targets.json"
REPORTS_DIR = DATASET_ROOT / "dataset" / "reports"

# Only these statuses represent a usable, exportable labelled example
# (mirrors EXPORTABLE_STATUSES in pipeline_benchmark.py / feature_exporter.py).
EXPORTABLE_STATUSES = {"SINGLE_PASS", "DOUBLE_PASS_AGREED", "GOLD_RESOLVED"}

TARGETS_SCHEMA_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Default per-(event_type, outcome) targets (task items #39-40).
#
# Weighted toward the more common/useful classes (PASS/CARRY/DUEL dominate
# real match footage and are the classes most training recipes need volume
# on), but taxonomy.py's rule holds here too: EVERY taxonomy-valid
# (event_type, outcome) combination gets a non-trivial (>0) target, including
# the rare/UNKNOWN-outcome buckets -- a target of 0 would silently tell
# collection "never bother labelling this," which is never true; even a rare
# class needs a handful of examples for validation/error-analysis. Same
# principle for the 9 real NO_EVENT hard-negative categories: each gets its
# own target so "hard negatives" doesn't collapse into one lazy bucket
# (task items #19-20).
#
# This adds up to ~430 total actions -- inside the 300-500 target range.
# Nothing here is sacred: edit the numbers directly, they are read fresh
# every run.
# ---------------------------------------------------------------------------
DEFAULT_EVENT_OUTCOME_TARGETS = {
    "PASS":  {"COMPLETED": 60, "INCOMPLETE": 30, "UNKNOWN": 5},
    "SHOT":  {"GOAL": 15, "SAVED": 20, "BLOCKED": 15, "OFF_TARGET": 15,
              "GOAL_LINE_CLEARANCE": 5, "UNKNOWN": 5},
    "CROSS": {"COMPLETED": 25, "INCOMPLETE": 25, "UNKNOWN": 5},
    "DUEL":  {"WON": 30, "LOST": 30, "UNKNOWN": 5},
    "CARRY": {"NONE": 40},
    "OTHER": {"UNKNOWN": 10},
    "UNKNOWN": {"UNKNOWN": 10},
}

DEFAULT_NO_EVENT_HARD_NEGATIVE_TARGETS = {
    "NEAR_BALL_NO_TOUCH": 9,
    "TEAMMATE_RECEIVES_NEARBY": 9,
    "SHOT_PASSES_NEARBY": 9,
    "CONTESTED_NO_DUEL": 9,
    "BALL_OCCLUDED": 9,
    "CAMERA_PANS_AWAY": 9,
    "REPLAY": 9,
    "STOPPAGE": 9,
    "OTHER": 9,
}


def _default_targets_doc() -> dict:
    total = (sum(v for outcomes in DEFAULT_EVENT_OUTCOME_TARGETS.values() for v in outcomes.values())
             + sum(DEFAULT_NO_EVENT_HARD_NEGATIVE_TARGETS.values()))
    return {
        "_notes": [
            "Hand-editable. Change any number and re-run collection_tracker.py -- targets "
            "are read fresh every run, never cached or regenerated unless you pass --init.",
            "Per task items #39-40: total is meant to land in the 300-500 labelled-action "
            "range, weighted toward common/useful classes, but with nothing at zero -- every "
            "taxonomy-valid (event_type, outcome) pair and every NO_EVENT hard-negative "
            "category should get at least a handful of examples.",
            "event_outcome_targets: event_type -> outcome -> target count. Must only use "
            "outcomes valid for that event_type per schema/taxonomy.py's "
            "OUTCOMES_BY_EVENT_TYPE -- collection_tracker.py will warn (not crash) if it "
            "finds a target key that taxonomy.py no longer recognises.",
            "no_event_hard_negative_targets: hard_negative_category -> target count. These "
            "are NO_EVENT clips specifically (task items #19-20) -- real hard negatives "
            "(near-miss, contested, occluded, etc.), not just trivial nothing-happening "
            "filler. A NO_EVENT record with no hard_negative_category set doesn't count "
            "toward any of these targets; see dataset_stats.json's no_event_hard_negative "
            "section for how many such 'uncategorised' negatives exist.",
            "Diversity matters as much as volume: hitting every number below from a single "
            "match is explicitly NOT the goal -- see this report's distinct match_id / "
            "source_video_id counts.",
        ],
        "schema_version": TARGETS_SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_target": total,
        "event_outcome_targets": DEFAULT_EVENT_OUTCOME_TARGETS,
        "no_event_hard_negative_targets": DEFAULT_NO_EVENT_HARD_NEGATIVE_TARGETS,
    }


def ensure_targets_file(targets_path: Path, force_init: bool) -> dict:
    """Creates targets_path with the default targets if it doesn't exist yet,
    or unconditionally overwrites it with defaults when force_init is True.
    Otherwise loads and returns whatever is already on disk -- this is the
    "trivially hand-editable" contract: a user's edits are never clobbered
    by an ordinary run."""
    if force_init or not targets_path.exists():
        doc = _default_targets_doc()
        targets_path.parent.mkdir(parents=True, exist_ok=True)
        targets_path.write_text(json.dumps(doc, indent=2))
        action = "(re)initialised" if force_init else "created (first run)"
        print(f"[collection_tracker] {action} default targets at {targets_path}")
        return doc
    try:
        return json.loads(targets_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        raise SystemExit(f"collection_targets.json exists but could not be read/parsed "
                          f"({e}) -- fix it by hand, or re-run with --init to reset it "
                          f"to defaults (this OVERWRITES any hand edits).")


def _validate_targets_against_taxonomy(targets: dict) -> list:
    """Returns a list of human-readable warnings (never raises) for target
    keys that no longer correspond to anything in schema/taxonomy.py -- e.g.
    a stale key left over after a taxonomy change. Warnings only: a
    hand-edited targets file is allowed to be ahead of or behind taxonomy.py
    temporarily, this tool just says so."""
    warnings = []
    eo_targets = targets.get("event_outcome_targets", {})
    for event_type, outcomes in eo_targets.items():
        if event_type not in taxonomy.EVENT_TYPES_ALL:
            warnings.append(f"event_outcome_targets has unknown event_type '{event_type}'")
            continue
        allowed = set(taxonomy.OUTCOMES_BY_EVENT_TYPE.get(event_type, []))
        for outcome in outcomes:
            if outcome not in allowed:
                warnings.append(f"event_outcome_targets['{event_type}']['{outcome}'] -- "
                                 f"'{outcome}' is not a valid outcome for {event_type} "
                                 f"(allowed: {sorted(allowed)})")
    hn_targets = targets.get("no_event_hard_negative_targets", {})
    valid_categories = set(c for c in taxonomy.HARD_NEGATIVE_CATEGORIES if c is not None)
    for category in hn_targets:
        if category not in valid_categories:
            warnings.append(f"no_event_hard_negative_targets has unknown category '{category}' "
                             f"(allowed: {sorted(valid_categories)})")
    return warnings


def load_annotations(annotations_dir: Path):
    """Same defensive contract as stats_report.py: missing directory or
    unparseable files degrade to an empty/partial list, never a crash."""
    records = []
    parse_errors = []
    if not annotations_dir.exists():
        return records, parse_errors
    for path in sorted(annotations_dir.glob("*.json")):
        try:
            records.append(json.loads(path.read_text()))
        except (json.JSONDecodeError, OSError) as e:
            parse_errors.append({"path": str(path), "error": str(e)})
    return records, parse_errors


def tally_progress(records: list, targets: dict) -> dict:
    eo_counts = {}   # event_type -> outcome -> count (exportable only)
    hn_counts = Counter()   # hard_negative_category -> count (exportable, NO_EVENT only)
    n_exportable = 0
    n_not_exportable = 0

    for ann in records:
        if ann.get("status") not in EXPORTABLE_STATUSES:
            n_not_exportable += 1
            continue
        n_exportable += 1
        hl = ann.get("human_label") or {}
        event_type = hl.get("event_type")
        outcome = hl.get("outcome")
        if event_type:
            eo_counts.setdefault(event_type, Counter())[outcome if outcome is not None else "NULL"] += 1
        if event_type == "NO_EVENT":
            hnc = hl.get("hard_negative_category")
            if hnc:
                hn_counts[hnc] += 1

    lines = []
    total_target = 0
    total_current = 0
    for event_type, outcomes in targets.get("event_outcome_targets", {}).items():
        for outcome, target in outcomes.items():
            current = eo_counts.get(event_type, Counter()).get(outcome, 0)
            lines.append({
                "label": f"{event_type}_{outcome}",
                "event_type": event_type, "outcome": outcome,
                "current": current, "target": target,
                "pct": round(100.0 * current / target, 1) if target else None,
            })
            total_target += target
            total_current += current

    for category, target in targets.get("no_event_hard_negative_targets", {}).items():
        current = hn_counts.get(category, 0)
        lines.append({
            "label": f"NO_EVENT_{category}",
            "event_type": "NO_EVENT", "hard_negative_category": category,
            "current": current, "target": target,
            "pct": round(100.0 * current / target, 1) if target else None,
        })
        total_target += target
        total_current += current

    return {
        "n_exportable_annotations": n_exportable,
        "n_non_exportable_annotations": n_not_exportable,
        "lines": lines,
        "total_current": total_current,
        "total_target": total_target,
        "overall_pct": round(100.0 * total_current / total_target, 1) if total_target else None,
    }


def diversity_report(records: list, source_videos_path: Path) -> dict:
    """Distinct match_id / source_video_id coverage, independent of the
    per-class target tally above -- task items #39-40's explicit
    'prioritize diversity over volume from one match'. Uses ALL annotation
    attempts found (any status) since collecting from a new match/source is
    a fact about the raw footage gathered, not about which attempts have
    since been resolved to an exportable status."""
    match_ids = set()
    source_video_ids = set()
    for ann in records:
        if ann.get("match_id"):
            match_ids.add(ann["match_id"])
        if ann.get("source_video_id"):
            source_video_ids.add(ann["source_video_id"])

    registry = {}
    registry_present = source_videos_path.exists()
    if registry_present:
        try:
            registry = json.loads(source_videos_path.read_text())
        except (json.JSONDecodeError, OSError):
            registry = {}

    registry_match_ids = set()
    for meta in registry.values():
        if isinstance(meta, dict) and meta.get("match_id"):
            registry_match_ids.add(meta["match_id"])

    return {
        "source_videos_registry_present": registry_present,
        "n_source_videos_registered": len(registry),
        "distinct_match_ids_in_annotations": len(match_ids),
        "distinct_source_video_ids_in_annotations": len(source_video_ids),
        "distinct_match_ids_in_registry": len(registry_match_ids),
        "match_ids": sorted(match_ids),
        "source_video_ids": sorted(source_video_ids),
    }


def print_progress(progress: dict, diversity: dict, warnings: list) -> None:
    print("=" * 78)
    print("SPANA COLLECTION PROGRESS")
    print("=" * 78)
    print(f"exportable annotations counted: {progress['n_exportable_annotations']} "
          f"(SINGLE_PASS / DOUBLE_PASS_AGREED / GOLD_RESOLVED)")
    print(f"non-exportable (e.g. unresolved DOUBLE_PASS_DISAGREEMENT): "
          f"{progress['n_non_exportable_annotations']}")
    if warnings:
        print("\nWARNINGS (collection_targets.json vs schema/taxonomy.py):")
        for w in warnings:
            print(f"  ! {w}")

    print("\n-- per-target progress --")
    for line in progress["lines"]:
        pct = f"{line['pct']:5.1f}%" if line["pct"] is not None else "  n/a"
        print(f"  {line['label']:32s} {line['current']:4d} / {line['target']:<4d} ({pct})")

    print(f"\nTOTAL: {progress['total_current']} / {progress['total_target']} "
          f"({progress['overall_pct']}%)" if progress['overall_pct'] is not None
          else f"\nTOTAL: {progress['total_current']} / {progress['total_target']}")

    print("\n-- diversity (task items #39-40: diversity over volume from one match) --")
    print(f"  distinct match_ids represented: {diversity['distinct_match_ids_in_annotations']}")
    print(f"  distinct source_video_ids represented: {diversity['distinct_source_video_ids_in_annotations']}")
    if diversity["source_videos_registry_present"]:
        print(f"  (source_videos.json registered: {diversity['n_source_videos_registered']} sources, "
              f"{diversity['distinct_match_ids_in_registry']} matches)")
    else:
        print("  (dataset/metadata/source_videos.json not found yet -- diversity counts inferred "
              "directly from annotations' match_id/source_video_id fields)")
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations-dir", type=Path, default=ANNOTATIONS_DIR)
    ap.add_argument("--source-videos", type=Path, default=SOURCE_VIDEOS_PATH)
    ap.add_argument("--targets", type=Path, default=TARGETS_PATH)
    ap.add_argument("--out", type=Path, default=REPORTS_DIR / "collection_progress.json")
    ap.add_argument("--init", action="store_true",
                     help="(Re)write dataset/metadata/collection_targets.json to the shipped "
                          "defaults. Safe on first run; OVERWRITES hand edits on a later run.")
    args = ap.parse_args()

    targets = ensure_targets_file(args.targets, force_init=args.init)
    warnings = _validate_targets_against_taxonomy(targets)

    records, parse_errors = load_annotations(args.annotations_dir)
    progress = tally_progress(records, targets)
    diversity = diversity_report(records, args.source_videos)

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "targets_path": str(args.targets),
        "targets_warnings": warnings,
        "n_json_parse_errors": len(parse_errors),
        "json_parse_errors": parse_errors,
        "progress": progress,
        "diversity": diversity,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    print_progress(progress, diversity, warnings)
    print(f"\nfull report written to {args.out}")


if __name__ == "__main__":
    main()

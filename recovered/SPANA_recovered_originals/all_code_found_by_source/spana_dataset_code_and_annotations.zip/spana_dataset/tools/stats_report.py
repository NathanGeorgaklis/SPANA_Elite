#!/usr/bin/env python3
"""
stats_report.py -- dataset composition/bias report (task item #33).

Reads every dataset/annotations/*.json (one file per ANNOTATION ATTEMPT --
see DATASET_SCHEMA.md sec 2) and produces a single JSON report plus a
human-readable summary, covering the questions the audit explicitly wants
answered before this dataset is trusted for training:

  - Is the class balance across (event_type, outcome) reasonable, or is one
    bucket (e.g. PASS_COMPLETED) drowning out everything else?
  - Is double-annotation coverage (task item #22, target 10-20% of clips)
    actually happening, or is the DOUBLE_PASS_AGREED/DISAGREEMENT machinery
    in annotation.schema.json going unused?
  - Is the source footage diverse (competition, camera type, pitch surface,
    lighting) or is this secretly a single-match, single-camera dataset?
  - Are clip/event durations sane, or is there a long tail of degenerate
    near-zero-length or absurdly long windows that would break fixed-width
    temporal binning downstream (feature_exporter.py's BIN_WIDTH_S)?
  - Are the NO_EVENT ("hard negative") clips actually hard negatives (task
    items #19-20 -- near-miss, contested, occluded, etc. via
    hard_negative_category) or just trivial "nothing happening" filler that
    would teach a model nothing useful about the boundary cases that matter?

THIS TOOL NEVER MODIFIES ANNOTATIONS, TRAINS ANYTHING, OR TOUCHES THE FROZEN
PRODUCTION PIPELINE. It only reads dataset/annotations/*.json and (optionally)
dataset/metadata/source_videos.json, and writes dataset/reports/dataset_stats.json.

Early-dataset behaviour is a first-class case, not an afterthought: zero or a
handful of annotation files, a missing or partially-populated
source_videos.json, or annotations that reference source_video_ids the
registry doesn't know about yet are all NORMAL states for a dataset still
being built by tools running in parallel (clip_extractor.py,
annotation_server.py, migrate_existing_annotations.py). Every section below
degrades to an explicit "no data yet" / zero-count shape instead of raising,
and any source_video_id that can't be resolved against the registry is
bucketed as "UNREGISTERED_SOURCE" rather than crashing the whole report.

Usage:
    python tools/stats_report.py [--annotations-dir dataset/annotations] \
        [--source-videos dataset/metadata/source_videos.json] \
        [--out dataset/reports/dataset_stats.json]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
DATASET_ROOT = TOOLS_DIR.parent
sys.path.insert(0, str(DATASET_ROOT))

from schema import taxonomy  # noqa: E402

ANNOTATIONS_DIR = DATASET_ROOT / "dataset" / "annotations"
SOURCE_VIDEOS_PATH = DATASET_ROOT / "dataset" / "metadata" / "source_videos.json"
REPORTS_DIR = DATASET_ROOT / "dataset" / "reports"

STATUSES = ["SINGLE_PASS", "DOUBLE_PASS_AGREED", "DOUBLE_PASS_DISAGREEMENT", "GOLD_RESOLVED"]

# Task item #22: double-annotation is meant to cover 10-20% of clips, not
# all of them (too expensive) and not none of them (no inter-annotator
# agreement signal at all). This is a target RANGE, not a hard gate -- the
# report prints where the dataset currently sits against it.
DOUBLE_ANNOTATION_TARGET_LOW = 0.10
DOUBLE_ANNOTATION_TARGET_HIGH = 0.20

UNREGISTERED = "UNREGISTERED_SOURCE"

DURATION_BUCKETS = [
    ("0-1s", 0.0, 1.0),
    ("1-2s", 1.0, 2.0),
    ("2-5s", 2.0, 5.0),
    ("5s+", 5.0, float("inf")),
]


def _bucket_label(value: float) -> str:
    for label, lo, hi in DURATION_BUCKETS:
        if lo <= value < hi:
            return label
    return DURATION_BUCKETS[-1][0]  # negative/degenerate durations fall into the top bucket, not lost


def load_annotations(annotations_dir: Path):
    """Reads every *.json in annotations_dir. Returns (records, parse_errors).
    A record that fails to parse as JSON is skipped and reported, never
    allowed to crash the whole run -- this directory is written by tools
    (annotation_server.py, migrate_existing_annotations.py) that may be
    mid-development or interrupted mid-write."""
    records = []
    parse_errors = []
    if not annotations_dir.exists():
        return records, parse_errors
    for path in sorted(annotations_dir.glob("*.json")):
        try:
            ann = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            parse_errors.append({"path": str(path), "error": str(e)})
            continue
        records.append((path, ann))
    return records, parse_errors


def load_source_videos(source_videos_path: Path) -> dict:
    """Returns the source_video_id -> metadata registry, or {} if the file
    doesn't exist yet (a normal early-dataset state -- other tools that
    populate this registry may not have run yet)."""
    if not source_videos_path.exists():
        return {}
    try:
        return json.loads(source_videos_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _safe_stats(values):
    """min/max/mean/median over a list of numbers, or all-None if empty --
    never raises statistics.StatisticsError on an empty/early dataset."""
    if not values:
        return {"n": 0, "min": None, "max": None, "mean": None, "median": None, "histogram": {}}
    hist = Counter(_bucket_label(v) for v in values)
    return {
        "n": len(values),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
        "mean": round(statistics.mean(values), 3),
        "median": round(statistics.median(values), 3),
        "histogram": {label: hist.get(label, 0) for label, _, _ in DURATION_BUCKETS},
    }


def build_report(annotations_dir: Path, source_videos_path: Path) -> dict:
    records, parse_errors = load_annotations(annotations_dir)
    registry = load_source_videos(source_videos_path)
    registry_present = source_videos_path.exists()

    # --- 1. counts by (event_type, outcome), and by event_subtype when set ---
    counts_by_event_outcome = defaultdict(Counter)   # event_type -> outcome -> count
    counts_by_event_subtype = defaultdict(Counter)   # event_type -> subtype -> count (only when set)

    # --- 2. status counts + double-annotation coverage ---
    status_counts = Counter()
    clip_attempt_counts = Counter()   # clip_id -> number of annotation attempts on file

    # --- 3. competition/source + camera/pitch/lighting, joined via registry ---
    counts_by_competition = Counter()
    counts_by_camera_type = Counter()
    counts_by_pitch_surface = Counter()
    counts_by_lighting = Counter()
    counts_by_source_video_id = Counter()
    counts_by_match_id = Counter()

    # --- 4. duration distributions ---
    gt_durations = []
    context_durations = []

    # --- 5. confidence distributions ---
    annotation_confidence_counts = Counter()
    manual_identity_confidence_counts = Counter()

    # --- 6. hard_negative_category, only among NO_EVENT ---
    no_event_total = 0
    no_event_with_category = 0
    hard_negative_category_counts = Counter()

    # --- 7. ambiguous / review_required ---
    ambiguous_counts = Counter()
    review_required_counts = Counter()

    record_errors = []

    for path, ann in records:
        try:
            status = ann.get("status")
            status_counts[status if status in STATUSES else "UNKNOWN_STATUS"] += 1

            clip_id = ann.get("clip_id")
            if clip_id:
                clip_attempt_counts[clip_id] += 1

            match_id = ann.get("match_id")
            if match_id:
                counts_by_match_id[match_id] += 1

            source_video_id = ann.get("source_video_id")
            if source_video_id:
                counts_by_source_video_id[source_video_id] += 1

            meta = registry.get(source_video_id) if (registry_present and source_video_id) else None
            competition = meta.get("competition") if meta else None
            camera_type = meta.get("camera_type") if meta else None
            pitch_surface = meta.get("pitch_surface") if meta else None
            lighting = meta.get("lighting") if meta else None
            counts_by_competition[competition or UNREGISTERED] += 1
            counts_by_camera_type[camera_type or UNREGISTERED] += 1
            counts_by_pitch_surface[pitch_surface or UNREGISTERED] += 1
            counts_by_lighting[lighting or UNREGISTERED] += 1

            hl = ann.get("human_label") or {}

            event_type = hl.get("event_type")
            outcome = hl.get("outcome")
            event_subtype = hl.get("event_subtype")
            if event_type:
                counts_by_event_outcome[event_type][outcome if outcome is not None else "NULL"] += 1
                if event_subtype is not None:
                    counts_by_event_subtype[event_type][event_subtype] += 1

            gt_win = hl.get("ground_truth_event_window") or {}
            if gt_win.get("start_time_s") is not None and gt_win.get("end_time_s") is not None:
                gt_durations.append(gt_win["end_time_s"] - gt_win["start_time_s"])
            ctx_win = hl.get("analysis_context_window") or {}
            if ctx_win.get("start_time_s") is not None and ctx_win.get("end_time_s") is not None:
                context_durations.append(ctx_win["end_time_s"] - ctx_win["start_time_s"])

            annotation_confidence_counts[hl.get("annotation_confidence") or "NULL"] += 1
            selected_player = hl.get("selected_player") or {}
            manual_identity_confidence_counts[selected_player.get("manual_identity_confidence") or "NULL"] += 1

            if event_type == "NO_EVENT":
                no_event_total += 1
                hnc = hl.get("hard_negative_category")
                if hnc:
                    no_event_with_category += 1
                    hard_negative_category_counts[hnc] += 1
                else:
                    hard_negative_category_counts["NONE_SET"] += 1

            ambiguous_counts[bool(hl.get("ambiguous", False))] += 1
            review_required_counts[bool(hl.get("review_required", False))] += 1

        except (AttributeError, TypeError) as e:
            # A malformed-but-parseable record (wrong shape) is reported,
            # not allowed to take the whole report down with it.
            record_errors.append({"path": str(path), "error": f"unexpected record shape: {e}"})

    # Double-annotation coverage rate (task item #22): fraction of DISTINCT
    # clip_ids that have >= 2 annotation attempts on file.
    total_distinct_clips = len(clip_attempt_counts)
    double_annotated_clips = sum(1 for n in clip_attempt_counts.values() if n >= 2)
    double_annotation_rate = (
        double_annotated_clips / total_distinct_clips if total_distinct_clips else None
    )
    if double_annotation_rate is None:
        coverage_status = "NO_DATA"
    elif double_annotation_rate < DOUBLE_ANNOTATION_TARGET_LOW:
        coverage_status = "BELOW_TARGET"
    elif double_annotation_rate > DOUBLE_ANNOTATION_TARGET_HIGH:
        coverage_status = "ABOVE_TARGET"
    else:
        coverage_status = "ON_TARGET"

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "taxonomy_version": taxonomy.TAXONOMY_VERSION,
        "n_annotation_files_found": len(records) + len(parse_errors),
        "n_annotation_files_read_ok": len(records),
        "n_json_parse_errors": len(parse_errors),
        "json_parse_errors": parse_errors,
        "n_record_shape_errors": len(record_errors),
        "record_shape_errors": record_errors,
        "source_videos_registry_present": registry_present,
        "n_source_videos_registered": len(registry),

        "counts_by_event_outcome": {et: dict(c) for et, c in counts_by_event_outcome.items()},
        "counts_by_event_subtype": {et: dict(c) for et, c in counts_by_event_subtype.items()},

        "counts_by_status": {s: status_counts.get(s, 0) for s in STATUSES + ["UNKNOWN_STATUS"]},
        "double_annotation_coverage": {
            "total_distinct_clip_ids": total_distinct_clips,
            "clip_ids_with_2plus_attempts": double_annotated_clips,
            "coverage_rate": round(double_annotation_rate, 4) if double_annotation_rate is not None else None,
            "target_range": [DOUBLE_ANNOTATION_TARGET_LOW, DOUBLE_ANNOTATION_TARGET_HIGH],
            "status": coverage_status,
        },

        "counts_by_competition": dict(counts_by_competition),
        "counts_by_camera_type": dict(counts_by_camera_type),
        "counts_by_pitch_surface": dict(counts_by_pitch_surface),
        "counts_by_lighting": dict(counts_by_lighting),
        "counts_by_match_id": dict(counts_by_match_id),
        "counts_by_source_video_id": dict(counts_by_source_video_id),

        "ground_truth_event_duration_s": _safe_stats(gt_durations),
        "analysis_context_duration_s": _safe_stats(context_durations),

        "annotation_confidence_distribution": dict(annotation_confidence_counts),
        "manual_identity_confidence_distribution": dict(manual_identity_confidence_counts),

        "no_event_hard_negative": {
            "n_no_event_records": no_event_total,
            "n_with_hard_negative_category_set": no_event_with_category,
            "fraction_with_category_set": (
                round(no_event_with_category / no_event_total, 4) if no_event_total else None
            ),
            "category_counts": dict(hard_negative_category_counts),
        },

        "ambiguous_counts": {str(k): v for k, v in ambiguous_counts.items()},
        "review_required_counts": {str(k): v for k, v in review_required_counts.items()},
    }
    return report


def print_summary(report: dict) -> None:
    print("=" * 78)
    print("SPANA DATASET STATS REPORT")
    print("=" * 78)
    print(f"generated_at: {report['generated_at']}")
    print(f"annotation files: {report['n_annotation_files_read_ok']} read OK, "
          f"{report['n_json_parse_errors']} parse errors, "
          f"{report['n_record_shape_errors']} shape errors")
    if report["n_annotation_files_read_ok"] == 0:
        print("\nNO ANNOTATION DATA YET -- all sections below are empty/zero by design.")

    print("\n-- counts by (event_type, outcome) --")
    if not report["counts_by_event_outcome"]:
        print("  (none)")
    for et in sorted(report["counts_by_event_outcome"]):
        for outcome, n in sorted(report["counts_by_event_outcome"][et].items()):
            print(f"  {et:10s} {outcome:20s} {n}")

    if report["counts_by_event_subtype"]:
        print("\n-- counts by event_subtype (only where set) --")
        for et in sorted(report["counts_by_event_subtype"]):
            for sub, n in sorted(report["counts_by_event_subtype"][et].items()):
                print(f"  {et:10s} {sub:15s} {n}")

    print("\n-- counts by status --")
    for s, n in report["counts_by_status"].items():
        print(f"  {s:25s} {n}")

    dac = report["double_annotation_coverage"]
    print("\n-- double-annotation coverage (task item #22 target: "
          f"{dac['target_range'][0]*100:.0f}-{dac['target_range'][1]*100:.0f}%) --")
    print(f"  distinct clips: {dac['total_distinct_clip_ids']}, "
          f"clips with >=2 attempts: {dac['clip_ids_with_2plus_attempts']}")
    print(f"  coverage_rate: {dac['coverage_rate']}  status: {dac['status']}")

    print("\n-- counts by competition (UNREGISTERED_SOURCE = not in source_videos.json) --")
    for k, n in sorted(report["counts_by_competition"].items(), key=lambda kv: -kv[1]):
        print(f"  {k:25s} {n}")
    print("-- counts by camera_type --")
    for k, n in sorted(report["counts_by_camera_type"].items(), key=lambda kv: -kv[1]):
        print(f"  {k:25s} {n}")
    print("-- counts by pitch_surface --")
    for k, n in sorted(report["counts_by_pitch_surface"].items(), key=lambda kv: -kv[1]):
        print(f"  {k:25s} {n}")
    print("-- counts by lighting --")
    for k, n in sorted(report["counts_by_lighting"].items(), key=lambda kv: -kv[1]):
        print(f"  {k:25s} {n}")
    print(f"-- distinct match_ids: {len(report['counts_by_match_id'])}, "
          f"distinct source_video_ids: {len(report['counts_by_source_video_id'])} --")

    for label, key in [("ground-truth event duration (s)", "ground_truth_event_duration_s"),
                        ("analysis context duration (s)", "analysis_context_duration_s")]:
        d = report[key]
        print(f"\n-- {label} --")
        print(f"  n={d['n']} min={d['min']} max={d['max']} mean={d['mean']} median={d['median']}")
        print(f"  histogram: {d['histogram']}")

    print("\n-- annotation_confidence distribution --")
    for k, n in report["annotation_confidence_distribution"].items():
        print(f"  {k:10s} {n}")
    print("-- manual_identity_confidence distribution --")
    for k, n in report["manual_identity_confidence_distribution"].items():
        print(f"  {k:10s} {n}")

    neg = report["no_event_hard_negative"]
    print("\n-- NO_EVENT hard_negative_category (task items #19-20) --")
    print(f"  n_no_event_records={neg['n_no_event_records']} "
          f"with_category_set={neg['n_with_hard_negative_category_set']} "
          f"fraction={neg['fraction_with_category_set']}")
    for k, n in sorted(neg["category_counts"].items(), key=lambda kv: -kv[1]):
        print(f"  {k:25s} {n}")

    print("\n-- ambiguous / review_required --")
    print(f"  ambiguous: {report['ambiguous_counts']}")
    print(f"  review_required: {report['review_required_counts']}")
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations-dir", type=Path, default=ANNOTATIONS_DIR)
    ap.add_argument("--source-videos", type=Path, default=SOURCE_VIDEOS_PATH)
    ap.add_argument("--out", type=Path, default=REPORTS_DIR / "dataset_stats.json")
    args = ap.parse_args()

    report = build_report(args.annotations_dir, args.source_videos)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    print_summary(report)
    print(f"\nfull report written to {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
pipeline_benchmark.py -- benchmarks the CURRENT, FROZEN production event
system against the canonical human-annotated dataset (audit item #41).

THIS TOOL NEVER MODIFIES THE EVENT SYSTEM. It runs the exact same read-only
capture path feature_exporter.py uses (clip_pipeline_runner.py ->
feature_extraction_harness.py -> football_analysis.engine.FootballAnalysisEngine,
completely unmodified) and only ever READS its output.

Two separate report sections, deliberately kept apart (audit item #42:
"do not blame event classification for bad perception"):

  1. EVENT BENCHMARK -- precision/recall/F1 per event type, a confusion
     matrix, and an "unknown/review rate" (how often the pipeline itself
     declines to commit to a type), all computed by comparing each
     annotation's human_label.event_type against the pipeline's own matched
     event (via event_matching / clip_pipeline_runner.find_matched_event).

  2. PERCEPTION BENCHMARK -- player-tracking coverage (fraction of frames in
     TRACKING vs OCCLUDED/REACQUIRING/LOST) and ball-availability coverage
     (fraction of frames CONFIRMED/REACQUIRING vs PREDICTED/OCCLUDED/LOST),
     aggregated across all benchmarked clips. Team classification is
     reported as NOT MEASURED here and why: engine.py's own per-clip
     analysis pass calls merge_other_player_candidates(..., team_classifier=
     None) -- exactly as this benchmark's harness does -- so the standalone
     TeamClassifier module is never exercised by this code path at all.
     Benchmarking it needs a separate integration test against that module
     directly, which is out of this task's scope (no production changes).

Usage:
    python pipeline_benchmark.py --annotations-dir dataset/annotations \
        --split TEST [--verify] --out dataset/reports/pipeline_benchmark.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
DATASET_ROOT = TOOLS_DIR.parent
sys.path.insert(0, str(TOOLS_DIR))

import clip_pipeline_runner as runner        # noqa: E402
import freeze_utils                           # noqa: E402
from schema import taxonomy                   # noqa: E402

ANNOTATIONS_DIR = DATASET_ROOT / "dataset" / "annotations"
SPLITS_PATH = DATASET_ROOT / "dataset" / "splits" / "assignment.json"
REPORTS_DIR = DATASET_ROOT / "dataset" / "reports"

EXPORTABLE_STATUSES = {"SINGLE_PASS", "DOUBLE_PASS_AGREED", "GOLD_RESOLVED"}

# The pipeline's own event "type" strings (ClipTagger output) map onto this
# dataset's taxonomy as follows -- UNKNOWN_RELEASE is the pipeline's own
# abstention type and corresponds to this dataset's UNKNOWN/OTHER/NO_EVENT
# bucket only in the loose sense of "the pipeline declined to commit"; it is
# reported separately as the review rate rather than folded into the
# confusion matrix as a false answer.
PIPELINE_TO_TAXONOMY = {
    "PASS": "PASS", "SHOT": "SHOT", "CROSS": "CROSS", "DUEL": "DUEL",
    "TOUCH": "OTHER", "RECEIVE": "OTHER", "CARRY": "CARRY", "GOAL": "SHOT",
}
BENCHMARK_EVENT_TYPES = ["PASS", "SHOT", "CROSS", "DUEL", "CARRY"]


def _load_split_assignment():
    if not SPLITS_PATH.exists():
        return None
    return json.loads(SPLITS_PATH.read_text())


def _select_annotations(annotations_dir: Path, split: str | None):
    assignment = _load_split_assignment() if split else None
    out = []
    for p in sorted(annotations_dir.glob("*.json")):
        ann = json.loads(p.read_text())
        if ann.get("status") not in EXPORTABLE_STATUSES:
            continue
        if split:
            if assignment is None:
                raise SystemExit(f"--split {split} requested but {SPLITS_PATH} does not exist yet "
                                  f"-- run split_generator.py first.")
            match_split = assignment.get(ann["match_id"])
            if match_split != split:
                continue
        out.append((p, ann))
    return out


def _perception_stats(capture) -> dict:
    tracking_counts = Counter(s.tracking_state for s in capture.snapshots)
    ball_counts = Counter(b.state for b in capture.ball_history)
    total_frames = len(capture.snapshots)
    total_ball = len(capture.ball_history)
    return {
        "frames": total_frames,
        "tracking_state_counts": dict(tracking_counts),
        "tracking_coverage_pct": round(100.0 * tracking_counts.get("TRACKING", 0) / total_frames, 2)
        if total_frames else None,
        "ball_state_counts": dict(ball_counts),
        "ball_confirmed_or_reacquiring_pct": round(
            100.0 * (ball_counts.get("CONFIRMED", 0) + ball_counts.get("REACQUIRING", 0)) / total_ball, 2)
        if total_ball else None,
    }


def run_benchmark(annotations_dir: Path, split: str | None, verify: bool, limit: int | None = None) -> dict:
    selected = _select_annotations(annotations_dir, split)
    if limit:
        selected = selected[:limit]

    confusion = defaultdict(Counter)   # confusion[human_type][pipeline_type_or_"NO_MATCH"/"REVIEW_REQUIRED"]
    n_total = 0
    n_review_required = 0
    n_no_match = 0
    failures = []
    perception_agg = {
        "clips": 0, "total_frames": 0, "tracking_state_counts": Counter(),
        "total_ball_frames": 0, "ball_state_counts": Counter(),
    }

    t0 = time.time()
    for path, ann in selected:
        human_type = ann["human_label"]["event_type"]
        n_total += 1
        try:
            loaded = runner.load_annotation_and_clip(path)
            capture = runner.run_pipeline(loaded, verify=verify)
        except runner.ClipPipelineError as e:
            failures.append({"annotation_id": ann["annotation_id"], "error": str(e)})
            confusion[human_type]["PIPELINE_ERROR"] += 1
            continue

        pstats = _perception_stats(capture)
        perception_agg["clips"] += 1
        perception_agg["total_frames"] += pstats["frames"]
        perception_agg["tracking_state_counts"].update(pstats["tracking_state_counts"])
        perception_agg["total_ball_frames"] += len(capture.ball_history)
        perception_agg["ball_state_counts"].update(pstats["ball_state_counts"])

        matched = runner.find_matched_event(loaded, capture)
        if matched is None:
            confusion[human_type]["NO_MATCH"] += 1
            n_no_match += 1
            continue
        if matched["type"] == "UNKNOWN_RELEASE" or matched.get("qualifiers") == ["REVIEW_REQUIRED"]:
            confusion[human_type]["REVIEW_REQUIRED"] += 1
            n_review_required += 1
            continue
        predicted = PIPELINE_TO_TAXONOMY.get(matched["type"], "OTHER")
        confusion[human_type][predicted] += 1

    # Precision/recall/F1 per taxonomy event type, computed from the confusion counts above.
    per_class = {}
    for etype in BENCHMARK_EVENT_TYPES:
        tp = confusion[etype][etype]
        fn = sum(v for k, v in confusion[etype].items() if k != etype)
        fp = sum(confusion[other][etype] for other in confusion if other != etype)
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        f1 = (2 * precision * recall / (precision + recall)) if (precision and recall) else None
        per_class[etype] = {
            "true_positives": tp, "false_positives": fp, "false_negatives": fn,
            "precision": round(precision, 4) if precision is not None else None,
            "recall": round(recall, 4) if recall is not None else None,
            "f1": round(f1, 4) if f1 is not None else None,
            "support": sum(confusion[etype].values()),
        }

    total_frames = perception_agg["total_frames"]
    total_ball = perception_agg["total_ball_frames"]
    tsc = perception_agg["tracking_state_counts"]
    bsc = perception_agg["ball_state_counts"]

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "production_files_sha256": freeze_utils.combined_production_sha256(),
        "split": split or "ALL_EXPORTABLE",
        "n_annotations_evaluated": n_total,
        "n_pipeline_errors": len(failures),
        "pipeline_errors": failures,
        "event_benchmark": {
            "confusion_matrix": {k: dict(v) for k, v in confusion.items()},
            "per_class": per_class,
            "unknown_review_rate": round(n_review_required / n_total, 4) if n_total else None,
            "no_match_rate": round(n_no_match / n_total, 4) if n_total else None,
        },
        "perception_benchmark": {
            "note": "Separate from event benchmark by design (audit item #42) -- do not "
                    "attribute a low event score to perception, or vice versa, without "
                    "checking this section.",
            "clips_evaluated": perception_agg["clips"],
            "player_tracking": {
                "total_frames": total_frames,
                "state_counts": dict(tsc),
                "tracking_coverage_pct": round(100.0 * tsc.get("TRACKING", 0) / total_frames, 2)
                if total_frames else None,
            },
            "ball_availability": {
                "total_frames": total_ball,
                "state_counts": dict(bsc),
                "confirmed_or_reacquiring_pct": round(
                    100.0 * (bsc.get("CONFIRMED", 0) + bsc.get("REACQUIRING", 0)) / total_ball, 2)
                if total_ball else None,
            },
            "team_classification": {
                "status": "NOT_MEASURED",
                "reason": "engine.py's own per-clip analysis pass calls merge_other_player_"
                          "candidates(..., team_classifier=None), exactly like this benchmark's "
                          "harness -- the standalone TeamClassifier module is not exercised by "
                          "this code path at all, so there is nothing honest to report here. "
                          "Benchmarking it requires a separate integration test against that "
                          "module directly.",
            },
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations-dir", type=Path, default=ANNOTATIONS_DIR)
    ap.add_argument("--split", choices=["TRAIN", "VALIDATION", "TEST", "GOLD_TEST"], default=None,
                     help="Benchmark only this split. Recommended: TEST or GOLD_TEST, never "
                          "TRAIN (audit item #25: GOLD_TEST is for final evaluation only -- "
                          "running this repeatedly against GOLD_TEST for threshold tuning "
                          "would itself violate its purpose, even though this tool cannot "
                          "modify the pipeline to act on the result).")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=REPORTS_DIR / "pipeline_benchmark.json")
    args = ap.parse_args()

    report = run_benchmark(args.annotations_dir, args.split, args.verify, args.limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["event_benchmark"]["per_class"], indent=2))
    print(f"\nunknown_review_rate={report['event_benchmark']['unknown_review_rate']} "
          f"no_match_rate={report['event_benchmark']['no_match_rate']}")
    print(f"tracking_coverage_pct={report['perception_benchmark']['player_tracking']['tracking_coverage_pct']} "
          f"ball_confirmed_pct={report['perception_benchmark']['ball_availability']['confirmed_or_reacquiring_pct']}")
    print(f"\nfull report written to {args.out}")


if __name__ == "__main__":
    main()

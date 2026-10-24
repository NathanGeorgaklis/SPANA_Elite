#!/usr/bin/env python3
"""
inter_annotator_agreement.py -- infrastructure-acceptance task item #28:
tooling/report fields for inter-annotator agreement, ready to run the moment
enough double-labelled clips exist. Trains nothing; computes no model
metric; this is a pairwise human/human agreement report only.

Per the task brief: "Do not calculate meaningless statistics until enough
double-labelled clips exist." This tool enforces that itself -- below
MIN_PAIRS_FOR_STATISTICS, it reports INSUFFICIENT_DATA and the exact pair
count found, and refuses to emit a percentage that would misleadingly imply
statistical weight it doesn't have.

Pairs are found via provenance.double_annotation (annotation_server.py's own
pairing bookkeeping -- see amend_annotation_file()/_apply_double_annotation_
logic()) rather than by re-deriving "same clip, two annotators" from
scratch, so this tool and the annotation tool always agree on what counts as
a pair.

Reported per pair:
  - event_type agreement (exact match)
  - outcome agreement (exact match)
  - timing difference: |interaction_time_s difference| and
    |release_time_s difference| in ABSOLUTE SOURCE-VIDEO SECONDS (never
    clip-relative -- see annotation.schema.json's frame_safety discipline),
    None when either side left that timestamp null
  - identity agreement: same team AND (same jersey_number, or both null)

Usage:
    python inter_annotator_agreement.py [--annotations-dir dataset/annotations]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
DATASET_ROOT = TOOLS_DIR.parent
ANNOTATIONS_DIR = DATASET_ROOT / "dataset" / "annotations"
REPORTS_DIR = DATASET_ROOT / "dataset" / "reports"

MIN_PAIRS_FOR_STATISTICS = 5


def find_double_annotation_pairs(annotations: list[dict]) -> list[tuple[dict, dict]]:
    by_id = {a["annotation_id"]: a for a in annotations if "annotation_id" in a}
    seen = set()
    pairs = []
    for a in annotations:
        da = a.get("provenance", {}).get("double_annotation")
        if not da or not da.get("paired_annotation_id"):
            continue
        other_id = da["paired_annotation_id"]
        other = by_id.get(other_id)
        if other is None:
            continue
        key = tuple(sorted([a["annotation_id"], other_id]))
        if key in seen:
            continue
        seen.add(key)
        pairs.append((a, other))
    return pairs


def compare_pair(a: dict, b: dict) -> dict:
    ha, hb = a["human_label"], b["human_label"]
    gwa, gwb = ha["ground_truth_event_window"], hb["ground_truth_event_window"]
    spa, spb = ha["selected_player"], hb["selected_player"]

    def _timing_diff(field):
        va, vb = gwa.get(field), gwb.get(field)
        return None if va is None or vb is None else round(abs(va - vb), 4)

    identity_agree = (spa.get("team") == spb.get("team")
                       and spa.get("jersey_number") == spb.get("jersey_number"))

    return {
        "annotation_id_a": a["annotation_id"], "annotation_id_b": b["annotation_id"],
        "clip_id": a.get("clip_id"),
        "event_type_agree": ha["event_type"] == hb["event_type"],
        "outcome_agree": ha.get("outcome") == hb.get("outcome"),
        "interaction_time_diff_s": _timing_diff("interaction_time_s"),
        "release_time_diff_s": _timing_diff("release_time_s"),
        "start_time_diff_s": _timing_diff("start_time_s"),
        "end_time_diff_s": _timing_diff("end_time_s"),
        "identity_agree": identity_agree,
        "final_status": a.get("status"),
    }


def build_report(annotations_dir: Path = ANNOTATIONS_DIR) -> dict:
    annotations = []
    for p in sorted(annotations_dir.glob("*.json")):
        try:
            annotations.append(json.loads(p.read_text()))
        except (json.JSONDecodeError, OSError):
            continue

    pairs = find_double_annotation_pairs(annotations)
    comparisons = [compare_pair(a, b) for a, b in pairs]

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_double_annotated_pairs_found": len(comparisons),
        "min_pairs_for_statistics": MIN_PAIRS_FOR_STATISTICS,
        "pairs": comparisons,
    }

    if len(comparisons) < MIN_PAIRS_FOR_STATISTICS:
        report["status"] = "INSUFFICIENT_DATA"
        report["note"] = (
            f"Only {len(comparisons)} double-annotated pair(s) exist (need >= "
            f"{MIN_PAIRS_FOR_STATISTICS} before any agreement PERCENTAGE is reported -- "
            f"per task item #28, 'do not calculate meaningless statistics until enough "
            f"double-labelled clips exist'). The current dataset has 0 double-annotated "
            f"clips at all (all 8 canonical annotations are SINGLE_PASS) -- this is "
            f"expected at this stage, not a defect. Individual pair-level comparisons "
            f"(if any) are still listed above/below for inspection; no aggregate rate is "
            f"computed."
        )
        return report

    n = len(comparisons)
    report["status"] = "COMPUTED"
    report["event_type_agreement_rate"] = round(sum(c["event_type_agree"] for c in comparisons) / n, 4)
    report["outcome_agreement_rate"] = round(sum(c["outcome_agree"] for c in comparisons) / n, 4)
    report["identity_agreement_rate"] = round(sum(c["identity_agree"] for c in comparisons) / n, 4)
    interaction_diffs = [c["interaction_time_diff_s"] for c in comparisons if c["interaction_time_diff_s"] is not None]
    release_diffs = [c["release_time_diff_s"] for c in comparisons if c["release_time_diff_s"] is not None]
    report["mean_interaction_time_diff_s"] = round(sum(interaction_diffs) / len(interaction_diffs), 4) if interaction_diffs else None
    report["mean_release_time_diff_s"] = round(sum(release_diffs) / len(release_diffs), 4) if release_diffs else None
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations-dir", type=Path, default=ANNOTATIONS_DIR)
    ap.add_argument("--out", type=Path, default=REPORTS_DIR / "inter_annotator_agreement.json")
    args = ap.parse_args()

    report = build_report(args.annotations_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    print(f"inter_annotator_agreement: {report['n_double_annotated_pairs_found']} pair(s) found, "
          f"status={report['status']}")
    if report["status"] == "COMPUTED":
        print(f"  event_type_agreement_rate: {report['event_type_agreement_rate']}")
        print(f"  outcome_agreement_rate:    {report['outcome_agreement_rate']}")
        print(f"  identity_agreement_rate:   {report['identity_agreement_rate']}")
    else:
        print(f"  {report['note']}")
    print(f"full report written to {args.out}")


if __name__ == "__main__":
    main()

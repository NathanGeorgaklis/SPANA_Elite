#!/usr/bin/env python3
"""
training_eligibility.py -- infrastructure-acceptance task item #12: an
EXPLICIT, computed statement of which existing canonical annotations may
enter a FUTURE training run, and which may not, and why. This tool trains
nothing -- it only classifies. Re-run any time the dataset changes; nothing
here is persisted onto the annotation records themselves (eligibility is a
function of current dataset state -- a record's own schema-valid content
doesn't change when, say, a clip gets extracted later and its frame mapping
becomes verified -- so this stays a REPORT, not a stored field, to avoid a
second copy of information that would need to be kept in sync by hand).

Rule set (verbatim from the task brief):

  TRAINING_ELIGIBLE requires ALL of:
    - schema valid (schema/annotation.schema.json + taxonomy.validate_event_record)
    - annotation_confidence in taxonomy.TRAINING_ELIGIBLE_CONFIDENCE (HIGH/MEDIUM)
    - selected-player identity acceptable: manual_identity_confidence != LOW
      (see dataset_validator.py's corrected identity semantics -- this project
      has no persistent player_id database, so identity acceptability is about
      visual-identification confidence, not the presence of a database ID)
    - not unresolved ambiguous: status != DOUBLE_PASS_DISAGREEMENT, AND
      human_label.ambiguous is not true, AND human_label.review_required is
      not true
    - source/frame mapping verified: the clip has been extracted
      (dataset/clips/<clip_id>/clip_time_map.json exists) AND
      dataset_validator.py's frame/fps consistency check for it raised no
      HARD error

  A record failing any one of the above is NOT_TRAINING_ELIGIBLE, and every
  failing reason is listed (not just the first) so a record blocked for two
  independent reasons shows both.

This module reuses dataset_validator.validate_all()'s own hard-error findings
rather than re-implementing schema/frame/fps checking a second time -- single
source of truth for "is this record's frame mapping actually verified."
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
sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(DATASET_ROOT))

import dataset_validator                    # noqa: E402
from schema import taxonomy                  # noqa: E402

ANNOTATIONS_DIR = DATASET_ROOT / "dataset" / "annotations"
CLIPS_DIR = DATASET_ROOT / "dataset" / "clips"
REPORTS_DIR = DATASET_ROOT / "dataset" / "reports"

# Hard-error "check" values (from dataset_validator.py's own findings) that
# mean "this record's frame/timestamp mapping is not verified" as opposed to
# some other, unrelated hard error (e.g. a duplicate annotation_id elsewhere
# in the dataset does not itself make THIS record's frame mapping bad).
#
# CORRECTED during infrastructure acceptance (task item #12/#7 cross-check):
# dataset_validator.py emits a single combined check named
# "frame_fps_consistency" for both the frame-range and fps-consistency logic
# (see dataset_validator.py's validate_all(), section (f)/(g)) -- it does NOT
# emit two separately-named checks "frame_range_consistency"/"fps_consistency".
# The previous version of this constant used those two non-existent names, so
# a real frame/fps hard error would never have matched here: it would have
# silently fallen through to `non_frame_hard_errors` and been reported as the
# more generic SCHEMA_OR_TAXONOMY_INVALID reason instead of the correct,
# more specific SOURCE_MAPPING_UNAVAILABLE reason. Verified against the real
# check name by reading dataset_validator.py's source directly, and by
# cross-checking frame_safety_acceptance_tests.py's FPS regression test
# (task item #7), which independently needed and confirmed the same name.
FRAME_MAPPING_CHECKS = {"frame_fps_consistency"}


def _hard_errors_by_annotation(report: dict) -> dict:
    by_id = {}
    for e in report.get("hard_errors", []):
        aid = e.get("annotation_id")
        if aid:
            by_id.setdefault(aid, []).append(e)
    return by_id


def classify(annotation: dict, hard_errors_for_this: list, clip_extracted: bool) -> dict:
    reasons = []
    human = annotation.get("human_label", {})
    sel = human.get("selected_player", {})

    non_frame_hard_errors = [e for e in hard_errors_for_this if e.get("check") not in FRAME_MAPPING_CHECKS]
    frame_hard_errors = [e for e in hard_errors_for_this if e.get("check") in FRAME_MAPPING_CHECKS]

    if non_frame_hard_errors:
        reasons.append({"code": "SCHEMA_OR_TAXONOMY_INVALID",
                         "detail": [e["check"] for e in non_frame_hard_errors]})

    conf = human.get("annotation_confidence")
    if conf not in taxonomy.TRAINING_ELIGIBLE_CONFIDENCE:
        reasons.append({"code": "LOW_ANNOTATION_CONFIDENCE", "detail": conf})

    if sel.get("manual_identity_confidence") == "LOW":
        reasons.append({"code": "PLAYER_IDENTITY_AMBIGUOUS", "detail": "manual_identity_confidence=LOW"})

    if annotation.get("status") == "DOUBLE_PASS_DISAGREEMENT":
        reasons.append({"code": "UNRESOLVED_DISAGREEMENT", "detail": "status=DOUBLE_PASS_DISAGREEMENT"})

    if human.get("ambiguous") or human.get("review_required"):
        reasons.append({"code": "DISPUTED_GROUND_TRUTH",
                         "detail": {"ambiguous": human.get("ambiguous"),
                                    "review_required": human.get("review_required")}})

    if not clip_extracted:
        reasons.append({"code": "SOURCE_MAPPING_UNAVAILABLE", "detail": "clip not yet extracted "
                         "(no dataset/clips/<clip_id>/clip_time_map.json)"})
    elif frame_hard_errors:
        reasons.append({"code": "SOURCE_MAPPING_UNAVAILABLE",
                         "detail": [e["check"] for e in frame_hard_errors]})

    return {
        "annotation_id": annotation["annotation_id"],
        "clip_id": annotation["clip_id"],
        "match_id": annotation["match_id"],
        "status": annotation["status"],
        "eligible": len(reasons) == 0,
        "reasons": reasons,
    }


def run(annotations_dir: Path = ANNOTATIONS_DIR, clips_dir: Path = CLIPS_DIR) -> dict:
    val_report = dataset_validator.validate_all(annotations_dir=annotations_dir, clips_dir=clips_dir)
    errors_by_id = _hard_errors_by_annotation(val_report)

    results = []
    for path in sorted(annotations_dir.glob("*.json")):
        try:
            ann = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if "annotation_id" not in ann:
            continue
        clip_extracted = (clips_dir / ann.get("clip_id", "")).joinpath("clip_time_map.json").exists()
        results.append(classify(ann, errors_by_id.get(ann["annotation_id"], []), clip_extracted))

    eligible = [r for r in results if r["eligible"]]
    not_eligible = [r for r in results if not r["eligible"]]
    reason_counts = Counter(rc["code"] for r in not_eligible for rc in r["reasons"])

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_annotations": len(results),
        "n_training_eligible": len(eligible),
        "n_not_training_eligible": len(not_eligible),
        "not_eligible_reason_counts": dict(reason_counts),
        "results": results,
    }
    return report


def print_summary(report: dict) -> None:
    print(f"training_eligibility: {report['n_annotations']} annotation(s) classified")
    print(f"  TRAINING_ELIGIBLE:     {report['n_training_eligible']}")
    print(f"  NOT_TRAINING_ELIGIBLE: {report['n_not_training_eligible']}")
    if report["not_eligible_reason_counts"]:
        print("  reasons (a record may have more than one):")
        for code, n in sorted(report["not_eligible_reason_counts"].items(), key=lambda kv: -kv[1]):
            print(f"    {code:<28} {n}")
    print()
    for r in report["results"]:
        verdict = "ELIGIBLE    " if r["eligible"] else "NOT_ELIGIBLE"
        codes = ",".join(rc["code"] for rc in r["reasons"]) or "-"
        print(f"  {verdict}  {r['annotation_id'][:8]}  clip={r['clip_id']:<28} {codes}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations-dir", type=Path, default=ANNOTATIONS_DIR)
    ap.add_argument("--clips-dir", type=Path, default=CLIPS_DIR)
    ap.add_argument("--out", type=Path, default=REPORTS_DIR / "training_eligibility.json")
    args = ap.parse_args()

    report = run(args.annotations_dir, args.clips_dir)
    print_summary(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nfull report written to {args.out}")


if __name__ == "__main__":
    main()

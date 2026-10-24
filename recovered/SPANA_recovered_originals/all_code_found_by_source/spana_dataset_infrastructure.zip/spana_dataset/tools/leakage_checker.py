#!/usr/bin/env python3
"""
leakage_checker.py -- cross-split data leakage checks for the SPANA dataset
(audit items #21, #24, #25, #31, #32: a real train/val/test/gold split is
only meaningful if no footage that informed training can also be evaluated
on, directly or by proxy).

Splitting happens at MATCH granularity (split_generator.py), specifically so
that clips sharing camera angle, kit colours, pitch and lighting never
straddle a split boundary. This module is the independent auditor of that
invariant -- it never assumes split_generator.py did its job correctly, it
re-derives everything from the on-disk files:

  (a) No match_id is literally present with two different split assignments.
      This should be structurally impossible once dataset/splits/assignment.json
      is loaded as a normal Python dict (a dict key can only hold one value) --
      but a hand-edited or concatenated JSON file can contain the same key
      twice, and the stdlib json module silently keeps only the LAST
      occurrence, discarding the earlier one with no warning at all. That is
      exactly the kind of silent-truncation bug this project's house style
      (see time_utils.py's module docstring) treats as unacceptable, so this
      module re-reads assignment.json's raw text itself (see
      `load_assignment_file`) rather than trusting a plain `json.loads`.

  (b) No two different match_ids that resolve to the SAME underlying
      source_video_id end up in different splits. Two match_ids are supposed
      to be two genuinely different matches, but a metadata mistake (the
      same source video registered, or re-registered, under two match_ids --
      e.g. a match re-uploaded after a filename change and re-ingested as
      "new") would put literally the same footage on both sides of a split
      boundary even though the match_ids themselves are cleanly separated.
      This is exactly the audit's leakage concern and is treated as a HARD
      finding when it happens. The source_video_id <-> match_id relationship
      is read from dataset/metadata/source_videos.json (the registry) when
      available, and cross-checked against the (match_id, source_video_id)
      pairs recorded directly on annotation records -- both are consulted
      because the registry may not be fully populated yet while annotations
      already reference real source videos (this is a legitimate early-
      dataset state, not a bug -- see the parallel-work note in the project
      brief).

  (c) Any annotation whose match_id has no entry at all in assignment.json
      yet. This is INFORMATIONAL, not an error: it just means that match
      hasn't been run through split_generator.py yet, which is completely
      normal while the dataset is still being built up.

Two ways to use this module:
  - Importable: `check_leakage(assignment, annotations)` -- used by
    dataset_validator.py so a single validation run folds leakage findings
    into the same report, rather than requiring a second manual step.
  - Standalone CLI:
        python leakage_checker.py --out dataset/reports/leakage_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
DATASET_ROOT = TOOLS_DIR.parent

ANNOTATIONS_DIR = DATASET_ROOT / "dataset" / "annotations"
SPLITS_PATH = DATASET_ROOT / "dataset" / "splits" / "assignment.json"
SOURCE_VIDEOS_PATH = DATASET_ROOT / "dataset" / "metadata" / "source_videos.json"
REPORTS_DIR = DATASET_ROOT / "dataset" / "reports"

VALID_SPLITS = ("TRAIN", "VALIDATION", "TEST", "GOLD_TEST")


class LeakageCheckerError(Exception):
    pass


# ---------------------------------------------------------------------------
# Defensive loaders -- never trust that a hand-editable JSON file was written
# correctly; verify it directly rather than assuming json.loads told the
# whole story (see module docstring, finding (a)).
# ---------------------------------------------------------------------------

def load_assignment_file(path: Path) -> tuple[dict, list[dict]]:
    """Reads dataset/splits/assignment.json and returns
    (assignment_dict, duplicate_key_findings).

    duplicate_key_findings is non-empty only if the raw JSON text contains
    the same match_id key more than once -- something a plain `json.loads`
    would silently paper over by keeping just the last occurrence. We detect
    this explicitly via `object_pairs_hook` so a hand-edited or badly
    concatenated assignment.json is caught here rather than producing a
    quietly-wrong split downstream.
    """
    if not path.exists():
        return {}, []

    duplicates: list[dict] = []

    def _pairs_hook(pairs):
        seen: dict[str, list] = defaultdict(list)
        for k, v in pairs:
            seen[k].append(v)
        out = {}
        for k, values in seen.items():
            if len(values) > 1 and len(set(values)) > 1:
                duplicates.append({
                    "match_id": k,
                    "conflicting_values": values,
                })
            out[k] = values[-1]  # mirror json.loads's own last-wins behaviour for `out`
        return out

    try:
        raw = path.read_text()
        assignment = json.loads(raw, object_pairs_hook=_pairs_hook)
    except json.JSONDecodeError as e:
        raise LeakageCheckerError(f"{path} is not valid JSON: {e}") from e

    if not isinstance(assignment, dict):
        raise LeakageCheckerError(f"{path} must contain a JSON object of match_id -> SPLIT_NAME")

    return assignment, duplicates


def load_annotations(annotations_dir: Path) -> list[dict]:
    """Loads every dataset/annotations/*.json file, skipping (and reporting
    on stderr, not raising) any file that isn't valid JSON -- a malformed
    annotation file is dataset_validator.py's job to flag as a hard error;
    leakage_checker only needs match_id/source_video_id and degrades
    gracefully around anything it can't parse."""
    out = []
    if not annotations_dir.exists():
        return out
    for p in sorted(annotations_dir.glob("*.json")):
        try:
            out.append(json.loads(p.read_text()))
        except json.JSONDecodeError as e:
            print(f"  (skipping unparseable annotation file {p.name}: {e})", file=sys.stderr)
    return out


def load_source_videos(path: Path) -> dict:
    """Loads dataset/metadata/source_videos.json (source_video_id -> record
    dict) if it exists. Returns {} if it doesn't -- the registry is being
    populated by other tools in parallel and may legitimately be empty or
    absent this early."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        print(f"  (warning: {path} is not valid JSON, ignoring for leakage checks: {e})", file=sys.stderr)
        return {}
    return data if isinstance(data, dict) else {}


# ---------------------------------------------------------------------------
# Core check
# ---------------------------------------------------------------------------

def check_leakage(assignment: dict, annotations: list[dict], source_videos: dict | None = None) -> dict:
    """Returns a structured leakage report. An empty set of findings under
    every key means clean.

    Parameters
    ----------
    assignment:
        Parsed dataset/splits/assignment.json, i.e. {match_id: SPLIT_NAME}.
        Callers that also want the raw-duplicate-key check (finding (a))
        should load this with `load_assignment_file` and fold its
        `duplicate_key_findings` into the report themselves (see `main()`
        below for the reference implementation) -- this function only
        checks the semantic content of whatever dict it is given, since a
        plain dict can never itself hold two values for one key.
    annotations:
        A list of parsed annotation records (or anything dict-shaped that at
        least has `match_id` / `source_video_id` -- this function is
        deliberately tolerant of otherwise-malformed records, since schema
        validity is dataset_validator.py's job, not this module's).
    source_videos:
        Optional parsed dataset/metadata/source_videos.json registry
        (source_video_id -> record dict, each ideally carrying its own
        `match_id`). May be None or {} if the registry doesn't exist yet.
    """
    source_videos = source_videos or {}

    # -- finding (a), semantic half: a dict can't literally hold two values
    # for one key, but we still sanity-check that every value is one of the
    # four known split names, since a typo here (e.g. "Train" vs "TRAIN")
    # would otherwise silently fail every downstream `== split` comparison.
    invalid_split_values = [
        {"match_id": mid, "value": split}
        for mid, split in assignment.items()
        if split not in VALID_SPLITS
    ]

    # -- gather (source_video_id -> {match_id: ...}) from every source we have.
    # A dict keyed by match_id (not a plain set) so we can also report, for
    # each source_video_id, which annotation/registry evidence produced each
    # match_id association -- useful when tracking down how the mix-up
    # happened, not just that it happened.
    svid_to_match_ids: dict[str, set] = defaultdict(set)
    svid_evidence: dict[str, dict] = defaultdict(lambda: defaultdict(list))

    for sv_id, rec in source_videos.items():
        if isinstance(rec, dict) and rec.get("match_id"):
            svid_to_match_ids[sv_id].add(rec["match_id"])
            svid_evidence[sv_id][rec["match_id"]].append("metadata/source_videos.json")

    match_ids_seen = set()
    annotation_match_ids = set()
    for ann in annotations:
        if not isinstance(ann, dict):
            continue
        match_id = ann.get("match_id")
        sv_id = ann.get("source_video_id")
        if match_id:
            match_ids_seen.add(match_id)
            annotation_match_ids.add(match_id)
        if sv_id and match_id:
            svid_to_match_ids[sv_id].add(match_id)
            ann_id = ann.get("annotation_id", "<unknown>")
            svid_evidence[sv_id][match_id].append(f"annotation:{ann_id}")

    # Also count match_ids that only appear in the registry (no annotations
    # yet) towards "known matches", for completeness of finding (c)'s
    # complement -- not required by the spec but harmless and informative.
    for sv_id, matches in svid_to_match_ids.items():
        match_ids_seen |= matches

    # -- finding (b): same source_video_id, different match_ids, AND those
    # match_ids resolve (via assignment) to genuinely different splits.
    source_video_split_conflicts = []
    source_video_shared_match_ids = []  # informational: same footage, multiple match_ids, but no split conflict (yet)
    for sv_id, matches in sorted(svid_to_match_ids.items()):
        if len(matches) < 2:
            continue
        splits_by_match = {m: assignment.get(m) for m in matches}
        resolved_splits = {s for s in splits_by_match.values() if s is not None}
        entry = {
            "source_video_id": sv_id,
            "match_ids": sorted(matches),
            "splits_by_match_id": splits_by_match,
            "evidence": {m: sorted(set(svid_evidence[sv_id][m])) for m in matches},
        }
        if len(resolved_splits) > 1:
            source_video_split_conflicts.append(entry)
        else:
            source_video_shared_match_ids.append(entry)

    # -- finding (c): annotated matches not yet split at all.
    unassigned_match_ids = sorted(m for m in annotation_match_ids if m not in assignment)

    hard_error_count = len(invalid_split_values) + len(source_video_split_conflicts)

    return {
        "invalid_split_values": invalid_split_values,
        "source_video_split_conflicts": source_video_split_conflicts,
        "source_video_shared_match_ids_same_split": source_video_shared_match_ids,
        "unassigned_match_ids": unassigned_match_ids,
        "n_known_match_ids": len(match_ids_seen),
        "n_assigned_match_ids": len(assignment),
        "hard_error_count": hard_error_count,
        "clean": hard_error_count == 0,
    }


def build_report(annotations_dir: Path = ANNOTATIONS_DIR, splits_path: Path = SPLITS_PATH,
                  source_videos_path: Path = SOURCE_VIDEOS_PATH) -> dict:
    assignment, duplicate_key_findings = load_assignment_file(splits_path)
    annotations = load_annotations(annotations_dir)
    source_videos = load_source_videos(source_videos_path)

    result = check_leakage(assignment, annotations, source_videos)
    result["duplicate_split_keys"] = duplicate_key_findings
    result["hard_error_count"] += len(duplicate_key_findings)
    result["clean"] = result["hard_error_count"] == 0
    result["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    result["assignment_path"] = str(splits_path)
    result["assignment_exists"] = splits_path.exists()
    result["annotations_dir"] = str(annotations_dir)
    result["n_annotations_scanned"] = len(annotations)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations-dir", type=Path, default=ANNOTATIONS_DIR)
    ap.add_argument("--splits-path", type=Path, default=SPLITS_PATH)
    ap.add_argument("--source-videos", type=Path, default=SOURCE_VIDEOS_PATH)
    ap.add_argument("--out", type=Path, default=REPORTS_DIR / "leakage_report.json")
    args = ap.parse_args()

    report = build_report(args.annotations_dir, args.splits_path, args.source_videos)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    print(f"leakage_checker: {report['n_annotations_scanned']} annotation(s) scanned, "
          f"{report['n_known_match_ids']} known match_id(s), "
          f"{report['n_assigned_match_ids']} assigned in {args.splits_path.name if report['assignment_exists'] else '(no assignment.json yet)'}")
    print(f"  duplicate_split_keys:            {len(report['duplicate_split_keys'])}")
    print(f"  invalid_split_values:            {len(report['invalid_split_values'])}")
    print(f"  source_video_split_conflicts:    {len(report['source_video_split_conflicts'])}  <-- HARD if nonzero")
    print(f"  unassigned_match_ids (info):     {len(report['unassigned_match_ids'])}")
    print(f"  clean: {report['clean']}")
    print(f"  full report written to {args.out}")

    sys.exit(0 if report["clean"] else 1)


if __name__ == "__main__":
    main()

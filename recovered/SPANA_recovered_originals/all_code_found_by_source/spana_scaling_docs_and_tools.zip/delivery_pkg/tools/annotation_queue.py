#!/usr/bin/env python3
"""
annotation_queue.py -- task item #24: ranks CANDIDATE action windows
(timestamps a human or the pipeline has flagged as worth annotating, not yet
turned into a canonical annotation) by a simple, deliberately un-fancy
combination of:

  1. event-class shortage  -- how far behind collection_targets.json this
     candidate's (guessed) event type currently is,
  2. match underrepresentation -- how few exportable annotations already
     exist for this candidate's match_id, relative to the per-match cap,
  3. camera-domain underrepresentation -- how few exportable annotations
     exist from a match sharing this candidate's camera_type,
  4. lighting-domain underrepresentation -- same, for lighting.

Per the task brief ("do not over-engineer scoring"), this is an unweighted
sum of four 0..1 gap ratios, not a tuned or learned ranking function, and it
is not itself an ML model or a production event-detection rule -- it never
reads, calls, or is called by football_analysis/ or player_tracker/.

A candidate whose match_id has already reached PER_MATCH_CAP is flagged and
sorted to the bottom regardless of its raw score (task item #21) -- breadth
across matches is enforced structurally, not just nudged by weighting.

This tool does NOT invent candidates -- it ranks whatever is fed to it via
--candidates (a JSON list of {candidate_id, match_id, event_type_hint,
notes}). Candidates come from a human reviewing footage, optionally aided by
the production pipeline SUGGESTING timestamps to look at (task item #22) --
never from treating pipeline output as if it were ground truth.

Usage:
    python annotation_queue.py --candidates candidates.json [--per-match-cap 6]
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
METADATA_DIR = DATASET_ROOT / "dataset" / "metadata"
sys.path.insert(0, str(TOOLS_DIR))

import collection_tracker as ctr   # noqa: E402
import source_registry as sr       # noqa: E402

DEFAULT_PER_MATCH_CAP = 6   # task item #21: recommended 5-8 during the early diversity stage


def _event_gap_ratios(targets: dict, tally: dict) -> dict:
    """label ('PASS_COMPLETED', 'NO_EVENT_BALL_OCCLUDED', ...) -> gap ratio
    in [0, 1], where 1.0 = completely unmet target, 0.0 = target already met."""
    ratios = {}
    for line in tally["lines"]:
        target = line["target"]
        current = line["current"]
        ratios[line["label"]] = max(0.0, min(1.0, 1.0 - (current / target if target else 1.0)))
    return ratios


def _per_match_counts(records: list) -> dict:
    counts: dict[str, int] = {}
    for ann in records:
        if ann.get("status") not in ctr.EXPORTABLE_STATUSES:
            continue
        mid = ann.get("match_id")
        if mid:
            counts[mid] = counts.get(mid, 0) + 1
    return counts


def _per_domain_counts(records: list, registry: dict, field: str) -> dict:
    """field in {'camera_type', 'lighting'}. Counts exportable annotations
    whose match_id resolves (via the registry) to each domain value."""
    match_to_value = {}
    for e in registry.values():
        match_to_value.setdefault(e.get("match_id"), e.get(field, "UNKNOWN"))
    counts: dict[str, int] = {}
    for ann in records:
        if ann.get("status") not in ctr.EXPORTABLE_STATUSES:
            continue
        value = match_to_value.get(ann.get("match_id"), "UNKNOWN")
        counts[value] = counts.get(value, 0) + 1
    return counts


def score_candidates(candidates: list, annotations_dir: Path = ANNOTATIONS_DIR,
                      registry_path: Path = sr.REGISTRY_PATH,
                      targets_path: Path = METADATA_DIR / "collection_targets.json",
                      per_match_cap: int = DEFAULT_PER_MATCH_CAP) -> dict:
    records, parse_errors = ctr.load_annotations(annotations_dir)
    targets = ctr.ensure_targets_file(targets_path, force_init=False)
    tally = ctr.tally_progress(records, targets)
    event_gaps = _event_gap_ratios(targets, tally)

    registry = sr.load_registry(registry_path)
    match_counts = _per_match_counts(records)
    camera_counts = _per_domain_counts(records, registry, "camera_type")
    lighting_counts = _per_domain_counts(records, registry, "lighting")
    max_match_count = max(match_counts.values(), default=0) or 1
    max_camera_count = max(camera_counts.values(), default=0) or 1
    max_lighting_count = max(lighting_counts.values(), default=0) or 1

    def _match_gap(match_id):
        return max(0.0, 1.0 - (match_counts.get(match_id, 0) / max_match_count))

    def _domain_gap(counts, value, max_count):
        return max(0.0, 1.0 - (counts.get(value, 0) / max_count))

    scored = []
    for c in candidates:
        match_id = c.get("match_id")
        event_hint = c.get("event_type_hint")
        reg_entry = next((e for e in registry.values() if e.get("match_id") == match_id), {})
        camera_type = reg_entry.get("camera_type", "UNKNOWN")
        lighting = reg_entry.get("lighting", "UNKNOWN")

        event_gap = max((v for k, v in event_gaps.items() if k.startswith(f"{event_hint}_")), default=0.5) \
            if event_hint else 0.5
        match_gap = _match_gap(match_id)
        camera_gap = _domain_gap(camera_counts, camera_type, max_camera_count)
        lighting_gap = _domain_gap(lighting_counts, lighting, max_lighting_count)

        score = event_gap + match_gap + camera_gap + lighting_gap   # unweighted sum, deliberately simple
        at_cap = match_counts.get(match_id, 0) >= per_match_cap

        scored.append({
            **c,
            "camera_type": camera_type, "lighting": lighting,
            "event_gap": round(event_gap, 3), "match_gap": round(match_gap, 3),
            "camera_gap": round(camera_gap, 3), "lighting_gap": round(lighting_gap, 3),
            "score": round(score, 3),
            "per_match_cap_reached": at_cap,
            "current_annotations_for_this_match": match_counts.get(match_id, 0),
        })

    # Cap-respecting candidates first, ranked by score descending; capped
    # candidates listed after, also by score, but clearly flagged (task #21:
    # breadth enforced structurally, not just via a scoring nudge).
    scored.sort(key=lambda c: (c["per_match_cap_reached"], -c["score"]))

    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "per_match_cap": per_match_cap,
        "n_candidates": len(scored),
        "n_over_cap": sum(1 for c in scored if c["per_match_cap_reached"]),
        "ranked_queue": scored,
        "current_match_counts": match_counts,
        "current_camera_domain_counts": camera_counts,
        "current_lighting_domain_counts": lighting_counts,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", type=Path, required=True,
                    help="JSON list of {candidate_id, match_id, event_type_hint, notes}")
    ap.add_argument("--per-match-cap", type=int, default=DEFAULT_PER_MATCH_CAP)
    ap.add_argument("--out", type=Path, default=REPORTS_DIR / "annotation_queue.json")
    args = ap.parse_args()

    candidates = json.loads(args.candidates.read_text())
    report = score_candidates(candidates, per_match_cap=args.per_match_cap)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    print(f"annotation_queue: {report['n_candidates']} candidate(s) ranked, "
          f"{report['n_over_cap']} already at/over the {report['per_match_cap']}-per-match cap")
    for c in report["ranked_queue"]:
        flag = " [CAP REACHED]" if c["per_match_cap_reached"] else ""
        print(f"  score={c['score']:<6} {c.get('candidate_id', '?'):<20} match={c.get('match_id')}{flag}")
    print(f"\nfull report written to {args.out}")


if __name__ == "__main__":
    main()

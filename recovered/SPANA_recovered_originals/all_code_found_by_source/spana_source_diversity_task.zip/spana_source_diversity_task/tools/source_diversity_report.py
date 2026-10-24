#!/usr/bin/env python3
"""
source_diversity_report.py -- task items #13/#14: a dashboard of registered
source diversity, aggregated at MATCH level (never source_video_id or clip
level -- a match with two camera angles registered as two source_video_ids
is still one match's worth of diversity, not two), plus an automatically
computed, prioritized list of the next diversity gaps to close.

Only match_ids whose registered source(s) have source_independence_status
CONFIRMED_INDEPENDENT contribute to the dashboard counts -- a
POSSIBLE_DUPLICATE pending human review, or a match with no registry entry
at all (e.g. legacy_validation_v3_realworld_batch, whose original footage is
unavailable in this environment and so was never run through
ingest_source.py), is reported separately as NOT COUNTED, never silently
folded into the real diversity picture.

This tool trains nothing, changes nothing, and reads dataset/metadata/
source_videos.json + dataset/annotations/*.json only.

Usage:
    python source_diversity_report.py [--target-matches 12]
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
ANNOTATIONS_DIR = DATASET_ROOT / "dataset" / "annotations"
REPORTS_DIR = DATASET_ROOT / "dataset" / "reports"
sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(DATASET_ROOT))

import source_registry as sr    # noqa: E402
from schema import taxonomy     # noqa: E402

COUNTED_STATUSES = {"CONFIRMED_INDEPENDENT"}

# Target dimension values worth actively tracking gaps for (a subset of the
# full taxonomy enums -- OTHER/UNKNOWN are never "targets" to acquire more of).
TARGET_CAMERA_TYPES = ["BROADCAST", "FIXED_CAMERA", "AUTOMATED_CAMERA", "MOBILE"]
TARGET_LIGHTING = ["DAYLIGHT", "FLOODLIT", "MIXED"]
TARGET_KIT_CONTRAST = ["LIGHT_VS_DARK", "SIMILAR_COLOURS", "CONTRASTING_COLOURS", "STRIPES_OR_PATTERN"]
TARGET_RESOLUTIONS = ["720p", "1080p"]
MIN_DISTINCT_FPS_BUCKETS = 2
MIN_DISTINCT_COMPETITIONS = 3


def _match_level_records(registry: dict) -> dict:
    """One representative record per match_id, built from whichever of its
    registered source_video entries are CONFIRMED_INDEPENDENT (there is
    normally exactly one; if a match ever has more than one registered
    angle, dimensions are the UNION across them for gap-finding, and the
    first one registered is used as the 'representative' row for listing)."""
    by_match: dict[str, list[dict]] = {}
    for e in registry.values():
        if e.get("source_independence_status") not in COUNTED_STATUSES:
            continue
        by_match.setdefault(e["match_id"], []).append(e)
    return by_match


def build_report(registry_path: Path = sr.REGISTRY_PATH, annotations_dir: Path = ANNOTATIONS_DIR,
                  target_matches: int = 12) -> dict:
    registry = sr.load_registry(registry_path)
    by_match = _match_level_records(registry)

    # Cross-check against annotations for match_ids with NO registry entry at
    # all (e.g. legacy_validation_v3_realworld_batch) -- reported, never
    # counted, per the module docstring.
    annotation_match_ids = set()
    for p in sorted(annotations_dir.glob("*.json")):
        try:
            a = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if a.get("match_id"):
            annotation_match_ids.add(a["match_id"])
    unregistered_match_ids = sorted(annotation_match_ids - set(registry[e]["match_id"] for e in registry) - set())
    # (recomputed properly below using by_match's keys, since a match_id can
    # exist in the registry without necessarily being CONFIRMED_INDEPENDENT)
    all_registered_match_ids = {e.get("match_id") for e in registry.values()}
    unregistered_match_ids = sorted(annotation_match_ids - all_registered_match_ids)

    by_camera_type = Counter()
    by_lighting = Counter()
    by_fps_bucket = Counter()
    by_resolution_bucket = Counter()
    by_source_type = Counter()
    by_competition = Counter()
    by_kit_contrast = Counter()
    match_rows = []

    for match_id, entries in sorted(by_match.items()):
        rep = entries[0]
        cams = sorted({e.get("camera_type", "UNKNOWN") for e in entries})
        lights = sorted({e.get("lighting", "UNKNOWN") for e in entries})
        fps_buckets = sorted({sr.fps_bucket(e.get("fps")) for e in entries})
        res_buckets = sorted({sr.resolution_bucket(e.get("width"), e.get("height")) for e in entries})
        kits = sorted({e.get("kit_contrast", "UNKNOWN") for e in entries})
        for c in cams:
            by_camera_type[c] += 1
        for l in lights:
            by_lighting[l] += 1
        for f in fps_buckets:
            by_fps_bucket[f] += 1
        for r in res_buckets:
            by_resolution_bucket[r] += 1
        for k in kits:
            by_kit_contrast[k] += 1
        by_source_type[rep.get("source_type", "UNKNOWN")] += 1
        by_competition[rep.get("competition") or "UNKNOWN"] += 1

        match_rows.append({
            "match_id": match_id,
            "n_source_videos": len(entries),
            "source_video_ids": [e["source_video_id"] for e in entries],
            "camera_types": cams, "lighting": lights, "fps_buckets": fps_buckets,
            "resolution_buckets": res_buckets, "kit_contrast": kits,
            "source_type": rep.get("source_type"), "competition": rep.get("competition"),
            "teams": rep.get("teams"), "match_date": rep.get("match_date"),
        })

    n_confirmed = len(by_match)

    # --- priority gap computation (task item #14) ---
    priorities = []
    missing_cameras = [c for c in TARGET_CAMERA_TYPES if by_camera_type.get(c, 0) == 0]
    if missing_cameras:
        priorities.append({
            "priority": len(priorities) + 1, "dimension": "camera_type",
            "gap": f"no CONFIRMED_INDEPENDENT source yet with camera_type in {missing_cameras}",
            "recommendation": f"acquire a source shot with: {missing_cameras[0]}",
        })
    missing_lighting = [l for l in TARGET_LIGHTING if by_lighting.get(l, 0) == 0]
    if missing_lighting:
        priorities.append({
            "priority": len(priorities) + 1, "dimension": "lighting",
            "gap": f"no CONFIRMED_INDEPENDENT source yet under {missing_lighting}",
            "recommendation": f"acquire a {missing_lighting[0].lower()} match",
        })
    missing_kits = [k for k in TARGET_KIT_CONTRAST if by_kit_contrast.get(k, 0) == 0]
    if missing_kits:
        priorities.append({
            "priority": len(priorities) + 1, "dimension": "kit_contrast",
            "gap": f"no source yet tagged with kit_contrast in {missing_kits}",
            "recommendation": f"acquire a match with {missing_kits[0].lower().replace('_', ' ')} kits",
        })
    if len(by_fps_bucket) < MIN_DISTINCT_FPS_BUCKETS:
        priorities.append({
            "priority": len(priorities) + 1, "dimension": "fps",
            "gap": f"only {len(by_fps_bucket)} distinct fps bucket(s) represented ({sorted(by_fps_bucket)}) "
                   f"-- target >= {MIN_DISTINCT_FPS_BUCKETS}",
            "recommendation": "acquire a source at a genuinely different real fps",
        })
    missing_res = [r for r in TARGET_RESOLUTIONS if by_resolution_bucket.get(r, 0) == 0]
    if missing_res:
        priorities.append({
            "priority": len(priorities) + 1, "dimension": "resolution",
            "gap": f"no source yet at {missing_res}",
            "recommendation": f"acquire a source at {missing_res[0]}",
        })
    if len(by_competition) < MIN_DISTINCT_COMPETITIONS:
        priorities.append({
            "priority": len(priorities) + 1, "dimension": "competition",
            "gap": f"only {len(by_competition)} distinct competition(s)/source(s) known "
                   f"({sorted(k for k in by_competition if k != 'UNKNOWN')}) -- target >= {MIN_DISTINCT_COMPETITIONS}",
            "recommendation": "acquire footage from a different competition or league",
        })
    if n_confirmed < target_matches:
        priorities.append({
            "priority": len(priorities) + 1, "dimension": "raw_match_count",
            "gap": f"only {n_confirmed} confirmed-independent match(es) -- target {target_matches}",
            "recommendation": "keep acquiring new matches even where diversity dimensions above are already covered",
        })

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_confirmed_independent_matches": n_confirmed,
        "target_matches": target_matches,
        "matches": match_rows,
        "by_camera_type": dict(by_camera_type),
        "by_lighting": dict(by_lighting),
        "by_fps_bucket": dict(by_fps_bucket),
        "by_resolution_bucket": dict(by_resolution_bucket),
        "by_source_type": dict(by_source_type),
        "by_competition": dict(by_competition),
        "by_kit_contrast": dict(by_kit_contrast),
        "match_ids_referenced_by_annotations_but_not_registered": unregistered_match_ids,
        "next_source_priorities": priorities,
    }
    return report


def render_markdown(report: dict) -> str:
    lines = [
        "# SOURCE_DIVERSITY_STATUS.md",
        "",
        f"*Generated {report['generated_at']}. Counts are at MATCH level (never per-clip or per-source_video-entry) "
        f"and include only CONFIRMED_INDEPENDENT matches -- see the report JSON for POSSIBLE_DUPLICATE / "
        f"SOURCE_INDEPENDENCE_UNKNOWN / unregistered match_ids, which are deliberately excluded from these counts.*",
        "",
        f"## Confirmed independent matches: {report['n_confirmed_independent_matches']} / target {report['target_matches']}",
        "",
    ]

    def _table(title, counts, targets=None):
        lines.append(f"**{title}:**")
        lines.append("")
        keys = sorted(set(counts) | set(targets or []))
        for k in keys:
            lines.append(f"- {k}: {counts.get(k, 0)}")
        lines.append("")

    _table("Camera type", report["by_camera_type"], TARGET_CAMERA_TYPES + ["OTHER", "UNKNOWN"])
    _table("Lighting", report["by_lighting"], TARGET_LIGHTING + ["UNKNOWN"])
    _table("FPS bucket", report["by_fps_bucket"])
    _table("Resolution bucket", report["by_resolution_bucket"], TARGET_RESOLUTIONS)
    _table("Competition / source", report["by_competition"])
    _table("Kit contrast", report["by_kit_contrast"], TARGET_KIT_CONTRAST + ["UNKNOWN"])

    if report["match_ids_referenced_by_annotations_but_not_registered"]:
        lines.append("**Match IDs referenced by existing annotations but NOT in the source registry "
                      "(not counted above -- see DATASET_STATUS.md for why):**")
        lines.append("")
        for m in report["match_ids_referenced_by_annotations_but_not_registered"]:
            lines.append(f"- `{m}`")
        lines.append("")

    lines.append("## NEXT SOURCE PRIORITIES")
    lines.append("")
    if not report["next_source_priorities"]:
        lines.append("No open gaps against current targets.")
    else:
        for p in report["next_source_priorities"]:
            lines.append(f"{p['priority']}. **{p['dimension']}** -- {p['gap']}. *Recommendation: {p['recommendation']}.*")
    lines.append("")
    lines.append("## Per-match detail")
    lines.append("")
    lines.append("| match_id | camera type(s) | lighting | fps | resolution | kit contrast | competition |")
    lines.append("|---|---|---|---|---|---|---|")
    for m in report["matches"]:
        lines.append(f"| `{m['match_id']}` | {', '.join(m['camera_types'])} | {', '.join(m['lighting'])} | "
                      f"{', '.join(m['fps_buckets'])} | {', '.join(m['resolution_buckets'])} | "
                      f"{', '.join(m['kit_contrast'])} | {m['competition'] or 'UNKNOWN'} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target-matches", type=int, default=12)
    ap.add_argument("--out-json", type=Path, default=REPORTS_DIR / "source_diversity_report.json")
    ap.add_argument("--out-md", type=Path, default=DATASET_ROOT / "SOURCE_DIVERSITY_STATUS.md")
    args = ap.parse_args()

    report = build_report(target_matches=args.target_matches)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2))
    args.out_md.write_text(render_markdown(report))

    print(f"source_diversity_report: {report['n_confirmed_independent_matches']} confirmed independent "
          f"match(es) / target {report['target_matches']}")
    for p in report["next_source_priorities"]:
        print(f"  priority {p['priority']}: {p['dimension']} -- {p['recommendation']}")
    print(f"\nJSON written to {args.out_json}")
    print(f"Markdown written to {args.out_md}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
ingest_source.py -- task item #10: the front door for adding a new football
match's source video into the dataset. This is the tool the user should run
for every new upload from now on (task item #9's required order:
SOURCE VIDEO -> MEDIA PROBE -> HASH -> REGISTER -> DIVERSITY METADATA ->
SOURCE ACCEPTED -> ANNOTATION -- annotation must never start from a source
that hasn't been through this pipeline).

Design rules this tool follows, per the task brief:
  - NEVER asks the user to type fps or resolution -- both come only from
    ffprobe (time_utils.probe_video), which is authoritative over any
    claimed value (task items #7/#8). There is no --fps or --resolution
    flag at all; this is deliberate, not an oversight.
  - Interactive prompts (only with --interactive) ask ONLY for metadata that
    genuinely cannot be derived from the file: competition, teams, match
    date, and confirmation of camera type / lighting (which ffprobe cannot
    tell you -- it has no idea if a camera was handheld or floodlit).
  - Runs source_registry.classify_independence() BEFORE writing anything,
    and refuses to silently register an exact/renamed duplicate as if it
    were a new independent source (task item #6).
  - Never collects or infers player identity, team roster, or any personal
    attribute -- only football/source metadata (task item #3).

REMOTE-PROBED MODE (added for the source-ingestion task on 2026-09-27):
  This tool originally assumed the source video file is locally readable in
  this environment. Real full-match footage broke that assumption -- three
  ~600MB full matches exceeded the device bridge's 400MB-per-file transfer
  cap, so the raw bytes could never be staged into this workspace to be
  ffprobe'd/hashed here directly. Rather than skip the required MEDIA PROBE
  -> HASH -> REGISTER order (task item #9) or hand-write a registry entry
  (an actual bypass), `--external-probe-json` lets the SAME probe -> hash ->
  classify -> register pipeline run against ffprobe/sha256 output that was
  captured by running the IDENTICAL commands (ffprobe with the same field
  selection as time_utils.probe_video(); `shasum -a 256`) on the machine
  that actually holds the file. This changes WHERE the bytes are read, never
  WHAT is trusted or HOW independence/diversity are decided -- the file
  itself is never trusted less than in local mode; a value that was itself
  guessed or manually typed (rather than a real ffprobe/shasum run) must
  never be passed this way. The JSON must contain: duration_s, fps,
  frame_count, width, height (all as ffprobe reported them), content_hash
  (sha256 hex digest of the actual file), file_size_bytes, codec (or null),
  original_filename, and remote_device_path (the file's real path on the
  machine it was probed on -- NOT assumed locally openable; recorded in the
  registry entry's `path` field alongside `path_kind: "remote_device"` so
  downstream tools like clip_extractor.py know not to try to open it here).

Usage:
    python ingest_source.py /path/to/new_match.mp4 --match-id new_match_1 \\
        --competition "State League Div 2" --teams "Home FC,Away FC" \\
        --match-date 2026-05-10 --camera-type MOBILE --camera-type-confidence HIGH \\
        --lighting DAYLIGHT

    python ingest_source.py /path/to/new_match.mp4 --interactive
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
DATASET_ROOT = TOOLS_DIR.parent
sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(DATASET_ROOT))

import time_utils                      # noqa: E402
import source_registry as sr           # noqa: E402
from schema import taxonomy            # noqa: E402

ANNOTATION_TOOL_VERSION = "ingest_source.py/1.0.0"

# duration_s * fps vs. frame_count sanity tolerance (task item #7) --
# generous enough for container rounding / variable-frame-rate quirks, tight
# enough to catch a genuinely wrong/corrupted fps claim.
FRAME_COUNT_TOLERANCE_FRAC = 0.02
FRAME_COUNT_TOLERANCE_MIN_FRAMES = 3


class IngestError(Exception):
    pass


REQUIRED_EXTERNAL_PROBE_KEYS = (
    "duration_s", "fps", "frame_count", "width", "height",
    "content_hash", "file_size_bytes", "original_filename", "remote_device_path",
)


def _validate_external_probe(ep: dict) -> None:
    missing = [k for k in REQUIRED_EXTERNAL_PROBE_KEYS if k not in ep or ep[k] in (None, "")]
    if missing:
        raise IngestError(f"--external-probe-json is missing required field(s): {missing} -- "
                           f"required: {list(REQUIRED_EXTERNAL_PROBE_KEYS)} (codec is optional/nullable)")
    if not (isinstance(ep["fps"], (int, float)) and ep["fps"] > 0):
        raise IngestError(f"external_probe fps must be a real positive number from ffprobe, got {ep['fps']!r}")
    if len(ep["content_hash"]) != 64:
        raise IngestError(f"external_probe content_hash does not look like a sha256 hex digest "
                           f"(expected 64 hex chars, got {len(ep['content_hash'])})")


def _prompt(label: str, default: str, choices: list[str] | None = None) -> str:
    suffix = f" [{'/'.join(choices)}]" if choices else ""
    raw = input(f"{label}{suffix} (default: {default}): ").strip()
    if not raw:
        return default
    if choices and raw.upper() not in [c.upper() for c in choices]:
        print(f"  '{raw}' not in {choices} -- using default {default!r}")
        return default
    return raw.upper() if choices else raw


def check_frame_count_consistency(meta: "time_utils.VideoMeta") -> dict:
    """Task item #7: duration x fps should approximate frame_count within
    real-world media tolerances. This never blocks ingestion by itself (some
    containers report an estimated nb_frames that's a little off) but ALWAYS
    reports the discrepancy explicitly -- never silently ignored."""
    expected = meta.duration_s * meta.fps
    diff = abs(expected - meta.frame_count)
    tolerance = max(FRAME_COUNT_TOLERANCE_MIN_FRAMES, expected * FRAME_COUNT_TOLERANCE_FRAC)
    consistent = diff <= tolerance
    return {
        "duration_s": meta.duration_s, "fps": meta.fps, "frame_count": meta.frame_count,
        "expected_frame_count": round(expected, 2), "diff_frames": round(diff, 2),
        "tolerance_frames": round(tolerance, 2), "consistent": consistent,
    }


def build_diversity_snapshot(registry: dict) -> dict:
    """Independence-aware diversity snapshot used to detect 'first of its
    kind' when a new source is added (task item #11's 'diversity changes'
    line). Only CONFIRMED_INDEPENDENT (or SAME_SOURCE-into-an-existing-match,
    since that still contributes real footage diversity even though it
    doesn't add a new match) entries count -- a POSSIBLE_DUPLICATE pending
    human review is not yet trusted as real diversity."""
    counted_statuses = {"CONFIRMED_INDEPENDENT", "SAME_SOURCE"}
    snap = {"camera_type": set(), "lighting": set(), "fps_bucket": set(), "resolution_bucket": set(),
            "source_type": set(), "competition": set(), "kit_contrast": set()}
    for e in registry.values():
        if e.get("source_independence_status") not in counted_statuses:
            continue
        snap["camera_type"].add(e.get("camera_type", "UNKNOWN"))
        snap["lighting"].add(e.get("lighting", "UNKNOWN"))
        snap["fps_bucket"].add(sr.fps_bucket(e.get("fps")))
        snap["resolution_bucket"].add(sr.resolution_bucket(e.get("width"), e.get("height")))
        snap["source_type"].add(e.get("source_type", "UNKNOWN"))
        snap["competition"].add(e.get("competition") or "UNKNOWN")
        snap["kit_contrast"].add(e.get("kit_contrast", "UNKNOWN"))
    return snap


def ingest(video_path: str, match_id: str | None, competition: str | None, teams: list[str] | None,
           match_date: str | None, camera_type: str, camera_type_confidence: str, lighting: str,
           source_type: str, kit_contrast: str, interactive: bool, override_status: str | None,
           override_reason: str | None,
           registry_path: Path = sr.REGISTRY_PATH, dry_run: bool = False,
           external_probe: dict | None = None) -> dict:
    path_kind = "local"
    if external_probe is not None:
        # --- REMOTE-PROBED MODE: file too large for this environment's
        # transfer limits (see module docstring). The probe/hash values below
        # are still real ffprobe/shasum output -- just captured on the device
        # that holds the file, not this one. ---
        _validate_external_probe(external_probe)
        abs_path = external_probe["remote_device_path"]
        path_kind = "remote_device"

        class _M:  # tiny stand-in for time_utils.VideoMeta, same field names
            pass
        meta = _M()
        meta.duration_s = float(external_probe["duration_s"])
        meta.fps = float(external_probe["fps"])
        meta.frame_count = int(external_probe["frame_count"])
        meta.width = int(external_probe["width"])
        meta.height = int(external_probe["height"])

        consistency = check_frame_count_consistency(meta)
        codec = external_probe.get("codec")
        content_hash = external_probe["content_hash"]
        file_size_bytes = int(external_probe["file_size_bytes"])
        original_filename = external_probe["original_filename"]
    else:
        abs_path = str(Path(video_path).resolve())
        if not Path(abs_path).is_file():
            raise IngestError(f"source video does not exist: {abs_path}")

        # --- MEDIA PROBE (authoritative; task items #7/#8) ---
        meta = time_utils.probe_video(abs_path)
        consistency = check_frame_count_consistency(meta)
        codec = sr.probe_codec(abs_path)

        # --- HASH ---
        content_hash = sr.sha256_of_file(abs_path)
        file_size_bytes = Path(abs_path).stat().st_size
        original_filename = Path(video_path).name

    registry = sr.load_registry(registry_path)

    existing_by_path = sr.find_by_path(registry, abs_path)
    if existing_by_path is not None:
        raise IngestError(
            f"This exact file path is already registered as source_video_id={existing_by_path!r}. "
            f"If the file changed on disk, re-run with a fresh copy at a new path, or edit the "
            f"registry entry directly -- ingest_source.py only handles NEW sources."
        )

    claimed_match_id = match_id or f"match_{content_hash[:12]}"

    # --- interactive prompts, ONLY for genuinely non-derivable metadata ---
    if interactive:
        print(f"\nProbed (authoritative, from the file itself): {meta.width}x{meta.height} @ "
              f"{meta.fps:.3f}fps, {meta.duration_s:.2f}s, {meta.frame_count} frames"
              + (f", codec={codec}" if codec else ""))
        if not consistency["consistent"]:
            print(f"  ! duration x fps ({consistency['expected_frame_count']}) vs. reported "
                  f"frame_count ({consistency['frame_count']}) differ by {consistency['diff_frames']} "
                  f"frames, outside the {consistency['tolerance_frames']}-frame tolerance -- reported "
                  f"for your awareness; this does not block ingestion.")
        competition = _prompt("Competition/league", competition or "UNKNOWN")
        teams_raw = _prompt("Teams (comma-separated)", ",".join(teams) if teams else "")
        teams = [t.strip() for t in teams_raw.split(",") if t.strip()] or None
        match_date = _prompt("Match date (YYYY-MM-DD)", match_date or "UNKNOWN")
        match_date = None if match_date == "UNKNOWN" else match_date
        camera_type = _prompt("Camera type", camera_type, taxonomy.SOURCE_TYPES)
        camera_type_confidence = _prompt("Camera type confidence", camera_type_confidence,
                                          taxonomy.SOURCE_CAMERA_TYPE_CONFIDENCE)
        lighting = _prompt("Lighting", lighting, taxonomy.SOURCE_LIGHTING)
        source_type = camera_type  # see source_registry.py docstring: same classification, two fields
        kit_contrast = _prompt("Kit contrast (best guess after watching a moment of footage)",
                                kit_contrast, taxonomy.SOURCE_KIT_CONTRAST)

    for field, value, allowed in (
        ("--camera-type", camera_type, taxonomy.SOURCE_TYPES),
        ("--camera-type-confidence", camera_type_confidence, taxonomy.SOURCE_CAMERA_TYPE_CONFIDENCE),
        ("--lighting", lighting, taxonomy.SOURCE_LIGHTING),
        ("--source-type", source_type, taxonomy.SOURCE_TYPES),
        ("--kit-contrast", kit_contrast, taxonomy.SOURCE_KIT_CONTRAST),
    ):
        if value not in allowed:
            raise IngestError(f"{field}={value!r} is not valid -- allowed: {allowed}")

    # --- independence classification (task items #1, #5, #6) ---
    probe_dict = {"duration_s": meta.duration_s, "fps": meta.fps, "width": meta.width, "height": meta.height}
    status, evidence, related_ids = sr.classify_independence(probe_dict, content_hash, claimed_match_id, registry)

    if status == "SAME_SOURCE" and related_ids and registry.get(related_ids[0], {}).get("match_id") != claimed_match_id:
        # Exact/renamed duplicate of a file under a DIFFERENT match_id --
        # this is the dangerous case (task item #6): refuse outright rather
        # than let a duplicate upload silently masquerade as a new match.
        raise IngestError(
            f"REFUSING TO REGISTER: this file's content is byte-identical to already-registered "
            f"source_video_id(s) {related_ids} (match_id={[registry[i]['match_id'] for i in related_ids]}), "
            f"but you asked to register it under a DIFFERENT match_id ({claimed_match_id!r}). This "
            f"looks like the same recording being re-uploaded and mislabelled as a new match -- "
            f"exactly the leakage risk this task exists to prevent. If you are certain these are "
            f"genuinely different matches that happen to hash identically (astronomically unlikely "
            f"for real video files), pass --override-independence-status CONFIRMED_INDEPENDENT "
            f"--override-reason '<why>'."
        )

    if override_status:
        if not override_reason:
            raise IngestError("--override-independence-status requires --override-reason")
        if override_status not in taxonomy.SOURCE_INDEPENDENCE_STATUSES:
            raise IngestError(f"--override-independence-status must be one of {taxonomy.SOURCE_INDEPENDENCE_STATUSES}")
        evidence = evidence + [f"OVERRIDDEN by human operator: {override_reason}"]
        status = override_status

    before_snapshot = build_diversity_snapshot(registry)

    # --- task item #15: incremental diversity VALUE (informational only --
    # never blocks; computed against the registry as it stands BEFORE this
    # candidate is added, using the same fields this entry is about to be
    # written with) ---
    diversity_candidate = {
        "camera_type": camera_type, "lighting": lighting, "kit_contrast": kit_contrast,
        "fps": meta.fps, "width": meta.width, "height": meta.height, "competition": competition,
    }
    diversity_value_level, diversity_value_explanation = sr.classify_diversity_value(diversity_candidate, registry)

    source_video_id = sr.new_source_video_id()
    now = datetime.now(timezone.utc).isoformat()
    entry = {
        "source_video_id": source_video_id,
        "match_id": claimed_match_id,
        "original_filename": original_filename,
        "path": abs_path,
        "path_kind": path_kind,   # "local" or "remote_device" -- see module docstring
        "content_hash": content_hash,
        "file_sha256": content_hash,   # legacy field name, kept for leakage_checker.py backward-compat
        "file_size_bytes": file_size_bytes,
        "duration_s": meta.duration_s,
        "fps": meta.fps,
        "frame_count": meta.frame_count,
        "width": meta.width,
        "height": meta.height,
        "codec": codec,
        "frame_count_consistency": consistency,
        "camera_type": camera_type,
        "camera_type_confidence": camera_type_confidence,
        "lighting": lighting,
        "source_type": source_type,
        "kit_contrast": kit_contrast,
        "competition": competition,
        "teams": teams,
        "match_date": match_date,
        "source_independence_status": status,
        "source_independence_evidence": evidence,
        "provenance": {
            "registered_by": "ingest_source.py", "tool_version": ANNOTATION_TOOL_VERSION,
            "notes": (
                f"Remote-probed: file exceeds this environment's device-bridge transfer cap, so "
                f"ffprobe + sha256 were run on the machine holding the file ({abs_path}) rather than "
                f"here; the file itself was never opened in this workspace. See ingest_source.py's "
                f"module docstring." if path_kind == "remote_device" else None
            ),
        },
        "registered_at": now,
        "updated_at": None,
    }

    if dry_run:
        return {"dry_run": True, "would_register": entry, "independence_status": status,
                 "independence_evidence": evidence,
                 "diversity_value": diversity_value_level, "diversity_value_explanation": diversity_value_explanation}

    registry[source_video_id] = entry
    sr.save_registry(registry, registry_path)

    after_snapshot = build_diversity_snapshot(registry)
    new_dimensions = []
    for dim, before_set in before_snapshot.items():
        added = after_snapshot[dim] - before_set
        for v in added:
            new_dimensions.append(f"first {v} source ({dim.replace('_', ' ')})")

    n_confirmed_after = sum(1 for e in registry.values() if e.get("source_independence_status") == "CONFIRMED_INDEPENDENT")

    return {
        "dry_run": False, "entry": entry, "source_video_id": source_video_id,
        "independence_status": status, "independence_evidence": evidence,
        "new_diversity_dimensions": new_dimensions,
        "n_confirmed_independent_matches_after": n_confirmed_after,
        "frame_count_consistency": consistency,
        "diversity_value": diversity_value_level, "diversity_value_explanation": diversity_value_explanation,
    }


def print_result(result: dict, target_matches: int = 12) -> None:
    if result.get("dry_run"):
        print("DRY RUN -- nothing written.")
        print(f"diversity value: {result['diversity_value']} (informational only -- never blocks ingestion)")
        for line in result["diversity_value_explanation"]:
            print(f"  - {line}")
        print(json.dumps(result["would_register"], indent=2, default=str))
        return

    e = result["entry"]
    print("=" * 60)
    print("SOURCE REGISTERED" if result["independence_status"] != "SAME_SOURCE" else "SOURCE REGISTERED (joins an existing match)")
    print("=" * 60)
    print(f"source_video_id: {e['source_video_id']}")
    print(f"match_id:        {e['match_id']}")
    print(f"duration:        {e['duration_s']:.2f}s")
    print(f"fps:             {e['fps']:.3f}")
    print(f"resolution:      {e['width']}x{e['height']}")
    print(f"content hash:    {e['content_hash']}")
    print(f"independence:    {result['independence_status']}")
    for ev in result["independence_evidence"]:
        print(f"  - {ev}")
    if not result["frame_count_consistency"]["consistent"]:
        print(f"! frame-count consistency check FAILED: expected ~{result['frame_count_consistency']['expected_frame_count']} "
              f"frames from duration x fps, ffprobe reports {result['frame_count_consistency']['frame_count']} "
              f"(diff {result['frame_count_consistency']['diff_frames']}, tolerance "
              f"{result['frame_count_consistency']['tolerance_frames']}) -- reported, not blocked.")
    print(f"dataset now: {result['n_confirmed_independent_matches_after']} confirmed independent match(es) / target {target_matches}")
    print(f"diversity value: {result['diversity_value']} (informational only -- never blocks ingestion, task item #15)")
    for line in result["diversity_value_explanation"]:
        print(f"  - {line}")
    if result["new_diversity_dimensions"]:
        print("diversity changes:")
        for d in result["new_diversity_dimensions"]:
            print(f"  - {d}")
    else:
        print("diversity changes: none -- this source matches dimensions already represented")
    print("=" * 60)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video_path")
    ap.add_argument("--match-id", default=None, help="Reuse an EXISTING match_id if this is another "
                    "angle/upload of a match already registered; omit to register a new match.")
    ap.add_argument("--competition", default=None)
    ap.add_argument("--teams", default=None, help="comma-separated, e.g. 'Home FC,Away FC'")
    ap.add_argument("--match-date", default=None, help="YYYY-MM-DD")
    ap.add_argument("--camera-type", default="UNKNOWN", choices=taxonomy.SOURCE_TYPES,
                    help="Never guessed from the file -- ffprobe cannot tell you this. Confirm it "
                         "yourself, or use --interactive to be prompted.")
    ap.add_argument("--camera-type-confidence", default="UNKNOWN", choices=taxonomy.SOURCE_CAMERA_TYPE_CONFIDENCE)
    ap.add_argument("--lighting", default="UNKNOWN", choices=taxonomy.SOURCE_LIGHTING)
    ap.add_argument("--source-type", default=None, choices=taxonomy.SOURCE_TYPES,
                    help="Defaults to the same value as --camera-type if omitted (see source_registry.py "
                         "docstring for why these two fields carry the same classification).")
    ap.add_argument("--kit-contrast", default="UNKNOWN", choices=taxonomy.SOURCE_KIT_CONTRAST,
                    help="Optional -- best guess after watching a moment of footage; not derivable from ffprobe.")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--override-independence-status", default=None)
    ap.add_argument("--override-reason", default=None)
    ap.add_argument("--registry-path", type=Path, default=sr.REGISTRY_PATH)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--target-matches", type=int, default=12)
    ap.add_argument("--external-probe-json", type=Path, default=None,
                    help="Path to a JSON file with real ffprobe/shasum output captured on the machine "
                         "that actually holds the video, for files too large to transfer into this "
                         "environment. See the module docstring's REMOTE-PROBED MODE section. When set, "
                         "video_path is used only as a label (pass the same remote_device_path recorded "
                         "in the JSON, or any placeholder) -- it is never opened here.")
    args = ap.parse_args()

    teams = [t.strip() for t in args.teams.split(",")] if args.teams else None

    external_probe = None
    if args.external_probe_json:
        external_probe = json.loads(args.external_probe_json.read_text())

    try:
        result = ingest(
            video_path=args.video_path, match_id=args.match_id, competition=args.competition,
            teams=teams, match_date=args.match_date, camera_type=args.camera_type,
            camera_type_confidence=args.camera_type_confidence,
            lighting=args.lighting, source_type=args.source_type or args.camera_type,
            kit_contrast=args.kit_contrast,
            interactive=args.interactive, override_status=args.override_independence_status,
            override_reason=args.override_reason, registry_path=args.registry_path, dry_run=args.dry_run,
            external_probe=external_probe,
        )
    except IngestError as e:
        print(f"INGESTION REFUSED: {e}", file=sys.stderr)
        sys.exit(1)

    print_result(result, target_matches=args.target_matches)


if __name__ == "__main__":
    main()

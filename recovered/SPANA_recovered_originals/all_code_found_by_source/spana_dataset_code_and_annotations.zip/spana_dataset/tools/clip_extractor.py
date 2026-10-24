#!/usr/bin/env python3
"""
clip_extractor.py -- CLI tool that turns a slice of a full source match video
into a self-contained training clip: `dataset/clips/<clip_id>/clip.mp4` plus
the `clip_time_map.json` sidecar (`time_utils.ClipTimeMap`) that every
downstream tool (annotation server, `feature_exporter.py`,
`clip_pipeline_runner.py`) reads timestamps through. It also owns the
`dataset/metadata/source_videos.json` registry that gives every full match
video a stable `source_video_id` independent of where the file happens to
live on disk.

WHY THIS FILE EXISTS THE WAY IT DOES
-------------------------------------------------------------------------
1. **`ffmpeg -ss` before `-i` is not frame-accurate under stream copy.**
   `-c copy` can only cut on keyframe boundaries, so a naive
   `ffmpeg -ss <t> -i in.mp4 -t <d> -c copy out.mp4` silently produces a clip
   whose first frame is "whichever keyframe was nearest `<t>`", not frame
   `<t>`. Re-encoding instead (`-c:v libx264 ... -r <source_fps>`) forces
   ffmpeg to decode every source frame and lay down a *new*, exact,
   constant-frame-rate stream, so the requested window is honoured to the
   frame rather than to the nearest keyframe. `-ss` is still placed before
   `-i` here (for fast seeking to the neighbourhood of the cut) but because
   we re-encode, ffmpeg's default `accurate_seek` behaviour decodes forward
   from the keyframe to the exact requested timestamp before encoding
   starts -- it is the re-encode, not the position of `-ss`, that buys
   frame accuracy.
2. **The fps match is verified, never assumed, after extraction.** This
   project has already shipped a hard-coded-fps bug once (see
   `time_utils.py`'s module docstring). `feature_exporter.py` and
   `clip_pipeline_runner.py` both hard-fail (via `ClipPipelineError`) if a
   clip's own probed fps doesn't match `clip_time_map.json`'s
   `source_fps` within 0.01 -- so this tool performs that exact check
   itself, immediately after writing the file, rather than letting a bad
   clip be discovered downstream with a much less actionable error. On
   mismatch the half-written clip directory is removed and extraction
   fails loudly; nothing partial is left for a later tool to trip over.
3. **The SOURCE video is never assumed to start at frame 0 relative to the
   clip.** Only the extracted clip itself is zero-based. `absolute_start_frame`
   /`absolute_end_frame` in `clip_time_map.json` are always expressed in the
   SOURCE video's own frame numbering (per `time_utils.ClipTimeMap`'s field
   documentation), computed from the source's real, ffprobe'd fps -- never
   from an assumed 25/30.
4. **`ground_truth_event_window` and `analysis_context_window` are kept
   separate** (DATASET_SCHEMA.md sec 2, task item #6): the pre-roll/post-roll
   padding used for model context is recorded in `analysis_context_window_
   absolute_s` only; the raw `--event-start-s`/`--event-end-s` the caller
   passed in are recorded, unpadded and unclamped except by the source
   video's own real duration, in `ground_truth_event_window_absolute_s`.
5. **IDs are opaque and filename-independent** (DATASET_SCHEMA.md task item
   #8). This implementation uses `uuid4().hex` (32 lowercase hex chars, no
   dashes) for both `source_video_id` and `clip_id` -- not a ULID as the
   schema doc's reference implementation mentions, but equally opaque,
   equally collision-resistant, and equally uninformative about the
   original filename or extraction order, which is the only property
   anything downstream actually relies on.

WHAT THIS FILE DELIBERATELY DOES NOT DO
-------------------------------------------------------------------------
It does not write an annotation record (`schema/annotation.schema.json`) --
that is `annotation_server.py`'s job, once a human has actually watched the
clip and labelled it. `--interaction-time-s`/`--release-time-s` are accepted
here purely as an early sanity check (so a typo'd timestamp is caught at
extraction time, before anyone spends time annotating the wrong window) --
their values are validated but NOT persisted into `clip_time_map.json`,
which has no field for them (see `time_utils.ClipTimeMap`); they belong in
`human_label.ground_truth_event_window` once the annotation record is
written.

This project's own module -- it does not import `football_analysis` or
`player_tracker`, and nothing there imports it.

Usage:
    python clip_extractor.py register-source --path /videos/match17.mp4 \\
        --match-id match17 --competition "U18 League" \\
        --camera-type BROADCAST_WIDE --pitch-surface GRASS --lighting DAY

    python clip_extractor.py extract --source-video-id <id> \\
        --event-start-s 812.4 --event-end-s 815.1 \\
        --interaction-time-s 813.0 --release-time-s 813.6 \\
        --pre-roll-s 3.0 --post-roll-s 2.0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

TOOLS_DIR = Path(__file__).resolve().parent
DATASET_ROOT = TOOLS_DIR.parent
REPO_ROOT = DATASET_ROOT.parent
for _p in (TOOLS_DIR, DATASET_ROOT, REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import time_utils                      # noqa: E402  (this project's own module)
from schema import taxonomy            # noqa: E402

SOURCE_VIDEOS_REGISTRY = DATASET_ROOT / "dataset" / "metadata" / "source_videos.json"
CLIPS_DIR = DATASET_ROOT / "dataset" / "clips"
CLIP_FILENAME = "clip.mp4"

# How far a newly-encoded clip's own probed fps may drift from the source
# video's probed fps before extraction is refused. Matches the exact
# tolerance clip_pipeline_runner.load_annotation_and_clip() and
# feature_exporter.py already enforce downstream -- a clip that passes this
# check here is guaranteed not to be rejected by either of those later.
FPS_MATCH_TOLERANCE = 0.01


class ClipExtractorError(Exception):
    pass


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# source_videos.json registry
# ---------------------------------------------------------------------------

def _load_registry() -> dict:
    if not SOURCE_VIDEOS_REGISTRY.exists():
        return {}
    return json.loads(SOURCE_VIDEOS_REGISTRY.read_text())


def _save_registry(registry: dict) -> None:
    SOURCE_VIDEOS_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    SOURCE_VIDEOS_REGISTRY.write_text(json.dumps(registry, indent=2, sort_keys=True))


def _sha256_of_file(path: str, chunk_size: int = 1 << 20) -> str:
    """Streamed SHA-256 of the raw file bytes -- INFRASTRUCTURE ADDITION
    (acceptance task item #3: 'use source file hash ... to establish
    independence'). This is the one piece of evidence that can prove two
    DIFFERENT filenames/source_video_ids are actually byte-identical
    content (e.g. the same match re-uploaded/re-ingested under a new name)
    even when nothing else about their metadata suggests a link -- probed
    fps/resolution/duration alone cannot catch that case, since a re-
    encoded-but-identical-content file could plausibly probe differently,
    while a straight copy/re-upload probes identically AND hashes
    identically. Computed once at registration time and stored on the
    registry entry; never recomputed elsewhere, so two entries can be
    compared for exact-content duplication without re-reading either file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_by_path(registry: dict, abs_path: str) -> Optional[str]:
    """Idempotency key for register-source: match by absolute on-disk path,
    never by filename alone (a filename is not stable -- DATASET_SCHEMA.md's
    'stable IDs, independent of filename' principle applies just as much to
    *finding* an existing registration as it does to naming a new one)."""
    for source_video_id, entry in registry.items():
        if entry.get("path") == abs_path:
            return source_video_id
    return None


def register_source(path: str, match_id: str, competition: Optional[str],
                     camera_type: str, pitch_surface: str, lighting: str) -> str:
    abs_path = str(Path(path).resolve())
    if not Path(abs_path).is_file():
        raise ClipExtractorError(f"source video does not exist: {abs_path}")

    # Validated against schema/taxonomy.py -- the single source of truth for
    # this vocabulary (task item #34's "technical/footage facts only").
    # Never silently coerced to UNKNOWN on a typo; a bad value is a caller
    # error and must be reported as one.
    for field_name, value, allowed in (
        ("--camera-type", camera_type, taxonomy.CAMERA_TYPES),
        ("--pitch-surface", pitch_surface, taxonomy.PITCH_SURFACES),
        ("--lighting", lighting, taxonomy.LIGHTING_CONDITIONS),
    ):
        if value not in allowed:
            raise ClipExtractorError(f"{field_name}={value!r} is not valid -- allowed: {allowed}")

    # fps/frame_count/duration/width/height are ALWAYS read from the file
    # itself (time_utils.probe_video), never taken on the caller's word --
    # this is the same discipline time_utils.py's own module docstring
    # insists on, applied one level up at the registry that everything else
    # bootstraps from.
    meta = time_utils.probe_video(abs_path)
    file_sha256 = _sha256_of_file(abs_path)

    registry = _load_registry()
    existing_id = _find_by_path(registry, abs_path)
    entry = {
        "path": abs_path,
        "match_id": match_id,
        "competition": competition,
        "camera_type": camera_type,
        "pitch_surface": pitch_surface,
        "lighting": lighting,
        "fps": meta.fps,
        "frame_count": meta.frame_count,
        "duration_s": meta.duration_s,
        "width": meta.width,
        "height": meta.height,
        "file_sha256": file_sha256,
    }
    if existing_id is not None:
        # Idempotent re-run: reuse the existing id, refresh the probed
        # technical fields (the file on disk may have been re-encoded since
        # last registration) and metadata fields (a caller correcting a
        # typo'd --competition should not have to know the id to fix it),
        # but keep the original registered_at rather than pretending this
        # video was only just discovered.
        entry["registered_at"] = registry[existing_id].get("registered_at", _now_iso())
        entry["updated_at"] = _now_iso()
        source_video_id = existing_id
    else:
        entry["registered_at"] = _now_iso()
        source_video_id = uuid.uuid4().hex

    registry[source_video_id] = entry
    _save_registry(registry)
    return source_video_id


# ---------------------------------------------------------------------------
# ffmpeg extraction
# ---------------------------------------------------------------------------

def _run_ffmpeg_extract(source_path: str, start_s: float, duration_s: float,
                         source_fps: float, out_path: Path) -> None:
    """Re-encodes [start_s, start_s + duration_s) of source_path into
    out_path at exactly source_fps, constant frame rate. Never stream-copies
    (see module docstring point #1) -- this is the one and only place ffmpeg
    is invoked, so every clip this tool ever produces goes through the same
    frame-accurate path."""
    cmd = [
        "ffmpeg", "-y", "-nostdin",
        "-ss", f"{start_s:.6f}",
        "-i", source_path,
        "-t", f"{duration_s:.6f}",
        # Optional audio map ('?' suffix): many of this project's source
        # videos (e.g. pipeline debug renders) have no audio stream at all,
        # and a hard `-map 0:a:0` would abort the whole extraction over a
        # stream that was never expected to exist.
        "-map", "0:v:0", "-map", "0:a:0?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        # -r plus -vsync cfr is the actual frame-accuracy fix: it forces the
        # OUTPUT to a genuine constant frame rate matching the source's real,
        # ffprobe'd fps, rather than inheriting whatever variable timestamps
        # the source container happened to carry.
        "-r", f"{source_fps:.8f}", "-vsync", "cfr",
        "-c:a", "aac", "-b:a", "128k",
        str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise ClipExtractorError(
            f"ffmpeg failed extracting {source_path} -> {out_path}:\n"
            f"{proc.stderr[-4000:]}"
        )


def extract_clip(source_video_id: str, event_start_s: float, event_end_s: float,
                  pre_roll_s: float, post_roll_s: float,
                  interaction_time_s: Optional[float] = None,
                  release_time_s: Optional[float] = None,
                  clip_id: Optional[str] = None) -> "time_utils.ClipTimeMap":
    registry = _load_registry()
    entry = registry.get(source_video_id)
    if entry is None:
        raise ClipExtractorError(
            f"unknown source_video_id={source_video_id!r} -- known ids: {sorted(registry)}. "
            f"Run 'register-source' first."
        )
    source_path = entry["path"]
    if not Path(source_path).is_file():
        raise ClipExtractorError(
            f"source video for {source_video_id} is registered at {source_path!r} but that "
            f"file no longer exists -- re-run register-source with its new location."
        )

    # Re-probed fresh rather than trusting the registry's cached fields:
    # the registry can go stale (file replaced on disk since registration),
    # and this is the one number (source duration) the clamping below can't
    # afford to get from a possibly-stale cache.
    source_meta = time_utils.probe_video(source_path)

    if event_end_s <= event_start_s:
        raise ClipExtractorError(f"--event-end-s ({event_end_s}) must be after --event-start-s ({event_start_s})")
    if event_start_s < 0:
        raise ClipExtractorError(f"--event-start-s ({event_start_s}) is negative")
    if event_start_s >= source_meta.duration_s:
        raise ClipExtractorError(
            f"--event-start-s ({event_start_s}) is at or past the source video's real duration "
            f"({source_meta.duration_s}s) -- this event cannot exist in this source video."
        )
    if pre_roll_s < 0 or post_roll_s < 0:
        raise ClipExtractorError(f"--pre-roll-s/--post-roll-s must be >= 0 (got {pre_roll_s}, {post_roll_s})")

    # interaction_time_s/release_time_s are sanity-checked here (task-level
    # requirement) but intentionally NOT written into clip_time_map.json --
    # see module docstring's "WHAT THIS FILE DELIBERATELY DOES NOT DO".
    for label, t in (("--interaction-time-s", interaction_time_s), ("--release-time-s", release_time_s)):
        if t is not None and not (event_start_s - 1e-6 <= t <= event_end_s + 1e-6):
            raise ClipExtractorError(
                f"{label} ({t}) must fall within [--event-start-s, --event-end-s] "
                f"([{event_start_s}, {event_end_s}])"
            )

    # The analysis_context_window is the padded window actually extracted,
    # clamped to the source video's REAL duration (never to an assumed one) --
    # DATASET_SCHEMA.md sec 2's "ground_truth_event_window and
    # analysis_context_window are separate objects" (task item #6): the
    # ground-truth window recorded below is exactly the caller's
    # --event-start-s/--event-end-s, unpadded.
    context_start_s = max(0.0, event_start_s - pre_roll_s)
    context_end_s = min(source_meta.duration_s, event_end_s + post_roll_s)
    if context_end_s <= context_start_s:
        raise ClipExtractorError(
            f"computed analysis_context_window [{context_start_s}, {context_end_s}] is empty or "
            f"inverted after clamping to source duration {source_meta.duration_s}s"
        )

    clip_id = clip_id or uuid.uuid4().hex
    clip_dir = CLIPS_DIR / clip_id
    try:
        clip_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise ClipExtractorError(
            f"clip_dir already exists: {clip_dir} -- clip_id must be unique; pass a different "
            f"--clip-id or remove the existing directory if this extraction is being redone "
            f"on purpose."
        )

    clip_video_path = clip_dir / CLIP_FILENAME
    try:
        _run_ffmpeg_extract(source_path, context_start_s, context_end_s - context_start_s,
                             source_meta.fps, clip_video_path)

        # Verify, don't assume (module docstring point #2): this is the exact
        # check clip_pipeline_runner.py and feature_exporter.py repeat
        # downstream, performed here first so a bad extraction is caught at
        # the source instead of surfacing later with a less specific error.
        clip_meta = time_utils.probe_video(str(clip_video_path))
        if abs(clip_meta.fps - source_meta.fps) > FPS_MATCH_TOLERANCE:
            raise ClipExtractorError(
                f"extracted clip's own probed fps ({clip_meta.fps}) does not match the source "
                f"video's fps ({source_meta.fps}) within {FPS_MATCH_TOLERANCE} -- refusing to "
                f"write a clip that feature_exporter.py/clip_pipeline_runner.py would later "
                f"hard-fail on. ffmpeg stderr may explain why the -r/-vsync cfr re-encode "
                f"didn't take effect."
            )
    except Exception:
        # Never leave a half-written clip directory behind -- a failed
        # extraction must be as if it never started, not a landmine for the
        # next tool that globs dataset/clips/*.
        shutil.rmtree(clip_dir, ignore_errors=True)
        raise

    # absolute_end_frame is derived from the CLIP's own actually-written
    # frame_count, not recomputed from context_end_s -- ffmpeg's exact frame
    # count for a given duration can differ by one frame from a naive
    # seconds_to_frames() rounding, and the clip file on disk is the ground
    # truth of what was actually produced (same "probe, don't assume"
    # discipline as everywhere else in this module).
    absolute_start_frame = time_utils.seconds_to_frames(context_start_s, source_meta.fps)
    absolute_end_frame = absolute_start_frame + clip_meta.frame_count

    # clip_path is stored relative to DATASET_ROOT (not absolute) so the
    # whole dataset/ tree stays portable -- clip_pipeline_runner._resolve_clip_path()
    # already expects exactly this: an absolute path is used as-is, anything
    # else is resolved against DATASET_ROOT.
    clip_path_relative = str(clip_video_path.relative_to(DATASET_ROOT))

    ctm = time_utils.ClipTimeMap(
        source_video_id=source_video_id,
        source_video_path=source_path,
        source_fps=source_meta.fps,
        clip_id=clip_id,
        clip_path=clip_path_relative,
        absolute_start_frame=absolute_start_frame,
        absolute_end_frame=absolute_end_frame,
        ground_truth_event_window_absolute_s=(event_start_s, event_end_s),
        analysis_context_window_absolute_s=(context_start_s, context_end_s),
    )
    (clip_dir / "clip_time_map.json").write_text(json.dumps(ctm.to_dict(), indent=2))
    return ctm


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("register-source", help="register (or re-register) a full source match video")
    p1.add_argument("--path", required=True, help="path to the source video file")
    p1.add_argument("--match-id", required=True)
    p1.add_argument("--competition", default=None)
    p1.add_argument("--camera-type", default="UNKNOWN", choices=taxonomy.CAMERA_TYPES)
    p1.add_argument("--pitch-surface", default="UNKNOWN", choices=taxonomy.PITCH_SURFACES)
    p1.add_argument("--lighting", default="UNKNOWN", choices=taxonomy.LIGHTING_CONDITIONS)

    p2 = sub.add_parser("extract", help="extract a training clip + clip_time_map.json from a registered source video")
    p2.add_argument("--source-video-id", required=True)
    p2.add_argument("--event-start-s", required=True, type=float, help="absolute source-video seconds")
    p2.add_argument("--event-end-s", required=True, type=float, help="absolute source-video seconds")
    p2.add_argument("--interaction-time-s", type=float, default=None,
                     help="absolute source-video seconds; sanity-checked only, not persisted here "
                          "(belongs in the annotation record's ground_truth_event_window)")
    p2.add_argument("--release-time-s", type=float, default=None,
                     help="absolute source-video seconds; sanity-checked only, not persisted here")
    p2.add_argument("--pre-roll-s", required=True, type=float)
    p2.add_argument("--post-roll-s", required=True, type=float)
    p2.add_argument("--clip-id", default=None, help="defaults to a generated uuid4 hex string")

    args = ap.parse_args()

    if args.cmd == "register-source":
        source_video_id = register_source(
            path=args.path, match_id=args.match_id, competition=args.competition,
            camera_type=args.camera_type, pitch_surface=args.pitch_surface, lighting=args.lighting,
        )
        print(f"source_video_id={source_video_id}")
        return

    if args.cmd == "extract":
        ctm = extract_clip(
            source_video_id=args.source_video_id,
            event_start_s=args.event_start_s, event_end_s=args.event_end_s,
            pre_roll_s=args.pre_roll_s, post_roll_s=args.post_roll_s,
            interaction_time_s=args.interaction_time_s, release_time_s=args.release_time_s,
            clip_id=args.clip_id,
        )
        print(f"clip_id={ctm.clip_id}")
        print(f"clip_path={ctm.clip_path}")
        print(f"analysis_context_window_absolute_s={list(ctm.analysis_context_window_absolute_s)}")
        print(f"absolute_start_frame={ctm.absolute_start_frame} absolute_end_frame={ctm.absolute_end_frame}")
        return


if __name__ == "__main__":
    try:
        main()
    except (ClipExtractorError, time_utils.TimeUtilsError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

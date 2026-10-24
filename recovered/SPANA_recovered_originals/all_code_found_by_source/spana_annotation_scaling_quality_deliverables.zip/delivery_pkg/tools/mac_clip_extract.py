#!/usr/bin/env python3
"""
mac_clip_extract.py -- Mac-side (device_bash) mirror of clip_extractor.py's
extract_clip()/_run_ffmpeg_extract() for source videos that are too large to
bridge into the cloud workspace (>400MB device-bridge transfer cap).

WHY THIS FILE EXISTS
-------------------------------------------------------------------------
clip_extractor.py's extract_clip() is the project's one canonical, already-
verified frame-accurate extraction path. It cannot run against these three
sources because it executes entirely in the cloud workspace and the source
files (youtube_game_1/2/3.mp4, 580-620MB each) only exist on the user's Mac.
This script is NOT a reimplementation of extraction logic -- it is the exact
same ffmpeg command, the exact same fps-verification-after-the-fact
discipline, and the exact same frame-math, run where the bytes actually are.
Nothing here changes WHAT is trusted (still: ffprobe the real file, never a
registry cache or a filename-based guess), only WHERE the commands execute.

It deliberately does not import time_utils.py or clip_extractor.py (those
modules and this project are not present on the Mac's isolated VM) -- it
re-implements the same three primitives (probe_video, seconds_to_frames,
the ffmpeg command) verbatim so the two paths cannot silently diverge. Any
change to clip_extractor.py's ffmpeg args or frame math must be mirrored
here by hand.

Output: writes <out-dir>/clip.mp4 and <out-dir>/clip_time_map.json on the
MAC. clip_time_map.json is a superset of time_utils.ClipTimeMap's schema --
every field ClipTimeMap.from_dict() reads (source_video_id, source_video_path,
source_fps, clip_id, clip_path, absolute_start_frame, absolute_end_frame,
ground_truth_event_window_absolute_s, analysis_context_window_absolute_s) is
present with those exact names, plus extra provenance fields the cloud-side
registration step needs (match_id, source_filename, source_start_timestamp,
source_end_timestamp, clip_start_frame, clip_end_frame, extraction_command,
duration_s, generated_at). clip_path is deliberately left as a RELATIVE
placeholder ("clips/<clip_id>/clip.mp4") to be finalized once the clip is
placed under dataset/clips/ in the cloud workspace -- there is no DATASET_ROOT
on the Mac for it to be relative to.

Usage:
    python3 mac_clip_extract.py \\
        --source-path /Users/nathangeorgaklis/Downloads/youtube_game_2.mp4 \\
        --source-video-id cde90b4d11194889943edaa3cb809314 \\
        --match-id npl_nsw_nws_vs_sdr \\
        --clip-id src2_clip_001 \\
        --event-start-s 3110.0 --event-end-s 3120.0 \\
        --pre-roll-s 5.0 --post-roll-s 5.0 \\
        --out-dir /Users/nathangeorgaklis/Downloads/_spana_clips_out/src2_clip_001
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

FPS_MATCH_TOLERANCE = 0.01


class ExtractError(Exception):
    pass


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def probe_video(path: str) -> dict:
    """Verbatim port of time_utils.probe_video()'s logic (ffprobe, never
    assumed/hard-coded fps). Returns {fps, frame_count, duration_s, width, height}."""
    p = Path(path)
    if not p.exists():
        raise ExtractError(f"video file does not exist: {path}")
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate,avg_frame_rate,nb_frames,width,height,duration",
        "-show_entries", "format=duration",
        "-of", "json", str(p),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        raise ExtractError(f"ffprobe failed on {path}: {out.stderr}")
    data = json.loads(out.stdout)
    stream = (data.get("streams") or [{}])[0]
    fmt = data.get("format") or {}

    def _parse_rate(rate_str):
        if not rate_str or rate_str == "0/0":
            return None
        num, _, den = rate_str.partition("/")
        den = den or "1"
        try:
            return float(num) / float(den)
        except (ValueError, ZeroDivisionError):
            return None

    fps = _parse_rate(stream.get("r_frame_rate")) or _parse_rate(stream.get("avg_frame_rate"))
    if not fps or fps <= 0:
        raise ExtractError(f"could not determine a valid fps for {path} (stream={stream})")

    duration_s = None
    for cand in (stream.get("duration"), fmt.get("duration")):
        if cand not in (None, "N/A", ""):
            try:
                duration_s = float(cand)
                break
            except ValueError:
                continue
    if duration_s is None:
        raise ExtractError(f"could not determine duration for {path}")

    nb_frames = stream.get("nb_frames")
    if nb_frames not in (None, "N/A", ""):
        frame_count = int(nb_frames)
    else:
        frame_count = int(round(duration_s * fps))

    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    if width <= 0 or height <= 0:
        raise ExtractError(f"could not determine resolution for {path}")

    return {"path": str(p), "fps": fps, "frame_count": frame_count,
            "duration_s": duration_s, "width": width, "height": height}


def seconds_to_frames(seconds: float, fps: float) -> int:
    if fps <= 0:
        raise ExtractError(f"invalid fps={fps}")
    return int(round(seconds * fps))


def run_ffmpeg_extract(source_path: str, start_s: float, duration_s: float,
                        source_fps: float, out_path: Path) -> list:
    """Byte-for-byte the same ffmpeg invocation as clip_extractor.py's
    _run_ffmpeg_extract() -- accurate seek via re-encode, never stream-copy,
    output pinned to the SOURCE's real fps via -r/-vsync cfr."""
    cmd = [
        "ffmpeg", "-y", "-nostdin",
        "-ss", f"{start_s:.6f}",
        "-i", source_path,
        "-t", f"{duration_s:.6f}",
        "-map", "0:v:0", "-map", "0:a:0?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-r", f"{source_fps:.8f}", "-vsync", "cfr",
        "-c:a", "aac", "-b:a", "128k",
        str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise ExtractError(
            f"ffmpeg failed extracting {source_path} -> {out_path}:\n{proc.stderr[-4000:]}"
        )
    return cmd


def extract(source_path: str, source_video_id: str, match_id: str, clip_id: str,
            event_start_s: float, event_end_s: float,
            pre_roll_s: float, post_roll_s: float, out_dir: str) -> dict:
    source_path = str(Path(source_path).resolve())
    source_meta = probe_video(source_path)  # ALWAYS re-probed here, never trusted from a registry cache

    if event_end_s <= event_start_s:
        raise ExtractError(f"--event-end-s ({event_end_s}) must be after --event-start-s ({event_start_s})")
    if event_start_s < 0:
        raise ExtractError(f"--event-start-s ({event_start_s}) is negative")
    if event_start_s >= source_meta["duration_s"]:
        raise ExtractError(
            f"--event-start-s ({event_start_s}) is at/past the source's real duration "
            f"({source_meta['duration_s']}s)"
        )
    if pre_roll_s < 0 or post_roll_s < 0:
        raise ExtractError("pre/post roll must be >= 0")

    context_start_s = max(0.0, event_start_s - pre_roll_s)
    context_end_s = min(source_meta["duration_s"], event_end_s + post_roll_s)
    if context_end_s <= context_start_s:
        raise ExtractError(f"empty/inverted context window [{context_start_s}, {context_end_s}]")

    out_dir_p = Path(out_dir)
    if out_dir_p.exists() and any(out_dir_p.iterdir()):
        raise ExtractError(f"out-dir already exists and is non-empty: {out_dir_p} -- refuse to overwrite")
    out_dir_p.mkdir(parents=True, exist_ok=True)
    clip_video_path = out_dir_p / "clip.mp4"

    try:
        cmd = run_ffmpeg_extract(source_path, context_start_s, context_end_s - context_start_s,
                                  source_meta["fps"], clip_video_path)

        clip_meta = probe_video(str(clip_video_path))
        if abs(clip_meta["fps"] - source_meta["fps"]) > FPS_MATCH_TOLERANCE:
            raise ExtractError(
                f"extracted clip's own probed fps ({clip_meta['fps']}) does not match source fps "
                f"({source_meta['fps']}) within {FPS_MATCH_TOLERANCE} -- refusing to keep this clip."
            )
    except Exception:
        shutil.rmtree(out_dir_p, ignore_errors=True)
        raise

    absolute_start_frame = seconds_to_frames(context_start_s, source_meta["fps"])
    absolute_end_frame = absolute_start_frame + clip_meta["frame_count"]

    ctm = {
        # --- exact time_utils.ClipTimeMap.from_dict() required fields ---
        "source_video_id": source_video_id,
        "source_video_path": source_path,
        "source_fps": source_meta["fps"],
        "clip_id": clip_id,
        "clip_path": f"dataset/clips/{clip_id}/clip.mp4",  # relative-to-DATASET_ROOT, matching
        # clip_extractor.py's clip_path_relative convention (CLIPS_DIR = DATASET_ROOT/dataset/clips).
        # BUGFIX (local-clip-extraction task): this used to be written as f"clips/{clip_id}/clip.mp4"
        # (missing the "dataset/" prefix), which clip_pipeline_runner.py's _resolve_clip_path() would
        # have resolved to the WRONG path. All 18 clip_time_map.json files produced by the first run of
        # this script were patched post-hoc; this fix means future runs no longer need that patch step.
        "absolute_start_frame": absolute_start_frame,
        "absolute_end_frame": absolute_end_frame,
        "ground_truth_event_window_absolute_s": [event_start_s, event_end_s],
        "analysis_context_window_absolute_s": [context_start_s, context_end_s],
        # --- superset provenance fields (this task's item #8) ---
        "match_id": match_id,
        "source_filename": Path(source_path).name,
        "source_start_timestamp": context_start_s,
        "source_end_timestamp": context_end_s,
        "clip_start_frame": 0,
        "clip_end_frame": clip_meta["frame_count"],
        "clip_frame_count": clip_meta["frame_count"],
        "clip_fps_probed": clip_meta["fps"],
        "duration_s": context_end_s - context_start_s,
        "extraction_command": cmd,
        "extraction_mode": "remote_device (device_bash on Mac; source never bridged to cloud workspace)",
        "generated_at": _now_iso(),
        "generated_by": "mac_clip_extract.py/1.0.0",
    }
    (out_dir_p / "clip_time_map.json").write_text(json.dumps(ctm, indent=2))

    clip_bytes = clip_video_path.stat().st_size
    return {"ctm": ctm, "clip_bytes": clip_bytes, "source_meta": source_meta, "clip_meta": clip_meta}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-path", required=True)
    ap.add_argument("--source-video-id", required=True)
    ap.add_argument("--match-id", required=True)
    ap.add_argument("--clip-id", required=True)
    ap.add_argument("--event-start-s", required=True, type=float)
    ap.add_argument("--event-end-s", required=True, type=float)
    ap.add_argument("--pre-roll-s", required=True, type=float)
    ap.add_argument("--post-roll-s", required=True, type=float)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    try:
        result = extract(
            source_path=args.source_path, source_video_id=args.source_video_id,
            match_id=args.match_id, clip_id=args.clip_id,
            event_start_s=args.event_start_s, event_end_s=args.event_end_s,
            pre_roll_s=args.pre_roll_s, post_roll_s=args.post_roll_s,
            out_dir=args.out_dir,
        )
    except ExtractError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps({
        "clip_id": args.clip_id,
        "clip_bytes": result["clip_bytes"],
        "source_fps": result["source_meta"]["fps"],
        "clip_fps_probed": result["clip_meta"]["fps"],
        "analysis_context_window_absolute_s": result["ctm"]["analysis_context_window_absolute_s"],
        "ground_truth_event_window_absolute_s": result["ctm"]["ground_truth_event_window_absolute_s"],
        "absolute_start_frame": result["ctm"]["absolute_start_frame"],
        "absolute_end_frame": result["ctm"]["absolute_end_frame"],
    }, indent=2))


if __name__ == "__main__":
    main()

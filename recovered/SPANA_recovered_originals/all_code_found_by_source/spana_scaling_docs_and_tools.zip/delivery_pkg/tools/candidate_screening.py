#!/usr/bin/env python3
"""candidate_screening.py -- ANNOTATION-SCALING-QUALITY task: the mandatory
pre-extraction screening stage (task items #4-8).

FULL MATCH -> candidate timestamp -> SCREENING -> ACCEPT/REJECT/REVIEW ->
only ACCEPTed candidates ever reach tools/mac_clip_extract.py.

This module is deliberately dependency-light (stdlib + ffmpeg/ffprobe/
ImageMagick `convert`/`montage` on PATH) so it runs unmodified either in the
cloud workspace (against already-extracted dataset/clips/*/clip.mp4) or on
the user's Mac via device_bash (against a full source match, which is never
bridged off the Mac -- see mac_clip_extract.py's own docstring for why).

Two-step workflow, matching the task's own diagram:

  1. `preview`  -- given a source video and a candidate timestamp, sample a
     ~10-20s window around it (task item #6) into a labelled contact sheet
     (task item #7: several frames before, the interaction moment, several
     frames after) for a human to actually look at. Writes contact_sheet.png
     plus a small manifest (frame timestamps) next to it. Never classifies
     anything itself -- screening is not automatic (task item #8).

  2. `record`   -- after a human has looked at the contact sheet and answered
     the 6-question checklist (task item #5) and the LIVE_PLAY/STOPPAGE/
     REPLAY/UNKNOWN question (task item #8), write a screening record
     conforming to schema/screening.schema.json. Only a record with
     screening_decision=ACCEPT should ever be passed to mac_clip_extract.py.

Nothing here imports or is imported by football_analysis/ or player_tracker/.
"""
import argparse
import json
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = "1.0.0"
TOOL_VERSION = "candidate_screening.py/1.0.0"


def _run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, **kw)
    if r.returncode != 0:
        raise RuntimeError(f"command failed ({cmd[0]}): {' '.join(str(c) for c in cmd)}\n{r.stderr.decode(errors='replace')[-800:]}")
    return r


def build_preview_contact_sheet(source_path, candidate_timestamp_s, out_dir,
                                 window_s=16.0, n_frames=8, tile_scale="380:214"):
    """Sample n_frames evenly across [candidate_timestamp_s - window_s/2,
    candidate_timestamp_s + window_s/2] (clamped to >= 0) and montage them
    into a single labelled contact sheet. Returns a dict manifest."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    half = window_s / 2.0
    win_start = max(0.0, candidate_timestamp_s - half)
    win_end = candidate_timestamp_s + half
    times = [win_start + i * (win_end - win_start) / (n_frames - 1) for i in range(n_frames)]

    tiles = []
    frame_records = []
    for i, t in enumerate(times):
        png = out_dir / f"frame_{i:02d}.png"
        _run(["ffmpeg", "-y", "-nostdin", "-ss", f"{t:.3f}", "-i", str(source_path),
              "-frames:v", "1", "-vf", f"scale={tile_scale}", str(png)])
        lbl = out_dir / f"frame_{i:02d}_lbl.png"
        marker = " <-- CANDIDATE" if abs(t - candidate_timestamp_s) < (win_end - win_start) / (2 * n_frames) else ""
        _run(["convert", str(png), "-gravity", "South", "-background", "black", "-splice", "0x16",
              "-pointsize", "11", "-fill", "yellow", "-annotate", "+0+2", f"t={t:.2f}s{marker}", str(lbl)])
        tiles.append(str(lbl))
        frame_records.append({"index": i, "timestamp_s": t})

    sheet_path = out_dir / "contact_sheet.png"
    _run(["montage"] + tiles + ["-tile", f"{n_frames}x1", "-geometry", "+2+2", str(sheet_path)])

    manifest = {
        "source_path": str(source_path),
        "candidate_timestamp_s": candidate_timestamp_s,
        "preview_window_s": {"start_time_s": win_start, "end_time_s": win_end},
        "n_frames": n_frames,
        "frames": frame_records,
        "contact_sheet_path": str(sheet_path),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generated_by": TOOL_VERSION,
    }
    (out_dir / "preview_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def write_screening_record(*, candidate_source_video_id, candidate_match_id,
                            candidate_timestamp_s, intended_event_type,
                            checklist, live_play_status, screening_decision,
                            screened_by, out_path,
                            preview_window_s=None, priority_tier=None,
                            priority_factors=None, training_eligibility_preview=None,
                            rejection_reason=None, resulting_clip_id=None, notes=None):
    assert live_play_status in ("LIVE_PLAY", "STOPPAGE", "REPLAY", "UNKNOWN")
    assert screening_decision in ("ACCEPT", "REJECT", "REVIEW")
    record = {
        "schema_version": SCHEMA_VERSION,
        "screening_id": str(uuid.uuid4()),
        "candidate_source_video_id": candidate_source_video_id,
        "candidate_match_id": candidate_match_id,
        "candidate_timestamp_s": candidate_timestamp_s,
        "intended_event_type": intended_event_type,
        "preview_window_s": preview_window_s,
        "checklist": checklist,
        "live_play_status": live_play_status,
        "priority_tier": priority_tier,
        "priority_factors": priority_factors,
        "training_eligibility_preview": training_eligibility_preview,
        "screening_decision": screening_decision,
        "rejection_reason": rejection_reason,
        "resulting_clip_id": resulting_clip_id,
        "screened_by": screened_by,
        "screened_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "notes": notes,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2))
    return record


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_preview = sub.add_parser("preview", help="Build a contact sheet for a candidate timestamp")
    p_preview.add_argument("--source-path", required=True)
    p_preview.add_argument("--timestamp-s", required=True, type=float)
    p_preview.add_argument("--out-dir", required=True)
    p_preview.add_argument("--window-s", type=float, default=16.0)
    p_preview.add_argument("--n-frames", type=int, default=8)

    args = ap.parse_args()
    if args.cmd == "preview":
        manifest = build_preview_contact_sheet(args.source_path, args.timestamp_s, args.out_dir,
                                                window_s=args.window_s, n_frames=args.n_frames)
        print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

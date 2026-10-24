"""event_engine_v1/benchmark_fixture.py

Reusable benchmark fixture schema + fail-closed provenance validator.

Created for Task 3D, directly motivated by Task 3C's finding: the seed/
label data for shot_close_range_01 was silently reused against a different
video that had been uploaded to the same path after the original
validation artifacts were generated. Nothing in that pipeline ever checked
that the file on disk still matched the file the seed data was created
against.

A BenchmarkFixture binds a benchmark clip's seed/label data to the EXACT
video file it was created against -- by content hash and probed metadata,
never by path alone. validate_provenance() refuses to say a fixture is
usable unless the file currently on disk matches on every one of those
axes. It never raises, never auto-repairs, and never reuses cached
tracking/ball output to "work around" a mismatch -- it only ever reports
the truth back to the caller, which is responsible for stopping the run.

This module is purely additive: it imports nothing from, and makes no
change to, any frozen player_tracker / player_tracker_mo / football_analysis
file. It has no side effects beyond reading the video file(s) it is pointed
at and (optionally) writing the fixture JSON the caller asks it to write.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    """Stream a file's sha256 without loading it fully into memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_rate(r: Optional[str]) -> Optional[float]:
    if not r or r == "0/0":
        return None
    num, _, den = r.partition("/")
    den = den or "1"
    try:
        return round(float(num) / float(den), 6)
    except (ValueError, ZeroDivisionError):
        return None


def probe_video(path: str) -> dict:
    """Read width/height/fps/frame_count/duration via ffprobe. Read-only;
    never re-encodes, transcodes, or writes to the video."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames",
        "-show_entries", "format=duration",
        "-of", "json",
        path,
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    data = json.loads(out)
    stream = (data.get("streams") or [{}])[0]
    fmt = data.get("format", {})

    fps = _parse_rate(stream.get("avg_frame_rate")) or _parse_rate(stream.get("r_frame_rate"))
    frame_count = stream.get("nb_frames")
    frame_count = int(frame_count) if frame_count not in (None, "N/A") else None
    duration = fmt.get("duration")
    duration = float(duration) if duration is not None else None

    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": duration,
    }


@dataclass
class BenchmarkFixture:
    """One benchmark clip's frozen identity + seed/label data, bound to the
    EXACT source file it was created against."""

    clip_id: str
    video_path: str
    video_sha256: str
    file_size_bytes: int

    width: int
    height: int
    fps: float
    frame_count: Optional[int]
    duration_seconds: float

    target_player_label: str
    seed_frame: int
    seed_bbox: list  # [x1, y1, x2, y2]

    expected_primary_event: str
    expected_outcome: str
    expected_event_window: str

    fixture_created_timestamp: str
    fixture_provenance_notes: str

    tracking_artifact_sha256: Optional[str] = None
    ball_artifact_sha256: Optional[str] = None

    @classmethod
    def create(
        cls,
        clip_id: str,
        video_path: str,
        target_player_label: str,
        seed_frame: int,
        seed_bbox: list,
        expected_primary_event: str,
        expected_outcome: str,
        expected_event_window: str,
        fixture_provenance_notes: str,
        tracking_artifact_path: Optional[str] = None,
        ball_artifact_path: Optional[str] = None,
    ) -> "BenchmarkFixture":
        """Build a fixture by actually hashing/probing the CURRENT file on
        disk at video_path. This is the only supported constructor -- every
        field is derived from the file itself, never hand-typed, so a
        fixture can never silently drift from the file it claims to
        describe."""
        p = Path(video_path)
        if not p.is_file():
            raise FileNotFoundError(f"video_path does not exist: {video_path}")

        video_hash = sha256_file(video_path)
        size = p.stat().st_size
        meta = probe_video(video_path)

        return cls(
            clip_id=clip_id,
            video_path=str(p),
            video_sha256=video_hash,
            file_size_bytes=size,
            width=meta["width"],
            height=meta["height"],
            fps=meta["fps"],
            frame_count=meta["frame_count"],
            duration_seconds=meta["duration_seconds"],
            target_player_label=target_player_label,
            seed_frame=seed_frame,
            seed_bbox=list(seed_bbox),
            expected_primary_event=expected_primary_event,
            expected_outcome=expected_outcome,
            expected_event_window=expected_event_window,
            fixture_created_timestamp=datetime.now(timezone.utc).isoformat(),
            fixture_provenance_notes=fixture_provenance_notes,
            tracking_artifact_sha256=(sha256_file(tracking_artifact_path) if tracking_artifact_path else None),
            ball_artifact_sha256=(sha256_file(ball_artifact_path) if ball_artifact_path else None),
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=False)

    def save(self, out_path: str) -> str:
        """Write the fixture to disk and return the sha256 of the fixture
        FILE itself (the 'freeze and hash the fixture' step) -- distinct
        from video_sha256, which hashes the video the fixture describes."""
        Path(out_path).write_text(self.to_json())
        return sha256_file(out_path)

    @classmethod
    def load(cls, path: str) -> "BenchmarkFixture":
        data = json.loads(Path(path).read_text())
        return cls(**data)


@dataclass
class ProvenanceCheckResult:
    status: str  # "MATCH" or "BENCHMARK_PROVENANCE_MISMATCH"
    clip_id: str
    checked_at: str
    mismatches: list
    current: dict

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=False)


def validate_provenance(fixture: BenchmarkFixture, video_path: Optional[str] = None) -> ProvenanceCheckResult:
    """Fail-closed source-identity check.

    Recomputes sha256 / resolution / fps / frame_count / duration from the
    file CURRENTLY on disk and compares every one of them against the
    frozen fixture. Returns a MATCH or BENCHMARK_PROVENANCE_MISMATCH
    result.

    This function does NOT run tracking, does NOT reuse cached tracking or
    ball output, and does NOT regenerate or overwrite the fixture under any
    circumstance. A mismatch must always surface to a human and stop the
    run -- it is never silently repaired or worked around. It never raises
    on a mismatch (see require_valid_provenance() for a raising variant);
    its only job is to report the truth.
    """
    path = video_path or fixture.video_path
    p = Path(path)
    mismatches = []

    if not p.is_file():
        mismatches.append(f"video_path does not exist: {path}")
        return ProvenanceCheckResult(
            status="BENCHMARK_PROVENANCE_MISMATCH",
            clip_id=fixture.clip_id,
            checked_at=datetime.now(timezone.utc).isoformat(),
            mismatches=mismatches,
            current={},
        )

    current_hash = sha256_file(path)
    current_size = p.stat().st_size
    current_meta = probe_video(path)
    current = {"video_sha256": current_hash, "file_size_bytes": current_size, **current_meta}

    if current_hash != fixture.video_sha256:
        mismatches.append(f"video_sha256 mismatch: fixture={fixture.video_sha256} current={current_hash}")
    if current_size != fixture.file_size_bytes:
        mismatches.append(f"file_size_bytes mismatch: fixture={fixture.file_size_bytes} current={current_size}")
    if current_meta["width"] != fixture.width or current_meta["height"] != fixture.height:
        mismatches.append(
            f"resolution mismatch: fixture={fixture.width}x{fixture.height} "
            f"current={current_meta['width']}x{current_meta['height']}"
        )
    if fixture.fps and current_meta["fps"] and abs(current_meta["fps"] - fixture.fps) > 0.01:
        mismatches.append(f"fps mismatch: fixture={fixture.fps} current={current_meta['fps']}")
    if (
        fixture.frame_count is not None
        and current_meta["frame_count"] is not None
        and current_meta["frame_count"] != fixture.frame_count
    ):
        mismatches.append(f"frame_count mismatch: fixture={fixture.frame_count} current={current_meta['frame_count']}")
    if (
        fixture.duration_seconds is not None
        and current_meta["duration_seconds"] is not None
        and abs(current_meta["duration_seconds"] - fixture.duration_seconds) > 0.05
    ):
        mismatches.append(
            f"duration mismatch: fixture={fixture.duration_seconds} current={current_meta['duration_seconds']}"
        )

    status = "BENCHMARK_PROVENANCE_MISMATCH" if mismatches else "MATCH"
    return ProvenanceCheckResult(
        status=status,
        clip_id=fixture.clip_id,
        checked_at=datetime.now(timezone.utc).isoformat(),
        mismatches=mismatches,
        current=current,
    )


class BenchmarkProvenanceMismatch(Exception):
    def __init__(self, result: ProvenanceCheckResult):
        self.result = result
        super().__init__(
            f"BENCHMARK_PROVENANCE_MISMATCH for clip '{result.clip_id}': {'; '.join(result.mismatches)}"
        )


def require_valid_provenance(fixture: BenchmarkFixture, video_path: Optional[str] = None) -> ProvenanceCheckResult:
    """Convenience wrapper: raises BenchmarkProvenanceMismatch on any
    non-MATCH result, so a caller that forgets to check .status explicitly
    cannot accidentally fall through into running tracking on a
    provenance-invalid file. Orchestration scripts that want to print a
    structured STOP report and exit cleanly (this task's own run scripts)
    call validate_provenance() directly and branch on .status instead."""
    result = validate_provenance(fixture, video_path)
    if result.status != "MATCH":
        raise BenchmarkProvenanceMismatch(result)
    return result

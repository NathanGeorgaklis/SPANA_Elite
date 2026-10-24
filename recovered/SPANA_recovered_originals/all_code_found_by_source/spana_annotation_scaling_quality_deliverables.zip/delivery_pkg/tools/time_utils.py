"""
time_utils.py -- explicit, no-silent-assumptions frame/timestamp conversion
utilities for the SPANA dataset pipeline (audit task item #7).

This project's own validation history includes a real, previously-shipped
frame-offset bug (production tracking snapshots numbered 0-based per tracking
run, a validation harness that assumed absolute==relative, and a separate
hard-coded-25fps bug in pre-roll window sizing -- see the architecture audit's
TECHNICAL_DEBT_AUDIT.md section 7). Every function here exists specifically to
make the mistakes that caused those bugs structurally hard to repeat:

  - fps is ALWAYS read from the actual video file (ffprobe), never assumed.
  - "frame 0" is ALWAYS qualified as either "absolute source frame" or
    "clip-relative frame" -- there is no bare `frame` field anywhere in this
    module's public API.
  - Every conversion function takes fps as an explicit, required argument.
    There is no default fps anywhere in this file.

This module is standalone: it does NOT import football_analysis or
player_tracker, and nothing in football_analysis/player_tracker imports it.
It is purely dataset-pipeline infrastructure.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


class TimeUtilsError(Exception):
    pass


@dataclass(frozen=True)
class VideoMeta:
    """Ground-truth metadata for one source video file, read directly from
    the file itself via ffprobe -- never assumed, never hard-coded."""
    path: str
    fps: float                 # exact, e.g. 29.97002997002997 -- NOT rounded to 25/30
    frame_count: int           # ffprobe's reported total frame count (nb_frames or estimated)
    duration_s: float
    width: int
    height: int

    def to_dict(self) -> dict:
        return asdict(self)


def probe_video(path: str) -> VideoMeta:
    """Reads real fps/frame_count/duration/resolution from a video file via
    ffprobe. Raises TimeUtilsError if the file can't be probed -- callers
    must handle this explicitly rather than falling back to a guessed fps."""
    p = Path(path)
    if not p.exists():
        raise TimeUtilsError(f"video file does not exist: {path}")
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate,avg_frame_rate,nb_frames,width,height,duration",
        "-show_entries", "format=duration",
        "-of", "json", str(p),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=True)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        raise TimeUtilsError(f"ffprobe failed on {path}: {e}") from e
    data = json.loads(out.stdout)
    stream = (data.get("streams") or [{}])[0]

    def _parse_rate(rate_str):
        if not rate_str or rate_str == "0/0":
            return None
        num, _, den = rate_str.partition("/")
        den = den or "1"
        try:
            return float(num) / float(den)
        except (ValueError, ZeroDivisionError):
            return None

    fps = _parse_rate(stream.get("avg_frame_rate")) or _parse_rate(stream.get("r_frame_rate"))
    if not fps or fps <= 0:
        raise TimeUtilsError(f"could not determine a real fps for {path} -- refusing to assume 25/30")

    duration_s = None
    for src in (stream.get("duration"), (data.get("format") or {}).get("duration")):
        if src is not None:
            try:
                duration_s = float(src)
                break
            except ValueError:
                continue

    nb_frames = stream.get("nb_frames")
    if nb_frames is not None:
        try:
            frame_count = int(nb_frames)
        except ValueError:
            frame_count = None
    else:
        frame_count = None
    if frame_count is None:
        if duration_s is None:
            raise TimeUtilsError(f"could not determine frame_count or duration for {path}")
        frame_count = int(round(duration_s * fps))
    if duration_s is None:
        duration_s = frame_count / fps

    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)

    return VideoMeta(path=str(p), fps=fps, frame_count=frame_count,
                      duration_s=duration_s, width=width, height=height)


# ---------------------------------------------------------------------------
# Pure conversions -- fps is always an explicit required argument.
# ---------------------------------------------------------------------------

def frames_to_seconds(frame_count: int, fps: float) -> float:
    if fps <= 0:
        raise TimeUtilsError(f"invalid fps={fps}")
    return frame_count / fps


def seconds_to_frames(seconds: float, fps: float) -> int:
    if fps <= 0:
        raise TimeUtilsError(f"invalid fps={fps}")
    return int(round(seconds * fps))


@dataclass(frozen=True)
class ClipTimeMap:
    """The explicit mapping between a extracted training clip's own
    clip-relative timeline and its source video's absolute timeline. Every
    extracted clip (clip_extractor.py) writes one of these as a JSON sidecar,
    and every downstream tool (annotation server, feature exporter, benchmark
    tool) reads timestamps through this object rather than re-deriving them.

    Field names deliberately spell out "absolute" vs "clip_relative"
    everywhere -- no bare "frame" or "start" field that could be misread as
    the other kind, which is exactly the class of bug this project has
    already shipped once (see module docstring).
    """
    source_video_id: str
    source_video_path: str
    source_fps: float                      # from VideoMeta.fps -- the one true fps
    clip_id: str
    clip_path: str
    absolute_start_frame: int              # this clip's frame 0, expressed in source-video frames
    absolute_end_frame: int                # exclusive
    ground_truth_event_window_absolute_s: tuple   # (start_s, end_s) in SOURCE video seconds
    analysis_context_window_absolute_s: tuple     # (start_s, end_s) in SOURCE video seconds

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ground_truth_event_window_absolute_s"] = list(self.ground_truth_event_window_absolute_s)
        d["analysis_context_window_absolute_s"] = list(self.analysis_context_window_absolute_s)
        return d

    @staticmethod
    def from_dict(d: dict) -> "ClipTimeMap":
        return ClipTimeMap(
            source_video_id=d["source_video_id"],
            source_video_path=d["source_video_path"],
            source_fps=d["source_fps"],
            clip_id=d["clip_id"],
            clip_path=d["clip_path"],
            absolute_start_frame=d["absolute_start_frame"],
            absolute_end_frame=d["absolute_end_frame"],
            ground_truth_event_window_absolute_s=tuple(d["ground_truth_event_window_absolute_s"]),
            analysis_context_window_absolute_s=tuple(d["analysis_context_window_absolute_s"]),
        )

    # -- conversions, always explicit about which timeline they're in/out of --

    def clip_relative_seconds_to_absolute_seconds(self, clip_relative_s: float) -> float:
        absolute_start_s = frames_to_seconds(self.absolute_start_frame, self.source_fps)
        return absolute_start_s + clip_relative_s

    def absolute_seconds_to_clip_relative_seconds(self, absolute_s: float) -> float:
        absolute_start_s = frames_to_seconds(self.absolute_start_frame, self.source_fps)
        return absolute_s - absolute_start_s

    def clip_relative_seconds_to_absolute_frame(self, clip_relative_s: float) -> int:
        absolute_s = self.clip_relative_seconds_to_absolute_seconds(clip_relative_s)
        return seconds_to_frames(absolute_s, self.source_fps)

    def absolute_frame_to_clip_relative_seconds(self, absolute_frame: int) -> float:
        clip_relative_frame = absolute_frame - self.absolute_start_frame
        return frames_to_seconds(clip_relative_frame, self.source_fps)

    def ground_truth_event_window_clip_relative_s(self) -> tuple:
        s, e = self.ground_truth_event_window_absolute_s
        return (self.absolute_seconds_to_clip_relative_seconds(s),
                self.absolute_seconds_to_clip_relative_seconds(e))

    def analysis_context_window_clip_relative_s(self) -> tuple:
        s, e = self.analysis_context_window_absolute_s
        return (self.absolute_seconds_to_clip_relative_seconds(s),
                self.absolute_seconds_to_clip_relative_seconds(e))


def load_clip_time_map(clip_dir: str) -> ClipTimeMap:
    path = Path(clip_dir) / "clip_time_map.json"
    if not path.exists():
        raise TimeUtilsError(f"no clip_time_map.json found in {clip_dir} -- was this clip "
                              f"extracted with clip_extractor.py?")
    return ClipTimeMap.from_dict(json.loads(path.read_text()))

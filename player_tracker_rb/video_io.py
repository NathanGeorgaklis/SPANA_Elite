# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R0/R0_TRACKER_INTERFACE_MAP.md §1.10-§1.11 (seed frame = initialization frame, frame_log frame f
#                 = video frame start_frame + f, no resume); R1/R1_DESIGN_DECISIONS.md D18, D20; F15 (.mov resume
#                 landed +10 frames late); R1-B instruction (Nathan, 2026-10-24)
"""
player_tracker_rb.video_io: sequential, fail-closed video reading for the R1 tracker.

Frame numbering (the recovered Pass-1 contract, VERIFIED_CODE + VERIFIED_OUTPUT):
  * The seed frame (video frame index `start_frame`) is the INITIALIZATION frame. It is read once and exposed as
    `reader.seed`, with relative frame 0. It is never a frame_log step.
  * The first logged step is the NEXT video frame: relative frame 1 = video frame start_frame + 1.
  * In general: relative frame f  <->  video frame index start_frame + f.

Time mapping (constant frame rate assumed and checked):
  * video_time_s = video_frame_index / fps   (absolute time in the source; this is frame_log v2.timestamp_s)
  * rel_time_s   = rel_frame / fps           (time since the seed; this is what the surviving event_adapter
                                              writes as snapshot timestamp: round(frame_number / fps, 4))
  * The container's own timestamp (CAP_PROP_POS_MSEC) is recorded per frame for diagnostics only. If it drifts
    from video_time_s by more than `timing_drift_tolerance_frames`, a timing-drift warning is counted.

Reading rules:
  * Reaching start_frame: `seek_mode` "decode_from_zero" (default) decodes and discards frames 0..start_frame-1
    with no container seek at all; "container_seek" performs ONE CAP_PROP_POS_FRAMES seek and verifies the
    reported position. After that, decoding is strictly sequential (`read()` only). No further seek, ever.
  * There is no checkpoint, no resume and no way to start in the middle of a run (F15). Nothing is written to disk.
  * Fail closed: an unreadable or missing source, a bad start frame, a hash mismatch, a frame-size change, a
    position mismatch, or a change to the source file during the run raises a VideoError subclass.
    The source file is stat-checked before every frame and fully re-hashed at close().

Frame-content fingerprint (see R1/R1B/FRAME_FINGERPRINT.md):
  * SHA-1 of a small greyscale thumbnail of each decoded frame. It exists ONLY to debug and verify frame alignment
    (e.g. that two runs, or two seek modes, decoded the same frame at the same index). It is not an appearance
    feature, carries no identity meaning, and must never be read by detection, appearance or state code.
"""
from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional

import cv2
import numpy as np

from .config import MS_PER_SECOND, TrackerConfig


# ---------------------------------------------------------------------------------------------
# Errors (all fail-closed)
# ---------------------------------------------------------------------------------------------
class VideoError(RuntimeError):
    """Base class: the video layer refuses to continue."""


class VideoNotFoundError(VideoError):
    pass


class VideoUnreadableError(VideoError):
    pass


class StartFrameOutOfRangeError(VideoError):
    pass


class SourceHashMismatchError(VideoError):
    pass


class SourceChangedError(VideoError):
    pass


class FrameAlignmentError(VideoError):
    pass


# ---------------------------------------------------------------------------------------------
# Hashing and fingerprints
# ---------------------------------------------------------------------------------------------
def sha256_file(path: str, chunk_bytes: int) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk_bytes), b""):
            h.update(block)
    return h.hexdigest()


_INTERPOLATIONS = {"INTER_AREA": cv2.INTER_AREA, "INTER_LINEAR": cv2.INTER_LINEAR,
                   "INTER_NEAREST": cv2.INTER_NEAREST}


def frame_fingerprint(image: np.ndarray, cfg: TrackerConfig) -> str:
    """SHA-1 of the greyscale thumbnail (size `frame_thumbnail_size`, resize `fingerprint_interpolation`).
    Debugging/alignment only. Not an identity feature."""
    if image is None or image.ndim < 2:
        raise ValueError("frame_fingerprint needs a 2-D (grey) or 3-D (BGR) image")
    grey = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    w, h = (int(v) for v in cfg.get("frame_thumbnail_size"))
    thumb = cv2.resize(grey, (w, h), interpolation=_INTERPOLATIONS[cfg.get("fingerprint_interpolation")])
    thumb = np.ascontiguousarray(thumb, dtype=np.uint8)
    hdr = f"{w}x{h}:".encode("ascii")
    return hashlib.sha1(hdr + thumb.tobytes()).hexdigest()


def fingerprint_method(cfg: TrackerConfig) -> Dict[str, Any]:
    w, h = (int(v) for v in cfg.get("frame_thumbnail_size"))
    return {"algorithm": "sha1(b'WxH:' + uint8 greyscale thumbnail bytes, row-major)",
            "thumbnail_size": [w, h], "colour_conversion": "cv2.COLOR_BGR2GRAY",
            "interpolation": cfg.get("fingerprint_interpolation"), "cv2_version": cv2.__version__,
            "purpose": "frame-alignment debugging only; not an identity or appearance feature"}


# ---------------------------------------------------------------------------------------------
# Data records
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class VideoFrame:
    rel_frame: int               # 0 = seed (initialization) frame; frame_log frames are 1, 2, ...
    video_frame_index: int       # start_frame + rel_frame
    video_time_s: float          # video_frame_index / fps
    rel_time_s: float            # rel_frame / fps
    pos_msec_reported: Optional[float]   # container timestamp, diagnostics only
    thumb_sha1: str              # frame-content fingerprint (alignment debugging only)
    image: np.ndarray = field(repr=False, compare=False)


@dataclass
class VideoInfo:
    path: str
    abs_path: str
    sha256: str
    size_bytes: int
    mtime_ns: int
    fps: float
    frame_width: int
    frame_height: int
    frame_count_reported: Optional[int]
    backend: str
    cv2_version: str
    seek_mode: str
    start_frame: int

    def as_record(self) -> Dict[str, Any]:
        return {"path": self.path, "abs_path": self.abs_path, "sha256": self.sha256, "size_bytes": self.size_bytes,
                "fps": self.fps, "frame_width": self.frame_width, "frame_height": self.frame_height,
                "frame_count_reported": self.frame_count_reported, "backend": self.backend,
                "cv2_version": self.cv2_version, "seek_mode": self.seek_mode, "start_frame": self.start_frame}


# ---------------------------------------------------------------------------------------------
# The reader
# ---------------------------------------------------------------------------------------------
class SequentialVideoReader:
    """Open a video, position at start_frame once, then decode strictly sequentially.

        with SequentialVideoReader(path, start_frame, cfg, expected_sha256=...) as r:
            seed = r.seed                 # relative frame 0 (initialization frame)
            for fr in r:                  # relative frames 1, 2, ... (frame_log steps)
                ...
        # leaving the block re-hashes the source and raises SourceChangedError if it changed.

    `capture_factory` exists only so tests can observe the calls made on the capture object.
    """

    def __init__(self, path: str, start_frame: int, cfg: TrackerConfig, expected_sha256: Optional[str] = None,
                 capture_factory: Callable[[str], Any] = cv2.VideoCapture):
        if isinstance(start_frame, bool) or not isinstance(start_frame, int):
            raise StartFrameOutOfRangeError(f"start_frame must be an int, got {start_frame!r}")
        if start_frame < 0:
            raise StartFrameOutOfRangeError(f"start_frame must be >= 0, got {start_frame}")
        if cfg.get("resume_allowed"):
            raise VideoError("resume is not supported")  # unreachable: config only allows False
        self._cfg = cfg
        self._path = path
        self._start = start_frame
        self._expected = expected_sha256
        self._factory = capture_factory
        self._cap = None
        self._closed = False
        self._iterating = False
        self._next_rel = 1
        self._seek_calls = 0
        self.info: Optional[VideoInfo] = None
        self.seed: Optional[VideoFrame] = None
        self.frames_decoded_after_seed = 0
        self.timing_drift_warnings = 0
        self.timing_drift_max_frames = 0.0
        self.position_reports_available = True
        self.eof_reached = False
        self.source_verified_at_close: Optional[bool] = None
        self._open()

    # ---- context manager ------------------------------------------------------------------------
    def __enter__(self) -> "SequentialVideoReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close(verify=exc_type is None)

    # ---- opening --------------------------------------------------------------------------------
    def _stat(self):
        try:
            st = os.stat(self._path)
        except OSError as e:
            raise SourceChangedError(f"source video no longer accessible: {e}") from e
        return st.st_size, st.st_mtime_ns

    def _open(self) -> None:
        path = self._path
        if not isinstance(path, str) or not path:
            raise VideoNotFoundError(f"invalid video path {path!r}")
        if not os.path.isfile(path):
            raise VideoNotFoundError(f"video not found: {path}")
        size0, mtime0 = self._stat()
        digest = sha256_file(path, self._cfg.get("hash_chunk_bytes"))
        if self._expected is not None and digest != self._expected:
            raise SourceHashMismatchError(f"source sha256 {digest} != expected {self._expected}")
        if (size0, mtime0) != self._stat():
            raise SourceChangedError("source video changed while it was being hashed")
        cap = self._factory(path)
        if cap is None or not cap.isOpened():
            raise VideoUnreadableError(f"OpenCV cannot open video: {path}")
        self._cap = cap
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not (math.isfinite(fps) and fps > 0):
            self._release()
            raise VideoUnreadableError(f"video reports no usable frame rate (fps={fps!r}): {path}")
        n_rep = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_count = int(round(n_rep)) if (math.isfinite(n_rep) and n_rep > 0) else None
        if frame_count is not None and self._start >= frame_count:
            self._release()
            raise StartFrameOutOfRangeError(
                f"start_frame {self._start} outside the video (reported frame count {frame_count})")
        try:
            backend = str(cap.getBackendName())
        except Exception:  # noqa: BLE001 - backend name is informational
            backend = "UNKNOWN"
        seek_mode = self._cfg.get("seek_mode")
        self._position(seek_mode)
        ok, img = cap.read()
        if not ok or img is None:
            self._release()
            if self._start == 0:
                raise VideoUnreadableError(f"could not decode the first frame: {path}")
            raise StartFrameOutOfRangeError(f"start_frame {self._start} could not be decoded (end of stream)")
        self._check_position_after_read(self._start)
        h_px, w_px = int(img.shape[0]), int(img.shape[1])
        self.info = VideoInfo(path=path, abs_path=os.path.abspath(path), sha256=digest, size_bytes=size0,
                              mtime_ns=mtime0, fps=fps, frame_width=w_px, frame_height=h_px,
                              frame_count_reported=frame_count, backend=backend, cv2_version=cv2.__version__,
                              seek_mode=seek_mode, start_frame=self._start)
        self.seed = self._make_frame(0, img)

    def _position(self, seek_mode: str) -> None:
        cap = self._cap
        if self._start == 0:
            return
        if seek_mode == "decode_from_zero":
            for i in range(self._start):
                ok = cap.grab()
                if not ok:
                    self._release()
                    raise StartFrameOutOfRangeError(
                        f"start_frame {self._start} outside the video (stream ended at frame {i})")
            return
        if seek_mode == "container_seek":
            self._seek_calls += 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(self._start))
            pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            if math.isfinite(pos) and pos >= 0 and int(round(pos)) != self._start:
                self._release()
                raise FrameAlignmentError(f"container seek to {self._start} landed at {pos}")
            return
        raise VideoError(f"unknown seek_mode {seek_mode!r}")

    def _check_position_after_read(self, video_index: int) -> None:
        pos = self._cap.get(cv2.CAP_PROP_POS_FRAMES)
        if not (math.isfinite(pos) and pos >= 0):
            self.position_reports_available = False
            return
        if int(round(pos)) != video_index + 1:
            raise FrameAlignmentError(
                f"decoder position {pos} after reading video frame {video_index} (expected {video_index + 1})")

    def _make_frame(self, rel: int, img: np.ndarray) -> VideoFrame:
        fps = self.info.fps
        vid = self._start + rel
        vt = vid / fps
        msec = self._cap.get(cv2.CAP_PROP_POS_MSEC)
        pos_msec = float(msec) if math.isfinite(msec) and msec >= 0 else None
        if pos_msec is not None:
            drift_frames = abs(pos_msec / MS_PER_SECOND - vt) * fps
            self.timing_drift_max_frames = max(self.timing_drift_max_frames, drift_frames)
            if drift_frames > self._cfg.get("timing_drift_tolerance_frames"):
                self.timing_drift_warnings += 1
        return VideoFrame(rel_frame=rel, video_frame_index=vid, video_time_s=vt, rel_time_s=rel / fps,
                          pos_msec_reported=pos_msec, thumb_sha1=frame_fingerprint(img, self._cfg), image=img)

    # ---- sequential iteration -------------------------------------------------------------------
    def __iter__(self) -> Iterator[VideoFrame]:
        if self._iterating or self._next_rel != 1:
            raise VideoError("frames can be iterated only once per reader (no rewind, no resume)")
        self._iterating = True
        return self._gen()

    def _gen(self) -> Iterator[VideoFrame]:
        cap_frames = self._cfg.get("max_frames")
        while not self._closed:
            if cap_frames is not None and self._next_rel > cap_frames:
                return
            if self._stat() != (self.info.size_bytes, self.info.mtime_ns):
                raise SourceChangedError("source video changed during the run")
            ok, img = self._cap.read()
            if not ok or img is None:
                self.eof_reached = True
                return
            rel = self._next_rel
            vid = self._start + rel
            self._check_position_after_read(vid)
            if (int(img.shape[0]), int(img.shape[1])) != (self.info.frame_height, self.info.frame_width):
                raise FrameAlignmentError(f"frame size changed at video frame {vid}: {img.shape[:2]}")
            fr = self._make_frame(rel, img)
            self._next_rel += 1
            self.frames_decoded_after_seed += 1
            yield fr

    # ---- closing --------------------------------------------------------------------------------
    def _release(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            finally:
                self._cap = None

    def close(self, verify: bool = True) -> None:
        """Release the decoder; with verify=True re-hash the source and fail closed if it changed."""
        if self._closed:
            return
        self._closed = True
        self._release()
        if verify and self.info is not None:
            if self._stat() != (self.info.size_bytes, self.info.mtime_ns):
                self.source_verified_at_close = False
                raise SourceChangedError("source video changed during the run (size/mtime)")
            if sha256_file(self._path, self._cfg.get("hash_chunk_bytes")) != self.info.sha256:
                self.source_verified_at_close = False
                raise SourceChangedError("source video content changed during the run (sha256)")
            self.source_verified_at_close = True

    @property
    def seek_calls(self) -> int:
        return self._seek_calls

    def alignment_summary(self) -> Dict[str, Any]:
        """Frame-alignment facts for the provenance sidecar (no image data)."""
        last_rel = self._next_rel - 1
        return {
            "frame_index_rule": "video_frame_index = start_frame + frame; frame 0 = seed (initialization) frame",
            "seed_video_frame_index": self._start,
            "seed_thumb_sha1": self.seed.thumb_sha1 if self.seed else None,
            "first_logged_frame": 1,
            "processed_frames": self.frames_decoded_after_seed,
            "last_logged_frame": last_rel if last_rel >= 1 else None,
            "last_video_frame_index": self._start + last_rel if last_rel >= 1 else None,
            "eof_reached": self.eof_reached,
            "time_rule": "video_time_s = video_frame_index / fps; rel_time_s = frame / fps",
            "timing_drift_warnings": self.timing_drift_warnings,
            "timing_drift_max_frames": self.timing_drift_max_frames,
            "position_reports_available": self.position_reports_available,
            "container_seek_calls": self._seek_calls,
        }


def thumbs_digest(thumbs: List[str]) -> str:
    """One SHA-256 over the ordered list of per-frame fingerprints (compact alignment summary)."""
    d = ThumbsDigest()
    for t in thumbs:
        d.update(t)
    return d.hexdigest()


class ThumbsDigest:
    """thumbs_digest computed incrementally, so a long run need not keep every fingerprint in memory."""

    def __init__(self) -> None:
        self._h = hashlib.sha256()
        self.count = 0

    def update(self, thumb: str) -> None:
        self._h.update(thumb.encode("ascii"))
        self._h.update(b"\n")
        self.count += 1

    def hexdigest(self) -> str:
        return self._h.hexdigest()

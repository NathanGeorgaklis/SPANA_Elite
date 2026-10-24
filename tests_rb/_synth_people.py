# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R1-E instruction (synthetic crossings, merges, gaps, pan/zoom, cuts); test helper only
"""Test helper: neutral synthetic people (TARGET, PERSON_A, PERSON_B, ...) moving in image coordinates, a
ground-truth record of which person produced every detection, and a runner that feeds a Population.

The helper owns the ground truth; the population never sees it. Tests use it to prove properties such as
"no tracklet ever holds detections of two different people" (no silent swap)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from player_tracker_rb.camera import FIRST_FRAME, CameraMotion, identity_motion, motion_from_matrix
from player_tracker_rb.config import TrackerConfig
from player_tracker_rb.detector import PersonCandidate
from player_tracker_rb.population import FrameResult, Population

W, H = 1280, 720


def cand(box: Sequence[float], score: float = 0.9, edge: bool = False) -> PersonCandidate:
    return PersonCandidate(tuple(float(v) for v in box), score, edge, "full_frame")


def still(width: int = W, height: int = H) -> CameraMotion:
    return motion_from_matrix([[1, 0, 0], [0, 1, 0]], width, height)


def pan(dx: float, dy: float = 0.0, width: int = W, height: int = H) -> CameraMotion:
    return motion_from_matrix([[1, 0, dx], [0, 1, dy]], width, height)


def zoom(s: float, width: int = W, height: int = H) -> CameraMotion:
    cx, cy = width / 2, height / 2
    return motion_from_matrix([[s, 0, cx - s * cx], [0, s, cy - s * cy]], width, height)


@dataclass
class Frame:
    """One synthetic frame: the detections given to the population and who produced each one."""
    dets: List[PersonCandidate]
    owners: List[Tuple[str, ...]]          # ground-truth people behind each detection (2+ for a merged box)
    camera: Optional[CameraMotion] = None  # None = still (or FIRST_FRAME on frame 0)


@dataclass
class Run:
    results: List[FrameResult]
    frames: List[Frame]
    population: Population
    fps: float

    def tracklet_owners(self) -> Dict[str, set]:
        """tracklet id -> set of ground-truth people whose detections it was observed on."""
        out: Dict[str, set] = {}
        for r, fr in zip(self.results, self.frames):
            for v in r.tracklets:
                if v.observed:
                    out.setdefault(v.tracklet_id, set()).update(fr.owners[v.detection_index])
        return out

    def records_json(self) -> str:
        return json.dumps([r.as_record() for r in self.results], sort_keys=True)

    def closed(self) -> Dict[str, str]:
        return {c.tracklet_id: c.reason for r in self.results for c in r.closed}


def run(frames: Sequence[Frame], fps: float = 30.0, cfg: Optional[TrackerConfig] = None, width: int = W,
        height: int = H, hook: Optional[Callable[[int, Population, FrameResult], None]] = None) -> Run:
    cfg = cfg or TrackerConfig.default()
    pop = Population(cfg, width, height)
    results = []
    for k, fr in enumerate(frames):
        cam = fr.camera if fr.camera is not None else (identity_motion(FIRST_FRAME) if k == 0
                                                       else still(width, height))
        res = pop.update(k, k / fps, fr.dets, cam)
        results.append(res)
        if hook:
            hook(k, pop, res)
    return Run(results, list(frames), pop, fps)


def box_at(cx: float, cy: float, w: float, h: float) -> Tuple[float, float, float, float]:
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def crossing(n_frames: int, fps: float, *, speed_h_per_s: float, dy_h: float, merge_iou: Optional[float],
             hide_back: bool = False, w: float = 32.0, h: float = 80.0, scale: float = 1.0,
             names=("PERSON_A", "PERSON_B")) -> List[Frame]:
    """Two same-size people walking towards each other along x and passing. Their boxes can be (merge_iou set)
    merged into one detection while their IoU exceeds merge_iou, or (hide_back) the rear one can be undetected
    while hidden. Coordinates scale with `scale` (for resolution-independence tests)."""
    w, h = w * scale, h * scale
    cy = 360.0 * scale
    v = speed_h_per_s * h / fps
    x0a, x0b = 400.0 * scale, 400.0 * scale + v * n_frames
    frames = []
    for k in range(n_frames):
        a = box_at(x0a + v * k, cy, w, h)
        b = box_at(x0b - v * k, cy + dy_h * h, w, h)
        ov = iou(a, b)
        if merge_iou is not None and ov >= merge_iou:
            m = (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
            frames.append(Frame([cand(m)], [tuple(names)]))
        elif hide_back and ov > 0:
            # PERSON_B is behind: only PERSON_A is detected
            frames.append(Frame([cand(a)], [(names[0],)]))
        else:
            frames.append(Frame([cand(a), cand(b)], [(names[0],), (names[1],)]))
    return frames


def walker(n_frames: int, fps: float, *, x0: float = 300.0, y0: float = 360.0, vx_h_per_s: float = 2.0,
           w: float = 32.0, h: float = 80.0, name: str = "TARGET", drop: Sequence[int] = ()) -> List[Frame]:
    frames = []
    for k in range(n_frames):
        b = box_at(x0 + vx_h_per_s * h / fps * k, y0, w, h)
        frames.append(Frame([] if k in drop else [cand(b)], [] if k in drop else [(name,)]))
    return frames


def combine(*seqs: Sequence[Frame]) -> List[Frame]:
    out = []
    for parts in zip(*seqs):
        dets, owners = [], []
        cam = None
        for p in parts:
            dets += p.dets
            owners += p.owners
            cam = cam or p.camera
        out.append(Frame(dets, owners, cam))
    return out


def as_array(box) -> np.ndarray:
    return np.asarray(box, dtype=np.float64)

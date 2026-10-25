# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R1-G instruction (Nathan, 2026-10-25: synthetic full-pipeline tests); test helper only
"""Test helper: synthetic END-TO-END clips for the integrated pipeline (player_tracker_rb.run_rb.run).

A clip is a real video file (FFV1, lossless, so rendered appearance survives encoding exactly) of people drawn by
_synth_scene on a textured pitch that the REAL camera estimator can measure:
  * the pitch has lines and blobs (background features); a camera pan shifts the whole pitch and every person;
  * each "shot" has its own pitch geometry; consecutive shots alternate brightness, so a shot change is a real hard
    cut (histogram change + flow failure) for the camera estimator.
Everything after the video file is the production pipeline: video_io -> detector backend -> camera -> population
-> target -> output records. The only test seams are the backends the config names "scripted":
  * a ScriptedDetector whose script is the scene's ideal detector output for each frame (one call per frame,
    starting at the seed frame), and
  * the SignatureEmbedding (identity signatures rendered into each person; see _synth_scene).
Ground truth (which person owns each detection) stays in this helper and is used only to score the output.
"""
from __future__ import annotations

import functools
import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from player_tracker_rb import run_rb as R
from player_tracker_rb.config import TrackerConfig
from player_tracker_rb.detector import ScriptedDetector
from tests_rb._synth_scene import Detection, Person, SceneFrame, SignatureEmbedding, detect, iou, render

W, H = 640, 360
FPS = 30.0
PW, PH = 28, 70            # person box size in px
SCRIPTED = {"detector_backend": "scripted", "appearance_backend": "scripted"}
PAN_MARGIN = 2000          # widest pan supported (px)


def pitch(shot: int, offset_x: int, width: int = W, height: int = H) -> np.ndarray:
    """Background for one frame of `shot`, seen through a camera panned by offset_x px."""
    if not 0 <= offset_x <= PAN_MARGIN:
        raise ValueError("pan offset outside the rendered pitch")
    return np.ascontiguousarray(_pitch_full(shot, width, height)[:, offset_x:offset_x + width])


@functools.lru_cache(maxsize=8)
def _pitch_full(shot: int, width: int, height: int) -> np.ndarray:
    rng = np.random.default_rng(1000 + shot)
    g = np.empty((height, width + PAN_MARGIN, 3), np.uint8)
    g[:] = (40, 130, 40) if shot % 2 == 0 else (30, 95, 35)
    sx, sy = 53 + 17 * (shot % 5), 47 + 11 * (shot % 5)
    for x in range(int(rng.integers(0, 40)), width + PAN_MARGIN, sx):
        cv2.line(g, (x, 0), (x + 30 - 25 * (shot % 2), height), (220, 230, 220), 2)
    for y in range(int(rng.integers(5, 30)), height, sy):
        cv2.line(g, (0, y), (width + PAN_MARGIN, y + 9 - 14 * (shot % 2)), (200, 210, 200), 2)
    for _ in range(60 + PAN_MARGIN // 10):
        x, y = rng.integers(0, width + PAN_MARGIN), rng.integers(0, height)
        cv2.circle(g, (int(x), int(y)), int(rng.integers(3, 9)),
                   (int(rng.integers(0, 80)), int(rng.integers(100, 200)), int(rng.integers(0, 80))), -1)
    g.setflags(write=False)
    return g


@dataclass
class ClipFrame:
    scene: SceneFrame          # people boxes in IMAGE coordinates (already panned)
    shot: int = 0
    pan_x: int = 0


@dataclass
class E2E:
    doc: Dict[str, Any]
    summary: Dict[str, Any]
    dets: List[List[Detection]]              # ground-truth detections per relative frame (0 = seed)
    owners: List[List[Tuple[str, ...]]]      # owner of each candidate the detector returned, per relative frame
    decision_log: List[Dict[str, Any]]
    paths: Dict[str, str]
    sizes: List[Tuple[int, Dict[str, int]]]

    # -- per frame accessors (f = frame_log frame, 1..N) ------------------------------------------------
    def target(self, f: int) -> Dict[str, Any]:
        return [t for t in self.doc["frame_log"][f - 1]["tracks"] if t["selected_player"]][0]

    def population(self, f: int) -> List[Dict[str, Any]]:
        return [t for t in self.doc["frame_log"][f - 1]["tracks"] if not t["selected_player"]]

    @property
    def n(self) -> int:
        return len(self.doc["frame_log"])

    def target_det_box(self, f: int, name: str = "TARGET") -> Optional[Tuple[float, ...]]:
        own = [d.box for d in self.dets[f] if d.owners == (name,)]
        return own[0] if own else None

    # -- scores ------------------------------------------------------------------------------------
    def swaps(self, name: str = "TARGET") -> List[int]:
        """Frames where the target record is trusted (ACTIVE) but its box is not the target's own detection."""
        out = []
        for f in range(1, self.n + 1):
            t = self.target(f)
            if t["state"] == "ACTIVE":
                own = self.target_det_box(f, name)
                if own is None or iou(own, t["observed_position_box"]) < 0.95:
                    out.append(f)
        return out

    def trusted_frames(self) -> List[int]:
        return [f for f in range(1, self.n + 1) if self.target(f)["v2"]["identity_trusted"]]

    def target_duplicated(self, name: str = "TARGET") -> List[int]:
        """Trusted frames where a population record sits on the target's own detection (target written twice)."""
        out = []
        for f in self.trusted_frames():
            own = self.target_det_box(f, name)
            if own is not None and any(iou(own, p["observed_position_box"]) > 0.5 for p in self.population(f)
                                       if p["v2"]["detected_this_frame"]):
                out.append(f)
        return out

    def body_written_twice(self, name: str = "TARGET") -> List[Tuple[int, str, str]]:
        """Rule 28 in EVERY frame: frames where T001's reported box is on the target's own body and a population
        record (detected or coasting) is also on that body."""
        out = []
        for f in range(1, self.n + 1):
            own = self.target_det_box(f, name)
            if own is None:
                continue
            t = self.target(f)
            if iou(own, t["observed_position_box"]) <= 0.5:
                continue
            for p in self.population(f):
                if iou(own, p["observed_position_box"]) > 0.5:
                    out.append((f, p["track_id"], t["v2"]["rb_state"]))
        return out

    def accepted_bank_frames(self) -> List[int]:
        return [line["frame"] for line in self.decision_log
                if ((line["target"].get("decision") or {}).get("bank") or {}).get("status") == "ACCEPTED"]

    def bank_violations(self, name: str = "TARGET") -> List[int]:
        """Frames whose accepted target bank sample was not a clean crop of the target alone: the sampled
        detection must be owned only by the target and no other person's TRUE box may touch the target's."""
        out = []
        for line in self.decision_log:
            f = line["frame"]
            t = line["target"]
            bank = (t.get("decision") or {}).get("bank") or {}
            if bank.get("status") != "ACCEPTED":
                continue
            di = t["detection_index"]
            if di is None or self.owners[f][di] != (name,):
                out.append(f)
                continue
            truth = self.scene_boxes[f]
            mine = truth[name]
            if any(iou(mine, b) > 0 for k, b in truth.items() if k != name):
                out.append(f)
        return out


def write_clip(path: str, frames: Sequence[ClipFrame], people: Dict[str, Person], fps: float = FPS,
               width: int = W, height: int = H) -> str:
    wr = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"FFV1"), fps, (width, height))
    if not wr.isOpened():
        raise RuntimeError("cannot open an FFV1 writer")
    for cf in frames:
        wr.write(render(cf.scene, people, width, height, background=pitch(cf.shot, cf.pan_x, width, height)))
    wr.release()
    return path


def detector_script(frames: Sequence[ClipFrame], start_frame: int, visible_min: float, width: int = W,
                    height: int = H) -> Tuple[List[List[List[float]]], List[List[Detection]]]:
    script, dets = [], []
    for cf in frames[start_frame:]:
        ds = detect(cf.scene, visible_min, width, height)
        dets.append(ds)
        script.append([[*d.box, 0.9] for d in ds])
    return script, dets


def run_clip(frames: Sequence[ClipFrame], people: Dict[str, Person], work_dir: str, *, start_frame: int = 0,
             seed_name: str = "TARGET", overrides: Optional[Dict[str, Any]] = None, visible_min: float = 0.35,
             debug: bool = True, name: str = "clip", fps: float = FPS,
             on_frame: Optional[Callable[[int, Dict[str, int]], None]] = None,
             video_path: Optional[str] = None, pose_gain: float = 0.0) -> E2E:
    """Write the clip, run the production pipeline on it, and load everything it wrote."""
    os.makedirs(work_dir, exist_ok=True)
    video = video_path or write_clip(os.path.join(work_dir, name + ".avi"), frames, people, fps)
    out_dir = os.path.join(work_dir, "out_" + name)
    os.makedirs(out_dir, exist_ok=False)
    ov = dict(SCRIPTED, **(overrides or {}))
    cfg = TrackerConfig.with_overrides(ov)
    script, dets = detector_script(frames, start_frame, visible_min)
    detector = ScriptedDetector(script, cfg)
    seed_box = list(frames[start_frame].scene.boxes[seed_name])
    sizes: List[Tuple[int, Dict[str, int]]] = []
    summary = R.run(video, start_frame, seed_box, out_dir=out_dir, debug=debug, config_overrides=ov,
                    detector=detector, embedder=SignatureEmbedding(people, pose_gain),
                    on_frame=(lambda f, s: sizes.append((f, s))) if on_frame is None else on_frame)
    paths = R.output_paths(out_dir, video)
    with open(paths["results"], encoding="ascii") as fh:
        doc = json.load(fh)
    log = []
    if debug:
        with open(paths["decision_log"], encoding="ascii") as fh:
            log = [json.loads(line) for line in fh]
    # owners of the candidates the detector actually returned (same NMS/clip path as the pipeline saw)
    owners = []
    replay = ScriptedDetector(script, cfg)
    blank = np.zeros((H, W, 3), np.uint8)
    for ds in dets:
        out = []
        for c in replay.detect(blank):
            best = max(ds, key=lambda d: iou(d.box, c.bbox)) if ds else None
            out.append(best.owners if best is not None and iou(best.box, c.bbox) > 0.5 else ())
        owners.append(out)
    e = E2E(doc, summary, dets, owners, log, paths, sizes)
    e.scene_boxes = [dict(cf.scene.boxes) for cf in frames[start_frame:]]
    return e


# ---- scene builders (image coordinates, frames at FPS) --------------------------------------------
def box(cx: float, cy: float) -> Tuple[float, float, float, float]:
    return (cx - PW / 2, cy - PH / 2, cx + PW / 2, cy + PH / 2)


def walk(n: int, start: Tuple[float, float], v: Tuple[float, float]) -> List[Tuple[float, float, float, float]]:
    return [box(start[0] + v[0] * k, start[1] + v[1] * k) for k in range(n)]


def gait_walk(n: int, start: Tuple[float, float], v: Tuple[float, float]) -> List[Tuple[float, float, float, float]]:
    """A walk whose box width cycles with the stride (pose variation for the pose-aware synthetic embedding)."""
    out = []
    for k in range(n):
        cx, cy = start[0] + v[0] * k, start[1] + v[1] * k
        w = PW - 8 + 8 * abs((k // 4) % 5 - 2)
        out.append((cx - w / 2, cy - PH / 2, cx + w / 2, cy + PH / 2))
    return out


def clip(positions: Dict[str, Sequence[Optional[Tuple[float, float, float, float]]]], *,
         shots: Optional[Sequence[int]] = None, pans: Optional[Sequence[int]] = None,
         drop: Optional[Callable[[int], Sequence[str]]] = None,
         merge: Optional[Callable[[int], Sequence[Tuple[str, str]]]] = None) -> List[ClipFrame]:
    """Frames from per-person box sequences (None = not in the scene). Pans shift every box by -pan_x."""
    n = max(len(v) for v in positions.values())
    out = []
    for k in range(n):
        px = pans[k] if pans else 0
        boxes = {}
        for name, seq in positions.items():
            b = seq[k] if k < len(seq) else None
            if b is not None:
                boxes[name] = (b[0] - px, b[1], b[2] - px, b[3])
        sf = SceneFrame(boxes, drop=list(drop(k)) if drop else [], merge=list(merge(k)) if merge else [])
        out.append(ClipFrame(sf, shots[k] if shots else 0, px))
    return out

# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R1-F instruction (synthetic identity scenarios: same-kit and opponent crossings, occlusion, merges,
#                 exits, gaps, reacquisition, cuts, ambiguity, provisional competitors); test helper only
"""Test helper: rendered synthetic scenes with ground truth for the R1-F target identity layer.

Appearance is fully controlled, so identity evidence is real and measurable (not scripted per frame):
  * every person is drawn as a box filled with a KIT colour (what the LAB cue sees: same kit = same colour), with a
    small head block (top HEAD_FRAC of the box, above the LAB bands) in a unique SIGNATURE colour;
  * SignatureEmbedding (a test EmbeddingBackend) counts the signature pixels of each person in a crop and returns the
    count-weighted mix of their identity vectors (unit length). A crop showing one person's head gives that person's
    vector; a merged or occluded crop gives a blend; no head gives a fixed background vector;
  * identity vectors are built with a chosen cosine to the TARGET vector, so "same kit, cosine 0.75" reproduces the
    hard case that colour cannot separate and the embedding separates only partly.
The population and the target tracker see only images and detections; the helper keeps the ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from player_tracker_rb import population as P
from player_tracker_rb import target as T
from player_tracker_rb.appearance import AppearanceExtractor, EmbeddingBackend, l2_normalise
from player_tracker_rb.camera import FIRST_FRAME, CameraMotion, identity_motion, motion_from_matrix
from player_tracker_rb.config import TrackerConfig
from player_tracker_rb.detector import PersonCandidate

W, H = 1280, 720
FPS = 30.0
DIM = 16
HEAD_FRAC = 0.12
BACKGROUND = (40, 120, 40)


def box_at(cx: float, cy: float, w: float, h: float) -> Tuple[float, float, float, float]:
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def covered(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / ((a[2] - a[0]) * (a[3] - a[1]))


@dataclass
class Person:
    name: str
    kit: Tuple[int, int, int]
    sig: Tuple[int, int, int]
    vec: np.ndarray = field(repr=False)


def make_people(spec: Sequence[Tuple[str, Tuple[int, int, int], float]]) -> Dict[str, Person]:
    """spec: (name, kit BGR, cosine of its identity vector to TARGET's). The first entry must be TARGET (cos 1).
    Signature colours are unique per person; identity vectors share only the TARGET component."""
    out = {}
    for k, (name, kit, cos) in enumerate(spec):
        v = np.zeros(DIM)
        v[0] = cos
        if k:
            v[k] = np.sqrt(max(0.0, 1.0 - cos * cos))
        sig = (200 - 7 * k, 30 + 11 * k, 220 - 13 * k)
        out[name] = Person(name, kit, sig, l2_normalise(v))
    return out


class SignatureEmbedding(EmbeddingBackend):
    def __init__(self, people: Dict[str, Person]):
        self.people = people
        bg = np.zeros(DIM)
        bg[DIM - 1] = 1.0
        self.bg = bg

    def embed(self, crop: np.ndarray) -> np.ndarray:
        v = np.zeros(DIM)
        for p in self.people.values():
            n = int(np.all(crop == np.asarray(p.sig, np.uint8), axis=2).sum())
            v += n * p.vec
        if not np.any(v):
            return self.bg.copy()
        return l2_normalise(v)

    def provenance(self):
        return {"backend": "test_signature"}


@dataclass
class SceneFrame:
    boxes: Dict[str, Tuple[float, float, float, float]]   # person -> true box (absent: not in the scene)
    order: Sequence[str] = ()                             # draw order far -> near (default: by box bottom)
    drop: Sequence[str] = ()                              # people present but not detected
    merge: Sequence[Tuple[str, str]] = ()                 # pairs detected as ONE box
    camera: Optional[CameraMotion] = None


@dataclass
class Detection:
    box: Tuple[float, float, float, float]
    owners: Tuple[str, ...]


def render(fr: SceneFrame, people: Dict[str, Person], width: int = W, height: int = H) -> np.ndarray:
    img = np.empty((height, width, 3), np.uint8)
    img[:] = BACKGROUND
    order = list(fr.order) or sorted(fr.boxes, key=lambda n: fr.boxes[n][3])
    for name in order:
        if name not in fr.boxes:
            continue
        x1, y1, x2, y2 = (int(round(v)) for v in fr.boxes[name])
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        p = people[name]
        img[y1:y2, x1:x2] = p.kit
        bh = fr.boxes[name][3] - fr.boxes[name][1]
        hy2 = int(round(fr.boxes[name][1] + HEAD_FRAC * bh))
        bw = fr.boxes[name][2] - fr.boxes[name][0]
        hx1 = int(round(fr.boxes[name][0] + bw / 4))
        hx2 = int(round(fr.boxes[name][2] - bw / 4))
        hy1 = max(0, y1)
        img[hy1:max(hy1, min(hy2, height)), max(0, hx1):min(width, hx2)] = p.sig
    return img


def detect(fr: SceneFrame, visible_min: float = 0.35, width: int = W, height: int = H) -> List[Detection]:
    """Ideal detector: one full box per sufficiently visible person (not dropped); merged pairs give one union
    box. Visibility = 1 - fraction covered by nearer people (drawn later)."""
    order = list(fr.order) or sorted(fr.boxes, key=lambda n: fr.boxes[n][3])
    merged_names = {n for pair in fr.merge for n in pair}
    dets = []
    for a, b in fr.merge:
        if a in fr.boxes and b in fr.boxes:
            ba, bb = fr.boxes[a], fr.boxes[b]
            dets.append(Detection((min(ba[0], bb[0]), min(ba[1], bb[1]), max(ba[2], bb[2]), max(ba[3], bb[3])),
                                  (a, b)))
    for k, name in enumerate(order):
        if name not in fr.boxes or name in fr.drop or name in merged_names:
            continue
        b = fr.boxes[name]
        inside = covered(b, (0, 0, width, height))
        if inside < 0.3:
            continue
        cov = max([covered(b, fr.boxes[m]) for m in order[k + 1:] if m in fr.boxes] or [0.0])
        if 1.0 - cov < visible_min:
            continue
        cb = (max(0.0, b[0]), max(0.0, b[1]), min(float(width), b[2]), min(float(height), b[3]))
        dets.append(Detection(cb, (name,)))
    return dets


@dataclass
class Result:
    frames: List[T.TargetFrame]
    dets: List[List[Detection]]
    pops: List[P.FrameResult]
    tracker: T.TargetTracker
    people: Dict[str, Person]

    def swaps(self, target: str = "TARGET") -> List[int]:
        """Frames where the target was TRACKING (trusted) on a detection that is not exactly the target."""
        out = []
        for k, (tf, ds) in enumerate(zip(self.frames, self.dets)):
            if tf.rb_state == T.TRACKING and ds[tf.detection_index].owners != (target,):
                out.append(k)
        return out

    def states(self) -> List[str]:
        return [f.rb_state for f in self.frames]

    def tracking_frames(self) -> List[int]:
        return [k for k, f in enumerate(self.frames) if f.rb_state == T.TRACKING]

    def bank_is_pure(self, target: str = "TARGET") -> bool:
        tv = self.people[target].vec
        return all(float(s.embedding @ tv) > 1.0 - 1e-9 for s in self.tracker.bank.samples)


def run(frames: Sequence[SceneFrame], people: Dict[str, Person], *, seed_name: str = "TARGET",
        cfg: Optional[TrackerConfig] = None, fps: float = FPS, width: int = W, height: int = H,
        visible_min: float = 0.35) -> Result:
    cfg = cfg or TrackerConfig.default()
    pop = P.Population(cfg, width, height)
    ext = AppearanceExtractor(cfg, SignatureEmbedding(people))
    tt = T.TargetTracker(cfg, pop, ext)
    out_frames, out_dets, out_pops = [], [], []
    for k, fr in enumerate(frames):
        img = render(fr, people, width, height)
        ds = detect(fr, visible_min, width, height)
        cands = [PersonCandidate(d.box, 0.9, d.box[0] <= 2 or d.box[2] >= width - 2, "synthetic") for d in ds]
        if k == 0:
            cam = identity_motion(FIRST_FRAME)
        else:
            cam = fr.camera if fr.camera is not None else motion_from_matrix([[1, 0, 0], [0, 1, 0]], width, height)
        res = pop.update(k, k / fps, cands, cam)
        tf = tt.seed(res, img, fr.boxes[seed_name]) if k == 0 else tt.update(res, img)
        out_frames.append(tf)
        out_dets.append(ds)
        out_pops.append(res)
    return Result(out_frames, out_dets, out_pops, tt, people)


# ---- scenario builders (people move in image coordinates; h = body height in px) ---------------------------
def linear(n: int, start: Tuple[float, float], v_px_per_frame: Tuple[float, float], w: float = 40, h: float = 100):
    return [box_at(start[0] + v_px_per_frame[0] * k, start[1] + v_px_per_frame[1] * k, w, h) for k in range(n)]


def still(n: int, at: Tuple[float, float], w: float = 40, h: float = 100):
    return [box_at(at[0], at[1], w, h) for _ in range(n)]

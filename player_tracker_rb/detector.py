# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R1/R1_DESIGN_DECISIONS.md D03, D17, D21, D22; R1/R1_GENERIC_TRACKER_REQUIREMENT.md (generic
#                 PERSON_CANDIDATE output, no role/team/colour); R1-C instruction (Nathan, 2026-10-24)
"""
player_tracker_rb.detector: the generic person-detector interface.

Every detector backend returns the same plain record for every person it sees: a `PersonCandidate`. A candidate
says only "a person-shaped object is here, with this detector score". It carries no identity, no team, no role,
no colour and no number. Which candidate (if any) is the selected target is decided later, by other layers.

Library-specific objects (ONNX sessions, raw tensors, model-specific result layouts) never leave a backend:
`detect()` returns a tuple of PersonCandidate made only of Python floats, ints, bools and strings.

This module also holds the backend-independent geometry used by every backend:
  * letterbox transform (image -> network input) and its exact inverse;
  * clipping to the image;
  * greedy non-maximum suppression in continuous pixel coordinates;
  * tile layout for the optional tiled mode (more network pixels per small person at a fixed model input).
"""
from __future__ import annotations

import abc
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import TrackerConfig

CANDIDATE_KIND = "PERSON_CANDIDATE"


class DetectorError(RuntimeError):
    """Base class for detector failures (fail closed)."""


class DetectorModelError(DetectorError):
    """Missing, unreadable, mismatched or incompatible model file."""


@dataclass(frozen=True)
class PersonCandidate:
    """One detected person. Generic: no identity, team, role, colour or number."""
    bbox: Tuple[float, float, float, float]    # [x1, y1, x2, y2] in source-image pixels, clipped to the image
    score: float                               # raw detector person score (0-1); NOT an identity confidence
    touches_image_edge: bool                   # box reaches the image border (person may be partly outside)
    detector_pass: str                         # "full_frame" or "tile_c{col}_r{row}" (diagnostics)
    kind: str = CANDIDATE_KIND

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    def as_record(self) -> Dict[str, Any]:
        return {"kind": self.kind, "bbox": list(self.bbox), "score": self.score,
                "touches_image_edge": self.touches_image_edge, "detector_pass": self.detector_pass}


class DetectorBackend(abc.ABC):
    """Interface every person detector implements."""

    @abc.abstractmethod
    def detect(self, image: np.ndarray) -> Tuple[PersonCandidate, ...]:
        """All people found in one BGR uint8 image, highest score first."""

    @abc.abstractmethod
    def provenance(self) -> Dict[str, Any]:
        """Backend name, model file, model sha256, licence status and settings (JSON-safe)."""

    def model_record(self) -> Optional[Dict[str, Any]]:
        """Model entry for the run provenance `models` list, or None if the backend uses no model file."""
        return None


# ---------------------------------------------------------------------------------------------
# Geometry shared by all backends
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Letterbox:
    """Image region -> square network input: scale by `scale`, place at the top-left, pad the rest.

    network = (image - origin) * scale   <=>   image = network / scale + origin
    """
    scale: float
    origin_x: float          # where the region starts in the source image (0 for a full-frame pass)
    origin_y: float
    region_w: int
    region_h: int
    input_size: int

    def to_image(self, boxes_net: np.ndarray) -> np.ndarray:
        out = np.asarray(boxes_net, dtype=np.float64).copy()
        out[:, 0::2] = out[:, 0::2] / self.scale + self.origin_x
        out[:, 1::2] = out[:, 1::2] / self.scale + self.origin_y
        return out

    def to_network(self, boxes_img: np.ndarray) -> np.ndarray:
        out = np.asarray(boxes_img, dtype=np.float64).copy()
        out[:, 0::2] = (out[:, 0::2] - self.origin_x) * self.scale
        out[:, 1::2] = (out[:, 1::2] - self.origin_y) * self.scale
        return out


def letterbox_for(region_w: int, region_h: int, input_size: int, origin_x: float = 0.0,
                  origin_y: float = 0.0) -> Letterbox:
    if region_w <= 0 or region_h <= 0:
        raise ValueError(f"empty region {region_w}x{region_h}")
    scale = min(input_size / region_h, input_size / region_w)
    return Letterbox(scale=scale, origin_x=float(origin_x), origin_y=float(origin_y), region_w=int(region_w),
                     region_h=int(region_h), input_size=int(input_size))


def resized_extent(lb: Letterbox) -> Tuple[int, int]:
    """(width, height) of the resized region inside the network input (the model's own int() truncation)."""
    return int(lb.region_w * lb.scale), int(lb.region_h * lb.scale)


def clip_boxes(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    out = np.asarray(boxes, dtype=np.float64).copy()
    out[:, 0::2] = np.clip(out[:, 0::2], 0.0, float(width))
    out[:, 1::2] = np.clip(out[:, 1::2], 0.0, float(height))
    return out


def box_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0.0, None) * np.clip(iy2 - iy1, 0.0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    return iou


def containment(inner: np.ndarray, outer: np.ndarray) -> np.ndarray:
    """Fraction of each inner box's area that lies inside each outer box (rows: inner, cols: outer)."""
    a = np.asarray(inner, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(outer, dtype=np.float64).reshape(-1, 4)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0.0, None) * np.clip(iy2 - iy1, 0.0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(area_a[:, None] > 0, inter / area_a[:, None], 0.0)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
    """Greedy NMS: keep the highest score, drop boxes with IoU > threshold, repeat. Ties broken by index
    (stable sort), so the result is deterministic."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if boxes.shape[0] == 0:
        return []
    order = np.argsort(-scores, kind="stable")
    keep: List[int] = []
    suppressed = np.zeros(boxes.shape[0], dtype=bool)
    iou = box_iou_matrix(boxes, boxes)
    for i in order:
        if suppressed[i]:
            continue
        keep.append(int(i))
        suppressed |= iou[i] > iou_threshold
    return keep


@dataclass(frozen=True)
class Tile:
    name: str
    x0: int
    y0: int
    x1: int
    y1: int


def tile_layout(width: int, height: int, cols: int, rows: int, overlap: float) -> List[Tile]:
    """Equal-size overlapping tiles covering the whole image (deterministic, row-major)."""
    if cols < 1 or rows < 1:
        raise ValueError("tile grid must be at least 1x1")
    tw = math.ceil(width / (cols - (cols - 1) * overlap)) if cols > 1 else width
    th = math.ceil(height / (rows - (rows - 1) * overlap)) if rows > 1 else height
    tw, th = min(tw, width), min(th, height)
    xs = [0] if cols == 1 else [round(c * (width - tw) / (cols - 1)) for c in range(cols)]
    ys = [0] if rows == 1 else [round(r * (height - th) / (rows - 1)) for r in range(rows)]
    return [Tile(f"tile_c{c}_r{r}", xs[c], ys[r], xs[c] + tw, ys[r] + th)
            for r in range(rows) for c in range(cols)]




# Cut-side flags for tile-pass boxes (bit mask).
CUT_LEFT = 1
CUT_TOP = CUT_LEFT << 1
CUT_RIGHT = CUT_TOP << 1
CUT_BOTTOM = CUT_RIGHT << 1


def _is_fragment_of(frag: np.ndarray, sides: int, whole: np.ndarray, cont: float, cont_thr: float,
                    span_thr: float) -> bool:
    """A cut box is a fragment of `whole` if it lies mostly inside it AND spans most of it across the cut."""
    if cont < cont_thr:
        return False
    fw, fh = frag[2] - frag[0], frag[3] - frag[1]
    ww, wh = whole[2] - whole[0], whole[3] - whole[1]
    if ww <= 0 or wh <= 0:
        return False
    if sides & (CUT_TOP | CUT_BOTTOM) and fw / ww >= span_thr:
        return True
    if sides & (CUT_LEFT | CUT_RIGHT) and fh / wh >= span_thr:
        return True
    return False


def seen_uncut_elsewhere(box: Sequence[float], sides: int, own: "Tile", tiles: Sequence["Tile"]) -> bool:
    """True if a box cut by an internal edge of tile `own` lies entirely inside another tile that extends past
    every cut side. That tile saw the same region without the cut, so the cut box adds nothing: whatever is there
    was seen whole by that tile (or by the full-frame pass). Tiles overlap, so this holds for every person shorter
    than the overlap who straddles a tile edge."""
    x1, y1, x2, y2 = box
    for t in tiles:
        if t == own or not (t.x0 <= x1 and t.y0 <= y1 and x2 <= t.x1 and y2 <= t.y1):
            continue
        if sides & CUT_LEFT and not t.x0 < own.x0:
            continue
        if sides & CUT_TOP and not t.y0 < own.y0:
            continue
        if sides & CUT_RIGHT and not t.x1 > own.x1:
            continue
        if sides & CUT_BOTTOM and not t.y1 > own.y1:
            continue
        return True
    return False


def merge_passes(boxes: np.ndarray, scores: np.ndarray, passes: Sequence[str], cut_sides: np.ndarray,
                 cfg: TrackerConfig) -> List[int]:
    """Merge the boxes of all passes. Returns kept indices.

    `cut_sides` is a bit mask per box (CUT_LEFT/TOP/RIGHT/BOTTOM): which internal tile edges the box touches.
    A box touching an internal tile edge may be a cut fragment of a person who continues into the next tile; it
    must never beat a whole box of the same person, whatever its score. But a partly hidden SECOND person behind
    someone also lies inside that someone's box, and must be kept. So:
      1. NMS over the whole (uncut) boxes;
      2. a cut box is dropped if it overlaps a kept whole box by IoU > detector_nms_iou, or if it is a fragment of
         one: at least detector_tile_cut_containment inside it AND spanning at least
         detector_tile_cut_min_span_ratio of it across the cut;
      3. the remaining cut boxes are NMS-ed among themselves and the same fragment rule is applied.
    Without tiling there are no cut boxes and this is plain NMS."""
    iou_thr = cfg.get("detector_nms_iou")
    cont_thr = cfg.get("detector_tile_cut_containment")
    span_thr = cfg.get("detector_tile_cut_min_span_ratio")
    is_cut = np.asarray(cut_sides) != 0
    whole = np.nonzero(~is_cut)[0]
    cut = np.nonzero(is_cut)[0]
    keep = [int(whole[k]) for k in nms(boxes[whole], scores[whole], iou_thr)] if whole.size else []
    if cut.size:
        survivors = []
        kb = boxes[keep] if keep else np.zeros((0, 4))
        iou = box_iou_matrix(boxes[cut], kb) if keep else np.zeros((cut.size, 0))
        cont = containment(boxes[cut], kb) if keep else np.zeros((cut.size, 0))
        for r, idx in enumerate(cut):
            if keep and np.any(iou[r] > iou_thr):
                continue
            if any(_is_fragment_of(boxes[idx], int(cut_sides[idx]), kb[c], cont[r, c], cont_thr, span_thr)
                   for c in range(len(keep))):
                continue
            survivors.append(int(idx))
        if survivors:
            sv = np.asarray(survivors)
            ck = [int(sv[k]) for k in nms(boxes[sv], scores[sv], iou_thr)]
            cb = boxes[ck]
            c2 = containment(cb, cb)
            for a, idx in enumerate(ck):
                if any(b != a and _is_fragment_of(boxes[idx], int(cut_sides[idx]), cb[b], c2[a, b], cont_thr,
                                                  span_thr) for b in range(len(ck))):
                    continue
                keep.append(idx)
    return keep


def finalize(boxes_img: np.ndarray, scores: np.ndarray, passes: Sequence[str], width: int, height: int,
             keep: Sequence[int], edge_margin_px: float) -> Tuple[PersonCandidate, ...]:
    """Clip kept boxes to the image, drop empty ones, and build PersonCandidate records (plain Python types)."""
    out: List[PersonCandidate] = []
    if len(keep) == 0:
        return tuple(out)
    clipped = clip_boxes(boxes_img[list(keep)], width, height)
    for row, idx in zip(clipped, keep):
        x1, y1, x2, y2 = (float(v) for v in row)
        if not (x2 > x1 and y2 > y1):
            continue
        m = float(edge_margin_px)
        edge = x1 <= m or y1 <= m or x2 >= float(width) - m or y2 >= float(height) - m
        out.append(PersonCandidate(bbox=(x1, y1, x2, y2), score=float(scores[idx]), touches_image_edge=bool(edge),
                                   detector_pass=str(passes[idx])))
    out.sort(key=lambda c: (-c.score, c.bbox))
    return tuple(out)


# ---------------------------------------------------------------------------------------------
# Scripted backend (tests and synthetic scenarios)
# ---------------------------------------------------------------------------------------------
class ScriptedDetector(DetectorBackend):
    """Returns pre-written boxes, one list per call, through the same NMS/clip path as a real backend.
    `script` is a list (one entry per detect() call) of lists of (x1, y1, x2, y2, score)."""

    def __init__(self, script: Sequence[Sequence[Sequence[float]]], cfg: TrackerConfig):
        self._script = [list(map(tuple, frame)) for frame in script]
        self._cfg = cfg
        self._calls = 0

    def detect(self, image: np.ndarray) -> Tuple[PersonCandidate, ...]:
        if self._calls >= len(self._script):
            raise DetectorError("scripted detector ran out of frames")
        rows = self._script[self._calls]
        self._calls += 1
        height, width = int(image.shape[0]), int(image.shape[1])
        if not rows:
            return ()
        arr = np.asarray(rows, dtype=np.float64)
        boxes, scores = arr[:, :4], arr[:, 4]
        mask = scores >= self._cfg.get("detector_min_score")
        idx = np.nonzero(mask)[0]
        boxes, scores = boxes[idx], scores[idx]
        passes = ["full_frame"] * len(idx)
        keep = nms(boxes, scores, self._cfg.get("detector_nms_iou"))
        return finalize(boxes, scores, passes, width, height, keep, self._cfg.get("detector_edge_touch_margin_px"))

    def provenance(self) -> Dict[str, Any]:
        return {"backend": "scripted", "model_file": None, "model_sha256": None, "model_status": None,
                "min_score": self._cfg.get("detector_min_score"), "nms_iou": self._cfg.get("detector_nms_iou")}

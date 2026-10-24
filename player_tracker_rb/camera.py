# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R1/R1_DESIGN_DECISIONS.md D13, D16 (global motion from background flow, hard-cut detection,
#                 cut invalidates position continuity); R1-E instruction (Nathan, 2026-10-24)
"""
player_tracker_rb.camera: generic global camera-motion estimation and hard-cut detection.

Nothing here knows about players, teams or the target. The estimator sees two consecutive frames and the boxes of
the people detected in them, and answers one question: how did the IMAGE move between the frames?

Method (per frame, D16):
  1. Grey image, resized to flow_work_width_px (all flow pixel parameters are at that working width, so they mean
     the same thing at any source resolution).
  2. Background features: goodFeaturesToTrack on the previous frame, with every person box (padded by
     flow_person_mask_pad_h x its own height) masked out, so moving players do not dominate the estimate.
  3. Pyramidal Lucas-Kanade flow to the current frame. Tracked points that land inside a current person box
     (padded) are discarded.
  4. A similarity transform (translation, uniform scale, rotation) by RANSAC (cv2.estimateAffinePartial2D).
  5. Bounds: a shift faster than camera_max_shift_frac_per_s, a zoom faster than camera_max_scale_ratio_per_s or a
     rotation faster than camera_max_rotation_rad_per_s is implausible, and the estimate is rejected.
  6. Hard cut: grey-histogram Bhattacharyya distance >= cut_hist_distance_min AND flow failure.

Status of every frame:
  FIRST_FRAME  no previous frame (no continuity to break);
  ESTIMATED    a plausible transform was measured (matrix maps previous-frame pixels to current-frame pixels);
  UNKNOWN      the image motion could not be measured (too few features, flow failure without a histogram change,
               or an implausible estimate). The matrix is the identity and is NOT a measurement;
  CUT          a hard cut: histogram change and flow failure together.

`continuity_broken` is True for CUT, and for UNKNOWN when camera_unknown_breaks_continuity is True (the default:
fail closed, because a cut between two similar-looking shots looks exactly like a flow failure).

Measured quantities are reported separately from decisions: n_features, n_tracked, inlier_fraction and
hist_distance are always recorded, `reasons` names every failed test, and a fitted transform that was rejected is
kept as `measured_matrix` (audit only, never applied).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import TrackerConfig

FIRST_FRAME = "FIRST_FRAME"
ESTIMATED = "ESTIMATED"
UNKNOWN = "UNKNOWN"
CUT = "CUT"
CAMERA_STATUSES = (FIRST_FRAME, ESTIMATED, UNKNOWN, CUT)

# reasons
TOO_FEW_FEATURES = "TOO_FEW_FEATURES"
TOO_FEW_TRACKED = "TOO_FEW_TRACKED"
NO_TRANSFORM = "NO_TRANSFORM"
LOW_INLIER_FRACTION = "LOW_INLIER_FRACTION"
IMPLAUSIBLE_SHIFT = "IMPLAUSIBLE_SHIFT"
IMPLAUSIBLE_SCALE = "IMPLAUSIBLE_SCALE"
IMPLAUSIBLE_ROTATION = "IMPLAUSIBLE_ROTATION"
HISTOGRAM_CHANGE = "HISTOGRAM_CHANGE"
RESOLUTION_CHANGED = "RESOLUTION_CHANGED"

Matrix = Tuple[Tuple[float, float, float], Tuple[float, float, float]]
IDENTITY: Matrix = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))


class CameraError(RuntimeError):
    """Invalid input to the camera estimator (fail closed)."""


@dataclass(frozen=True)
class CameraMotion:
    """How the image moved from the previous frame to this one (see the module docstring)."""
    status: str
    matrix: Matrix                    # previous-frame px -> current-frame px (identity unless ESTIMATED)
    shift_px: Tuple[float, float]     # displacement of the image centre under `matrix`
    scale: float                      # uniform scale of `matrix` (1.0 unless ESTIMATED)
    rotation_rad: float
    n_features: int                   # background features found in the previous frame
    n_tracked: int                    # features tracked into this frame (outside person boxes)
    inlier_fraction: Optional[float]  # RANSAC inliers / n_features (None if no transform was fitted)
    hist_distance: Optional[float]    # grey-histogram Bhattacharyya distance to the previous frame
    reasons: Tuple[str, ...]
    continuity_broken: bool
    measured_matrix: Optional[Matrix] = None  # a fitted transform that was REJECTED (audit only; never applied)

    @property
    def compensated(self) -> bool:
        """True only when `matrix` is a measurement."""
        return self.status == ESTIMATED

    def warp_point(self, x: float, y: float) -> Tuple[float, float]:
        (a, b, tx), (c, d, ty) = self.matrix
        return a * x + b * y + tx, c * x + d * y + ty

    def warp_vector(self, vx: float, vy: float) -> Tuple[float, float]:
        (a, b, _), (c, d, _) = self.matrix
        return a * vx + b * vy, c * vx + d * vy

    def warp_box(self, box: Sequence[float]) -> Tuple[float, float, float, float]:
        """Box centre moved by the transform; width and height multiplied by its scale (the box stays axis-aligned:
        the small broadcast roll is applied to the centre only)."""
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        hw, hh = (box[2] - box[0]) / 2 * self.scale, (box[3] - box[1]) / 2 * self.scale
        nx, ny = self.warp_point(cx, cy)
        return nx - hw, ny - hh, nx + hw, ny + hh

    def as_record(self) -> Dict[str, Any]:
        return {"status": self.status, "matrix": [list(r) for r in self.matrix], "shift_px": list(self.shift_px),
                "scale": self.scale, "rotation_rad": self.rotation_rad, "n_features": self.n_features,
                "n_tracked": self.n_tracked, "inlier_fraction": self.inlier_fraction,
                "hist_distance": self.hist_distance, "reasons": list(self.reasons),
                "continuity_broken": self.continuity_broken,
                "measured_matrix": None if self.measured_matrix is None else [list(r) for r in self.measured_matrix]}


def identity_motion(status: str, *, reasons: Sequence[str] = (), n_features: int = 0, n_tracked: int = 0,
                    inlier_fraction: Optional[float] = None, hist_distance: Optional[float] = None,
                    continuity_broken: bool = False, measured_matrix: Optional[Matrix] = None) -> CameraMotion:
    if status == ESTIMATED:
        raise CameraError("identity_motion is not a measurement; ESTIMATED needs a fitted matrix")
    return CameraMotion(status, IDENTITY, (0.0, 0.0), 1.0, 0.0, int(n_features), int(n_tracked), inlier_fraction,
                        hist_distance, tuple(reasons), bool(continuity_broken), measured_matrix)


def motion_from_matrix(matrix: Sequence[Sequence[float]], width: int, height: int, *, n_features: int = 0,
                       n_tracked: int = 0, inlier_fraction: Optional[float] = None,
                       hist_distance: Optional[float] = None) -> CameraMotion:
    """An ESTIMATED CameraMotion from a 2x3 similarity matrix (source pixels). Also used by tests."""
    m = np.asarray(matrix, dtype=np.float64).reshape(2, 3)
    if not np.all(np.isfinite(m)):
        raise CameraError("non-finite camera matrix")
    a, c = float(m[0, 0]), float(m[1, 0])
    scale = math.hypot(a, c)
    rot = math.atan2(c, a)
    cx, cy = width / 2, height / 2
    nx = m[0, 0] * cx + m[0, 1] * cy + m[0, 2]
    ny = m[1, 0] * cx + m[1, 1] * cy + m[1, 2]
    mat: Matrix = (tuple(float(v) for v in m[0]), tuple(float(v) for v in m[1]))  # type: ignore[assignment]
    return CameraMotion(ESTIMATED, mat, (float(nx - cx), float(ny - cy)), float(scale), float(rot),
                        int(n_features), int(n_tracked), inlier_fraction, hist_distance, (), False)


def _check_image(image: Any) -> None:
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3
            or image.shape[0] < 1 or image.shape[1] < 1):
        raise CameraError("camera estimator needs a non-empty BGR uint8 image of shape (h, w, 3)")


def _check_boxes(boxes: Sequence[Sequence[float]]) -> List[Tuple[float, float, float, float]]:
    out = []
    for b in boxes:
        if len(b) != 4 or not all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in b):
            raise CameraError(f"person box must be 4 finite numbers, got {b!r}")
        out.append(tuple(float(v) for v in b))
    return out


class CameraMotionEstimator:
    """Frame-to-frame global image motion and hard-cut detection. Call update() once per frame, in order."""

    def __init__(self, cfg: TrackerConfig):
        self._cfg = cfg
        self._prev_work: Optional[np.ndarray] = None
        self._prev_hist: Optional[np.ndarray] = None
        self._prev_boxes: List[Tuple[float, float, float, float]] = []
        self._prev_shape: Optional[Tuple[int, int]] = None

    def reset(self) -> None:
        self.__init__(self._cfg)

    # -- helpers ---------------------------------------------------------------------------------------------
    def _resize_width(self, gray: np.ndarray, width: int) -> Tuple[np.ndarray, float]:
        h, w = gray.shape[:2]
        s = width / w
        nh = max(1, int(round(h * s)))
        interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
        return cv2.resize(gray, (int(width), nh), interpolation=interp), s

    def _hist(self, gray: np.ndarray) -> np.ndarray:
        small, _ = self._resize_width(gray, self._cfg.get("cut_work_width_px"))
        bins = self._cfg.get("cut_hist_bins")
        levels = float(np.iinfo(np.uint8).max + 1)
        hist = cv2.calcHist([small], [0], None, [bins], [0.0, levels])
        total = float(hist.sum())
        return (hist / total).astype(np.float32) if total > 0 else hist.astype(np.float32)

    def _mask(self, shape: Tuple[int, int], boxes: Sequence[Sequence[float]], s: float) -> np.ndarray:
        mask = np.full(shape, np.iinfo(np.uint8).max, np.uint8)
        pad_h = self._cfg.get("flow_person_mask_pad_h")
        for b in boxes:
            pad = pad_h * (b[3] - b[1])
            x1, y1 = int(math.floor((b[0] - pad) * s)), int(math.floor((b[1] - pad) * s))
            x2, y2 = int(math.ceil((b[2] + pad) * s)), int(math.ceil((b[3] + pad) * s))
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(shape[1], x2), min(shape[0], y2)
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 0
        return mask

    # -- main ------------------------------------------------------------------------------------------------
    def update(self, image: np.ndarray, person_boxes: Sequence[Sequence[float]], dt_s: float) -> CameraMotion:
        """Motion from the previous frame to `image`. `person_boxes` are this frame's detections (source px);
        `dt_s` is the time since the previous frame (1 / fps for consecutive frames)."""
        _check_image(image)
        boxes = _check_boxes(person_boxes)
        if not (isinstance(dt_s, (int, float)) and math.isfinite(float(dt_s)) and dt_s > 0):
            raise CameraError(f"dt_s must be a positive finite number of seconds, got {dt_s!r}")
        c = self._cfg
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        work, s = self._resize_width(gray, c.get("flow_work_width_px"))
        hist = self._hist(gray)
        shape = (int(image.shape[0]), int(image.shape[1]))
        prev_work, prev_hist, prev_boxes, prev_shape = self._prev_work, self._prev_hist, self._prev_boxes, \
            self._prev_shape
        self._prev_work, self._prev_hist, self._prev_boxes, self._prev_shape = work, hist, boxes, shape

        if prev_work is None:
            return identity_motion(FIRST_FRAME)
        hist_d = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
        hist_change = hist_d >= c.get("cut_hist_distance_min")
        if prev_shape != shape:
            return identity_motion(CUT, reasons=(RESOLUTION_CHANGED,), hist_distance=hist_d, continuity_broken=True)

        reasons: List[str] = []
        n_feat = n_tracked = 0
        inlier_frac: Optional[float] = None
        motion: Optional[CameraMotion] = None
        pts = cv2.goodFeaturesToTrack(prev_work, maxCorners=c.get("flow_max_corners"),
                                      qualityLevel=c.get("flow_quality_level"),
                                      minDistance=c.get("flow_min_distance_px"),
                                      mask=self._mask(prev_work.shape, prev_boxes, s))
        n_feat = 0 if pts is None else int(len(pts))
        if n_feat < c.get("flow_min_features"):
            reasons.append(TOO_FEW_FEATURES)
        else:
            win = c.get("flow_lk_win_px")
            nxt, st, _err = cv2.calcOpticalFlowPyrLK(prev_work, work, pts, None, winSize=(win, win),
                                                     maxLevel=c.get("flow_lk_max_level"))
            ok = st.reshape(-1).astype(bool)
            p0, p1 = pts.reshape(-1, 2)[ok], nxt.reshape(-1, 2)[ok]
            cur_mask = self._mask(work.shape, boxes, s)
            xi = np.clip(np.round(p1[:, 0]).astype(int), 0, work.shape[1] - 1)
            yi = np.clip(np.round(p1[:, 1]).astype(int), 0, work.shape[0] - 1)
            inside = (p1[:, 0] >= 0) & (p1[:, 0] < work.shape[1]) & (p1[:, 1] >= 0) & (p1[:, 1] < work.shape[0])
            keep = inside & (cur_mask[yi, xi] > 0)
            p0, p1 = p0[keep], p1[keep]
            n_tracked = int(len(p0))
            if n_tracked < c.get("flow_min_features"):
                reasons.append(TOO_FEW_TRACKED)
            else:
                m, inl = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC,
                                                     ransacReprojThreshold=c.get("flow_ransac_reproj_px"))
                if m is None or inl is None:
                    reasons.append(NO_TRANSFORM)
                else:
                    inlier_frac = float(inl.reshape(-1).astype(bool).sum()) / n_feat
                    if inlier_frac <= c.get("cut_flow_inlier_fraction_max"):
                        reasons.append(LOW_INLIER_FRACTION)
                    else:
                        full = m.astype(np.float64).copy()
                        full[:, 2] /= s      # working px -> source px (the linear part is unchanged)
                        motion = motion_from_matrix(full, shape[1], shape[0], n_features=n_feat,
                                                    n_tracked=n_tracked, inlier_fraction=inlier_frac,
                                                    hist_distance=hist_d)
                        shift = math.hypot(*motion.shift_px) / shape[1]
                        if shift > c.get("camera_max_shift_frac_per_s") * dt_s:
                            reasons.append(IMPLAUSIBLE_SHIFT)
                        lim = math.log(c.get("camera_max_scale_ratio_per_s")) * dt_s
                        if not motion.scale > 0 or abs(math.log(motion.scale)) > lim:
                            reasons.append(IMPLAUSIBLE_SCALE)
                        if abs(motion.rotation_rad) > c.get("camera_max_rotation_rad_per_s") * dt_s:
                            reasons.append(IMPLAUSIBLE_ROTATION)

        if not reasons and motion is not None:
            return motion
        measured = None if motion is None else motion.matrix
        if hist_change:
            return identity_motion(CUT, reasons=tuple(reasons) + (HISTOGRAM_CHANGE,), n_features=n_feat,
                                   n_tracked=n_tracked, inlier_fraction=inlier_frac, hist_distance=hist_d,
                                   continuity_broken=True, measured_matrix=measured)
        return identity_motion(UNKNOWN, reasons=tuple(reasons), n_features=n_feat, n_tracked=n_tracked,
                               inlier_fraction=inlier_frac, hist_distance=hist_d,
                               continuity_broken=bool(c.get("camera_unknown_breaks_continuity")),
                               measured_matrix=measured)

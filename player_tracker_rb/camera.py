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
  6. Photometric check: the previous frame is warped with the fitted transform and compared with this frame over
     the background (person boxes of BOTH frames masked; both blurred by cut_residual_blur_px and mean-centred, so a uniform
     exposure change does not count). residual = mean|a' - b'| / (mean|a'| + mean|b'|), in [0, 1] and independent
     of the image's texture contrast. A transform that leaves a residual
     >= cut_residual_min does not explain the image change (e.g. a static broadcast overlay dominated the features
     across a cut): the estimate is rejected (HIGH_RESIDUAL), which is a flow failure.
  7. Hard cut: grey-histogram Bhattacharyya distance >= cut_hist_distance_min AND flow failure.
  8. Scale corroboration (NEW_R2, R2 DEV F8). A similarity fit to a panning broadcast camera over a deep scene
     (pitch near, stands far) can report a steady false zoom (DEV: x0.66 over 4 s while people's heights rose).
     People are the scale that matters downstream, so the fitted scale is checked against them: each previous person
     box is moved by the fitted transform (centre only) and matched one-to-one to a current box by IoU >=
     camera_scale_match_iou_min; with >= camera_scale_check_min_people matches the frame's people log-height ratio
     is their median (a box nested >= population_nested_containment in a larger one counts once). Over the last
     camera_scale_check_window_s AND camera_scale_check_long_window_s of ESTIMATED frames (people measured on at
     least half of each), if exp|sum log(fitted scale) - sum people log-ratio| > camera_scale_disagreement_ratio_max
     in either the scale is UNCORROBORATED until both are back within camera_scale_release_ratio: the frame is
     reported as a TRANSLATION-ONLY motion (scale 1, no rotation, the image centre moved as the fit moves it),
     reason SCALE_UNCORROBORATED, the full fit kept as measured_matrix. Fail closed: no size change is
     propagated on a scale the people contradict. Without enough people the fit is used as before (no evidence
     either way). The window always accumulates the FITTED scale, so a sustained false zoom stays rejected.

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
HIGH_RESIDUAL = "HIGH_RESIDUAL"
RESOLUTION_CHANGED = "RESOLUTION_CHANGED"
SCALE_UNCORROBORATED = "SCALE_UNCORROBORATED"   # NEW_R2 F8: the fitted scale disagrees with people's heights

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
    inlier_fraction: Optional[float]  # RANSAC inliers / n_tracked (None if no transform was fitted)
    hist_distance: Optional[float]    # grey-histogram Bhattacharyya distance to the previous frame
    reasons: Tuple[str, ...]
    continuity_broken: bool
    measured_matrix: Optional[Matrix] = None  # a fitted transform that was REJECTED (audit only; never applied)
    residual: Optional[float] = None          # photometric residual of the fitted transform (0-1), if one was fitted

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
                "measured_matrix": None if self.measured_matrix is None else [list(r) for r in self.measured_matrix],
                "residual": self.residual}


def identity_motion(status: str, *, reasons: Sequence[str] = (), n_features: int = 0, n_tracked: int = 0,
                    inlier_fraction: Optional[float] = None, hist_distance: Optional[float] = None,
                    continuity_broken: bool = False, measured_matrix: Optional[Matrix] = None,
                    residual: Optional[float] = None) -> CameraMotion:
    if status == ESTIMATED:
        raise CameraError("identity_motion is not a measurement; ESTIMATED needs a fitted matrix")
    return CameraMotion(status, IDENTITY, (0.0, 0.0), 1.0, 0.0, int(n_features), int(n_tracked), inlier_fraction,
                        hist_distance, tuple(reasons), bool(continuity_broken), measured_matrix, residual)


def motion_from_matrix(matrix: Sequence[Sequence[float]], width: int, height: int, *, n_features: int = 0,
                       n_tracked: int = 0, inlier_fraction: Optional[float] = None,
                       hist_distance: Optional[float] = None, residual: Optional[float] = None) -> CameraMotion:
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
                        int(n_features), int(n_tracked), inlier_fraction, hist_distance, (), False, None, residual)


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
        self._t = 0.0
        # F8: (t, log fitted scale, people log-height ratio or None) of recent ESTIMATED frames
        self._scale_window: List[Tuple[float, float, Optional[float]]] = []
        self._scale_rejected = False                         # F8: hysteresis state of the scale check

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

    def _residual(self, prev_work: np.ndarray, work: np.ndarray, m: np.ndarray, mask: np.ndarray) -> float:
        """Normalised photometric residual in [0, 1] between the warped previous frame a and this frame b over the
        background (`mask` > 0 and inside the warped image), both blurred and mean-centred (a', b'):
            residual = mean|a' - b'| / (mean|a'| + mean|b'|)
        0 when the transform explains the change; about 0.5-0.7 for unrelated images, at any texture contrast.
        1.0 if nothing is comparable."""
        sigma = self._cfg.get("cut_residual_blur_px")
        h, w = work.shape[:2]
        a = cv2.GaussianBlur(prev_work, (0, 0), sigma).astype(np.float32)
        b = cv2.GaussianBlur(work, (0, 0), sigma).astype(np.float32)
        warped = cv2.warpAffine(a, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=-1.0)
        valid = (warped >= 0) & (mask > 0)
        if not np.any(valid):
            return 1.0
        da, db = warped[valid] - warped[valid].mean(), b[valid] - b[valid].mean()
        contrast = float(np.abs(da).mean() + np.abs(db).mean())
        if contrast <= 0.0:
            return 1.0
        return float(np.abs(da - db).mean() / contrast)

    def _people_log_ratio(self, motion: "CameraMotion", prev_boxes: Sequence[Sequence[float]],
                          boxes: Sequence[Sequence[float]]) -> Optional[float]:
        """F8: median log height ratio of people matched between the frames (previous box centre moved by the fitted
        transform, size kept; one-to-one greedy by IoU), or None with too few matches."""
        c = self._cfg
        prev_boxes, boxes = self._bodies(prev_boxes), self._bodies(boxes)
        if not prev_boxes or not boxes:
            return None
        cur = np.asarray(boxes, dtype=np.float64)
        moved = []
        for b in prev_boxes:
            nx, ny = motion.warp_point((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
            hw, hh = (b[2] - b[0]) / 2, (b[3] - b[1]) / 2
            moved.append((nx - hw, ny - hh, nx + hw, ny + hh))
        m = np.asarray(moved, dtype=np.float64)
        ix = np.clip(np.minimum(m[:, None, 2], cur[None, :, 2]) - np.maximum(m[:, None, 0], cur[None, :, 0]), 0, None)
        iy = np.clip(np.minimum(m[:, None, 3], cur[None, :, 3]) - np.maximum(m[:, None, 1], cur[None, :, 1]), 0, None)
        inter = ix * iy
        area_m = (m[:, 2] - m[:, 0]) * (m[:, 3] - m[:, 1])
        area_c = (cur[:, 2] - cur[:, 0]) * (cur[:, 3] - cur[:, 1])
        union = area_m[:, None] + area_c[None, :] - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        ratios, used_i, used_j = [], set(), set()
        for k in np.argsort(-iou, axis=None):
            i, j = divmod(int(k), iou.shape[1])
            if iou[i, j] < c.get("camera_scale_match_iou_min"):
                break
            if i in used_i or j in used_j:
                continue
            used_i.add(i)
            used_j.add(j)
            h0, h1 = prev_boxes[i][3] - prev_boxes[i][1], boxes[j][3] - boxes[j][1]
            if h0 > 0 and h1 > 0:
                ratios.append(math.log(h1 / h0))
        if len(ratios) < c.get("camera_scale_check_min_people"):
            return None
        return float(np.median(ratios))

    def _bodies(self, boxes: Sequence[Sequence[float]]) -> List[Sequence[float]]:
        """F8: one box per body: a box lying >= population_nested_containment inside another, larger box (a
        duplicate, part or alternative box) is dropped, so one person's boxes count once."""
        c = self._cfg
        keep = []
        for i, b in enumerate(boxes):
            area = (b[2] - b[0]) * (b[3] - b[1])
            nested = False
            for j, o in enumerate(boxes):
                if i == j:
                    continue
                oarea = (o[2] - o[0]) * (o[3] - o[1])
                ix = max(0.0, min(b[2], o[2]) - max(b[0], o[0]))
                iy = max(0.0, min(b[3], o[3]) - max(b[1], o[1]))
                if area > 0 and ix * iy / area >= c.get("population_nested_containment") and \
                        (oarea > area or (oarea == area and j < i)):
                    nested = True
                    break
            if not nested:
                keep.append(b)
        return keep

    def _window_disagreement(self, t: float, win: float, min_span: float) -> Optional[float]:
        """F8: |sum log fitted scale - sum people log ratio| over the last `win` s of ESTIMATED frames, or None when
        less than `min_span` s is available or people were measured on fewer than half of its frames."""
        rows = [x for x in self._scale_window if t - x[0] <= win]
        if not rows or t - rows[0][0] < min_span:
            return None
        pairs = [(ls, pr) for (_, ls, pr) in rows if pr is not None]
        if 2 * len(pairs) < len(rows):
            return None
        return abs(sum(ls for ls, _ in pairs) - sum(pr for _, pr in pairs))

    def _scale_uncorroborated(self, t: float, log_scale: float, people: Optional[float]) -> Tuple[bool, Dict[str, Any]]:
        """F8: add this ESTIMATED frame and say whether the fitted scale is contradicted by people, over a short
        window (a fast false zoom) and a long one (a slow drift). Hysteresis: once rejected, the scale stays rejected
        until both windows agree within camera_scale_release_ratio (no flapping at the threshold)."""
        c = self._cfg
        short_w, long_w = c.get("camera_scale_check_window_s"), c.get("camera_scale_check_long_window_s")
        self._scale_window.append((t, log_scale, people))
        self._scale_window = [x for x in self._scale_window if t - x[0] <= max(short_w, long_w)]
        # the long window decides as soon as the short one could (it then holds all the history up to long_w)
        min_span = short_w * c.get("camera_scale_check_min_span_fraction")
        ds = [d for d in (self._window_disagreement(t, short_w, min_span),
                          self._window_disagreement(t, long_w, min_span)) if d is not None]
        info: Dict[str, Any] = {"disagreement_ratio": math.exp(max(ds)) if ds else None}
        worst = max(ds) if ds else None
        if self._scale_rejected:
            if worst is not None and worst <= math.log(c.get("camera_scale_release_ratio")):
                self._scale_rejected = False
        elif worst is not None and worst > math.log(c.get("camera_scale_disagreement_ratio_max")):
            self._scale_rejected = True
        return self._scale_rejected, info

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

        self._t += dt_s
        if prev_work is None:
            self._scale_window, self._scale_rejected = [], False
            return identity_motion(FIRST_FRAME)
        hist_d = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
        hist_change = hist_d >= c.get("cut_hist_distance_min")
        if prev_shape != shape:
            self._scale_window, self._scale_rejected = [], False
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
                    inlier_frac = float(inl.reshape(-1).astype(bool).sum()) / n_tracked
                    if inlier_frac <= c.get("cut_flow_inlier_fraction_max"):
                        reasons.append(LOW_INLIER_FRACTION)
                    else:
                        full = m.astype(np.float64).copy()
                        full[:, 2] /= s      # working px -> source px (the linear part is unchanged)
                        prev_mask = self._mask(prev_work.shape, prev_boxes, s)
                        warped_prev_mask = cv2.warpAffine(prev_mask, m, (work.shape[1], work.shape[0]),
                                                          flags=cv2.INTER_NEAREST, borderValue=0)
                        resid = self._residual(prev_work, work, m, np.minimum(cur_mask, warped_prev_mask))
                        motion = motion_from_matrix(full, shape[1], shape[0], n_features=n_feat,
                                                    n_tracked=n_tracked, inlier_fraction=inlier_frac,
                                                    hist_distance=hist_d, residual=resid)
                        if resid >= c.get("cut_residual_min"):
                            reasons.append(HIGH_RESIDUAL)
                        shift = math.hypot(*motion.shift_px) / shape[1]
                        if shift > c.get("camera_max_shift_frac_per_s") * dt_s:
                            reasons.append(IMPLAUSIBLE_SHIFT)
                        lim = math.log(c.get("camera_max_scale_ratio_per_s")) * dt_s
                        if not motion.scale > 0 or abs(math.log(motion.scale)) > lim:
                            reasons.append(IMPLAUSIBLE_SCALE)
                        if abs(motion.rotation_rad) > c.get("camera_max_rotation_rad_per_s") * dt_s:
                            reasons.append(IMPLAUSIBLE_ROTATION)

        if not reasons and motion is not None:
            people = self._people_log_ratio(motion, prev_boxes, boxes)
            bad, _info = self._scale_uncorroborated(self._t, math.log(motion.scale), people)
            if not bad:
                return motion
            # F8: translation-only motion that moves the image centre as the fit does (unbiased by where the
            # features happen to lie, unlike their median flow under a false zoom); the full fit is audit only
            dx, dy = motion.shift_px
            tr = motion_from_matrix([[1.0, 0.0, dx], [0.0, 1.0, dy]], shape[1], shape[0], n_features=n_feat,
                                    n_tracked=n_tracked, inlier_fraction=inlier_frac, hist_distance=hist_d,
                                    residual=motion.residual)
            return CameraMotion(tr.status, tr.matrix, tr.shift_px, tr.scale, tr.rotation_rad, tr.n_features,
                                tr.n_tracked, tr.inlier_fraction, tr.hist_distance, (SCALE_UNCORROBORATED,), False,
                                motion.matrix, tr.residual)
        self._scale_window, self._scale_rejected = [], False   # no measured continuity: the check starts again
        measured = None if motion is None else motion.matrix
        resid = None if motion is None else motion.residual
        if hist_change:
            return identity_motion(CUT, reasons=tuple(reasons) + (HISTOGRAM_CHANGE,), n_features=n_feat,
                                   n_tracked=n_tracked, inlier_fraction=inlier_frac, hist_distance=hist_d,
                                   continuity_broken=True, measured_matrix=measured, residual=resid)
        return identity_motion(UNKNOWN, reasons=tuple(reasons), n_features=n_feat, n_tracked=n_tracked,
                               inlier_fraction=inlier_frac, hist_distance=hist_d,
                               continuity_broken=bool(c.get("camera_unknown_breaks_continuity")),
                               measured_matrix=measured, residual=resid)

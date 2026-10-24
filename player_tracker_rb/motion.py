# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R1/R1_DESIGN_DECISIONS.md D06 (observed vs predicted vs held), D09 (motion is a veto only),
#                 D13 (seconds and body heights), D16 (camera compensation, cuts); R1-E instruction (Nathan, 2026-10-24)
"""
player_tracker_rb.motion: a bounded, camera-compensated constant-velocity box model for ONE body.

Motion can say "this box is physically implausible for that body" (a veto). It can never say "this box IS that
body": nothing in this module produces an identity score.

State (per body): the last OBSERVED box (kept in current-image coordinates by applying each frame's camera
transform to it), its time, a velocity (px/s, camera-compensated, so it is motion relative to the background) and a
log-height rate (1/s, for players approaching or leaving the camera).

Update from an observation at time t (dt = t - t_obs > 0):
    alpha  = 1 - exp(-dt / motion_velocity_tau_s)                  (the same smoothing at any fps)
    v     += alpha * ((c_obs - c_ref) / dt - v),  |v| <= motion_veto_max_speed_h_per_s * h
    r     += alpha * (ln(h_obs / h_ref) / dt - r), |r| <= ln(motion_veto_max_scale_ratio_per_s)
A PARTIAL observation (part of the body hidden or outside the image) moves the reference box but updates neither
v nor r, and the next observation after a partial one does not update them either (its centre jump is not motion).

Prediction at time t: extrapolate for de = min(t - t_obs, motion_max_extrapolation_s):
    c = c_ref + v * de,   w, h = w_ref, h_ref * exp(r * de)
source PREDICTED while t - t_obs <= motion_max_extrapolation_s, HELD after that (camera-compensated only). A
prediction is never an observation: Prediction.source is never DETECTION.

Plausibility check of a box against the state (the veto):
    speed:  |c_box - c_ref| <= (motion_jitter_h + motion_veto_max_speed_h_per_s * dt) * h_pred
    scale:  h_box / h_pred within [1/k, k],  k = motion_scale_jitter_ratio * motion_veto_max_scale_ratio_per_s**dt
            (skipped when the box or the reference is PARTIAL: a hidden part changes the height, not the body)
h is the predicted box height, so every limit is in body heights and seconds (D13).

Camera: apply_camera() moves the reference box and rotates/scales the velocity with each frame's ESTIMATED
transform. An UNKNOWN frame cannot be compensated; the state records that (camera_compensated becomes False until
the next observation). After a CUT the caller must discard the state: a position from another shot is meaningless.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Dict, Sequence, Tuple

from .camera import CameraMotion
from .config import TrackerConfig

DETECTION = "DETECTION"
PREDICTED = "PREDICTED"
HELD = "HELD"

SPEED = "SPEED"
SCALE = "SCALE"
NO_CONTINUITY = "NO_CONTINUITY"


class MotionError(RuntimeError):
    """Invalid motion input (fail closed)."""


Box = Tuple[float, float, float, float]


def _finite_box(box: Sequence[float]) -> Box:
    if len(box) != 4 or not all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in box):
        raise MotionError(f"box must be 4 finite numbers, got {box!r}")
    b = tuple(float(v) for v in box)
    if not (b[2] > b[0] and b[3] > b[1]):
        raise MotionError(f"box must have positive width and height, got {box!r}")
    return b  # type: ignore[return-value]


def centre(box: Sequence[float]) -> Tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def box_from(cx: float, cy: float, w: float, h: float) -> Box:
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


@dataclass(frozen=True)
class MotionState:
    ref_box: Box                    # last observed box, camera-compensated into the current image
    t_obs: float                    # time of that observation (s)
    vx: float = 0.0                 # px/s (camera-compensated)
    vy: float = 0.0
    log_h_rate: float = 0.0         # 1/s
    n_obs: int = 1
    ref_partial: bool = False
    camera_compensated: bool = True  # every frame since the observation had an ESTIMATED (or no) camera change

    def as_record(self) -> Dict[str, Any]:
        return {"ref_box": list(self.ref_box), "t_obs": self.t_obs, "velocity_px_per_s": [self.vx, self.vy],
                "log_height_rate_per_s": self.log_h_rate, "n_obs": self.n_obs, "ref_partial": self.ref_partial,
                "camera_compensated": self.camera_compensated}


@dataclass(frozen=True)
class Prediction:
    box: Box
    source: str                     # PREDICTED or HELD, never DETECTION
    dt_s: float                     # time since the last observation
    extrapolated_s: float           # how much of dt_s was extrapolated with the velocity
    camera_compensated: bool


@dataclass(frozen=True)
class MotionCheck:
    plausible: bool
    reasons: Tuple[str, ...]
    displacement_h: float
    speed_h_per_s: float
    scale_ratio: float
    allowed_displacement_h: float
    allowed_scale_ratio: float

    def as_record(self) -> Dict[str, Any]:
        return {"plausible": self.plausible, "reasons": list(self.reasons), "displacement_h": self.displacement_h,
                "speed_h_per_s": self.speed_h_per_s, "scale_ratio": self.scale_ratio,
                "allowed_displacement_h": self.allowed_displacement_h,
                "allowed_scale_ratio": self.allowed_scale_ratio}


class MotionModel:
    """Stateless operations on MotionState (see the module docstring for every formula)."""

    def __init__(self, cfg: TrackerConfig):
        self._cfg = cfg

    def start(self, box: Sequence[float], t: float, *, partial: bool = False) -> MotionState:
        return MotionState(ref_box=_finite_box(box), t_obs=float(t), ref_partial=bool(partial))

    def apply_camera(self, st: MotionState, cam: CameraMotion) -> MotionState:
        if cam.compensated:
            vx, vy = cam.warp_vector(st.vx, st.vy)
            return replace(st, ref_box=_finite_box(cam.warp_box(st.ref_box)), vx=vx, vy=vy)
        if cam.continuity_broken:
            raise MotionError("camera continuity is broken; discard the motion state instead of compensating it")
        if cam.status == "FIRST_FRAME":
            return st
        return replace(st, camera_compensated=False)

    def _dt(self, st: MotionState, t: float) -> float:
        dt = float(t) - st.t_obs
        if not math.isfinite(dt) or dt <= 0:
            raise MotionError(f"time must be after the last observation (dt={dt!r})")
        return dt

    def predict(self, st: MotionState, t: float) -> Prediction:
        dt = self._dt(st, t)
        horizon = self._cfg.get("motion_max_extrapolation_s")
        de = min(dt, horizon)
        cx, cy = centre(st.ref_box)
        w = st.ref_box[2] - st.ref_box[0]
        h = st.ref_box[3] - st.ref_box[1]
        g = math.exp(st.log_h_rate * de)
        box = box_from(cx + st.vx * de, cy + st.vy * de, w * g, h * g)
        return Prediction(box=box, source=PREDICTED if dt <= horizon else HELD, dt_s=dt, extrapolated_s=de,
                          camera_compensated=st.camera_compensated)

    def check(self, st: MotionState, box: Sequence[float], t: float, *, partial: bool = False) -> MotionCheck:
        b = _finite_box(box)
        c = self._cfg
        dt = self._dt(st, t)
        pred = self.predict(st, t)
        h_pred = pred.box[3] - pred.box[1]
        bx, by = centre(b)
        rx, ry = centre(st.ref_box)
        disp_h = math.hypot(bx - rx, by - ry) / h_pred
        allowed_h = c.get("motion_jitter_h") + c.get("motion_veto_max_speed_h_per_s") * dt
        ratio = (b[3] - b[1]) / h_pred
        k = c.get("motion_scale_jitter_ratio") * c.get("motion_veto_max_scale_ratio_per_s") ** dt
        reasons = []
        if disp_h > allowed_h:
            reasons.append(SPEED)
        if not (partial or st.ref_partial) and not (1.0 / k <= ratio <= k):
            reasons.append(SCALE)
        return MotionCheck(plausible=not reasons, reasons=tuple(reasons), displacement_h=disp_h,
                           speed_h_per_s=disp_h / dt, scale_ratio=ratio, allowed_displacement_h=allowed_h,
                           allowed_scale_ratio=k)

    def update(self, st: MotionState, box: Sequence[float], t: float, *, partial: bool = False) -> MotionState:
        b = _finite_box(box)
        c = self._cfg
        dt = self._dt(st, t)
        vx, vy, r = st.vx, st.vy, st.log_h_rate
        if not partial and not st.ref_partial:
            alpha = 1.0 - math.exp(-dt / c.get("motion_velocity_tau_s"))
            bx, by = centre(b)
            rx, ry = centre(st.ref_box)
            h_obs, h_ref = b[3] - b[1], st.ref_box[3] - st.ref_box[1]
            vx += alpha * ((bx - rx) / dt - vx)
            vy += alpha * ((by - ry) / dt - vy)
            vmax = c.get("motion_veto_max_speed_h_per_s") * h_obs
            speed = math.hypot(vx, vy)
            if speed > vmax:
                vx, vy = vx * vmax / speed, vy * vmax / speed
            r += alpha * (math.log(h_obs / h_ref) / dt - r)
            rmax = math.log(c.get("motion_veto_max_scale_ratio_per_s"))
            r = max(-rmax, min(rmax, r))
        return MotionState(ref_box=b, t_obs=float(t), vx=vx, vy=vy, log_h_rate=r, n_obs=st.n_obs + 1,
                           ref_partial=bool(partial), camera_compensated=True)

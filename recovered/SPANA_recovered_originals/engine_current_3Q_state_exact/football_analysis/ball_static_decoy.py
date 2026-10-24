"""
ball_static_decoy.py -- STATIC-DECOY TASK: StaticDecoyTracker + generic
pitch-region classification.

Core principle (rule #2): a match ball is a dynamic object; a printed
football on an advertising board (or a static league-logo graphic) is part
of the static SCENE. Detector confidence and ball-like appearance alone
must never be enough to make a candidate the active match-ball track --
this module adds the missing signals: whether a candidate keeps
reappearing in the same place relative to the BACKGROUND (not just the
same pixel coordinates -- see ball_background_motion.py for camera-motion
compensation), whether its surrounding image patch stays suspiciously
identical over time, and whether it sits on the playable grass or embedded
in pitch-side signage/crowd/scoreboard graphics.

This is entirely additive, downstream evidence: nothing here talks to
player_tracker, nothing here decides anything about possession/events, and
nothing here permanently blacklists a screen location (rule #6/#23.7) --
a location's accumulated evidence resets after STALE_AFTER_FRAMES of not
being proposed as a candidate at all, so a real ball that happens to pause
near where a decoy once was is judged completely fresh, not tainted by
that decoy's old history.

Rule #10's core distinction -- "static relative to background + off-pitch/
board context" (reject) vs. "temporarily stationary match ball on pitch"
(do not reject) -- is implemented as a MULTIPLICATIVE combination
(background_static_probability x off_pitch_weight), never motion alone:
a perfectly still candidate sitting ON_PITCH (a genuine free-kick/penalty
ball) or right next to the tracked player scores near-zero "decoy_score"
regardless of how long it has been stationary, because off_pitch_weight
(or the near-player override) is near zero for it.
"""
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .geometry import center, dist
from .visual_pitch_calibration import _grass_mask, _pitch_horizon   # read-only reuse, rule #8 -- this
                                                                       # module is never modified or re-imported
                                                                       # with side effects, only its two pure
                                                                       # segmentation helpers are called

BIN_SIZE = 20.0                # px -- spatial hash bucket for "the same recurring screen location"
PATCH_HALF = 10                # px half-size of the appearance patch sampled around a candidate
PATCH_RESIZE = 16               # patches are resized to this before comparison, so patch position jitter
                                 # of a few pixels doesn't itself look like a change
STATIC_EMA_RATE = 0.35          # how fast background_static_probability rises on a static observation
RESIDUAL_STATIC_THRESH = 6.0    # px/frame -- residual (background-compensated) motion at/below this
                                 # counts as "static" for this observation
STALE_AFTER_FRAMES = 45         # a bin not proposed as a candidate for this long is forgotten entirely --
                                 # rule #6/#23.7: never a permanent blacklist
DECOY_SCORE_REJECT = 0.5        # decoy_score at/above this is reported as a STATIC_DECOY/OFF_PITCH_CANDIDATE
NEAR_PLAYER_RADIUS = 150.0      # px -- rule #10's "near a player" override radius
GRASS_LOCAL_WINDOW = 18         # px half-size of the local grass-fraction sampling window
GRASS_ON_PITCH_FRACTION = 0.30
NEAR_PITCH_BAND = 45.0          # px either side of the pitch horizon counted as NEAR_PITCH rather than OFF/ON

OFF_PITCH_WEIGHT = {"ON_PITCH": 0.0, "NEAR_PITCH": 0.55, "OFF_PITCH": 1.0, "UNKNOWN": 0.3}

# TASK 3E fix (see DUEL_LOST_STATIC_DECOY_ROOT_CAUSE.md): a small, compact,
# color-coincidence "grass" blob -- e.g. dark/desaturated pixels inside a
# broadcast sponsor-logo or scorebug graphic that happen to fall inside
# visual_pitch_calibration._grass_mask()'s HSV range, then get bridged by
# that function's own internal 9x9 morphological CLOSE into one small solid
# island -- is categorically different from a genuine patch of pitch: real
# grass, wherever the camera finds it, is a large, continuous expanse, not
# an isolated few-dozen-pixel speck. This filter is a LOCAL, read-only
# refinement of the MASK THIS MODULE ALREADY OWNS a copy of (PitchRegionClassifier's
# self._mask) -- it never edits visual_pitch_calibration.py itself (rule #8
# is preserved: that module's two helpers are still called as pure, unmodified
# functions), so it has zero effect on any other caller of _grass_mask/
# _pitch_horizon (goal detection, release geometry, pitch timeline, clip
# tagging, calibration overlays -- all untouched). The threshold is not a
# hardcoded pixel count or clip-specific coordinate: it is expressed as a
# multiple of this module's own existing GRASS_LOCAL_WINDOW sampling-window
# area, i.e. "a surviving grass component must be meaningfully larger than
# the window we sample it through, or it cannot be trusted as a real patch
# of pitch at this scale."
MIN_GRASS_COMPONENT_WINDOWS = 8


def _filter_small_grass_components(mask, min_area):
    """Zero out connected components of `mask` smaller than `min_area`
    pixels. Read-only w.r.t. visual_pitch_calibration.py -- operates only on
    a mask array this module already holds a private copy of."""
    binary = (mask > 0).astype(np.uint8)
    n, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return mask
    keep = np.zeros(n, dtype=bool)
    keep[0] = False  # background label
    for i in range(1, n):
        keep[i] = stats[i, cv2.CC_STAT_AREA] >= min_area
    filtered = np.where(keep[labels], mask, 0).astype(mask.dtype)
    return filtered


@dataclass
class DecoyEvidence:
    background_static_probability: float
    residual_object_motion: Optional[float]
    frames_stationary: int
    patch_similarity: Optional[float]
    pitch_region: str          # ON_PITCH | NEAR_PITCH | OFF_PITCH | UNKNOWN
    near_player: bool
    decoy_score: float          # 0..1, combines background_static_probability with pitch/player context
    classification: str         # REAL_BALL_CANDIDATE | STATIC_DECOY | OFF_PITCH_CANDIDATE


@dataclass
class _BinHistory:
    last_frame: int
    last_pos: tuple
    last_bg_cum: tuple
    static_probability: float
    frames_stationary: int
    last_patch: Optional[np.ndarray]


def _extract_patch(frame, cx, cy):
    h, w = frame.shape[:2]
    x1, y1 = max(0, int(cx - PATCH_HALF)), max(0, int(cy - PATCH_HALF))
    x2, y2 = min(w, int(cx + PATCH_HALF)), min(h, int(cy + PATCH_HALF))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (PATCH_RESIZE, PATCH_RESIZE))


def _patch_similarity(a, b):
    if a is None or b is None:
        return None
    res = cv2.matchTemplate(a.astype(np.float32), b.astype(np.float32), cv2.TM_CCOEFF_NORMED)
    return float(max(0.0, min(1.0, (res[0][0] + 1.0) / 2.0)))


class PitchRegionClassifier:
    """Thin, read-only wrapper around visual_pitch_calibration.py's own
    grass-segmentation helpers (rule #8: reused as-is, never modified, no
    homography required). Computed at a downsampled resolution purely for
    speed -- this runs every frame for every ball candidate, unlike that
    module's own once-every-N-frames goal calibration sampling."""

    def __init__(self, downsample_width=240):
        self.downsample_width = downsample_width
        self._cache_frame_id = None
        self._mask = None
        self._horizon = None
        self._scale = 1.0

    def prepare(self, frame, frame_id):
        if self._cache_frame_id == frame_id:
            return
        h, w = frame.shape[:2]
        self._scale = self.downsample_width / float(w)
        small = cv2.resize(frame, (self.downsample_width, max(1, int(h * self._scale))))
        raw_mask = _grass_mask(small)
        # TASK 3E fix: strip small color-coincidence blobs (see
        # MIN_GRASS_COMPONENT_WINDOWS docstring above) before this mask is
        # used for either the local grass-fraction sample or the horizon --
        # visual_pitch_calibration.py's own _grass_mask/_pitch_horizon are
        # still called unmodified; only this module's private copy of their
        # output is refined.
        win = max(1, int(GRASS_LOCAL_WINDOW * self._scale))
        window_area = (2 * win) ** 2
        min_area = MIN_GRASS_COMPONENT_WINDOWS * window_area
        self._mask = _filter_small_grass_components(raw_mask, min_area)
        self._horizon = _pitch_horizon(self._mask)
        self._cache_frame_id = frame_id

    def classify(self, cx, cy):
        if self._mask is None:
            return "UNKNOWN"
        h, w = self._mask.shape[:2]
        sx, sy = cx * self._scale, cy * self._scale
        ix, iy = int(round(sx)), int(round(sy))
        if not (0 <= ix < w and 0 <= iy < h):
            return "UNKNOWN"

        win = max(1, int(GRASS_LOCAL_WINDOW * self._scale))
        x1, y1 = max(0, ix - win), max(0, iy - win)
        x2, y2 = min(w, ix + win), min(h, iy + win)
        local_grass_fraction = float((self._mask[y1:y2, x1:x2] > 0).mean()) if (x2 > x1 and y2 > y1) else 0.0

        if local_grass_fraction >= GRASS_ON_PITCH_FRACTION:
            return "ON_PITCH"

        horizon_y = float(self._horizon[ix])
        band = NEAR_PITCH_BAND * self._scale
        if abs(sy - horizon_y) <= band or sy >= horizon_y:
            return "NEAR_PITCH"
        return "OFF_PITCH"


class StaticDecoyTracker:
    """Per-clip (per-BallTracker-instance) memory of recurring candidate
    screen locations. See module docstring for the no-permanent-blacklist
    guarantee.

    Bins are keyed on BACKGROUND-MOTION-COMPENSATED position
    (cx - bg_cum_x, cy - bg_cum_y), not raw image-space position (rule #4:
    "static must NOT mean pixel coordinates don't move"). A candidate
    attached to the background keeps a near-constant compensated position
    even while its raw pixel position drifts every frame during a camera
    pan, so its accumulated history survives the pan intact instead of
    resetting every time it crosses a raw-pixel bin boundary. A genuinely
    independently-moving ball's compensated position keeps changing too
    (that's exactly what "independent motion" means), so it still lands in
    fresh bins as expected -- this only changes what "the same recurring
    location" means, not whether independent motion is recognised as such.
    """

    def __init__(self):
        self._bins = {}
        self.pitch_classifier = PitchRegionClassifier()

    def _bin_key(self, cx, cy, bg_cum=(0.0, 0.0)):
        return int((cx - bg_cum[0]) // BIN_SIZE), int((cy - bg_cum[1]) // BIN_SIZE)

    def _residual(self, hist, cx, cy, frame_idx, bg_cum):
        dt = frame_idx - hist.last_frame
        if dt <= 0:
            return None
        moved = ((cx - hist.last_pos[0]) ** 2 + (cy - hist.last_pos[1]) ** 2) ** 0.5
        bg_moved = ((bg_cum[0] - hist.last_bg_cum[0]) ** 2 + (bg_cum[1] - hist.last_bg_cum[1]) ** 2) ** 0.5
        # residual = |candidate displacement - background displacement|, per frame.
        # Using magnitudes (not vector subtraction) keeps this robust and cheap; since
        # a genuine background-attached object's displacement vector should be very
        # close in both magnitude AND direction to the background's own, while a truly
        # independently-moving ball's displacement is generally much larger in
        # magnitude than whatever the (usually ~0, on these static-camera clips)
        # background motion is -- see ball_background_motion.py's own module docstring.
        return abs(moved - bg_moved) / dt

    def evaluate(self, box, frame_idx, frame, bg_cum, player_bbox):
        """Score this candidate using history accumulated STRICTLY BEFORE
        this frame. Call observe() afterwards to fold this frame's data in
        for next time."""
        cx, cy = center(box)
        self.pitch_classifier.prepare(frame, frame_idx)
        pitch_region = self.pitch_classifier.classify(cx, cy)
        near_player = player_bbox is not None and dist((cx, cy), center(player_bbox)) < NEAR_PLAYER_RADIUS

        key = self._bin_key(cx, cy, bg_cum)
        hist = self._bins.get(key)
        if hist is None or (frame_idx - hist.last_frame) > STALE_AFTER_FRAMES:
            static_probability, residual, frames_stationary, patch_sim = 0.0, None, 0, None
        else:
            static_probability = hist.static_probability
            residual = self._residual(hist, cx, cy, frame_idx, bg_cum)
            frames_stationary = hist.frames_stationary
            patch_sim = _patch_similarity(hist.last_patch, _extract_patch(frame, cx, cy))

        off_pitch_weight = OFF_PITCH_WEIGHT.get(pitch_region, 0.3)
        decoy_score = static_probability * off_pitch_weight
        if near_player:
            decoy_score *= 0.25   # rule #10: a ball-shaped detection right at a player's feet is very
                                    # unlikely to be a stray pitch-side decoy, whatever its motion history
        decoy_score = max(0.0, min(1.0, decoy_score))

        if decoy_score >= DECOY_SCORE_REJECT:
            classification = "OFF_PITCH_CANDIDATE" if pitch_region == "OFF_PITCH" else "STATIC_DECOY"
        else:
            classification = "REAL_BALL_CANDIDATE"

        return DecoyEvidence(round(static_probability, 3), round(residual, 2) if residual is not None else None,
                              frames_stationary, round(patch_sim, 3) if patch_sim is not None else None,
                              pitch_region, near_player, round(decoy_score, 3), classification)

    def observe(self, box, frame_idx, frame, bg_cum):
        """Folds this frame's observation of `box` into that screen
        location's history, for future evaluate() calls. Must be called
        AFTER evaluate() for the same candidate/frame, never before."""
        cx, cy = center(box)
        key = self._bin_key(cx, cy, bg_cum)
        hist = self._bins.get(key)
        patch = _extract_patch(frame, cx, cy)

        if hist is None or (frame_idx - hist.last_frame) > STALE_AFTER_FRAMES:
            self._bins[key] = _BinHistory(last_frame=frame_idx, last_pos=(cx, cy), last_bg_cum=bg_cum,
                                           static_probability=0.0, frames_stationary=0, last_patch=patch)
            return

        residual = self._residual(hist, cx, cy, frame_idx, bg_cum)
        patch_sim = _patch_similarity(hist.last_patch, patch)
        is_static_this_obs = residual is not None and residual <= RESIDUAL_STATIC_THRESH
        # patch similarity CORROBORATES motion evidence, it never decides alone (rule #7)
        if is_static_this_obs and patch_sim is not None and patch_sim < 0.5:
            is_static_this_obs = False

        if is_static_this_obs:
            hist.frames_stationary += 1
            hist.static_probability += STATIC_EMA_RATE * (1.0 - hist.static_probability)
        else:
            # a candidate that turns out to be moving independently loses trust fast --
            # much faster than it built up -- so a real ball that happens to reuse a
            # screen location a decoy once occupied is judged on its own merits almost
            # immediately (rule #10)
            hist.frames_stationary = 0
            hist.static_probability *= 0.35

        hist.last_frame = frame_idx
        hist.last_pos = (cx, cy)
        hist.last_bg_cum = bg_cum
        hist.last_patch = patch

    def reset(self):
        self._bins = {}
        self.pitch_classifier = PitchRegionClassifier()

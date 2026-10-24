# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R1/R1_DESIGN_DECISIONS.md D04, D08, D12; R1 approval §4 and §20; R1/R1_GENERIC_TRACKER_REQUIREMENT.md;
#                 R1-D instruction (Nathan, 2026-10-24: separate cues, neither alone establishes identity);
#                 R1/R1D/R1D_STAGE_RECORD.md (review findings applied)
"""
player_tracker_rb.appearance: generic appearance cues for any selected person.

Two cues, kept separate end to end:
  * LAB part colour (tracker-owned, no model): two L1-normalised, SOFT-BINNED 3-D LAB histograms, one for an
    upper-body band and one for a lower-body band of the box, using only the central part of the box width.
    Soft (trilinear) binning: each pixel spreads its weight over the two nearest bin centres in every channel, so a
    tiny colour shift (white balance, compression) moves the histogram smoothly instead of flipping whole bins. This
    matters most for neutral kits, whose a/b values sit at the middle of the range.
  * Generic embedding (behind EmbeddingBackend; the default is DINOv2 ViT-S/14, a GENERIC visual embedding, not a
    football or person re-identification model).

Similarities (all bounded to [0, 1]; non-finite inputs raise AppearanceError):
  * lab_similarity       = mean over the two bands of the Bhattacharyya coefficient sum_i sqrt(p_i * q_i).
                           1 = identical colour distributions, 0 = no overlap.
  * embedding_similarity = max(0, cosine(u, v)) of unit-length embeddings (unit length is checked).
  * composite            = w_lab * lab + w_emb * emb   (config weights, sum 1). Reported, never used alone.

identity_support() is the only appearance-level verdict. It needs ALL of: lab >= lab_floor (colour not incompatible),
emb >= embedding_floor (the embedding agrees), composite >= appearance_floor, all values finite, and no crop flag
saying the pixels may belong to someone else (MERGED_SUSPECTED, OCCLUDED_BY_OTHER). A missing cue gives no support.
So colour alone, or the embedding alone, never establishes identity. Contrast against competing people is a later
layer (target state, R1-F): support here is necessary, not sufficient.

Crop quality (documented formula, range [0, 1]):
  quality = inside_fraction * min(1, h_px / crop_good_height_px) * (1 - covered_fraction)
  inside_fraction  = share of the box area inside the image (detector boxes are already clipped, so for them this is
                     1 and border truncation is reported by the TOUCHES_IMAGE_EDGE flag instead);
  covered_fraction = share of the box area covered by other people's boxes (union). The caller MUST pass the other
                     boxes explicitly (possibly an empty list); there is no silent default.
  valid (usable for comparison) needs h_px >= crop_min_height_px and inside_fraction >= crop_min_inside_fraction.
  Flags: LOW_RESOLUTION, MOSTLY_OUTSIDE, TOUCHES_IMAGE_EDGE, OCCLUDED_BY_OTHER, MERGED_SUSPECTED (caller's flag).

Appearance bank: a bounded memory of trusted observations of ONE physical person (the selected target, or later a
competitor). The target bank accepts a sample only from a trusted, uncontested, complete, good-quality observation
that is itself appearance-consistent with what the bank already holds (see AppearanceBank.update_refusals). Nothing
in this module knows who the person is.
"""
from __future__ import annotations

import abc
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import TrackerConfig

# Crop-quality flags
LOW_RESOLUTION = "LOW_RESOLUTION"
MOSTLY_OUTSIDE = "MOSTLY_OUTSIDE"
TOUCHES_IMAGE_EDGE = "TOUCHES_IMAGE_EDGE"
OCCLUDED_BY_OTHER = "OCCLUDED_BY_OTHER"
MERGED_SUSPECTED = "MERGED_SUSPECTED"
EMBEDDING_UNAVAILABLE = "EMBEDDING_UNAVAILABLE"
# Flags meaning "these pixels may belong to someone else": appearance must not support identity.
CONTAMINATION_FLAGS = (MERGED_SUSPECTED, OCCLUDED_BY_OTHER)


class AppearanceError(RuntimeError):
    pass


class EmbeddingModelError(AppearanceError):
    pass


def _finite(*xs: Any) -> bool:
    return all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x)) for x in xs)


# ---------------------------------------------------------------------------------------------
# Crop geometry and quality
# ---------------------------------------------------------------------------------------------
def _box_area(b: Sequence[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def covered_fraction(box: Sequence[float], others: Sequence[Sequence[float]]) -> float:
    """Share of the box area covered by the union of the other boxes (rasterised on the integer pixel grid that
    contains the box)."""
    x1, y1, x2, y2 = (int(np.floor(box[0])), int(np.floor(box[1])), int(np.ceil(box[2])), int(np.ceil(box[3])))
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0 or not len(others):
        return 0.0
    mask = np.zeros((h, w), dtype=bool)
    for o in others:
        ox1 = max(int(np.floor(o[0])), x1) - x1
        oy1 = max(int(np.floor(o[1])), y1) - y1
        ox2 = min(int(np.ceil(o[2])), x2) - x1
        oy2 = min(int(np.ceil(o[3])), y2) - y1
        if ox2 > ox1 and oy2 > oy1:
            mask[oy1:oy2, ox1:ox2] = True
    return float(mask.mean())


@dataclass(frozen=True)
class CropQuality:
    height_px: float
    inside_fraction: float
    covered_fraction: float
    quality: float
    valid: bool
    flags: Tuple[str, ...]

    def as_record(self) -> Dict[str, Any]:
        return {"height_px": self.height_px, "inside_fraction": self.inside_fraction,
                "covered_fraction": self.covered_fraction, "quality": self.quality, "valid": self.valid,
                "flags": list(self.flags)}


def assess_crop(box: Sequence[float], width: int, height: int, cfg: TrackerConfig, *,
                others: Sequence[Sequence[float]], merged_suspected: bool = False) -> CropQuality:
    """Quality of a person box as an appearance sample (formula in the module docstring). `others` is required."""
    if len(box) != 4 or not _finite(*box):
        raise AppearanceError(f"box must be 4 finite numbers, got {box!r}")
    area = _box_area(box)
    cx1, cy1 = max(0.0, box[0]), max(0.0, box[1])
    cx2, cy2 = min(float(width), box[2]), min(float(height), box[3])
    inside = _box_area((cx1, cy1, cx2, cy2)) / area if area > 0 else 0.0
    h_px = max(0.0, cy2 - cy1)
    cov = covered_fraction((cx1, cy1, cx2, cy2), others) if inside > 0 else 0.0
    q = inside * min(1.0, h_px / cfg.get("crop_good_height_px")) * (1.0 - cov)
    flags: List[str] = []
    if h_px < cfg.get("crop_min_height_px"):
        flags.append(LOW_RESOLUTION)
    if inside < cfg.get("crop_min_inside_fraction"):
        flags.append(MOSTLY_OUTSIDE)
    m = cfg.get("detector_edge_touch_margin_px")
    if box[0] <= m or box[1] <= m or box[2] >= width - m or box[3] >= height - m:
        flags.append(TOUCHES_IMAGE_EDGE)
    if cov > cfg.get("crop_max_covered_fraction"):
        flags.append(OCCLUDED_BY_OTHER)
    if merged_suspected:
        flags.append(MERGED_SUSPECTED)
    valid = LOW_RESOLUTION not in flags and MOSTLY_OUTSIDE not in flags
    return CropQuality(height_px=h_px, inside_fraction=inside, covered_fraction=cov, quality=float(q),
                       valid=valid, flags=tuple(flags))


def crop_image(image: np.ndarray, box: Sequence[float]) -> Optional[np.ndarray]:
    """The image pixels inside the box, clipped to the image; None if empty."""
    if len(box) != 4 or not _finite(*box):
        raise AppearanceError(f"box must be 4 finite numbers, got {box!r}")
    h, w = image.shape[:2]
    x1, y1 = max(0, int(round(box[0]))), max(0, int(round(box[1])))
    x2, y2 = min(w, int(round(box[2]))), min(h, int(round(box[3])))
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2]


# ---------------------------------------------------------------------------------------------
# LAB part-colour cue (tracker-owned)
# ---------------------------------------------------------------------------------------------
def _soft_hist3(values: np.ndarray, bins: int) -> np.ndarray:
    """Trilinear soft histogram of N x 3 uint8 values over bins^3 cells, bin centres at (i + 0.5) * 256 / bins.
    Each value splits its unit weight between the two nearest centres per channel (clamped at the ends)."""
    levels = float(np.iinfo(np.uint8).max) + 1.0
    half = 1.0 / 2                                                    # bin centres sit half a bin in
    pos = values.astype(np.float64) * (bins / levels) - half         # continuous bin coordinate per channel
    lo = np.floor(pos)
    frac = pos - lo
    lo = lo.astype(np.int64)
    hist = np.zeros(bins ** 3, dtype=np.float64)
    for dl in (0, 1):
        for da in (0, 1):
            for db in (0, 1):
                idx = np.clip(lo + np.array([dl, da, db]), 0, bins - 1)
                wgt = (np.where(dl, frac[:, 0], 1 - frac[:, 0]) * np.where(da, frac[:, 1], 1 - frac[:, 1])
                       * np.where(db, frac[:, 2], 1 - frac[:, 2]))
                np.add.at(hist, (idx[:, 0] * bins + idx[:, 1]) * bins + idx[:, 2], wgt)
    return hist


def lab_descriptor(crop: np.ndarray, cfg: TrackerConfig) -> Optional[np.ndarray]:
    """Two L1-normalised soft LAB histograms (upper band, lower band), stacked as shape (2, bins**3). None if a band
    has fewer than lab_min_band_pixels pixels. Bands are fractions of the crop height; only the central
    lab_center_width_fraction of the width is used."""
    if crop is None or crop.ndim != 3 or crop.size == 0 or crop.dtype != np.uint8:
        return None
    h, w = crop.shape[:2]
    keep = cfg.get("lab_center_width_fraction")
    cx1 = int(round(w * (1.0 - keep) / 2))
    cx2 = max(cx1 + 1, int(round(w * (1.0 + keep) / 2)))
    lab = cv2.cvtColor(np.ascontiguousarray(crop), cv2.COLOR_BGR2LAB)
    bins = cfg.get("lab_bins_per_channel")
    hists = []
    for band in (cfg.get("lab_upper_band"), cfg.get("lab_lower_band")):
        y1, y2 = int(round(h * band[0])), int(round(h * band[1]))
        region = lab[y1:max(y2, y1 + 1), cx1:cx2].reshape(-1, lab.shape[2])
        if region.shape[0] < cfg.get("lab_min_band_pixels"):
            return None
        hist = _soft_hist3(region, bins)
        hists.append(hist / hist.sum())
    return np.stack(hists, 0)


def lab_similarity(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    """Mean Bhattacharyya coefficient over the bands, in [0, 1]; None if either descriptor is missing."""
    if a is None or b is None:
        return None
    if a.shape != b.shape or not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise AppearanceError("LAB descriptors must have the same shape and be finite")
    bc = np.sqrt(np.clip(a, 0.0, None) * np.clip(b, 0.0, None)).sum(axis=1)
    return float(np.clip(bc.mean(), 0.0, 1.0))


# ---------------------------------------------------------------------------------------------
# Embedding cue (generic interface)
# ---------------------------------------------------------------------------------------------
class EmbeddingBackend(abc.ABC):
    """Generic visual embedding of a person crop. Not an identity model."""

    @abc.abstractmethod
    def embed(self, crop: np.ndarray) -> np.ndarray:
        """Unit-length float64 vector for one BGR uint8 crop."""

    @abc.abstractmethod
    def provenance(self) -> Dict[str, Any]:
        """Backend, model file, sha256, licence status, input size, pooling (JSON-safe)."""

    def model_record(self) -> Optional[Dict[str, Any]]:
        return None


def l2_normalise(v: Any) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).ravel()
    n = float(np.linalg.norm(v))
    if not np.all(np.isfinite(v)) or not math.isfinite(n) or n <= 0:
        raise AppearanceError("cannot normalise a zero or non-finite vector")
    return v / n


def _check_unit(v: np.ndarray, tol: float) -> None:
    if v.ndim != 1 or not np.all(np.isfinite(v)) or abs(float(np.linalg.norm(v)) - 1.0) > tol:
        raise AppearanceError("embedding must be a finite unit-length vector")


def embedding_similarity(a: Optional[np.ndarray], b: Optional[np.ndarray], tol: float) -> Optional[float]:
    """max(0, cosine) of unit-length vectors, in [0, 1]; None if either is missing. Non-unit or non-finite vectors
    raise (a broken backend must not be hidden by clipping)."""
    if a is None or b is None:
        return None
    if a.shape != b.shape:
        raise AppearanceError(f"embedding shapes differ: {a.shape} vs {b.shape}")
    _check_unit(a, tol)
    _check_unit(b, tol)
    return float(np.clip(np.dot(a, b), 0.0, 1.0))


class ScriptedEmbedding(EmbeddingBackend):
    """Test backend: returns pre-written vectors in call order (normalised)."""

    def __init__(self, vectors: Sequence[Sequence[float]]):
        self._v = [l2_normalise(np.asarray(v, dtype=np.float64)) for v in vectors]
        self._i = 0

    def embed(self, crop: np.ndarray) -> np.ndarray:
        if self._i >= len(self._v):
            raise AppearanceError("scripted embedding ran out of vectors")
        v = self._v[self._i]
        self._i += 1
        return v

    def provenance(self) -> Dict[str, Any]:
        return {"backend": "scripted"}


# ---------------------------------------------------------------------------------------------
# Feature, similarity, identity support
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class AppearanceFeature:
    lab: Optional[np.ndarray] = field(repr=False)
    embedding: Optional[np.ndarray] = field(repr=False)
    quality: CropQuality

    @property
    def usable(self) -> bool:
        return self.quality.valid and self.lab is not None and self.embedding is not None


@dataclass(frozen=True)
class AppearanceSimilarity:
    lab: Optional[float]
    embedding: Optional[float]
    composite: Optional[float]

    def as_record(self) -> Dict[str, Any]:
        return {"lab": self.lab, "embedding": self.embedding, "composite": self.composite}


def compare(a: AppearanceFeature, b: AppearanceFeature, cfg: TrackerConfig) -> AppearanceSimilarity:
    tol = cfg.get("embedding_norm_tolerance")
    lab = lab_similarity(a.lab, b.lab)
    emb = embedding_similarity(a.embedding, b.embedding, tol)
    comp = None
    if lab is not None and emb is not None:
        comp = cfg.get("composite_weight_lab") * lab + cfg.get("composite_weight_embedding") * emb
    return AppearanceSimilarity(lab=lab, embedding=emb, composite=comp)


def identity_support(sim: AppearanceSimilarity, cfg: TrackerConfig,
                     flags: Sequence[str] = ()) -> Tuple[bool, Tuple[str, ...]]:
    """Whether appearance alone is CONSISTENT with the same person (necessary, not sufficient: contrast against
    competitors is decided later). `flags` are the crop-quality flags of the observation being judged. Returns
    (supported, reasons-if-not). Non-finite values never support."""
    reasons = []
    if sim.lab is None:
        reasons.append("LAB_MISSING")
    elif not _finite(sim.lab):
        reasons.append("LAB_NON_FINITE")
    elif sim.lab < cfg.get("lab_floor"):
        reasons.append("LAB_INCOMPATIBLE")
    if sim.embedding is None:
        reasons.append("EMBEDDING_MISSING")
    elif not _finite(sim.embedding):
        reasons.append("EMBEDDING_NON_FINITE")
    elif sim.embedding < cfg.get("embedding_floor"):
        reasons.append("EMBEDDING_BELOW_FLOOR")
    if sim.composite is None or not _finite(sim.composite) or sim.composite < cfg.get("appearance_floor"):
        reasons.append("COMPOSITE_BELOW_FLOOR")
    for f in CONTAMINATION_FLAGS:
        if f in flags:
            reasons.append(f"CROP_{f}")
    return (not reasons), tuple(reasons)


class AppearanceExtractor:
    """Builds AppearanceFeature records from an image and a person box."""

    def __init__(self, cfg: TrackerConfig, embedder: Optional[EmbeddingBackend]):
        self._cfg = cfg
        self._embedder = embedder

    def extract(self, image: np.ndarray, box: Sequence[float], *, others: Sequence[Sequence[float]],
                merged_suspected: bool = False) -> AppearanceFeature:
        h, w = image.shape[:2]
        q = assess_crop(box, w, h, self._cfg, others=others, merged_suspected=merged_suspected)
        crop = crop_image(image, box)
        if crop is None or not q.valid:
            return AppearanceFeature(lab=None, embedding=None, quality=q)
        lab = lab_descriptor(crop, self._cfg)
        emb = None
        if self._embedder is not None:
            emb = self._embedder.embed(crop)
            _check_unit(emb, self._cfg.get("embedding_norm_tolerance"))
        else:
            q = CropQuality(q.height_px, q.inside_fraction, q.covered_fraction, q.quality, q.valid,
                            q.flags + (EMBEDDING_UNAVAILABLE,))
        return AppearanceFeature(lab=lab, embedding=emb, quality=q)


# ---------------------------------------------------------------------------------------------
# Appearance bank (trusted memory of one physical person)
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class BankUpdateContext:
    """Facts supplied by the caller (the target-state layer) about the observation offered to the bank."""
    identity_trusted: bool                         # the target layer is confident this box is the person
    merged_suspected: bool = False                 # the box may contain two people
    max_competitor_iou: float = 0.0                # largest IoU between this box and any other person's box
    seconds_since_reacquisition: Optional[float] = None   # None = never reacquired


# Bank decision codes
ACCEPTED = "ACCEPTED"
REDUNDANT = "REDUNDANT"
REFUSED = "REFUSED"
NOT_STORED_CAPACITY = "NOT_STORED_CAPACITY"


class AppearanceBank:
    """Bounded multi-view memory of one person's appearance.

    add() stores a sample only if update_refusals() is empty. The first stored sample is the anchor and is never
    evicted. A sample almost identical to a stored one (embedding similarity >= 1 - bank_min_novelty) is not stored
    (REDUNDANT). When over capacity, the most redundant non-anchor sample (highest embedding similarity to another
    sample) is evicted, which keeps different views (front/back, poses); if that is the new sample itself, the result
    is NOT_STORED_CAPACITY. Outliers cannot enter in the first place because of the self-consistency check.
    """

    def __init__(self, cfg: TrackerConfig, require_trust: bool = True, capacity: Optional[int] = None):
        self._cfg = cfg
        self.require_trust = require_trust
        cap = capacity if capacity is not None else cfg.get("bank_max_samples")
        if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
            raise AppearanceError(f"bank capacity must be an int >= 1, got {cap!r}")
        self.capacity = cap
        self.samples: List[AppearanceFeature] = []
        self.rejections: Dict[str, int] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def update_refusals(self, feat: AppearanceFeature, ctx: BankUpdateContext) -> Tuple[str, ...]:
        c = self._cfg
        q = feat.quality
        r: List[str] = []
        if not _finite(q.quality, q.covered_fraction, q.inside_fraction, q.height_px):
            r.append("NON_FINITE_INPUT")
        if not feat.usable:
            r.append("FEATURE_NOT_USABLE")
        if not (_finite(q.quality) and q.quality >= c.get("bank_min_crop_quality")):
            r.append("LOW_QUALITY")
        for flag in (OCCLUDED_BY_OTHER, MERGED_SUSPECTED, EMBEDDING_UNAVAILABLE, TOUCHES_IMAGE_EDGE):
            if flag in q.flags:
                r.append(flag)
        if not (_finite(q.covered_fraction) and q.covered_fraction <= c.get("bank_max_covered_fraction")):
            if OCCLUDED_BY_OTHER not in r:
                r.append("PARTLY_COVERED_BY_OTHER")
        if ctx.merged_suspected and MERGED_SUSPECTED not in r:
            r.append(MERGED_SUSPECTED)
        if self.require_trust:
            if ctx.identity_trusted is not True:
                r.append("IDENTITY_NOT_TRUSTED")
            if not (_finite(ctx.max_competitor_iou) and ctx.max_competitor_iou <= c.get("bank_contact_iou")):
                r.append("CONTESTED_BY_OTHER_PERSON")
            s = ctx.seconds_since_reacquisition
            if s is not None and not (_finite(s) and s >= c.get("post_reacq_quarantine_s")):
                r.append("POST_REACQUISITION_QUARANTINE")
        if not r and self.samples:
            best, _ = self.best_match(feat)
            if best is None or not identity_support(best, c)[0]:
                r.append("BANK_SELF_INCONSISTENT")
        return tuple(r)

    def add(self, feat: AppearanceFeature, ctx: BankUpdateContext) -> Tuple[str, Tuple[str, ...]]:
        refusals = self.update_refusals(feat, ctx)
        if refusals:
            for x in refusals:
                self.rejections[x] = self.rejections.get(x, 0) + 1
            return REFUSED, refusals
        tol = self._cfg.get("embedding_norm_tolerance")
        if self.samples:
            nearest = max(embedding_similarity(feat.embedding, s.embedding, tol) for s in self.samples)
            if nearest >= 1.0 - self._cfg.get("bank_min_novelty"):
                return REDUNDANT, ()
        self.samples.append(feat)
        if len(self.samples) > self.capacity:
            evicted = self._evict()
            if evicted == len(self.samples):          # the new sample itself was the one removed
                return NOT_STORED_CAPACITY, ()
        return ACCEPTED, ()

    def _evict(self) -> int:
        """Remove the most redundant non-anchor sample; return its index (before removal)."""
        if len(self.samples) == 1:
            return 0
        embs = np.stack([s.embedding for s in self.samples], 0)
        sims = embs @ embs.T
        np.fill_diagonal(sims, -np.inf)
        redundancy = sims.max(axis=1)
        redundancy[0] = -np.inf                        # the anchor is never evicted
        i = int(np.argmax(redundancy))
        del self.samples[i]
        return i if i < len(self.samples) else len(self.samples)

    def best_match(self, feat: AppearanceFeature) -> Tuple[Optional[AppearanceSimilarity], Optional[int]]:
        """Similarity of a feature to the bank: the sample with the highest composite similarity (cue values come
        from that same sample, never mixed across samples). (None, None) if the bank is empty or the feature is not
        usable."""
        if not self.samples or not feat.usable:
            return None, None
        best, best_i = None, None
        for i, s in enumerate(self.samples):
            sim = compare(feat, s, self._cfg)
            if sim.composite is not None and (best is None or sim.composite > best.composite):
                best, best_i = sim, i
        return best, best_i

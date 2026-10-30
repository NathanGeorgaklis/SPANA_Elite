# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R1/R1_DESIGN_DECISIONS.md D05, D06, D07, D08, D10, D11, D12, D13, D14, D15, D16, D18, D19;
#                 R1/R1_GENERIC_TRACKER_REQUIREMENT.md; R1-F instruction (Nathan, 2026-10-25)
"""
player_tracker_rb.target: the identity layer for ONE target person (TARGET) among generic people.

The target is seeded by a human box at the seed frame. From then on, a detection becomes the target's OBSERVED box
only through contrastive identity evidence. Nothing here knows about teams, numbers, colours, clips or frames.

Inputs per frame: the Population's FrameResult for this frame (R1-E: tracklets, observations, competitors, banks)
and the image. update() first lets the population sample its own tracklet banks for this frame (competitor
appearance), then decides the target.

SEED (D18): the max-IoU body detection (>= seed_min_iou) to the human seed box, refused (SeedError, fail closed)
if it is merged-suspected, merge-sized against the seed box itself (merged_area_ratio / merged_aspect_band: the
only single-body size reference on the first frame), has an unusable crop, or cannot anchor the bank.

States (D05): TRACKING (trusted, observed), OCCLUDED, AMBIGUOUS, LOST, REACQUIRING (verifying a candidate).
Only TRACKING is trusted, and only TRACKING has an observed box (D06): every other state writes the target's
PREDICTED or HELD box, and the contested candidate box (if any) separately in observed_or_contested_box.

CANDIDATE EVALUATION (one detection i; D08; no fast path: applied even when it is the only box):
  * MERGED: the box is MERGED_SUSPECTED, or its tracklet carries MERGE_LINEAGE, or (while position is valid and
    the target has a size reference) it is merge-sized against the target's recent median size (merged_area_ratio
    / merged_aspect_band) -> it may hold two people and can never confirm identity (D11).
  * MOTION (veto only): while positional continuity is valid and a motion prior exists, the target motion model
    must find the box physically plausible (motion.py). Motion never adds identity evidence.
  * APPEARANCE: the crop's similarity to the target bank (best single sample) must pass identity_support()
    (lab_floor, embedding_floor, appearance_floor; contaminated crops fail). Reacquisition uses the stricter
    composite floor reacq_appearance_floor.
  * CONTRAST against every competitor that could be this box (Population.competitors around the box:
    observations, hidden tracklets, tracklets closed within contact_memory_s, with their banks). EXCLUDED, so the
    target never competes with itself: the candidate's own tracklet, every tracklet the target was confirmed on
    UNLESS that tracklet is observed this frame on a box outside the candidate's nested group (then it is a separate
    body beside the candidate and competes; R1-F review H1), and boxes of the candidate's own nested group
    (part-body / alternative boxes of the same body).
      contrast = target_similarity - best competitor-bank similarity on this same crop
    must be >= contrastive_margin (ordinary) or the reacquisition margin (below). With no banked competitor the
    contrast is not defined and does not block.
  * UNKNOWN COMPETITOR IN CONTACT: a competitor in contact with the box (either box covers >=
    identity_contact_cover_min of the other) whose appearance is unknown (no bank yet, e.g. provisional) and whose
    size could be the target's (the motion SCALE bound against the target height) blocks the box: contrast cannot
    be verified (D10). A body that cannot physically be the target (a ball-sized fragment) does not compete; crop
    contamination is handled by the crop-quality and bank rules.
  * AMBIGUITY: another box in the gate (logged with role RIVAL, judged by the ordinary criteria) that also looks
    like the target (no motion veto, passes the appearance floor, and is not more similar to another known person
    than to the target) within ambiguity_margin of the candidate's target similarity blocks it (D08-5).

ORDINARY ASSOCIATION (CONTINUATION): the target's current population tracklet was observed this frame. The
tracklet is geometric evidence of one body with no contact since the last frame (R1-E), but its box must still pass
every test above (ordinary floors and margin). Pass -> TRACKING; fail -> AMBIGUOUS with the reasons.

NOT OBSERVED: the target's tracklet is hidden -> OCCLUDED (OCCLUDER_OVERLAP with the occluders' ids if a body
covers >= occluder_cover_min of the predicted box, else NO_CANDIDATE_IN_GATE); a merged box over the predicted
target -> OCCLUDED if the predicted centre is inside it, else AMBIGUOUS (MERGED_DETECTION_SUSPECTED; D11). The
tracklet closed: CONTACT_AMBIGUITY -> AMBIGUOUS (COMPETING_CANDIDATE), MERGED_CONTACT -> D11 as above,
LEFT_FRAME -> LOST (TARGET_OUT_OF_FRAME), UNMATCHED_TIMEOUT -> LOST (AGED_OUT). A closed tracklet is never
continued: the target can then only come back through REACQUISITION.

REACQUISITION (D15; any box that is not the continuing tracklet, after any break):
  * candidates: every non-merged, non-alternative detection whose tracklet is not a KNOWN OTHER (a tracked body
    observed as a separate box in the same frame as the target at the seed, or while the target was confirmed by
    continuation on one tracklet for >= target_known_other_min_s; cleared when a reacquisition is accepted and
    pruned to tracklets the population still knows); while positional continuity is valid and the target
    was confirmed less than lost_long_s ago, only inside the search gate; otherwise anywhere (appearance only);
  * per frame test: as above with reacq_appearance_floor and the margin
        m(a) = min(reacq_contrastive_margin + reacq_margin_growth_per_s * a, reacq_margin_cap)
    where a = seconds since the target was last confirmed (longer absence -> stricter);
  * verification: the best passing candidate is followed by its population tracklet. It is accepted only when
    (i) at least reacq_window_s has elapsed since verification started, (ii) it passed in at least
    reacq_min_hit_fraction of the window's frames, (iii) it passes in the accepting frame, and (iv) no other box
    (including the target's own continuing box, if observed) was within ambiguity_margin of it in that frame. If its tracklet closes, or another candidate becomes clearly
    better, verification restarts. During verification the state is REACQUIRING (untrusted).
  * acceptance: TRACKING, target_episode + 1, a reacquisition event (D07), and a bank quarantine of
    post_reacq_quarantine_s.

AGING (D14): OCCLUDED for longer than max_occluded_s, or AMBIGUOUS for longer than max_ambiguous_s (both counted
from the last confirmation) -> LOST (AGED_OUT); the current tracklet is released (only reacquisition can return).
CAMERA: a continuity break (CUT, UNKNOWN when configured, or an estimator reset) -> LOST (CAMERA_CUT); the motion
state is discarded (position_valid False), and reacquisition is appearance-only. Otherwise the motion state and
recent heights follow the measured camera transform (D16).

SEARCH GATE (D13): radius = min(gate_base_h + gate_growth_h_per_s * u, gate_max_h) target heights around the
predicted centre, u = seconds since the last confirmation; frozen while AMBIGUOUS (gate_grows_while_ambiguous).
h = median height of the target's non-partial observed boxes in the last target_height_window_s (camera-scaled).

APPEARANCE BANK (D12): updated only on a TRACKING frame of a VERIFIED episode (trusted), from a box that the population also considers
clean (not CONTESTED / PARTIAL / TOUCHES_IMAGE_EDGE / DUPLICATE_ABSORBED / IN_MERGED_REGION / MERGE_LINEAGE /
merged, containing no other body's box), isolated from every other body outside its own nested group (IoU <=
bank_contact_iou, covered <= bank_max_covered_fraction), in a camera-verified frame, outside the post-reacquisition
quarantine, at most once per target_bank_sample_interval_s; then AppearanceBank's own quality and
self-consistency checks apply.

IDENTITY CONFIDENCE (identity_confidence_rb, range [0, 1], UNCALIBRATED):
  * seed frame: 1.0 (human-provided identity);
  * TRACKING: 0.5 + 0.5 * min(a, k), with
        a = clip((target_similarity - floor) / (1 - floor), 0, 1)           (appearance margin above its floor)
        k = clip((contrast - margin) / (1 - margin), 0, 1); with no banked competitor k = 1 for continuation of a
            VERIFIED episode (the tracklet itself is geometric evidence of one body), k = 0 otherwise
    so a trusted frame is in [0.5, 1] and the weakest piece of evidence decides;
  * UNVERIFIED EPISODE (R1-G; review H1, L4): an episode is contrast-VERIFIED from the seed (human identification),
    or once a TRACKING frame (or the accepting reacquisition frame) beats a CONFUSABLE competitor by the margin: one
    whose bank similarity to the crop reaches appearance_floor. A reacquisition accepted otherwise opens an
    UNVERIFIED episode: its observed frames stay rb_state TRACKING but identity_trusted = False, reason
    INSUFFICIENT_CONTRAST, confidence min(formula, untrusted_identity_confidence_ceiling); the target bank is not
    updated (EPISODE_UNVERIFIED) and no body is recorded as a known other. Event evidence: contrast_available (any
    banked competitor contrasted) and contrast_verified (a confusable one); decision log: episode_contrast_verified;
  * every untrusted frame: min(untrusted_identity_confidence_ceiling,
        last trusted confidence * 0.5 ** (seconds since confirmation / confidence_half_life_s)).
Separate quantities are reported, never folded into this one: raw detector score, target similarity, best
competitor similarity.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .appearance import (AppearanceBank, AppearanceExtractor, AppearanceFeature, AppearanceSimilarity,
                         BankUpdateContext, identity_support)
from .camera import ESTIMATED, FIRST_FRAME
from .config import TrackerConfig
from .detector import box_iou_matrix, containment
from .motion import DETECTION, HELD, PREDICTED, MotionModel, MotionState, centre
from . import population as P

# states (D05; the same strings as output_schema.RB_STATES)
TRACKING = "TRACKING"
OCCLUDED = "OCCLUDED"
AMBIGUOUS = "AMBIGUOUS"
LOST = "LOST"
REACQUIRING = "REACQUIRING"

# uncertainty reasons (output_schema.UNCERTAINTY_REASONS vocabulary)
OCCLUDER_OVERLAP = "OCCLUDER_OVERLAP"
COMPETING_CANDIDATE = "COMPETING_CANDIDATE"
MERGED_DETECTION_SUSPECTED = "MERGED_DETECTION_SUSPECTED"
REACQUISITION_UNVERIFIED = "REACQUISITION_UNVERIFIED"
BELOW_APPEARANCE_FLOOR = "BELOW_APPEARANCE_FLOOR"
INSUFFICIENT_CONTRAST = "INSUFFICIENT_CONTRAST"
EPISODE_UNVERIFIED = "EPISODE_UNVERIFIED"          # bank refusal: the episode's identity is not contrast-verified
MOTION_VETO = "MOTION_VETO"
NO_CANDIDATE_IN_GATE = "NO_CANDIDATE_IN_GATE"
TARGET_OUT_OF_FRAME = "TARGET_OUT_OF_FRAME"
CAMERA_CUT = "CAMERA_CUT"
LOW_QUALITY_CROP = "LOW_QUALITY_CROP"
AGED_OUT = "AGED_OUT"

# candidate roles and detailed (decision-log) reasons
CONTINUATION = "CONTINUATION"
REACQUISITION = "REACQUISITION"
RIVAL = "RIVAL"          # another in-gate box judged by the ORDINARY criteria only to see if it rivals the target
UNKNOWN_COMPETITOR_IN_CONTACT = "UNKNOWN_COMPETITOR_IN_CONTACT"
RIVAL_CANDIDATE = "RIVAL_CANDIDATE"
MORE_LIKE_A_COMPETITOR = "MORE_LIKE_A_COMPETITOR"

_DETAIL_TO_REASON = {
    "MERGED": MERGED_DETECTION_SUSPECTED, MOTION_VETO: MOTION_VETO,
    "LAB_MISSING": LOW_QUALITY_CROP, "EMBEDDING_MISSING": LOW_QUALITY_CROP, "NON_FINITE": LOW_QUALITY_CROP,
    "LAB_NON_FINITE": LOW_QUALITY_CROP, "EMBEDDING_NON_FINITE": LOW_QUALITY_CROP,
    "CROP_MERGED_SUSPECTED": MERGED_DETECTION_SUSPECTED, "CROP_OCCLUDED_BY_OTHER": OCCLUDER_OVERLAP,
    "FEATURE_NOT_USABLE": LOW_QUALITY_CROP, "EMPTY_TARGET_BANK": LOW_QUALITY_CROP,
    "LAB_INCOMPATIBLE": BELOW_APPEARANCE_FLOOR, "EMBEDDING_BELOW_FLOOR": BELOW_APPEARANCE_FLOOR,
    "COMPOSITE_BELOW_FLOOR": BELOW_APPEARANCE_FLOOR, "BELOW_REACQ_FLOOR": BELOW_APPEARANCE_FLOOR,
    "INSUFFICIENT_CONTRAST": INSUFFICIENT_CONTRAST, UNKNOWN_COMPETITOR_IN_CONTACT: COMPETING_CANDIDATE,
    RIVAL_CANDIDATE: COMPETING_CANDIDATE,
}

# observations whose box the target bank must refuse (population flags)
_BANK_REFUSE_FLAGS = (P.CONTESTED, P.PARTIAL, P.TOUCHES_IMAGE_EDGE, P.DUPLICATE_ABSORBED, P.IN_MERGED_REGION,
                      P.MERGED_SUSPECTED, P.NESTED, P.ALTERNATIVE, P.SIZE_ANOMALY)

Box = Tuple[float, float, float, float]


class SeedError(RuntimeError):
    """The seed cannot start a trustworthy target (fail closed; D18)."""


class TargetError(RuntimeError):
    """Invalid use of the target tracker (fail closed)."""


def _wh(b: Sequence[float]) -> Tuple[float, float]:
    return b[2] - b[0], b[3] - b[1]


def _cover(a: Box, b: Box) -> float:
    """Fraction of a's area inside b."""
    return float(containment(np.asarray([a]), np.asarray([b]))[0, 0])


# ---------------------------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class CandidateEvaluation:
    """The full evidence for one detection considered as the target in one frame (decision log, rule 19)."""
    detection_index: int
    tracklet_id: Optional[str]
    box: Box
    role: str
    in_gate: bool
    distance_h: Optional[float]
    merged: bool
    motion: Optional[Dict[str, Any]]
    quality_flags: Tuple[str, ...]
    target_similarity: Optional[Dict[str, Any]]
    best_competitor: Optional[str]
    best_competitor_similarity: Optional[float]
    contrast: Optional[float]
    margin: float
    unknown_contacts: Tuple[str, ...]
    passed: bool
    details: Tuple[str, ...]

    @property
    def composite(self) -> Optional[float]:
        return None if self.target_similarity is None else self.target_similarity.get("composite")

    def as_record(self) -> Dict[str, Any]:
        return {"detection_index": self.detection_index, "tracklet_id": self.tracklet_id, "box": list(self.box),
                "role": self.role, "in_gate": self.in_gate, "distance_h": self.distance_h, "merged": self.merged,
                "motion": self.motion, "quality_flags": list(self.quality_flags),
                "target_similarity": self.target_similarity, "best_competitor": self.best_competitor,
                "best_competitor_similarity": self.best_competitor_similarity, "contrast": self.contrast,
                "margin": self.margin, "unknown_contacts": list(self.unknown_contacts), "passed": self.passed,
                "details": list(self.details)}


@dataclass(frozen=True)
class ReacquisitionEvent:
    episode: int
    lost_from_frame: int
    uncertain_from_frame: int
    reacq_started_frame: int
    reacq_accepted_frame: int
    uncertain_duration_s: float
    evidence: Dict[str, Any]

    def as_record(self) -> Dict[str, Any]:
        return {"episode": self.episode, "lost_from_frame": self.lost_from_frame,
                "uncertain_from_frame": self.uncertain_from_frame, "reacq_started_frame": self.reacq_started_frame,
                "reacq_accepted_frame": self.reacq_accepted_frame,
                "uncertain_duration_s": self.uncertain_duration_s, "evidence": self.evidence}


@dataclass(frozen=True)
class TargetFrame:
    """The target decision for one frame. The first block is exactly output_schema.V2_TARGET_FIELDS (without
    record_kind); the rest is context for the output writer and the decision log."""
    rb_state: str
    observation_source: str
    bbox_is_observed: bool
    bbox_is_held: bool
    bbox_is_predicted: bool
    held_age_s: float
    last_confirmed_frame: Optional[int]
    time_since_confirmed_s: Optional[float]
    observed_or_contested_box: Optional[Box]
    observation_tracklet_id: Optional[str]
    target_episode: int
    identity_trusted: bool
    identity_confidence_rb: float
    identity_uncertainty_reasons: Tuple[str, ...]
    occluder_track_ids: Tuple[str, ...]
    merged_suspected: bool
    raw_detection_conf: Optional[float]
    reidentified_this_frame: bool
    target_similarity: Optional[float]
    best_competitor_similarity: Optional[float]
    # context
    rel_frame: int
    time_s: float
    bbox: Box
    detection_index: Optional[int]
    predicted_inside_image: bool
    position_valid: bool
    target_height_px: Optional[float]
    decision: Dict[str, Any] = field(compare=False)

    def v2_fields(self) -> Dict[str, Any]:
        """The v2 TARGET block (output_schema.V2_TARGET_FIELDS)."""
        return {"record_kind": "TARGET", "rb_state": self.rb_state, "observation_source": self.observation_source,
                "bbox_is_observed": self.bbox_is_observed, "bbox_is_held": self.bbox_is_held,
                "bbox_is_predicted": self.bbox_is_predicted, "held_age_s": self.held_age_s,
                "last_confirmed_frame": self.last_confirmed_frame,
                "time_since_confirmed_s": self.time_since_confirmed_s,
                "observed_or_contested_box": (None if self.observed_or_contested_box is None
                                              else list(self.observed_or_contested_box)),
                "observation_tracklet_id": self.observation_tracklet_id, "target_episode": self.target_episode,
                "identity_trusted": self.identity_trusted, "identity_confidence_rb": self.identity_confidence_rb,
                "identity_uncertainty_reasons": list(self.identity_uncertainty_reasons),
                "occluder_track_ids": list(self.occluder_track_ids), "merged_suspected": self.merged_suspected,
                "raw_detection_conf": self.raw_detection_conf,
                "reidentified_this_frame": self.reidentified_this_frame,
                "target_similarity": self.target_similarity,
                "best_competitor_similarity": self.best_competitor_similarity}

    def as_record(self) -> Dict[str, Any]:
        rec = self.v2_fields()
        rec.update({"rel_frame": self.rel_frame, "time_s": self.time_s, "bbox": list(self.bbox),
                    "detection_index": self.detection_index, "predicted_inside_image": self.predicted_inside_image,
                    "position_valid": self.position_valid, "target_height_px": self.target_height_px,
                    "decision": self.decision})
        return rec


@dataclass
class _Verification:
    tracklet_id: str
    start_t: float
    start_frame: int
    history: List[Tuple[float, bool]] = field(default_factory=list)


# ---------------------------------------------------------------------------------------------
# The target tracker
# ---------------------------------------------------------------------------------------------

def _seed_geometry(seed_box: Sequence[float], box: Sequence[float]) -> Dict[str, float]:
    """Scale-free geometry of a detection against the human seed box: the fraction of the detection inside the seed
    box, the centre offset in seed-box heights (x, y), height and width ratios detection / seed box, and the gap
    from the detection's bottom edge (feet line) up to the seed box's bottom edge, in seed heights."""
    sx0, sy0, sx1, sy1 = (float(v) for v in seed_box)
    bx0, by0, bx1, by1 = (float(v) for v in box)
    inter = max(0.0, min(sx1, bx1) - max(sx0, bx0)) * max(0.0, min(sy1, by1) - max(sy0, by0))
    area = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    sh = sy1 - sy0
    return {"contain": inter / area if area > 0 else 0.0,
            "dx_h": ((bx0 + bx1) - (sx0 + sx1)) / 2.0 / sh,
            "dy_h": ((by0 + by1) - (sy0 + sy1)) / 2.0 / sh,
            "h_ratio": (by1 - by0) / sh, "w_ratio": (bx1 - bx0) / (sx1 - sx0),
            "bottom_gap_h": (sy1 - by1) / sh}

class TargetTracker:
    def __init__(self, cfg: TrackerConfig, population: P.Population, extractor: AppearanceExtractor):
        self._cfg = cfg
        self._pop = population
        self._ext = extractor
        self._mm = MotionModel(cfg)
        self._bank = AppearanceBank(cfg, require_trust=True)
        self._motion: Optional[MotionState] = None
        self._sizes: List[Tuple[float, float, float]] = []   # (t, w, h px) of non-partial confirmed boxes
        self._anomaly_tracklet: Optional[str] = None   # the target's tracklet while it carries a SIZE_ANOMALY
        self._last_h: Optional[float] = None
        self._state: Optional[str] = None
        self._base_state: str = LOST                        # the non-verifying state under REACQUIRING
        self._base_reasons: Tuple[str, ...] = ()
        self._episode = 1
        # the current episode's identity has been CONTRAST-VERIFIED (beat a confusable competitor; the seed is a human
        # identification and counts as verified). An episode opened by an unverified reacquisition reports observed
        # boxes as untrusted until a TRACKING frame passes that contrast (see the module docstring)
        self._contrast_verified = True
        self._tracklet: Optional[str] = None                 # the target's current population tracklet
        self._target_tracklets: Set[str] = set()             # every tracklet the target was confirmed on
        self._known_others: Set[str] = set()                 # tracklets seen as separate bodies beside the target
        self._tracklet_since_t: Optional[float] = None       # when the target was first confirmed on its tracklet
        self._failed = False                                 # a refused seed leaves the tracker unusable
        self._last_confirm_t: Optional[float] = None
        self._last_confirm_frame: Optional[int] = None
        self._last_trusted_conf = 0.0
        self._uncertain_from: Optional[Tuple[int, float]] = None
        self._lost_from_frame: Optional[int] = None
        self._ambiguous_since_t: Optional[float] = None
        self._verif: Optional[_Verification] = None
        self._reacq_t: Optional[float] = None
        self._last_bank_t: Optional[float] = None
        self._events: List[ReacquisitionEvent] = []
        self._last_frame: Optional[int] = None
        self._position_valid = False
        self._last_box: Optional[Box] = None
        self._prev_t: Optional[float] = None

    # -- public --------------------------------------------------------------------------------------------
    @property
    def bank(self) -> AppearanceBank:
        return self._bank

    @property
    def reacquisition_events(self) -> Tuple[ReacquisitionEvent, ...]:
        return tuple(self._events)

    def state_sizes(self) -> Dict[str, int]:
        """Sizes of everything the target tracker keeps between frames (diagnostics: long-run memory checks)."""
        return {"target_size_history": len(self._sizes), "target_tracklets": len(self._target_tracklets),
                "target_known_others": len(self._known_others), "target_bank_samples": len(self._bank),
                "target_verification_history": len(self._verif.history) if self._verif is not None else 0,
                "target_reacquisition_events": len(self._events)}

    @property
    def current_tracklet(self) -> Optional[str]:
        """The population tracklet the target is following (its own geometric chain), or None once released."""
        return self._tracklet

    @property
    def verifying_tracklet(self) -> Optional[str]:
        """The tracklet under reacquisition verification, or None."""
        return self._verif.tracklet_id if self._verif is not None else None

    @property
    def target_tracklets(self) -> Tuple[str, ...]:
        return tuple(sorted(self._target_tracklets))

    # -- helpers -------------------------------------------------------------------------------------------
    def _check_frame(self, res: P.FrameResult, image: np.ndarray) -> None:
        if not isinstance(res, P.FrameResult) or res is not self._pop.last:
            raise TargetError("update() needs the population's latest FrameResult (call Population.update first)")
        if self._last_frame is not None and res.rel_frame <= self._last_frame:
            raise TargetError("frames must be processed in increasing order")
        if not isinstance(image, np.ndarray) or image.ndim != 3:
            raise TargetError("image must be an (h, w, 3) array")

    def _height(self, t: float) -> Optional[float]:
        win = self._cfg.get("target_height_window_s")
        recent = [hh for (ts, _ww, hh) in self._sizes if t - ts <= win]
        if recent:
            return float(np.median(recent))
        return self._last_h

    def _size_ref(self, t: float) -> Optional[Tuple[float, float]]:
        """The target's ESTABLISHED recent size (median w, h of non-partial confirmed boxes within
        target_height_window_s, spanning at least population_size_ref_min_s), or None."""
        c = self._cfg
        recent = [(ts, ww, hh) for (ts, ww, hh) in self._sizes if t - ts <= c.get("target_height_window_s")]
        if len(recent) < 2 or recent[-1][0] - recent[0][0] < c.get("population_size_ref_min_s"):
            return None
        return float(np.median([x[1] for x in recent])), float(np.median([x[2] for x in recent]))

    def _group(self, res: P.FrameResult, i: int) -> Set[int]:
        """Detection indices of i's nested group (i, the boxes nested in it, and its host chain)."""
        obs = res.observations
        root = i
        while obs[root].nested_in is not None:
            root = obs[root].nested_in
        members = {root}
        changed = True
        while changed:
            changed = False
            for o in obs:
                if o.nested_in is not None and o.nested_in in members and o.index not in members:
                    members.add(o.index)
                    changed = True
        return members

    @staticmethod
    def _unproven_partners(res: P.FrameResult, i: int) -> Set[str]:
        """Tracklets whose distinctness from observation i's own tracklet is unproven (born substantially inside one
        another's box and never seen apart: possibly the same body; population F3b)."""
        tid = res.observations[i].tracklet_id
        v = res.tracklet(tid) if tid else None
        return set(v.unproven_distinct_from) if v is not None else set()

    def _other_boxes(self, res: P.FrameResult, i: int, *, for_bank: bool = False) -> List[Box]:
        """Every other body's box for crop contamination: other observations outside i's nested group, and hidden
        tracklets' predicted boxes except the target's own tracklets. Like a nested part box, a tracklet of unproven
        distinctness from i's own tracklet may be the same body and is not counted (NEW_R2 F3b), except for the bank,
        which never takes a crop that may touch a second person (for_bank=True keeps every other body)."""
        grp = self._group(res, i)
        same = set() if for_bank else self._unproven_partners(res, i)
        boxes = [o.bbox for o in res.observations if o.is_body and o.index not in grp and
                 (o.tracklet_id is None or o.tracklet_id not in same)]
        boxes += [v.box for v in res.tracklets if not v.observed and v.tracklet_id not in self._target_tracklets
                  and v.tracklet_id not in same]
        return boxes

    def _feature(self, cache: Dict[int, AppearanceFeature], res: P.FrameResult, image: np.ndarray,
                 i: int) -> AppearanceFeature:
        if i not in cache:
            o = res.observations[i]
            cache[i] = self._ext.extract(image, o.bbox, others=self._other_boxes(res, i),
                                         merged_suspected=P.MERGED_SUSPECTED in o.flags)
        return cache[i]

    def _merged(self, res: P.FrameResult, i: int) -> bool:
        o = res.observations[i]
        if P.MERGED_SUSPECTED in o.flags:
            return True
        v = res.tracklet(o.tracklet_id) if o.tracklet_id else None
        return v is not None and P.MERGE_LINEAGE in v.flags

    def _reacq_margin(self, absence_s: float) -> float:
        c = self._cfg
        return min(c.get("reacq_contrastive_margin") + c.get("reacq_margin_growth_per_s") * absence_s,
                   c.get("reacq_margin_cap"))

    def _gate(self, t: float) -> float:
        c = self._cfg
        if self._last_confirm_t is None:
            return c.get("gate_base_h")
        ref_t = t
        if not c.get("gate_grows_while_ambiguous") and self._ambiguous_since_t is not None:
            ref_t = self._ambiguous_since_t                 # frozen from the start of the ambiguity
        u = max(0.0, ref_t - self._last_confirm_t)
        return min(c.get("gate_base_h") + c.get("gate_growth_h_per_s") * u, c.get("gate_max_h"))

    def _in_gate(self, box: Box, pred_box: Optional[Box], gate_h: Optional[float], h: float) -> bool:
        if pred_box is None or gate_h is None:
            return True
        pcx, pcy = centre(pred_box)
        bcx, bcy = centre(box)
        if math.hypot(bcx - pcx, bcy - pcy) / h <= gate_h:
            return True
        return float(box_iou_matrix(np.asarray([pred_box]), np.asarray([box]))[0, 0]) > 0

    def _evaluate(self, cache: Dict[int, AppearanceFeature], res: P.FrameResult, image: np.ndarray, i: int, *,
                  role: str, pred_box: Optional[Box], gate_h: Optional[float], h: float,
                  absence_s: float) -> CandidateEvaluation:
        c = self._cfg
        t = res.time_s
        o = res.observations[i]
        details: List[str] = []
        merged = self._merged(res, i)
        # D11 at the target level: a box much larger or wider than the target's own established recent size may
        # hold two people (applied while position/scale continuity holds)
        ref = self._size_ref(t) if (self._position_valid and pred_box is not None) else None
        if not merged and P.SIZE_ANOMALY in o.flags:
            merged = True       # an uncorroborated size change may hold an unseen second body: abstain (NEW_R2)
        if not merged and ref is not None:
            w, hh = _wh(o.bbox)
            lo, hi = c.get("merged_aspect_band")
            aspect_rel = (w / hh) / (ref[0] / ref[1])
            if (w * hh) / (ref[0] * ref[1]) >= c.get("merged_area_ratio") or \
                    (aspect_rel > hi and hh / ref[1] >= c.get("population_partial_height_ratio")):
                merged = True
        if merged:
            details.append("MERGED")
        # position (gate) and motion veto, only while positional continuity is valid
        in_gate, dist_h, mrec = True, None, None
        if pred_box is not None and gate_h is not None:
            pcx, pcy = centre(pred_box)
            bcx, bcy = centre(o.bbox)
            dist_h = math.hypot(bcx - pcx, bcy - pcy) / h
            overlaps = float(box_iou_matrix(np.asarray([pred_box]), np.asarray([o.bbox]))[0, 0]) > 0
            in_gate = dist_h <= gate_h or overlaps
        if pred_box is not None and self._motion is not None and self._position_valid and t > self._motion.t_obs:
            chk = self._mm.check(self._motion, o.bbox, t,
                                 partial=P.PARTIAL in o.flags or P.TOUCHES_IMAGE_EDGE in o.flags)
            mrec = chk.as_record()
            if not chk.plausible:
                details.append(MOTION_VETO)
        # appearance
        feat = self._feature(cache, res, image, i)
        sim: Optional[AppearanceSimilarity] = None
        if not len(self._bank):
            details.append("EMPTY_TARGET_BANK")
        elif not feat.usable:
            details.append("FEATURE_NOT_USABLE")
        else:
            sim, _ = self._bank.best_match(feat)
            ok, why = identity_support(sim, c, feat.quality.flags)
            details.extend(why)
            if role == REACQUISITION and (sim.composite is None or sim.composite < c.get("reacq_appearance_floor")):
                if "COMPOSITE_BELOW_FLOOR" not in details:
                    details.append("BELOW_REACQ_FLOOR")
        # contrast against every competitor that could be this box
        margin = self._reacq_margin(absence_s) if role == REACQUISITION else c.get("contrastive_margin")
        best_ref, best_sim, unknown = None, None, []
        grp = self._group(res, i)
        # the target never competes with itself: its own tracklets are excluded, EXCEPT one observed this frame on
        # a box outside this candidate's group - that is a separate body (the target cannot be in two places), so
        # it competes like anyone else (rule 28)
        seen_elsewhere = {v.tracklet_id for v in res.tracklets
                          if v.observed and v.tracklet_id in self._target_tracklets and v.detection_index not in grp}
        excl_t = (set(self._target_tracklets) - seen_elsewhere) | ({o.tracklet_id} if o.tracklet_id else set())
        comps = self._pop.competitors(o.bbox, h, exclude_tracklet_ids=sorted(excl_t),
                                      exclude_detection_indices=sorted(grp))
        same_body = self._unproven_partners(res, i)
        for cp in comps:
            if cp.kind == P.RECENTLY_CLOSED and cp.closed_age_s is not None and \
                    cp.closed_age_s > c.get("contact_memory_s"):
                continue
            has_bank = cp.bank is not None and len(cp.bank) > 0
            if not has_bank:
                # an unproven partner of this box's tracklet (possibly the same body) is not an unknown person in
                # contact, as a nested part box is not (NEW_R2 F3b); with a bank it still competes below
                if cp.tracklet_id is not None and cp.tracklet_id in same_body:
                    continue
                if cp.position_valid and cp.kind != P.RECENTLY_CLOSED and self._could_be_target(cp.box, h, t):
                    cm = c.get("identity_contact_cover_min")
                    if _cover(o.bbox, cp.box) >= cm or _cover(cp.box, o.bbox) >= cm:
                        unknown.append(cp.ref)
                continue
            if feat.usable:
                cs, _ = cp.bank.best_match(feat)
                if cs is not None and cs.composite is not None and (best_sim is None or cs.composite > best_sim):
                    best_ref, best_sim = cp.ref, cs.composite
        contrast = None
        if sim is not None and sim.composite is not None and best_sim is not None:
            contrast = sim.composite - best_sim
            if contrast < margin:
                details.append("INSUFFICIENT_CONTRAST")
        if unknown:
            details.append(UNKNOWN_COMPETITOR_IN_CONTACT)
        return CandidateEvaluation(
            detection_index=i, tracklet_id=o.tracklet_id, box=o.bbox, role=role, in_gate=in_gate,
            distance_h=dist_h, merged=merged, motion=mrec, quality_flags=tuple(feat.quality.flags),
            target_similarity=None if sim is None else sim.as_record(), best_competitor=best_ref,
            best_competitor_similarity=best_sim, contrast=contrast, margin=margin, unknown_contacts=tuple(unknown),
            passed=not details, details=tuple(details))

    def _could_be_target(self, box: Box, h: float, t: float) -> bool:
        """Scale plausibility: could a box of this height be the target's body now? Same bound as the motion
        SCALE veto (motion_scale_jitter_ratio x motion_veto_max_scale_ratio_per_s ** dt, dt = seconds since the
        target was last confirmed). A body far smaller or larger than the target (a ball-sized fragment, a person
        much nearer the camera) cannot be confused with it."""
        c = self._cfg
        dt = 0.0 if self._last_confirm_t is None else max(0.0, t - self._last_confirm_t)
        k = c.get("motion_scale_jitter_ratio") * c.get("motion_veto_max_scale_ratio_per_s") ** dt
        ratio = _wh(box)[1] / h
        return 1.0 / k <= ratio <= k

    def _looks_like_target(self, ev: CandidateEvaluation) -> bool:
        """A rival target candidate: physically plausible (no motion veto), passes the appearance floors and is
        not more like another known person."""
        if MOTION_VETO in ev.details:
            return False
        floor_fail = {"LAB_MISSING", "EMBEDDING_MISSING", "LAB_NON_FINITE", "EMBEDDING_NON_FINITE",
                      "LAB_INCOMPATIBLE", "EMBEDDING_BELOW_FLOOR", "COMPOSITE_BELOW_FLOOR", "FEATURE_NOT_USABLE",
                      "EMPTY_TARGET_BANK"}
        if ev.composite is None or floor_fail & set(ev.details):
            return False
        return ev.contrast is None or ev.contrast >= 0.0

    def _rivals(self, evals: Sequence[CandidateEvaluation], best: CandidateEvaluation) -> List[CandidateEvaluation]:
        m = self._cfg.get("ambiguity_margin")
        return [e for e in evals if e.detection_index != best.detection_index and e.in_gate
                and self._looks_like_target(e) and best.composite is not None
                and e.composite >= best.composite - m]

    def _confidence_trusted(self, ev: CandidateEvaluation) -> float:
        c = self._cfg
        floor = c.get("reacq_appearance_floor") if ev.role == REACQUISITION else c.get("appearance_floor")
        a = min(1.0, max(0.0, (ev.composite - floor) / (1.0 - floor)))
        if ev.contrast is None:
            # no banked competitor: nothing to confuse with for a CONTINUATION (geometric chain) of a verified
            # episode; but for a REACQUISITION, or any frame of an episode that a reacquisition opened without
            # contrast, nothing positive separates the target from an unseen look-alike -> minimal trust
            k = 1.0 if ev.role == CONTINUATION and self._contrast_verified else 0.0
        else:
            k = min(1.0, max(0.0, (ev.contrast - ev.margin) / (1.0 - ev.margin)))
        return 1.0 / 2 + min(a, k) / 2

    def _confusable_contrast(self, ev: CandidateEvaluation) -> bool:
        """A passed contrast VERIFIES identity only if the competitor could be confused with the target: its bank's
        similarity to this crop reaches appearance_floor. Beating an obviously different person says nothing about
        an unseen look-alike (R1-G review L4)."""
        return ev.contrast is not None and ev.best_competitor_similarity is not None and \
            ev.best_competitor_similarity >= self._cfg.get("appearance_floor")

    def _confidence_untrusted(self, t: float) -> float:
        c = self._cfg
        if self._last_confirm_t is None:
            return 0.0
        decay = (1.0 / 2) ** ((t - self._last_confirm_t) / c.get("confidence_half_life_s"))   # half-life
        return min(c.get("untrusted_identity_confidence_ceiling"), self._last_trusted_conf * decay)

    def _bank_update(self, res: P.FrameResult, image: np.ndarray, i: int,
                     cache: Dict[int, AppearanceFeature]) -> Dict[str, Any]:
        """D12: offer the confirmed box to the target bank if every condition holds."""
        c = self._cfg
        t = res.time_s
        o = res.observations[i]
        refusals: List[str] = [f for f in _BANK_REFUSE_FLAGS if f in o.flags]
        v = res.tracklet(o.tracklet_id) if o.tracklet_id else None
        if v is not None and P.MERGE_LINEAGE in v.flags:
            refusals.append(P.MERGE_LINEAGE)
        grp = self._group(res, i)
        others = self._other_boxes(res, i, for_bank=True)
        # another body's box inside this one (not a part of the same body)
        for other in res.observations:
            if other.is_body and other.index not in grp and \
                    _cover(other.bbox, o.bbox) >= c.get("population_nested_containment"):
                refusals.append(P.CONTAINS_OTHER)
                break
        max_iou = float(box_iou_matrix(np.asarray([o.bbox]), np.asarray(others)).max()) if others else 0.0
        if max_iou > c.get("bank_contact_iou"):
            refusals.append("NOT_ISOLATED")
        seed_frame = self._last_frame is None
        if res.camera.status != ESTIMATED and not res.camera.continuity_broken and not seed_frame:
            refusals.append("CAMERA_UNVERIFIED")
        if self._last_bank_t is not None and t - self._last_bank_t < c.get("target_bank_sample_interval_s"):
            refusals.append("SAMPLE_NOT_DUE")
        if refusals:
            return {"status": "REFUSED", "refusals": sorted(set(refusals))}
        feat = self._feature(cache, res, image, i)
        ctx = BankUpdateContext(identity_trusted=True, merged_suspected=False, max_competitor_iou=max_iou,
                                seconds_since_reacquisition=None if self._reacq_t is None else t - self._reacq_t)
        status, why = self._bank.add(feat, ctx)
        self._last_bank_t = t
        return {"status": status, "refusals": list(why)}

    def _confirm(self, res: P.FrameResult, image: np.ndarray, ev: CandidateEvaluation, conf: float,
                 cache: Dict[int, AppearanceFeature]) -> Dict[str, Any]:
        """Bookkeeping for a TRACKING frame on detection ev.detection_index."""
        t = res.time_s
        o = res.observations[ev.detection_index]
        if self._confusable_contrast(ev):                    # the episode's identity is now contrast-verified
            self._contrast_verified = True
        partial = P.PARTIAL in o.flags or P.TOUCHES_IMAGE_EDGE in o.flags
        if self._motion is None or not self._position_valid:
            self._motion = self._mm.start(o.bbox, t, partial=partial)
        else:
            self._motion = self._mm.update(self._motion, o.bbox, t, partial=partial)
        if not partial:
            self._sizes.append((t,) + _wh(o.bbox))
            self._last_h = _wh(o.bbox)[1]
        self._sizes = [x for x in self._sizes if t - x[0] <= self._cfg.get("target_height_window_s")]
        self._position_valid = True
        if o.tracklet_id != self._tracklet or self._tracklet_since_t is None:
            self._tracklet_since_t = t
        self._tracklet = o.tracklet_id
        # every other tracked body observed in this frame, outside the target's own nested group, is a known
        # different person (two separate boxes at once): it can never become the target by reacquisition. Only
        # recorded at the seed (the human identified the target there) or once the target has been confirmed on
        # the same tracklet for target_known_other_min_s (one trusted frame is not enough to rule a body out for its
        # whole tracklet)
        grp = self._group(res, ev.detection_index)
        seeding = self._last_confirm_t is None              # the seed box is the human's own identification
        settled = seeding or (self._contrast_verified and ev.role == CONTINUATION and
                              t - self._tracklet_since_t >= self._cfg.get("target_known_other_min_s"))
        for other in (res.observations if settled else ()):
            if other.is_body and other.index not in grp and other.tracklet_id and \
                    other.tracklet_id != o.tracklet_id and other.tracklet_id not in self._target_tracklets:
                self._known_others.add(other.tracklet_id)
        if o.tracklet_id:
            self._target_tracklets.add(o.tracklet_id)
        self._last_confirm_t, self._last_confirm_frame = t, res.rel_frame
        self._last_trusted_conf = conf
        self._uncertain_from = None
        self._lost_from_frame = None
        self._ambiguous_since_t = None
        self._verif = None
        if not self._contrast_verified:                      # an unverified identity never shapes the bank
            return {"status": "REFUSED", "refusals": [EPISODE_UNVERIFIED]}
        return self._bank_update(res, image, ev.detection_index, cache)

    def _enter_uncertain(self, res: P.FrameResult, state: str) -> None:
        if self._uncertain_from is None:
            self._uncertain_from = (res.rel_frame, res.time_s)
        if state == LOST and self._lost_from_frame is None:
            self._lost_from_frame = res.rel_frame
        if state == AMBIGUOUS and self._ambiguous_since_t is None:
            self._ambiguous_since_t = res.time_s
        if state != AMBIGUOUS:
            self._ambiguous_since_t = None

    # -- seed ----------------------------------------------------------------------------------------------
    def seed(self, res: P.FrameResult, image: np.ndarray, seed_bbox: Sequence[float]) -> TargetFrame:
        """Start the target from a human seed box at the seed frame (D18). Fails closed (SeedError) if no single,
        clean detection matches the seed, or if its crop cannot anchor the appearance bank."""
        if self._state is not None or self._failed:
            raise TargetError("seed() may be called once, on the first frame")
        self._check_frame(res, image)
        c = self._cfg
        sb = tuple(float(v) for v in seed_bbox)
        if len(sb) != 4 or not all(math.isfinite(v) for v in sb) or not (sb[2] > sb[0] and sb[3] > sb[1]):
            raise SeedError(f"seed box must be 4 finite numbers with positive size, got {seed_bbox!r}")
        bodies = [o for o in res.observations if o.is_body]
        if not bodies:
            raise SeedError("no person detection at the seed frame")
        ious = box_iou_matrix(np.asarray([sb]), np.asarray([o.bbox for o in bodies]))[0]
        geo = [_seed_geometry(sb, o.bbox) for o in bodies]
        k = int(np.argmax(ious))
        if ious[k] >= c.get("seed_min_iou"):
            rule = "IOU"                                     # D18, unchanged
        else:
            # Containment path (NEW_R2): a human seed box looser than the detector's body box. The detection must
            # lie almost wholly inside the seed box, near its centre, and be body-sized against it; and it must be
            # the only detection substantially inside the box (otherwise the box may hold two people).
            ok = [i for i, g in enumerate(geo)
                  if g["contain"] >= c.get("seed_contain_min")
                  and abs(g["dx_h"]) <= c.get("seed_contain_max_centre_offset")
                  and abs(g["dy_h"]) <= c.get("seed_contain_max_centre_offset")
                  and g["h_ratio"] >= c.get("seed_contain_min_height_ratio")
                  and g["w_ratio"] >= c.get("seed_contain_min_width_ratio")
                  and abs(g["bottom_gap_h"]) <= c.get("seed_contain_max_bottom_gap")]
            if not ok:
                raise SeedError(f"no detection overlaps the seed box by IoU >= {c.get('seed_min_iou')} "
                                f"(best {float(ious[k]):.3f}) and none is contained, centred, body-sized and on the "
                                f"seed box's feet line")
            if len(ok) > 1:
                raise SeedError(f"seed box is ambiguous: {len(ok)} detections are contained in it")
            k = ok[0]
            others = [i for i, g in enumerate(geo) if i != k and g["contain"] >= c.get("seed_contain_competitor_max")]
            if others:
                raise SeedError(f"seed box is ambiguous: {len(others)} other detection(s) lie at least "
                                f"{c.get('seed_contain_competitor_max')} inside it")
            rule = "CONTAINMENT"
        o = bodies[k]
        if self._merged(res, o.index):
            raise SeedError("the detection matching the seed is merged-suspected (may hold two people)")
        # the human's seed box is the one trustworthy single-body size reference at the seed frame: a detection
        # that is merge-sized against it (D11 area / aspect test) may hold the target plus someone else
        (dw, dh), (sw, sh) = _wh(o.bbox), _wh(sb)
        if (dw * dh) / (sw * sh) >= c.get("merged_area_ratio") or \
                (dw / dh) / (sw / sh) > c.get("merged_aspect_band")[1]:
            raise SeedError("the detection matching the seed is merge-sized against the seed box "
                            f"(area x{(dw * dh) / (sw * sh):.2f}, aspect x{(dw / dh) / (sw / sh):.2f})")
        cache: Dict[int, AppearanceFeature] = {}
        feat = self._feature(cache, res, image, o.index)
        if not feat.usable:
            raise SeedError(f"the seed crop is not usable: {feat.quality.flags}")
        ev = CandidateEvaluation(o.index, o.tracklet_id, o.bbox, "SEED", True, None, False, None,
                                 tuple(feat.quality.flags), None, None, None, None, 0.0, (), True, ())
        self._state = TRACKING
        bank = self._confirm(res, image, ev, 1.0, cache)
        if bank["status"] != "ACCEPTED":
            self._failed = True                              # never usable after a refused seed (fail closed)
            raise SeedError(f"the seed crop cannot anchor the target appearance bank: {bank['refusals']}")
        self._last_frame = res.rel_frame
        fr = self._frame(res, TRACKING, (), o.index, ev, 1.0, False, {"seed": True, "seed_iou": float(ious[k]),
                                                                     "seed_rule": rule,
                                                                     "seed_containment": geo[k]["contain"],
                                                                     "seed_centre_offset_h": (geo[k]["dx_h"],
                                                                                              geo[k]["dy_h"]),
                                                                     "seed_height_ratio": geo[k]["h_ratio"],
                                                                     "bank": bank, "transition": "SEED"})
        self._prev_t = res.time_s
        return fr

    # -- per frame -----------------------------------------------------------------------------------------
    def update(self, res: P.FrameResult, image: np.ndarray) -> TargetFrame:
        if self._state is None or self._failed:
            raise TargetError("seed() first (and successfully)")
        self._check_frame(res, image)
        c = self._cfg
        t = res.time_s
        self._pop.sample_appearance(res.rel_frame, image, self._ext)
        alive = {v.tracklet_id for v in res.tracklets} | {m.tracklet_id for m in self._pop.recently_closed()}
        self._known_others &= alive                          # bounded: only bodies the population still knows
        self._target_tracklets &= alive | ({self._tracklet} if self._tracklet else set())
        prev_state = self._state
        cache: Dict[int, AppearanceFeature] = {}
        log: Dict[str, Any] = {"from": prev_state}

        # ---- camera ----------------------------------------------------------------------------------------
        cam = res.camera
        cut = cam.continuity_broken or cam.status == FIRST_FRAME
        if cut:
            self._motion = None
            self._position_valid = False
            self._tracklet = None
            self._tracklet_since_t = None
            self._verif = None
            self._sizes, self._last_h = [], None              # scale from another shot means nothing here
            log["camera"] = "CONTINUITY_BROKEN"
        elif self._motion is not None:
            self._motion = self._mm.apply_camera(self._motion, cam)
            if cam.compensated:
                self._sizes = [(ts, ww * cam.scale, hh * cam.scale) for (ts, ww, hh) in self._sizes]
                if self._last_h is not None:
                    self._last_h *= cam.scale
                if self._last_box is not None:
                    self._last_box = cam.warp_box(self._last_box)

        h = self._height(t) or float(np.median([_wh(o.bbox)[1] for o in res.observations] or [1.0]))
        pred = self._mm.predict(self._motion, t) if (self._motion is not None and t > self._motion.t_obs) else None
        pred_box: Optional[Box] = pred.box if pred is not None else None
        absence = 0.0 if self._last_confirm_t is None else t - self._last_confirm_t
        prior = (self._position_valid and pred_box is not None and absence < c.get("lost_long_s"))
        gate_h = self._gate(t) if prior else None
        log.update({"h": h, "gate_h": gate_h, "absence_s": absence, "positional_prior": prior})

        state: Optional[str] = None
        reasons: List[str] = []
        det_idx: Optional[int] = None
        contested_box: Optional[Box] = None
        occluders: Tuple[str, ...] = ()
        merged_flag = False
        chosen: Optional[CandidateEvaluation] = None
        evals: List[CandidateEvaluation] = []
        conf_trusted: Optional[float] = None
        reidentified = False

        if cut:
            state, reasons = LOST, [CAMERA_CUT]
        else:
            views = {v.tracklet_id: v for v in res.tracklets}
            closed = {cl.tracklet_id: cl for cl in res.closed}
            tv = views.get(self._tracklet) if self._tracklet else None
            continuity_broken = False
            if tv is not None and tv.observed:
                if P.SIZE_ANOMALY in res.observations[tv.detection_index].flags:
                    self._anomaly_tracklet = self._tracklet
                elif self._anomaly_tracklet == self._tracklet:
                    # NEW_R2: the target's own tracklet carried a size anomaly (possibly an unseen second body in
                    # its box) that has now ended. Geometry cannot say which body the tracklet holds now, so its
                    # continuity is not identity evidence: the target comes back only through reacquisition (a new
                    # episode, untrusted and never banked until contrast-verified; R1-G)
                    continuity_broken = True
                    self._tracklet, self._anomaly_tracklet = None, None
                    log["continuity_broken"] = "SIZE_ANOMALY_ENDED"
            if continuity_broken:
                state, reasons = AMBIGUOUS, [COMPETING_CANDIDATE]
            elif tv is not None and tv.observed:
                # ---- ordinary association: continuation of the target's tracklet ----------------------------
                ev = self._evaluate(cache, res, image, tv.detection_index, role=CONTINUATION, pred_box=pred_box,
                                    gate_h=gate_h, h=h, absence_s=absence)
                evals.append(ev)
                others = []
                grp = self._group(res, ev.detection_index)
                for o in res.observations:
                    if o.is_body and o.index not in grp and self._in_gate(o.bbox, pred_box, gate_h, h):
                        others.append(self._evaluate(cache, res, image, o.index, role=RIVAL,
                                                     pred_box=pred_box, gate_h=gate_h, h=h, absence_s=absence))
                evals.extend(others)
                details = list(ev.details)
                rivals = self._rivals(others, ev) if ev.passed else []
                if rivals:
                    details.append(RIVAL_CANDIDATE)
                    log["continuation_rivals"] = [e.detection_index for e in rivals]
                if not details:
                    state, chosen = TRACKING, ev
                    conf_trusted = self._confidence_trusted(ev)
                else:
                    state = AMBIGUOUS
                    reasons = sorted({_DETAIL_TO_REASON.get(d, COMPETING_CANDIDATE) for d in details})
                    contested_box, det_idx = ev.box, ev.detection_index
                    merged_flag = ev.merged
            elif tv is not None:
                # ---- the target's tracklet is hidden -----------------------------------------------------------
                state, reasons, occluders, merged_flag = self._unobserved(res, pred_box)
            elif self._tracklet is not None and self._tracklet in closed:
                reason = closed[self._tracklet].reason
                self._tracklet = None
                if reason == P.MERGED_CONTACT:
                    state, reasons, occluders, merged_flag = self._unobserved(res, pred_box)
                    if not merged_flag:
                        state, reasons, merged_flag = AMBIGUOUS, [MERGED_DETECTION_SUSPECTED], True
                elif reason == P.CONTACT_AMBIGUITY:
                    state, reasons = AMBIGUOUS, [COMPETING_CANDIDATE]
                elif reason == P.LEFT_FRAME:
                    state, reasons = LOST, [TARGET_OUT_OF_FRAME]
                else:
                    state, reasons = LOST, [AGED_OUT]
            else:
                # no tracklet of its own: the underlying uncertain state carries on (it ages below)
                state = self._base_state
                reasons = list(self._base_reasons) or [NO_CANDIDATE_IN_GATE]
                if state == OCCLUDED:
                    # refresh what covers the predicted box now (reason and occluder ids must agree)
                    s2, r2, occ2, m2 = self._unobserved(res, pred_box)
                    if s2 in (OCCLUDED, AMBIGUOUS):
                        state, reasons, occluders, merged_flag = s2, r2, occ2, m2

        # ---- aging (D14) ------------------------------------------------------------------------------------
        if state in (OCCLUDED, AMBIGUOUS) and self._last_confirm_t is not None:
            limit = c.get("max_occluded_s") if state == OCCLUDED else c.get("max_ambiguous_s")
            if t - self._last_confirm_t > limit:
                state, reasons = LOST, sorted(set(reasons) | {AGED_OUT})
                self._tracklet = None
        if state == LOST:
            self._tracklet = None                           # only reacquisition can return from LOST
            if pred_box is not None and self._position_valid and \
                    self._pop.inside_fraction(pred_box) < c.get("population_min_inside_fraction"):
                reasons = sorted(set(reasons) | {TARGET_OUT_OF_FRAME})

        # ---- reacquisition (any non-TRACKING frame) ---------------------------------------------------------
        if state != TRACKING:
            self._enter_uncertain(res, state)
            self._base_state, self._base_reasons = state, tuple(reasons)
            acc = self._reacquire(cache, res, image, pred_box if prior else None, gate_h, h, absence, evals)
            log["reacquisition"] = acc["log"]
            if acc["accepted"] is not None:
                ev = acc["accepted"]
                chosen, state, reidentified = ev, TRACKING, True
                self._contrast_verified = self._confusable_contrast(ev)
                conf_trusted = self._confidence_trusted(ev)
                self._episode += 1
                uf_frame, uf_t = self._uncertain_from
                self._events.append(ReacquisitionEvent(
                    episode=self._episode,
                    lost_from_frame=self._lost_from_frame if self._lost_from_frame is not None else uf_frame,
                    uncertain_from_frame=uf_frame, reacq_started_frame=acc["started_frame"],
                    reacq_accepted_frame=res.rel_frame, uncertain_duration_s=t - uf_t,
                    evidence=acc["evidence"]))
                self._reacq_t = t
                self._known_others = set()                   # a new episode: earlier co-observations are reset
            elif self._verif is not None:
                state = REACQUIRING
                reasons = sorted(set(reasons) | {REACQUISITION_UNVERIFIED})
                ve = acc["verifying"]
                if ve is not None:
                    contested_box, det_idx = ve.box, ve.detection_index
            elif acc.get("best_failing") is not None:
                # a box that looks like the target was found but not accepted: say why
                bf = acc["best_failing"]
                reasons = sorted(set(reasons) | {_DETAIL_TO_REASON.get(d, COMPETING_CANDIDATE) for d in bf.details}
                                 | ({COMPETING_CANDIDATE} if not bf.details else set()))
                if contested_box is None:
                    contested_box, det_idx = bf.box, bf.detection_index

        # ---- write ---------------------------------------------------------------------------------------------
        trusted = True
        if state == TRACKING:
            log["bank"] = self._confirm(res, image, chosen, conf_trusted, cache)
            log["episode_contrast_verified"] = self._contrast_verified
            det_idx = chosen.detection_index
            trusted = self._contrast_verified
            conf = conf_trusted if trusted else min(conf_trusted, c.get("untrusted_identity_confidence_ceiling"))
            self._last_trusted_conf = conf
        else:
            conf = self._confidence_untrusted(t)
            if not reasons:
                reasons = [NO_CANDIDATE_IN_GATE]
        log["candidates"] = [e.as_record() for e in evals]
        log["transition"] = f"{prev_state}->{state}"
        self._state = state
        self._last_frame = res.rel_frame
        if self._anomaly_tracklet != self._tracklet:
            # the anomaly marker belongs to the target's current tracklet only: once the target has left it (LOST,
            # cut, closed, reacquired elsewhere) a later return is already a new episode (review round 4 L1)
            self._anomaly_tracklet = None
        frame = self._frame(res, state, tuple(sorted(set(reasons))), det_idx, chosen, conf, reidentified, log,
                            trusted=trusted,
                            contested_box=contested_box, occluders=occluders, merged=merged_flag, pred=pred)
        self._prev_t = t
        return frame

    def _unobserved(self, res: P.FrameResult, pred_box: Optional[Box]) -> Tuple[str, List[str], Tuple[str, ...], bool]:
        """State when the target has no box of its own this frame (D11, D14)."""
        c = self._cfg
        if pred_box is None:
            return LOST, [NO_CANDIDATE_IN_GATE], (), False
        pcx, pcy = centre(pred_box)
        for o in res.observations:
            if P.MERGED_SUSPECTED in o.flags and _cover(pred_box, o.bbox) >= c.get("merged_overlap_min"):
                inside = o.bbox[0] <= pcx <= o.bbox[2] and o.bbox[1] <= pcy <= o.bbox[3]
                return (OCCLUDED if inside else AMBIGUOUS), [MERGED_DETECTION_SUSPECTED], (), True
        occ = []
        own = set(self._target_tracklets) | ({self._verif.tracklet_id} if self._verif is not None else set())
        for o in res.observations:
            if o.is_body and o.tracklet_id not in own and _cover(pred_box, o.bbox) >= c.get("occluder_cover_min"):
                occ.append(o.tracklet_id or o.ref)
        if self._pop.inside_fraction(pred_box) < c.get("population_min_inside_fraction"):
            return LOST, [TARGET_OUT_OF_FRAME], (), False
        if occ:
            return OCCLUDED, [OCCLUDER_OVERLAP], tuple(sorted(occ)), False
        return OCCLUDED, [NO_CANDIDATE_IN_GATE], (), False

    def _reacquire(self, cache: Dict[int, AppearanceFeature], res: P.FrameResult, image: np.ndarray,
                   pred_box: Optional[Box], gate_h: Optional[float], h: float, absence: float,
                   evals: List[CandidateEvaluation]) -> Dict[str, Any]:
        c = self._cfg
        t = res.time_s
        cands: List[CandidateEvaluation] = []
        # the target's own continuing box, if observed this frame, is a rival of every other candidate: a box
        # elsewhere cannot be the target while the target's own tracklet is seen and looks as much like it
        cont = [e for e in evals if e.role == CONTINUATION and self._tracklet is not None
                and e.tracklet_id == self._tracklet]
        excluded_known = []
        for o in res.observations:
            if not o.is_body or P.ALTERNATIVE in o.flags:
                continue
            if o.tracklet_id in self._known_others:
                excluded_known.append(o.tracklet_id)        # seen beside the confirmed target: not the target
                continue
            if self._tracklet is not None and o.tracklet_id == self._tracklet:
                continue                                    # the current (failing) continuation is not re-tested
            if not self._in_gate(o.bbox, pred_box, gate_h, h):
                continue
            cands.append(self._evaluate(cache, res, image, o.index, role=REACQUISITION, pred_box=pred_box,
                                        gate_h=gate_h, h=h, absence_s=absence))
        evals.extend(cands)
        passing = sorted([e for e in cands if e.passed and e.tracklet_id is not None],
                         key=lambda e: (-e.composite, e.detection_index))
        out: Dict[str, Any] = {"accepted": None, "verifying": None, "started_frame": None, "evidence": None,
                               "log": {"candidates": len(cands), "passing": [e.detection_index for e in passing],
                                       "excluded_known_others": sorted(excluded_known)}}
        lookalikes = sorted([e for e in cands if self._looks_like_target(e) or
                             (e.composite is not None and e.composite >= c.get("appearance_floor"))],
                            key=lambda e: (-e.composite, e.detection_index))
        out["best_failing"] = lookalikes[0] if lookalikes and not lookalikes[0].passed else None
        best = passing[0] if passing else None
        rivals = self._rivals(cands + cont, best) if best is not None else []
        hit = best is not None and not rivals
        v = self._verif
        if v is not None:
            vt = res.tracklet(v.tracklet_id)
            if vt is None:                                   # its tracklet closed: restart
                v = None
            elif best is not None and best.tracklet_id != v.tracklet_id:
                cur = next((e for e in cands if e.tracklet_id == v.tracklet_id), None)
                if cur is None or not cur.passed or \
                        best.composite - (cur.composite or 0.0) >= c.get("ambiguity_margin"):
                    v = None                                 # another candidate is clearly better: restart
        if v is None and best is not None and not rivals:
            v = _Verification(best.tracklet_id, t, res.rel_frame)
        if v is None:
            self._verif = None
            out["log"]["rivals"] = [e.detection_index for e in rivals]
            return out
        cur = next((e for e in cands if e.tracklet_id == v.tracklet_id), None)
        cur_hit = cur is not None and cur.passed and hit and best.tracklet_id == v.tracklet_id
        v.history.append((t, cur_hit))
        win = c.get("reacq_window_s")
        v.history = [(ts, ok) for (ts, ok) in v.history if t - ts < win + self._frame_dt(res) / 2]   # bounded
        window = [ok for (_ts, ok) in v.history]
        frac = sum(window) / len(window)
        dt = self._frame_dt(res)
        elapsed = t - v.start_t
        self._verif = v
        out["started_frame"] = v.start_frame
        out["log"].update({"verifying": v.tracklet_id, "elapsed_s": elapsed, "hit_fraction": frac,
                           "rivals": [e.detection_index for e in rivals]})
        if not any(window) and elapsed >= win - dt / 2:
            self._verif = None                               # a whole window without a pass: abandon
            out["log"]["abandoned"] = v.tracklet_id
            return out
        if cur_hit and elapsed >= win - dt / 2 and frac >= c.get("reacq_min_hit_fraction"):
            out["accepted"] = cur
            out["evidence"] = {"tracklet_id": v.tracklet_id, "window_s": win, "hit_fraction": frac,
                               "contrast_available": cur.contrast is not None,
                               "contrast_verified": self._confusable_contrast(cur),
                               "frames_in_window": len(window), "absence_s": absence,
                               "positional_prior": pred_box is not None, "margin": cur.margin,
                               "target_similarity": cur.composite, "best_competitor": cur.best_competitor,
                               "best_competitor_similarity": cur.best_competitor_similarity,
                               "contrast": cur.contrast}
            return out
        out["verifying"] = cur                               # may be None (its tracklet is hidden this frame)
        out["verifying_tracklet"] = v.tracklet_id
        return out

    def _frame_dt(self, res: P.FrameResult) -> float:
        """Time since the previous processed frame (0 on the first)."""
        return res.time_s - self._prev_t if self._prev_t is not None else 0.0

    def _frame(self, res: P.FrameResult, state: str, reasons: Tuple[str, ...], det_idx: Optional[int],
               ev: Optional[CandidateEvaluation], conf: float, reidentified: bool, log: Dict[str, Any], *,
               contested_box: Optional[Box] = None, occluders: Tuple[str, ...] = (), merged: bool = False,
               pred: Any = None, trusted: bool = True) -> TargetFrame:
        t = res.time_s
        if state == TRACKING:
            o = res.observations[det_idx]
            box, src = o.bbox, DETECTION
            held_age = 0.0
            raw = float(o.score)
            tsim = ev.composite if ev is not None else None
            csim = ev.best_competitor_similarity if ev is not None else None
            tracklet = o.tracklet_id
            self._last_box = box
        else:
            if pred is not None:
                box, src = pred.box, pred.source
            else:
                box, src = self._last_box, HELD
            held_age = 0.0 if self._last_confirm_t is None else t - self._last_confirm_t
            raw = None if det_idx is None else float(res.observations[det_idx].score)
            cand = None
            for e in log.get("candidates", []):
                if e["detection_index"] == det_idx:
                    cand = e
            tsim = None if cand is None or cand["target_similarity"] is None else cand["target_similarity"]["composite"]
            csim = None if cand is None else cand["best_competitor_similarity"]
            tracklet = None if det_idx is None else res.observations[det_idx].tracklet_id
        # with no valid position (after a cut) the box is only a placeholder: the target is not known to be outside
        inside = (not self._position_valid and state != TRACKING) or \
            self._pop.inside_fraction(box) >= self._cfg.get("population_min_inside_fraction")
        return TargetFrame(
            rb_state=state, observation_source=src, bbox_is_observed=src == DETECTION, bbox_is_held=src == HELD,
            bbox_is_predicted=src == PREDICTED, held_age_s=held_age, last_confirmed_frame=self._last_confirm_frame,
            time_since_confirmed_s=None if self._last_confirm_t is None else t - self._last_confirm_t,
            observed_or_contested_box=contested_box if state != TRACKING else box,
            observation_tracklet_id=tracklet, target_episode=self._episode,
            identity_trusted=state == TRACKING and trusted, identity_confidence_rb=float(conf),
            identity_uncertainty_reasons=(() if trusted else (INSUFFICIENT_CONTRAST,)) if state == TRACKING else reasons,
            occluder_track_ids=tuple(occluders), merged_suspected=bool(merged), raw_detection_conf=raw,
            reidentified_this_frame=reidentified, target_similarity=tsim, best_competitor_similarity=csim,
            rel_frame=res.rel_frame, time_s=t, bbox=tuple(float(v) for v in box), detection_index=det_idx,
            predicted_inside_image=bool(inside), position_valid=self._position_valid,
            target_height_px=self._height(t), decision=log)

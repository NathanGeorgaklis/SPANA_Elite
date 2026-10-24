# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R1/R1_DESIGN_DECISIONS.md D06, D08, D10, D11, D13, D16; R1/R1_GENERIC_TRACKER_REQUIREMENT.md
#                 (every nearby person competes; no team/number/colour); R1-E instruction (Nathan, 2026-10-24)
"""
player_tracker_rb.population: the generic population of people in the image, as short-lived tracklets.

What a tracklet is: a chain of detections that motion and geometry say is ONE physical body, with no gap during
which another body could have taken its place. It makes no identity claim, has no team, role, number or colour,
and has no promotion stage. Every detected person is a competitor from the frame it is first seen.

Tracklets are deliberately short-lived. Whenever geometry cannot guarantee continuity, the tracklet ENDS and a new
one starts, with the old ids recorded as `possible_predecessors`. Linking identities across those breaks is the
job of the identity layers (R1-F / R1-G), with appearance and contrast, not of the population.

Per frame (update), in order:
  0. Camera. If the frame's CameraMotion breaks continuity (CUT, or UNKNOWN when configured), every live tracklet
     and merged region is closed (CAMERA_CUT / CAMERA_UNKNOWN) and remembered positions are marked invalid.
     Otherwise every remembered box is moved by the ESTIMATED transform (pan/zoom compensation).
  1. Duplicates and nesting. Two detections with IoU >= population_duplicate_iou are one body detected twice: the
     lower-scored one is DUPLICATE_SUPPRESSED (kept in the record, never a tracklet or competitor) and the kept one
     is DUPLICATE_ABSORBED (never an appearance sample). A detection lying >= population_nested_containment inside
     a larger one is NESTED (a part-body or alternative box of the same person, or a body hidden that deep); the
     larger is CONTAINS_OTHER. Nothing is deleted for overlapping: nested boxes stay observations and competitors.
  2. Association (geometry only). A live tracklet may take a detection (nested or not) if IoU(predicted box,
     detection) >= population_match_iou_min and the motion check passes (motion.py: bounded speed and scale; a
     veto, never proof; a box shorter than population_partial_height_ratio x the prediction is checked as partial).
     A HIDDEN tracklet may take one only if it was missed for <= population_max_gap_s AND no other body touched
     its predicted box during the gap. Greedy by IoU, ties broken by tracklet then detection order. Afterwards an
     unassigned nested box, or an unassigned box containing an assigned nested box, is an ALTERNATIVE: it starts no
     tracklet and takes no part in merge tests.
  3. Merged detections (D11). A detection is MERGED_SUSPECTED if
       MULTI_CLAIM      it covers >= merged_overlap_min of the predicted boxes of two or more physically distinct
                        tracklets that have no other detection this frame, each observed within
                        population_max_gap_s and each filling >= merged_claim_min_box_fraction of the detection
                        (two bodies, one box; also catches a neighbour that "disappears" into a box; a small far body
                        behind a large near one is occlusion, not a merge). A hidden tracklet and its possible
                        successor count as one body;
       AREA_RATIO       its area is >= merged_area_ratio x the ESTABLISHED size of the tracklet it continues;
       ASPECT_WIDE      its w/h is wider than merged_aspect_band x that reference's while its height is not reduced;
       PERSISTENT_MERGE it lies in a merged region and fails those size tests against the region's reference;
       UNRESOLVED_MERGE it lies in a merged region that still holds more bodies than have come out of it.
     A size reference is ESTABLISHED only once the tracklet's non-partial sizes span population_size_ref_min_s
     (median over population_size_ref_window_s); a box that merely covers another body is never size-tested
     against that body. A merged detection is never a tracklet: its claimants are closed (MERGED_CONTACT); it
     stays an observation (and a competitor) flagged MERGED_SUSPECTED. Its box becomes a merged region (camera-
     compensated) holding n >= 2 bodies, whose reference is its largest established claimant size. The region is
     RESOLVED (removed) when at least n separate bodies are observed in it, and EXPIRES
     population_merged_region_hold_s after the last size or multi-claim evidence; on the expiry frame it still
     passes its claimants on as possible predecessors. Detections in a region are IN_MERGED_REGION and are never
     appearance samples, because a second body may still be hidden in it.
  4. Contested links. A tracklet-detection link is ambiguous if a rival detection (for that tracklet) or a rival
     tracklet (for that detection) has IoU within population_assoc_ambiguity_margin of it. Every tracklet in the
     connected group is closed (CONTACT_AMBIGUITY) and each of the group's detections starts a new tracklet whose
     possible_predecessors are the closed ones. A crossing therefore never produces a silent swap.
  5. Births. Any other unassigned detection starts a new tracklet. Hidden tracklets whose predicted box it covers
     (>= population_contact_cover_min) are recorded as possible predecessors (and the new id as their possible
     successor). The hidden tracklet is NOT closed: it may be a different person still hidden behind someone.
  6. Unmatched tracklets become HIDDEN with a PREDICTED (then HELD) box, never an observed one. They close when
     the predicted box is mostly outside the image (LEFT_FRAME) or after population_max_unmatched_s
     (UNMATCHED_TIMEOUT). Contact with any other body during the gap is recorded and bars re-association.
  7. Overlap context: every overlapping pair among the frame's bodies (observations and hidden predictions), with
     IoU and both cover fractions.
  8. Appearance-bank eligibility (contamination guard): only an observed, uncontested, unmerged, non-partial,
     non-duplicate tracklet box that contains no other detection and is isolated (max IoU with any other body <= bank_contact_iou and covered
     <= bank_max_covered_fraction), in a frame with a measured or first camera, and at most once per
     population_bank_sample_interval_s. sample_appearance() then applies the AppearanceBank's own quality checks.
     The population never touches the target's bank.

Observed vs inferred: TrackletView.observation_source is DETECTION only when a detection was assigned this frame
(observed_box is then that detection's box); otherwise it is PREDICTED or HELD and observed_box is None.

Determinism: no randomness; every ordering is explicit (ids are issued in a fixed order; ties broken by index).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .appearance import AppearanceBank, AppearanceExtractor, BankUpdateContext, covered_fraction
from .camera import CUT, ESTIMATED, FIRST_FRAME, CameraMotion
from .config import TrackerConfig
from .detector import PersonCandidate, box_iou_matrix, containment
from .motion import DETECTION, MotionModel, MotionState, Prediction, centre

# tracklet status
NEW = "NEW"            # born this frame (observed)
ACTIVE = "ACTIVE"      # continued on a detection this frame (observed)
HIDDEN = "HIDDEN"      # alive, not observed this frame

# close reasons
UNMATCHED_TIMEOUT = "UNMATCHED_TIMEOUT"
LEFT_FRAME = "LEFT_FRAME"
CAMERA_CUT = "CAMERA_CUT"
CAMERA_UNKNOWN = "CAMERA_UNKNOWN"
CONTACT_AMBIGUITY = "CONTACT_AMBIGUITY"
MERGED_CONTACT = "MERGED_CONTACT"

# observation / tracklet flags
DUPLICATE_SUPPRESSED = "DUPLICATE_SUPPRESSED"
DUPLICATE_ABSORBED = "DUPLICATE_ABSORBED"
MERGED_SUSPECTED = "MERGED_SUSPECTED"
PARTIAL = "PARTIAL"
TOUCHES_IMAGE_EDGE = "TOUCHES_IMAGE_EDGE"
CONTESTED = "CONTESTED"
OVERLAPS_OTHER = "OVERLAPS_OTHER"
REASSOCIATED_AFTER_GAP = "REASSOCIATED_AFTER_GAP"
CONTACT_DURING_GAP = "CONTACT_DURING_GAP"
CAMERA_UNCOMPENSATED = "CAMERA_UNCOMPENSATED"
NESTED = "NESTED"                         # lies inside a larger detection (alternative box or hidden body)
ALTERNATIVE = "ALTERNATIVE"               # an alternative box of a nested group: no tracklet, still a competitor
CONTAINS_OTHER = "CONTAINS_OTHER"         # contains a nested detection: never an appearance sample
IN_MERGED_REGION = "IN_MERGED_REGION"     # overlaps where a merged detection was seen within population_max_unmatched_s

# merge reasons
MULTI_CLAIM = "MULTI_CLAIM"
AREA_RATIO = "AREA_RATIO"
ASPECT_WIDE = "ASPECT_WIDE"
PERSISTENT_MERGE = "PERSISTENT_MERGE"
UNRESOLVED_MERGE = "UNRESOLVED_MERGE"

# bank refusals decided by the population (the bank adds its own)
NOT_OBSERVED = "NOT_OBSERVED"
NOT_ISOLATED = "NOT_ISOLATED"
CAMERA_UNVERIFIED = "CAMERA_UNVERIFIED"
SAMPLE_NOT_DUE = "SAMPLE_NOT_DUE"

# contact events
EVENT_AMBIGUOUS = "AMBIGUOUS_ASSOCIATION"
EVENT_MERGED = "MERGED_DETECTION"

# competitor kinds
OBSERVATION = "OBSERVATION"
HIDDEN_BODY = "HIDDEN"
RECENTLY_CLOSED = "RECENTLY_CLOSED"
LAST_SEEN = "LAST_SEEN"

Box = Tuple[float, float, float, float]


class PopulationError(RuntimeError):
    """Invalid input to the population (fail closed)."""


def _box(b: Sequence[float]) -> Box:
    if len(b) != 4 or not all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in b):
        raise PopulationError(f"box must be 4 finite numbers, got {b!r}")
    out = tuple(float(v) for v in b)
    if not (out[2] > out[0] and out[3] > out[1]):
        raise PopulationError(f"box must have positive width and height, got {b!r}")
    return out  # type: ignore[return-value]


def _wh(b: Sequence[float]) -> Tuple[float, float]:
    return b[2] - b[0], b[3] - b[1]


def det_ref(index: int) -> str:
    return f"D{index}"


# ---------------------------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Observation:
    """One detection in this frame, with the population's context. Never carries an identity."""
    index: int
    bbox: Box
    score: float
    touches_image_edge: bool
    tracklet_id: Optional[str]                # the tracklet this box is the observation of (None: none)
    duplicate_of: Optional[int]               # suppressed duplicate of that detection index
    absorbed: Tuple[int, ...]                 # duplicates suppressed into this one
    nested_in: Optional[int]                  # NESTED inside that (larger) detection index
    flags: Tuple[str, ...]
    merge_reasons: Tuple[str, ...]
    claimants: Tuple[str, ...]                # tracklets whose predicted box this box covers (merge test)
    possible_predecessors: Tuple[str, ...]
    max_iou_other: float                      # largest IoU with any other body this frame
    covered_fraction: float                   # fraction of this box covered by other bodies

    @property
    def ref(self) -> str:
        return det_ref(self.index)

    @property
    def is_body(self) -> bool:
        return self.duplicate_of is None

    def as_record(self) -> Dict[str, Any]:
        return {"ref": self.ref, "index": self.index, "bbox": list(self.bbox), "score": self.score,
                "touches_image_edge": self.touches_image_edge, "tracklet_id": self.tracklet_id,
                "duplicate_of": self.duplicate_of, "absorbed": list(self.absorbed), "nested_in": self.nested_in,
                "flags": list(self.flags),
                "merge_reasons": list(self.merge_reasons), "claimants": list(self.claimants),
                "possible_predecessors": list(self.possible_predecessors), "max_iou_other": self.max_iou_other,
                "covered_fraction": self.covered_fraction}


@dataclass(frozen=True)
class TrackletView:
    """A live tracklet as of this frame. `box` is the observed box when observation_source is DETECTION, else the
    predicted/held box; `observed_box` is None unless observed this frame."""
    tracklet_id: str
    status: str
    observation_source: str
    box: Box
    observed_box: Optional[Box]
    detection_index: Optional[int]
    first_frame: int
    first_time_s: float
    last_observed_frame: int
    time_since_observed_s: float
    age_s: float
    provisional: bool
    flags: Tuple[str, ...]
    possible_predecessors: Tuple[str, ...]
    possible_successors: Tuple[str, ...]
    bank_samples: int
    bank_eligible: bool
    bank_refusals: Tuple[str, ...]
    camera_compensated: bool
    is_competitor: bool = True

    @property
    def observed(self) -> bool:
        return self.observation_source == DETECTION

    def as_record(self) -> Dict[str, Any]:
        return {"tracklet_id": self.tracklet_id, "status": self.status,
                "observation_source": self.observation_source, "box": list(self.box),
                "observed_box": None if self.observed_box is None else list(self.observed_box),
                "detection_index": self.detection_index, "first_frame": self.first_frame,
                "first_time_s": self.first_time_s, "last_observed_frame": self.last_observed_frame,
                "time_since_observed_s": self.time_since_observed_s, "age_s": self.age_s,
                "provisional": self.provisional, "flags": list(self.flags),
                "possible_predecessors": list(self.possible_predecessors),
                "possible_successors": list(self.possible_successors), "bank_samples": self.bank_samples,
                "bank_eligible": self.bank_eligible, "bank_refusals": list(self.bank_refusals),
                "camera_compensated": self.camera_compensated, "is_competitor": self.is_competitor}


@dataclass(frozen=True)
class ClosedTracklet:
    tracklet_id: str
    reason: str
    closed_frame: int
    closed_time_s: float
    last_box: Box                    # last known box (camera-compensated while position_valid)
    last_observed_frame: int
    position_valid: bool             # False after a continuity break (cut): the position means nothing now
    possible_successors: Tuple[str, ...]
    bank: AppearanceBank = field(compare=False, repr=False)

    def as_record(self) -> Dict[str, Any]:
        return {"tracklet_id": self.tracklet_id, "reason": self.reason, "closed_frame": self.closed_frame,
                "closed_time_s": self.closed_time_s, "last_box": list(self.last_box),
                "last_observed_frame": self.last_observed_frame, "position_valid": self.position_valid,
                "possible_successors": list(self.possible_successors), "bank_samples": len(self.bank)}


@dataclass(frozen=True)
class ContactEvent:
    kind: str
    tracklets: Tuple[str, ...]       # closed by the event
    detections: Tuple[int, ...]
    births: Tuple[str, ...]          # tracklets started from the event's detections

    def as_record(self) -> Dict[str, Any]:
        return {"kind": self.kind, "tracklets": list(self.tracklets), "detections": list(self.detections),
                "births": list(self.births)}


@dataclass(frozen=True)
class OverlapPair:
    a: str
    b: str
    iou: float
    a_covered_by_b: float
    b_covered_by_a: float

    def as_record(self) -> Dict[str, Any]:
        return {"a": self.a, "b": self.b, "iou": self.iou, "a_covered_by_b": self.a_covered_by_b,
                "b_covered_by_a": self.b_covered_by_a}


@dataclass(frozen=True)
class FrameResult:
    rel_frame: int
    time_s: float
    camera: CameraMotion
    observations: Tuple[Observation, ...]
    tracklets: Tuple[TrackletView, ...]
    closed: Tuple[ClosedTracklet, ...]
    contact_events: Tuple[ContactEvent, ...]
    overlap_pairs: Tuple[OverlapPair, ...]

    def tracklet(self, tracklet_id: str) -> Optional[TrackletView]:
        for t in self.tracklets:
            if t.tracklet_id == tracklet_id:
                return t
        return None

    def as_record(self) -> Dict[str, Any]:
        return {"rel_frame": self.rel_frame, "time_s": self.time_s, "camera": self.camera.as_record(),
                "observations": [o.as_record() for o in self.observations],
                "tracklets": [t.as_record() for t in self.tracklets],
                "closed": [c.as_record() for c in self.closed],
                "contact_events": [e.as_record() for e in self.contact_events],
                "overlap_pairs": [p.as_record() for p in self.overlap_pairs]}


@dataclass(frozen=True)
class Competitor:
    """A body that could be confused with a query box. Provisional or not, every body competes."""
    ref: str                         # detection ref "D<i>" or tracklet id
    kind: str                        # OBSERVATION, HIDDEN or RECENTLY_CLOSED
    tracklet_id: Optional[str]
    box: Box
    box_source: str                  # DETECTION, PREDICTED, HELD or LAST_SEEN
    position_valid: bool
    distance_h: Optional[float]      # centre distance to the query box in h (None if the position is invalid)
    provisional: Optional[bool]
    flags: Tuple[str, ...]
    bank: Optional[AppearanceBank] = field(compare=False, repr=False, default=None)

    def as_record(self) -> Dict[str, Any]:
        return {"ref": self.ref, "kind": self.kind, "tracklet_id": self.tracklet_id, "box": list(self.box),
                "box_source": self.box_source, "position_valid": self.position_valid,
                "distance_h": self.distance_h, "provisional": self.provisional, "flags": list(self.flags),
                "bank_samples": None if self.bank is None else len(self.bank)}


# ---------------------------------------------------------------------------------------------
# Internal mutable state
# ---------------------------------------------------------------------------------------------
@dataclass
class _Tracklet:
    tid: str
    order: int
    motion: MotionState
    first_frame: int
    first_time: float
    last_obs_frame: int
    bank: AppearanceBank
    sizes: List[Tuple[float, float, float]]  # (t, w, h) of NON-partial observations (camera-scaled)
    contact_during_gap: bool = False
    last_sample_t: Optional[float] = None
    predecessors: Tuple[str, ...] = ()
    successors: List[str] = field(default_factory=list)


@dataclass
class _MergedRegion:
    box: Box
    ref_wh: Optional[Tuple[float, float]]  # established single-body size reference (largest claimant), or None
    claimants: Tuple[str, ...]
    n_bodies: int                         # bodies believed inside (>= 2)
    last_evidence_t: float                # last frame with size or multi-claim evidence of a merge here


def _distinct_bodies(tids: Sequence[str], by_tid: Dict[str, "_Tracklet"]) -> int:
    """Number of physically distinct bodies among tracklets: two tracklets linked as possible predecessor and
    successor of each other may be ONE body (a hidden tracklet and its re-detection), so they count once."""
    uf = _UnionFind()
    for a in tids:
        uf.find(a)
        for b in tids:
            if a < b and (b in by_tid[a].predecessors or b in by_tid[a].successors
                          or a in by_tid[b].predecessors or a in by_tid[b].successors):
                uf.union(a, b)
    return len({uf.find(a) for a in tids})


class _UnionFind:
    def __init__(self) -> None:
        self.parent: Dict[Any, Any] = {}

    def find(self, x: Any) -> Any:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: Any, b: Any) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # deterministic root: the smaller key (tracklets "t:<order>", detections "d:<index>")
            lo, hi = sorted((ra, rb))
            self.parent[hi] = lo


# ---------------------------------------------------------------------------------------------
# The population
# ---------------------------------------------------------------------------------------------
class Population:
    def __init__(self, cfg: TrackerConfig, image_width: int, image_height: int):
        if not (isinstance(image_width, int) and isinstance(image_height, int) and image_width > 0
                and image_height > 0):
            raise PopulationError(f"image size must be positive ints, got {image_width!r} x {image_height!r}")
        self._cfg = cfg
        self._w, self._h = image_width, image_height
        self._mm = MotionModel(cfg)
        self._live: List[_Tracklet] = []
        self._regions: List[_MergedRegion] = []
        self._memory: List[ClosedTracklet] = []
        self._counter = 0
        self._order = 0
        self._last: Optional[FrameResult] = None
        self._last_frame: Optional[int] = None
        self._last_t: Optional[float] = None

    # -- ids -------------------------------------------------------------------------------------------------
    def _new_id(self) -> str:
        c = self._cfg
        while True:
            self._counter += 1
            tid = f"{c.get('population_track_id_prefix')}{self._counter:0{c.get('population_track_id_digits')}d}"
            if tid != c.get("target_track_id"):
                return tid

    # -- helpers ---------------------------------------------------------------------------------------------
    def _inside_fraction(self, b: Box) -> float:
        w, h = _wh(b)
        ix = max(0.0, min(float(self._w), b[2]) - max(0.0, b[0]))
        iy = max(0.0, min(float(self._h), b[3]) - max(0.0, b[1]))
        return ix * iy / (w * h)

    def _touches_edge(self, b: Box) -> bool:
        m = self._cfg.get("detector_edge_touch_margin_px")
        return b[0] <= m or b[1] <= m or b[2] >= self._w - m or b[3] >= self._h - m

    def _size_ref(self, tr: _Tracklet, t: float) -> Optional[Tuple[float, float]]:
        """Median (w, h) of the tracklet's non-partial observations within population_size_ref_window_s, or None
        if they do not yet span population_size_ref_min_s (not ESTABLISHED)."""
        c = self._cfg
        tr.sizes = [x for x in tr.sizes if t - x[0] <= c.get("population_size_ref_window_s")]
        if len(tr.sizes) < 2 or tr.sizes[-1][0] - tr.sizes[0][0] < c.get("population_size_ref_min_s"):
            return None
        return float(np.median([x[1] for x in tr.sizes])), float(np.median([x[2] for x in tr.sizes]))

    def _close(self, tr: _Tracklet, reason: str, frame: int, t: float, last_box: Box, position_valid: bool,
               closed: List[ClosedTracklet]) -> None:
        ct = ClosedTracklet(tr.tid, reason, frame, t, last_box, tr.last_obs_frame, position_valid,
                            tuple(tr.successors), tr.bank)
        closed.append(ct)

    def _birth(self, box: Box, frame: int, t: float, partial: bool, predecessors: Sequence[str]) -> _Tracklet:
        self._order += 1
        tr = _Tracklet(tid=self._new_id(), order=self._order,
                       motion=self._mm.start(box, t, partial=partial), first_frame=frame, first_time=t,
                       last_obs_frame=frame, sizes=[] if partial else [(t,) + _wh(box)],
                       bank=AppearanceBank(self._cfg, require_trust=True,
                                           capacity=self._cfg.get("population_bank_max_samples")),
                       predecessors=tuple(sorted(set(predecessors))))
        return tr

    # -- main ------------------------------------------------------------------------------------------------
    def update(self, rel_frame: int, time_s: float, candidates: Sequence[PersonCandidate],
               camera: CameraMotion) -> FrameResult:
        c = self._cfg
        if isinstance(rel_frame, bool) or not isinstance(rel_frame, int):
            raise PopulationError(f"rel_frame must be an int, got {rel_frame!r}")
        if not (isinstance(time_s, (int, float)) and math.isfinite(float(time_s))):
            raise PopulationError(f"time_s must be finite, got {time_s!r}")
        if self._last_frame is not None and not (rel_frame > self._last_frame and time_s > self._last_t):
            raise PopulationError("frames must be processed in strictly increasing frame and time order")
        if not isinstance(camera, CameraMotion):
            raise PopulationError("camera must be a CameraMotion")
        t = float(time_s)
        dets: List[PersonCandidate] = list(candidates)
        for d in dets:
            if not isinstance(d, PersonCandidate):
                raise PopulationError(f"candidates must be PersonCandidate, got {type(d).__name__}")
        boxes: List[Box] = [_box(d.bbox) for d in dets]
        closed: List[ClosedTracklet] = []
        events: List[ContactEvent] = []

        # ---- 0. camera ----------------------------------------------------------------------------------
        if camera.continuity_broken:
            reason = CAMERA_CUT if camera.status == CUT else CAMERA_UNKNOWN
            for tr in self._live:
                self._close(tr, reason, rel_frame, t, tr.motion.ref_box, False, closed)
            self._live, self._regions = [], []
            self._memory = [_invalidate(m) for m in self._memory]
        else:
            for tr in self._live:
                tr.motion = self._mm.apply_camera(tr.motion, camera)
                if camera.compensated:
                    tr.sizes = [(ts, w * camera.scale, h * camera.scale) for ts, w, h in tr.sizes]
            if camera.compensated:
                for rg in self._regions:
                    rg.box = camera.warp_box(rg.box)
                    if rg.ref_wh is not None:
                        rg.ref_wh = (rg.ref_wh[0] * camera.scale, rg.ref_wh[1] * camera.scale)
                self._memory = [_warp_memory(m, camera) for m in self._memory]

        live = list(self._live)
        preds: Dict[str, Prediction] = {tr.tid: self._mm.predict(tr.motion, t) for tr in live}
        pbox = {tid: p.box for tid, p in preds.items()}

        # ---- 1. duplicates ------------------------------------------------------------------------------
        n = len(dets)
        dup_of: List[Optional[int]] = [None] * n
        absorbed: Dict[int, List[int]] = {i: [] for i in range(n)}
        if n:
            iou_dd = box_iou_matrix(np.asarray(boxes), np.asarray(boxes))
            order = sorted(range(n), key=lambda i: (-dets[i].score, i))
            for pos, i in enumerate(order):
                if dup_of[i] is not None:
                    continue
                for j in order[pos + 1:]:
                    if dup_of[j] is None and iou_dd[i, j] >= c.get("population_duplicate_iou"):
                        dup_of[j] = i
                        absorbed[i].append(j)
        bodies = [i for i in range(n) if dup_of[i] is None]
        nested_in: List[Optional[int]] = [None] * n
        if len(bodies) > 1:
            cont_dd = containment(np.asarray([boxes[i] for i in bodies]), np.asarray([boxes[i] for i in bodies]))
            area = {i: _wh(boxes[i])[0] * _wh(boxes[i])[1] for i in bodies}
            for a, j in enumerate(bodies):
                hosts = [(cont_dd[a, b], -i) for b, i in enumerate(bodies)
                         if i != j and area[i] > area[j] and cont_dd[a, b] >= c.get("population_nested_containment")]
                if hosts:
                    nested_in[j] = -max(hosts)[1]
        contains_other = {nested_in[j] for j in bodies if nested_in[j] is not None}

        def group_of(i: int) -> int:
            while nested_in[i] is not None:
                i = nested_in[i]
            return i
        group = {i: group_of(i) for i in bodies}

        # ---- 2. association ----------------------------------------------------------------------------
        edge = {i: self._touches_edge(boxes[i]) for i in range(n)}
        eligible = []
        for tr in live:
            gap = t - tr.motion.t_obs
            hidden = tr.last_obs_frame != self._last_frame
            if hidden and (gap > c.get("population_max_gap_s") or tr.contact_during_gap):
                continue
            eligible.append(tr)
        iou_td: Dict[Tuple[str, int], float] = {}
        if live and bodies:
            m = box_iou_matrix(np.asarray([pbox[tr.tid] for tr in live]), np.asarray([boxes[i] for i in bodies]))
            for a, tr in enumerate(live):
                for b, i in enumerate(bodies):
                    iou_td[(tr.tid, i)] = float(m[a, b])
        valid: Dict[Tuple[str, int], float] = {}
        for tr in eligible:
            for i in bodies:
                v = iou_td.get((tr.tid, i), 0.0)
                if v >= c.get("population_match_iou_min"):
                    short = (boxes[i][3] - boxes[i][1]) < c.get("population_partial_height_ratio") * \
                        (pbox[tr.tid][3] - pbox[tr.tid][1])
                    chk = self._mm.check(tr.motion, boxes[i], t, partial=edge[i] or short)
                    if chk.plausible:
                        valid[(tr.tid, i)] = v
        order_of = {tr.tid: tr.order for tr in live}
        assigned_t: Dict[str, int] = {}
        assigned_d: Dict[int, str] = {}
        for (tid, i), v in sorted(valid.items(), key=lambda kv: (-kv[1], order_of[kv[0][0]], kv[0][1])):
            if tid in assigned_t or i in assigned_d:
                continue
            assigned_t[tid], assigned_d[i] = i, tid
        # A nested box and its container are alternative observations (possibly of one body). An unassigned
        # nested box, or an unassigned box containing an assigned nested box, is an ALTERNATIVE: it starts no
        # tracklet and takes part in no merge test (it stays an observation and a competitor).
        hosts_of_assigned: Set[int] = set()
        for i in assigned_d:
            j = nested_in[i]
            while j is not None:
                hosts_of_assigned.add(j)
                j = nested_in[j]
        alternative = {i for i in bodies if i not in assigned_d
                       and (nested_in[i] is not None or i in hosts_of_assigned)}
        track = [i for i in bodies if i not in alternative]        # observed bodies that are or start tracklets

        # ---- 3. merged detections ----------------------------------------------------------------------
        by_tid = {tr.tid: tr for tr in live}
        cont_pd: Dict[Tuple[str, int], float] = {}
        if live and track:
            cm = containment(np.asarray([pbox[tr.tid] for tr in live]), np.asarray([boxes[i] for i in track]))
            for a, tr in enumerate(live):
                for b, i in enumerate(track):
                    cont_pd[(tr.tid, i)] = float(cm[a, b])
        merge_reasons: Dict[int, List[str]] = {i: [] for i in track}
        claimants: Dict[int, List[str]] = {i: [] for i in track}
        partial: Dict[int, bool] = {}
        match_min = c.get("population_match_iou_min")
        region_of: Dict[int, int] = {}
        in_region: Dict[int, List[int]] = {k: [] for k in range(len(self._regions))}
        if self._regions and track:
            rm = box_iou_matrix(np.asarray([rg.box for rg in self._regions]), np.asarray([boxes[i] for i in track]))
            for b, i in enumerate(track):
                k = int(np.argmax(rm[:, b]))                 # first maximum: deterministic
                if rm[k, b] >= match_min:
                    region_of[i] = k
                    in_region[k].append(i)
        resolved = {k for k, ds in in_region.items() if len(ds) >= self._regions[k].n_bodies}
        hold = c.get("population_merged_region_hold_s")
        # an expired region no longer makes a box merged, but this frame's detections in it still inherit its
        # claimants as possible predecessors (and are refused as appearance samples); it is removed at the end
        expired = {k for k, rg in enumerate(self._regions) if t - rg.last_evidence_t > hold}
        lo, hi = c.get("merged_aspect_band")
        refs = {tr.tid: self._size_ref(tr, t) for tr in live}
        for i in track:
            cl = [tr.tid for tr in live
                  if cont_pd.get((tr.tid, i), 0.0) >= c.get("merged_overlap_min")
                  and assigned_t.get(tr.tid, i) == i]
            claimants[i] = cl
            det_area = _wh(boxes[i])[0] * _wh(boxes[i])[1]
            strong = [tid for tid in cl
                      if t - by_tid[tid].motion.t_obs <= c.get("population_max_gap_s")
                      and cont_pd[(tid, i)] * _wh(pbox[tid])[0] * _wh(pbox[tid])[1]
                      >= c.get("merged_claim_min_box_fraction") * det_area]
            if len(strong) >= 2 and _distinct_bodies(strong, by_tid) >= 2:
                merge_reasons[i].append(MULTI_CLAIM)
            # size reference: only the body this box would continue, or an evidenced merged region. A box that
            # merely covers another (smaller, hidden) body is not compared with it: that is occlusion.
            ref_wh: Optional[Tuple[float, float]] = None
            if i in assigned_d:
                ref_wh = refs[assigned_d[i]]
            elif i in region_of:
                ref_wh = self._regions[region_of[i]].ref_wh
            w, h = _wh(boxes[i])
            is_partial = edge[i]
            if ref_wh is not None:
                rw, rh = ref_wh
                hr = h / rh
                aspect_rel = (w / h) / (rw / rh)
                size_reasons = []
                if (w * h) / (rw * rh) >= c.get("merged_area_ratio"):
                    size_reasons.append(AREA_RATIO)
                if aspect_rel > hi and hr >= c.get("population_partial_height_ratio"):
                    size_reasons.append(ASPECT_WIDE)
                if size_reasons and i in region_of and not cl and i not in assigned_d:
                    size_reasons.append(PERSISTENT_MERGE)
                merge_reasons[i].extend(size_reasons)
                if hr < c.get("population_partial_height_ratio") or aspect_rel < lo:
                    is_partial = True
            if not merge_reasons[i] and i in region_of and region_of[i] not in resolved \
                    and region_of[i] not in expired:
                merge_reasons[i].append(UNRESOLVED_MERGE)
            partial[i] = is_partial
        merged = {i for i in track if merge_reasons[i]}

        # ---- 4. contested links -----------------------------------------------------------------------
        margin = c.get("population_assoc_ambiguity_margin")
        uf = _UnionFind()
        contested_pairs = []
        for tid, i in sorted(assigned_t.items(), key=lambda kv: order_of[kv[0]]):
            if i in merged:
                continue
            v = valid[(tid, i)]
            rival_d = [j for j in bodies if j != i and group[j] != group[i] and (tid, j) in valid
                       and valid[(tid, j)] >= v - margin]
            rival_t = [u for u in order_of if u != tid and (u, i) in valid and valid[(u, i)] >= v - margin]
            if rival_d or rival_t:
                contested_pairs.append(tid)
                uf.union(("t", order_of[tid]), ("d", i))
                for j in rival_d:
                    uf.union(("t", order_of[tid]), ("d", j))
                    if j in assigned_d:
                        uf.union(("d", j), ("t", order_of[assigned_d[j]]))
                for u in rival_t:
                    uf.union(("d", i), ("t", order_of[u]))
                    if u in assigned_t:
                        uf.union(("t", order_of[u]), ("d", assigned_t[u]))
        groups: Dict[Any, Tuple[Set[str], Set[int]]] = {}
        tid_by_order = {tr.order: tr.tid for tr in live}
        for key in list(uf.parent):
            root = uf.find(key)
            g = groups.setdefault(root, (set(), set()))
            if key[0] == "t":
                g[0].add(tid_by_order[key[1]])
            else:
                g[1].add(key[1])
        contested_t: Set[str] = set()
        contested_d: Set[int] = set()
        for ts, ds in groups.values():
            contested_t |= ts
            contested_d |= {d for d in ds if d not in merged}

        # ---- resolve -----------------------------------------------------------------------------------
        closed_now: Set[str] = set()
        predecessors: Dict[int, Set[str]] = {i: set() for i in track}
        # merged detections: never a tracklet; their claimants end
        for i in sorted(merged):
            if i in assigned_d:
                tid = assigned_d.pop(i)
                assigned_t.pop(tid, None)
            ended = []
            for tid in claimants[i]:
                if tid not in closed_now and tid not in contested_t:
                    tr = by_tid[tid]
                    self._close(tr, MERGED_CONTACT, rel_frame, t, pbox[tid], True, closed)
                    closed_now.add(tid)
                    ended.append(tid)
            events.append(ContactEvent(EVENT_MERGED, tuple(ended), (i,), ()))
            evidence = any(r != UNRESOLVED_MERGE for r in merge_reasons[i])
            src = list(claimants[i])
            est = [refs[tid] for tid in src if refs[tid] is not None]
            ref_wh = max(est, key=lambda wh: (wh[0] * wh[1], wh)) if est else None
            if i in region_of:
                rg = self._regions[region_of[i]]
                members = [boxes[j] for j in in_region[region_of[i]]]
                rg.box = (min(b[0] for b in members), min(b[1] for b in members), max(b[2] for b in members),
                          max(b[3] for b in members))
                if evidence:
                    rg.last_evidence_t = t
                    rg.claimants = tuple(sorted(set(rg.claimants) | set(src)))
                    if src:
                        rg.n_bodies = max(rg.n_bodies, _distinct_bodies(src, by_tid))
                    if ref_wh is not None and (rg.ref_wh is None or
                                               ref_wh[0] * ref_wh[1] > rg.ref_wh[0] * rg.ref_wh[1]):
                        rg.ref_wh = ref_wh
            elif evidence:
                n_in = _distinct_bodies(src, by_tid) if src else 0
                self._regions.append(_MergedRegion(boxes[i], ref_wh, tuple(sorted(set(src))), max(2, n_in), t))
        # contested groups: every tracklet in the group ends; every detection starts again
        births: Dict[int, _Tracklet] = {}
        for root in sorted(groups, key=lambda r: (r[0], r[1])):
            ts, ds = groups[root]
            ended = []
            for tid in sorted(ts, key=lambda u: order_of[u]):
                if tid in closed_now:
                    continue
                tr = by_tid[tid]
                last = boxes[assigned_t[tid]] if tid in assigned_t else pbox[tid]
                self._close(tr, CONTACT_AMBIGUITY, rel_frame, t, last, True, closed)
                closed_now.add(tid)
                ended.append(tid)
                assigned_t.pop(tid, None)
            for d in ds:
                if assigned_d.get(d) in ts:
                    assigned_d.pop(d)
                predecessors.setdefault(d, set()).update(ts)
            dd = tuple(sorted(d for d in ds if d not in merged))
            events.append(ContactEvent(EVENT_AMBIGUOUS, tuple(ended), dd, ()))
        # clean continuations
        observed_now: Dict[str, int] = {}
        flags_now: Dict[str, Set[str]] = {}
        for tid, i in sorted(assigned_t.items(), key=lambda kv: order_of[kv[0]]):
            tr = by_tid[tid]
            fl: Set[str] = set()
            if tr.last_obs_frame != self._last_frame:
                fl.add(REASSOCIATED_AFTER_GAP)
            tr.motion = self._mm.update(tr.motion, boxes[i], t, partial=partial[i])
            if not partial[i]:
                tr.sizes.append((t,) + _wh(boxes[i]))
            tr.last_obs_frame = rel_frame
            tr.contact_during_gap = False
            observed_now[tid] = i
            flags_now[tid] = fl
        # births
        hidden_now = [tr for tr in live if tr.tid not in observed_now and tr.tid not in closed_now]
        for i in track:
            if i in merged or i in assigned_d:
                continue
            for tr in hidden_now:
                if cont_pd.get((tr.tid, i), 0.0) >= c.get("population_contact_cover_min"):
                    predecessors[i].add(tr.tid)
            if i in region_of:
                predecessors[i].update(self._regions[region_of[i]].claimants)
            tr = self._birth(boxes[i], rel_frame, t, partial[i], predecessors[i])
            births[i] = tr
            observed_now[tr.tid] = i
            flags_now[tr.tid] = {CONTESTED} if i in contested_d else set()
        for tr in hidden_now:
            for i, b in births.items():
                if tr.tid in b.predecessors and b.tid not in tr.successors:
                    tr.successors.append(b.tid)
        # fill the births into the contact events
        events = [ContactEvent(e.kind, e.tracklets, e.detections,
                               tuple(births[d].tid for d in e.detections if d in births)) for e in events]
        # memory successors
        succ_map: Dict[str, List[str]] = {}
        for i, b in births.items():
            for p in b.predecessors:
                succ_map.setdefault(p, []).append(b.tid)
        closed = [ClosedTracklet(ct.tracklet_id, ct.reason, ct.closed_frame, ct.closed_time_s, ct.last_box,
                                 ct.last_observed_frame, ct.position_valid,
                                 tuple(sorted(set(ct.possible_successors) | set(succ_map.get(ct.tracklet_id, ())))),
                                 ct.bank) for ct in closed]

        # ---- 6. unmatched tracklets -----------------------------------------------------------------------
        new_live: List[_Tracklet] = []
        for tr in live:
            if tr.tid in closed_now:
                continue
            if tr.tid in observed_now:
                new_live.append(tr)
                continue
            p = preds[tr.tid]
            if self._inside_fraction(p.box) < c.get("population_min_inside_fraction"):
                self._close(tr, LEFT_FRAME, rel_frame, t, p.box, True, closed)
                continue
            if p.dt_s > c.get("population_max_unmatched_s"):
                self._close(tr, UNMATCHED_TIMEOUT, rel_frame, t, p.box, True, closed)
                continue
            new_live.append(tr)
        new_live.extend(births[i] for i in sorted(births))
        self._live = new_live
        hidden_live = [tr for tr in new_live if tr.tid not in observed_now]

        # body boxes this frame: observations (non-duplicates) and hidden predictions
        body_refs: List[str] = [det_ref(i) for i in bodies] + [tr.tid for tr in hidden_live]
        body_boxes: List[Box] = [boxes[i] for i in bodies] + [pbox[tr.tid] for tr in hidden_live]
        cover_min = c.get("population_contact_cover_min")
        if body_boxes:
            bb = np.asarray(body_boxes)
            iou_bb = box_iou_matrix(bb, bb)
            cov_bb = containment(bb, bb)
        for k, tr in enumerate(hidden_live):
            row = len(bodies) + k
            for j in range(len(body_boxes)):
                if j != row and cov_bb[row, j] >= cover_min:
                    tr.contact_during_gap = True
                    break

        # merged regions expire when not continued
        self._regions = [rg for k, rg in enumerate(self._regions) if k not in resolved and k not in expired]

        # ---- 7. overlap context --------------------------------------------------------------------------
        pairs: List[OverlapPair] = []
        nb = len(body_boxes)
        for a in range(nb):
            for b in range(a + 1, nb):
                if iou_bb[a, b] > 0 or cov_bb[a, b] > 0 or cov_bb[b, a] > 0:
                    pairs.append(OverlapPair(body_refs[a], body_refs[b], float(iou_bb[a, b]),
                                             float(cov_bb[a, b]), float(cov_bb[b, a])))

        # ---- observations ----------------------------------------------------------------------------------
        det_tracklet = {i: tid for tid, i in observed_now.items()}
        obs: List[Observation] = []
        body_index = {i: k for k, i in enumerate(bodies)}
        for i in range(n):
            fl: List[str] = []
            max_iou, cov = 0.0, 0.0
            if dup_of[i] is not None:
                fl.append(DUPLICATE_SUPPRESSED)
            else:
                k = body_index[i]
                others = [body_boxes[j] for j in range(nb) if j != k]
                max_iou = float(max((iou_bb[k, j] for j in range(nb) if j != k), default=0.0))
                cov = covered_fraction(boxes[i], others) if others else 0.0
                if absorbed[i]:
                    fl.append(DUPLICATE_ABSORBED)
                if nested_in[i] is not None:
                    fl.append(NESTED)
                if i in alternative:
                    fl.append(ALTERNATIVE)
                if i in contains_other:
                    fl.append(CONTAINS_OTHER)
                if i in merged:
                    fl.append(MERGED_SUSPECTED)
                if partial.get(i):
                    fl.append(PARTIAL)
                if i in contested_d:
                    fl.append(CONTESTED)
                if i in region_of:
                    fl.append(IN_MERGED_REGION)
                if max_iou > 0 or cov > 0:
                    fl.append(OVERLAPS_OTHER)
            if edge[i]:
                fl.append(TOUCHES_IMAGE_EDGE)
            preds_i = tuple(sorted(births[i].predecessors)) if i in births else ()
            obs.append(Observation(index=i, bbox=boxes[i], score=float(dets[i].score),
                                   touches_image_edge=bool(dets[i].touches_image_edge),
                                   tracklet_id=det_tracklet.get(i), duplicate_of=dup_of[i],
                                   absorbed=tuple(absorbed[i]), nested_in=nested_in[i], flags=tuple(fl),
                                   merge_reasons=tuple(merge_reasons.get(i, ())),
                                   claimants=tuple(claimants.get(i, ())), possible_predecessors=preds_i,
                                   max_iou_other=max_iou, covered_fraction=cov))

        # ---- 8. views and bank eligibility ---------------------------------------------------------------
        views: List[TrackletView] = []
        cam_ok = camera.status in (ESTIMATED, FIRST_FRAME) or camera.continuity_broken
        interval = c.get("population_bank_sample_interval_s")
        for tr in self._live:
            i = observed_now.get(tr.tid)
            fl = set(flags_now.get(tr.tid, set()))
            refusals: List[str] = []
            if i is None:
                p = preds[tr.tid]
                box, src, obs_box = p.box, p.source, None
                refusals.append(NOT_OBSERVED)
                if tr.contact_during_gap:
                    fl.add(CONTACT_DURING_GAP)
                comp = p.camera_compensated
            else:
                o = obs[i]
                box, src, obs_box = boxes[i], DETECTION, boxes[i]
                comp = True
                if o.max_iou_other > 0 or o.covered_fraction > 0:
                    fl.add(OVERLAPS_OTHER)
                if PARTIAL in o.flags:
                    fl.add(PARTIAL)
                if TOUCHES_IMAGE_EDGE in o.flags:
                    fl.add(TOUCHES_IMAGE_EDGE)
                if DUPLICATE_ABSORBED in o.flags:
                    fl.add(DUPLICATE_ABSORBED)
                if IN_MERGED_REGION in o.flags:
                    fl.add(IN_MERGED_REGION)
                if CONTAINS_OTHER in o.flags:
                    fl.add(CONTAINS_OTHER)
                for flag in (CONTESTED, PARTIAL, TOUCHES_IMAGE_EDGE, DUPLICATE_ABSORBED, IN_MERGED_REGION,
                             CONTAINS_OTHER):
                    if flag in fl:
                        refusals.append(flag)
                if not (o.max_iou_other <= c.get("bank_contact_iou")
                        and o.covered_fraction <= c.get("bank_max_covered_fraction")):
                    refusals.append(NOT_ISOLATED)
                if not cam_ok:
                    refusals.append(CAMERA_UNVERIFIED)
                if tr.last_sample_t is not None and t - tr.last_sample_t < interval:
                    refusals.append(SAMPLE_NOT_DUE)
            if not comp:
                fl.add(CAMERA_UNCOMPENSATED)
            views.append(TrackletView(
                tracklet_id=tr.tid, status=(NEW if tr.first_frame == rel_frame else
                                            (ACTIVE if i is not None else HIDDEN)),
                observation_source=src, box=box, observed_box=obs_box, detection_index=i,
                first_frame=tr.first_frame, first_time_s=tr.first_time, last_observed_frame=tr.last_obs_frame,
                time_since_observed_s=t - tr.motion.t_obs, age_s=t - tr.first_time,
                provisional=(t - tr.first_time) < c.get("population_provisional_s"), flags=tuple(sorted(fl)),
                possible_predecessors=tr.predecessors, possible_successors=tuple(tr.successors),
                bank_samples=len(tr.bank), bank_eligible=not refusals, bank_refusals=tuple(refusals),
                camera_compensated=comp))

        # ---- memory ------------------------------------------------------------------------------------------
        self._memory.extend(closed)
        keep_s = c.get("population_closed_memory_s")
        self._memory = [m for m in self._memory if t - m.closed_time_s <= keep_s]
        res = FrameResult(rel_frame=rel_frame, time_s=t, camera=camera, observations=tuple(obs),
                          tracklets=tuple(views), closed=tuple(closed), contact_events=tuple(events),
                          overlap_pairs=tuple(pairs))
        self._last, self._last_frame, self._last_t = res, rel_frame, t
        return res

    # -- queries ---------------------------------------------------------------------------------------------
    @property
    def last(self) -> Optional[FrameResult]:
        return self._last

    def recently_closed(self) -> Tuple[ClosedTracklet, ...]:
        return tuple(self._memory)

    def bank_of(self, tracklet_id: str) -> Optional[AppearanceBank]:
        for tr in self._live:
            if tr.tid == tracklet_id:
                return tr.bank
        for m in self._memory:
            if m.tracklet_id == tracklet_id:
                return m.bank
        return None

    def competitors(self, box: Sequence[float], h_px: float, *, exclude_tracklet_ids: Sequence[str] = (),
                    exclude_detection_indices: Sequence[int] = ()) -> Tuple[Competitor, ...]:
        """Every body of the last frame that could be confused with `box`: observations (tracked or not, merged
        or not, provisional or not), hidden tracklets, and recently closed tracklets (with their banks). A body is
        included if it overlaps `box` or its centre is within competitor_radius_h x h_px. A recently closed
        tracklet whose position is invalid (after a cut) is always included, with distance_h None."""
        if self._last is None:
            return ()
        q = _box(box)
        if not (isinstance(h_px, (int, float)) and math.isfinite(float(h_px)) and h_px > 0):
            raise PopulationError(f"h_px must be a positive finite number, got {h_px!r}")
        radius = self._cfg.get("competitor_radius_h")
        ex_t, ex_d = set(exclude_tracklet_ids), set(exclude_detection_indices)
        qx, qy = centre(q)
        res = self._last
        views = {v.tracklet_id: v for v in res.tracklets}
        out: List[Competitor] = []

        def near(b: Box) -> Tuple[bool, float]:
            bx, by = centre(b)
            dist = math.hypot(bx - qx, by - qy) / h_px
            ov = float(box_iou_matrix(np.asarray([q]), np.asarray([b]))[0, 0]) > 0
            return ov or dist <= radius, dist

        for o in res.observations:
            if not o.is_body or o.index in ex_d or (o.tracklet_id is not None and o.tracklet_id in ex_t):
                continue
            ok, dist = near(o.bbox)
            if ok:
                v = views.get(o.tracklet_id) if o.tracklet_id else None
                out.append(Competitor(o.ref, OBSERVATION, o.tracklet_id, o.bbox, DETECTION, True, dist,
                                      None if v is None else v.provisional, o.flags,
                                      None if o.tracklet_id is None else self.bank_of(o.tracklet_id)))
        for v in res.tracklets:
            if v.observed or v.tracklet_id in ex_t:
                continue
            ok, dist = near(v.box)
            if ok:
                out.append(Competitor(v.tracklet_id, HIDDEN_BODY, v.tracklet_id, v.box, v.observation_source, True,
                                      dist, v.provisional, v.flags, self.bank_of(v.tracklet_id)))
        for m in self._memory:
            if m.tracklet_id in ex_t:
                continue
            if m.position_valid:
                ok, dist = near(m.last_box)
                if not ok:
                    continue
                out.append(Competitor(m.tracklet_id, RECENTLY_CLOSED, m.tracklet_id, m.last_box, LAST_SEEN, True,
                                      dist, None, (m.reason,), m.bank))
            else:
                out.append(Competitor(m.tracklet_id, RECENTLY_CLOSED, m.tracklet_id, m.last_box, LAST_SEEN, False,
                                      None, None, (m.reason,), m.bank))
        kind_rank = {OBSERVATION: 0, HIDDEN_BODY: 1, RECENTLY_CLOSED: 2}
        out.sort(key=lambda x: (kind_rank[x.kind], math.inf if x.distance_h is None else x.distance_h, x.ref))
        return tuple(out)

    def sample_appearance(self, image: np.ndarray, extractor: AppearanceExtractor) -> Dict[str, Tuple[str, Tuple[str, ...]]]:
        """Offer this frame's bank-eligible tracklet boxes to their own banks. Returns {tracklet_id: (status,
        refusals)} for every tracklet that was offered a sample."""
        res = self._last
        if res is None:
            return {}
        if image.shape[0] != self._h or image.shape[1] != self._w:
            raise PopulationError("image size differs from the population's image size")
        bodies = [o for o in res.observations if o.is_body]
        hidden = [v for v in res.tracklets if not v.observed]
        out: Dict[str, Tuple[str, Tuple[str, ...]]] = {}
        live = {tr.tid: tr for tr in self._live}
        for v in res.tracklets:
            if not v.bank_eligible:
                continue
            i = v.detection_index
            others = [o.bbox for o in bodies if o.index != i] + [h.box for h in hidden]
            o = res.observations[i]
            feat = extractor.extract(image, o.bbox, others=others, merged_suspected=False)
            ctx = BankUpdateContext(identity_trusted=True, merged_suspected=False,
                                    max_competitor_iou=o.max_iou_other, seconds_since_reacquisition=None)
            tr = live[v.tracklet_id]
            out[v.tracklet_id] = tr.bank.add(feat, ctx)
            tr.last_sample_t = res.time_s
        return out


def _invalidate(m: ClosedTracklet) -> ClosedTracklet:
    return ClosedTracklet(m.tracklet_id, m.reason, m.closed_frame, m.closed_time_s, m.last_box,
                          m.last_observed_frame, False, m.possible_successors, m.bank)


def _warp_memory(m: ClosedTracklet, cam: CameraMotion) -> ClosedTracklet:
    if not m.position_valid:
        return m
    return ClosedTracklet(m.tracklet_id, m.reason, m.closed_frame, m.closed_time_s, cam.warp_box(m.last_box),
                          m.last_observed_frame, True, m.possible_successors, m.bank)

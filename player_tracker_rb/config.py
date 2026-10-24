# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R1/R1_DESIGN_DECISIONS.md (D05-D22); R0/R0_CONSTANTS_AND_THRESHOLDS.csv (historical values);
#                 R0/R0_TRACKER_INTERFACE_MAP.md §4-§5 (contract constants); R1/R1_APPROVALS.md (configuration rule)
"""
player_tracker_rb.config: the single home of every behavioural constant in the R1 tracker.

The rule, from the R1 approval:
  Every behavioural threshold lives here. No undocumented magic values inside implementation functions.

Each parameter is a `Param` record holding:
  - name, type, units, meaning, default;
  - source: HISTORICAL_REFERENCE or NEW_R1;
  - evidence_level: the evidence ladder from the charter;
  - validation_status: how far the VALUE has been validated for R1;
  - role: how the value may be used.

Roles:
  BEHAVIOURAL     drives a tracker decision; changing it is a recorded parameter change.
  CONTRACT        fixed by the historical Pass-1 JSON contract; changing it breaks compatibility.
  RUNTIME         execution, provenance or determinism settings; no identity semantics.
  REFERENCE_ONLY  a recovered historical value kept for audit and comparison. Tracker code must
                  never read it. `TrackerConfig.get()` refuses REFERENCE_ONLY names.

Units:
  Time limits are in seconds. They become frames with the video's own fps: see `seconds_to_frames`.
  Image geometry is in units of target height "h", the median height of recent observed target
  boxes: see `h_to_px`. Pixels appear only where the quantity really is about image resolution
  (crop size, optical-flow feature spacing) or where the historical contract fixes it.

Historical exemplars cited in evidence strings: HE1 = frozen 3N Pass-1 output sha256 88246efe..., HE2 = frozen 3N
Pass-1 output sha256 40fa0df1... (fixtures/historical_contract/, HISTORICAL_CONTRACT_HASHES.sha256). They are
contract evidence only; no clip, player or frame knowledge enters any decision (R1/R1_GENERIC_TRACKER_REQUIREMENT.md).

Every NEW_R1 value is UNVALIDATED until synthetic tests (R1) and DEV tracking labels (R2) say otherwise.
Values must never be tuned on event results.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

CONFIG_REGISTRY_VERSION = "r1b-1"

# Unit constants (definitions, not tunable behaviour).
MS_PER_SECOND = 1000.0

# ---------------------------------------------------------------------------------------------
# Vocabularies used by the registry itself
# ---------------------------------------------------------------------------------------------
SOURCES = ("HISTORICAL_REFERENCE", "NEW_R1")
EVIDENCE_LEVELS = ("VERIFIED_CODE", "VERIFIED_OUTPUT", "VERIFIED_TEST", "DOCUMENTED", "INFERRED",
                   "UNKNOWN", "DESIGN_CHOICE")
VALIDATION_STATUSES = ("UNVALIDATED", "SYNTHETIC_TESTED", "BENCHMARK_DEV_CHECKED", "R2_VALIDATED",
                       "CONTRACT_VERIFIED", "NOT_APPLICABLE")
ROLES = ("BEHAVIOURAL", "CONTRACT", "RUNTIME", "REFERENCE_ONLY")


@dataclass(frozen=True)
class Param:
    """One configuration parameter and everything known about it."""
    name: str
    type: str                 # "float" | "int" | "bool" | "str" | "pair" | "mapping" | "optional_int"
    units: str
    meaning: str
    default: Any
    source: str               # SOURCES
    evidence_level: str       # EVIDENCE_LEVELS
    validation_status: str    # VALIDATION_STATUSES
    role: str                 # ROLES
    group: str
    evidence: str = ""        # where the value or rule comes from
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed: Optional[Tuple[Any, ...]] = None
    historical_value: Any = None   # set when a NEW_R1 parameter has a recorded historical counterpart

    def as_record(self) -> Dict[str, Any]:
        return {
            "name": self.name, "type": self.type, "units": self.units, "meaning": self.meaning,
            "default": _jsonable(self.default), "source": self.source,
            "evidence_level": self.evidence_level, "validation_status": self.validation_status,
            "role": self.role, "group": self.group, "evidence": self.evidence,
            "min_value": self.min_value, "max_value": self.max_value,
            "allowed": None if self.allowed is None else [_jsonable(a) for a in self.allowed],
            "historical_value": _jsonable(self.historical_value),
        }


def _jsonable(v: Any) -> Any:
    if isinstance(v, tuple):
        return [_jsonable(x) for x in v]
    if isinstance(v, Mapping):
        return {str(k): _jsonable(x) for k, x in sorted(v.items(), key=lambda kv: str(kv[0]))}
    return v


def _p(name, type_, units, meaning, default, source, evidence_level, validation_status, role, group,
       evidence="", min_value=None, max_value=None, allowed=None, historical_value=None) -> Param:
    return Param(name, type_, units, meaning, default, source, evidence_level, validation_status, role,
                 group, evidence, min_value, max_value, allowed, historical_value)


N, H = "NEW_R1", "HISTORICAL_REFERENCE"
UNV = "UNVALIDATED"

# ---------------------------------------------------------------------------------------------
# The registry. Ordered by group. Each name is unique (enforced at import time).
# ---------------------------------------------------------------------------------------------
PARAMS: Tuple[Param, ...] = (
    # ---- contract (historical Pass-1 JSON, kept bit-for-bit in meaning) -----------------------
    _p("legacy_identity_trust_threshold", "float", "confidence (0-1)",
       "actor_attribution trusts a frame only if identity_confidence >= this (MEDIUM_THRESH).",
       0.5, H, "VERIFIED_CODE", "CONTRACT_VERIFIED", "CONTRACT", "contract",
       "event_engine_v1/actor_attribution.py:112-114; football_analysis/confidence.py:12 MEDIUM_THRESH", 0.0, 1.0),
    _p("detection_confidence_by_position_source", "mapping", "quantised confidence",
       "The historical detection_confidence field is quantised by position_source; it is NOT a detector score.",
       MappingProxyType({"CONFIRMED_DETECTION": 1.0, "AMBIGUOUS_DETECTION": 0.5,
                         "WEAK_SPATIAL_RECOVERY": 0.5, "PREDICTED": 0.0}),
       H, "VERIFIED_OUTPUT", "CONTRACT_VERIFIED", "CONTRACT", "contract",
       "historical exemplars HE1 (88246efe) and HE2 (40fa0df1): every frame_log record except state REIDENTIFYING "
       "(see detection_confidence_reidentifying). R0 erratum E1: R0 said 'all records'."),
    _p("detection_confidence_reidentifying", "float", "quantised confidence",
       "Historical detection_confidence on REIDENTIFYING records, which carry position_source "
       "CONFIRMED_DETECTION but were written 0.0 (11 of 11 such records in the frozen exemplars).",
       0.0, H, "VERIFIED_OUTPUT", "CONTRACT_VERIFIED", "CONTRACT", "contract",
       "HE1 (1 record) and HE2 (10 records); found by the R1-A contract tests", 0.0, 1.0),
    _p("box_round_dp", "int", "decimal places",
       "Rounding for bbox and observed_position_box coordinates.", 1, H, "VERIFIED_OUTPUT",
       "CONTRACT_VERIFIED", "CONTRACT", "contract", "HE1 and HE2: every box coordinate has 1 dp", 0, 6),
    _p("confidence_round_dp", "int", "decimal places",
       "Rounding for historical confidence fields.", 3, H, "VERIFIED_OUTPUT", "CONTRACT_VERIFIED",
       "CONTRACT", "contract", "HE2: confidence fields have at most 3 dp", 0, 6),
    _p("camera_shift_round_dp", "int", "decimal places", "Rounding for frame_log camera_shift [dx, dy] px.",
       1, H, "VERIFIED_OUTPUT", "CONTRACT_VERIFIED", "CONTRACT", "contract", "HE1 and HE2 camera_shift", 0, 6),
    _p("state_pct_round_dp", "int", "decimal places", "Rounding for summary.selected_state_pct values.",
       1, H, "VERIFIED_OUTPUT", "CONTRACT_VERIFIED", "CONTRACT", "contract", "HE1 summary selected_state_pct (1 dp values)", 0, 6),
    _p("timestamp_round_dp", "int", "decimal places",
       "Rounding for timestamps (event_adapter uses round(frame / fps, 4)).", 4, H, "VERIFIED_CODE",
       "CONTRACT_VERIFIED", "CONTRACT", "contract", "player_tracker_mo/event_adapter.py:129,144", 0, 9),
    _p("target_track_id", "str", "id", "track_id / identity_id / online_track_id of the one logical target.",
       "T001", N, "DESIGN_CHOICE", "NOT_APPLICABLE", "CONTRACT", "contract",
       "D07 (approved by Nathan); historical seed track was T001 in both frozen JSONs"),
    _p("population_track_id_prefix", "str", "id prefix",
       "Prefix for non-target population tracklet ids (historical T### form).", "T", H, "VERIFIED_OUTPUT",
       "CONTRACT_VERIFIED", "CONTRACT", "contract", "HE1 track ids T001..T035"),
    _p("track_source_label", "str", "label", "Historical per-track 'source' field value.", "ONLINE", H,
       "VERIFIED_OUTPUT", "CONTRACT_VERIFIED", "CONTRACT", "contract", "HE1 and HE2: source == ONLINE in every record"),
    _p("untrusted_identity_confidence_ceiling", "float", "confidence (0-1)",
       "When identity_trusted is false, the historical identity_confidence is forced to at most this value, "
       "so that surviving actor_attribution treats the frame as untrusted.",
       0.49, N, "DESIGN_CHOICE", UNV, "CONTRACT", "contract", "D19", 0.0, 1.0),

    # ---- video / provenance (D18, D20) -------------------------------------------------------
    _p("resume_allowed", "bool", "flag", "Checkpoint/resume of a Pass-1 run. Must stay False.", False, N,
       "DOCUMENTED", "NOT_APPLICABLE", "RUNTIME", "video",
       "F15: a .mov resume landed +10 frames late (TASK3N STALE_VS_FRESH_REGISTRATION)", allowed=(False,)),
    _p("seed_min_iou", "float", "IoU ratio",
       "The run is refused unless a raw detection at the seed frame overlaps the seed box by at least this.",
       0.5, N, "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "video", "D18", 0.0, 1.0),
    _p("frame_thumbnail_size", "pair", "px (w, h)",
       "Size of the grey thumbnail whose SHA-1 is stored per frame for registration checks.", (64, 36), N,
       "DESIGN_CHOICE", "NOT_APPLICABLE", "RUNTIME", "video", "D20"),
    _p("max_frames", "optional_int", "frames",
       "Optional cap on processed frames after the seed; None processes to the end of the video.", None, N,
       "DESIGN_CHOICE", "NOT_APPLICABLE", "RUNTIME", "video",
       "historical default 450 (run_mo); R1 processes the whole clip unless capped", 1, None,
       historical_value=450),
    _p("seek_mode", "str", "name",
       "How the reader reaches start_frame. decode_from_zero decodes and discards frames 0..start_frame-1 "
       "(no container seek at all); container_seek does ONE CAP_PROP_POS_FRAMES seek and then verifies the "
       "reported position. Either way, decoding after start_frame is strictly sequential with no further seeks.",
       "decode_from_zero", N, "DESIGN_CHOICE", "NOT_APPLICABLE", "RUNTIME", "video",
       "D20; F15 (a container-level resume on .mov landed +10 frames late); the R1 label-sheet tool "
       "already decodes from frame 0 for this reason", allowed=("decode_from_zero", "container_seek")),
    _p("hash_chunk_bytes", "int", "bytes", "Read size used when hashing the source video with SHA-256.", 1048576,
       N, "DESIGN_CHOICE", "NOT_APPLICABLE", "RUNTIME", "video", "R1-B provenance (performance only)", 4096,
       268435456),
    _p("fingerprint_interpolation", "str", "OpenCV interpolation name",
       "Resize method for the frame-content fingerprint thumbnail (debugging/alignment only).", "INTER_AREA", N,
       "DESIGN_CHOICE", "NOT_APPLICABLE", "RUNTIME", "video", "R1-B frame fingerprint (R1/R1B/FRAME_FINGERPRINT.md)",
       allowed=("INTER_AREA", "INTER_LINEAR", "INTER_NEAREST")),
    _p("timing_drift_tolerance_frames", "float", "frames",
       "If the container's reported timestamp differs from video_frame_index / fps by more than this many "
       "frame periods, the frame is counted as a timing-drift warning (variable frame rate suspected).", 0.5, N,
       "DESIGN_CHOICE", UNV, "RUNTIME", "video", "R1-B video-time mapping (constant-fps assumption check)",
       0.0, 1000.0),

    # ---- detector (D03, D17) -----------------------------------------------------------------
    _p("detector_backend", "str", "name", "Person detector backend.", "yolox_onnx", N, "DESIGN_CHOICE",
       "NOT_APPLICABLE", "RUNTIME", "detector", "D03; R1 approval §2 (YOLOX-s DEVELOPMENT/EVALUATION)",
       allowed=("yolox_onnx", "scripted")),
    _p("detector_input_size", "int", "px (square network input)",
       "Detector network input size. To be chosen on development footage by person-detection quality only (D17).", 640, N,
       "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "detector", "D17; R1 approval §3", 320, 1920,
       allowed=(640, 960, 1280)),
    _p("detector_min_score", "float", "detector score (0-1)",
       "Minimum raw person score kept. Low on purpose: partial bodies must still appear as competitors.",
       0.10, N, "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "detector",
       "TASK3M: a partially visible target body scored 0.47-0.59; historical Pass-2 conf 0.15 (HISTORICAL_REFERENCE)",
       0.0, 1.0, historical_value=0.15),
    _p("detector_nms_iou", "float", "IoU ratio", "Non-maximum-suppression IoU for person boxes.", 0.65, N,
       "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "detector", "D03", 0.0, 1.0),
    _p("detection_min_height_h", "float", "h",
       "Detections shorter than this many target heights are ignored as competitors (noise).", 0.25, N,
       "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "detector", "D13", 0.0, 2.0),

    # ---- runtime determinism (D20, D21) ------------------------------------------------------
    _p("onnx_intra_op_threads", "int", "threads", "ONNX Runtime intra-op thread count (fixed for determinism).",
       1, N, "DESIGN_CHOICE", "NOT_APPLICABLE", "RUNTIME", "runtime", "D20", 1, 64),
    _p("onnx_inter_op_threads", "int", "threads", "ONNX Runtime inter-op thread count.", 1, N,
       "DESIGN_CHOICE", "NOT_APPLICABLE", "RUNTIME", "runtime", "D20", 1, 64),
    _p("onnx_sequential_execution", "bool", "flag", "ONNX Runtime sequential execution mode.", True, N,
       "DESIGN_CHOICE", "NOT_APPLICABLE", "RUNTIME", "runtime", "D20", allowed=(True,)),

    # ---- appearance (D04, D12) ---------------------------------------------------------------
    _p("appearance_backend", "str", "name", "Appearance backend.", "composite_lab_dinov2", N, "DESIGN_CHOICE",
       "NOT_APPLICABLE", "RUNTIME", "appearance", "D04; R1 approval §4",
       allowed=("composite_lab_dinov2", "lab_only", "scripted")),
    _p("composite_weight_lab", "float", "weight",
       "Weight of the LAB part-colour similarity in the composite similarity.", 0.35, N, "DESIGN_CHOICE", UNV,
       "BEHAVIOURAL", "appearance", "D04: colour is supporting evidence only", 0.0, 1.0),
    _p("composite_weight_embedding", "float", "weight",
       "Weight of the generic embedding similarity (DINOv2, NOT player ReID) in the composite.", 0.65, N,
       "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "appearance", "D04", 0.0, 1.0),
    _p("lab_upper_band", "pair", "fraction of box height (top, bottom)",
       "Vertical band of the box used for the upper-body (shirt) colour descriptor.", (0.15, 0.50), N,
       "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "appearance", "D04", 0.0, 1.0),
    _p("lab_lower_band", "pair", "fraction of box height (top, bottom)",
       "Vertical band used for the lower-body (shorts) colour descriptor.", (0.50, 0.80), N,
       "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "appearance", "D04", 0.0, 1.0),
    _p("lab_bins_per_channel", "int", "bins", "Histogram bins per LAB channel.", 8, N, "DESIGN_CHOICE", UNV,
       "BEHAVIOURAL", "appearance", "D04", 2, 64),
    _p("crop_min_height_px", "int", "px (image resolution)",
       "Crops shorter than this give an invalid appearance feature (too few pixels to describe).", 24, N,
       "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "appearance", "D12 crop quality", 4, 512),
    _p("crop_min_inside_fraction", "float", "fraction of box area",
       "Minimum fraction of the box inside the image for a valid appearance feature.", 0.6, N,
       "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "appearance", "D12 crop quality", 0.0, 1.0),

    # ---- identity decision (D08, D09) --------------------------------------------------------
    _p("appearance_floor", "float", "composite similarity (0-1)",
       "A box may become the target's observed box only if its similarity to the target bank is >= this.",
       0.55, N, "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "identity",
       "D08 test 1. Historical floor 0.30 was a cosine on yolo26n-reid: different scale, not comparable",
       0.0, 1.0, historical_value=0.30),
    _p("contrastive_margin", "float", "similarity difference",
       "Target similarity minus the best competitor similarity on the same box must be >= this.", 0.08, N,
       "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "identity", "D08 test 2; R1 approval §18", 0.0, 1.0),
    _p("ambiguity_margin", "float", "similarity difference",
       "The best target candidate must beat the second-best target candidate by >= this.", 0.06, N,
       "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "identity",
       "D08 test 5. Historical assignment margin 0.08 (a near-miss of 0.081 decided a historical identity swap, F1)",
       0.0, 1.0, historical_value=0.08),
    _p("contact_memory_s", "float", "s",
       "How long a competitor that overlapped the target keeps its pre-contact appearance for contrast.",
       2.0, N, "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "identity", "D08", 0.0, 30.0),

    # ---- geometry and motion (D13, R1 approval §19: motion is a veto only) -------------------
    _p("target_height_window_s", "float", "s",
       "Window of recent observed target boxes whose median height defines h.", 1.0, N, "DESIGN_CHOICE", UNV,
       "BEHAVIOURAL", "geometry", "D13", 0.0, 30.0),
    _p("gate_base_h", "float", "h", "Search-gate radius around the predicted target centre when just observed.",
       0.75, N, "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "geometry",
       "D13. Historical base_gate 90 px (unscaled)", 0.0, 20.0, historical_value=90.0),
    _p("gate_growth_h_per_s", "float", "h per s", "Gate growth while the target is unobserved (not while AMBIGUOUS).",
       2.0, N, "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "geometry",
       "D13. Historical gate_growth 35 px per frame", 0.0, 50.0, historical_value=35.0),
    _p("gate_max_h", "float", "h", "Cap on the search-gate radius.", 3.0, N, "DESIGN_CHOICE", UNV,
       "BEHAVIOURAL", "geometry", "D13. Historical max_gate 350 px", 0.0, 50.0, historical_value=350.0),
    _p("gate_grows_while_ambiguous", "bool", "flag", "Whether the gate grows during AMBIGUOUS. Must stay False.",
       False, N, "DOCUMENTED", "NOT_APPLICABLE", "BEHAVIOURAL", "geometry",
       "F6: unbounded gate growth during AMBIGUOUS led to a relock 234.8 px away", allowed=(False,)),
    _p("motion_veto_max_speed_h_per_s", "float", "h per s",
       "Camera-compensated target speed above this vetoes a candidate. A veto only: never proves identity.",
       8.0, N, "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "geometry", "D09; R1 approval §19", 0.0, 100.0),
    _p("motion_veto_max_scale_ratio_per_s", "float", "ratio per s",
       "Box-height change faster than this factor per second vetoes a candidate.", 1.8, N, "DESIGN_CHOICE", UNV,
       "BEHAVIOURAL", "geometry", "D09", 1.0, 100.0),

    # ---- population / competitors (D10) ------------------------------------------------------
    _p("population_match_iou_min", "float", "IoU ratio",
       "Minimum IoU to continue a population tracklet (geometry only; no identity claim).", 0.3, N,
       "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "population", "D10", 0.0, 1.0),
    _p("population_max_unmatched_s", "float", "s", "A population tracklet unmatched this long is dropped.",
       0.5, N, "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "population", "D10", 0.0, 30.0),
    _p("population_bank_max_samples", "int", "samples", "Appearance samples kept per population tracklet.",
       8, N, "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "population", "D10", 1, 1000),
    _p("competitor_radius_h", "float", "h",
       "Any detection-backed body whose box centre is within this of the target gate is a competitor, from "
       "its first frame, regardless of team label or age.", 1.5, N, "DESIGN_CHOICE", UNV, "BEHAVIOURAL",
       "population", "D10; R1 approval §E", 0.0, 20.0),

    # ---- merged-detection guard (D11) --------------------------------------------------------
    _p("merged_area_ratio", "float", "ratio to recent median target area",
       "A candidate box larger than this multiple of the target's recent median area is merged-suspect.", 1.6, N,
       "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "merged", "D11; F2 (a historical merged box for two players measured x3.1 the target area)",
       1.0, 100.0),
    _p("merged_aspect_band", "pair", "ratio to recent median aspect (w/h)",
       "Aspect ratio outside this band relative to the target's median is merged-suspect.", (0.6, 1.6), N,
       "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "merged", "D11", 0.0, 100.0),
    _p("merged_overlap_min", "float", "fraction of predicted box area",
       "A box covering at least this fraction of BOTH the target's and a competitor's predicted boxes is "
       "merged-suspect.", 0.3, N, "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "merged", "D11", 0.0, 1.0),

    # ---- appearance bank hygiene (D12, R1 approval §20) --------------------------------------
    _p("bank_max_samples", "int", "samples", "Target appearance bank capacity (diversity-pruned).", 24, N,
       "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "bank", "D12", 1, 1000),
    _p("bank_contact_iou", "float", "IoU ratio",
       "No bank update if any competitor box overlaps the target box by more than this.", 0.05, N,
       "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "bank", "D12", 0.0, 1.0),
    _p("bank_min_crop_quality", "float", "quality (0-1)", "No bank update from a crop below this quality.",
       0.6, N, "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "bank", "D12", 0.0, 1.0),
    _p("bank_min_novelty", "float", "similarity difference",
       "A new sample is added only if it differs from its nearest bank sample by at least this.", 0.02, N,
       "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "bank", "D12 diversity pruning", 0.0, 1.0),
    _p("post_reacq_quarantine_s", "float", "s", "No bank update for this long after a reacquisition.", 1.0, N,
       "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "bank", "D12", 0.0, 30.0),

    # ---- target state limits (D14) -----------------------------------------------------------
    _p("max_occluded_s", "float", "s", "OCCLUDED becomes LOST after this long.", 1.0, N, "DESIGN_CHOICE", UNV,
       "BEHAVIOURAL", "state", "D14. Historical occlusion_to_lost_frames 10 (frames, not seconds)", 0.0, 30.0,
       historical_value=10),
    _p("max_ambiguous_s", "float", "s", "AMBIGUOUS becomes LOST after this long (it must age out).", 1.0, N,
       "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "state", "D14; F6 (AMBIGUOUS never aged out historically)",
       0.0, 30.0),
    _p("occluder_cover_min", "float", "fraction of predicted target box",
       "If a competitor covers at least this fraction of the target's predicted box, an unobserved target "
       "is OCCLUDED rather than LOST.", 0.3, N, "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "state", "D14", 0.0, 1.0),
    _p("confidence_half_life_s", "float", "s",
       "identity_confidence_rb halves for every this-many seconds the target goes unconfirmed.", 0.5, N,
       "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "state", "D14", 0.001, 30.0),

    # ---- reacquisition (D15) -----------------------------------------------------------------
    _p("reacq_appearance_floor", "float", "composite similarity (0-1)",
       "Appearance floor during reacquisition (stricter than ordinary association).", 0.62, N,
       "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "reacquisition", "D15", 0.0, 1.0),
    _p("reacq_contrastive_margin", "float", "similarity difference",
       "Contrastive margin during reacquisition, before any absence growth.", 0.12, N, "DESIGN_CHOICE", UNV,
       "BEHAVIOURAL", "reacquisition", "D15. Historical re-entry margin 0.08", 0.0, 1.0, historical_value=0.08),
    _p("reacq_margin_growth_per_s", "float", "similarity difference per s of absence",
       "Longer absence raises the reacquisition margin by this much per second.", 0.02, N, "DESIGN_CHOICE", UNV,
       "BEHAVIOURAL", "reacquisition", "D15; R1 approval §F (longer absence, stronger evidence)", 0.0, 1.0),
    _p("reacq_margin_cap", "float", "similarity difference", "Upper limit of the grown reacquisition margin.",
       0.25, N, "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "reacquisition", "D15", 0.0, 1.0),
    _p("reacq_window_s", "float", "s", "Verification window W for a reacquisition candidate.", 0.5, N,
       "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "reacquisition", "D15", 0.0, 30.0),
    _p("reacq_min_hit_fraction", "float", "fraction of window frames",
       "The candidate must pass in at least this fraction of the window's frames (K of W).", 0.6, N,
       "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "reacquisition", "D15", 0.0, 1.0),
    _p("lost_long_s", "float", "s", "After this long LOST, the positional prior is dropped (appearance only).",
       3.0, N, "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "reacquisition", "D15", 0.0, 600.0),
    _p("positional_prior_half_life_s", "float", "s",
       "The positional prior's weight in reacquisition halves every this-many seconds since confirmation.", 1.0,
       N, "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "reacquisition", "D15", 0.001, 600.0),

    # ---- camera motion and cuts (D16) --------------------------------------------------------
    _p("cut_hist_distance_min", "float", "Bhattacharyya distance (0-1)",
       "A frame-to-frame grey-histogram distance at least this, together with flow failure, is a hard cut.",
       0.5, N, "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "camera", "D16", 0.0, 1.0),
    _p("cut_flow_inlier_fraction_max", "float", "fraction of tracked features",
       "Optical flow with fewer surviving inliers than this counts as flow failure.", 0.2, N, "DESIGN_CHOICE",
       UNV, "BEHAVIOURAL", "camera", "D16", 0.0, 1.0),
    _p("flow_max_corners", "int", "features", "Maximum background features for global-motion estimation.",
       300, N, "DESIGN_CHOICE", "NOT_APPLICABLE", "RUNTIME", "camera", "D16", 10, 10000),
    _p("flow_quality_level", "float", "OpenCV corner quality (0-1)", "goodFeaturesToTrack quality level.",
       0.01, N, "DESIGN_CHOICE", "NOT_APPLICABLE", "RUNTIME", "camera", "D16", 0.0, 1.0),
    _p("flow_min_distance_px", "float", "px (image resolution)", "Minimum spacing between flow features.",
       8.0, N, "DESIGN_CHOICE", "NOT_APPLICABLE", "RUNTIME", "camera", "D16", 1.0, 1000.0),
    _p("flow_person_mask_pad_h", "float", "h", "Padding around person boxes excluded from background flow.",
       0.1, N, "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "camera", "D16", 0.0, 5.0),

    # ---- REFERENCE_ONLY: recovered historical values. Never read by tracker code. -------------
    _p("hist_assoc_weight_appearance", "float", "weight", "Historical linear association weight (audit only).",
       0.40, H, "VERIFIED_OUTPUT", "NOT_APPLICABLE", "REFERENCE_ONLY", "historical",
       "least-squares fit, task3m association_capture.json (116 candidates, max err 0.0007)"),
    _p("hist_assoc_weight_motion", "float", "weight", "Historical association weight (audit only).", 0.20, H,
       "VERIFIED_OUTPUT", "NOT_APPLICABLE", "REFERENCE_ONLY", "historical", "same fit"),
    _p("hist_assoc_weight_iou_last_box", "float", "weight", "Historical association weight (audit only).", 0.10,
       H, "VERIFIED_OUTPUT", "NOT_APPLICABLE", "REFERENCE_ONLY", "historical", "same fit"),
    _p("hist_assoc_weight_colour_hsv", "float", "weight",
       "Historical HSV colour weight (audit only; inverted under contamination, F1e).", 0.15, H,
       "VERIFIED_OUTPUT", "NOT_APPLICABLE", "REFERENCE_ONLY", "historical", "same fit"),
    _p("hist_assoc_weight_size", "float", "weight", "Historical association weight (audit only).", 0.05, H,
       "VERIFIED_OUTPUT", "NOT_APPLICABLE", "REFERENCE_ONLY", "historical", "same fit"),
    _p("hist_min_appearance_floor", "float", "cosine similarity (yolo26n-reid)",
       "Historical appearance floor (audit only).", 0.30, H, "DOCUMENTED", "NOT_APPLICABLE", "REFERENCE_ONLY",
       "historical", "TASK3M_REID_APPEARANCE_AUDIT.md s2"),
    _p("hist_ambiguity_margin", "float", "score margin", "Historical assignment ambiguity margin (audit only).",
       0.08, H, "DOCUMENTED", "NOT_APPLICABLE", "REFERENCE_ONLY", "historical", "TASK3M_MOTION_ASSOCIATION_AUDIT.md s2"),
    _p("hist_base_gate_px", "float", "px", "Historical base gate (audit only).", 90.0, H, "DOCUMENTED",
       "NOT_APPLICABLE", "REFERENCE_ONLY", "historical", "TASK3C finding 4"),
    _p("hist_gate_growth_px_per_frame", "float", "px per frame", "Historical gate growth (audit only).", 35.0, H,
       "DOCUMENTED", "NOT_APPLICABLE", "REFERENCE_ONLY", "historical", "TASK3C finding 4"),
    _p("hist_max_gate_px", "float", "px", "Historical max gate (audit only).", 350.0, H, "DOCUMENTED",
       "NOT_APPLICABLE", "REFERENCE_ONLY", "historical", "TASK3C finding 4"),
    _p("hist_fast_path_radius_px", "float", "px",
       "Historical identity fast-path radius (audit only; the fast path is defect F1 and is not rebuilt).", 45.0,
       H, "DOCUMENTED", "NOT_APPLICABLE", "REFERENCE_ONLY", "historical", "TASK3M_MOTION_ASSOCIATION_AUDIT.md s2"),
    _p("hist_occlusion_to_lost_frames", "int", "frames", "Historical OCCLUDED->LOST limit (audit only).", 10, H,
       "DOCUMENTED", "NOT_APPLICABLE", "REFERENCE_ONLY", "historical", "TASK3C"),
    _p("hist_max_frames_default", "int", "frames", "Historical run_mo max_frames default (audit only).", 450, H,
       "VERIFIED_CODE", "NOT_APPLICABLE", "REFERENCE_ONLY", "historical", "pipeline.py:217"),
)

# ---------------------------------------------------------------------------------------------
# Cross-parameter rules (checked on every configuration, including overrides)
# ---------------------------------------------------------------------------------------------
def _cross_checks(v: Mapping[str, Any]) -> list:
    errs = []
    if abs(v["composite_weight_lab"] + v["composite_weight_embedding"] - 1.0) > 1e-9:
        errs.append("composite_weight_lab + composite_weight_embedding must equal 1.0")
    if not v["untrusted_identity_confidence_ceiling"] < v["legacy_identity_trust_threshold"]:
        errs.append("untrusted_identity_confidence_ceiling must be < legacy_identity_trust_threshold")
    if v["reacq_appearance_floor"] < v["appearance_floor"]:
        errs.append("reacq_appearance_floor must be >= appearance_floor (reacquisition is stricter)")
    if v["reacq_contrastive_margin"] < v["contrastive_margin"]:
        errs.append("reacq_contrastive_margin must be >= contrastive_margin (reacquisition is stricter)")
    if v["reacq_margin_cap"] < v["reacq_contrastive_margin"]:
        errs.append("reacq_margin_cap must be >= reacq_contrastive_margin")
    if v["gate_base_h"] > v["gate_max_h"]:
        errs.append("gate_base_h must be <= gate_max_h")
    for band in ("lab_upper_band", "lab_lower_band", "merged_aspect_band"):
        lo, hi = v[band]
        if not lo < hi:
            errs.append(f"{band} must be (low, high) with low < high")
    if v["merged_aspect_band"][0] >= 1.0 or v["merged_aspect_band"][1] <= 1.0:
        errs.append("merged_aspect_band must contain 1.0 (the target's own aspect)")
    if v["lost_long_s"] < v["max_ambiguous_s"] or v["lost_long_s"] < v["max_occluded_s"]:
        errs.append("lost_long_s must be >= max_ambiguous_s and >= max_occluded_s")
    dcm = v["detection_confidence_by_position_source"]
    if set(dcm) != {"CONFIRMED_DETECTION", "AMBIGUOUS_DETECTION", "WEAK_SPATIAL_RECOVERY", "PREDICTED"}:
        errs.append("detection_confidence_by_position_source must cover exactly the 4 historical position sources")
    return errs


class ConfigError(ValueError):
    """Raised for unknown names, wrong types, out-of-range values or failed cross-checks."""


def _check_value(p: Param, value: Any) -> Optional[str]:
    t = p.type
    if t == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            return f"{p.name}: expected a finite number, got {value!r}"
        x = float(value)
    elif t in ("int", "optional_int"):
        if value is None and t == "optional_int":
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            return f"{p.name}: expected an int, got {value!r}"
        x = value
    elif t == "bool":
        if not isinstance(value, bool):
            return f"{p.name}: expected a bool, got {value!r}"
        x = None
    elif t == "str":
        if not isinstance(value, str) or not value:
            return f"{p.name}: expected a non-empty str, got {value!r}"
        x = None
    elif t == "pair":
        if (not isinstance(value, (tuple, list)) or len(value) != 2 or
                any(isinstance(e, bool) or not isinstance(e, (int, float)) for e in value)):
            return f"{p.name}: expected a pair of numbers, got {value!r}"
        x = None
        for e in value:
            if p.min_value is not None and e < p.min_value:
                return f"{p.name}: element {e} below minimum {p.min_value}"
            if p.max_value is not None and e > p.max_value:
                return f"{p.name}: element {e} above maximum {p.max_value}"
    elif t == "mapping":
        if not isinstance(value, Mapping):
            return f"{p.name}: expected a mapping, got {value!r}"
        x = None
    else:
        return f"{p.name}: unknown parameter type {t!r}"
    if x is not None:
        if p.min_value is not None and x < p.min_value:
            return f"{p.name}: {x} below minimum {p.min_value}"
        if p.max_value is not None and x > p.max_value:
            return f"{p.name}: {x} above maximum {p.max_value}"
    if p.allowed is not None and value not in p.allowed:
        return f"{p.name}: {value!r} not in allowed {p.allowed!r}"
    return None


def _freeze(t: str, value: Any) -> Any:
    if t == "pair":
        return tuple(float(e) if isinstance(e, float) else e for e in value)
    if t == "mapping":
        return MappingProxyType(dict(value))
    if t == "float":
        return float(value)
    return value


_BY_NAME: Dict[str, Param] = {}
for _param in PARAMS:
    if _param.name in _BY_NAME:
        raise ConfigError(f"duplicate parameter name {_param.name}")
    _BY_NAME[_param.name] = _param
REGISTRY: Mapping[str, Param] = MappingProxyType(_BY_NAME)


@dataclass(frozen=True)
class TrackerConfig:
    """Validated, immutable configuration. Build with `TrackerConfig.default()` or `.with_overrides()`."""
    values: Mapping[str, Any]
    overrides: Tuple[Tuple[str, Any], ...] = field(default=())

    @staticmethod
    def default() -> "TrackerConfig":
        return TrackerConfig.with_overrides({})

    @staticmethod
    def with_overrides(overrides: Mapping[str, Any]) -> "TrackerConfig":
        errs = []
        vals = {}
        for name, p in REGISTRY.items():
            vals[name] = p.default
        for name, value in overrides.items():
            if name not in REGISTRY:
                errs.append(f"unknown parameter {name!r}")
                continue
            if REGISTRY[name].role == "REFERENCE_ONLY":
                errs.append(f"{name} is REFERENCE_ONLY and cannot be overridden")
                continue
            vals[name] = value
        for name, value in vals.items():
            e = _check_value(REGISTRY[name], value)
            if e:
                errs.append(e)
        if not errs:
            errs.extend(_cross_checks(vals))
        if errs:
            raise ConfigError("; ".join(errs))
        frozen = {k: _freeze(REGISTRY[k].type, v) for k, v in vals.items()}
        ov = tuple(sorted((k, _jsonable(_freeze(REGISTRY[k].type, v))) for k, v in overrides.items()))
        return TrackerConfig(MappingProxyType(frozen), ov)

    def get(self, name: str) -> Any:
        """Value of a usable parameter. REFERENCE_ONLY values are refused: they must not drive behaviour."""
        p = REGISTRY.get(name)
        if p is None:
            raise ConfigError(f"unknown parameter {name!r}")
        if p.role == "REFERENCE_ONLY":
            raise ConfigError(f"{name} is REFERENCE_ONLY (historical audit value) and must not be used by tracker code")
        return self.values[name]

    __getitem__ = get

    def reference_value(self, name: str) -> Any:
        """Explicit accessor for REFERENCE_ONLY values, for audit and report code only."""
        p = REGISTRY.get(name)
        if p is None or p.role != "REFERENCE_ONLY":
            raise ConfigError(f"{name!r} is not a REFERENCE_ONLY parameter")
        return self.values[name]

    def as_record(self) -> Dict[str, Any]:
        """Every value (JSON-safe, sorted) plus the overrides applied."""
        return {"registry_version": CONFIG_REGISTRY_VERSION,
                "values": {k: _jsonable(self.values[k]) for k in sorted(self.values)},
                "overrides": [[k, v] for k, v in self.overrides]}

    def config_hash(self) -> str:
        """SHA-256 of the canonical JSON of `as_record()` (recorded in every tracking JSON's provenance)."""
        blob = json.dumps(self.as_record(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(blob.encode("ascii")).hexdigest()


def registry_records() -> list:
    """The full registry as JSON-safe records, in declaration order (for docs and audits)."""
    return [p.as_record() for p in PARAMS]


def params_by(**criteria: Any) -> Tuple[Param, ...]:
    """Filter the registry, e.g. params_by(role="BEHAVIOURAL", source="NEW_R1")."""
    return tuple(p for p in PARAMS if all(getattr(p, k) == v for k, v in criteria.items()))


# ---------------------------------------------------------------------------------------------
# Unit conversions. The only sanctioned way to turn seconds into frames and h into pixels.
# ---------------------------------------------------------------------------------------------
_EPS = 1e-9


def seconds_to_frames(seconds: float, fps: float) -> int:
    """Smallest whole number of frames covering `seconds` at `fps`. 0 s -> 0 frames; any positive
    duration -> at least 1 frame. The same limit in seconds gives the same real duration at any fps."""
    if not (isinstance(fps, (int, float)) and fps > 0 and math.isfinite(fps)):
        raise ValueError(f"fps must be a positive finite number, got {fps!r}")
    if seconds < 0 or not math.isfinite(seconds):
        raise ValueError(f"seconds must be a non-negative finite number, got {seconds!r}")
    if seconds == 0:
        return 0
    return max(1, int(math.ceil(seconds * fps - _EPS)))


def frames_to_seconds(frames: int, fps: float) -> float:
    if not (isinstance(fps, (int, float)) and fps > 0 and math.isfinite(fps)):
        raise ValueError(f"fps must be a positive finite number, got {fps!r}")
    return frames / float(fps)


def h_to_px(h_units: float, target_height_px: float) -> float:
    """Convert a distance in target heights to pixels, given the current target height in pixels."""
    if not (target_height_px > 0 and math.isfinite(target_height_px)):
        raise ValueError(f"target_height_px must be positive and finite, got {target_height_px!r}")
    return float(h_units) * float(target_height_px)


def px_to_h(px: float, target_height_px: float) -> float:
    if not (target_height_px > 0 and math.isfinite(target_height_px)):
        raise ValueError(f"target_height_px must be positive and finite, got {target_height_px!r}")
    return float(px) / float(target_height_px)


def per_second_to_per_frame(rate_per_s: float, fps: float) -> float:
    """Convert a rate expressed per second (e.g. h per s) into the same quantity per frame."""
    if not (isinstance(fps, (int, float)) and fps > 0 and math.isfinite(fps)):
        raise ValueError(f"fps must be a positive finite number, got {fps!r}")
    return float(rate_per_s) / float(fps)


def iter_params(group: Optional[str] = None) -> Iterable[Param]:
    return (p for p in PARAMS if group is None or p.group == group)

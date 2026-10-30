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

CONFIG_REGISTRY_VERSION = "r2b-1"   # r2b-1: +population_part_birth_containment (R2 DEV F3b); r2a-1: +seed_contain_* (R2 DEV F1), +merged_uncorroborated_area_ratio, +merged_anomaly_max_s, +merged_anomaly_clear_area_ratio, +merged_anomaly_clear_s (R2 DEV F3); r1f-2: -population_merge_emerge_pad_h, +target_known_other_min_s (R1-F review)

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
    type: str                 # "float" | "int" | "bool" | "str" | "pair" | "tuple" | "mapping" | "optional_int"
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
    # Seed association by containment (R2 DEV F1, NEW_R2): a human seed box is often looser than a detector body
    # box. When no detection reaches seed_min_iou, a detection may still be the boxed body if it lies almost wholly
    # inside the seed box, is centred in it, is body-sized against it, and is the ONLY detection substantially
    # inside it. Any second detection substantially inside the box makes the seed ambiguous (fail closed).
    _p("seed_contain_min", "float", "fraction of the detection's area",
       "Containment seed path: the detection must lie at least this much inside the human seed box (the human box is "
       "taken as the outer bound of the boxed body; the margin allows limbs a tight detector box may include).",
       0.9, N, "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "video",
       "NEW_R2 (R2 DEV remediation F1). DEV seeds refused by IoU: the matching detection measured 1.00 inside the "
       "seed box (all four); seeds accepted by IoU measured 0.87-1.00",
       0.5, 1.0),
    _p("seed_contain_max_centre_offset", "float", "seed-box heights",
       "Containment seed path: the detection's centre must lie within this distance of the seed box centre, in each "
       "axis (scale-free). A body inside a loose human box but well off its centre may be someone else.",
       0.15, N, "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "video",
       "NEW_R2 (R2 DEV remediation F1). DEV seeds: matching detections measured <= 0.06 seed heights off centre; a "
       "neighbour at shoulder distance is about 0.25-0.45 seed heights off", 0.0, 1.0),
    _p("seed_contain_min_height_ratio", "float", "ratio to the seed box height",
       "Containment seed path: the detection must be at least this tall relative to the seed box. A much smaller "
       "body inside the human box is more likely a farther person or a fragment than the boxed body.",
       0.5, N, "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "video",
       "NEW_R2 (R2 DEV remediation F1). DEV seeds: matching detections measured 0.62-0.94 of the seed height; a "
       "farther, smaller person is also excluded by seed_contain_max_bottom_gap (feet line)", 0.0, 1.0),
    _p("seed_contain_min_width_ratio", "float", "ratio to the seed box width",
       "Containment seed path: the detection must be at least this wide relative to the seed box. A human box much "
       "wider than one body (drawn over several people) is not a looser box around one body.", 0.45, N,
       "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "video",
       "NEW_R2 (R2 DEV remediation F1, review M1). DEV seeds refused by IoU: 0.55-0.68 of the seed width", 0.0, 1.0),
    _p("seed_contain_max_bottom_gap", "float", "seed-box heights",
       "Containment seed path: the detection's bottom edge (feet line) must be within this distance of the seed box's "
       "bottom edge. A smaller person standing farther away inside the human box has a higher feet line.", 0.2, N,
       "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "video",
       "NEW_R2 (R2 DEV remediation F1, review M2). DEV seeds refused by IoU: 0.07-0.16 seed heights", 0.0, 1.0),
    _p("seed_contain_competitor_max", "float", "fraction of the other detection's area",
       "Containment seed path: if any other body detection lies at least this much inside the seed box, the box may "
       "hold two people and the seed is refused as ambiguous (fail closed). Applies only to the containment path; "
       "the IoU path is unchanged (D18).",
       0.25, N, "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "video",
       "NEW_R2 (R2 DEV remediation F1). DEV seeds: competing detections measured <= 0.16 inside the seed box except "
       "in crowded seeds (0.46-0.79), which this refuses", 0.0, 1.0),
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
    _p("detector_model_file", "str", "path (relative to the project root, or absolute)",
       "ONNX person-detector model file. Loaded only if its SHA-256 equals detector_model_sha256.",
       "models/yolox_s.onnx", N, "DESIGN_CHOICE", "NOT_APPLICABLE", "RUNTIME", "detector",
       "R1/R1C/R1C_MODEL_RECORD.md; official Megvii YOLOX release 0.1.1rc0 asset yolox_s.onnx"),
    _p("detector_model_sha256", "str", "sha256 hex",
       "Pinned SHA-256 of the detector model. A mismatch refuses the run (D22).",
       "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063", N, "VERIFIED_CODE", "NOT_APPLICABLE",
       "RUNTIME", "detector",
       "sha256 of https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx, "
       "downloaded 2026-10-24 (R1/R1C/R1C_MODEL_RECORD.md)"),
    _p("detector_model_status", "str", "licence status",
       "Licence status of the detector model, copied into provenance (R1_MODEL_AND_LICENSE_RECORD.md).",
       "LICENCE_UNCLEAR", N, "DOCUMENTED", "NOT_APPLICABLE", "RUNTIME", "detector",
       "R1 approval §2: YOLOX-s DEVELOPMENT/EVALUATION; not PRODUCTION_APPROVED until the COCO question is resolved",
       allowed=("PRODUCTION_APPROVED", "DEVELOPMENT_ONLY", "LICENCE_UNCLEAR", "NOT_APPROVED")),
    _p("detector_person_class_index", "int", "class index",
       "Index of the 'person' class in the detector's class outputs (COCO class 0).", 0, H, "VERIFIED_CODE",
       "CONTRACT_VERIFIED", "CONTRACT", "detector",
       "YOLOX yolox/data/datasets/coco_classes.py: COCO_CLASSES[0] == 'person'", 0, 1000),
    _p("detector_letterbox_pad_value", "int", "uint8 grey level",
       "Padding value of the detector's letterbox (model preprocessing convention).", 114, H, "VERIFIED_CODE",
       "CONTRACT_VERIFIED", "CONTRACT", "detector",
       "YOLOX yolox/data/data_augment.py preproc(): np.ones(...) * 114, image placed top-left, BGR 0-255 float, "
       "no mean/std normalisation (verified on real footage 2026-10-24: the normalised variant scores ~0.001)", 0, 255),
    _p("detector_strides", "tuple", "px per output cell",
       "Output strides of the detector heads, used to decode grid offsets.", (8, 16, 32), H, "VERIFIED_CODE",
       "CONTRACT_VERIFIED", "CONTRACT", "detector",
       "YOLOX yolox/utils/demo_utils.py demo_postprocess(): strides [8, 16, 32] (p6=False); 8400 = 80^2+40^2+20^2",
       1, 1024),
    _p("detector_input_channels", "int", "channels", "Colour channels of the detector input tensor (BGR).", 3, H,
       "VERIFIED_CODE", "CONTRACT_VERIFIED", "CONTRACT", "detector",
       "official yolox_s.onnx input 'images' shape [1, 3, 640, 640] (checked at load)", 1, 4),
    _p("detector_box_fields", "int", "values per prediction before class scores",
       "Leading values of each detector prediction: cx, cy, w, h, objectness.", 5, H, "VERIFIED_CODE",
       "CONTRACT_VERIFIED", "CONTRACT", "detector",
       "YOLOX demo/ONNXRuntime/onnx_inference.py: boxes = p[:, :4]; scores = p[:, 4:5] * p[:, 5:]", 1, 64),
    _p("detector_tiling", "str", "mode",
       "none: one full-frame pass. grid: overlapping tiles (plus an optional full-frame pass) so small people get "
       "more network pixels at the model's fixed input size; results are merged by global NMS.", "grid", N,
       "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "detector",
       "D17; the official YOLOX ONNX models have a fixed 640x640 input. Selected by the pre-declared rule in "
       "tools_rb/detector_eval.py on one development clip, detection quality only (R1/R1C/R1C_STAGE_RECORD.md): "
       "full-frame 640 small-person pool recall 0.82, 2x2 tiles + full frame 0.99. Proxy, not ground truth",
       allowed=("none", "grid")),
    _p("detector_tile_grid", "pair", "tiles (columns, rows)", "Tile grid used when detector_tiling is grid.",
       (2, 2), N, "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "detector", "D17", 1, 8),
    _p("detector_tile_overlap_fraction", "float", "fraction of tile size",
       "Overlap between neighbouring tiles, so a person cut by one tile edge is whole in the neighbour.", 0.2, N,
       "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "detector", "D17", 0.0, 0.9),
    _p("detector_tile_include_full_frame", "bool", "flag",
       "With tiling, also run one full-frame pass (keeps people larger than a tile whole).", True, N,
       "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "detector", "D17"),
    _p("detector_tile_edge_margin_fraction", "float", "fraction of tile size",
       "A tile-pass box within this fraction of the tile size from an internal tile edge counts as touching it "
       "(possible cut fragment). Network box edges are coarse, so exact equality is not enough.", 0.02, N,
       "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "detector", "R1-C development check: a cut box ended 1 px inside the edge",
       0.0, 0.5),
    _p("detector_edge_touch_margin_px", "float", "px (image resolution)",
       "A box within this many pixels of the image border is flagged touches_image_edge (person may be partly "
       "outside the frame).", 2.0, N, "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "detector",
       "R1-C: boxes cut by the bottom border ended at 719 of 720 px", 0.0, 64.0),
    _p("detector_tile_cut_min_span_ratio", "float", "ratio",
       "A cut box is treated as a fragment of a larger box only if it spans at least this fraction of that box "
       "across the cut (its width for a top/bottom cut, its height for a left/right cut). A narrower box inside "
       "another person's box is a partly hidden second person and is kept.", 0.7, N, "DESIGN_CHOICE", UNV,
       "BEHAVIOURAL", "detector",
       "R1-C development check: containment alone removed a partly occluded second person (F2/F5: occluded "
       "competitors must stay visible)", 0.0, 1.0),
    _p("detector_tile_cut_containment", "float", "fraction of box area",
       "A tile-pass box that touches an internal tile edge and lies at least this much inside another kept box is "
       "dropped as a cut fragment. Boxes touching the real image edge are never dropped by this rule.", 0.8, N,
       "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "detector", "D17; F5 (fragments of one body acting as extra people)",
       0.0, 1.0),
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
       allowed=("composite_lab_dinov2", "scripted")),
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
    _p("crop_good_height_px", "int", "px (image resolution)",
       "Crop height at which resolution stops limiting quality: quality's height term is min(1, h / this).", 64, N,
       "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "appearance", "R1-D crop quality formula (R1/R1D/R1D_STAGE_RECORD.md)",
       4, 1024),
    _p("crop_max_covered_fraction", "float", "fraction of box area",
       "A crop more than this much covered by other people's boxes is flagged OCCLUDED_BY_OTHER and is never "
       "bank-eligible (its pixels may belong to someone else).", 0.3, N, "DESIGN_CHOICE", UNV, "BEHAVIOURAL",
       "appearance", "D12; R1 approval §20 (no bank update from heavily overlapped observations)", 0.0, 1.0),
    _p("bank_max_covered_fraction", "float", "fraction of box area",
       "Bank updates are refused if more than this share of the box is covered by other people's boxes. Stricter "
       "than the IoU contact test, which shrinks as an occluder gets larger.", 0.05, N, "DESIGN_CHOICE", UNV,
       "BEHAVIOURAL", "bank", "R1-D review finding 2 (a large close occluder passed the IoU test at 30% coverage)",
       0.0, 1.0),
    _p("lab_min_band_pixels", "int", "pixels",
       "A LAB colour band with fewer pixels than this gives no LAB descriptor (histogram too sparse to compare).",
       150, N, "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "appearance",
       "R1-D review finding 9 (a 24 px crop gives ~70 pixels over 512 bins)", 1, 100000),
    _p("embedding_norm_tolerance", "float", "absolute difference from 1",
       "Embeddings must be unit length within this tolerance; otherwise the backend is broken (fail closed).",
       1e-6, N, "DESIGN_CHOICE", "NOT_APPLICABLE", "RUNTIME", "appearance", "R1-D review finding 10", 0.0, 0.1),
    _p("lab_center_width_fraction", "float", "fraction of box width",
       "Central share of the box width used for the LAB colour bands (box sides are mostly background).", 0.6, N,
       "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "appearance", "R1-D LAB descriptor", 0.05, 1.0),
    _p("lab_floor", "float", "LAB similarity (Bhattacharyya coefficient, 0-1)",
       "Per-cue veto: below this LAB similarity the colours are incompatible and appearance cannot support "
       "identity. Above it, colour still proves nothing on its own.", 0.3, N, "DESIGN_CHOICE", UNV,
       "BEHAVIOURAL", "appearance", "R1-D; R1/R1_GENERIC_TRACKER_REQUIREMENT.md (colour may only reject)", 0.0, 1.0),
    _p("embedding_floor", "float", "embedding similarity (max(0, cosine), 0-1)",
       "Per-cue requirement: the generic embedding must reach this similarity for appearance to support "
       "identity, so colour can never establish identity alone.", 0.70, N, "DESIGN_CHOICE", UNV, "BEHAVIOURAL",
       "appearance", "R1-D; neither cue alone establishes identity. Calibrated by the pre-declared rule in "
       "tools_rb/appearance_eval.py (P5 of same-person CLS similarity 0.741 on the development clip, floored to "
       "0.05; R1/R1D/appearance_eval/calibration_run.json); was 0.5 before calibration, which is below most "
       "different-person similarities (median ~0.61). In-sample, weak labels, one clip: UNVALIDATED until R2",
       0.0, 1.0),
    _p("embedding_model_file", "str", "path (relative to the project root, or absolute)",
       "ONNX generic appearance-embedding model (DINOv2 ViT-S/14 backbone; NOT a football ReID model).",
       "models/dinov2_vits14_cls_patchmean_224x112.onnx", N, "DESIGN_CHOICE", "NOT_APPLICABLE", "RUNTIME",
       "appearance", "R1/R1D/R1D_MODEL_RECORD.md; exported by tools_rb/export_models.py"),
    _p("embedding_model_sha256", "str", "sha256 hex",
       "Pinned SHA-256 of the embedding model. A mismatch refuses to load.",
       "0ab62713cf7759dd820b53576b39010cb0fe8dcfbe22af444c945c519bf9efea", N, "VERIFIED_CODE", "NOT_APPLICABLE",
       "RUNTIME", "appearance", "tools_rb/export_models.py output, byte-reproducible (two exports, same hash)"),
    _p("embedding_model_status", "str", "licence status",
       "Licence status of the embedding model, copied into provenance.", "LICENCE_UNCLEAR", N, "DOCUMENTED",
       "NOT_APPLICABLE", "RUNTIME", "appearance",
       "R1 approval §4: DEVELOPMENT/EVALUATION; Apache-2.0 confirmed by repo and model card, pending Nathan's sign-off",
       allowed=("PRODUCTION_APPROVED", "DEVELOPMENT_ONLY", "LICENCE_UNCLEAR", "NOT_APPROVED")),
    _p("embedding_input_hw", "pair", "px (height, width)", "Embedding network input size (16 x 8 patches of 14 px).",
       (224, 112), H, "VERIFIED_CODE", "CONTRACT_VERIFIED", "CONTRACT", "appearance",
       "exported model input pixel_values [1, 3, 224, 112] (dinov2 EXPORT_RECORD.json)", 14, 1024),
    _p("embedding_mean_rgb", "tuple", "fraction (0-1) per RGB channel",
       "Normalisation mean of the embedding model input.", (0.485, 0.456, 0.406), H, "VERIFIED_CODE",
       "CONTRACT_VERIFIED", "CONTRACT", "appearance", "facebook/dinov2-small preprocessor_config.json image_mean",
       0.0, 1.0),
    _p("embedding_std_rgb", "tuple", "fraction (0-1) per RGB channel",
       "Normalisation std of the embedding model input.", (0.229, 0.224, 0.225), H, "VERIFIED_CODE",
       "CONTRACT_VERIFIED", "CONTRACT", "appearance", "facebook/dinov2-small preprocessor_config.json image_std",
       0.001, 1.0),
    _p("embedding_pooling", "str", "mode",
       "cls: the final-layernorm CLS token. cls_patchmean: CLS and the mean patch token, each L2-normalised, "
       "concatenated and L2-normalised again.", "cls", N, "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "appearance",
       "R1-D; chosen by the pre-declared rule in tools_rb/appearance_eval.py", allowed=("cls", "cls_patchmean")),

    # ---- identity decision (D08, D09) --------------------------------------------------------
    _p("appearance_floor", "float", "composite similarity (0-1)",
       "A box may become the target's observed box only if its similarity to the target bank is >= this.",
       0.80, N, "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "identity",
       "D08 test 1. R1-D calibration (tools_rb/appearance_eval.py: P5 of same-person composite 0.806 with soft LAB "
       "binning, floored to 0.05; was 0.55; R1/R1D/appearance_eval/calibration_run.json). UNVALIDATED until R2. "
       "Historical floor 0.30 was a cosine on yolo26n-reid: different scale, not comparable",
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
    _p("motion_velocity_tau_s", "float", "s",
       "Time constant of the exponential smoothing of a tracklet's camera-compensated velocity and log-height "
       "rate (alpha = 1 - exp(-dt / tau), so the smoothing is the same at any fps).", 0.2, N, "DESIGN_CHOICE",
       UNV, "BEHAVIOURAL", "geometry", "R1-E motion model (NEW)", 0.001, 30.0),
    _p("motion_max_extrapolation_s", "float", "s",
       "A box is extrapolated with its velocity for at most this long after its last observation; after that "
       "it is HELD (camera-compensated only, no further extrapolation).", 0.3, N, "DESIGN_CHOICE", UNV,
       "BEHAVIOURAL", "geometry", "R1-E motion model (NEW); D06 observed vs predicted vs held", 0.0, 30.0),
    _p("motion_jitter_h", "float", "h",
       "Detector box-centre jitter allowed on top of the speed limit when checking motion plausibility.",
       0.25, N, "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "geometry", "R1-E motion model (NEW)", 0.0, 20.0),
    _p("motion_scale_jitter_ratio", "float", "ratio",
       "Detector box-height jitter allowed on top of motion_veto_max_scale_ratio_per_s.", 1.25, N,
       "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "geometry", "R1-E motion model (NEW)", 1.0, 100.0),

    # ---- population / competitors (D10) ------------------------------------------------------
    _p("population_match_iou_min", "float", "IoU ratio",
       "Minimum IoU to continue a population tracklet (geometry only; no identity claim).", 0.3, N,
       "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "population", "D10", 0.0, 1.0),
    _p("population_max_unmatched_s", "float", "s", "A population tracklet unmatched this long is dropped.",
       0.5, N, "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "population", "D10", 0.0, 30.0),
    _p("population_bank_max_samples", "int", "samples", "Appearance samples kept per population tracklet.",
       8, N, "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "population", "D10", 1, 1000),
    _p("competitor_radius_h", "float", "h",
       "Any detection-backed body whose box centre is within this of the query box centre (in R1-F: the target "
       "gate), or that overlaps the query box, is a competitor, from its first frame, regardless of team label "
       "or age.", 1.5, N, "DESIGN_CHOICE", UNV, "BEHAVIOURAL",
       "population", "D10; R1 approval §E", 0.0, 20.0),
    _p("population_duplicate_iou", "float", "IoU ratio",
       "Two same-frame detections with IoU at least this are one body detected twice: the lower-scored one is "
       "suppressed as a separate tracklet but kept as evidence, and the kept one is flagged (never a bank "
       "sample). Below this, overlapping boxes are separate people.", 0.85, N, "DESIGN_CHOICE", UNV,
       "BEHAVIOURAL", "population", "R1-E (NEW); detector NMS already removes IoU > detector_nms_iou", 0.0, 1.0),
    _p("population_assoc_ambiguity_margin", "float", "IoU difference",
       "A tracklet-detection link is ambiguous if a rival detection or rival tracklet is within this IoU of "
       "it. Ambiguous links end the tracklets involved (no continuity is claimed through the contact).", 0.1, N,
       "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "population", "R1-E (NEW); F1/F4 identity switch at contact", 0.0, 1.0),
    _p("population_max_gap_s", "float", "s",
       "A tracklet missed for at most this long may continue on a detection, but only if no other body touched "
       "its predicted box during the gap. Longer gaps always start a new tracklet.", 0.2, N, "DESIGN_CHOICE",
       UNV, "BEHAVIOURAL", "population", "R1-E (NEW)", 0.0, 30.0),
    _p("population_contact_cover_min", "float", "fraction of predicted box area",
       "A hidden tracklet is in contact with another body if that body covers at least this fraction of its "
       "predicted box.", 0.1, N, "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "population", "R1-E (NEW)", 0.0, 1.0),
    _p("population_partial_height_ratio", "float", "ratio to predicted box height",
       "A detection shorter than this fraction of its tracklet's predicted height is PARTIAL (part of the body "
       "hidden or outside the image). PARTIAL boxes do not update velocity or size.", 0.75, N, "DESIGN_CHOICE",
       UNV, "BEHAVIOURAL", "population", "R1-E (NEW)", 0.0, 1.0),
    _p("population_min_inside_fraction", "float", "fraction of predicted box area",
       "A hidden tracklet whose predicted box has less than this fraction inside the image has left the frame "
       "and is closed.", 0.5, N, "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "population", "R1-E (NEW)", 0.0, 1.0),
    _p("population_provisional_s", "float", "s",
       "A tracklet younger than this is labelled provisional. Informational only: provisional tracklets are "
       "competitors exactly like older ones.", 0.3, N, "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "population",
       "R1-E (NEW); D10 (historical PROVISIONAL tracks were invisible to contests, defect F4)", 0.0, 30.0),
    _p("population_bank_sample_interval_s", "float", "s",
       "Minimum time between appearance samples offered to one tracklet's bank.", 0.25, N, "DESIGN_CHOICE",
       UNV, "BEHAVIOURAL", "population", "R1-E (NEW)", 0.0, 30.0),
    _p("population_closed_memory_s", "float", "s",
       "Closed tracklets (with their appearance banks) stay available as recent competitors for this long.",
       2.0, N, "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "population", "R1-E (NEW); D08 pre-contact competitor "
       "appearance", 0.0, 600.0),
    _p("population_merged_region_hold_s", "float", "s",
       "A merged region (where one detection covered two or more bodies) stays UNRESOLVED for this long after "
       "the last size/multi-claim evidence, unless at least as many separate bodies come out of it first. While "
       "unresolved, a single box inside it is MERGED_SUSPECTED (a second body may still be hidden in it).", 0.5,
       N, "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "population", "R1-E (NEW); D11 merged-detection guard", 0.0,
       30.0),
    _p("population_nested_containment", "float", "fraction of the smaller box's area",
       "A detection lying at least this much inside a larger same-frame detection is NESTED: kept as an "
       "observation and a competitor, but it starts no tracklet and claims nothing (detectors return part-body "
       "boxes inside a person's full box; a second person that hidden is not trackable from geometry either). The "
       "containing box is CONTAINS_OTHER and never an appearance sample.", 0.75, N, "DESIGN_CHOICE", UNV,
       "BEHAVIOURAL", "population", "R1-E (NEW); R1-E development-clip population check: at 0.9, alternative "
       "boxes of one person (smaller box 0.75-0.9 inside the larger) still became separate tracklets and "
       "produced false MULTI_CLAIM merges; side-by-side people in contact measured 0.33-0.62. In-sample, "
       "UNVALIDATED.", 0.0, 1.0),
    _p("population_part_birth_containment", "float", "fraction of the smaller box's area",
       "A tracklet born with at least this fraction of the smaller of (its box, the box of a body observed in the "
       "same frame, older and at least 1/population_partial_height_ratio taller) inside the other has UNPROVEN "
       "distinctness from that body: possibly its part box, or a farther person partly behind it. It is a tracklet "
       "and a competitor; when the older body keeps its own detection the younger is not a second MULTI_CLAIM "
       "claimant, and neither corroborates a size change of the other. Proven distinct once the two are observed "
       "in one frame overlapping less than this, or the younger is observed as tall as a whole body.",
       0.5, N, "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "population",
       "NEW_R2 (R2 DEV remediation F3b). DEV 37a109c: 95 of 223 MULTI_CLAIM merge frames had a claimant born "
       ">= 0.5 inside another observed body; visually checked same-body part boxes were born 0.55-0.77 inside. "
       "R1-E measured side-by-side people in contact at 0.33-0.62 (these start unproven when first seen that "
       "close and become proven once seen apart). Must be <= population_nested_containment.", 0.0, 1.0),
    _p("population_size_ref_window_s", "float", "s",
       "A tracklet's size reference is the median width and height of its non-partial observed boxes in this "
       "recent window (camera-scaled).", 1.0, N, "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "population",
       "R1-E (NEW); D11 (recent median), D13", 0.0, 30.0),
    _p("population_size_ref_min_s", "float", "s",
       "A size reference is ESTABLISHED only when its non-partial observations span at least this long. Size-"
       "based merge tests (AREA_RATIO, ASPECT_WIDE, PERSISTENT_MERGE) need an established reference: a box seen "
       "once may itself have been partial.", 0.2, N, "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "population",
       "R1-E (NEW); R1-E development-clip population check (part-box references)", 0.0, 30.0),
    _p("population_part_box_min_height_ratio", "float", "ratio to the containing box's height",
       "A detection inside a larger one is NESTED (an alternative or part-body box of the same person) only if it "
       "is at least this tall relative to the larger box. A much smaller box inside is another, farther body "
       "that the larger one occludes: both are tracked bodies (the larger is still CONTAINS_OTHER).", 0.3, N,
       "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "population",
       "R1-E (NEW); R1-E code review M2. Development-clip part boxes measured 0.36-0.58 of the host height", 0.0,
       1.0),
    _p("merged_claim_min_box_fraction", "float", "fraction of the detection's area",
       "A tracklet counts towards a MULTI_CLAIM merge only if its predicted box fills at least this fraction of "
       "the detection (and it was observed within population_max_gap_s). A small far body behind a large near "
       "body fills little of the near body's box: that is occlusion, not one box for two bodies.", 0.25, N,
       "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "population",
       "R1-E (NEW); D11; R1-E development-clip population check (occluders were flagged as merges)", 0.0, 1.0),
    _p("population_track_id_digits", "int", "digits",
       "Zero-padded width of the numeric part of a population tracklet id (prefix + number).", 3, H,
       "VERIFIED_OUTPUT", "CONTRACT_VERIFIED", "CONTRACT", "population",
       "historical online ids T001, T064, T076 (R0 plan; blind_r2_009, shot_close_range)", 1, 12),

    # ---- merged-detection guard (D11) --------------------------------------------------------
    _p("merged_area_ratio", "float", "ratio to a body's established recent median area",
       "A box larger than this multiple of the established recent median area of the body it continues (the "
       "target's in R1-F; a population tracklet's in R1-E) is merged-suspect.", 1.6, N,
       "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "merged", "D11; F2 (a historical merged box for two players measured x3.1 the target area)",
       1.0, 100.0),
    _p("merged_uncorroborated_area_ratio", "float", "ratio to a body's established recent median area",
       "Size-only merge evidence (AREA_RATIO / ASPECT_WIDE) on a single body continuing its own tracklet, with no "
       "possible second body touching its box, is a SIZE_ANOMALY (abstained on, not closed; see merged_anomaly_max_s) "
       "unless the box is at least this multiple of the reference area: a growth no single body's pose explains "
       "(two bodies side by side are about x2).", 2.5, N, "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "merged",
       "NEW_R2 (R2 DEV remediation F3). Development footage: a single isolated body's pose widened its box up to x1.65 "
       "its 1 s median area at constant height; F2 historical merged box x3.1", 1.0, 100.0),
    _p("merged_anomaly_max_s", "float", "s",
       "A SIZE_ANOMALY (single body, no possible second body, below merged_uncorroborated_area_ratio) that has not "
       "ended after this long, and whose box is not near its held single-body size, is treated as a merge "
       "(SUSTAINED_SIZE_ANOMALY; the tracklet closes; merged region and lineage as for any merge). Seconds, so the rule "
       "is frame-rate free.", 1.0, N, "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "merged",
       "NEW_R2 (R2 DEV remediation F3, reviews H1/H3): longer than a single limb extension or stride; an unseen second "
       "body that stays merged is still merged after this long. Not calibrated: to be measured on DEV", 0.0, 10.0),
    _p("merged_anomaly_clear_area_ratio", "float", "ratio to the held single-body reference",
       "A SIZE_ANOMALY ends only on boxes at most this large (area, and aspect relative to the reference) against the "
       "held single-body size, not partial and not at the image edge, sustained for merged_anomaly_clear_s. A union "
       "of two bodies stays clearly wider than one body even on its narrowest frames; a single body's stride or limb "
       "extension returns close to its normal size.", 1.25, N, "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "merged",
       "NEW_R2 (R2 DEV remediation F3, review round 2 H1: a union alternating x2.0 / x1.45 ended the anomaly on its "
       "x1.45 frames). Not calibrated", 1.0, 10.0),
    _p("merged_anomaly_clear_s", "float", "s",
       "How long a SIZE_ANOMALY's box must stay near its held single-body size (merged_anomaly_clear_area_ratio) before "
       "the anomaly ends. One near-size or partial frame never ends it.", 0.2, N, "DESIGN_CHOICE", UNV,
       "BEHAVIOURAL", "merged", "NEW_R2 (R2 DEV remediation F3, review round 2 H1). Not calibrated", 0.0, 10.0),
    _p("merged_aspect_band", "pair", "ratio to recent median aspect (w/h)",
       "Aspect ratio outside this band relative to the target's median is merged-suspect.", (0.6, 1.6), N,
       "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "merged", "D11", 0.0, 100.0),
    _p("merged_overlap_min", "float", "fraction of predicted box area",
       "A box covering at least this fraction of the predicted boxes of two or more distinct tracked bodies (D11: "
       "the target's and a competitor's) is merged-suspect (subject to merged_claim_min_box_fraction).", 0.3, N, "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "merged", "D11", 0.0, 1.0),

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
    _p("identity_contact_cover_min", "float", "fraction of the smaller box's area",
       "A competitor is IN CONTACT with a target candidate box if either box covers at least this fraction of "
       "the other. A competitor in contact whose appearance is unknown (no bank yet, e.g. provisional) blocks the "
       "candidate: contrast cannot be verified (D10).", 0.1, N, "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "identity",
       "R1-F (NEW); D08, D10", 0.0, 1.0),
    _p("target_known_other_min_s", "float", "s",
       "A body seen beside the confirmed target becomes a KNOWN OTHER person (never reacquired as the target "
       "while its tracklet lives) at the seed frame, or once the target has been confirmed on the same tracklet, by "
       "continuation, for at least this long: one trusted frame is not enough to rule a body out.", 0.2, N, "DESIGN_CHOICE",
       UNV, "BEHAVIOURAL", "identity", "R1-F (NEW); R1-F code review M2", 0.0, 30.0),
    _p("target_bank_sample_interval_s", "float", "s",
       "Minimum time between samples offered to the target appearance bank.", 0.25, N, "DESIGN_CHOICE", UNV,
       "BEHAVIOURAL", "bank", "R1-F (NEW); D12", 0.0, 30.0),
    _p("reacq_appearance_floor", "float", "composite similarity (0-1)",
       "Appearance floor during reacquisition (stricter than ordinary association).", 0.85, N,
       "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "reacquisition",
       "D15. Raised from 0.62 to appearance_floor + 0.05 when R1-D calibrated appearance_floor to 0.80 (the "
       "cross-check requires reacquisition to stay stricter). To be revisited in R1-F", 0.0, 1.0),
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
    _p("cut_flow_inlier_fraction_max", "float", "fraction of tracked background features",
       "Optical flow whose RANSAC inliers are at most this fraction of the background features successfully "
       "tracked into this frame (features landing on a person box are discarded, not counted as failures) counts "
       "as flow failure.", 0.2, N, "DESIGN_CHOICE",
       UNV, "BEHAVIOURAL", "camera", "D16", 0.0, 1.0),
    _p("flow_max_corners", "int", "features", "Maximum background features for global-motion estimation.",
       300, N, "DESIGN_CHOICE", "NOT_APPLICABLE", "RUNTIME", "camera", "D16", 10, 10000),
    _p("flow_quality_level", "float", "OpenCV corner quality (0-1)", "goodFeaturesToTrack quality level.",
       0.01, N, "DESIGN_CHOICE", "NOT_APPLICABLE", "RUNTIME", "camera", "D16", 0.0, 1.0),
    _p("flow_min_distance_px", "float", "px (flow working resolution)",
       "Minimum spacing between flow features, at the flow working width (flow_work_width_px).",
       8.0, N, "DESIGN_CHOICE", "NOT_APPLICABLE", "RUNTIME", "camera", "D16", 1.0, 1000.0),
    _p("flow_person_mask_pad_h", "float", "h", "Padding around person boxes excluded from background flow.",
       0.1, N, "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "camera", "D16", 0.0, 5.0),
    _p("cut_hist_bins", "int", "bins", "Number of grey-level bins of the cut-detection histogram.", 32, N,
       "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "camera", "D16; R1-E (NEW)", 2, 256),
    _p("cut_work_width_px", "int", "px (cut-histogram working width)",
       "Frames are resized to this width before the cut histogram (normalised, so resolution-independent).",
       160, N, "DESIGN_CHOICE", "NOT_APPLICABLE", "RUNTIME", "camera", "R1-E (NEW)", 8, 4096),
    _p("flow_work_width_px", "int", "px (flow working width)",
       "Frames are resized to this width for global-motion flow, so every flow pixel parameter means the same "
       "thing at any source resolution. The estimate is scaled back to source pixels.", 640, N,
       "DESIGN_CHOICE", "NOT_APPLICABLE", "RUNTIME", "camera", "R1-E (NEW); R1 approval §29 (scale-normalised)",
       32, 8192),
    _p("flow_lk_win_px", "int", "px (flow working resolution)", "Lucas-Kanade window size.", 21, N,
       "DESIGN_CHOICE", "NOT_APPLICABLE", "RUNTIME", "camera", "R1-E (NEW); OpenCV default family", 3, 255),
    _p("flow_lk_max_level", "int", "pyramid levels", "Lucas-Kanade pyramid levels above the base image.", 3, N,
       "DESIGN_CHOICE", "NOT_APPLICABLE", "RUNTIME", "camera", "R1-E (NEW)", 0, 10),
    _p("flow_ransac_reproj_px", "float", "px (flow working resolution)",
       "RANSAC inlier threshold for the global similarity transform.", 2.0, N, "DESIGN_CHOICE",
       "NOT_APPLICABLE", "RUNTIME", "camera", "R1-E (NEW)", 0.1, 100.0),
    _p("flow_min_features", "int", "features",
       "Fewer background features (found, or successfully tracked) than this means the camera motion cannot "
       "be estimated.", 30, N, "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "camera", "R1-E (NEW)", 3, 10000),
    _p("camera_max_shift_frac_per_s", "float", "fraction of frame width per s",
       "An estimated image-centre shift faster than this is implausible (estimate rejected: UNKNOWN).", 3.0, N,
       "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "camera", "R1-E (NEW): bound on camera motion", 0.0, 1000.0),
    _p("camera_max_scale_ratio_per_s", "float", "ratio per s",
       "An estimated zoom faster than this factor per second is implausible (estimate rejected: UNKNOWN).",
       3.0, N, "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "camera", "R1-E (NEW): bound on camera zoom", 1.0, 1000.0),
    _p("camera_max_rotation_rad_per_s", "float", "rad per s",
       "An estimated image rotation faster than this is implausible (estimate rejected: UNKNOWN).", 0.5, N,
       "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "camera", "R1-E (NEW): bound on camera roll", 0.0, 100.0),
    _p("cut_residual_min", "float", "normalised residual (0-1)",
       "A fitted camera transform whose normalised photometric residual mean|a'-b'| / (mean|a'| + mean|b'|) "
       "(a = previous frame warped onto this one, b = this frame; background only, blurred, mean-centred) is at "
       "least this does not explain the image change and is rejected (HIGH_RESIDUAL; e.g. a static broadcast "
       "overlay dominating the features across a cut). Synthetic: true pan/zoom ~0.02, unrelated shots ~0.5.",
       0.25, N, "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "camera", "R1-E (NEW); R1-E code review H2", 0.0, 1.0),
    _p("cut_residual_blur_px", "float", "px (flow working resolution)",
       "Gaussian sigma applied to both frames before the photometric residual (suppresses noise and "
       "interpolation error).", 1.5, N, "DESIGN_CHOICE", "NOT_APPLICABLE", "RUNTIME", "camera", "R1-E (NEW)",
       0.1, 50.0),
    _p("camera_unknown_breaks_continuity", "bool", "flag",
       "If True, a frame whose camera motion cannot be estimated (UNKNOWN) breaks position continuity exactly "
       "like a detected cut (fail closed: a cut between similar-looking shots is not distinguishable from a "
       "flow failure).", True, N, "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "camera",
       "R1-E (NEW); R1 approval §20 fail closed", allowed=(True, False)),

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
    if v["contact_memory_s"] > v["population_closed_memory_s"]:
        errs.append("contact_memory_s must be <= population_closed_memory_s (the population keeps closed "
                    "competitors, with their banks, only that long)")
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
    if v["merged_uncorroborated_area_ratio"] < v["merged_area_ratio"]:
        errs.append("merged_uncorroborated_area_ratio must be >= merged_area_ratio")
    if v["merged_anomaly_clear_area_ratio"] >= v["merged_area_ratio"]:
        errs.append("merged_anomaly_clear_area_ratio must be < merged_area_ratio (a merge-sized box never clears)")
    if v["population_part_birth_containment"] > v["population_nested_containment"]:
        errs.append("population_part_birth_containment must be <= population_nested_containment")
    if v["population_merged_region_hold_s"] > v["population_closed_memory_s"]:
        errs.append("population_merged_region_hold_s must be <= population_closed_memory_s")
    if v["population_max_gap_s"] > v["population_max_unmatched_s"]:
        errs.append("population_max_gap_s must be <= population_max_unmatched_s")
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
    elif t == "tuple":
        if (not isinstance(value, (tuple, list)) or len(value) == 0 or
                any(isinstance(e, bool) or not isinstance(e, (int, float)) for e in value)):
            return f"{p.name}: expected a non-empty sequence of numbers, got {value!r}"
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
    if t in ("pair", "tuple"):
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

# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R0/R0_TRACKER_INTERFACE_MAP.md §0-§6 (historical Pass-1 JSON contract, recovered from frozen
#                 historical exemplars HE1 (88246efe) and HE2 (40fa0df1) plus surviving consumers event_adapter.py and
#                 actor_attribution.py); R1/R1_DESIGN_DECISIONS.md D01, D05, D06, D07, D19, D20
"""
player_tracker_rb.output_schema: the Pass-1 tracking JSON contract, historical part plus additive v2 part.

The file is `{video_basename}_mo_tracking_results.json`:
    {"summary": {...15 historical keys..., "v2": {...}},
     "frame_log": [{...7 historical keys..., "v2": {...}}, ...],
     "final_tracks": [ track_record, ... ]}

Each track record holds the 29 historical fields, plus an optional "v2" block.

Rules (R1 approval §17):
  * Historical keys keep their historical names, types and meaning. Nothing is silently redefined.
  * All new R1 information sits only under keys named "v2". `strip_v2()` removes them, and the result must
    still validate as a historical document (`validate_document(strip_v2(doc), require_v2=False)`).
  * The top-level layout is unchanged: no new top-level keys.

This module only defines and checks the contract. It makes no tracking decisions.
"""
from __future__ import annotations

import copy
import json
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .config import TrackerConfig

SCHEMA_NAME = "spana_mo_tracking_results"
V2_SCHEMA_VERSION = "2.0.0-r1a"
TRACKER_NAME = "player_tracker_rb"
V2_KEY = "v2"

# ---------------------------------------------------------------------------------------------
# Historical contract (VERIFIED_OUTPUT from both frozen exemplars unless noted)
# ---------------------------------------------------------------------------------------------
TOP_LEVEL_KEYS: Tuple[str, ...] = ("summary", "frame_log", "final_tracks")

SUMMARY_KEYS: Tuple[str, ...] = (
    "video", "start_frame", "fps", "processed_frames", "processing_time_s", "selected_state_pct",
    "n_tracks_ever_created", "n_tracks_final_active", "clip_output", "final_confirmed_count",
    "final_provisional_count", "spawn_guard_stats", "identity_conflicts_logged",
    "identity_conflict_counts", "health_metrics",
)
# Summary keys that surviving Pass-2 code reads at runtime (ATP:270,308,325; BF:320-322).
SUMMARY_RUNTIME_REQUIRED: Tuple[str, ...] = ("fps", "start_frame", "processed_frames", "identity_conflict_counts")

HEALTH_METRIC_KEYS: Tuple[str, ...] = (
    "frames_processed", "total_tracks_spawned", "new_tracks_per_frame", "tracks_per_100_frames",
    "provisional_to_confirmed_ratio", "median_track_lifetime_frames", "duplicate_likelihood",
    "live_association_eligible_dormant_count", "pruned_from_live_association_count",
)

FRAME_KEYS: Tuple[str, ...] = ("frame", "is_cut", "camera_shift", "n_detections", "n_live_tracks",
                               "overlap_contests", "tracks")

# The 29 historical per-track fields, in the historical order, with their JSON types.
# "box" = list of 4 numbers [x1, y1, x2, y2] in px. "opt_str" = str or None. "null" = always None.
TRACK_FIELDS: Tuple[Tuple[str, str], ...] = (
    ("track_id", "str"), ("identity_id", "str"), ("online_track_id", "str"), ("source", "str"),
    ("state", "str"), ("age", "int"), ("last_seen_frame", "int"), ("bbox", "box"), ("team", "str"),
    ("team_confidence", "num"), ("number_candidate", "opt_str"), ("selected_player", "bool"),
    ("detection_confidence", "num"), ("motion_confidence", "num"), ("appearance_confidence", "num"),
    ("reid_confidence", "num"), ("overall_track_confidence", "num"), ("visibility_confidence", "num"),
    ("occlusion_status", "str"), ("merged_into", "merged"), ("position_confidence", "num"),
    ("track_continuity_confidence", "num"), ("identity_confidence", "num"),
    ("selected_player_confidence", "num"), ("observed_position_box", "box"),
    ("position_is_confirmed", "bool"), ("position_source", "str"),
    ("frames_since_observed_position", "int"), ("frames_since_confirmed_identity", "int"),
)
TRACK_FIELD_NAMES: Tuple[str, ...] = tuple(n for n, _ in TRACK_FIELDS)
CONFIDENCE_FIELDS: Tuple[str, ...] = tuple(n for n, t in TRACK_FIELDS if t == "num")

# Historical online state vocabulary (7 observed values; TERMINATED appears only as a pipeline default).
HISTORICAL_STATES: Tuple[str, ...] = ("ACTIVE", "PROVISIONAL", "AMBIGUOUS", "OCCLUDED", "TEMPORARILY_LOST",
                                      "REIDENTIFYING", "OUT_OF_FRAME")
# final_tracks additionally uses TERMINATED (HE1 20 of 35, HE2 45 of 70 final records; never in frame_log),
# and there merged_into may name the absorbing track (HE1 'T014' etc.; always None in frame_log).
# Both are VERIFIED_OUTPUT facts found by validating the frozen exemplars in R1-A.
FINAL_TRACK_EXTRA_STATES: Tuple[str, ...] = ("TERMINATED",)
OCCLUSION_STATUSES: Tuple[str, ...] = ("NONE", "PARTIAL", "FULL", "AMBIGUOUS", "WEAK_SPATIAL_RECOVERY")
POSITION_SOURCES: Tuple[str, ...] = ("CONFIRMED_DETECTION", "AMBIGUOUS_DETECTION", "PREDICTED",
                                     "WEAK_SPATIAL_RECOVERY")
TEAMS: Tuple[str, ...] = ("TEAM_A", "TEAM_B", "UNKNOWN")

# Historical online state -> legacy TrackingSnapshot state, with the evidence for each row
# (R0_TRACKER_INTERFACE_MAP §5.2). Pass 2 trusts only TRACKING. event_adapter defaults unknown keys to LOST.
LEGACY_STATE_EVIDENCE: Mapping[str, Tuple[str, str]] = {
    "ACTIVE": ("TRACKING", "VERIFIED_OUTPUT"),
    "OCCLUDED": ("OCCLUDED", "VERIFIED_OUTPUT"),
    "AMBIGUOUS": ("OCCLUDED", "VERIFIED_OUTPUT"),
    "TEMPORARILY_LOST": ("LOST", "DOCUMENTED"),
    "REIDENTIFYING": ("REACQUIRING", "DOCUMENTED"),
    "OUT_OF_FRAME": ("LOST", "INFERRED"),          # via the .get(..., "LOST") default only
}
LEGACY_STATES: Tuple[str, ...] = ("TRACKING", "OCCLUDED", "REACQUIRING", "LOST")
LEGACY_DEFAULT_STATE = "LOST"

# ---------------------------------------------------------------------------------------------
# R1 (v2) vocabulary
# ---------------------------------------------------------------------------------------------
RB_STATES: Tuple[str, ...] = ("TRACKING", "OCCLUDED", "AMBIGUOUS", "LOST", "REACQUIRING")
OBSERVATION_SOURCES: Tuple[str, ...] = ("DETECTION", "HELD", "PREDICTED", "NONE")
RECORD_KINDS: Tuple[str, ...] = ("TARGET", "POPULATION")
UNCERTAINTY_REASONS: Tuple[str, ...] = (
    "OCCLUDER_OVERLAP", "COMPETING_CANDIDATE", "MERGED_DETECTION_SUSPECTED", "REACQUISITION_UNVERIFIED",
    "BELOW_APPEARANCE_FLOOR", "INSUFFICIENT_CONTRAST", "MOTION_VETO", "NO_CANDIDATE_IN_GATE",
    "TARGET_OUT_OF_FRAME", "CAMERA_CUT", "LOW_QUALITY_CROP", "AGED_OUT",
)

# D05: R1 state -> historical online `state` written to the record.
# LOST splits on whether the predicted box is still inside the image.
def historical_state_for(rb_state: str, predicted_inside_image: bool = True) -> str:
    if rb_state == "TRACKING":
        return "ACTIVE"
    if rb_state == "OCCLUDED":
        return "OCCLUDED"
    if rb_state == "AMBIGUOUS":
        return "AMBIGUOUS"
    if rb_state == "REACQUIRING":
        return "REIDENTIFYING"
    if rb_state == "LOST":
        return "TEMPORARILY_LOST" if predicted_inside_image else "OUT_OF_FRAME"
    raise ValueError(f"unknown R1 state {rb_state!r}")


def historical_detection_confidence(cfg: TrackerConfig, state: str, position_source: str) -> float:
    """The historical, quantised detection_confidence (NOT a detector score). VERIFIED_OUTPUT: quantised by
    position_source, except REIDENTIFYING records, which were written 0.0 although CONFIRMED_DETECTION."""
    if state == "REIDENTIFYING":
        return float(cfg.get("detection_confidence_reidentifying"))
    return float(cfg.get("detection_confidence_by_position_source")[position_source])


def legacy_state_for(historical_state: str) -> str:
    """What the surviving event_adapter would produce from this historical state (unknown -> LOST)."""
    row = LEGACY_STATE_EVIDENCE.get(historical_state)
    return row[0] if row else LEGACY_DEFAULT_STATE


# Historical (state, occlusion_status, position_source) written for each R1 situation. Every triple here was
# observed in HE1 (R0_TRACKER_INTERFACE_MAP §4, joint table), so R1 never emits a combination unseen in
# the historical outputs.
OBSERVED_HISTORICAL_TRIPLES: Tuple[Tuple[str, str, str], ...] = (
    ("ACTIVE", "NONE", "CONFIRMED_DETECTION"),
    ("PROVISIONAL", "NONE", "CONFIRMED_DETECTION"),
    ("TEMPORARILY_LOST", "FULL", "PREDICTED"),
    ("ACTIVE", "WEAK_SPATIAL_RECOVERY", "WEAK_SPATIAL_RECOVERY"),
    ("AMBIGUOUS", "AMBIGUOUS", "AMBIGUOUS_DETECTION"),
    ("OCCLUDED", "PARTIAL", "PREDICTED"),
    ("OUT_OF_FRAME", "FULL", "PREDICTED"),
    ("REIDENTIFYING", "FULL", "CONFIRMED_DETECTION"),
)
# The triples R1 itself writes for the TARGET record, by R1 state (and LOST in/out of image).
# ACTIVE+WEAK_SPATIAL_RECOVERY is deliberately never written: R1 has no weak recovery path.
TARGET_TRIPLE_BY_RB_STATE: Mapping[Tuple[str, bool], Tuple[str, str, str]] = {
    ("TRACKING", True): ("ACTIVE", "NONE", "CONFIRMED_DETECTION"),
    ("OCCLUDED", True): ("OCCLUDED", "PARTIAL", "PREDICTED"),
    ("AMBIGUOUS", True): ("AMBIGUOUS", "AMBIGUOUS", "AMBIGUOUS_DETECTION"),
    ("REACQUIRING", True): ("REIDENTIFYING", "FULL", "CONFIRMED_DETECTION"),
    ("LOST", True): ("TEMPORARILY_LOST", "FULL", "PREDICTED"),
    ("LOST", False): ("OUT_OF_FRAME", "FULL", "PREDICTED"),
}
# Population tracklets (non-target bodies). They are written with the historical non-selected look:
# detected this frame -> ACTIVE/NONE/CONFIRMED_DETECTION; coasting -> TEMPORARILY_LOST/FULL/PREDICTED.
POPULATION_TRIPLE_DETECTED = ("ACTIVE", "NONE", "CONFIRMED_DETECTION")
POPULATION_TRIPLE_COASTING = ("TEMPORARILY_LOST", "FULL", "PREDICTED")

# v2 block fields: name -> type. Types: "bool", "int", "num", "str", "opt_int", "opt_num", "opt_str",
# "opt_box", "box", "list_str", "enum:<VOCAB>".
V2_TARGET_FIELDS: Mapping[str, str] = {
    "record_kind": "enum:RECORD_KINDS",
    "rb_state": "enum:RB_STATES",
    "observation_source": "enum:OBSERVATION_SOURCES",
    "bbox_is_observed": "bool",
    "bbox_is_held": "bool",
    "bbox_is_predicted": "bool",
    "held_age_s": "num",
    "last_confirmed_frame": "opt_int",
    "time_since_confirmed_s": "opt_num",
    "observed_or_contested_box": "opt_box",
    "observation_tracklet_id": "opt_str",
    "target_episode": "int",
    "identity_trusted": "bool",
    "identity_confidence_rb": "num",
    "identity_uncertainty_reasons": "list_reason",
    "occluder_track_ids": "list_str",
    "merged_suspected": "bool",
    "raw_detection_conf": "opt_num",
    "reidentified_this_frame": "bool",
    "target_similarity": "opt_num",
    "best_competitor_similarity": "opt_num",
}
V2_POPULATION_FIELDS: Mapping[str, str] = {
    "record_kind": "enum:RECORD_KINDS",
    "detected_this_frame": "bool",
    "raw_detection_conf": "opt_num",
    "is_competitor": "bool",
    "first_frame": "int",
}
V2_FRAME_FIELDS: Mapping[str, str] = {
    "video_frame_index": "int",
    "timestamp_s": "num",
    "frame_thumb_sha1": "str",
    "cut_detected": "bool",
    "camera_shift_h": "opt_pair",
    "target_height_px": "opt_num",
}
V2_SUMMARY_FIELDS: Mapping[str, str] = {
    "schema": "str",
    "schema_version": "str",
    "tracker": "str",
    "tracker_version": "str",
    "provenance": "dict",
    "config_hash": "str",
    "reacquisition_events": "list_dict",
    "rb_state_pct": "dict",
    "notes": "list_str",
}
PROVENANCE_FIELDS: Mapping[str, str] = {
    "video_path": "str",
    "video_sha256": "str",
    "fps": "num",
    "frame_count": "opt_int",
    "decode_mode": "str",
    "seed_frame": "int",
    "seed_bbox": "box",
    "models": "list_dict",
    "config_hash": "str",
    "git_commit": "opt_str",
    "git_dirty": "opt_bool",
    "schema_version": "str",
    "tracker_version": "str",
    "run_started_utc": "str",
}
REACQ_EVENT_FIELDS: Mapping[str, str] = {
    "episode": "int",
    "lost_from_frame": "int",
    "uncertain_from_frame": "int",
    "reacq_started_frame": "int",
    "reacq_accepted_frame": "int",
    "uncertain_duration_s": "num",
    "evidence": "dict",
}
DECODE_MODE = "SEQUENTIAL_NO_RESUME"

_VOCABS = {"RECORD_KINDS": RECORD_KINDS, "RB_STATES": RB_STATES, "OBSERVATION_SOURCES": OBSERVATION_SOURCES}


# ---------------------------------------------------------------------------------------------
# Type helpers
# ---------------------------------------------------------------------------------------------
def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _is_box(x: Any) -> bool:
    return isinstance(x, list) and len(x) == 4 and all(_is_num(e) for e in x)


def _check_type(kind: str, x: Any) -> bool:
    if kind == "str":
        return isinstance(x, str)
    if kind == "opt_str":
        return x is None or isinstance(x, str)
    if kind == "int":
        return isinstance(x, int) and not isinstance(x, bool)
    if kind == "opt_int":
        return x is None or (isinstance(x, int) and not isinstance(x, bool))
    if kind == "num":
        return _is_num(x)
    if kind == "opt_num":
        return x is None or _is_num(x)
    if kind == "bool":
        return isinstance(x, bool)
    if kind == "opt_bool":
        return x is None or isinstance(x, bool)
    if kind == "null":
        return x is None
    if kind == "box":
        return _is_box(x)
    if kind == "opt_box":
        return x is None or _is_box(x)
    if kind == "opt_pair":
        return x is None or (isinstance(x, list) and len(x) == 2 and all(_is_num(e) for e in x))
    if kind == "list_str":
        return isinstance(x, list) and all(isinstance(e, str) for e in x)
    if kind == "list_reason":
        return isinstance(x, list) and all(e in UNCERTAINTY_REASONS for e in x)
    if kind == "list_dict":
        return isinstance(x, list) and all(isinstance(e, dict) for e in x)
    if kind == "dict":
        return isinstance(x, dict)
    if kind.startswith("enum:"):
        return x in _VOCABS[kind[len("enum:"):]]
    raise ValueError(f"unknown schema type {kind!r}")


def _check_fields(spec: Mapping[str, str], obj: Any, where: str, errors: List[str], exact: bool = True) -> None:
    if not isinstance(obj, dict):
        errors.append(f"{where}: expected an object")
        return
    missing = [k for k in spec if k not in obj]
    if missing:
        errors.append(f"{where}: missing {missing}")
    if exact:
        extra = [k for k in obj if k not in spec]
        if extra:
            errors.append(f"{where}: unexpected {extra}")
    for k, kind in spec.items():
        if k in obj and not _check_type(kind, obj[k]):
            errors.append(f"{where}.{k}: bad value {obj[k]!r} for type {kind}")


# ---------------------------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------------------------
def validate_track_record(rec: Any, where: str = "track", require_v2: bool = False,
                          cfg: Optional[TrackerConfig] = None, context: str = "frame_log") -> List[str]:
    """Check one track record: 29 historical fields with historical types and vocabularies, plus the
    optional v2 block and its invariants. context is "frame_log" or "final_tracks": the historical contract
    differs slightly between them (TERMINATED state and non-null merged_into occur only in final_tracks)."""
    if context not in ("frame_log", "final_tracks"):
        raise ValueError(f"unknown context {context!r}")
    errors: List[str] = []
    if not isinstance(rec, dict):
        return [f"{where}: expected an object"]
    allowed = set(TRACK_FIELD_NAMES) | {V2_KEY}
    missing = [n for n in TRACK_FIELD_NAMES if n not in rec]
    extra = [k for k in rec if k not in allowed]
    if missing:
        errors.append(f"{where}: missing historical fields {missing}")
    if extra:
        errors.append(f"{where}: non-historical keys outside 'v2': {extra}")
    for name, kind in TRACK_FIELDS:
        if kind == "merged":
            kind = "opt_str" if context == "final_tracks" else "null"
        if name in rec and not _check_type(kind, rec[name]):
            errors.append(f"{where}.{name}: bad value {rec[name]!r} for historical type {kind}")
    states = HISTORICAL_STATES + (FINAL_TRACK_EXTRA_STATES if context == "final_tracks" else ())
    if rec.get("state") not in states:
        errors.append(f"{where}.state: {rec.get('state')!r} not in historical vocabulary")
    if rec.get("occlusion_status") not in OCCLUSION_STATUSES:
        errors.append(f"{where}.occlusion_status: {rec.get('occlusion_status')!r} not in historical vocabulary")
    if rec.get("position_source") not in POSITION_SOURCES:
        errors.append(f"{where}.position_source: {rec.get('position_source')!r} not in historical vocabulary")
    if rec.get("team") not in TEAMS:
        errors.append(f"{where}.team: {rec.get('team')!r} not in historical vocabulary")
    for n in CONFIDENCE_FIELDS:
        v = rec.get(n)
        if _is_num(v) and not (0.0 <= float(v) <= 1.0):
            errors.append(f"{where}.{n}: {v} outside [0, 1]")
    spc, idc = rec.get("selected_player_confidence"), rec.get("identity_confidence")
    if _is_num(spc) and _is_num(idc) and isinstance(rec.get("selected_player"), bool) and context == "frame_log":
        if not rec["selected_player"] and spc != 0.0:
            errors.append(f"{where}: selected_player_confidence must be 0.0 on a non-selected record")
        if rec["selected_player"] and spc > idc:
            errors.append(f"{where}: selected_player_confidence must not exceed identity_confidence")
    if isinstance(rec.get("position_is_confirmed"), bool) and rec.get("position_source") in POSITION_SOURCES:
        if rec["position_is_confirmed"] != (rec["position_source"] == "CONFIRMED_DETECTION"):
            errors.append(f"{where}: position_is_confirmed must equal (position_source == CONFIRMED_DETECTION)")
    if V2_KEY in rec:
        if context == "final_tracks":
            errors.append(f"{where}: final_tracks records carry no v2 block (the per-frame v2 is authoritative)")
        else:
            errors.extend(_validate_track_v2(rec, where, cfg or TrackerConfig.default()))
    elif require_v2:
        errors.append(f"{where}: missing v2 block")
    return errors


def _validate_track_v2(rec: dict, where: str, cfg: TrackerConfig) -> List[str]:
    errors: List[str] = []
    v2 = rec[V2_KEY]
    if not isinstance(v2, dict):
        return [f"{where}.v2: expected an object"]
    kind = v2.get("record_kind")
    if kind == "TARGET":
        _check_fields(V2_TARGET_FIELDS, v2, f"{where}.v2", errors)
        if errors:
            return errors
        errors.extend(target_invariant_errors(rec, cfg, where))
    elif kind == "POPULATION":
        _check_fields(V2_POPULATION_FIELDS, v2, f"{where}.v2", errors)
        if errors:
            return errors
        if rec["selected_player"]:
            errors.append(f"{where}: a POPULATION record must not be selected_player")
        triple = (rec["state"], rec["occlusion_status"], rec["position_source"])
        want = POPULATION_TRIPLE_DETECTED if v2["detected_this_frame"] else POPULATION_TRIPLE_COASTING
        if triple != want:
            errors.append(f"{where}: population triple {triple} != {want}")
    else:
        errors.append(f"{where}.v2.record_kind: {kind!r} not in {RECORD_KINDS}")
    return errors


def target_invariant_errors(rec: dict, cfg: TrackerConfig, where: str = "target") -> List[str]:
    """Safety invariants for the selected target record (D05, D06, D19). These are what keep the surviving
    Pass-2 consumers safe; they are checked on every output."""
    errors: List[str] = []
    v2 = rec[V2_KEY]
    rb = v2["rb_state"]
    # selected
    if not rec["selected_player"]:
        errors.append(f"{where}: TARGET record must be selected_player")
    # identity fields
    tid = cfg.get("target_track_id")
    for k in ("track_id", "identity_id", "online_track_id"):
        if rec[k] != tid:
            errors.append(f"{where}.{k}: {rec[k]!r} != logical target id {tid!r}")
    # state mapping (D05): exact historical triple for this R1 state
    inside = rec["state"] != "OUT_OF_FRAME"
    want = TARGET_TRIPLE_BY_RB_STATE.get((rb, inside if rb == "LOST" else True))
    got = (rec["state"], rec["occlusion_status"], rec["position_source"])
    if want is None or got != want:
        errors.append(f"{where}: rb_state {rb} written as {got}, expected {want}")
    # non-TRACKING never reaches legacy TRACKING
    if rb != "TRACKING" and legacy_state_for(rec["state"]) == "TRACKING":
        errors.append(f"{where}: non-TRACKING rb_state {rb} maps to legacy TRACKING")
    # observed <=> CONFIRMED_DETECTION (in ACTIVE) <=> TRACKING (D06)
    observed = v2["bbox_is_observed"]
    if observed != (rb == "TRACKING"):
        errors.append(f"{where}: bbox_is_observed={observed} but rb_state={rb}")
    if observed != (rec["state"] == "ACTIVE" and rec["position_source"] == "CONFIRMED_DETECTION"):
        errors.append(f"{where}: bbox_is_observed must equal (state ACTIVE and CONFIRMED_DETECTION)")
    flags = (v2["bbox_is_observed"], v2["bbox_is_held"], v2["bbox_is_predicted"])
    if sum(flags) != 1:
        errors.append(f"{where}: exactly one of bbox_is_observed/held/predicted must be true, got {flags}")
    src_flag = {"DETECTION": "bbox_is_observed", "HELD": "bbox_is_held", "PREDICTED": "bbox_is_predicted"}
    src = v2["observation_source"]
    if src == "NONE" or v2[src_flag[src]] is not True:
        errors.append(f"{where}: observation_source {src} inconsistent with the bbox flags")
    if observed and v2["held_age_s"] != 0:
        errors.append(f"{where}: an observed box must have held_age_s == 0")
    if not observed and rec["frames_since_observed_position"] == 0:
        errors.append(f"{where}: an unobserved box must have frames_since_observed_position > 0")
    if observed and rec["frames_since_observed_position"] != 0:
        errors.append(f"{where}: an observed box must have frames_since_observed_position == 0")
    # identity trust (D19)
    trusted = v2["identity_trusted"]
    if trusted and rb != "TRACKING":
        errors.append(f"{where}: identity_trusted requires rb_state TRACKING")
    if trusted and v2["merged_suspected"]:
        errors.append(f"{where}: identity_trusted with merged_suspected")
    if trusted and v2["identity_uncertainty_reasons"]:
        errors.append(f"{where}: identity_trusted with uncertainty reasons {v2['identity_uncertainty_reasons']}")
    thr = cfg.get("legacy_identity_trust_threshold")
    if not trusted and rec["identity_confidence"] >= thr:
        errors.append(f"{where}: untrusted identity but identity_confidence {rec['identity_confidence']} >= {thr}")
    if not trusted and not v2["identity_uncertainty_reasons"]:
        errors.append(f"{where}: untrusted identity must name at least one uncertainty reason")
    if rec["selected_player_confidence"] != rec["identity_confidence"]:
        errors.append(f"{where}: R1 writes selected_player_confidence == identity_confidence on the target")
    if v2["merged_suspected"] and rb == "TRACKING":
        errors.append(f"{where}: merged-suspected detection cannot establish TRACKING")
    # detection_confidence keeps historical quantised semantics
    want_dc = historical_detection_confidence(cfg, rec["state"], rec["position_source"])
    if rec["detection_confidence"] != want_dc:
        errors.append(f"{where}: detection_confidence {rec['detection_confidence']} must be the historical "
                      f"quantised value {want_dc} for {rec['state']}/{rec['position_source']}")
    if v2["target_episode"] < 1:
        errors.append(f"{where}: target_episode must be >= 1")
    return errors


def validate_frame_record(fr: Any, where: str = "frame", require_v2: bool = False,
                          cfg: Optional[TrackerConfig] = None) -> List[str]:
    errors: List[str] = []
    if not isinstance(fr, dict):
        return [f"{where}: expected an object"]
    missing = [k for k in FRAME_KEYS if k not in fr]
    extra = [k for k in fr if k not in FRAME_KEYS and k != V2_KEY]
    if missing:
        errors.append(f"{where}: missing historical keys {missing}")
    if extra:
        errors.append(f"{where}: non-historical keys outside 'v2': {extra}")
    if missing:
        return errors
    if not (isinstance(fr["frame"], int) and not isinstance(fr["frame"], bool) and fr["frame"] >= 1):
        errors.append(f"{where}.frame: must be an int >= 1 (frame_log frame f = video frame start_frame + f)")
    if not isinstance(fr["is_cut"], bool):
        errors.append(f"{where}.is_cut: must be bool")
    if not (isinstance(fr["camera_shift"], list) and len(fr["camera_shift"]) == 2 and
            all(_is_num(e) for e in fr["camera_shift"])):
        errors.append(f"{where}.camera_shift: must be [dx, dy]")
    for k in ("n_detections", "n_live_tracks"):
        if not (isinstance(fr[k], int) and not isinstance(fr[k], bool) and fr[k] >= 0):
            errors.append(f"{where}.{k}: must be a non-negative int")
    oc = fr["overlap_contests"]
    if not (isinstance(oc, list) and all(isinstance(p, list) and len(p) == 2 and
                                         all(isinstance(t, str) for t in p) for p in oc)):
        errors.append(f"{where}.overlap_contests: must be a list of [track_id, track_id] pairs (never an int)")
    if not isinstance(fr["tracks"], list):
        errors.append(f"{where}.tracks: must be a list")
        return errors
    c = cfg or TrackerConfig.default()
    for i, t in enumerate(fr["tracks"]):
        errors.extend(validate_track_record(t, f"{where}.tracks[{i}]", require_v2=require_v2, cfg=c))
    n_sel = sum(1 for t in fr["tracks"] if isinstance(t, dict) and t.get("selected_player") is True)
    if n_sel > 1:
        errors.append(f"{where}: {n_sel} selected_player records (at most 1 allowed)")
    if require_v2 and n_sel != 1:
        errors.append(f"{where}: R1 output must carry exactly one selected (target) record, got {n_sel}")
    ids = [t.get("track_id") for t in fr["tracks"] if isinstance(t, dict)]
    if len(ids) != len(set(ids)):
        errors.append(f"{where}: duplicate track_id in tracks")
    if fr.get("n_live_tracks") != len(fr["tracks"]) and require_v2:
        errors.append(f"{where}: n_live_tracks must equal len(tracks) in R1 output")
    if V2_KEY in fr:
        _check_fields(V2_FRAME_FIELDS, fr[V2_KEY], f"{where}.v2", errors)
        if isinstance(fr[V2_KEY], dict) and fr[V2_KEY].get("cut_detected") is not None:
            if fr[V2_KEY]["cut_detected"] != fr["is_cut"]:
                errors.append(f"{where}: v2.cut_detected must equal is_cut")
    elif require_v2:
        errors.append(f"{where}: missing v2 block")
    return errors


def validate_summary(s: Any, require_v2: bool = False) -> List[str]:
    errors: List[str] = []
    if not isinstance(s, dict):
        return ["summary: expected an object"]
    missing = [k for k in SUMMARY_KEYS if k not in s]
    extra = [k for k in s if k not in SUMMARY_KEYS and k != V2_KEY]
    if missing:
        errors.append(f"summary: missing historical keys {missing}")
    if extra:
        errors.append(f"summary: non-historical keys outside 'v2': {extra}")
    if missing:
        return errors
    types = {"video": "str", "start_frame": "int", "fps": "num", "processed_frames": "int",
             "processing_time_s": "num", "selected_state_pct": "dict", "n_tracks_ever_created": "int",
             "n_tracks_final_active": "int", "clip_output": "opt_str", "final_confirmed_count": "int",
             "final_provisional_count": "int", "spawn_guard_stats": "dict", "identity_conflicts_logged": "int",
             "identity_conflict_counts": "dict", "health_metrics": "dict"}
    for k, kind in types.items():
        if not _check_type(kind, s[k]):
            errors.append(f"summary.{k}: bad value {s[k]!r} for type {kind}")
    if _is_num(s["fps"]) and not s["fps"] > 0:
        errors.append("summary.fps: must be > 0")
    for k in s["selected_state_pct"] if isinstance(s["selected_state_pct"], dict) else []:
        if k not in HISTORICAL_STATES:
            errors.append(f"summary.selected_state_pct: key {k!r} not a historical state")
    for k, v in (s["identity_conflict_counts"].items() if isinstance(s["identity_conflict_counts"], dict) else []):
        if not (isinstance(k, str) and k.count("|") == 1 and isinstance(v, int)):
            errors.append(f"summary.identity_conflict_counts: entry {k!r}: {v!r} must be 'Tid|Tid' -> int")
    if isinstance(s["health_metrics"], dict):
        hm_missing = [k for k in HEALTH_METRIC_KEYS if k not in s["health_metrics"]]
        if hm_missing:
            errors.append(f"summary.health_metrics: missing {hm_missing}")
    if V2_KEY in s:
        v2 = s[V2_KEY]
        _check_fields(V2_SUMMARY_FIELDS, v2, "summary.v2", errors)
        if isinstance(v2, dict):
            if isinstance(v2.get("provenance"), dict):
                _check_fields(PROVENANCE_FIELDS, v2["provenance"], "summary.v2.provenance", errors)
                prov = v2["provenance"]
                if prov.get("decode_mode") not in (None, DECODE_MODE):
                    errors.append(f"summary.v2.provenance.decode_mode must be {DECODE_MODE}")
                if prov.get("config_hash") is not None and prov.get("config_hash") != v2.get("config_hash"):
                    errors.append("summary.v2.provenance.config_hash must equal summary.v2.config_hash")
            for i, ev in enumerate(v2.get("reacquisition_events") or []):
                _check_fields(REACQ_EVENT_FIELDS, ev, f"summary.v2.reacquisition_events[{i}]", errors)
            for k in (v2.get("rb_state_pct") or {}):
                if k not in RB_STATES:
                    errors.append(f"summary.v2.rb_state_pct: key {k!r} not an R1 state")
    elif require_v2:
        errors.append("summary: missing v2 block")
    return errors


def validate_document(doc: Any, require_v2: bool = False, cfg: Optional[TrackerConfig] = None) -> List[str]:
    """Validate a whole tracking JSON document. require_v2=False checks the historical contract only and
    accepts both frozen historical exemplars; require_v2=True additionally checks every R1 rule."""
    if not isinstance(doc, dict):
        return ["document: expected an object"]
    errors: List[str] = []
    if sorted(doc.keys()) != sorted(TOP_LEVEL_KEYS):
        errors.append(f"document: top-level keys must be exactly {list(TOP_LEVEL_KEYS)}, got {list(doc.keys())}")
        return errors
    c = cfg or TrackerConfig.default()
    errors.extend(validate_summary(doc["summary"], require_v2=require_v2))
    fl = doc["frame_log"]
    if not isinstance(fl, list):
        return errors + ["frame_log: must be a list"]
    for i, fr in enumerate(fl):
        errors.extend(validate_frame_record(fr, f"frame_log[{i}]", require_v2=require_v2, cfg=c))
    frames = [fr.get("frame") for fr in fl if isinstance(fr, dict)]
    if frames and frames != list(range(1, len(frames) + 1)):
        errors.append("frame_log: frames must run 1, 2, ..., N with no gaps (frame f = video frame start_frame + f)")
    s = doc["summary"]
    if isinstance(s, dict) and isinstance(s.get("processed_frames"), int) and s["processed_frames"] != len(fl):
        errors.append(f"summary.processed_frames {s['processed_frames']} != len(frame_log) {len(fl)}")
    ft = doc["final_tracks"]
    if not isinstance(ft, list):
        errors.append("final_tracks: must be a list")
    else:
        for i, t in enumerate(ft):
            errors.extend(validate_track_record(t, f"final_tracks[{i}]", require_v2=False, cfg=c,
                                                context="final_tracks"))
        n_sel_final = sum(1 for t in ft if isinstance(t, dict) and t.get("selected_player") is True)
        if n_sel_final > 1:
            errors.append(f"final_tracks: {n_sel_final} selected_player records (at most 1 allowed)")
    if require_v2:
        errors.extend(_v2_document_checks(doc, c))
    return errors


def _v2_document_checks(doc: dict, cfg: TrackerConfig) -> List[str]:
    errors: List[str] = []
    s = doc["summary"]
    v2 = s.get(V2_KEY) or {}
    prov = v2.get("provenance") or {}
    if _is_num(prov.get("fps")) and _is_num(s.get("fps")) and float(prov["fps"]) != float(s["fps"]):
        errors.append("summary.v2.provenance.fps must equal summary.fps")
    if isinstance(prov.get("seed_frame"), int) and prov.get("seed_frame") != s.get("start_frame"):
        errors.append("summary.v2.provenance.seed_frame must equal summary.start_frame")
    for i, fr in enumerate(doc["frame_log"]):
        fv2 = fr.get(V2_KEY) if isinstance(fr, dict) else None
        if isinstance(fv2, dict) and isinstance(fv2.get("video_frame_index"), int):
            if fv2["video_frame_index"] != s["start_frame"] + fr["frame"]:
                errors.append(f"frame_log[{i}]: v2.video_frame_index must equal start_frame + frame")
    # Reacquisition events must agree with the episode sequence of the target.
    episodes = []
    for fr in doc["frame_log"]:
        for t in fr.get("tracks", []):
            if isinstance(t, dict) and t.get("selected_player") and isinstance(t.get(V2_KEY), dict):
                episodes.append(t[V2_KEY].get("target_episode"))
    if episodes:
        if episodes[0] != 1:
            errors.append("target_episode must start at 1")
        increments = sum(1 for a, b in zip(episodes, episodes[1:]) if b != a)
        if any(isinstance(a, int) and isinstance(b, int) and b not in (a, a + 1)
               for a, b in zip(episodes, episodes[1:])):
            errors.append("target_episode may only stay the same or increase by 1")
        n_events = len(v2.get("reacquisition_events") or [])
        if increments != n_events:
            errors.append(f"{increments} episode increments but {n_events} reacquisition_events")
    return errors


# ---------------------------------------------------------------------------------------------
# strip / canonical serialisation
# ---------------------------------------------------------------------------------------------
def strip_v2(doc: dict) -> dict:
    """Deep copy with every v2 block removed. The result must validate as a historical document."""
    out = copy.deepcopy(doc)
    out.get("summary", {}).pop(V2_KEY, None)
    for fr in out.get("frame_log", []):
        fr.pop(V2_KEY, None)
        for t in fr.get("tracks", []):
            t.pop(V2_KEY, None)
    for t in out.get("final_tracks", []):
        t.pop(V2_KEY, None)
    return out


def dumps_canonical(doc: dict) -> str:
    """Deterministic JSON text (D20): sorted keys, fixed separators, no NaN/Infinity, newline at end.
    Float rounding is applied when records are built (see the builders below), not here."""
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n"


# ---------------------------------------------------------------------------------------------
# Builders: the only way R1 code should create records, so rounding and field order are uniform.
# ---------------------------------------------------------------------------------------------
def _round_box(box: Sequence[float], dp: int) -> List[float]:
    if len(box) != 4:
        raise ValueError(f"box must have 4 numbers, got {box!r}")
    return [round(float(v), dp) for v in box]


def _round_conf(x: float, dp: int) -> float:
    v = round(float(x), dp)
    if not (0.0 <= v <= 1.0):
        raise ValueError(f"confidence {x!r} outside [0, 1]")
    return v


def build_track_record(cfg: TrackerConfig, *, track_id: str, state: str, age: int, last_seen_frame: int,
                       bbox: Sequence[float], observed_position_box: Sequence[float], occlusion_status: str,
                       position_source: str, selected_player: bool, confidences: Mapping[str, float],
                       frames_since_observed_position: int, frames_since_confirmed_identity: int,
                       team: str = "UNKNOWN", number_candidate: Optional[str] = None,
                       v2: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Build one record with the 29 historical fields in historical order, then the optional v2 block.

    `confidences` supplies the historical confidence fields except detection_confidence (always the historical
    quantised value, see historical_detection_confidence) and selected_player_confidence. Historically that was
    0.0 on every non-selected record, equal to identity_confidence on ACTIVE selected records, and lower than it
    (about x0.85 OCCLUDED, x0.55 AMBIGUOUS; formula UNKNOWN) otherwise. R1 writes identity_confidence for the
    selected record in every state (allowed: never above identity_confidence) and 0.0 otherwise.
    Missing confidence fields raise.
    """
    bdp = cfg.get("box_round_dp")
    cdp = cfg.get("confidence_round_dp")
    need = [n for n in CONFIDENCE_FIELDS if n not in ("detection_confidence", "selected_player_confidence")]
    missing = [n for n in need if n not in confidences]
    if missing:
        raise ValueError(f"missing confidence fields {missing}")
    unexpected = [n for n in confidences if n not in need]
    if unexpected:
        raise ValueError(f"unexpected confidence fields {unexpected}")
    if position_source not in POSITION_SOURCES:
        raise ValueError(f"unknown position_source {position_source!r}")
    det_conf = historical_detection_confidence(cfg, state, position_source)
    idc = _round_conf(confidences["identity_confidence"], cdp)
    rec: Dict[str, Any] = {
        "track_id": track_id, "identity_id": track_id, "online_track_id": track_id,
        "source": cfg.get("track_source_label"), "state": state, "age": int(age),
        "last_seen_frame": int(last_seen_frame), "bbox": _round_box(bbox, bdp), "team": team,
        "team_confidence": _round_conf(confidences["team_confidence"], cdp),
        "number_candidate": number_candidate, "selected_player": bool(selected_player),
        "detection_confidence": float(det_conf),
        "motion_confidence": _round_conf(confidences["motion_confidence"], cdp),
        "appearance_confidence": _round_conf(confidences["appearance_confidence"], cdp),
        "reid_confidence": _round_conf(confidences["reid_confidence"], cdp),
        "overall_track_confidence": _round_conf(confidences["overall_track_confidence"], cdp),
        "visibility_confidence": _round_conf(confidences["visibility_confidence"], cdp),
        "occlusion_status": occlusion_status, "merged_into": None,
        "position_confidence": _round_conf(confidences["position_confidence"], cdp),
        "track_continuity_confidence": _round_conf(confidences["track_continuity_confidence"], cdp),
        "identity_confidence": idc,
        "selected_player_confidence": idc if selected_player else 0.0,
        "observed_position_box": _round_box(observed_position_box, bdp),
        "position_is_confirmed": position_source == "CONFIRMED_DETECTION",
        "position_source": position_source,
        "frames_since_observed_position": int(frames_since_observed_position),
        "frames_since_confirmed_identity": int(frames_since_confirmed_identity),
    }
    assert tuple(rec.keys()) == TRACK_FIELD_NAMES
    if v2 is not None:
        rec[V2_KEY] = _round_v2(dict(v2), cfg)
    return rec


def _round_v2(v2: Dict[str, Any], cfg: TrackerConfig) -> Dict[str, Any]:
    bdp = cfg.get("box_round_dp")
    cdp = cfg.get("confidence_round_dp")
    tdp = cfg.get("timestamp_round_dp")
    out = {}
    for k, v in v2.items():
        if k in ("observed_or_contested_box",) and v is not None:
            v = _round_box(v, bdp)
        elif k in ("identity_confidence_rb", "raw_detection_conf", "target_similarity",
                   "best_competitor_similarity") and v is not None:
            v = round(float(v), cdp)
        elif k in ("held_age_s", "time_since_confirmed_s", "timestamp_s", "uncertain_duration_s") and v is not None:
            v = round(float(v), tdp)
        out[k] = v
    return out


def build_frame_record(cfg: TrackerConfig, *, frame: int, is_cut: bool, camera_shift_px: Sequence[float],
                       n_detections: int, overlap_contests: Iterable[Sequence[str]],
                       tracks: List[Dict[str, Any]], v2: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    sdp = cfg.get("camera_shift_round_dp")
    pairs = [[str(a), str(b)] for a, b in overlap_contests]
    rec: Dict[str, Any] = {
        "frame": int(frame), "is_cut": bool(is_cut),
        "camera_shift": [round(float(camera_shift_px[0]), sdp), round(float(camera_shift_px[1]), sdp)],
        "n_detections": int(n_detections), "n_live_tracks": len(tracks),
        "overlap_contests": pairs, "tracks": tracks,
    }
    if v2 is not None:
        fv = dict(v2)
        if fv.get("timestamp_s") is not None:
            fv["timestamp_s"] = round(float(fv["timestamp_s"]), cfg.get("timestamp_round_dp"))
        if fv.get("camera_shift_h") is not None:
            fv["camera_shift_h"] = [round(float(e), cfg.get("confidence_round_dp")) for e in fv["camera_shift_h"]]
        if fv.get("target_height_px") is not None:
            fv["target_height_px"] = round(float(fv["target_height_px"]), cfg.get("box_round_dp"))
        rec[V2_KEY] = fv
    return rec


def selected_state_pct(historical_states: Sequence[str], dp: int) -> Dict[str, float]:
    """summary.selected_state_pct: percentage of frames per historical state of the selected record."""
    n = len(historical_states)
    if n == 0:
        return {}
    out: Dict[str, float] = {}
    for s in historical_states:
        out[s] = out.get(s, 0) + 1
    return {k: round(100.0 * v / n, dp) for k, v in out.items()}

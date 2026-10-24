#!/usr/bin/env python3
"""
annotation_server.py -- local Flask web app for producing canonical human
annotations (schema/annotation.schema.json) over extracted clips
(dataset/clips/<clip_id>/). This is the ONLY tool in this project that
writes human_label data, and it is the tool responsible for producing the
tracking_init.json sidecar that feature_exporter.py's "TRACKING-INIT
CONTRACT" (see that file's module docstring) requires before Pass-1
tracking can run.

ARCHITECTURE (deliberately boring, per the task's "do not over-engineer
this"): plain Flask, server-rendered Jinja for the two tiny pieces of
per-page dynamic data (which view, which clip_id), and a single vanilla-JS
file (static/annotator/app.js) that does the rest via small JSON APIs under
/api/. No frontend build step, no React/Vue, no database -- the filesystem
under dataset/ IS the database, exactly as directed. All three "pages"
(clip list, annotate-one-clip, resolve-a-disagreement) are the same
index.html template; app.js switches behaviour on `window.ANNOTATOR_VIEW`.

DESIGN DECISIONS WORTH FLAGGING FOR FUTURE READERS
---------------------------------------------------

1. CLIP VIDEO FILENAME ASSUMPTION. clip_extractor.py does not exist yet at
   the time this file was written. Per the governing task spec, this server
   assumes the extracted clip video is named literally `clip.mp4` inside
   `dataset/clips/<clip_id>/` (see `_find_clip_video()` below), falling back
   to "whatever single .mp4 is in that directory" if `clip.mp4` isn't found.
   **Double-check this assumption against clip_extractor.py's actual output
   naming once it exists** -- if it names things differently, only
   `_find_clip_video()` needs to change.

2. `analysis_context_window` IS NEVER FREELY EDITED BY THE ANNOTATOR. It is
   always read straight from the clip's own `clip_time_map.json`
   (`ClipTimeMap.analysis_context_window_absolute_s`) and written into the
   submitted record verbatim. This is not a UI restriction for its own
   sake: `feature_exporter.py` hard-fails (see its "Sanity check against the
   clip's OWN recorded context window" comment) if an annotation's
   `analysis_context_window` drifts from the clip's own recorded window by
   more than ~1.5 frames. Only the four `ground_truth_event_window`
   timestamps (start/interaction/release/end) are annotator-controlled, via
   the four marker buttons, because those describe the true event -- never
   how much pre/post-roll context a model gets, per DATASET_SCHEMA.md sec 2.

3. `match_id` RESOLUTION. `ClipTimeMap` carries `source_video_id` but not
   `match_id` -- `match_id` lives in the (not-yet-populated, at the time of
   writing) `dataset/metadata/source_videos.json` registry described in
   DATASET_SCHEMA.md. `_resolve_match_id()` looks that registry up by
   `source_video_id` and falls back to using `source_video_id` itself as
   `match_id` if the registry is absent or has no entry for it. This keeps
   the annotator usable before that registry exists; once
   metadata/source_videos.json is populated by another tool, real match_ids
   will simply start showing up with no code change here.

4. DOUBLE-ANNOTATION PAIRING RULE. A new submission is treated as the
   "second pass" of an existing clip only when exactly one annotation with
   status SINGLE_PASS already exists for that clip_id (see
   `_apply_double_annotation_logic()`). That pairing compares only
   `event_type` + `outcome` (per the task spec) and sets BOTH records'
   `status` (DOUBLE_PASS_AGREED or DOUBLE_PASS_DISAGREEMENT) and
   `provenance.double_annotation`; neither file's `human_label` content is
   ever touched. If a clip already has two settled passes (AGREED,
   DISAGREEMENT, or GOLD_RESOLVED) and a further annotation is submitted
   anyway, it is written as an independent SINGLE_PASS record rather than
   forced into a pairing that the schema/spec don't define -- this is a
   deliberate simplification for a case the task spec does not cover, not
   an oversight.

5. PIPELINE-SUGGESTION / `weak_label` DISPLAY: SKIPPED FOR V1, ON PURPOSE.
   Task item #5 explicitly says this is optional and "acceptable and
   simplest to skip pipeline-suggestion display entirely for this first
   version." This server never calls the frozen pipeline and never
   populates `weak_label`. What matters structurally (and IS implemented):
   `human_label` is built exclusively from the annotator's own form
   submission -- there is no code path anywhere in this file that reads a
   pipeline/weak-label value into `human_label`.

6. Every dropdown enum in the UI is served from `/api/taxonomy`, which is a
   thin JSON view over `schema/taxonomy.py`. No event type / subtype /
   outcome / confidence list is duplicated in app.js or index.html.

Run with:
    python3 annotation_server.py                # http://127.0.0.1:5050
    python3 annotation_server.py --port 8000 --debug
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_from_directory

TOOLS_DIR = Path(__file__).resolve().parent
DATASET_ROOT = TOOLS_DIR.parent
REPO_ROOT = DATASET_ROOT.parent
for _p in (TOOLS_DIR, DATASET_ROOT, REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import time_utils                          # noqa: E402  (this project's own module)
from schema import taxonomy                # noqa: E402  (single source of truth for enums)

try:
    import jsonschema
except ImportError:  # pragma: no cover -- the task guarantees this is installed
    jsonschema = None

ANNOTATION_TOOL_VERSION = "1.0.0"
ANNOTATION_SCHEMA_VERSION = "1.0.0"

CLIPS_DIR = DATASET_ROOT / "dataset" / "clips"
ANNOTATIONS_DIR = DATASET_ROOT / "dataset" / "annotations"
SOURCE_VIDEOS_REGISTRY = DATASET_ROOT / "dataset" / "metadata" / "source_videos.json"
ANNOTATION_SCHEMA_PATH = DATASET_ROOT / "schema" / "annotation.schema.json"

STATIC_DIR = TOOLS_DIR / "static" / "annotator"

# Statuses that represent a "settled first pass" a second submission for the
# same clip can be paired against. Only SINGLE_PASS is eligible -- see design
# decision #4 in the module docstring.
PAIRABLE_STATUS = "SINGLE_PASS"

app = Flask(
    __name__,
    static_folder=str(STATIC_DIR),
    static_url_path="/static",
    template_folder=str(STATIC_DIR),
)


class AnnotationServerError(Exception):
    """Raised for any client-fixable problem (bad clip_id, missing
    clip_time_map.json, taxonomy violation, schema violation, ...). Always
    caught at the route boundary and turned into a 400 JSON error -- never
    allowed to become an opaque 500."""


# ---------------------------------------------------------------------------
# Filesystem helpers -- dataset/ is the database, so these are the only
# functions in this file that touch disk under dataset/.
# ---------------------------------------------------------------------------

def _load_annotation_schema() -> dict:
    return json.loads(ANNOTATION_SCHEMA_PATH.read_text())


_ANNOTATION_SCHEMA = _load_annotation_schema()


def list_clip_ids() -> list:
    if not CLIPS_DIR.exists():
        return []
    return sorted(p.name for p in CLIPS_DIR.iterdir() if p.is_dir())


def _find_clip_video(clip_dir: Path):
    """See design decision #1 in the module docstring: assumes `clip.mp4`,
    falls back to any single .mp4 present. Returns the filename (relative to
    clip_dir), or None if no video file is found."""
    preferred = clip_dir / "clip.mp4"
    if preferred.exists():
        return "clip.mp4"
    candidates = sorted(clip_dir.glob("*.mp4"))
    return candidates[0].name if candidates else None


def load_time_map_safe(clip_id: str):
    """Returns (ClipTimeMap or None, error_message or None)."""
    clip_dir = CLIPS_DIR / clip_id
    try:
        return time_utils.load_clip_time_map(str(clip_dir)), None
    except time_utils.TimeUtilsError as e:
        return None, str(e)


def load_tracking_init_safe(clip_id: str):
    p = CLIPS_DIR / clip_id / "tracking_init.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _all_annotation_paths() -> list:
    if not ANNOTATIONS_DIR.exists():
        return []
    return sorted(ANNOTATIONS_DIR.glob("*.json"))


def load_all_annotations() -> list:
    out = []
    for p in _all_annotation_paths():
        try:
            out.append(json.loads(p.read_text()))
        except (json.JSONDecodeError, OSError):
            continue  # a corrupt file must not take the whole UI down
    return out


def annotations_for_clip(clip_id: str, all_annotations=None) -> list:
    all_annotations = all_annotations if all_annotations is not None else load_all_annotations()
    return [a for a in all_annotations if a.get("clip_id") == clip_id]


def get_annotation_by_id(annotation_id: str):
    p = ANNOTATIONS_DIR / f"{annotation_id}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def write_annotation_file(record: dict) -> Path:
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ANNOTATIONS_DIR / f"{record['annotation_id']}.json"
    out_path.write_text(json.dumps(record, indent=2))
    return out_path


def compute_clip_status(clip_id: str, all_annotations=None) -> str:
    """One of: NONE, SINGLE_PASS, DOUBLE_PASS_AGREED,
    DOUBLE_PASS_DISAGREEMENT, GOLD_RESOLVED (task item #1). GOLD_RESOLVED
    takes priority for display if present -- it is the final answer for that
    clip even though the two disagreeing records also still exist on disk."""
    anns = annotations_for_clip(clip_id, all_annotations)
    if not anns:
        return "NONE"
    statuses = {a.get("status") for a in anns}
    if "GOLD_RESOLVED" in statuses:
        return "GOLD_RESOLVED"
    if "DOUBLE_PASS_DISAGREEMENT" in statuses:
        return "DOUBLE_PASS_DISAGREEMENT"
    if "DOUBLE_PASS_AGREED" in statuses:
        return "DOUBLE_PASS_AGREED"
    return "SINGLE_PASS"


def _resolve_match_id(source_video_id: str) -> str:
    """See design decision #3 in the module docstring."""
    if SOURCE_VIDEOS_REGISTRY.exists():
        try:
            registry = json.loads(SOURCE_VIDEOS_REGISTRY.read_text())
            entry = registry.get(source_video_id)
            if isinstance(entry, dict) and entry.get("match_id"):
                return entry["match_id"]
        except (json.JSONDecodeError, OSError):
            pass
    return source_video_id  # documented fallback -- see design decision #3


# ---------------------------------------------------------------------------
# Record building -- the one place human_label + frame_safety get assembled,
# shared by the normal submit path and the gold-resolution path so both
# produce structurally identical, schema-valid records.
# ---------------------------------------------------------------------------

def _require(payload: dict, key: str):
    if key not in payload or payload[key] in (None, ""):
        raise AnnotationServerError(f"missing required field: {key}")
    return payload[key]


def _nullable(payload: dict, key: str):
    return payload.get(key) if payload.get(key) not in ("",) else None


def build_human_label(payload: dict, ctm: "time_utils.ClipTimeMap") -> tuple:
    """Builds the `human_label` object plus its matching `frame_safety`
    object from a JSON payload whose event-window timestamps are
    CLIP-RELATIVE seconds (what the <video> element's currentTime gives the
    browser) and converts every one of them to absolute source-video seconds
    via the clip's own ClipTimeMap, per DATASET_SCHEMA.md sec 2 / task item
    #7. Returns (human_label_dict, frame_safety_dict).
    """
    event_type = _require(payload, "event_type")
    event_subtype = payload.get("event_subtype") or None
    outcome = payload.get("outcome") or None

    problems = taxonomy.validate_event_record(event_type, event_subtype, outcome)
    if problems:
        raise AnnotationServerError("; ".join(problems))

    abstention_reason = payload.get("abstention_reason") or None
    if abstention_reason is not None:
        if event_type not in ("OTHER", "UNKNOWN"):
            raise AnnotationServerError("abstention_reason is only valid when event_type is OTHER or UNKNOWN")
        if abstention_reason not in taxonomy.ABSTENTION_REASONS:
            raise AnnotationServerError(f"abstention_reason '{abstention_reason}' not in {taxonomy.ABSTENTION_REASONS}")

    hard_negative_category = payload.get("hard_negative_category") or None
    if hard_negative_category is not None:
        if event_type != "NO_EVENT":
            raise AnnotationServerError("hard_negative_category is only meaningful when event_type is NO_EVENT")
        if hard_negative_category not in taxonomy.HARD_NEGATIVE_CATEGORIES:
            raise AnnotationServerError(
                f"hard_negative_category '{hard_negative_category}' not in {taxonomy.HARD_NEGATIVE_CATEGORIES}"
            )

    selected_player_in = payload.get("selected_player") or {}
    team = _require(selected_player_in, "team")
    if team not in taxonomy.TEAMS:
        raise AnnotationServerError(f"selected_player.team '{team}' not in {taxonomy.TEAMS}")
    manual_identity_confidence = _require(selected_player_in, "manual_identity_confidence")
    if manual_identity_confidence not in taxonomy.MANUAL_IDENTITY_CONFIDENCE_LEVELS:
        raise AnnotationServerError(
            f"selected_player.manual_identity_confidence '{manual_identity_confidence}' "
            f"not in {taxonomy.MANUAL_IDENTITY_CONFIDENCE_LEVELS}"
        )
    selected_player = {
        "player_id": _nullable(selected_player_in, "player_id"),
        "jersey_number": selected_player_in.get("jersey_number") if selected_player_in.get("jersey_number") not in ("", None) else None,
        "team": team,
        "manual_identity_confidence": manual_identity_confidence,
    }

    annotation_confidence = _require(payload, "annotation_confidence")
    if annotation_confidence not in taxonomy.ANNOTATION_CONFIDENCE_LEVELS:
        raise AnnotationServerError(f"annotation_confidence '{annotation_confidence}' not in {taxonomy.ANNOTATION_CONFIDENCE_LEVELS}")

    window_in = payload.get("ground_truth_event_window_clip_relative_s") or {}
    start_cr = _require(window_in, "start_time_s")
    end_cr = _require(window_in, "end_time_s")
    interaction_cr = window_in.get("interaction_time_s")
    release_cr = window_in.get("release_time_s")
    try:
        start_cr, end_cr = float(start_cr), float(end_cr)
        interaction_cr = float(interaction_cr) if interaction_cr is not None else None
        release_cr = float(release_cr) if release_cr is not None else None
    except (TypeError, ValueError) as e:
        raise AnnotationServerError(f"ground_truth_event_window has a non-numeric timestamp: {e}")

    def to_abs(cr):
        return None if cr is None else ctm.clip_relative_seconds_to_absolute_seconds(cr)

    start_abs, end_abs = to_abs(start_cr), to_abs(end_cr)
    interaction_abs, release_abs = to_abs(interaction_cr), to_abs(release_cr)

    ground_truth_event_window = {
        "start_time_s": start_abs,
        "interaction_time_s": interaction_abs,
        "release_time_s": release_abs,
        "end_time_s": end_abs,
    }

    # analysis_context_window: NEVER taken from the client -- see design
    # decision #2 in the module docstring.
    ctx_start_abs, ctx_end_abs = ctm.analysis_context_window_absolute_s
    analysis_context_window = {"start_time_s": ctx_start_abs, "end_time_s": ctx_end_abs}

    human_label = {
        "selected_player": selected_player,
        "event_type": event_type,
        "event_subtype": event_subtype,
        "outcome": outcome,
        "abstention_reason": abstention_reason,
        "hard_negative_category": hard_negative_category,
        "ground_truth_event_window": ground_truth_event_window,
        "analysis_context_window": analysis_context_window,
        "receiver_player_id": _nullable(payload, "receiver_player_id"),
        "opponent_player_id": _nullable(payload, "opponent_player_id"),
        "annotation_confidence": annotation_confidence,
        "ambiguous": bool(payload.get("ambiguous", False)),
        "review_required": bool(payload.get("review_required", False)),
        "notes": payload.get("notes") or "",
    }

    def to_frame(abs_s):
        return None if abs_s is None else time_utils.seconds_to_frames(abs_s, ctm.source_fps)

    frame_safety = {
        "source_fps": ctm.source_fps,
        "absolute_start_frame": ctm.absolute_start_frame,
        "absolute_end_frame": ctm.absolute_end_frame,
        "ground_truth_event_window_absolute_frames": {
            "start": to_frame(start_abs),
            "interaction": to_frame(interaction_abs),
            "release": to_frame(release_abs),
            "end": to_frame(end_abs),
        },
    }

    return human_label, frame_safety


def build_annotation_record(clip_id: str, payload: dict, status: str, provenance_extra: dict = None) -> dict:
    ctm, err = load_time_map_safe(clip_id)
    if ctm is None:
        raise AnnotationServerError(f"cannot build annotation for clip '{clip_id}': {err}")

    human_label, frame_safety = build_human_label(payload, ctm)

    annotator_id = _require(payload, "annotator_id")
    provenance = {
        "annotator_id": annotator_id,
        "annotated_at": datetime.now(timezone.utc).isoformat(),
        "annotation_tool_version": ANNOTATION_TOOL_VERSION,
        "double_annotation": None,
        "gold_label": None,
        "migrated_from": None,
    }
    if provenance_extra:
        provenance.update(provenance_extra)

    record = {
        "schema_version": ANNOTATION_SCHEMA_VERSION,
        "annotation_id": str(uuid.uuid4()),
        "clip_id": clip_id,
        "match_id": _resolve_match_id(ctm.source_video_id),
        "source_video_id": ctm.source_video_id,
        "status": status,
        "human_label": human_label,
        "provenance": provenance,
        "frame_safety": frame_safety,
    }
    _validate_record(record)
    return record


def _validate_record(record: dict):
    if jsonschema is None:
        return  # task guarantees jsonschema is installed; degrade gracefully rather than crash
    try:
        jsonschema.validate(record, _ANNOTATION_SCHEMA)
    except jsonschema.ValidationError as e:
        raise AnnotationServerError(f"record failed schema validation: {e.message}") from e


def _apply_double_annotation_logic(clip_id: str, new_record: dict):
    """See design decision #4 in the module docstring. Mutates new_record's
    status/provenance in place; if a pairing happens, also rewrites the
    paired first-pass record's file on disk (status + provenance only --
    its human_label is never touched). Returns the (possibly None) first
    record dict for the caller's response payload."""
    existing = [a for a in annotations_for_clip(clip_id) if a.get("status") == PAIRABLE_STATUS]

    if len(existing) == 0:
        new_record["status"] = "SINGLE_PASS"
        return None

    # If more than one SINGLE_PASS record somehow exists (shouldn't happen,
    # since this function is the only thing that promotes SINGLE_PASS out of
    # that state), pair with the oldest one and leave the rest alone rather
    # than guessing at a three-way comparison the schema doesn't model.
    first = sorted(existing, key=lambda a: a.get("provenance", {}).get("annotated_at", ""))[0]

    first_label = first["human_label"]
    new_label = new_record["human_label"]
    agree = (first_label["event_type"] == new_label["event_type"] and first_label["outcome"] == new_label["outcome"])
    new_status = "DOUBLE_PASS_AGREED" if agree else "DOUBLE_PASS_DISAGREEMENT"

    new_record["status"] = new_status
    new_record["provenance"]["double_annotation"] = {
        "is_second_pass": True,
        "paired_annotation_id": first["annotation_id"],
    }

    first["status"] = new_status
    first.setdefault("provenance", {})["double_annotation"] = {
        "is_second_pass": False,
        "paired_annotation_id": new_record["annotation_id"],
    }
    _validate_record(first)
    write_annotation_file(first)
    return first


# ---------------------------------------------------------------------------
# Page routes -- all three render the same template; app.js branches on
# window.ANNOTATOR_VIEW / window.ANNOTATOR_CLIP_ID.
# ---------------------------------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html", view="list", clip_id=None)


@app.route("/annotate/<clip_id>")
def annotate_view(clip_id):
    return render_template("index.html", view="annotate", clip_id=clip_id)


@app.route("/resolve/<clip_id>")
def resolve_view(clip_id):
    return render_template("index.html", view="resolve", clip_id=clip_id)


# ---------------------------------------------------------------------------
# Video / sidecar file serving -- read-only, scoped to one clip's own
# directory (send_from_directory rejects path traversal on its own).
# ---------------------------------------------------------------------------

@app.route("/clips/<clip_id>/<path:filename>")
def serve_clip_file(clip_id, filename):
    clip_dir = CLIPS_DIR / clip_id
    if not clip_dir.is_dir():
        abort(404)
    return send_from_directory(str(clip_dir), filename)


# ---------------------------------------------------------------------------
# JSON APIs
# ---------------------------------------------------------------------------

@app.route("/api/taxonomy")
def api_taxonomy():
    return jsonify({
        "taxonomy_version": taxonomy.TAXONOMY_VERSION,
        "event_types": taxonomy.EVENT_TYPES,
        "event_types_all": taxonomy.EVENT_TYPES_ALL,
        "abstention_reasons": taxonomy.ABSTENTION_REASONS,
        "outcomes_by_event_type": taxonomy.OUTCOMES_BY_EVENT_TYPE,
        "subtypes_by_event_type": taxonomy.SUBTYPES_BY_EVENT_TYPE,
        "hard_negative_categories": taxonomy.HARD_NEGATIVE_CATEGORIES,
        "annotation_confidence_levels": taxonomy.ANNOTATION_CONFIDENCE_LEVELS,
        "manual_identity_confidence_levels": taxonomy.MANUAL_IDENTITY_CONFIDENCE_LEVELS,
        "teams": taxonomy.TEAMS,
    })


@app.route("/api/clips")
def api_clips():
    all_annotations = load_all_annotations()
    clips = []
    for clip_id in list_clip_ids():
        anns = annotations_for_clip(clip_id, all_annotations)
        clips.append({
            "clip_id": clip_id,
            "status": compute_clip_status(clip_id, all_annotations),
            "annotation_count": len(anns),
            "has_video": _find_clip_video(CLIPS_DIR / clip_id) is not None,
            "has_time_map": (CLIPS_DIR / clip_id / "clip_time_map.json").exists(),
        })
    return jsonify({"clips": clips})


@app.route("/api/clip/<clip_id>")
def api_clip_detail(clip_id):
    clip_dir = CLIPS_DIR / clip_id
    if not clip_dir.is_dir():
        return jsonify({"error": f"no such clip directory: {clip_id}"}), 404

    ctm, time_map_error = load_time_map_safe(clip_id)
    video_filename = _find_clip_video(clip_dir)
    all_annotations = load_all_annotations()
    anns = annotations_for_clip(clip_id, all_annotations)

    return jsonify({
        "clip_id": clip_id,
        "status": compute_clip_status(clip_id, all_annotations),
        "video_filename": video_filename,
        "video_url": f"/clips/{clip_id}/{video_filename}" if video_filename else None,
        "time_map": ctm.to_dict() if ctm else None,
        "time_map_error": time_map_error,
        "tracking_init": load_tracking_init_safe(clip_id),
        "annotations": [
            {
                "annotation_id": a["annotation_id"],
                "status": a["status"],
                "event_type": a["human_label"]["event_type"],
                "event_subtype": a["human_label"]["event_subtype"],
                "outcome": a["human_label"]["outcome"],
                "annotator_id": a["provenance"]["annotator_id"],
                "annotated_at": a["provenance"]["annotated_at"],
                "human_label": a["human_label"],
            }
            for a in anns
        ],
    })


@app.route("/api/clip/<clip_id>/tracking_init", methods=["POST"])
def api_save_tracking_init(clip_id):
    """Writes dataset/clips/<clip_id>/tracking_init.json per
    feature_exporter.py's TRACKING-INIT CONTRACT. This is the only place in
    the whole spana_dataset/ tree that produces this file."""
    clip_dir = CLIPS_DIR / clip_id
    if not clip_dir.is_dir():
        return jsonify({"error": f"no such clip directory: {clip_id}"}), 404

    payload = request.get_json(force=True, silent=True) or {}
    try:
        init_frame = int(_require(payload, "init_frame"))
        init_box = payload.get("init_box")
        if not (isinstance(init_box, list) and len(init_box) == 4 and all(isinstance(v, (int, float)) for v in init_box)):
            raise AnnotationServerError("init_box must be [x1, y1, x2, y2]")
        player_id = payload.get("player_id") or None
        jersey_number = payload.get("jersey_number")
        jersey_number = int(jersey_number) if jersey_number not in (None, "") else None
        max_frames = payload.get("max_frames")
        max_frames = int(max_frames) if max_frames not in (None, "") else None
    except AnnotationServerError as e:
        return jsonify({"error": str(e)}), 400
    except (TypeError, ValueError) as e:
        return jsonify({"error": f"invalid tracking_init payload: {e}"}), 400

    record = {
        "player_id": player_id,
        "jersey_number": jersey_number,
        "init_frame": init_frame,
        "init_box": [float(v) for v in init_box],
        "max_frames": max_frames,
    }
    out_path = clip_dir / "tracking_init.json"
    out_path.write_text(json.dumps(record, indent=2))
    return jsonify({"ok": True, "tracking_init": record, "path": str(out_path)})


@app.route("/api/clip/<clip_id>/submit", methods=["POST"])
def api_submit_annotation(clip_id):
    clip_dir = CLIPS_DIR / clip_id
    if not clip_dir.is_dir():
        return jsonify({"error": f"no such clip directory: {clip_id}"}), 404

    payload = request.get_json(force=True, silent=True) or {}
    try:
        record = build_annotation_record(clip_id, payload, status="SINGLE_PASS")
        paired_first = _apply_double_annotation_logic(clip_id, record)
        # status may have just changed (SINGLE_PASS -> DOUBLE_PASS_*); re-validate
        # before writing since we're about to persist it.
        _validate_record(record)
        write_annotation_file(record)
    except AnnotationServerError as e:
        return jsonify({"error": str(e)}), 400

    return jsonify({
        "ok": True,
        "annotation_id": record["annotation_id"],
        "status": record["status"],
        "paired_with": paired_first["annotation_id"] if paired_first else None,
        "clip_status": compute_clip_status(clip_id),
    })


@app.route("/api/clip/<clip_id>/gold_resolve", methods=["POST"])
def api_gold_resolve(clip_id):
    """Implements task item #23's disagreement resolution: reads the two
    original DOUBLE_PASS_DISAGREEMENT records (never mutating them) and
    writes a brand-new, third GOLD_RESOLVED record whose
    provenance.gold_label.source_annotation_ids points back at them."""
    clip_dir = CLIPS_DIR / clip_id
    if not clip_dir.is_dir():
        return jsonify({"error": f"no such clip directory: {clip_id}"}), 404

    payload = request.get_json(force=True, silent=True) or {}
    try:
        source_ids = payload.get("source_annotation_ids")
        if not (isinstance(source_ids, list) and len(source_ids) >= 2):
            raise AnnotationServerError("source_annotation_ids must be an array of at least 2 annotation_ids")
        sources = []
        for sid in source_ids:
            src = get_annotation_by_id(sid)
            if src is None:
                raise AnnotationServerError(f"source annotation not found: {sid}")
            if src.get("clip_id") != clip_id:
                raise AnnotationServerError(f"source annotation {sid} belongs to a different clip")
            if src.get("status") != "DOUBLE_PASS_DISAGREEMENT":
                raise AnnotationServerError(
                    f"source annotation {sid} has status={src.get('status')!r}, expected DOUBLE_PASS_DISAGREEMENT"
                )
            sources.append(src)

        resolved_by = _require(payload, "resolved_by")
        resolution_notes = payload.get("resolution_notes") or ""

        gold_provenance_extra = {
            "gold_label": {
                "resolved_by": resolved_by,
                "resolved_at": datetime.now(timezone.utc).isoformat(),
                "source_annotation_ids": [s["annotation_id"] for s in sources],
                "resolution_notes": resolution_notes,
            }
        }
        # The gold record's own top-level provenance.annotator_id is the
        # reviewer who resolved it -- payload doesn't need a separate
        # annotator_id field, so default it from resolved_by if not given.
        payload.setdefault("annotator_id", resolved_by)

        record = build_annotation_record(clip_id, payload, status="GOLD_RESOLVED", provenance_extra=gold_provenance_extra)
        write_annotation_file(record)
    except AnnotationServerError as e:
        return jsonify({"error": str(e)}), 400

    return jsonify({"ok": True, "annotation_id": record["annotation_id"], "status": record["status"]})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5050)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
feature_exporter.py -- CLI tool that turns one canonical annotation
(schema/annotation.schema.json) into one canonical feature record
(schema/feature.schema.json), by running the FROZEN production pipeline
read-only (via feature_extraction_harness.py) over the annotation's clip and
reading out its already-computed per-frame buffers and release-arbitration
evidence.

THIS FILE DOES NOT TRAIN ANYTHING (audit item #48). It does not fit, tune,
or select a model of any kind. Its entire job is: run the existing,
unmodified pipeline once per clip, resample its output into fixed temporal
bins, mask missing values explicitly, and write the result to disk in a
versioned, reproducible format.

TRACKING-INIT CONTRACT (documented here because no tool writes it yet --
annotation_server.py, not yet built, is expected to produce it as part of
the annotator drawing/selecting the tracked player's initial box):
each clip directory (dataset/clips/<clip_id>/) must contain, alongside
clip_time_map.json, a `tracking_init.json` sidecar:
    {
      "player_id": "P1",
      "jersey_number": 9,               # optional, may be null
      "init_frame": 0,                  # CLIP-RELATIVE frame the box was drawn on
      "init_box": [x1, y1, x2, y2],      # pixel coords in the CLIP's own frame
      "max_frames": 150                 # optional; defaults to the whole clip
    }
Without this file, feature export cannot run Pass 1 tracking and fails with
a clear, actionable error rather than guessing an initial box.

Usage:
    python feature_exporter.py export --annotation dataset/annotations/<id>.json [--verify]
    python feature_exporter.py export-batch --annotations-dir dataset/annotations [--verify]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
DATASET_ROOT = TOOLS_DIR.parent
REPO_ROOT = DATASET_ROOT.parent
sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(DATASET_ROOT))
sys.path.insert(0, str(REPO_ROOT))

import time_utils                                    # noqa: E402  (this project's own module)
import event_matching                                 # noqa: E402
import freeze_utils                                    # noqa: E402
from schema import taxonomy                            # noqa: E402
import clip_pipeline_runner as runner                  # noqa: E402

FEATURE_SCHEMA_VERSION = "1.0.0"
FEATURE_EXPORTER_VERSION = "1.0.0"
BIN_WIDTH_S = 0.1     # documented in DATASET_SCHEMA.md sec 3; bump feature_schema_version to change this

ANNOTATIONS_DIR = DATASET_ROOT / "dataset" / "annotations"
CLIPS_DIR = DATASET_ROOT / "dataset" / "clips"
FEATURES_DIR = DATASET_ROOT / "dataset" / "features"

# Only these statuses represent a settled human label -- an unresolved
# disagreement must never be exported as if it were a usable ground truth
# (mirrors audit item #37's "human label is never auto-filled" spirit: an
# UNRESOLVED disagreement is not yet a human label at all).
EXPORTABLE_STATUSES = {"SINGLE_PASS", "DOUBLE_PASS_AGREED", "GOLD_RESOLVED"}


class FeatureExportError(Exception):
    pass


def masked(value, valid: bool) -> dict:
    return {"value": (value if valid else None), "valid": bool(valid)}


def _extract_evidence_number(evidence: dict, keys) -> tuple:
    """Reads the first present numeric value under `keys` out of an
    already-computed evidence dict. Returns (value, True) if found,
    (None, False) otherwise. Never computes a new number itself -- this is
    read-only extraction of an existing production value, not a new
    heuristic (audit item #45: no new production logic)."""
    if not isinstance(evidence, dict):
        return None, False
    for k in keys:
        v = evidence.get(k)
        if isinstance(v, (int, float)):
            return float(v), True
    return None, False


def _build_heuristic_scores(matched_event: dict | None) -> dict:
    empty_candidate_keys = ["PASS", "SHOT", "CROSS", "DUEL", "UNKNOWN_RELEASE"]
    out = {k: None for k in empty_candidate_keys}
    out["pipeline_winner"] = None
    out["pipeline_decision"] = None
    if matched_event is None:
        return out

    arbitration = (matched_event.get("evidence") or {}).get("arbitration")
    if not arbitration:
        return out

    candidates = arbitration.get("candidates")
    if candidates:
        # candidates is {event_type: Candidate.to_dict()} -- verbatim from event_arbitration.py
        for etype, cdict in candidates.items():
            if etype in out:
                out[etype] = {
                    "score": cdict.get("score"),
                    "confidence": cdict.get("confidence"),
                    "effective_score": cdict.get("effective_score"),
                    "evidence": cdict.get("evidence", {}),
                }
        ranked = arbitration.get("ranked") or []
        out["pipeline_winner"] = ranked[0] if ranked else None
        out["pipeline_decision"] = arbitration.get("decision")
    else:
        # Authoritative short-circuit path (resolve_release's confirmed-
        # teammate-reception branch) never builds a candidates dict --
        # report what IS known (the winning type and the stated reason)
        # rather than fabricating five candidate scores that were never
        # computed for this release.
        out["pipeline_winner"] = matched_event.get("type")
        out["pipeline_decision"] = arbitration.get("reason") or arbitration.get("decision")
    return out


def export_one(annotation_path: Path, out_root: Path = FEATURES_DIR, bin_width_s: float = BIN_WIDTH_S,
                verify: bool = False, force: bool = False) -> Path:
    ann = json.loads(Path(annotation_path).read_text())

    if ann.get("schema_version") != "1.0.0":
        raise FeatureExportError(f"unsupported annotation schema_version: {ann.get('schema_version')!r}")
    if ann["status"] not in EXPORTABLE_STATUSES:
        raise FeatureExportError(
            f"annotation {ann['annotation_id']} has status={ann['status']!r} -- only "
            f"{sorted(EXPORTABLE_STATUSES)} may be exported as features. An unresolved "
            f"DOUBLE_PASS_DISAGREEMENT must go through gold resolution first."
        )

    human = ann["human_label"]
    problems = taxonomy.validate_event_record(human["event_type"], human["event_subtype"], human["outcome"])
    if problems:
        raise FeatureExportError(f"annotation {ann['annotation_id']} fails taxonomy validation: {problems}")

    out_path = out_root / FEATURE_SCHEMA_VERSION / f"{ann['annotation_id']}.json"
    if out_path.exists() and not force:
        print(f"  skip (already exported): {out_path}")
        return out_path

    clip_id = ann["clip_id"]
    try:
        loaded = runner.load_annotation_and_clip(annotation_path)
        capture = runner.run_pipeline(loaded, verify=verify)
    except runner.ClipPipelineError as e:
        raise FeatureExportError(str(e)) from e
    ctm = loaded.clip_time_map
    clip_meta = loaded.clip_meta

    snapshots_by_frame = {s.frame_number: s for s in capture.snapshots}
    ball_by_frame = {b.frame_number: b for b in capture.ball_history}
    poss_by_frame = {p.frame: p for p in capture.possession_frames}

    gt = human["ground_truth_event_window"]
    ctx = human["analysis_context_window"]

    align_absolute_s = gt["interaction_time_s"] if gt["interaction_time_s"] is not None else gt["start_time_s"]
    align_clip_rel_s = ctm.absolute_seconds_to_clip_relative_seconds(align_absolute_s)

    ctx_start_clip_rel = ctm.absolute_seconds_to_clip_relative_seconds(ctx["start_time_s"])
    ctx_end_clip_rel = ctm.absolute_seconds_to_clip_relative_seconds(ctx["end_time_s"])
    # Sanity check against the clip's OWN recorded context window -- these should always
    # agree (both derive from the same clip_time_map.json); a mismatch beyond one frame's
    # worth of rounding means the annotation and the clip sidecar have drifted apart.
    ctm_ctx_start, ctm_ctx_end = ctm.analysis_context_window_clip_relative_s()
    tol_s = 1.5 / clip_meta.fps
    if abs(ctx_start_clip_rel - ctm_ctx_start) > tol_s or abs(ctx_end_clip_rel - ctm_ctx_end) > tol_s:
        raise FeatureExportError(
            f"annotation {ann['annotation_id']}'s analysis_context_window does not match "
            f"clip {clip_id}'s own clip_time_map.json -- refusing to export with an "
            f"inconsistent frame/time basis."
        )

    matched_event = runner.find_matched_event(loaded, capture)

    release_clip_rel_s = None
    if gt["release_time_s"] is not None:
        release_clip_rel_s = ctm.absolute_seconds_to_clip_relative_seconds(gt["release_time_s"])

    n_bins = max(1, math.ceil((ctx_end_clip_rel - ctx_start_clip_rel) / bin_width_s))
    last_valid_frame = max(snapshots_by_frame) if snapshots_by_frame else -1

    release_bin_index = None
    if release_clip_rel_s is not None:
        release_bin_index = int((release_clip_rel_s - ctx_start_clip_rel) // bin_width_s)

    temporal_bins = []
    for i in range(n_bins):
        bin_start = ctx_start_clip_rel + i * bin_width_s
        bin_centre = bin_start + bin_width_s / 2.0
        t_offset_s = round(bin_centre - align_clip_rel_s, 4)

        nearest_frame = time_utils.seconds_to_frames(bin_centre, clip_meta.fps)
        frame_in_range = 0 <= nearest_frame <= last_valid_frame
        resolved_frame = nearest_frame if frame_in_range else None

        source_frame_absolute = (ctm.clip_relative_seconds_to_absolute_frame(bin_centre)
                                  if frame_in_range else None)

        snap = snapshots_by_frame.get(resolved_frame) if resolved_frame is not None else None
        snap_prev = snapshots_by_frame.get(resolved_frame - 1) if resolved_frame is not None else None
        snap_next = snapshots_by_frame.get(resolved_frame + 1) if resolved_frame is not None else None
        player_valid = snap is not None and snap.tracking_state != "LOST"

        vel_valid = (player_valid and snap_prev is not None and snap_next is not None
                     and snap_prev.tracking_state != "LOST" and snap_next.tracking_state != "LOST")
        vx = ((snap_next.centre_x - snap_prev.centre_x) / (2.0 / clip_meta.fps)) if vel_valid else None
        vy = ((snap_next.centre_y - snap_prev.centre_y) / (2.0 / clip_meta.fps)) if vel_valid else None

        bts = ball_by_frame.get(resolved_frame) if resolved_frame is not None else None
        ball_valid = bts is not None and bts.centre_x is not None and bts.centre_y is not None

        pf = poss_by_frame.get(resolved_frame) if resolved_frame is not None else None

        raw = {
            "player_x": masked(snap.centre_x if player_valid else None, player_valid),
            "player_y": masked(snap.centre_y if player_valid else None, player_valid),
            "player_velocity_x": masked(vx, vel_valid),
            "player_velocity_y": masked(vy, vel_valid),
            "player_tracking_state": masked(snap.tracking_state if snap else None, snap is not None),
            "player_tracking_confidence": masked(snap.tracking_confidence if snap else None, snap is not None),
            "ball_x": masked(bts.centre_x if ball_valid else None, ball_valid),
            "ball_y": masked(bts.centre_y if ball_valid else None, ball_valid),
            "ball_velocity_x": masked(bts.velocity_x if ball_valid else None, ball_valid),
            "ball_velocity_y": masked(bts.velocity_y if ball_valid else None, ball_valid),
            "ball_state": masked(bts.state if bts else None, bts is not None),
            "ball_confidence": masked(bts.confidence if bts else None, bts is not None),
        }

        pbd_valid = player_valid and ball_valid
        pbd = (math.hypot(snap.centre_x - bts.centre_x, snap.centre_y - bts.centre_y)
               if pbd_valid else None)

        teammate_d = pf.nearest_teammate_distance if pf else None
        if teammate_d is None and pf is not None and pf.nearest_other_team == "TEAMMATE":
            teammate_d = pf.nearest_other_distance
        unknown_d = pf.nearest_unknown_distance if pf else None
        if unknown_d is None and pf is not None and pf.nearest_other_team == "UNKNOWN":
            unknown_d = pf.nearest_other_distance
        opponent_d = pf.nearest_other_distance if (pf is not None and pf.nearest_other_team == "OPPONENT") else None

        possession_valid = pf is not None and not pf.paused
        possession_val = (1.0 if pf.in_possession else 0.0) if possession_valid else None

        derived = {
            "player_ball_distance": masked(pbd, pbd_valid),
            "nearest_teammate_distance": masked(teammate_d, teammate_d is not None),
            "nearest_opponent_distance": masked(opponent_d, opponent_d is not None),
            "nearest_unknown_distance": masked(unknown_d, unknown_d is not None),
            "possession_probability": masked(possession_val, possession_valid),
            # Not yet computable without a real pixel-to-pitch homography -- PitchContext
            # only exposes an attack-direction proxy today, not per-frame pitch coordinates.
            # Left explicitly invalid rather than faked (audit item #30); revisit once a real
            # calibration exists.
            "pitch_position_x": masked(None, False),
            "pitch_position_y": masked(None, False),
            "receiver_likelihood": masked(None, False),
            "goal_direction_score": masked(None, False),
            "release_width_estimate": masked(None, False),
        }

        # receiver_likelihood / goal_direction_score / release_width_estimate: populated
        # ONLY on the bin nearest release_time, and only by reading values the frozen
        # pipeline's own arbitration evidence already computed for this release -- never
        # recomputed here (audit items #26, #28, #45).
        if matched_event is not None and release_bin_index is not None and i == release_bin_index:
            evidence = matched_event.get("evidence") or {}
            receiver_evidence = evidence.get("receiver_evidence")
            if isinstance(receiver_evidence, dict):
                status = receiver_evidence.get("status")
                if status == "RECEIVER_CONFIRMED":
                    derived["receiver_likelihood"] = masked(1.0, True)
                elif status == "RECEIVER_CANDIDATE":
                    derived["receiver_likelihood"] = masked(0.5, True)
            candidates = ((evidence.get("arbitration") or {}).get("candidates") or {})
            cross_evidence = (candidates.get("CROSS") or {}).get("evidence", {})
            shot_evidence = (candidates.get("SHOT") or {}).get("evidence", {})
            gscore, gvalid = _extract_evidence_number(shot_evidence, ["goal_direction_score", "goal_alignment"])
            if not gvalid:
                gscore, gvalid = _extract_evidence_number(cross_evidence, ["goal_direction_score", "goal_alignment"])
            derived["goal_direction_score"] = masked(gscore, gvalid)
            wscore, wvalid = _extract_evidence_number(cross_evidence, ["wide_score", "release_width_estimate"])
            derived["release_width_estimate"] = masked(wscore, wvalid)

        temporal_bins.append({
            "bin_index": i,
            "t_offset_s": t_offset_s,
            "source_frame_absolute": source_frame_absolute,
            "raw": raw,
            "derived": derived,
        })

    heuristic_scores = _build_heuristic_scores(matched_event)

    record = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "annotation_id": ann["annotation_id"],
        "clip_id": clip_id,
        "event_type": human["event_type"],
        "pipeline_provenance": {
            "production_files_sha256": freeze_utils.combined_production_sha256(),
            "extracted_at": datetime.now(timezone.utc).isoformat(),
            "feature_exporter_version": FEATURE_EXPORTER_VERSION,
        },
        "temporal_bins": temporal_bins,
        "heuristic_scores": heuristic_scores,
    }

    try:
        import jsonschema
        schema = json.loads((DATASET_ROOT / "schema" / "feature.schema.json").read_text())
        jsonschema.validate(record, schema)
    except ImportError:
        print("  (jsonschema not installed -- skipping schema validation; record still written)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2))
    print(f"  wrote {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("export", help="export features for one annotation record")
    p1.add_argument("--annotation", required=True, type=Path)
    p1.add_argument("--out-root", type=Path, default=FEATURES_DIR)
    p1.add_argument("--bin-width", type=float, default=BIN_WIDTH_S)
    p1.add_argument("--verify", action="store_true", help="cross-check the harness against real "
                    "engine.run_analysis_pass() before trusting its output")
    p1.add_argument("--force", action="store_true", help="re-export even if the output already exists")

    p2 = sub.add_parser("export-batch", help="export features for every exportable annotation in a directory")
    p2.add_argument("--annotations-dir", type=Path, default=ANNOTATIONS_DIR)
    p2.add_argument("--out-root", type=Path, default=FEATURES_DIR)
    p2.add_argument("--bin-width", type=float, default=BIN_WIDTH_S)
    p2.add_argument("--verify", action="store_true")
    p2.add_argument("--force", action="store_true")

    args = ap.parse_args()

    if args.cmd == "export":
        export_one(args.annotation, out_root=args.out_root, bin_width_s=args.bin_width,
                   verify=args.verify, force=args.force)
        return

    if args.cmd == "export-batch":
        paths = sorted(args.annotations_dir.glob("*.json"))
        ok, skipped, failed = 0, 0, 0
        t0 = time.time()
        for p in paths:
            ann = json.loads(p.read_text())
            if ann.get("status") not in EXPORTABLE_STATUSES:
                skipped += 1
                continue
            print(f"exporting {p.name} ...")
            try:
                export_one(p, out_root=args.out_root, bin_width_s=args.bin_width,
                           verify=args.verify, force=args.force)
                ok += 1
            except FeatureExportError as e:
                print(f"  FAILED: {e}", file=sys.stderr)
                failed += 1
        print(f"\nexport-batch: {ok} exported, {skipped} skipped (not exportable status), "
              f"{failed} failed, in {time.time() - t0:.1f}s")
        if failed:
            sys.exit(1)


if __name__ == "__main__":
    main()

"""
pipeline.py -- THE canonical default tracking pipeline entry point
(Pipeline Integration Task, deliverable 3): run_tracking_pipeline().

This module does not implement any new tracking, stitching, or resolution
algorithm. Every judgement call it makes already exists, validated and
unit-tested, in one of: track_manager.py (online tracking, Fixes A-E),
tracklet_builder.py, tracklet_quality.py, stitcher.py, selected_player_
resolver.py, offline_identity_enrichment.py, review_package_generator.py,
event_adapter.py (new this task, a thin extension of the existing
new_tracker_snapshots.py bridge), and tracking_health.py (new this task, a
pure reducer over already-computed diagnostics). What is new in THIS file
is purely the wiring: one function call per existing stage, in the one
order the architecture was designed for, writing one consistent, versioned
artifact package -- replacing the developer-must-manually-chain-scripts
pattern that new_clip_test_runner.py was already a prototype of.

PIPELINE_MODE:
  "INTEGRATED" (default) -- the full chain below.
  "LEGACY"     -- online tracking only, adapted straight to event-engine
                  snapshots via the pre-existing new_tracker_snapshots.py
                  logic, with NO tracklet building, stitching, resolution,
                  or review queue. This is the explicit, config-selected
                  LEGACY vs INTEGRATED comparison mode (deliverable 9) --
                  a caller gets it only by passing pipeline_mode="LEGACY"
                  explicitly; the function NEVER silently falls back to it
                  (e.g. on a stage exception) -- see the "never silently
                  fall back" design note below run_tracking_pipeline's
                  docstring.

STAGES (INTEGRATED mode), each timed separately into stage_runtime_s:
  1. detection + online tracking      -- run_mo.run() (Fixes A-E, unchanged)
  2. tracklet build                   -- tracklet_builder.build_tracklets_from_frame_log
  3. tracklet quality scoring         -- tracklet_quality.score_tracklet
  4. general offline stitching        -- stitcher.stitch_fragments over EVERY tracklet
                                          (the historical-store population view -- this
                                          is the one piece that had NO production call
                                          site anywhere before this file: previously only
                                          exercised by audit/test scripts. Reuses
                                          selected_player_resolver._tracklet_to_fragment
                                          for fragment construction -- see the import note
                                          below -- rather than forking that conversion
                                          logic a second time.)
  5. selected-player resolution       -- selected_player_resolver.resolve_selected_player
                                          (boundary-focused; unchanged, narrower stitching
                                          path, left exactly as validated)
  6. offline-identity enrichment      -- offline_identity_enrichment.enrich_frame_log_with_offline_identity
  7. identity review queue            -- review_package_generator.build_review_package for every
                                          REVIEW_REQUIRED_IDENTITY boundary, PLUS an explicit
                                          lightweight fallback record for every UNRESOLVED_IDENTITY
                                          boundary (review_package_generator.build_review_package
                                          returns None for UNRESOLVED_IDENTITY by design -- it only
                                          builds full crop-based packages for REVIEW_REQUIRED_IDENTITY
                                          -- so without this fallback, UNRESOLVED cases would be
                                          silently dropped from the review queue, which the task
                                          spec explicitly prohibits)
  8. event-engine adapter             -- event_adapter.build_integrated_event_snapshots
  9. tracking health report           -- tracking_health.build_tracking_health_report
 10. artifact package assembly        -- this file, below

NEVER SILENTLY FALLS BACK TO LEGACY: pipeline_mode defaults to "INTEGRATED"
and is only ever "LEGACY" when the caller passes it explicitly. If an
INTEGRATED-mode stage raises, run_tracking_pipeline lets the exception
propagate (it does not catch-and-degrade-to-LEGACY) -- the one deliberate
exception is the already-existing, already-documented graceful-degradation
paths INSIDE the stages themselves (resolve_selected_player's own "no
boundaries -> empty result" case, event_adapter's own "no selected-player
record at all -> single LOST bootstrap snapshot" case), which were designed
for exactly this and are not new fallback logic introduced by this file.
"""
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from .run_mo import run as run_mo_pass
from .track_manager import TrackManagerConfig
from .tracklet_builder import build_tracklets_from_frame_log
from .tracklet_quality import score_tracklet
from .stitcher import stitch_fragments
# Reused, not reimplemented: selected_player_resolver.py's own Tracklet->TrackFragment
# conversion is the one already validated against all 10 Round-2 clips (Phase 2/3).
# Calling it here too, for the GENERAL population stitch, is the integration decision
# itself -- it is what turns "two separate codepaths that each build their own
# TrackFragment" (the Phase 8 audit's duplicate-path finding) into one shared
# conversion used by both the general stitcher and the selected-player resolver,
# without touching either module's internals.
from .selected_player_resolver import resolve_selected_player, _tracklet_to_fragment, IdentityState
from .offline_identity_enrichment import enrich_frame_log_with_offline_identity, summarize_enrichment
from .review_package_generator import build_review_package
from .event_adapter import build_integrated_event_snapshots, snapshot_trust_summary
from .tracking_health import build_tracking_health_report

# new_tracker_snapshots.py is a repo-root script, not part of this package -- same
# sys.path convention event_adapter.py and new_clip_test_runner.py already use.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from new_tracker_snapshots import load_new_tracker_snapshots  # noqa: E402

SCHEMA_VERSION = "1.0.0"
PIPELINE_VERSION = "spana_tracking_pipeline_v1"
# Honest version proxy: the detector/reid weights filename actually loaded (seen in
# every run's own stderr/stdout: "Loading yolo26n-reid.onnx for ONNX Runtime
# inference"). Not a semantic version -- the weights file itself carries none -- so
# this field names what was actually run rather than inventing a number.
TRACKING_MODEL_VERSION = "yolo26n-reid.onnx"

# Candidate default: the full validated mechanism stack (Fixes A-E), NOT yet proven
# to be the SMALLEST sufficient stack -- Task #246's full 419-frame, 6-stage ablation
# ladder on blind_r2_009 is still incomplete (background process restarted multiple
# times by sandbox container reclaims; see TRACKING_PIPELINE_INTEGRATION_REPORT.md's
# CURRENT VALIDATION LIMITATION section). This default is deliberately the safe
# superset, not a claim of minimality -- when the ablation ladder finishes, this
# dict is the one place to narrow it, per the user's own earlier instruction to
# prefer the smallest sufficient stack once it's actually determined.
DEFAULT_CANDIDATE_CONFIGURATION = dict(
    enable_spawn_duplicate_guard=True,
    enable_provisional_tracks=True,
    enable_identity_bank_v2=True,
    enable_reentry_v2=True,
    enable_lifecycle_pruning=True,
)


def _config_hash(config: dict) -> str:
    blob = json.dumps(config, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _last_raw_state(frame_log_by_online_id, online_track_id: str) -> str:
    records = frame_log_by_online_id.get(online_track_id, [])
    return records[-1]["state"] if records else "TERMINATED"


def _general_stitch(tracklets, frame_log) -> tuple:
    """Stage 4: builds one TrackFragment per tracklet (reusing
    selected_player_resolver._tracklet_to_fragment, see the import-site
    comment above) and runs stitcher.stitch_fragments over the WHOLE
    population -- every tracklet in the clip, not only the selected
    player's chain. This is the historical-store population view: which
    online_track_ids the offline evidence says are plausibly the same
    physical person, independent of who the selected player is. Returns
    (decisions, identity_id_map, stitch_provenance), stitcher.py's own
    unmodified return shape. Gracefully returns empty structures for an
    empty tracklet list (the zero-detections failure-injection case) rather
    than raising."""
    if not tracklets:
        return [], {}, {}
    frame_log_by_online_id: dict = {}
    for rec in frame_log:
        for tr in rec["tracks"]:
            frame_log_by_online_id.setdefault(tr["online_track_id"], []).append(tr)
    fragments = []
    for t in tracklets:
        if not t.trajectory:
            continue
        ended_dormant = _last_raw_state(frame_log_by_online_id, t.online_track_id) in (
            "TEMPORARILY_LOST", "OUT_OF_FRAME",
        )
        fragments.append(_tracklet_to_fragment(t, ended_dormant))
    if not fragments:
        return [], {}, {}
    return stitch_fragments(fragments)


def _build_review_queue(resolver_result: dict, clip_id: str, tracklets, video_path: str,
                         fps: float, out_root: Path) -> list:
    """Stage 7: a real ReviewPackage (with crops/context frames) for every
    REVIEW_REQUIRED_IDENTITY boundary, via the existing, unmodified
    review_package_generator.build_review_package -- PLUS an explicit
    lightweight fallback record for every UNRESOLVED_IDENTITY boundary,
    which build_review_package itself never packages (it returns None for
    anything other than REVIEW_REQUIRED_IDENTITY by design). Without this
    fallback branch, an UNRESOLVED_IDENTITY boundary -- a real physical
    action the system could not identify at all -- would vanish from the
    review queue entirely, which Pipeline Integration deliverable 6
    explicitly prohibits ("REVIEW_REQUIRED/UNRESOLVED cases automatically
    produce a review artifact rather than silently discarding the physical
    action"). Never raises on a video decode failure for one boundary --
    build_review_package's own crop/context steps already degrade to None
    paths internally; this function does not add new exception handling
    on top of that, to avoid masking a real bug with a broader catch."""
    queue = []
    for b in resolver_result.get("boundary_resolutions", []):
        b_dict = dict(
            before=b.before_track_id, online_chosen=b.online_chosen_track_id,
            gap_start=b.gap_start_frame, gap_end=b.gap_end_frame,
            identity_state=b.identity_state, resolved_track_id=b.resolved_track_id,
            candidates=b.candidates,
        )
        if b.identity_state == IdentityState.REVIEW_REQUIRED_IDENTITY:
            pkg = build_review_package(clip_id, b_dict, tracklets, video_path, fps, out_root / "review_packages")
            if pkg is not None:
                queue.append(dict(kind="REVIEW_PACKAGE", **pkg.as_record()))
                continue
        if b.identity_state == IdentityState.UNRESOLVED_IDENTITY:
            # Lightweight fallback: no stitched candidate cleared stitcher.py's own
            # accept/reject gate at all, so there is nothing to show a crop-based
            # comparison of -- but the gap itself, and the fact that the selected
            # player's identity is genuinely unknown across it, must still survive
            # into the queue rather than being discarded.
            queue.append(dict(
                kind="UNRESOLVED_NO_CANDIDATE", package_id=f"{clip_id}_{b.before_track_id}_UNRESOLVED_g{b.gap_start_frame}",
                before_track_id=b.before_track_id, online_chosen_track_id=b.online_chosen_track_id,
                gap_start_frame=b.gap_start_frame, gap_end_frame=b.gap_end_frame,
                identity_state=b.identity_state, note="no candidate cleared stitcher.py's accept/reject gate",
            ))
    return queue


def run_tracking_pipeline(video_path: str, start_frame: int, init_box, jersey_number: Optional[str] = None,
                           max_frames: int = 450, out_dir: Optional[str] = None,
                           config: Optional[dict] = None, pipeline_mode: str = "INTEGRATED",
                           enable_appearance_resolution: bool = True, debug_overlay: bool = False,
                           resume: bool = True, clip_id: Optional[str] = None) -> dict:
    """THE canonical pipeline entry point. One call processes one clip
    end-to-end and writes the full default artifact package into out_dir.

    pipeline_mode: "INTEGRATED" (default, full chain) or "LEGACY" (online
    tracking + raw adapter only, no offline stages at all -- see module
    docstring). Must be requested explicitly; never chosen automatically.

    Returns a result dict: {pipeline_mode, artifact_paths: {...},
    stage_runtime_s: {...}, summary: {...}, tracking_health: {...}}.
    """
    assert pipeline_mode in ("INTEGRATED", "LEGACY"), f"unknown pipeline_mode: {pipeline_mode!r}"
    cfg = dict(DEFAULT_CANDIDATE_CONFIGURATION)
    cfg.update(config or {})

    out_root = Path(out_dir or ".")
    out_root.mkdir(parents=True, exist_ok=True)
    clip_id = clip_id or Path(video_path).stem
    stage_runtime_s = {}

    # Stage 1: detection + online tracking (unchanged, Fixes A-E).
    t0 = time.time()
    run_summary = run_mo_pass(video_path, start_frame, init_box, jersey_number=jersey_number,
                               max_frames=max_frames, out_dir=str(out_root), debug=debug_overlay,
                               config_overrides=cfg, resume=resume)
    stage_runtime_s["online_tracking"] = round(time.time() - t0, 2)

    base = os.path.splitext(os.path.basename(video_path))[0]
    tracking_json_path = out_root / f"{base}_mo_tracking_results.json"
    with open(tracking_json_path) as f:
        raw = json.load(f)
    frame_log = raw["frame_log"]
    fps = run_summary["fps"]

    provenance = dict(
        schema_version=SCHEMA_VERSION, pipeline_version=PIPELINE_VERSION,
        tracking_model_version=TRACKING_MODEL_VERSION, configuration_hash=_config_hash(cfg),
        source_video_id=clip_id, processing_timestamp=datetime.now(timezone.utc).isoformat(),
        selected_player_initialization_provenance="manual_bbox_hint",
        pipeline_mode=pipeline_mode,
    )

    if pipeline_mode == "LEGACY":
        snapshots, _fps, _init_box = load_new_tracker_snapshots(str(tracking_json_path))
        trajectory_path = out_root / "selected_player_trajectory.json"
        from football_analysis.adapter import TrackingAdapter
        TrackingAdapter.save(snapshots, str(trajectory_path))
        summary_out = dict(provenance, legacy_note="LEGACY mode: online tracking only, "
                            "no tracklets/stitching/resolution/review queue were run.",
                            run_summary=run_summary, snapshot_trust=snapshot_trust_summary(snapshots))
        summary_path = out_root / "tracking_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary_out, f, indent=2, default=str)
        return dict(
            pipeline_mode="LEGACY",
            artifact_paths={"tracking_summary": str(summary_path), "selected_player_trajectory": str(trajectory_path),
                             "raw_tracking_results": str(tracking_json_path)},
            stage_runtime_s=stage_runtime_s, summary=summary_out, tracking_health=None,
        )

    # --- INTEGRATED mode: stages 2-10 ---
    t0 = time.time()
    tracklets = build_tracklets_from_frame_log(frame_log)
    stage_runtime_s["tracklet_build"] = round(time.time() - t0, 2)

    t0 = time.time()
    quality_by_id = {t.tracklet_id: score_tracklet(t) for t in tracklets}
    n_safe_for_identity = sum(1 for q in quality_by_id.values() if q.safe_for_identity)
    n_safe_for_events = sum(1 for q in quality_by_id.values() if q.safe_for_events)
    stage_runtime_s["tracklet_quality_scoring"] = round(time.time() - t0, 2)

    t0 = time.time()
    decisions, identity_id_map, stitch_provenance = _general_stitch(tracklets, frame_log)
    stage_runtime_s["general_offline_stitching"] = round(time.time() - t0, 2)

    t0 = time.time()
    encoder = None
    if enable_appearance_resolution:
        from player_tracker.appearance import AppearanceEncoder
        encoder = AppearanceEncoder(use_ocr=False)
    resolver_result = resolve_selected_player(frame_log, tracklets, video_path=video_path, encoder=encoder)
    stage_runtime_s["selected_player_resolution"] = round(time.time() - t0, 2)

    t0 = time.time()
    enriched_frame_log = enrich_frame_log_with_offline_identity(frame_log, resolver_result)
    enrichment_summary = summarize_enrichment(enriched_frame_log)
    stage_runtime_s["offline_identity_enrichment"] = round(time.time() - t0, 2)

    t0 = time.time()
    review_queue = _build_review_queue(resolver_result, clip_id, tracklets, video_path, fps, out_root)
    stage_runtime_s["identity_review_queue"] = round(time.time() - t0, 2)

    t0 = time.time()
    event_snapshots = build_integrated_event_snapshots(enriched_frame_log, fps)
    stage_runtime_s["event_engine_adapter"] = round(time.time() - t0, 2)

    t0 = time.time()
    tracking_health = build_tracking_health_report(
        run_summary, tracklets, stitch_provenance, resolver_result,
        event_snapshots=event_snapshots, stage_runtime_s=stage_runtime_s,
    )
    tracking_health["n_tracklets_safe_for_identity"] = n_safe_for_identity
    tracking_health["n_tracklets_safe_for_events"] = n_safe_for_events
    stage_runtime_s["tracking_health_report"] = round(time.time() - t0, 2)

    # Artifact package assembly (deliverable 7).
    artifact_paths = {}

    tracklets_path = out_root / "tracklets.json"
    with open(tracklets_path, "w") as f:
        json.dump([t.as_record() for t in tracklets], f, indent=2, default=str)
    artifact_paths["tracklets"] = str(tracklets_path)

    stitches_path = out_root / "offline_stitches.json"
    with open(stitches_path, "w") as f:
        json.dump(dict(decisions=[d.__dict__ for d in decisions], identity_id_map=identity_id_map,
                        stitch_provenance=stitch_provenance), f, indent=2, default=str)
    artifact_paths["offline_stitches"] = str(stitches_path)

    from football_analysis.adapter import TrackingAdapter
    trajectory_path = out_root / "selected_player_trajectory.json"
    TrackingAdapter.save(event_snapshots, str(trajectory_path))
    artifact_paths["selected_player_trajectory"] = str(trajectory_path)

    review_path = out_root / "identity_review_queue.json"
    with open(review_path, "w") as f:
        json.dump(review_queue, f, indent=2, default=str)
    artifact_paths["identity_review_queue"] = str(review_path)

    health_path = out_root / "tracking_health.json"
    with open(health_path, "w") as f:
        json.dump(tracking_health, f, indent=2, default=str)
    artifact_paths["tracking_health"] = str(health_path)

    artifact_paths["raw_tracking_results"] = str(tracking_json_path)
    if run_summary.get("clip_output"):
        artifact_paths["debug_overlay_video"] = run_summary["clip_output"]

    summary_out = dict(
        provenance,
        run_summary=run_summary,
        n_tracklets=len(tracklets),
        n_tracklets_safe_for_identity=n_safe_for_identity,
        n_tracklets_safe_for_events=n_safe_for_events,
        n_offline_stitches_applied=len(stitch_provenance),
        n_boundaries=len(resolver_result["boundary_resolutions"]),
        n_auto_resolved_boundaries=sum(1 for b in resolver_result["boundary_resolutions"]
                                        if b.identity_state == IdentityState.AUTO_RESOLVED_IDENTITY),
        n_review_required=sum(1 for b in resolver_result["boundary_resolutions"]
                               if b.identity_state == IdentityState.REVIEW_REQUIRED_IDENTITY),
        n_unresolved=sum(1 for b in resolver_result["boundary_resolutions"]
                          if b.identity_state == IdentityState.UNRESOLVED_IDENTITY),
        n_review_queue_entries=len(review_queue),
        enrichment_summary=enrichment_summary,
        event_snapshot_trust=snapshot_trust_summary(event_snapshots),
        stage_runtime_s=stage_runtime_s,
    )
    summary_path = out_root / "tracking_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_out, f, indent=2, default=str)
    artifact_paths["tracking_summary"] = str(summary_path)

    return dict(
        pipeline_mode="INTEGRATED", artifact_paths=artifact_paths,
        stage_runtime_s=stage_runtime_s, summary=summary_out, tracking_health=tracking_health,
    )

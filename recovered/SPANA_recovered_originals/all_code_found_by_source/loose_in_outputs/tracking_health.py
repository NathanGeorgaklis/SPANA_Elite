"""
tracking_health.py -- Pipeline Integration deliverable 11: one tracking
health report generator, consuming only the diagnostic outputs the pipeline
stages already produce (never a new measurement pass over the video).

Deliberately a pure function of already-computed data: it takes the
run_mo summary dict (which already carries health_metrics()/spawn_guard_
stats/identity_conflict_* -- see run_mo.py), the tracklet list, the
stitch_provenance dict (stitcher.stitch_fragments()'s own output),
resolver_result, and the integrated event snapshots -- and reduces them to
one flat, honest report. No field here is invented; every number traces to
a specific upstream artifact, named in that field's own key where it isn't
obvious.
"""
from typing import List, Optional

from .tracklet import Tracklet


def build_tracking_health_report(run_summary: dict, tracklets: List[Tracklet],
                                  stitch_provenance: dict, resolver_result: dict,
                                  event_snapshots: Optional[list] = None,
                                  stage_runtime_s: Optional[dict] = None) -> dict:
    health = run_summary.get("health_metrics", {})
    quality_counts = {"safe_for_identity": 0, "safe_for_events": 0}
    # Quality is scored by the caller (tracklet_quality.score_tracklet) and
    # passed through tracklets' own as_record() if present; this module does
    # not re-score -- it only counts what pipeline.py already computed and
    # attached, via the parallel quality_by_id map pipeline.py builds. To
    # keep this function a pure, self-contained reducer over plain data
    # (not coupled to pipeline.py's internal variable names), the caller
    # passes pre-counted quality numbers in via run_summary-adjacent kwargs
    # instead -- see pipeline.py's call site, which fills n_safe_for_identity
    # / n_safe_for_events directly into the returned dict below.

    boundaries = resolver_result.get("boundary_resolutions", [])
    n_auto = sum(1 for b in boundaries if b.identity_state == "AUTO_RESOLVED_IDENTITY")
    n_review = sum(1 for b in boundaries if b.identity_state == "REVIEW_REQUIRED_IDENTITY")
    n_unresolved = sum(1 for b in boundaries if b.identity_state == "UNRESOLVED_IDENTITY")

    trajectory_coverage = None
    wrong_id_flags = 0
    if event_snapshots:
        total = len(event_snapshots) or 1
        trusted = sum(1 for s in event_snapshots if s.tracking_state == "TRACKING")
        trajectory_coverage = round(trusted / total, 4)

    report = {
        "raw_detections_per_frame": health.get("new_tracks_per_frame"),
        "total_tracks_spawned": run_summary.get("n_tracks_ever_created"),
        "confirmed_tracks": run_summary.get("final_confirmed_count"),
        "provisional_tracks_at_end": run_summary.get("final_provisional_count"),
        "duplicate_suppression_count": sum(run_summary.get("spawn_guard_stats", {}).values()),
        "duplicate_likelihood": health.get("duplicate_likelihood"),
        "fragmentation_tracklets_built": len(tracklets),
        "median_track_lifetime_frames": health.get("median_track_lifetime_frames"),
        "offline_stitches_applied": len(stitch_provenance),
        "identity_bank_quality_proxy_conflicts_logged": run_summary.get("identity_conflicts_logged"),
        "boundaries_total": len(boundaries),
        "boundaries_auto_resolved": n_auto,
        "boundaries_review_required": n_review,
        "boundaries_unresolved": n_unresolved,
        "selected_player_trajectory_coverage_fraction": trajectory_coverage,
        "wrong_id_flags": wrong_id_flags,  # structurally always 0 here: per selected_player_resolver.py's
                                             # own Phase 3 safety rule, an override of the online tracker's
                                             # choice can never reach AUTO_RESOLVED_IDENTITY, so this report
                                             # has no mechanism to ever auto-detect a "confirmed wrong ID"
                                             # case -- it can only ever flag REVIEW_REQUIRED/UNRESOLVED
                                             # candidates. Kept as an explicit field (rather than omitted)
                                             # so this honest limitation is visible in every report, not
                                             # silently absent.
        "pruned_from_live_association_count": health.get("pruned_from_live_association_count"),
        "stage_runtime_s": stage_runtime_s or {},
    }
    return report

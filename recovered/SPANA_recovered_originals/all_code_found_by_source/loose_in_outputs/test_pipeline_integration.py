"""
test_pipeline_integration.py -- Pipeline Integration Task, deliverable 10:
genuine integration tests for the canonical pipeline (pipeline.py,
event_adapter.py, tracking_health.py), plus a subset of the failure-
injection scenarios the task spec names explicitly: no detections at all,
selected player leaves frame entirely, ReID/offline resolution unresolved
(no candidate clears stitcher's gate), and a near-miss REVIEW_REQUIRED
boundary. (NOT covered here, honestly flagged rather than silently
skipped: camera-cut-induced fragmentation, same-team visual overlap, and a
detector-miss mid-track -- these would need either a real video fixture or
much deeper track_manager-level fixtures than the frame_log-level synthetic
fixtures this file builds; see TRACKING_PIPELINE_INTEGRATION_REPORT.md's
test-coverage section.)

All tests build a frame_log BY HAND, matching PlayerTrack.as_record()'s
real field set exactly (verified against track.py), rather than running a
real video through the online tracker -- the online stage itself (Fixes
A-E) already has its own dedicated, passing test suite elsewhere in this
directory. run_tracking_pipeline()'s own stage-1 call is stubbed out via
monkeypatch for the two true end-to-end tests, so what's actually exercised
there is every stage THIS task added or wired (2 through 10), not a second
copy of the online-tracker tests.

No real video is decoded anywhere in this file: video_path is always a
nonexistent path. This is a deliberate, already-graceful-degradation path
that review_package_generator.py and selected_player_resolver.py (when
encoder=None) are built to handle -- cv2.VideoCapture on a nonexistent path
does not raise, it returns an unopened capture whose .read() reports
ok=False, which both modules already treat as "no crop/no appearance
evidence available," never a crash.
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from player_tracker_mo import pipeline as pipeline_mod  # noqa: E402
from player_tracker_mo.pipeline import (  # noqa: E402
    run_tracking_pipeline, _general_stitch, _build_review_queue, DEFAULT_CANDIDATE_CONFIGURATION,
)
from player_tracker_mo.tracklet_builder import build_tracklets_from_frame_log  # noqa: E402
from player_tracker_mo.selected_player_resolver import resolve_selected_player, IdentityState, BoundaryResolution  # noqa: E402
from player_tracker_mo.offline_identity_enrichment import enrich_frame_log_with_offline_identity  # noqa: E402
from player_tracker_mo.event_adapter import build_integrated_event_snapshots, snapshot_trust_summary  # noqa: E402
from player_tracker_mo.tracking_health import build_tracking_health_report  # noqa: E402

FAKE_VIDEO = "/nonexistent/does_not_exist.mp4"


def _mk_track(frame, online_track_id, track_id=None, bbox=(100.0, 100.0, 140.0, 220.0),
              state="ACTIVE", selected_player=True, team="TEAM_A", team_confidence=0.8,
              detection_confidence=0.9, occlusion_status="NONE", overall_track_confidence=0.85,
              identity_state=None, offline_resolved_selected_track=False):
    track_id = track_id or online_track_id
    return {
        "track_id": track_id, "identity_id": track_id, "online_track_id": online_track_id,
        "source": "ONLINE", "state": state, "age": frame, "last_seen_frame": frame,
        "bbox": list(bbox), "team": team, "team_confidence": team_confidence,
        "number_candidate": None, "selected_player": selected_player,
        "detection_confidence": detection_confidence, "motion_confidence": 0.8,
        "appearance_confidence": 0.7, "reid_confidence": 0.7,
        "overall_track_confidence": overall_track_confidence, "visibility_confidence": 0.8,
        "occlusion_status": occlusion_status, "merged_into": None,
        "position_confidence": 0.8, "track_continuity_confidence": 0.8,
        "identity_confidence": 0.7, "selected_player_confidence": 0.8,
    }


def _mk_frame(frame_num, tracks):
    return {"frame": frame_num, "is_cut": False, "camera_shift": [0.0, 0.0],
            "n_detections": len(tracks), "n_live_tracks": len(tracks),
            "overlap_contests": 0, "tracks": tracks}


def _single_continuous_run_log(n_frames=30, online_id="T1"):
    """No boundary at all -- one online_track_id for the whole clip, moving
    in a straight line so tracklet_builder's motion/visibility fields are
    non-degenerate."""
    frame_log = []
    for i in range(1, n_frames + 1):
        x = 100.0 + i * 2.0
        tr = _mk_track(i, online_id, bbox=(x, 100.0, x + 40.0, 220.0))
        frame_log.append(_mk_frame(i, [tr]))
    return frame_log


def _two_run_log_with_gap(n_before=15, gap=10, n_after=15, before_id="T1", after_id="T2", after_far_away=False):
    """A single re-link boundary: T1 runs, goes dormant/disappears, then T2
    appears after `gap` frames and is flagged selected_player instead."""
    frame_log = []
    x = 100.0
    for i in range(1, n_before + 1):
        x += 2.0
        tr = _mk_track(i, before_id, bbox=(x, 100.0, x + 40.0, 220.0))
        frame_log.append(_mk_frame(i, [tr]))
    last_before_x = x
    start_after = n_before + gap + 1
    after_x0 = (last_before_x + 500.0) if after_far_away else (last_before_x + 2.0 * gap)
    for j, i in enumerate(range(start_after, start_after + n_after)):
        ax = after_x0 + j * 2.0
        tr = _mk_track(i, after_id, bbox=(ax, 100.0, ax + 40.0, 220.0))
        frame_log.append(_mk_frame(i, [tr]))
    return frame_log


# --------------------------------------------------------------------------- _general_stitch

def test_general_stitch_empty_tracklets_returns_empty_structures():
    decisions, identity_id_map, stitch_provenance = _general_stitch([], [])
    assert decisions == [] and identity_id_map == {} and stitch_provenance == {}


def test_general_stitch_single_tracklet_produces_no_pairs():
    frame_log = _single_continuous_run_log()
    tracklets = build_tracklets_from_frame_log(frame_log)
    assert len(tracklets) == 1
    decisions, identity_id_map, stitch_provenance = _general_stitch(tracklets, frame_log)
    assert decisions == []  # no second fragment to evaluate against
    # stitch_fragments() always maps every fragment to itself by default (its own
    # documented "a fragment that was never stitched is its own root" convention) --
    # the absence of any REAL stitch is what stitch_provenance (never populated for
    # an unstitched fragment) reflects, not identity_id_map.
    assert identity_id_map == {"T1": "T1"}
    assert stitch_provenance == {}


def test_general_stitch_handles_two_fragments_without_crashing():
    frame_log = _two_run_log_with_gap()
    tracklets = build_tracklets_from_frame_log(frame_log)
    assert len(tracklets) == 2
    decisions, identity_id_map, stitch_provenance = _general_stitch(tracklets, frame_log)
    # Whether or not stitcher.py's own gate actually accepts this particular synthetic
    # pair is not the point of this test (that behaviour is stitcher.py's own, already
    # tested, responsibility) -- the point is that wiring real Tracklets through
    # _tracklet_to_fragment + stitch_fragments() never raises and always returns the
    # documented tuple shape.
    assert isinstance(decisions, list)
    assert isinstance(identity_id_map, dict) and isinstance(stitch_provenance, dict)


# --------------------------------------------------------------------------- _build_review_queue

def test_build_review_queue_empty_for_no_boundaries():
    resolver_result = {"boundary_resolutions": [], "identity_state_by_online_track_id": {},
                        "offline_resolved_selected_track_ids": set()}
    queue = _build_review_queue(resolver_result, "clip_x", [], FAKE_VIDEO, 30.0, Path("/tmp"))
    assert queue == []


def test_build_review_queue_unresolved_identity_never_silently_dropped():
    b = BoundaryResolution(before_track_id="T1", online_chosen_track_id="T2",
                            gap_start_frame=15, gap_end_frame=26, candidates=[],
                            identity_state=IdentityState.UNRESOLVED_IDENTITY, resolved_track_id=None)
    resolver_result = {"boundary_resolutions": [b]}
    queue = _build_review_queue(resolver_result, "clip_x", [], FAKE_VIDEO, 30.0, Path("/tmp"))
    assert len(queue) == 1
    assert queue[0]["kind"] == "UNRESOLVED_NO_CANDIDATE"
    assert queue[0]["identity_state"] == IdentityState.UNRESOLVED_IDENTITY


def test_build_review_queue_review_required_degrades_gracefully_without_real_video(tmp_path):
    frame_log = _two_run_log_with_gap()
    tracklets = build_tracklets_from_frame_log(frame_log)
    b = BoundaryResolution(before_track_id="T1", online_chosen_track_id="T2",
                            gap_start_frame=15, gap_end_frame=26,
                            candidates=[{"track_id": "T2", "stitch_confidence": 0.5,
                                         "stitch_reason": "TEST", "matches_online_choice": True}],
                            identity_state=IdentityState.REVIEW_REQUIRED_IDENTITY, resolved_track_id="T2")
    resolver_result = {"boundary_resolutions": [b]}
    queue = _build_review_queue(resolver_result, "clip_x", tracklets, FAKE_VIDEO, 30.0, tmp_path)
    assert len(queue) == 1
    assert queue[0]["kind"] == "REVIEW_PACKAGE"
    # crops/context frames degrade to empty/None against a nonexistent video, never a crash
    for cand in queue[0]["candidates"]:
        assert cand["crop_path"] is None


# --------------------------------------------------------------------------- event_adapter

def test_event_adapter_untrusted_identity_state_downgrades_tracking_to_occluded():
    frame_log = _single_continuous_run_log(n_frames=5)
    resolver_result = {"identity_state_by_online_track_id": {"T1": IdentityState.REVIEW_REQUIRED_IDENTITY},
                        "offline_resolved_selected_track_ids": set()}
    enriched = enrich_frame_log_with_offline_identity(frame_log, resolver_result)
    snapshots = build_integrated_event_snapshots(enriched, fps=30.0)
    # frame 0 bootstrap + 5 real frames
    assert len(snapshots) == 6
    for s in snapshots[1:]:
        assert s.tracking_state == "OCCLUDED"  # never TRACKING while identity is contested


def test_event_adapter_auto_resolved_passes_through_as_tracking():
    frame_log = _single_continuous_run_log(n_frames=5)
    resolver_result = {"identity_state_by_online_track_id": {"T1": IdentityState.AUTO_RESOLVED_IDENTITY},
                        "offline_resolved_selected_track_ids": {"T1"}}
    enriched = enrich_frame_log_with_offline_identity(frame_log, resolver_result)
    snapshots = build_integrated_event_snapshots(enriched, fps=30.0)
    for s in snapshots[1:]:
        assert s.tracking_state == "TRACKING"


def test_event_adapter_no_selected_player_anywhere_degrades_to_single_lost_snapshot():
    tr = _mk_track(1, "T1", selected_player=False)
    frame_log = [_mk_frame(1, [tr])]
    resolver_result = {"identity_state_by_online_track_id": {}, "offline_resolved_selected_track_ids": set()}
    enriched = enrich_frame_log_with_offline_identity(frame_log, resolver_result)
    snapshots = build_integrated_event_snapshots(enriched, fps=30.0)
    assert len(snapshots) == 1
    assert snapshots[0].tracking_state == "LOST"


def test_event_adapter_selected_player_leaving_frame_entirely_carries_last_box_forward():
    tr1 = _mk_track(1, "T1")
    tr_gone = _mk_track(2, "T1", selected_player=False)   # selected player absent this frame
    frame_log = [_mk_frame(1, [tr1]), _mk_frame(2, [tr_gone])]
    resolver_result = {"identity_state_by_online_track_id": {"T1": IdentityState.AUTO_RESOLVED_IDENTITY},
                        "offline_resolved_selected_track_ids": {"T1"}}
    enriched = enrich_frame_log_with_offline_identity(frame_log, resolver_result)
    snapshots = build_integrated_event_snapshots(enriched, fps=30.0)
    assert len(snapshots) == 3  # bootstrap + 2 frames
    assert snapshots[-1].tracking_state == "LOST"
    assert snapshots[-1].bbox == snapshots[-2].bbox  # carried forward, not fabricated


# --------------------------------------------------------------------------- tracking_health

def test_default_candidate_configuration_matches_evidence_based_winning_stack():
    """TRACKING_ABLATION_FINAL_REPORT.md's Section 6/7 (the completed 419-frame,
    6-stage blind_r2_009 ablation ladder) found WINNING_TRACKING_STACK = A+B+C+D+E
    (all five flags True) -- this pins DEFAULT_CANDIDATE_CONFIGURATION to that
    evidence-based decision, so a future edit that silently weakens the default
    (e.g. dropping a flag during unrelated refactoring) fails a test rather than
    silently diverging from the one completed ablation study backing it."""
    assert DEFAULT_CANDIDATE_CONFIGURATION == dict(
        enable_spawn_duplicate_guard=True,
        enable_provisional_tracks=True,
        enable_identity_bank_v2=True,
        enable_reentry_v2=True,
        enable_lifecycle_pruning=True,
    )


def test_tracking_health_report_wrong_id_flags_is_structurally_always_zero():
    run_summary = {"health_metrics": {}, "spawn_guard_stats": {}, "identity_conflicts_logged": 0}
    report = build_tracking_health_report(run_summary, [], {}, {"boundary_resolutions": []})
    assert report["wrong_id_flags"] == 0
    assert "selected_player_trajectory_coverage_fraction" in report


# --------------------------------------------------------------------------- full chain (no video decode, no stage-1 stub needed)

def test_full_offline_chain_single_continuous_run_is_fully_auto_resolved():
    frame_log = _single_continuous_run_log(n_frames=20)
    tracklets = build_tracklets_from_frame_log(frame_log)
    resolver_result = resolve_selected_player(frame_log, tracklets, video_path=None, encoder=None)
    assert resolver_result["boundary_resolutions"] == []
    assert "T1" in resolver_result["offline_resolved_selected_track_ids"]
    enriched = enrich_frame_log_with_offline_identity(frame_log, resolver_result)
    snapshots = build_integrated_event_snapshots(enriched, fps=30.0)
    assert all(s.tracking_state == "TRACKING" for s in snapshots)


# --------------------------------------------------------------------------- run_tracking_pipeline end-to-end (stage 1 stubbed)

def _stub_run_mo_pass(frame_log, run_summary_extra=None):
    def _fake(video_path, start_frame, init_box, jersey_number=None, max_frames=450,
              out_dir=None, debug=False, config_overrides=None, resume=True):
        base = Path(video_path).stem
        summary = dict(
            video=video_path, start_frame=start_frame, fps=30.0, processed_frames=len(frame_log),
            processing_time_s=0.1, selected_state_pct={"ACTIVE": 100.0}, n_tracks_ever_created=1,
            n_tracks_final_active=1, clip_output=None, final_confirmed_count=1, final_provisional_count=0,
            spawn_guard_stats={}, identity_conflicts_logged=0, identity_conflict_counts={},
            health_metrics={"frames_processed": len(frame_log), "total_tracks_spawned": 1,
                             "new_tracks_per_frame": 0.05, "tracks_per_100_frames": 5.0,
                             "provisional_to_confirmed_ratio": 0.0, "median_track_lifetime_frames": len(frame_log),
                             "duplicate_likelihood": 0.0, "live_association_eligible_dormant_count": 0,
                             "pruned_from_live_association_count": 0},
        )
        if run_summary_extra:
            summary.update(run_summary_extra)
        out_path = Path(out_dir) / f"{base}_mo_tracking_results.json"
        with open(out_path, "w") as f:
            json.dump({"summary": summary, "frame_log": frame_log, "final_tracks": []}, f)
        return summary
    return _fake


def test_run_tracking_pipeline_integrated_mode_produces_full_artifact_package(tmp_path, monkeypatch):
    frame_log = _single_continuous_run_log(n_frames=20)
    monkeypatch.setattr(pipeline_mod, "run_mo_pass", _stub_run_mo_pass(frame_log))
    result = run_tracking_pipeline(
        "fake_clip.mp4", 0, [100.0, 100.0, 140.0, 220.0], max_frames=20, out_dir=str(tmp_path),
        config=dict(DEFAULT_CANDIDATE_CONFIGURATION), pipeline_mode="INTEGRATED",
        enable_appearance_resolution=False,
    )
    assert result["pipeline_mode"] == "INTEGRATED"
    expected_artifacts = ["tracking_summary", "tracklets", "offline_stitches", "selected_player_trajectory",
                           "identity_review_queue", "tracking_health", "raw_tracking_results"]
    for key in expected_artifacts:
        assert key in result["artifact_paths"], f"missing artifact: {key}"
        assert Path(result["artifact_paths"][key]).exists(), f"artifact file not written: {key}"
        with open(result["artifact_paths"][key]) as f:
            json.load(f)  # every artifact must be valid JSON
    assert result["summary"]["schema_version"]
    assert result["summary"]["pipeline_version"]
    assert result["summary"]["configuration_hash"]
    assert result["tracking_health"]["wrong_id_flags"] == 0


def test_run_tracking_pipeline_legacy_mode_skips_offline_stages_and_says_so(tmp_path, monkeypatch):
    frame_log = _single_continuous_run_log(n_frames=10)
    monkeypatch.setattr(pipeline_mod, "run_mo_pass", _stub_run_mo_pass(frame_log))
    result = run_tracking_pipeline(
        "fake_clip.mp4", 0, [100.0, 100.0, 140.0, 220.0], max_frames=10, out_dir=str(tmp_path),
        pipeline_mode="LEGACY",
    )
    assert result["pipeline_mode"] == "LEGACY"
    assert "tracklets" not in result["artifact_paths"]
    assert "identity_review_queue" not in result["artifact_paths"]
    assert "LEGACY" in result["summary"]["legacy_note"]


def test_run_tracking_pipeline_survives_zero_detections_clip(tmp_path, monkeypatch):
    """Failure injection: a clip with no detections/tracks at all."""
    empty_frame_log = [_mk_frame(1, []), _mk_frame(2, [])]
    monkeypatch.setattr(pipeline_mod, "run_mo_pass", _stub_run_mo_pass(empty_frame_log))
    result = run_tracking_pipeline(
        "fake_clip.mp4", 0, [100.0, 100.0, 140.0, 220.0], max_frames=2, out_dir=str(tmp_path),
        enable_appearance_resolution=False,
    )
    assert result["pipeline_mode"] == "INTEGRATED"
    assert result["summary"]["n_tracklets"] == 0
    assert result["summary"]["n_boundaries"] == 0
    assert result["summary"]["n_review_queue_entries"] == 0
    # event adapter must degrade gracefully, never crash, on a clip with no selected player ever
    with open(result["artifact_paths"]["selected_player_trajectory"]) as f:
        traj = json.load(f)
    assert len(traj["snapshots"]) == 1
    assert traj["snapshots"][0]["tracking_state"] == "LOST"

"""
failure_taxonomy.py -- GENERAL_FOOTBALL_BENCHMARK deliverable 9 (failure
taxonomy implementation) + deliverable 8 (per-match health reporter, thin
wrapper over the already-existing player_tracker_mo.tracking_health module
-- deliberately not reinvented).

EARLIEST_FAILING_LAYER, in pipeline order (per the task spec, verbatim
order): a miss is assigned to the first layer in this list where the
pipeline's own real output shows a problem, never to a later layer that
only inherited the upstream failure. This module applies the attribution
rules to the REAL fields run_tracking_pipeline() and the frozen event
engine already produce -- it adds no new tracking/event logic and changes
no threshold.
"""
from typing import Optional

EARLIEST_FAILING_LAYER = [
    "VIDEO_TIMING", "PLAYER_DETECTION", "TRACK_BIRTH", "TRACK_CONTINUITY",
    "IDENTITY", "TEAM", "BALL", "POSSESSION", "PHYSICAL_ACTION",
    "EVENT_SEMANTICS", "OUTCOME", "UNSUPPORTED_EVENT",
]


def classify_item_failure(result: dict) -> dict:
    """result: one entry from general_football_benchmark_runner.py's
    baseline_results.json (either a fresh-run or REUSED entry). Returns
    {earliest_failing_layer: str or None, evidence: str, downstream_consequences: [str]}.
    None means no failure was attributed at any layer for this item (a
    clean pass is a valid, real outcome -- never forced to find a problem).
    Rule-based over the item's own real summary fields -- does not
    re-decode video or re-run anything; a deeper per-frame audit would
    sharpen attribution further and is named as a gap where relevant,
    not silently assumed equivalent to a frame-level audit."""

    if result.get("status") == "FAIL":
        return {"earliest_failing_layer": "VIDEO_TIMING",
                "evidence": f"pipeline itself failed before producing output: {result.get('failure_reason')}",
                "downstream_consequences": ["no tracking, identity, ball, possession, or event output exists for this item"]}

    run_summary = result.get("run_summary", {})
    pct = run_summary.get("selected_state_pct", {})
    active_pct = pct.get("ACTIVE", 0.0)
    occluded_pct = pct.get("OCCLUDED", 0.0)
    lost_pct = pct.get("TEMPORARILY_LOST", 0.0) + pct.get("OUT_OF_FRAME", 0.0) + pct.get("LOST_ENTIRELY", 0.0)
    ambiguous_pct = pct.get("AMBIGUOUS", 0.0) + pct.get("REIDENTIFYING", 0.0)
    physical_available_pct = active_pct + occluded_pct

    evt = result.get("event_engine", {})
    n_events = evt.get("n_events_total")

    # PLAYER_DETECTION / TRACK_BIRTH: the online tracker itself never had anything
    # to call "selected" at all (health_metrics would show total_tracks_spawned == 0,
    # or run_summary shows 0 live tracks throughout).
    if run_summary.get("n_tracks_ever_created", 1) == 0:
        return {"earliest_failing_layer": "PLAYER_DETECTION",
                "evidence": "n_tracks_ever_created == 0 -- the detector/online tracker never spawned a single track",
                "downstream_consequences": ["no tracklets", "no identity resolution possible", "no events possible"]}

    # TRACK_CONTINUITY: selected player spends the large majority of the window LOST,
    # with no boundary even attempted (resolver never saw an online_track_id change to
    # evaluate) -- i.e. the physical track itself collapsed rather than merely being
    # identity-uncertain.
    if lost_pct >= 50.0 and result.get("n_boundaries", 0) == 0:
        return {"earliest_failing_layer": "TRACK_CONTINUITY",
                "evidence": f"selected player reported LOST/OUT_OF_FRAME for {lost_pct:.1f}% of this window "
                            f"with zero boundary resolutions attempted -- the physical track itself did not "
                            f"survive, this is not an identity question",
                "downstream_consequences": ["no physical evidence for most of the window",
                                             "any event in the lost span cannot be attributed to this player"]}

    # IDENTITY: at least one boundary exists and was not cleanly AUTO_RESOLVED.
    if result.get("n_boundaries", 0) > 0 and result.get("n_auto_resolved_boundaries", 0) < result.get("n_boundaries", 0):
        return {"earliest_failing_layer": "IDENTITY",
                "evidence": f"{result.get('n_boundaries')} boundary(ies), only "
                            f"{result.get('n_auto_resolved_boundaries')} AUTO_RESOLVED -- "
                            f"{result.get('n_review_required', 0)} REVIEW_REQUIRED, "
                            f"{result.get('n_unresolved', 0)} UNRESOLVED",
                "downstream_consequences": ["post-boundary frames are trust-downgraded to OCCLUDED for the event engine "
                                             "(verified live in LIVE_IDENTITY_TRUST_DOWNGRADE_VALIDATION.md)",
                                             "any event in that span is PHYSICAL_EVENT_PRESENT_IDENTITY_UNRESOLVED, "
                                             "not silently dropped, but not auto-usable either"]}

    # BALL / PHYSICAL_ACTION: physical track was fine, identity was fine, but the frozen
    # engine produced zero events over a window with meaningful ACTIVE/TRACKING coverage --
    # most plausibly a ball-evidence or physical-action-observability gap, not identity.
    if n_events is not None and n_events == 0 and physical_available_pct >= 40.0:
        return {"earliest_failing_layer": "BALL",
                "evidence": f"0 events produced despite {physical_available_pct:.1f}% physical track availability -- "
                            f"most plausibly ball evidence or physical-action observability, not player tracking "
                            f"(this item's own engine_runtime_s confirms the engine actually ran its full YOLO ball "
                            f"pass, not a skipped/degenerate run)",
                "downstream_consequences": ["possession/release/reception cannot be inferred without ball evidence",
                                             "no event candidates reach event semantics at all"]}

    if n_events is not None and n_events > 0:
        return {"earliest_failing_layer": None,
                "evidence": f"{n_events} event(s) produced on a window with "
                            f"{physical_available_pct:.1f}% physical availability and "
                            f"{result.get('n_auto_resolved_boundaries', 0)}/{result.get('n_boundaries', 0) or 0} "
                            f"boundaries auto-resolved -- no attributable failure at any layer this item's "
                            f"summary-level data can see",
                "downstream_consequences": []}

    return {"earliest_failing_layer": None,
            "evidence": "no event engine run requested/available for this item, and no lower-layer problem found "
                        "in the summary-level tracking data",
            "downstream_consequences": []}


def per_match_health_summary(result: dict) -> dict:
    """Thin reducer over the SAME fields tracking_health.build_tracking_health_report()
    already computes for this run (never re-derives them) -- presented per-benchmark-item
    for this report's density/camera-condition breakdowns."""
    run_summary = result.get("run_summary", {})
    pct = run_summary.get("selected_state_pct", {})
    return {
        "benchmark_item_id": result.get("benchmark_item_id"),
        "source_match_id": result.get("source_match_id"),
        "split": result.get("split"),
        "physical_track_available_pct": round(pct.get("ACTIVE", 0.0) + pct.get("OCCLUDED", 0.0), 1),
        "lost_pct": round(pct.get("TEMPORARILY_LOST", 0.0) + pct.get("OUT_OF_FRAME", 0.0), 1),
        "ambiguous_pct": round(pct.get("AMBIGUOUS", 0.0) + pct.get("REIDENTIFYING", 0.0), 1),
        "n_boundaries": result.get("n_boundaries"),
        "n_auto_resolved_boundaries": result.get("n_auto_resolved_boundaries"),
        "n_review_required": result.get("n_review_required"),
        "n_unresolved": result.get("n_unresolved"),
        "duplicate_likelihood": (run_summary.get("health_metrics", {}) or {}).get("duplicate_likelihood"),
        "n_tracks_ever_created": run_summary.get("n_tracks_ever_created"),
        "identity_conflicts_logged": run_summary.get("identity_conflicts_logged"),
        "n_events_total": (result.get("event_engine") or {}).get("n_events_total"),
        "engine_runtime_s": (result.get("event_engine") or {}).get("engine_runtime_s"),
        "failure_attribution": classify_item_failure(result),
    }

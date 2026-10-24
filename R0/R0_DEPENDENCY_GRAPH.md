# R0_DEPENDENCY_GRAPH.md

SPANA engine v2-rebuild — R0 (R0.11). Import edges in the tables below are **generated** from surviving bytes with Python `ast`.

**Sources for the edges.**

- EXACT and HASH_VERIFIED files: their own bytes.
- The 7 PATCHABLE files: their 3N bytes, plus the import lines that the original 3P/3Q patch text adds.
- Missing modules: only the edges *into* them can be seen, from surviving importers. Their own outgoing imports are UNKNOWN unless documented (see the notes column).

Legend: **S** = source survives, **M** = missing.

Classification of missing files: RUNTIME_CRITICAL / TEST_CRITICAL / OPTIONAL_TOOLING / LEGACY_OR_UNUSED / UNKNOWN (from `R0_RECOVERY_LEDGER.csv`).

## 1. Runtime chain (as it ran in 3N–3Q)

```
PASS 1 (never imported by Pass 2; boundary = JSON file)
  player_tracker_mo.run_mo.run  [M]
    └─ track_manager.MultiPlayerTrackManager  [M]
         ├─ player_tracker.detector.PlayerDetector  [M]   (YOLO person, classes=[0])
         ├─ player_tracker.appearance.AppearanceEncoder  [M]   (yolo26n-reid.onnx via ONNX Runtime; HSV colour; EasyOCR)
         ├─ player_tracker.camera / motion / association / identity / identity_manager / geometry  [M] (DOCUMENTED usage)
         ├─ identity_bank_mo.IdentityBankV2 [M], reentry [M], team_track [M] → football_analysis.possession_detector.TeamClassifier [S-3N] (!! cross-pass coupling, F14)
         └─ track.PlayerTrack.as_record()  [M]
    ⇒ {base}_mo_tracking_results.json   (VERIFIED_OUTPUT: GOLD 88246efe, PC 40fa0df1)

PASS 2 (event_engine_v1.analyze_target_player [S])
  json.load(tracking JSON)
  → player_tracker_mo.tracklet_builder [M] → selected_player_resolver [M] (+ stitcher [M], AppearanceEncoder [M])
  → offline_identity_enrichment [M]
  → event_adapter [S]  ── imports new_tracker_snapshots.NEW_TO_LEGACY_STATE [M] and football_analysis.adapter [M]
  → football_analysis.adapter.TrackingAdapter.save [M]
  → football_analysis.engine.FootballAnalysisEngine.run_analysis_pass [M]   (own YOLO: yolov8s.pt [M] persons+ball; yolov8x.pt [M] heavy ball)
        → event_detector [S-3N/patch] → ball_tracker [S] (+7 ball_* [M], ball_static_decoy [S]), interaction_detector [M],
          possession_detector [S-3N/patch], possession_segmentation [M], event_classifier [M], receiver_eligibility [S-HV], controlled_duel [S],
          ball_interaction_credibility [S], visual_pitch_calibration [M], release_geometry [M]
        → event_qualifier [S-3N/patch] → event_arbitration [S-3N/patch] → event_candidates [S], shot_candidate [S-3N/patch], controlled_duel [S]
        → confidence [S-3N/patch] → clip_tagger [M] (raw event dicts)
  → actor_attribution [S] → possession_events [S-HV] → player_stats [S-3N/patch]
  → event_clip_exporter [S]
```

**Blocking-path summary.**

- Pass 1 cannot run at all: every module on its path is missing.
- Pass 2 cannot import:
  - `analyze_target_player` fails at import on `player_tracker.appearance`, `player_tracker_mo.*`, `football_analysis.adapter`, `engine` and `colour`;
  - `event_adapter` fails at import on `new_tracker_snapshots` and `football_analysis.adapter`.

## 2. Module table

| S/M | Module | Imports (surviving-visible) | Imported by (surviving) | Runtime phase | Data consumed | Data produced | Role | Missing-file class |
|---|---|---|---|---|---|---|---|---|
| M | `event_engine_v1/__init__.py` | UNKNOWN (missing) | — | — | — | — | — | TEST_CRITICAL |
| S | `event_engine_v1/actor_attribution.py` | confidence | analyze_target_player | Pass 2 post | enriched frame_log, raw events, target_status, identity_conflict_counts | actor assessment per event | Pass-2 | (survives) |
| S | `event_engine_v1/analyze_target_player.py` | actor_attribution, player_stats, possession_events, adapter, colour, engine, appearance, event_adapter, offline_identity_enrichment, selected_player_resolver, tracklet_builder | — | Pass 2 orchestrator | mo tracking JSON, video | event_log.json, player_analysis.json, snapshots cache, raw events | Pass-2 | (survives) |
| S | `event_engine_v1/benchmark_fixture.py` | — | test_benchmark_fixture_guard, test_temporal_coverage_guard | tooling | fixture JSON, video file | provenance/temporal/evaluation verdicts | tools/tests | (survives) |
| S | `event_engine_v1/event_clip_exporter.py` | — | — | post | video, event log | per-event mp4 clips | Pass-2 output | (survives) |
| S | `event_engine_v1/player_stats.py` | — | analyze_target_player | post | event log | stats, stat confidence | Pass-2 | (survives) |
| S | `event_engine_v1/possession_events.py` | — | analyze_target_player | post | stat-eligible events | POSSESSION_WON/LOST | Pass-2 | (survives) |
| S | `event_engine_v1/test_benchmark_fixture_guard.py` | benchmark_fixture | — | tests | — | — | tests | (survives) |
| S | `event_engine_v1/test_temporal_coverage_guard.py` | benchmark_fixture | — | tests | — | — | tests | (survives) |
| M | `football_analysis/__init__.py` | UNKNOWN (missing) | — | Pass 2 | — | — | Pass-2 | UNKNOWN |
| M | `football_analysis/adapter.py` | UNKNOWN (missing) | analyze_target_player, event_adapter, pipeline | Pass 2 input | snapshot list / JSON | TrackingSnapshot, TrackingAdapter.save/load | Pass-2 | RUNTIME_CRITICAL |
| M | `football_analysis/ball_background_motion.py` | UNKNOWN (missing) | ball_tracker | Pass 2 | — | — | Pass-2 | RUNTIME_CRITICAL |
| M | `football_analysis/ball_confidence.py` | UNKNOWN (missing) | ball_tracker | Pass 2 | — | — | Pass-2 | RUNTIME_CRITICAL |
| M | `football_analysis/ball_debug_overlay.py` | UNKNOWN (missing) | — | Pass 2 | — | — | Pass-2 | OPTIONAL_TOOLING |
| M | `football_analysis/ball_gating.py` | UNKNOWN (missing) | ball_tracker | Pass 2 | — | — | Pass-2 | RUNTIME_CRITICAL |
| S | `football_analysis/ball_interaction_credibility.py` | interaction_detector, possession_detector | controlled_duel, event_detector | Pass 2 | — | — | Pass-2 | (survives) |
| M | `football_analysis/ball_motion.py` | UNKNOWN (missing) | ball_tracker | Pass 2 | — | — | Pass-2 | RUNTIME_CRITICAL |
| M | `football_analysis/ball_roi_search.py` | UNKNOWN (missing) | ball_tracker | Pass 2 | — | — | Pass-2 | RUNTIME_CRITICAL |
| S | `football_analysis/ball_static_decoy.py` | geometry, visual_pitch_calibration | ball_tracker | Pass 2 | — | — | Pass-2 | (survives) |
| M | `football_analysis/ball_temporal_bridge.py` | UNKNOWN (missing) | ball_tracker | Pass 2 | — | — | Pass-2 | RUNTIME_CRITICAL |
| M | `football_analysis/ball_track_state.py` | UNKNOWN (missing) | ball_tracker | Pass 2 | — | — | Pass-2 | RUNTIME_CRITICAL |
| S | `football_analysis/ball_tracker.py` | ball_background_motion, ball_confidence, ball_gating, ball_motion, ball_roi_search, ball_static_decoy, ball_temporal_bridge, ball_track_state, geometry | event_detector | Pass 2 | — | — | Pass-2 | (survives) |
| M | `football_analysis/clip_tagger.py` | UNKNOWN (missing) | — | Pass 2 | — | — | Pass-2 | RUNTIME_CRITICAL |
| M | `football_analysis/colour.py` | UNKNOWN (missing) | analyze_target_player, possession_detector | Pass 2 | — | — | Pass-2 | RUNTIME_CRITICAL |
| S | `football_analysis/confidence.py` | — | actor_attribution | Pass 2 | — | — | Pass-2 | (survives) |
| S | `football_analysis/controlled_duel.py` | ball_interaction_credibility, event_candidates | event_arbitration, event_detector | Pass 2 | — | — | Pass-2 | (survives) |
| M | `football_analysis/debug_overlay.py` | UNKNOWN (missing) | — | Pass 2 | — | — | Pass-2 | OPTIONAL_TOOLING |
| M | `football_analysis/engine.py` | UNKNOWN (missing) | analyze_target_player | Pass 2 driver | video, snapshots, tracked colour hist | raw events JSON (clip_tagger shape) | Pass-2 | RUNTIME_CRITICAL |
| S | `football_analysis/event_arbitration.py` | controlled_duel, event_candidates, event_classifier, geometry, pitch_context, release_geometry, shot_candidate | event_qualifier | Pass 2 | — | — | Pass-2 | (survives) |
| S | `football_analysis/event_candidates.py` | geometry, goalkeeper_context, pitch_context, release_geometry | controlled_duel, event_arbitration | Pass 2 | — | — | Pass-2 | (survives) |
| M | `football_analysis/event_classifier.py` | UNKNOWN (missing) | event_arbitration, event_detector, shot_candidate | Pass 2 | — | — | Pass-2 | RUNTIME_CRITICAL |
| S | `football_analysis/event_detector.py` | ball_interaction_credibility, ball_tracker, controlled_duel, event_classifier, geometry, interaction_detector, possession_detector, possession_segmentation, receiver_eligibility, release_geometry, visual_pitch_calibration | — | Pass 2 | — | — | Pass-2 | (survives) |
| S | `football_analysis/event_qualifier.py` | event_arbitration, shot_candidate | — | Pass 2 | — | — | Pass-2 | (survives) |
| M | `football_analysis/geometry.py` | UNKNOWN (missing) | ball_static_decoy, ball_tracker, event_arbitration, event_candidates, event_detector, opponent_detection, possession_detector, receiver_eligibility, shot_candidate | Pass 2 | — | — | Pass-2 | RUNTIME_CRITICAL |
| M | `football_analysis/goalkeeper_context.py` | UNKNOWN (missing) | event_candidates, shot_candidate | Pass 2 | — | — | Pass-2 | RUNTIME_CRITICAL |
| M | `football_analysis/interaction_detector.py` | UNKNOWN (missing) | ball_interaction_credibility, event_detector | Pass 2 | — | — | Pass-2 | RUNTIME_CRITICAL |
| S | `football_analysis/opponent_detection.py` | geometry | receiver_eligibility | Pass 2 | — | — | Pass-2 | (survives) |
| M | `football_analysis/pitch_context.py` | UNKNOWN (missing) | event_arbitration, event_candidates, shot_candidate | Pass 2 | — | — | Pass-2 | RUNTIME_CRITICAL |
| S | `football_analysis/possession_detector.py` | colour, geometry | ball_interaction_credibility, event_detector | Pass 2 | — | — | Pass-2 | (survives) |
| M | `football_analysis/possession_segmentation.py` | UNKNOWN (missing) | event_detector | Pass 2 | — | — | Pass-2 | RUNTIME_CRITICAL |
| S | `football_analysis/receiver_eligibility.py` | geometry, opponent_detection | event_detector | Pass 2 | — | — | Pass-2 | (survives) |
| M | `football_analysis/release_geometry.py` | UNKNOWN (missing) | event_arbitration, event_candidates, event_detector, shot_candidate | Pass 2 | — | — | Pass-2 | RUNTIME_CRITICAL |
| M | `football_analysis/run.py` | UNKNOWN (missing) | — | Pass 2 | — | — | Pass-2 | OPTIONAL_TOOLING |
| S | `football_analysis/shot_candidate.py` | event_classifier, geometry, goalkeeper_context, pitch_context, release_geometry | event_arbitration, event_qualifier | Pass 2 | — | — | Pass-2 | (survives) |
| M | `football_analysis/visual_calibration_overlay.py` | UNKNOWN (missing) | — | Pass 2 | — | — | Pass-2 | OPTIONAL_TOOLING |
| M | `football_analysis/visual_pitch_calibration.py` | UNKNOWN (missing) | ball_static_decoy, event_detector | Pass 2 | — | — | Pass-2 | RUNTIME_CRITICAL |
| M | `football_analysis/visual_pitch_timeline.py` | UNKNOWN (missing) | — | Pass 2 | — | — | Pass-2 | UNKNOWN |
| M | `new_tracker_snapshots.py` | UNKNOWN (missing) | event_adapter, pipeline | Pass 1→2 boundary | raw tracking JSON | NEW_TO_LEGACY_STATE, snapshots | Pass-1/2 adapter | RUNTIME_CRITICAL |
| M | `player_tracker/__init__.py` | UNKNOWN (missing) | — | Pass 1 | — | — | Pass-1 | RUNTIME_CRITICAL |
| M | `player_tracker/appearance.py` | UNKNOWN (missing) | analyze_target_player, pipeline | Pass 1 + Pass 2 pre | frame, box | embedding, colour hist, OCR number | Pass-1 | RUNTIME_CRITICAL |
| M | `player_tracker/association.py` | UNKNOWN (missing) | — | Pass 1 | — | — | Pass-1 | RUNTIME_CRITICAL |
| M | `player_tracker/camera.py` | UNKNOWN (missing) | — | Pass 1 | — | — | Pass-1 | RUNTIME_CRITICAL |
| M | `player_tracker/debug_viz.py` | UNKNOWN (missing) | — | Pass 1 | — | — | Pass-1 | OPTIONAL_TOOLING |
| M | `player_tracker/detector.py` | UNKNOWN (missing) | — | Pass 1 | frame | person boxes | Pass-1 | RUNTIME_CRITICAL |
| M | `player_tracker/geometry.py` | UNKNOWN (missing) | — | Pass 1 | — | — | Pass-1 | RUNTIME_CRITICAL |
| M | `player_tracker/identity.py` | UNKNOWN (missing) | — | Pass 1 | — | — | Pass-1 | RUNTIME_CRITICAL |
| M | `player_tracker/identity_manager.py` | UNKNOWN (missing) | — | Pass 1 | — | — | Pass-1 | RUNTIME_CRITICAL |
| M | `player_tracker/motion.py` | UNKNOWN (missing) | — | Pass 1 | — | — | Pass-1 | RUNTIME_CRITICAL |
| M | `player_tracker/run.py` | UNKNOWN (missing) | — | Pass 1 | — | — | Pass-1 | LEGACY_OR_UNUSED |
| M | `player_tracker/tracker.py` | UNKNOWN (missing) | — | Pass 1 | — | — | Pass-1 | LEGACY_OR_UNUSED |
| M | `player_tracker_mo/__init__.py` | UNKNOWN (missing) | — | Pass 1 | — | — | Pass-1 | RUNTIME_CRITICAL |
| M | `player_tracker_mo/association_mo.py` | UNKNOWN (missing) | — | Pass 1 | — | — | Pass-1 | RUNTIME_CRITICAL |
| M | `player_tracker_mo/composite_identity_score.py` | UNKNOWN (missing) | — | Pass 1 | — | — | Pass-1 | UNKNOWN |
| S | `player_tracker_mo/event_adapter.py` | adapter, new_tracker_snapshots | analyze_target_player, pipeline | Pass 1→2 boundary | enriched frame_log, fps | List[TrackingSnapshot] | Pass-1/2 adapter | (survives) |
| M | `player_tracker_mo/identity_bank_mo.py` | UNKNOWN (missing) | — | Pass 1 | — | — | Pass-1 | RUNTIME_CRITICAL |
| M | `player_tracker_mo/identity_descriptor_v2.py` | UNKNOWN (missing) | — | Pass 1 | — | — | Pass-1 | RUNTIME_CRITICAL |
| M | `player_tracker_mo/new_clip_test_runner.py` | UNKNOWN (missing) | — | Pass 1 | — | — | Pass-1 | OPTIONAL_TOOLING |
| M | `player_tracker_mo/offline_identity_enrichment.py` | UNKNOWN (missing) | analyze_target_player, pipeline | Pass 2 pre | frame_log, resolver result | enriched frame_log | Pass-1/2 | RUNTIME_CRITICAL |
| S | `player_tracker_mo/pipeline.py` | adapter, new_tracker_snapshots, appearance, event_adapter, offline_identity_enrichment, review_package_generator, run_mo, selected_player_resolver, stitcher, track_manager, tracking_health, tracklet_builder, tracklet_quality | — | Pass 1 (+offline) | video, seed, config | artifact package (tracking JSON, tracklets, stitches, trajectory, review queue, health) | Pass-1 tools | (survives) |
| M | `player_tracker_mo/reentry.py` | UNKNOWN (missing) | — | Pass 1 | — | — | Pass-1 | RUNTIME_CRITICAL |
| M | `player_tracker_mo/review_package_generator.py` | UNKNOWN (missing) | pipeline | Pass 1 | — | — | Pass-1 | OPTIONAL_TOOLING |
| M | `player_tracker_mo/run_mo.py` | UNKNOWN (missing) | pipeline | Pass 1 | video, seed frame/box, config | {base}_mo_tracking_results.json, summary, debug mp4 | Pass-1 | RUNTIME_CRITICAL |
| M | `player_tracker_mo/selected_player_resolver.py` | UNKNOWN (missing) | analyze_target_player, pipeline | Pass 2 pre (offline identity) | frame_log, tracklets, video, encoder | boundary resolutions, identity states | Pass-1/2 | RUNTIME_CRITICAL |
| M | `player_tracker_mo/stitcher.py` | UNKNOWN (missing) | pipeline | offline identity | TrackFragments | stitch decisions | Pass-1/2 | RUNTIME_CRITICAL |
| M | `player_tracker_mo/team_track.py` | UNKNOWN (missing) | — | Pass 1 | — | — | Pass-1 | RUNTIME_CRITICAL |
| M | `player_tracker_mo/track.py` | UNKNOWN (missing) | — | Pass 1 | — | PlayerTrack.as_record() (29 fields) | Pass-1 | RUNTIME_CRITICAL |
| M | `player_tracker_mo/track_manager.py` | UNKNOWN (missing) | pipeline | Pass 1 | frames, detections, embeddings | per-frame records (debug_log), health metrics | Pass-1 | RUNTIME_CRITICAL |
| S | `player_tracker_mo/tracking_health.py` | tracklet | pipeline | Pass 1 diagnostics | run summary, tracklets, stitch provenance, resolver result, snapshots | tracking_health dict | tools | (survives) |
| M | `player_tracker_mo/tracklet.py` | UNKNOWN (missing) | tracking_health | Pass 1 offline / Pass 2 pre | — | Tracklet | Pass-1/2 | RUNTIME_CRITICAL |
| M | `player_tracker_mo/tracklet_builder.py` | UNKNOWN (missing) | analyze_target_player, pipeline | Pass 2 pre (offline identity) | frame_log | tracklets | Pass-1/2 | RUNTIME_CRITICAL |
| M | `player_tracker_mo/tracklet_quality.py` | UNKNOWN (missing) | pipeline | Pass 1 | — | — | Pass-1 | OPTIONAL_TOOLING |

## 3. Notes on edges not visible in surviving imports

- `football_analysis/engine.py` (missing) is the caller of `event_detector`, `event_qualifier` (constructor injection, per CURRENT_ARCHITECTURE.md), `confidence`, `clip_tagger` and probably `opponent_detection` (the local high-sensitivity pass). These are DOCUMENTED and INFERRED; the bytes do not survive.
- `player_tracker_mo/track_manager.py` (missing) uses `player_tracker` detector, appearance, camera and matcher objects (DOCUMENTED, TRACKING_PIPELINE_INTEGRATION_MAP §1). It also uses `identity_manager.register_confirmed/register_unconfirmed` (SEG_D §3), `identity_bank_mo.IdentityBankV2` (IPR:143) and `team_track` → `football_analysis.possession_detector.TeamClassifier` (TASK3M). **That last edge is the historical Pass-1 → Pass-2-code coupling.**
- `selected_player_resolver` (missing) imports `stitcher.evaluate_stitch` and `tracklet_builder.populate_appearance_samples` (DOCUMENTED, integration map §1).
- `validation/lib/run_engine_pass.py` and `validation/lib/reid_backward.py` were never in the manifest. They are referenced only (`analyze_target_player.py:22-24,59-61`; earlier_frame_reid_report.md) and are not needed by the 3Q runtime chain.

## 4. Counts (82 manifest entries)

| Class (missing only) | Count |
|---|---|
| LEGACY_OR_UNUSED | 2 |
| OPTIONAL_TOOLING | 8 |
| RUNTIME_CRITICAL | 46 |
| TEST_CRITICAL | 1 |
| UNKNOWN | 3 |

Surviving: 22 manifest entries (14 EXACT, 1 HASH_VERIFIED, 7 PATCHABLE), plus 2 post-3N files (controlled_duel EXACT, receiver_eligibility HASH_VERIFIED).

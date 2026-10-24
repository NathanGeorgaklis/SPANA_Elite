# SPANA recovered originals (2026-10-24)

Only files that **survived as saved bytes** are packaged here. Nothing was rewritten from memory.
Each engine file was matched by SHA-256 against the records written at the time (the Task 3N 82-file manifest, plus the 3P and 3Q integrity checks).
The rebuild I made earlier (`spana_engine_surviving_files_3Q_state.zip`) was **not used as a source**.

Searched: every file and every ZIP (including nested ZIPs) in the session outputs folder, the Claude project docs, and Nathan's Mac home folder (excluding Library).
Nothing new was found on the Mac. The uploads folder is empty.

## 1. `engine_current_3Q_state_exact/`: 15 files, byte-exact, current version

| File | Bytes | sha256 | Found in |
|---|---|---|---|
| `event_engine_v1/actor_attribution.py` | 9,317 | ac2003830a66 | task3f_actor_confidence.zip → event_engine_v1/actor_attribution.py |
| `event_engine_v1/analyze_target_player.py` | 20,278 | 1f15a21644e8 | task3f_actor_confidence.zip → event_engine_v1/analyze_target_player.py |
| `event_engine_v1/benchmark_fixture.py` | 18,303 | 0e5f7d1d5ac6 | task3h_duel_regrounding.zip → benchmark_fixture.py |
| `event_engine_v1/event_clip_exporter.py` | 4,269 | c574d6e071e4 | task3a_pass_completed_01_deliverables.zip → event_clip_exporter.py |
| `event_engine_v1/test_benchmark_fixture_guard.py` | 2,893 | 36e9000d1339 | task3d_benchmark_guard_and_duel_run.zip → task3d_deliverables/test_benchmark_fixture_guard.py |
| `event_engine_v1/test_temporal_coverage_guard.py` | 4,399 | 99452da59d23 | task3h_duel_regrounding.zip → test_temporal_coverage_guard.py |
| `football_analysis/ball_interaction_credibility.py` | 13,446 | be2027eb5109 | task3g_ball_validity_and_temporal_guard.zip → football_analysis/ball_interaction_credibility.py |
| `football_analysis/ball_static_decoy.py` | 15,794 | d98f2224c757 | task3e_static_decoy_fix.zip → football_analysis/ball_static_decoy.py |
| `football_analysis/ball_tracker.py` | 25,763 | c819415d0571 | task3g_ball_validity_and_temporal_guard.zip → football_analysis/ball_tracker.py |
| `football_analysis/controlled_duel.py` | 28,587 | c5705cefbcf9 | task3q_controlled_duel.zip → task3q/code_snapshot/controlled_duel.py |
| `football_analysis/event_candidates.py` | 44,909 | bc91abef45cb | <loose> → event_candidates.py |
| `football_analysis/opponent_detection.py` | 18,906 | 1d1198c1b805 | task3l_other_player_context.zip → ../football_analysis/opponent_detection.py |
| `player_tracker_mo/event_adapter.py` | 7,951 | 418d54219ff1 | <loose> → event_adapter.py |
| `player_tracker_mo/pipeline.py` | 21,266 | 98aa28b60a58 | <loose> → pipeline.py |
| `player_tracker_mo/tracking_health.py` | 4,527 | 67dc6b3f823a | <loose> → tracking_health.py |

## 2. `engine_3Q_hash_verified_from_original_patch/`: 2 files

These were produced by applying the saved original 3P/3Q diff files to the 3N originals. They are **not** recovered copies.
They are included because their bytes match hashes recorded at the time: `receiver_eligibility.py` (3P and 3Q INTEGRITY_CHECK) and `possession_events.py` (3Q CODE_AT_RUN). That makes them byte-identical to what I built.

| File | Bytes | sha256 | Source |
|---|---|---|---|
| `event_engine_v1/possession_events.py` | 6,704 | f0ac5dc97496 | rebuilt from original 3Q patch |
| `football_analysis/receiver_eligibility.py` | 8,755 | f1ccd2bc94b8 | rebuilt from original 3P/3Q patch |

## 3. `engine_3N_versions_of_files_changed_in_3P_3Q/`: 8 files, byte-exact but older (pre-3P)

| File | Bytes | sha256 | Found in |
|---|---|---|---|
| `event_engine_v1/player_stats.py` | 7,873 | 30321eb2a34b | task3f_actor_confidence.zip → event_engine_v1/player_stats.py |
| `event_engine_v1/possession_events.py` | 5,541 | fcb3357347b0 | task3f_actor_confidence.zip → event_engine_v1/possession_events.py |
| `football_analysis/confidence.py` | 3,103 | 41a94f8e628d | task3q_controlled_duel.zip → task3q/pre_task3q_originals/confidence.py |
| `football_analysis/event_arbitration.py` | 21,245 | 86ebeb58a12d | <loose> → event_arbitration.py |
| `football_analysis/event_detector.py` | 18,023 | 81b72abd5c13 | task3g_ball_validity_and_temporal_guard.zip → football_analysis/event_detector.py |
| `football_analysis/event_qualifier.py` | 2,051 | 40b8ea44dbb3 | task3q_controlled_duel.zip → task3q/pre_task3q_originals/event_qualifier.py |
| `football_analysis/possession_detector.py` | 30,260 | 6790b47f1a95 | task3l_other_player_context.zip → ../football_analysis/possession_detector.py |
| `football_analysis/shot_candidate.py` | 65,651 | 94f58fc72a3b | task3p_receiver_eligibility.zip → task3p/pre_task3p_originals/shot_candidate.py |

## 4. `original_patches_and_integrity_records/`

These are the original 3P/3Q diffs, the 3N manifest, and the 3P/3Q integrity checks, as saved at the time.

## 5. `all_code_found_by_source/`

Every `.py`, `.sh` and `.patch` file found anywhere: test files, diagnostic scripts and dataset tools.
Each one sits under the ZIP or folder it came from, with that folder structure preserved. Older engine versions appear here (for example `loose_in_outputs/engine_updated.py`, dated Sep 22), but they are **not** the current engine.

## 6. `task3r/`

Contents:
- Nathan's labels v1 and v2, exactly as received;
- the receipt;
- the reconciliation docs;
- the label sheet;
- the sealed reviewer-2 file and its SEAL.

The clip label pack (`task3r_label_pack.zip`, 22 MB) is already in Downloads and was left out.

## 7. `INVENTORY_all_files_found.csv`

Lists every file found, with its path and size. This includes the full contents of every ZIP.

---

# MISSING: not recovered

## A. Current (3Q) versions not recovered: 7 files

Only the older 3N version of each survives (section 3).
The 3Q version could be regenerated by applying the original patches in section 4. The 3Q integrity record says the 3Q state equals "3N originals + TASK3P_CODE_DIFF.patch" for possession_detector and shot_candidate. However, **no hash of the 3Q version was recorded** for any of the seven, so a regenerated copy cannot be proven identical.

- `event_engine_v1/player_stats.py`
- `football_analysis/confidence.py`
- `football_analysis/event_arbitration.py`
- `football_analysis/event_detector.py`
- `football_analysis/event_qualifier.py`
- `football_analysis/possession_detector.py`
- `football_analysis/shot_candidate.py`

## B. Not found at all: 60 of the 82 manifest entries

**player_tracker/** (12)

- `player_tracker/__init__.py`
- `player_tracker/appearance.py`
- `player_tracker/association.py`
- `player_tracker/camera.py`
- `player_tracker/debug_viz.py`
- `player_tracker/detector.py`
- `player_tracker/geometry.py`
- `player_tracker/identity.py`
- `player_tracker/identity_manager.py`
- `player_tracker/motion.py`
- `player_tracker/run.py`
- `player_tracker/tracker.py`

**player_tracker_mo/** (18)

- `player_tracker_mo/__init__.py`
- `player_tracker_mo/association_mo.py`
- `player_tracker_mo/composite_identity_score.py`
- `player_tracker_mo/identity_bank_mo.py`
- `player_tracker_mo/identity_descriptor_v2.py`
- `player_tracker_mo/new_clip_test_runner.py`
- `player_tracker_mo/offline_identity_enrichment.py`
- `player_tracker_mo/reentry.py`
- `player_tracker_mo/review_package_generator.py`
- `player_tracker_mo/run_mo.py`
- `player_tracker_mo/selected_player_resolver.py`
- `player_tracker_mo/stitcher.py`
- `player_tracker_mo/team_track.py`
- `player_tracker_mo/track.py`
- `player_tracker_mo/track_manager.py`
- `player_tracker_mo/tracklet.py`
- `player_tracker_mo/tracklet_builder.py`
- `player_tracker_mo/tracklet_quality.py`

**football_analysis/** (25)

- `football_analysis/__init__.py`
- `football_analysis/adapter.py`
- `football_analysis/ball_background_motion.py`
- `football_analysis/ball_confidence.py`
- `football_analysis/ball_debug_overlay.py`
- `football_analysis/ball_gating.py`
- `football_analysis/ball_motion.py`
- `football_analysis/ball_roi_search.py`
- `football_analysis/ball_temporal_bridge.py`
- `football_analysis/ball_track_state.py`
- `football_analysis/clip_tagger.py`
- `football_analysis/colour.py`
- `football_analysis/debug_overlay.py`
- `football_analysis/engine.py`
- `football_analysis/event_classifier.py`
- `football_analysis/geometry.py`
- `football_analysis/goalkeeper_context.py`
- `football_analysis/interaction_detector.py`
- `football_analysis/pitch_context.py`
- `football_analysis/possession_segmentation.py`
- `football_analysis/release_geometry.py`
- `football_analysis/run.py`
- `football_analysis/visual_calibration_overlay.py`
- `football_analysis/visual_pitch_calibration.py`
- `football_analysis/visual_pitch_timeline.py`

**event_engine_v1/** (1)

- `event_engine_v1/__init__.py (0-byte file per manifest)`

**other** (4)

- `new_tracker_snapshots.py`
- `yolo26n-reid.onnx`
- `yolov8s.pt`
- `yolov8x.pt`

## C. Task 3R working files

These existed only in the wiped session workspace. They were not found in any saved location, and the transcript that recorded them has since been compacted:

- `TASK3R_SCORING_RULES.md`
- `score_benchmark.py`
- `run_benchmark.py`
- `LABELING_INSTRUCTIONS.md` (the copy inside `task3r_label_pack.zip` survives)
- `make_sheet.py`
- `cutscan.py`
- `seed_candidates.py`
- `code_state.sh`
- the clip-cutting script

The raw extracts R01–R10 and their extraction scripts and logs survive on the Mac, in `~/Downloads/_spana_task3r_work/`.

## What this means

Pass 1 (the player tracker: `run_mo`, `track_manager`, `selected_player_resolver` and others) is not recoverable.
Neither is most of the Pass-2 core (`engine`, `adapter`, `interaction_detector`, `possession_segmentation`, `event_classifier` and others), nor the YOLO and re-ID models.
Any replacement code would be new code. It would have to be re-validated from scratch, not treated as the version that produced the 3A–3Q results.

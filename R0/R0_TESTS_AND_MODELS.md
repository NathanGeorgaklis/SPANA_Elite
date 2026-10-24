# R0_TESTS_AND_MODELS.md

SPANA engine v2-rebuild, R0. Read-only. Written 2026-10-24.

Sections: **1** tracker test inventory (R0.6) · **2** engine test inventory (context) · **3** tracking data and fixture inventory (R0.7) · **4** model recovery (R0.8).

"Runnable now" was checked in a scratch copy of the surviving tree:
- Python 3 with pytest;
- no models and no source video;
- no engine file modified.

Raw output is in `R0_TEST_COLLECTION_RAW.txt`.

---

## 1. Surviving tracker-related tests (R0.6)

A test counts as "tracker-related" if it exercises Pass-1 code, the Pass-1 output contract, the seed/fixture format, or per-frame identity consumption.

| # | Path (in `SPANA_recovered_originals.zip`) | sha256 (12) | What it checks | Dependencies | Runnable now | Expected inputs | Expected outputs | Behaviour or historical internals? |
|---|---|---|---|---|---|---|---|---|
| T1 | `all_code_found_by_source/loose_in_outputs/test_pipeline_integration.py` (16 tests) | 3914d7f37d18 | `pipeline._general_stitch`, `_build_review_queue` (UNRESOLVED never dropped), `event_adapter` (downgrade, pass-through, no-selected → 1 LOST, leave-frame → carry box + LOST), `DEFAULT_CANDIDATE_CONFIGURATION` pin, health `wrong_id_flags == 0`, full offline chain on a single run → AUTO_RESOLVED, and `run_tracking_pipeline` INTEGRATED, LEGACY and zero-detection with Stage 1 stubbed | **missing:** run_mo, track_manager, tracklet_builder, stitcher, selected_player_resolver, offline_identity_enrichment, review_package_generator, tracklet, new_tracker_snapshots, football_analysis.adapter. **Surviving:** pipeline, event_adapter, tracking_health | **NO** (`ModuleNotFoundError: player_tracker_mo.run_mo`) | hand-built frame_log matching `PlayerTrack.as_record()` (without the 5 Stage-B position fields), fake video path | listed in `R0_TRACKER_INTERFACE_MAP.md` §6 | **Mostly behaviour and contract.** It is the best surviving statement of adapter and degradation semantics. Two tests pin internals: `_general_stitch` and the config dict. Not in the 3N manifest, so its version is unproven (REPO_ROOT = `parents[2]`). |
| T2 | `task3f_actor_confidence.zip/test_actor_attribution.py` | 405e4cd6a2f2 | Actor windows, per-frame trust rules, resolver factor, stats eligibility | `event_engine_v1.actor_attribution` (exact), `analyze_target_player` (exact, but its top-level imports need missing `player_tracker.appearance` and others), `player_stats` | **NO** (`No module named 'player_tracker'`, through the analyze_target_player import) | synthetic frame_log plus raw events | actor verdicts and stats | **Behaviour.** It consumes Pass-1 fields: state, identity_state, position_source, occlusion_status, overlap_contests, identity_confidence. **Version note:** it is a 3F-era file and later 3P/3Q changes to player_stats could affect it. |
| T3 | `event_engine_v1/test_benchmark_fixture_guard.py` (EXACT) | 36e9000d1339 | Seed fixture create/save/load round-trip and the provenance guard (hash mismatch → BENCHMARK_PROVENANCE_MISMATCH) | `benchmark_fixture` (exact) plus the real `dual_lost_01.mov` at a `/root/.claude/uploads/…` path | **NO** (`FileNotFoundError` on the video; the file is not on the Mac) | real video | MATCH / MISMATCH | **Behaviour.** Documents the seed format. |
| T4 | `event_engine_v1/test_temporal_coverage_guard.py` (EXACT) | 99452da59d23 | Tracking range from the mo summary (`start_frame`, `processed_frames`, `fps`) versus the label window | `benchmark_fixture` | **NO** (video path at collection) | fixtures | VALID / INVALID | **Behaviour.** Consumes `summary.start_frame/fps/processed_frames`. |
| T5 | `player_tracker_mo/tests/` (155 tests incl. `test_position_provenance.py` 7, `test_track_freshness.py` 7) | — | Online tracker Fixes A–E and position provenance | — | **MISSING.** Referenced by SEG_D_STAGE_B §3; never in a manifest. | — | — | — |

**Tracker tests surviving: 4 (T1–T4). Runnable now: 0.** T3 and T4 become runnable only with the original `duel_lost_01.mov`, which is not on the Mac (the earlier inventory also found it absent), or by repointing them at another video. That would be a test change and needs approval.

## 2. Other surviving tests (event engine; listed for R3/R4)

| Path | sha256 (12) | Area | Runnable now | Blocking missing module |
|---|---|---|---|---|
| `task3q_controlled_duel.zip/task3q/test_controlled_duel.py` | 368b114bb8a4 | DUEL path B (19 tests per status) | NO | ball_track_state, event_classifier |
| `task3p_receiver_eligibility.zip/task3p/test_receiver_eligibility.py` | 8265437c9dd2 | receiver rule (13) | NO | geometry |
| `loose/test_event_arbitration.py` | ef1dec5ac95f | arbitration | NO | ball_track_state, pitch_context, goalkeeper_context |
| `loose/test_pass_evidence_fix.py` | b2de2762e47d | PASS evidence | NO | ball_track_state |
| `task3g…/test_ball_physical_validity.py` | a680741370bf | ball credibility gate | NO | possession_segmentation, geometry |
| `loose/test_ball_static_decoy.py`, `task3e…/test_ball_static_decoy_grass_mask_fix.py`, `…/test_pitch_region.py` | a0f33d76149a, ca0016ba7ca4, ffe18bc99d3f | static decoy | NO (not attempted in full; ball_static_decoy imports missing geometry and visual_pitch_calibration) | geometry |
| `loose/test_opponent_detection.py`, `task3l…/test_other_player_context.py` | b28f9b09aa3f, 3547af4523d2 | other-player context | NO (opponent_detection imports missing geometry) | geometry |
| `SPANA_V2…/test_ab_event_matcher.py` | 92deccfe8410 | dataset tooling | not attempted | — |
| `loose/milestone2/test_touch_detector.py`, `…/test_touch_episode.py` | 4fed0e8d78e0, 8c74cea343b1 | Milestone-2 prototypes, superseded | not attempted | — |

The historical full-regression count (180/180 after 3Q, per the project status doc) **cannot be reproduced now**.

---

## 3. Tracking data and fixture inventory (R0.7)

Holdout content (`videoplayback (2).mp4`, ESFC v BCFC) was **not opened, decoded or hashed** in R0. Its SHA-256 below comes from TASK3J_VIDEO_INVENTORY.md.

| # | Item | Location (verified today) | Hash check | Contents useful for tracking | Class |
|---|---|---|---|---|---|
| X1 | **GOLD_DUEL_01 extract** `GOLD_DUEL_01_syo_mar_3036_3048.mp4` | Mac `~/Downloads/_spana_task3j_work/gold/` | **164541…3fd5e9: MATCH today** (also GOLD_DUEL_01_HASHES.sha256) | 1280×720, 30 fps, 360 frames; lossless re-encode, framemd5-verified against source | **VALID_REFERENCE** (video) |
| X2 | GOLD fixture `GOLD_DUEL_01_BENCHMARK_FIXTURE.json` | task3j zip | 7929bb28…: MATCH (recorded in 3K/3N guards) | seed_frame 180, seed_bbox [635.3, 304.1, 757.8, 438.3]; seed confirmed by raw YOLOv8x detection conf 0.939 | **VALID_REFERENCE** (seed) |
| X3 | GOLD human label plus sign-off | task3j zip; `specs/GOLD_DUEL_01_HUMAN_SIGNOFF.json`; Mac task3j_work | 05fe67c6… and 423ad2ba…: MATCH | **Human-signed** (Nathan, 2026-10-19). Target = SOL #23 and opponent = MAR #4, both HIGH. #23 legible at listed times. Overlap frames 262–274 named as the identity-risk moment. No same-kit player within 10 m. **This is clip-level identity truth, not per-frame boxes.** | **VALID_REFERENCE** (identity at clip and moment level) |
| X4 | **GOLD frozen 3N Pass-1 JSON** | task3n zips | **88246efe…: MATCH** `PASS1_FROZEN_SHA256.txt` | Full 29-field frame_log, 179 frames, all tracks | **VALID_REFERENCE for the output contract.** **DIAGNOSTIC_ONLY as tracking truth** (it is model output) |
| X5 | GOLD frozen 3N snapshots and Pass-2 outputs | task3n zips | listed in `OUTPUT_HASHES_BEFORE_COMPARISON.txt` | adapter output (`{"snapshots": […]}`, 180) | VALID_REFERENCE (adapter contract) |
| X6 | `TASK3N_GOLD_PASS1_TRACE_250_285.csv`, `TASK3M_TARGET_IDENTITY_FRAME_TRACE.csv` (36 frames, 50 columns) | task3n / task3m zips | not separately recorded | per-frame physical-player labels (#23 / #4 / merged / hidden) around the collision | **DIAGNOSTIC_ONLY.** Visual labels by Claude, not human-signed. A human could confirm them to promote them to R2 truth for frames 250–285 |
| X7 | `gold_target_chain.json`, `gold_seed_detections.json` | task3j zip | — | naive nearest-box chain (switched #23→#4 at 266) and seed detections | DIAGNOSTIC_ONLY (negative evidence of naive chaining) |
| X8 | GOLD Task 3M replay captures (`replay/`, `replay_3lflag_on/`) | task3m zip | determinism 109/110 vs the scored run | raw detections with confidences, score matrices, bank admissions | DIAGNOSTIC_ONLY |
| X9 | GOLD stale (pre-3L) Task 3K Pass-1 JSON | task3k zip (`PASS1_OUTPUT_HASH.txt`) | not re-hashed in R0 | historical swap run | DIAGNOSTIC_ONLY. Negative evidence for F1 |
| X10 | pass_completed_01 fixture `PASS_COMPLETED_01_BENCHMARK_FIXTURE.json` | task3g zip | 8cdbf0aa… (no independent historical record found) | seed_frame 28, seed_bbox [191.6, 850.4, 302.3, 1094.3]. **Circular:** "seed_bbox = first selected-player box recorded by the Pass-1 run". `tracking_artifact_sha256` aba3464b… = the **misregistered stale** JSON | DIAGNOSTIC_ONLY for tracking (seed provenance is circular); valid for event truth only |
| X11 | **pass_completed_01 frozen 3N Pass-1 JSON** | task3n zips | **40fa0df1…: MATCH** | 364 frames, selected ACTIVE 100%, single id T001, no contacts | **VALID_REFERENCE for the output contract.** DIAGNOSTIC_ONLY as tracking truth |
| X12 | pass_completed_01 source video `f5bd8ce0-pass_completed_01.mov` (a2e45703…, 22,656,639 B) | **NOT FOUND** in ~/Documents or ~/Downloads (by name or size, including zip listings) | — | — | **UNKNOWN** (cannot re-run Pass 1) |
| X13 | duel_lost_01 (9c94e481…) | video not found | — | — | **REJECTED** (REJECTED_BENCHMARK_REGISTRY: seed was a bystander; contest in progress at frame 0). Negative evidence: front/back view change gave 65 false "self-inconsistency" frames (TASK3M) |
| X14 | src2_clip_003 (24922747…) | not found locally | — | — | **REJECTED** (REJECTED_IDENTITY_AMBIGUITY; includes a same-kit crossing at 4.0–4.3 s). Negative evidence |
| X15 | shot_close_range_01 | not found | — | — | **REJECTED** (BENCHMARK_PROVENANCE_MISMATCH: seed on empty grass). Negative evidence |
| X16 | blind_r2_001, blind_r2_009 (640×360, 30 fps) | sources not located; `~/Downloads/blind_r2_009_tracker_trace.csv` (75 KB) survives; `ablation_ladder_009_summary.json` in specs | — | ablation metrics; boundary T001→T064 (UNRESOLVED) documented | DIAGNOSTIC_ONLY (single-source regression clips; "must never be represented as proof of general tracking quality", DEFAULT_TRACKING_PIPELINE_SPEC §5) |
| X17 | SEG_A–SEG_E (Milestone 1/2) episode JSONs `seg_*_episodes*.json` | `~/Downloads/` | — | Milestone-2 touch-episode outputs, not tracking JSONs | UNKNOWN for tracking (not examined in depth) |
| X18 | Task 3R raw extracts R01–R10 | `~/Downloads/_spana_task3r_work/` | `extract_log*.txt` framemd5 match=YES | 10 windows from videoplayback.mp4, videoplayback (1).mp4 and youtube_game_1/2/3 | **UNKNOWN / RESERVED.** Task 3R is frozen. Must not be used for tracker development or tuning. R2 use only with your explicit approval |
| X19 | Source broadcasts `videoplayback.mp4` (09aecc9b…, SYO v MAR, multi-camera) and `videoplayback (1).mp4` (e8bd4d5f…, MCFC v MIT, single low camera, floodlit) | `~/Downloads/` | registered in `source_videos.json` (TASK3J) | full matches. Candidate pool for new R2 tracking clips, excluding 3R windows, GENERAL_BENCHMARK_V2 windows and the GOLD window | **VALID_REFERENCE (source)** once new windows are human-labelled |
| X20 | `youtube_game_1/2/3.mp4` | `~/Downloads/` | hashes in `_spana_task3r_work/source_sha256.txt` | used by Task 3R | UNKNOWN (provenance documentation not reviewed in R0) |
| X21 | **`videoplayback (2).mp4` (b9f34c4a…, ESFC v BCFC)** | `~/Downloads/` | not touched | — | **HOLDOUT. Do not inspect.** |

**Tracker fixtures available now (usable for R1/R2 without re-labelling): 1.** GOLD_DUEL_01 is the only clip with a verified source video, a non-circular verified seed, human-signed clip-level identity and a frozen reference output (X1–X5). **Output-contract references: 2** (X4, X11).

**Gap.** There are **no human per-frame tracking annotations** (target boxes or identity per frame) anywhere in the surviving evidence. R2 needs new human labels (see `R0_TRACKER_ACCEPTANCE_TARGETS.md`).

---

## 4. Model recovery (R0.8)

R0 downloaded nothing. The hashes come from the 3N manifest (`3N_CODE_BASELINE_SHA256.txt`, byte-identical in the archive and the task3n zips, sha256 6eeebc12…). All three were re-verified "OK" after the 3N runs (`CODE_BASELINE_RECHECK.txt`).

| Model | File survives? | Historical SHA-256 | Source | Version | Licence / source | Role (evidence) | Exact reconstruction possible? |
|---|---|---|---|---|---|---|---|
| `yolov8s.pt` | **NO** (not on the Mac: Documents/Downloads searched by name; earlier inventory searched home) | `1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a` | UNKNOWN (no download URL recorded). Plausibly the Ultralytics release asset (INFERRED from name and `ultralytics.YOLO()` loading) | UNKNOWN | UNKNOWN (Ultralytics models are AGPL-3.0 if so; INFERRED) | Pass-2 engine full-frame person+ball detector, classes [0,32], conf 0.15 (DOCUMENTED). Possibly the Pass-1 PlayerDetector (CONFLICT: YOLOv8s vs yolo26n-reid; see behaviour evidence §1) | **UNKNOWN.** A web search for the hash found no public record. Possible later test: download the official asset in R1 (with approval) and compare hashes |
| `yolov8x.pt` | **NO** | `3df4ada6b4dad6d657868f2fdf7faecfb34dcfccf3a25c4b82079064718524c8` | UNKNOWN (as above) | UNKNOWN | UNKNOWN | Heavy ball re-detection model (older engine code `YOLO("yolov8x.pt")`); seed confirmation in Task 3J (DOCUMENTED) | **UNKNOWN** (same test possible) |
| `yolo26n-reid.onnx` | **NO** | `8529c383197ae4c468eda535d1b165f8b4162cf17bf5fbcff49c7cb6455bc0bb` | **UNKNOWN. No document records its origin.** A web search for the exact filename and hash found no match. "YOLO26" is a public Ultralytics model family, but no public "-reid" ONNX artefact by this name was found | UNKNOWN | UNKNOWN | DOCUMENTED as the person-ReID embedding behind `AppearanceEncoder.embed()` (earlier_frame_reid_report.md:46). pipeline.py:107-111 names it `TRACKING_MODEL_VERSION`. One report table lists it as the "Detector" (**conflict recorded**). Loaded via ONNX Runtime 1.25.0, CPU | **NO** (origin unknown). **Do not substitute in R0** |
| EasyOCR weights (jersey OCR) | NO (not in the manifest) | not recorded | EasyOCR (DOCUMENTED) | UNKNOWN | UNKNOWN | `AppearanceEncoder` OCR. `use_ocr=False` in Pass 2. PC.json `number_candidate` shows spurious values | UNKNOWN |

**Tracker models exactly recovered: 0 of 3.**

Implication for R1, which needs your approval:
- R1 needs at least a person detector, and very probably an appearance embedding.
- Whichever models R1 uses will be **NEW** dependencies.
- They must be pinned by hash and recorded in the tracking JSON.
- They must never be described as the historical models unless their hashes match the values above.

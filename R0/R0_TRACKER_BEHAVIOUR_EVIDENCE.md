# R0_TRACKER_BEHAVIOUR_EVIDENCE.md

SPANA engine v2-rebuild, R0. Read-only. Written 2026-10-24.

**What this is.** Evidence about how the *historical* selected-player tracker (`player_tracker_mo` plus the shared `player_tracker` components) behaved. None of its source survives. Each item below is labelled with one of:

- **VERIFIED_OUTPUT**: the item is visible in a frozen, hash-verified output.
- **VERIFIED_CODE**: the item is visible in a surviving caller or replay script, never in tracker internals.
- **DOCUMENTED**: the item is stated in a historical report.
- **INFERRED**: the item is reasoned from evidence but not stated anywhere.
- **UNKNOWN**: there is no evidence for the item.

**What this is not.** This is not a specification for R1. Historical behaviour includes known defects; see `R0_TRACKER_FAILURE_CATALOGUE.md`. R1 treats this document as design constraints, not as code to copy.

Primary sources:
- **TRACKING_ABLATION_FINAL_REPORT.md** (the ablation report)
- **TASK3M_MOTION_ASSOCIATION_AUDIT.md**, **TASK3M_REID_APPEARANCE_AUDIT.md**, **TASK3M_OFFLINE_RESOLVER_AUDIT.md**, **TASK3M_FIX_LAYER_OPTIONS.md** (the 3M audits)
- **TASK3N_GOLD_TRACKING_COMPARISON.md** (3N)
- **SEG_D_STALE_TRACK_WRONG_ACTOR_INVESTIGATION.md** and **SEG_D_STAGE_B_TRACKER_POSITION_FIX.md** (SEG_D)
- **TASK3C_TRACKING_DIAGNOSIS_REPORT.md** (3C)
- **CURRENT_ARCHITECTURE.md**
- `instrumented_pass1_replay.py` and its `replay/association_capture.json` (the IPR capture)
- The two frozen 3N tracking JSONs.

---

## 1. Detector model

| Item | Evidence | Finding |
|---|---|---|
| Class used by Pass 1 | VERIFIED_CODE: `player_tracker.detector.PlayerDetector().detect_persons(frame)` in 6 surviving diagnostic scripts; `mgr.detector` in IPR:69 | YOLO person detector wrapper |
| Predict call | VERIFIED_CODE: IPR:70 | `d.model.predict(frame, classes=[0], conf=d.conf, imgsz=d.imgsz, verbose=False)`, so person class only |
| Return | VERIFIED_CODE: diagnose_raw_detections.py:28-33 | list of `(x1, y1, x2, y2[, conf])` |
| Which weights file Pass 1 used | **CONFLICT** | **UNKNOWN.** CURRENT_ARCHITECTURE.md calls it a "YOLOv8 person detection wrapper". 90_MINUTE_MATCH_ARCHITECTURE.md says "PlayerDetector (YOLOv8s)". END_TO_END_TRACKING_PIPELINE_REPORT.md table lists "Detector (yolo26n-reid.onnx)". pipeline.py:107-111 says only that the run logs "Loading yolo26n-reid.onnx". |
| `d.conf` / `d.imgsz` values | UNKNOWN | never recorded |
| Detection confidences seen | VERIFIED_OUTPUT (IPR capture) | 0.47–0.91 on GOLD around the collision; partial bodies 0.47–0.59 |
| Pass-2 detector (separate pass) | DOCUMENTED (inspect_missing_defender.py:10) and VERIFIED_CODE (analyze_target_player.py:105) | `yolov8s.pt`, classes [0, 32] (person, ball), conf 0.15; a heavy `yolov8x.pt` factory for ball re-detection |

## 2. Detection confidence handling

| Item | Evidence | Finding |
|---|---|---|
| Per-track field | VERIFIED_OUTPUT (GOLD, all 1,713 records) | `detection_confidence` is **not the YOLO score**. It is quantised by position source: CONFIRMED_DETECTION → 1.0 (1,167 records; 1 exception at 0.0), AMBIGUOUS_DETECTION / WEAK_SPATIAL_RECOVERY → 0.5, PREDICTED → 0.0. The raw YOLO confidence is never written to the frame_log |
| Use in association | UNKNOWN | not among the six score components (§4) |

## 3. Appearance / ReID

| Item | Evidence | Finding |
|---|---|---|
| Encoder | VERIFIED_CODE: `player_tracker.appearance.AppearanceEncoder`, `.embed(frame, box)` (IPR:43, 109) | ONNX ReID embedding |
| Model file | DOCUMENTED: earlier_frame_reid_report.md:46 | `yolo26n-reid.onnx` is the "real football person-ReID embedding". Runtime log: "Loading yolo26n-reid.onnx for ONNX Runtime inference… ONNX Runtime 1.25.0 with CPUExecutionProvider" (task3b/3c logs) |
| Embedding dimension | UNKNOWN | — |
| Similarity | DOCUMENTED | cosine (`.cosine()`) |
| Colour cue in association | DOCUMENTED: TECHNICAL_DEBT_AUDIT.md; TASK3M_REID §4 | HSV shirt histogram (`player_tracker.appearance.shirt_color_hist`). Note this is distinct from Pass 2's LAB KitDescriptor. The HSV cue cannot separate two blues under contamination; the LAB cue can |
| Jersey OCR | DOCUMENTED (CURRENT_ARCHITECTURE.md:104) + VERIFIED_OUTPUT | EasyOCR. PC.json `number_candidate` holds spurious values ('3331', '33315'); GOLD holds all None |
| Identity bank (Fix C) | VERIFIED_CODE: `IdentityBankV2.consider(embedding, quality, appearance_sim, is_ambiguous_this_frame) -> bool`, `.n_trusted_samples` (IPR:60, 143-147) | Quality-gated, multi-sample. In GOLD it refused every sample from 262 to 273 because appearance was below the admission floor, so it stayed clean and enabled recovery (TASK3M_REID §6). Admission floor value is UNKNOWN |
| Representative embedding | VERIFIED_CODE: `mgr._representative_embedding(t)` (IPR:104) | The bank exposes a single representative vector. How it is aggregated is UNKNOWN |
| ReID behaviour on AMBIGUOUS frames | DOCUMENTED: TASK3M_REID §5 | Logged `appearance_confidence` and `reid_confidence` are halved, which lowers `identity_confidence`. This does **not** change the association decision |

## 4. Association

| Item | Evidence | Finding |
|---|---|---|
| Functions (module attributes of track_manager) | VERIFIED_CODE (IPR:80-83) | `compute_identity_gate_matrix(tracks, detections)`, `build_score_matrix(tracks, detections) -> (scores, scored)`, `assign(tracks, detections, scores)`, `detect_overlap_contests(tracks)` |
| Assignment record | VERIFIED_CODE (IPR:120-122) | `track_id, detection_index, score, ambiguous, contested_with, reject_reason` |
| Score components | VERIFIED_OUTPUT (IPR capture, `as_log_dict`) | `appearance_similarity, motion_similarity, iou, colour_similarity, number_similarity, size_consistency, final_score, result` |
| **Score weights** | **VERIFIED_OUTPUT.** R0 fitted least squares over 116 ACCEPTED candidates; maximum absolute error 0.0007. This matches DOCUMENTED TASK3M_REID §5 | appearance **0.40**, motion **0.20**, IoU-to-last-box **0.10**, colour **0.15**, size **0.05**. The `number_similarity` weight is **UNKNOWN**: the component was 0 in every capture. 0.10 would be INFERRED only if the weights sum to 1 |
| Rejection reasons | VERIFIED_OUTPUT | `outside motion gate (Npx > Npx)` (649 of 812); `no strong identity signal (appearance=…, colour=…)` (45); `appearance below floor (N < N)` (2) |
| Assignment method | DOCUMENTED | Hungarian assignment over the "scorable" (non-PROVISIONAL, non-DORMANT) tracks |
| Ambiguity margin | DOCUMENTED (TASK3M_MOTION §2, item 5) | 0.08: if the best candidate does not beat the next-best by more than this, the frame is AMBIGUOUS |
| Appearance floor | DOCUMENTED (TASK3M_REID §2) | `MIN_APPEARANCE_FLOOR = 0.30` |
| Identity-gate fast path | DOCUMENTED (TASK3M_MOTION §2) | If an ACTIVE track has exactly one colour-plausible detection inside its gate and within 45 px of its last confirmed centre, `require_identity_gate = False`: the box is accepted on position and size without the appearance check. **This is a known defect** |
| Spot checks | VERIFIED_CODE (IPR:90) | `mgr.cfg.spot_check_interval`. Every Nth frame the gate is forced on. N is UNKNOWN |

## 5. Motion continuity

| Item | Evidence | Finding |
|---|---|---|
| Model | DOCUMENTED (CURRENT_ARCHITECTURE.md) | 4-state constant-velocity Kalman (`MotionPredictor`). Internal `motion.position` (IPR:54) |
| Gate radius | DOCUMENTED (SEG_D §3; TASK3C) + VERIFIED_OUTPUT (gate 125 = 90 + 35×1 in IPR capture) | `gate = min(max_gate, base_gate + gate_growth × frames_since_confirmed)`, with base 90 px, growth 35 px per frame and max 350 px. **Pixel-based and not scale-normalised.** The gate grows while a track is AMBIGUOUS, which is a defect |
| Ambiguous frames update motion | DOCUMENTED (SEG_D §3; TASK3M) | AMBIGUOUS calls `motion.update(center(det))` toward the contested detection but does **not** update `current_box`. After a wrong confirmation this lets the filter adopt the occluder's velocity |
| Camera compensation | DOCUMENTED | `CameraMotionEstimator` (optical-flow pan estimate plus hard-cut detection) outputs `camera_shift` and `is_cut`. No hard cut was ever observed in a surviving output |

## 6. Identity gates and promotion

| Item | Evidence | Finding |
|---|---|---|
| PROVISIONAL birth (Fix B) | DOCUMENTED (ablation report §1) | New detections start PROVISIONAL and must accumulate evidence before promotion |
| Promotion rule | DOCUMENTED (TASK3M_MOTION; TASK3N Part 6) | `_try_promote_provisional` requires at least 60% team-vote agreement. The minimum frame count is UNKNOWN. On GOLD, #4's track was promoted at frame 195 (15 frames after birth) once the 3L team classifier voted TEAM_B |
| PROVISIONAL termination | DOCUMENTED (TASK3M) | Terminated on a "provisional miss streak 2" (T003 terminated after 2 starved frames) |
| **PROVISIONAL tracks invisible to association** | DOCUMENTED (TASK3M Part 11, item 1) | Excluded from the Hungarian matrix, the assignment contest and `detect_overlap_contests`. **Known defect** |
| Team classifier coupling | DOCUMENTED (TASK3M; TASK3N) | Pass 1 imports `football_analysis.possession_detector.TeamClassifier` through `team_track.TrackAwareTeamClassifier`. A Pass-2 change (3L) changed Pass-1 tracking. **R1 must not share mutable logic with Pass 2** |
| Spawn-duplicate guard (Fix A) | DOCUMENTED + VERIFIED_OUTPUT (`spawn_guard_stats`) | Leftover detections are classified SAFE_NEW_TRACK / POSSIBLE_DUPLICATE / AMBIGUOUS_OWNERSHIP before a new track spawns |

## 7. Occlusion logic

| Item | Evidence | Finding |
|---|---|---|
| OCCLUDED state | VERIFIED_OUTPUT | Combination OCCLUDED + PARTIAL + PREDICTED. GOLD 266–268: no detection, box held at #23's last confirmed position |
| OCCLUDED → TEMPORARILY_LOST | DOCUMENTED (TASK3C, citing TrackManagerConfig) | `occlusion_to_lost_frames = 10` |
| TEMPORARILY_LOST / OUT_OF_FRAME | VERIFIED_OUTPUT | TEMPORARILY_LOST + FULL + PREDICTED (308 records); OUT_OF_FRAME + FULL + PREDICTED (12). Both are members of the DORMANT set (pipeline.py:162-164; IPR:132) |
| Dormant termination | DOCUMENTED (SEG_D) | `terminate_after_dormant_frames` (value UNKNOWN); lifecycle pruning (Fix E) |

## 8. Ambiguity

| Item | Evidence | Finding |
|---|---|---|
| AMBIGUOUS state | VERIFIED_OUTPUT | AMBIGUOUS + AMBIGUOUS + AMBIGUOUS_DETECTION. Box not updated (`current_box` held); `observed_position_box` = contested detection |
| Trigger | DOCUMENTED | best-versus-second margin of 0.08 or less |
| Ageing | DOCUMENTED (SEG_D §3) | **AMBIGUOUS never ages out** (no decay or termination path). A track stayed AMBIGUOUS for 77 frames with a frozen box and then relocked 234.8 px away. **Known defect** |
| Pass-2 mapping | VERIFIED_OUTPUT | AMBIGUOUS → legacy OCCLUDED |

## 9. LOST state

| Item | Evidence | Finding |
|---|---|---|
| Online "LOST" | VERIFIED_OUTPUT | There is no `LOST` online state. The nearest equivalents are TEMPORARILY_LOST and OUT_OF_FRAME |
| Legacy LOST | VERIFIED_CODE (event_adapter.py:100-109, 124-133, 135) | Emitted when no selected record exists, or for any state missing from the table |
| Old single-target tracker | DOCUMENTED (CURRENT_ARCHITECTURE.md) | LOST rescans the full frame every 5th frame forever and reacquires after 3 consistent hits (`reacquire_frames = 3`), with no long-absence gate. This applies to `player_tracker/tracker.py`, not the mo tracker |

## 10. Held-bbox behaviour

| Item | Evidence | Finding |
|---|---|---|
| `bbox` = `identity.current_box` | DOCUMENTED (SEG_D §3) | Written only by `register_confirmed()`; frozen during AMBIGUOUS, OCCLUDED and dormant states |
| `observed_position_box` | VERIFIED_OUTPUT + DOCUMENTED (SEG_D Stage B) | Additive field: the latest contested or predicted position. It differs from `bbox` in 442 of 545 non-confirmed GOLD records |
| Held box reaching Pass 2 | VERIFIED_CODE (event_adapter.py:141) | The adapter passes `bbox` (the held box), never `observed_position_box`. Its only trust signal is state → OCCLUDED |
| Consequence | DOCUMENTED (TASK3N; TASK3O) | Pass 2 searched around a stale held box and admitted the target's own re-emerging body as a separate TEAMMATE receiver, producing a false PASS |

## 11. Re-entry and reacquisition

| Item | Evidence | Finding |
|---|---|---|
| Fix D global margin-aware re-entry | DOCUMENTED (ablation report §1) | All dormant candidates compete. The winner must beat the next-best by `REENTRY_AMBIGUITY_MARGIN = 0.08`; otherwise a `ReentryConflict` is logged and appears in `identity_conflict_counts` |
| REIDENTIFYING state | VERIFIED_OUTPUT + DOCUMENTED | Seen once in GOLD and 10 times in PC, each as REIDENTIFYING + FULL + CONFIRMED_DETECTION. The live trace (LIVE_IDENTITY_TRUST_DOWNGRADE_VALIDATION §4) shows it mapped to legacy REACQUIRING at the moment `selected_player` moved to a new track id (blind_r2_009 T001 → T064). The internal decision rule is UNKNOWN |
| Weak spatial recovery | VERIFIED_OUTPUT + DOCUMENTED | ACTIVE + WEAK_SPATIAL_RECOVERY + WEAK_SPATIAL_RECOVERY (103 GOLD records), from `_try_spatial_recovery`. Recovery by position, without identity confirmation |
| Conflict visibility | DOCUMENTED (SEG_D §3) | The conflict log is populated only inside the DORMANT re-entry loop. AMBIGUOUS→relock and fast-path swaps never log a conflict |
| Reacquisition after GOLD collision | VERIFIED_OUTPUT (3N) | OCCLUDED 266–268, then AMBIGUOUS 269–271, then confirmed on #23 at 272 (visually checked by Claude in 3N, not by a person) |

## 12. Same-team ambiguity

- **DOCUMENTED.** IDENTITY_FEASIBILITY_GATE.md formalises that a contest between same-kit players caps identity evidence at WEAK unless a number is legible.
- **No surviving selected-player tracker output contains a same-kit crossing that was examined.** The GOLD human label notes "No same-kit player is within 10 m of the target between the seed and the label end."
- Tracker behaviour in a same-team crossing is therefore **UNKNOWN** from frozen outputs.

## 13. Opponent overlap

| Item | Evidence | Finding |
|---|---|---|
| GOLD 262–274 (different kits) | VERIFIED_OUTPUT (3N JSON) + DOCUMENTED (3M, 3N) | Current code: no swap. Stale pre-3L code: swap 266–273 through the fast path plus PROVISIONAL invisibility |
| Merged detection | DOCUMENTED (TASK3M_MOTION §1) | At 266 the detector returned one box for the overlapped pair (#23's kit-pixel share 0.086). #23 went undetected at 267–268 |

## 14. Seed initialisation

| Item | Evidence | Finding |
|---|---|---|
| API | VERIFIED_CODE (IPR:162-164) | Seek to `seed_frame`, read it, `initialize(frame0, seed_box, jersey_number=None)` |
| Seed track state | VERIFIED_OUTPUT | T001 ACTIVE from frame 1, `detection_confidence 1.0`, team TEAM_A in both clips |
| Seed verification practice | DOCUMENTED (TASK3D; TASK3J; TASK3C) | The seed box must overlap a raw detection at the seed frame (GOLD: YOLOv8x conf 0.939). The rejected shot_close_range_01 seed was 168.7 px from any detection |
| Seed taken from output | DOCUMENTED (PASS_COMPLETED_01 fixture notes) | `seed_bbox` = the first selected-player box recorded by an earlier Pass-1 run. This is circular provenance; see `R0_TESTS_AND_MODELS.md` §3 |

## 15. Track IDs

| Item | Evidence | Finding |
|---|---|---|
| Format | VERIFIED_OUTPUT | `T` + 3 digits, sequential by birth; the seed is T001 |
| `identity_id` / `online_track_id` | VERIFIED_OUTPUT | Equal to `track_id` in every surviving record. The offline stitcher can map ids (`identity_id_map`), but the online frame_log never showed a remap |
| Selected id change | DOCUMENTED (TASK3M_OFFLINE_RESOLVER) | The resolver creates a boundary only when the selected `online_track_id` changes. In both frozen runs it never changed (T001 throughout) |

## 16. Resolver behaviour

| Item | Evidence | Finding |
|---|---|---|
| Scope | DOCUMENTED (resolver lines 198-201, quoted in TASK3M) | Boundary-only. "Silent drift WITHIN a run is a separate, track_manager-level problem" |
| Safety rule | DOCUMENTED | An override of the online choice can never reach AUTO_RESOLVED. The position stream is never substituted |
| Live boundary behaviour | DOCUMENTED (LIVE_IDENTITY_TRUST_DOWNGRADE_VALIDATION.md) | blind_r2_009 boundary T001 → T064 at frames 89–90: UNRESOLVED_IDENTITY, zero candidates, 29/29 frames downgraded to OCCLUDED |
| Frozen-3N runs | VERIFIED_OUTPUT (both runs have one online id) | `target_status` RESOLVED on both clips |

## 17. Temporal smoothing

- Team labels: position-bucket temporal smoothing inside TeamClassifier (DOCUMENTED, CURRENT_ARCHITECTURE.md).
- No temporal smoothing of the selected box or state is documented. **UNKNOWN.**

## 18. Frame-level identity evidence available historically

The per-frame identity evidence shown below is all VERIFIED_OUTPUT. The 29 track fields are listed in `R0_TRACKER_INTERFACE_MAP.md` §4.

The three-way state on the selected track (`state`, `occlusion_status`, `position_source`) moved together. `identity_confidence` was the only scalar that changed during the GOLD swap: 0.36–0.61 during the swap, against a typical 0.70–0.76.

The following were **absent** historically (DOCUMENTED, TASK3M Part 8):
- any record of *which* other track was the competitor or occluder for the selected track. `overlap_contests` covered non-PROVISIONAL tracks only;
- any per-frame "physical identity verified" signal, as distinct from "the matcher accepted a box".

## 19. Runtime and performance

- **DOCUMENTED** (ablation report §2). Fix B made the tracker about 4.8× slower: 3,300–3,460 s for 419 frames on a 2-CPU sandbox, compared with about 690 s without B. Per-frame time grew with live track count, which reached about 150.
- **VERIFIED_OUTPUT.** GOLD took 912 s for 179 frames; PC took 3,185 s for 364 frames.
- **INFERRED.** The cost grows with the number of tracks. R1 should bound the population it tracks.

## 20. Things historically claimed but never independently verified

- ACTIVE at the blind_r2_009 contested moment was "the tracker's own label, not independent ground truth" (ablation report §3).
- The GOLD per-frame "physical player in selected box" labels in TASK3N/TASK3M traces are **Claude visual checks**, not human-signed annotations.
- Test counts conflict across reports:

| Source | Test count |
|---|---|
| SEG_D Stage B | 155 tracker tests |
| END_TO_END_TRACKING_PIPELINE_REPORT | 148 (132 + 16) |
| 3Q status | 180/180 (engine plus receiver) |

These are different suites at different times. None of them survive in runnable form.

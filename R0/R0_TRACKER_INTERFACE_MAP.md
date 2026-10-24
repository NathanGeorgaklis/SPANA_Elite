# R0_TRACKER_INTERFACE_MAP.md

SPANA engine v2-rebuild — R0 (read-only recovery). Written 2026-10-24.

This document records **what downstream code expects from Pass 1 (the selected-player tracker)**, recovered only from surviving bytes and frozen outputs. It does not describe how the old tracker worked internally; that is in `R0_TRACKER_BEHAVIOUR_EVIDENCE.md`.

## How to read the evidence

Every finding carries: **EVIDENCE LEVEL · SOURCE FILE:LINE · EXACT OBSERVED USAGE**.

Source-file abbreviations. All are hash-checked as described in `R0_RECOVERY_LEDGER.csv`:

| Abbrev. | File | Status |
|---|---|---|
| `EA` | `player_tracker_mo/event_adapter.py` | EXACT_RECOVERED (418d5421) |
| `PL` | `player_tracker_mo/pipeline.py` | EXACT_RECOVERED (98aa28b6) |
| `TH` | `player_tracker_mo/tracking_health.py` | EXACT_RECOVERED (67dc6b3f) |
| `ATP` | `event_engine_v1/analyze_target_player.py` | EXACT_RECOVERED (1f15a216) |
| `AA` | `event_engine_v1/actor_attribution.py` | EXACT_RECOVERED (ac200383) |
| `BF` | `event_engine_v1/benchmark_fixture.py` | EXACT_RECOVERED (0e5f7d1d) |
| `CD` | `football_analysis/controlled_duel.py` | EXACT_RECOVERED (c5705cef) |
| `RE` | `football_analysis/receiver_eligibility.py` | HASH_VERIFIED (f1ccd2bc) |
| `ED3N`, `PD3N`, `CF3N`, `EARB3N` | `event_detector.py`, `possession_detector.py`, `confidence.py`, `event_arbitration.py` | 3N bytes, byte-exact (3Q bytes = patch-derived only) |
| `TPI` | `loose_in_outputs/test_pipeline_integration.py` | survives; not in the manifest, so its version is unproven |
| `RFP1` | `task3n/run_fresh_pass1.py` | Task 3N driver, from the task zip |
| `IPR` | `task3m/instrumented_pass1_replay.py` | Task 3M read-only replay, from the task zip |
| `GOLD.json` | `task3n/gold/pass1/GOLD_DUEL_01_syo_mar_3036_3048_mo_tracking_results.json` | **VERIFIED_OUTPUT**: sha256 88246efe…1dc5 equals `PASS1_FROZEN_SHA256.txt` |
| `PC.json` | `task3n/pass_completed_01/pass1/f5bd8ce0-pass_completed_01_mo_tracking_results.json` | **VERIFIED_OUTPUT**: sha256 40fa0df1… equals `PASS1_FROZEN_SHA256.txt` |
| `GOLD.snap`, `PC.snap` | `task3n/<clip>/pass2/*_integrated_snapshots.json` | Task 3N frozen Pass-2 adapter output |

---

## 0. The single most important finding

**The historical Pass-1 → Pass-2 boundary is a file, not a function call.**

- Pass 1 (`player_tracker_mo.run_mo.run`) writes `{video_basename}_mo_tracking_results.json` with the top-level keys `summary`, `frame_log` and `final_tracks`.
  - VERIFIED_CODE: `PL:247-251`, `base = os.path.splitext(os.path.basename(video_path))[0]` and `tracking_json_path = out_root / f"{base}_mo_tracking_results.json"`.
  - VERIFIED_OUTPUT: GOLD.json and PC.json.
- Pass 2 (`analyze_target_player`) **only ever reads that JSON**. It never calls the tracker.
  - VERIFIED_CODE: `ATP:269-271`, `d = json.load(open(mo_tracking_json_path))`, `fps = d["summary"]["fps"]`, `frame_log = d["frame_log"]`.
  - DOCUMENTED: `ATP:13-16`, "produced separately, ahead of this module … a precomputed input to this orchestrator".

Two frozen, hash-verified examples of this file survive. **This JSON is therefore the strongest recoverable interface, and the natural contract for R1.** If R1 emits a schema-compatible JSON, every surviving or rebuilt Pass-2 consumer can be tested against it without coupling to tracker internals.

---

## 1. Pass-1 entry point and seed format

| # | Finding | Evidence | Source:line | Exact observed usage |
|---|---|---|---|---|
| 1.1 | Pass-1 function name and signature | VERIFIED_CODE | `PL:80`, `PL:242-244` | `from .run_mo import run as run_mo_pass`; `run_mo_pass(video_path, start_frame, init_box, jersey_number=jersey_number, max_frames=max_frames, out_dir=str(out_root), debug=debug_overlay, config_overrides=cfg, resume=resume)` |
| 1.2 | Defaults | VERIFIED_CODE | `TPI:268-269` (stub mirrors the real signature) | `def _fake(video_path, start_frame, init_box, jersey_number=None, max_frames=450, out_dir=None, debug=False, config_overrides=None, resume=True)` |
| 1.3 | How 3N actually invoked it | VERIFIED_CODE | `RFP1:14` | `run(fx.video_path, fx.seed_frame, list(fx.seed_bbox), jersey_number=None, out_dir=out_dir, debug=True)` |
| 1.4 | Return value is the summary dict | VERIFIED_CODE | `PL:252`, `PL:355-356`, `RFP1:15-16` | `fps = run_summary["fps"]`; `run_summary.get("clip_output")`; keys printed: video, start_frame, processed_frames, processing_time_s, selected_state_pct, identity_conflicts_logged |
| 1.5 | **Seed format** = video frame index plus one box `[x1, y1, x2, y2]` in pixels | VERIFIED_CODE | `BF:104-105` | `seed_frame: int`; `seed_bbox: list  # [x1, y1, x2, y2]` |
| 1.6 | Seed provenance label | VERIFIED_CODE | `PL:258` | `selected_player_initialization_provenance="manual_bbox_hint"` |
| 1.7 | Optional jersey hint exists but was not used in 3N | VERIFIED_CODE | `PL:216`, `RFP1:14` | `jersey_number: Optional[str] = None`; 3N passed `jersey_number=None` |
| 1.8 | Config overrides are a flat dict of five booleans | VERIFIED_CODE | `PL:121-127` | `DEFAULT_CANDIDATE_CONFIGURATION = dict(enable_spawn_duplicate_guard=True, enable_provisional_tracks=True, enable_identity_bank_v2=True, enable_reentry_v2=True, enable_lifecycle_pruning=True)` |
| 1.9 | Internal construction API | VERIFIED_CODE | `IPR:159-176` | `MultiPlayerTrackManager(config=TrackManagerConfig())`; `cap.set(CAP_PROP_POS_FRAMES, SEED_FRAME)`; `mgr.initialize(frame0, SEED_BOX, jersey_number=None)`; loop `mgr.step(frame)`; `mgr.debug_log` = frame_log |
| 1.10 | `resume=True` checkpointing | DOCUMENTED | TASK3N_FINAL_REPORT; `STALE_VS_FRESH_REGISTRATION.txt` | Resume landed 10 frames late on a .mov (IoU 1.000 at offset +10 from frame 171). **Rule carried forward: no resume.** |
| 1.11 | Seed → first frame | VERIFIED_CODE + VERIFIED_OUTPUT | `IPR:162-170`; GOLD.json | The seed frame itself is read for `initialize`; the first logged step is the next frame (`frame: 1`). frame_log frame *f* = video frame `start_frame + f` (also `ATP:284-307`; TASK3M: "engine frame = extract − 180") |

**Conflict recorded.** `EA:39-41` says frame_log is "1-indexed step() calls; frame 0 is the synthetic bootstrap record". `ATP:285-287` says frame_log frames are "0-based RELATIVE to … start_frame". Both are consistent with the data if "0" means the seed frame itself: frame_log starts at 1 in both JSONs, and the adapter synthesises the frame-0 snapshot. R1 must reproduce the data, not either prose description.

---

## 2. Tracking JSON: `summary` block

VERIFIED_OUTPUT (GOLD.json, PC.json). The keys are identical in both files:

| Key | Type | GOLD example | Consumed by (surviving) |
|---|---|---|---|
| `video` | str | path to the extract | — |
| `start_frame` | int | 180 | `ATP:308` (`start_frame_offset`), `BF:321` |
| `fps` | float | 30.0 (PC 56.502242152466366) | `ATP:270`, `PL:252`, `BF:320` |
| `processed_frames` | int | 179 | `BF:322` |
| `processing_time_s` | float | 912.0 | — |
| `selected_state_pct` | dict state→% | `{ACTIVE 93.9, OCCLUDED 1.7, AMBIGUOUS 4.5}` | — |
| `n_tracks_ever_created` | int | 35 | `TH:51` |
| `n_tracks_final_active` | int | 7 | — |
| `clip_output` | str/None | debug mp4 path | `PL:355-356` |
| `final_confirmed_count` | int | 15 | `TH:52` |
| `final_provisional_count` | int | 1 | `TH:53` |
| `spawn_guard_stats` | dict | `{SAFE_NEW_TRACK 26, POSSIBLE_DUPLICATE 64, AMBIGUOUS_OWNERSHIP 8}` | `TH:54` |
| `identity_conflicts_logged` | int | 0 | `TH:58` |
| `identity_conflict_counts` | dict `"Tid|Tid"`→int | PC: `{"T006|T017": 1}` | `ATP:325` → `AA:183-187` (splits on `"|"`) |
| `health_metrics` | dict (9 keys) | frames_processed, total_tracks_spawned, new_tracks_per_frame, tracks_per_100_frames, provisional_to_confirmed_ratio, median_track_lifetime_frames, duplicate_likelihood, live_association_eligible_dormant_count, pruned_from_live_association_count | `TH:24,49,53,57,72` |

Runtime-required by surviving Pass-2 code: **`fps`, `start_frame`, `identity_conflict_counts`** (`processed_frames` too, for the benchmark temporal guard). The rest is health or diagnostics.

---

## 3. Tracking JSON: `frame_log` (per-frame records)

VERIFIED_OUTPUT. It has one record per processed frame: 179 in GOLD, 364 in PC. Every record has exactly these 7 keys:

| Key | Type | Observed | Consumer |
|---|---|---|---|
| `frame` | int, starts at 1 | 1…179 | `AA:179-180` (`build_frame_index`), `EA:122`, `PL:155-157` |
| `is_cut` | bool | False in every frame of both clips | — (Pass 2 runs its own cut logic; UNKNOWN) |
| `camera_shift` | [dx, dy] float | GOLD frame 1 `[-12.1, 7.4]`; mostly `[0.0, -0.0]` | — |
| `n_detections` | int | 8, 20 | — |
| `n_live_tracks` | int | 1… | — |
| `overlap_contests` | list of [track_id, track_id] | GOLD 18 frames non-empty, e.g. `[["T001","T003"]]` | **`AA:110`**: `any(ids & set(pair) for pair in rec.get("overlap_contests", []) or [])` |
| `tracks` | list of track records (§4) | **all** live tracks, not only the target | `EA:88-91`, `AA:86-90`, `PL:155-157` |

**Conflict.** `TPI:73-74` builds synthetic frames with `"overlap_contests": 0` (an int). The frozen outputs always hold a list. `AA:110` tolerates both, via `or []`. R1 must emit a list.

---

## 4. Per-track record (29 fields): the historical `PlayerTrack.as_record()`

VERIFIED_OUTPUT: all 1,713 (GOLD) and 8,184 (PC) records have the same 29 keys and the types below. `final_tracks` uses the same record shape. The field list is DOCUMENTED as `PlayerTrack.as_record()` (TARGET_PLAYER_ADAPTER_COMPATIBILITY_REPORT.md §2; SEG_D_STAGE_B §2).

Legend: **R** = read by surviving runtime code; **R\*** = read by a missing module per documentation; **D** = diagnostic only.

| Field | Type | Observed values / range | Use | Evidence for use |
|---|---|---|---|---|
| `track_id` | str `T###` | T001… | R | `AA:109,196`; `EA` via record |
| `identity_id` | str | equals track_id in every record observed | D | VERIFIED_OUTPUT |
| `online_track_id` | str | equals track_id in every record observed | **R** | `AA:109,117,146,195`; `PL:157,162` (tracklet grouping); resolver splits runs on change of this (TASK3M_OFFLINE_RESOLVER_AUDIT) |
| `source` | str | `ONLINE` only | D | — |
| `state` | str enum | ACTIVE, PROVISIONAL, AMBIGUOUS, OCCLUDED, TEMPORARILY_LOST, REIDENTIFYING, OUT_OF_FRAME (TERMINATED appears only as pipeline default `PL:137`) | **R** | `EA:135`, `AA:101`, `PL:162-164` |
| `age` | int | frames | D | — |
| `last_seen_frame` | int | — | R\* | SEG_D (freshness gate in milestone tools) |
| `bbox` | [x1,y1,x2,y2] float, 1 dp | — | **R** | `EA:111,141` → snapshot bbox |
| `team` | str | TEAM_A, TEAM_B, UNKNOWN | D (Pass 1) | team vote drives promotion internally (TASK3N) |
| `team_confidence` | float | 0–1 | D | — |
| `number_candidate` | str/None | GOLD all None; PC `'26','10','13','24','8','6','1','3331','33315'` | D | OCR output; spurious values seen ('33315') |
| `selected_player` | bool | exactly 1 True per frame in both JSONs | **R** | `EA:89`, `AA:88`, resolver |
| `detection_confidence` | float | quantised: 1.0 CONFIRMED_DETECTION / 0.5 AMBIGUOUS_DETECTION or WEAK_SPATIAL_RECOVERY / 0.0 PREDICTED (GOLD, all records) — **not** the YOLO score | D | VERIFIED_OUTPUT |
| `motion_confidence` | float | — | D | — |
| `appearance_confidence` | float | — | D | halved on AMBIGUOUS frames (TASK3M_REID §5) |
| `reid_confidence` | float | — | D | recorded, not gated (`AA:118`; TASK3F §2) |
| `overall_track_confidence` | float | — | **R** | `EA:146` → `tracking_confidence` (verified 179/179 and 364/364 equal in the snapshots) |
| `visibility_confidence` | float | — | D | — |
| `occlusion_status` | str enum | NONE, PARTIAL, FULL, AMBIGUOUS, WEAK_SPATIAL_RECOVERY | **R** | `AA:107` (`!= "NONE"` → untrusted) |
| `merged_into` | None | always None | D | — |
| `position_confidence` | float | — | R\* | SEG_D freshness gate (milestone tool) |
| `track_continuity_confidence` | float | — | D | — |
| `identity_confidence` | float | — | **R** | `AA:112-114` (`< MEDIUM_THRESH 0.5` → untrusted) |
| `selected_player_confidence` | float | equals identity_confidence on the selected track in observed examples | D | — |
| `observed_position_box` | [x1,y1,x2,y2] | differs from `bbox` in 442/545 non-confirmed GOLD records | D (not consumed!) | SEG_D Stage B added it. **Not passed to Pass 2 by EA**. TASK3O §6: "Pass-1 context is dropped by the frozen event_adapter" |
| `position_is_confirmed` | bool | True iff position_source == CONFIRMED_DETECTION (observed) | D | — |
| `position_source` | str enum | CONFIRMED_DETECTION, AMBIGUOUS_DETECTION, PREDICTED, WEAK_SPATIAL_RECOVERY | **R** | `AA:105` (`!= "CONFIRMED_DETECTION"` → untrusted) |
| `frames_since_observed_position` | int | — | D | — |
| `frames_since_confirmed_identity` | int | — | D | — |

**Joint (state, occlusion_status, position_source) combinations observed** (GOLD, all tracks). This is VERIFIED_OUTPUT. It is the *only* recoverable statement of how the old states co-occurred:

| state | occlusion_status | position_source | count |
|---|---|---|---|
| ACTIVE | NONE | CONFIRMED_DETECTION | 813 |
| PROVISIONAL | NONE | CONFIRMED_DETECTION | 354 |
| TEMPORARILY_LOST | FULL | PREDICTED | 308 |
| ACTIVE | WEAK_SPATIAL_RECOVERY | WEAK_SPATIAL_RECOVERY | 103 |
| AMBIGUOUS | AMBIGUOUS | AMBIGUOUS_DETECTION | 63 |
| OCCLUDED | PARTIAL | PREDICTED | 59 |
| OUT_OF_FRAME | FULL | PREDICTED | 12 |
| REIDENTIFYING | FULL | CONFIRMED_DETECTION | 1 |

**Held vs observed box (already present historically).** On an OCCLUDED selected frame (GOLD frame 86):
- `bbox` = held last confirmed box `[874.6, 296.6, 918.3, 388.7]`;
- `observed_position_box` = predicted `[872.1, 295.0, 915.8, 387.1]`;
- `position_source = PREDICTED`.

VERIFIED_OUTPUT. The distinction exists in Pass 1 output, but **Pass 2 never sees it**.

---

## 5. Who reads what: field-level consumer map

### 5.1 `event_adapter.build_integrated_event_snapshots` (EXACT; requires missing `new_tracker_snapshots.py`)

| Finding | Evidence | Line | Usage |
|---|---|---|---|
| Import-time hard dependency | VERIFIED_CODE | `EA:58-61` | `from new_tracker_snapshots import NEW_TO_LEGACY_STATE`, at repo root via sys.path. **EA cannot be imported until a `new_tracker_snapshots` module exists.** |
| Input | VERIFIED_CODE | `EA:71-72` | `(enriched_frame_log: List[dict], fps: float, player_id="P1")` |
| Selected record = first track with `selected_player` truthy | VERIFIED_CODE | `EA:87-91` | `for tr in rec["tracks"]: if tr.get("selected_player"): return tr` |
| No selected record anywhere → single LOST bootstrap | VERIFIED_CODE | `EA:100-109` | bbox [0,0,0,0], conf 0.0, `"LOST"` |
| Frame-0 bootstrap uses the **first selected record's bbox**, not the seed bbox | VERIFIED_CODE + VERIFIED_OUTPUT | `EA:111-117`; GOLD.snap[0] = [659.5, 309.3, 758.2, 444.9] ≠ fixture seed [635.3, 304.1, 757.8, 438.3] | `tracking_confidence=1.0, tracking_state="TRACKING"` |
| Frame with no selected record → carry the last box, LOST, conf 0 | VERIFIED_CODE | `EA:124-133` | `bbox=list(last.bbox) … tracking_state="LOST"` |
| State mapping | VERIFIED_CODE | `EA:135` | `NEW_TO_LEGACY_STATE.get(sel["state"], "LOST")` |
| Identity downgrade | VERIFIED_CODE | `EA:136-139` | if legacy == TRACKING and `identity_state` ∈ {REVIEW_REQUIRED_IDENTITY, UNRESOLVED_IDENTITY} → OCCLUDED |
| Timestamp | VERIFIED_CODE | `EA:129,144` | `round(frame_number / fps, 4)` |
| Confidence | VERIFIED_CODE | `EA:146` | `sel.get("overall_track_confidence", 0.0)` |
| Output fields | VERIFIED_CODE | `EA:142-148` | player_id, frame_number, timestamp, bbox, centre_x, centre_y (2 dp), tracking_confidence, tracking_state |

### 5.2 `NEW_TO_LEGACY_STATE` (missing `new_tracker_snapshots.py`): recovered partially

| Online state | Legacy state | Evidence |
|---|---|---|
| ACTIVE | TRACKING | VERIFIED_OUTPUT: 168/168 GOLD and 364/364 PC selected frames, joining frame_log against frozen snapshots. All identity_state values were AUTO_RESOLVED here, so no downgrade interferes |
| OCCLUDED | OCCLUDED | VERIFIED_OUTPUT: 3/3 GOLD |
| AMBIGUOUS | OCCLUDED | VERIFIED_OUTPUT: 8/8 GOLD (downgrade applies only to TRACKING, so this is the table, not the downgrade) |
| TEMPORARILY_LOST | LOST | DOCUMENTED: LIVE_IDENTITY_TRUST_DOWNGRADE_VALIDATION.md §4 live trace, blind_r2_009 frames 85–89 (`TEMPORARILY_LOST … LOST -> LOST`); not in surviving frozen outputs |
| REIDENTIFYING | REACQUIRING | DOCUMENTED: same trace, frame 90 (`REIDENTIFYING … REACQUIRING -> REACQUIRING`) |
| PROVISIONAL, OUT_OF_FRAME | **UNKNOWN** | never the selected state in any surviving record or trace (OUT_OF_FRAME → LOST via the `.get` default is INFERRED only) |
| (any unlisted key) | LOST | VERIFIED_CODE: `EA:135` default |

The legacy vocabulary is DOCUMENTED as `{TRACKING, OCCLUDED, REACQUIRING, LOST}` (TARGET_PLAYER_ADAPTER_COMPATIBILITY_REPORT.md §1). LIVE_IDENTITY_TRUST_DOWNGRADE_VALIDATION.md §3 also gives a separate *diagnostic* grouping: ACTIVE/OCCLUDED → AVAILABLE, REIDENTIFYING/AMBIGUOUS → AMBIGUOUS, TEMPORARILY_LOST/OUT_OF_FRAME/TERMINATED → LOST. That grouping is not the adapter table.

**Selected-id changes did occur historically** (DOCUMENTED). After a loss, the online tracker moved `selected_player` onto a new track id: blind_r2_009 T001 → T064 at frame 90, and shot_close_range_01 T001 → T076 at frame 116. In both cases the resolver judged the boundary UNRESOLVED or reviewed it. Neither of the two surviving frozen JSONs contains such a change.

### 5.3 `TrackingSnapshot` (missing `football_analysis/adapter.py`)

| Finding | Evidence | Source | Usage |
|---|---|---|---|
| Field set | VERIFIED_CODE (constructor calls) + VERIFIED_OUTPUT (snapshot JSON) | `EA:105-108,142-148`; GOLD.snap | `player_id, frame_number, timestamp, bbox, centre_x, centre_y, tracking_confidence, tracking_state` |
| Allowed states | DOCUMENTED | TARGET_PLAYER_ADAPTER_COMPATIBILITY_REPORT.md §1 | TRACKING, OCCLUDED, REACQUIRING, LOST |
| Save format | VERIFIED_OUTPUT | GOLD.snap, PC.snap | `{"snapshots": [ {…8 fields…}, … ]}`, with length = frame_log length + 1 (180 / 365) |
| Save API | VERIFIED_CODE | `ATP:282`, `PL:266,341` | `TrackingAdapter.save(snapshots, path)` (static/class method) |
| Load path into the engine | VERIFIED_CODE | `ATP:101-106` | `FootballAnalysisEngine(fps=fps).run_analysis_pass(video_path, filtered_snapshots, tracked_color_hist, start_frame=start_frame_offset, conf=0.15, out_json_path=…)` |

### 5.4 Pass-2 football_analysis consumers of `TrackingSnapshot` (surviving)

| Consumer | Evidence | Line | Usage |
|---|---|---|---|
| `event_detector.process_frame` | VERIFIED_CODE (3N bytes) | `ED3N:78,100-101,117,141` | `snapshot.frame_number`, `snapshot.bbox` (ball-tracker context, team reference sample, other-box exclusion) |
| event evidence `tracking_states` | VERIFIED_CODE | `ED3N:28-31,230,244` | counts of `s.tracking_state` over the segment |
| `confidence` tracking quality | VERIFIED_CODE | `CF3N:28-31` | `states.get("TRACKING")`, `states.get("OCCLUDED")` |
| possession pauses unless TRACKING | VERIFIED_CODE | `PD3N:383,424-429` | `if interaction_frame.tracking_state != "TRACKING": … paused=True` |
| arbitration | VERIFIED_CODE | `EARB3N:326` | `pf.tracking_state == "TRACKING" and pf.ball_distance <= 40.0` |
| receiver eligibility (3P) | VERIFIED_CODE | `RE:96-101,142,149-158` | a non-TRACKING target disables reception confirmation by nearby teammate/unknown boxes |
| controlled duel (3Q) | VERIFIED_CODE | `CD:21,123,126,268` | `snapshots[i].bbox`; `tracking_state == "TRACKING"` gates the contest and identity grounding |
| other-player boxes | VERIFIED_CODE | `ED3N:84-88` (docstring) | "person boxes that are NOT the tracked player (already excluded by the caller via IoU against snapshot.bbox)". **Pass 2 computes other players from its own YOLO pass (yolov8s, classes [0,32], conf 0.15), not from Pass-1 tracks.** |

**Consequence for R1.** Pass 2 treats only `TRACKING` as "the target is really here". OCCLUDED, REACQUIRING and LOST all mean "do not use for possession or receiver confirmation". Any R1 state that is not confidently observed **must map to a non-TRACKING legacy state**.

### 5.5 `actor_attribution` (EXACT): per-frame identity trust

VERIFIED_CODE `AA:93-120`. A frame is trusted only if **all** of these hold:

```
selected record present                                            (AA:97-99)
state == "ACTIVE"                                                   (AA:101)
identity_state not in {REVIEW_REQUIRED_IDENTITY, UNRESOLVED_IDENTITY} (AA:103)
position_source == "CONFIRMED_DETECTION"                            (AA:105)
occlusion_status == "NONE"   (default "NONE" if absent)             (AA:107)
neither online_track_id nor track_id appears in overlap_contests    (AA:109-111)
identity_confidence >= MEDIUM_THRESH (0.5, from confidence.py)       (AA:112-114)
```

Clip level:
- `RESOLVER_FACTOR` RESOLVED 1.0, REVIEW_REQUIRED 0.5, UNRESOLVED 0.0 (`AA:77`);
- more than one `online_track_id` in the window caps the score below MEDIUM (`AA:151-153`);
- an `identity_conflict_counts` pair involving the selected ids halves it (`AA:154-156`, `AA:183-187`).

**Limitation (DOCUMENTED, TASK3M_OFFLINE_RESOLVER_AUDIT Part 8).** This tests the track's *self-reported health*, not physical identity. A wrong-person track that is ACTIVE, detection-backed and uncontested passes.

### 5.6 Offline resolver chain (all missing; interface recoverable from callers and tests)

| Function | Evidence | Source | Observed contract |
|---|---|---|---|
| `build_tracklets_from_frame_log(frame_log) -> List[Tracklet]` | VERIFIED_CODE | `ATP:64,273`, `PL:82,282`, `TPI:117-118` | one Tracklet per `online_track_id`; `.tracklet_id`, `.online_track_id`, `.trajectory`, `.as_record()` (`PL:159-165,286,330`) |
| `resolve_selected_player(frame_log, tracklets, video_path=, encoder=)` | VERIFIED_CODE | `ATP:275`, `PL:300`, `TPI:257` | returns dict: `boundary_resolutions` (list of `BoundaryResolution`), `identity_state_by_online_track_id` (dict), `offline_resolved_selected_track_ids` (set) (`TPI:146-147,184-185,258-259`) |
| `BoundaryResolution` fields | VERIFIED_CODE | `TPI:153-155`, `PL:190-195` | before_track_id, online_chosen_track_id, gap_start_frame, gap_end_frame, candidates (list of dicts with track_id, stitch_confidence, stitch_reason, matches_online_choice), identity_state, resolved_track_id |
| `IdentityState` values | VERIFIED_CODE | `PL:196,201,367-371`, `ATP:317-319` | AUTO_RESOLVED_IDENTITY, REVIEW_REQUIRED_IDENTITY, UNRESOLVED_IDENTITY (compared to plain strings in ATP, so string-valued) |
| Single continuous run → AUTO_RESOLVED, no boundaries | VERIFIED_CODE (test) + DOCUMENTED | `TPI:254-262`; TASK3M_OFFLINE_RESOLVER_AUDIT items 2-3 | runs split only on `online_track_id` change; within-run drift is out of scope by design |
| Safety rule: an override of the online choice never reaches AUTO_RESOLVED | DOCUMENTED | `EA:11-20`; TRACKING_PIPELINE_INTEGRATION_MAP §3 | the position stream is never substituted |
| `enrich_frame_log_with_offline_identity(frame_log, resolver_result)` | VERIFIED_CODE | `ATP:277`, `TPI:186,198` | adds `identity_state` and `offline_resolved_selected_track` to selected records, per online id |
| `target_status` derivation | VERIFIED_CODE | `ATP:315-321` | UNRESOLVED if any unresolved boundary, else REVIEW_REQUIRED if any review boundary, else RESOLVED |
| `AppearanceEncoder(use_ocr=False)` | VERIFIED_CODE | `ATP:63,274`, `PL:298-299`, `IPR:31,43,109` | `.embed(frame, box)` → vector or None |

### 5.7 Health report (EXACT, diagnostic only)

VERIFIED_CODE `TH:20-75`. It reads `run_summary.health_metrics`, `n_tracks_ever_created`, `final_confirmed_count`, `final_provisional_count`, `spawn_guard_stats`, `identity_conflicts_logged` and `boundary_resolutions[].identity_state`, and counts snapshots with `tracking_state == "TRACKING"`. `wrong_id_flags` is structurally always 0 (`TH:64-71`): the old system had no way to detect a confirmed wrong ID.

---

## 6. Error and degradation behaviour (historical contract)

| Situation | Behaviour | Evidence |
|---|---|---|
| No selected player in any frame | one LOST snapshot, never a crash | VERIFIED_CODE `EA:100-109`; TEST `TPI:204-211` |
| Selected player absent in a frame | carry the previous bbox, LOST, conf 0 | VERIFIED_CODE `EA:124-133`; TEST `TPI:214-224` |
| Zero detections in the clip | pipeline completes; 0 tracklets, 0 boundaries, 1 LOST snapshot | TEST `TPI:326-342` |
| UNRESOLVED boundary | must appear in the review queue (`UNRESOLVED_NO_CANDIDATE`) | VERIFIED_CODE `PL:201-212`; TEST `TPI:152-160` |
| INTEGRATED stage raises | the exception propagates; no silent LEGACY fallback | DOCUMENTED `PL:61-69` |
| Video unreadable at seek | `RuntimeError` in Pass 2 | VERIFIED_CODE `ATP:97-98` |

---

## 7. Interface completeness score

Counting the 18 distinct Pass-1 interface elements that surviving code depends on:

| Element | Status |
|---|---|
| JSON top-level layout (`summary`/`frame_log`/`final_tracks`) | **FULLY SPECIFIED** |
| `summary.fps`, `start_frame`, `processed_frames`, `identity_conflict_counts` | **FULLY SPECIFIED** |
| frame_log record keys | **FULLY SPECIFIED** |
| `overlap_contests` format | **FULLY SPECIFIED** |
| 29 per-track fields: names and types | **FULLY SPECIFIED** |
| `selected_player` exactly-one-per-frame | **FULLY SPECIFIED** (observed; an absence path is also defined) |
| `state` vocabulary | **FULLY SPECIFIED** (7 observed values) |
| `occlusion_status` vocabulary | **FULLY SPECIFIED** |
| `position_source` vocabulary | **FULLY SPECIFIED** |
| seed format (`seed_frame`, `seed_bbox` xyxy px) | **FULLY SPECIFIED** |
| frame numbering vs video frames | **FULLY SPECIFIED** (from data) |
| `run_mo.run` signature | **FULLY SPECIFIED** |
| `TrackingSnapshot` fields and save format | **FULLY SPECIFIED** |
| `NEW_TO_LEGACY_STATE` | **PARTIALLY SPECIFIED** (3 of 7 VERIFIED_OUTPUT + 2 DOCUMENTED; PROVISIONAL, OUT_OF_FRAME UNKNOWN) |
| semantics/formula of confidence fields (`identity_confidence`, `overall_track_confidence`, …) | **PARTIALLY SPECIFIED** (consumers and thresholds known; computation UNKNOWN) |
| resolver / tracklet / enrichment return shapes | **PARTIALLY SPECIFIED** |
| `is_cut` / `camera_shift` semantics | **PARTIALLY SPECIFIED** (types known; no cut ever observed) |
| `AppearanceEncoder` embedding model and dimensions | **UNKNOWN** (model origin unknown; dimension not recorded) |

**Totals: FULLY 13 · PARTIALLY 4 · UNKNOWN 1.**

---

## 8. What R1 may add (all NEW, additive, and labelled)

These are **not** historical. They are listed here only to show where the historical contract leaves gaps that R1 must fill explicitly:

- **NEW** per-frame `identity_uncertainty_reason` (e.g. `OCCLUDER_OVERLAP`, `COMPETING_CANDIDATE`, `MERGED_DETECTION_SUSPECTED`, `REACQUISITION_UNVERIFIED`). Historically the only per-frame uncertainty carriers were `state`/`occlusion_status`/`position_source`.
- **NEW** `occluder_track_ids`. Historically, `overlap_contests` was blind to PROVISIONAL tracks (TASK3M).
- **NEW** `bbox_is_held` (explicit). Historically this was derivable only as `position_source != CONFIRMED_DETECTION`, and Pass 2 never received it.
- **NEW** `reidentified_this_frame`. Historically REIDENTIFYING was a state, seen once in GOLD, with an unknown legacy mapping.
- **NEW** schema version field on the tracking JSON. Historically the version lived only in the pipeline summary (`PL:105-111`).

The R1 plan proposes these as an **additive v2 sidecar block inside each track record**, so the 29 historical fields stay byte-shape-compatible.

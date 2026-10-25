# R1-G output mapping: how `run_rb` fills the Pass-1 contract

**Contract:** `player_tracker_rb/output_schema.py`, v2 schema `2.0.0-r1a` (approved in R1-A). **R1-G does not change the contract**: no key, type, unit or vocabulary was added, removed or redefined. Every frame record is validated with `require_v2=True` before it is written, and the finished document is validated again in the tests, both as-is and after `strip_v2` (historical contract).

One behaviour-neutral serialisation fix was made in `output_schema.build_frame_record`: `camera_shift` values are written as `0.0`, never `-0.0`.

File: `{video_basename}_mo_tracking_results.json`. Sidecars: `{video_basename}_rb_video_provenance.json` (always) and `{video_basename}_rb_decision_log.jsonl` (with `debug=True`).

## Frame numbering

| Item | Rule |
|---|---|
| Seed | Video frame `start_frame` is relative frame 0, the initialization frame. It is **not** a `frame_log` record. |
| `frame_log[].frame` | `f = 1 … N`; video frame `start_frame + f` (`v2.video_frame_index`). |
| `v2.timestamp_s` | `video_frame_index / fps` (absolute video time). |
| Internal tracker time | `f / fps` (time since the seed). |

## TARGET record (one per frame, `selected_player = true`)

| Field | Value | Semantics |
|---|---|---|
| `track_id`, `identity_id`, `online_track_id` | `T001` (`target_track_id`) | One logical target for the whole run (D07). |
| `state`, `occlusion_status`, `position_source` | `TARGET_TRIPLE_BY_RB_STATE[(rb_state, inside)]` | D05. Only TRACKING → ACTIVE / CONFIRMED_DETECTION. |
| `bbox` | Last **confirmed** (observed) box; the seed detection until the first confirmation | Historical meaning (D06; HE1 frame 86). |
| `observed_position_box` | This frame's observed (TRACKING), predicted or held box | `v2.observation_source` says which. |
| `age` | `f` | Frames since the seed. |
| `last_seen_frame` | `v2.last_confirmed_frame` (0 = seed) | |
| `frames_since_observed_position`, `frames_since_confirmed_identity` | `0` when TRACKING, else `f - last_confirmed_frame` | |
| `identity_confidence` | `v2.identity_confidence_rb` when trusted; else `min(…, 0.49)` | D19. Keeps the surviving consumers' 0.5 threshold safe. |
| `selected_player_confidence` | = `identity_confidence` | Contract rule. |
| `overall_track_confidence` | = `identity_confidence` | event_adapter turns it into the snapshot `tracking_confidence`. |
| `detection_confidence` | Historical quantised value by state / position_source | NOT a detector score; the raw score is `v2.raw_detection_conf`. |
| `appearance_confidence`, `reid_confidence` | Composite appearance similarity of the reported box to the target bank (0 if none) | **NEW semantics.** |
| `motion_confidence` | 1 while positional continuity is valid, else 0 | **NEW semantics.** |
| `visibility_confidence`, `position_confidence`, `track_continuity_confidence` | 1 when observed (TRACKING), else 0 | **NEW semantics.** |
| `team`, `team_confidence`, `number_candidate` | `UNKNOWN`, 0, null | The generic tracker has no team or number logic. |
| `v2` | `TargetFrame.v2_fields()`: exactly `V2_TARGET_FIELDS` | |

### Uncontrasted reacquisition: capped and flagged

A reacquisition accepted with no banked competitor to contrast against is recorded as follows:
- The event has `summary.v2.reacquisition_events[].evidence.contrast_available = false`.
- The acceptance frame, **and every later frame of that episode**, has `identity_confidence_rb = 0.5`, the minimal trusted value.
- The cap lifts only when a TRACKING frame passes a real contrast against a banked competitor. That frame is logged as `episode_contrast_verified` in the decision log.

## POPULATION records (every other live tracklet, `selected_player = false`)

| Field | Value |
|---|---|
| `track_id` | The population tracklet id (`T002`, `T003`, …; `T001` is never issued). |
| `state` / `occlusion_status` / `position_source` | Detected: ACTIVE / NONE / CONFIRMED_DETECTION. Coasting: TEMPORARILY_LOST / FULL / PREDICTED. |
| `bbox` | Last observed box. |
| `observed_position_box` | This frame's observed, predicted or held box. |
| `age` | `f - first_frame`. |
| `last_seen_frame` | Last observed frame. |
| `identity_confidence` | 0: no identity claim. |
| `overall_track_confidence` | Detector score when detected, else 0. |
| `v2` | `record_kind = POPULATION`, `detected_this_frame`, `raw_detection_conf`, `is_competitor`, `first_frame`. |

**Physical-entity exclusion (project rule 28).** These tracklets are never written as a separate person:
- every tracklet the target was confirmed on;
- the tracklet the target's decision is about this frame (`v2.observation_tracklet_id`).

In `overlap_contests` they are written as `T001`. A merged box that physically contains the target is still written, because it also contains another person. Pass 2 must not treat a population record overlapping an untrusted target as independent of it (an R1-H question).

## Frame record

| Field | Value |
|---|---|
| `is_cut`, `v2.cut_detected` | Camera status CUT (histogram change + flow failure). A continuity break from UNKNOWN motion does not set `is_cut`; it does put CAMERA_CUT in the target's reasons. |
| `camera_shift` | Image-centre displacement in px from the camera estimate (0 unless ESTIMATED). |
| `v2.camera_shift_h` | `camera_shift` divided by the target height px; null while the height is unknown. |
| `n_detections` | Raw person candidates returned by the detector. |
| `n_live_tracks` | `len(tracks)`. |
| `overlap_contests` | Every overlapping pair of bodies (population overlap context) as written track ids, sorted. A target in a contest makes the surviving `actor_attribution` distrust the frame, which is conservative. |

## Summary

| Field | Value |
|---|---|
| `video`, `start_frame`, `fps`, `processed_frames` | From the reader. |
| `processing_time_s` | Wall-clock time. |
| `selected_state_pct` | Historical states of the TARGET record. |
| `n_tracks_ever_created` | 1 + distinct population ids written. |
| `n_tracks_final_active` | Tracks in the last frame. |
| `final_confirmed_count` / `final_provisional_count` | Last-frame tracks by population provisional flag (the target counts as confirmed). |
| `clip_output` | null (no debug video). |
| `spawn_guard_stats` | `{}`: no spawn guard in R1. |
| `identity_conflict_counts` | `T001\|Txxx` → number of frames in which Txxx was the competing hypothesis (best competitor, unknown contact, or reacquisition rival) that kept the target from being confirmed (F9 fix). |
| `identity_conflicts_logged` | Number of such frames. |
| `health_metrics` | `frames_processed`; `total_tracks_spawned` (population tracklets, the target's own included); `new_tracks_per_frame`; `tracks_per_100_frames`; `median_track_lifetime_frames` (written population records); the others are null because R1 has no such mechanism. |
| `v2` | Schema, versions, `provenance` (contract block), `config_hash`, `reacquisition_events`, `rb_state_pct`, `notes` (these semantics in short). |

`final_tracks` holds the last written record of every track ever written, without its v2 block. A track absent from the last frame has `state = TERMINATED`.

## Determinism and streaming

- **Determinism.** Identical inputs, code and config give a document that is byte-identical except `summary.processing_time_s` and `summary.v2.provenance.run_started_utc` (`run_rb.WALL_CLOCK_FIELDS`). `run_rb.compare_documents` reports the first other difference, if any. The decision log contains no wall-clock field and is byte-identical.
- **Streaming.** Frame records are streamed to `*.rb_partial` and the canonical document is assembled at the end. The result is byte-identical to `output_schema.dumps_canonical` of the whole document (tested). Memory is bounded by live tracks plus the number of distinct tracks (`final_tracks`), not by clip length.

## Fail closed

The run is refused (`RunRefusedError`) and nothing is written when any of these hold:
- `jersey_number` is given;
- `out_dir` is missing;
- an output already exists;
- the backends don't match the config (production config with a substitute backend, or a scripted config without its backend);
- the provenance (git state) is unknown;
- the seed is refused.

Other failures:
- A source-hash mismatch or any video error raises from `video_io`; nothing is written.
- Any contract violation raises `ContractViolationError`; the partial output is deleted.

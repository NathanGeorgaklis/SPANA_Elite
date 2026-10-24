# R0_R1_TRACKER_REBUILD_PLAN.md

SPANA engine v2-rebuild — **plan for R1 (selected-player tracking).** This is a proposal only. **Nothing here is built. R1 needs your approval before it starts.**

Everything R1 creates is **NEW CODE**. Each new Python file carries this header:

```python
# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R0_TRACKER_INTERFACE_MAP.md, R0_TRACKER_FAILURE_CATALOGUE.md, <specific items>
```

---

## 0. Scope

**In scope.** Select a player from a seed, track that physical player, know when identity is uncertain, and emit an auditable per-frame record. The record must satisfy the historical Pass-1 JSON contract, plus clearly labelled NEW fields.

**Out of scope for R1:**
- events, possession, ball, stats;
- the offline resolver and enrichment;
- any change to surviving recovered files;
- Task 3R;
- the holdout.

---

## 1. Architecture: the smallest design that satisfies the recovered contract

**Contract, not code.** The historical boundary is the tracking JSON (`R0_TRACKER_INTERFACE_MAP.md` §0). R1 therefore does **not** recreate any of the 30 historical `player_tracker`/`player_tracker_mo` module names. It reimplements the *contract* in one new package.

**One package, no Pass-2 imports.** `player_tracker_rb/` imports nothing from `football_analysis/`, `player_tracker/` or `player_tracker_mo/`. This prevents a repeat of F14, where a Pass-2 change silently altered tracking.

**No `player_tracker_mo_rb/` in R1.** The offline layer (tracklets, resolver, enrichment) is a Pass-2-side concern and is deferred to R3. Building it now would be "recreating files because they existed".

### Files to create (all NEW) under `~/Documents/SPANA_Engine_Rebuild/`

| # | Path | Responsibility | Evidence basis |
|---|---|---|---|
| 1 | `player_tracker_rb/__init__.py` | Package marker. Exposes `run`, `SCHEMA_VERSION` and `TRACKER_VERSION`. | — |
| 2 | `player_tracker_rb/config.py` | Frozen dataclass of every parameter. Geometry is in **target body heights (h)**, time in **seconds**. The config is hashed into the output. | F16; charter §22 |
| 3 | `player_tracker_rb/video_io.py` | One seek to `start_frame`, then **sequential** decode only. Records fps from the container and a per-frame content hash (a downscaled-frame hash). No checkpoint/resume. | F15; TASK3F §2 |
| 4 | `player_tracker_rb/detector.py` | Person detector wrapper with a pinned model path and SHA-256. Returns `(box, raw_conf)`. Refuses to run if the model hash does not match the config. | behaviour evidence §1; R0.8 |
| 5 | `player_tracker_rb/appearance.py` | (a) ReID embedding (model to be approved, §3). (b) **NEW** LAB upper/lower-body kit descriptor, written inside this package (not imported from Pass 2). (c) A multi-view target bank. (d) Per-competitor short banks. | F1 (HSV inverts, LAB separates); F3; TASK3M §14 "multi-view reference" |
| 6 | `player_tracker_rb/motion.py` | Constant-velocity filter in normalised units. Global camera-motion estimate (sparse optical flow on the background, excluding person boxes). Hard-cut detector. | behaviour evidence §5; F11 |
| 7 | `player_tracker_rb/population.py` | Lightweight tracks for **all other people**, so that competitors and occluders are known. Uses IoU/motion association only. It makes **no identity claims** and has no promotion gating, so any detection-backed body counts as a competitor immediately. | F4 |
| 8 | `player_tracker_rb/target_state.py` | **The selected-target state machine** (§2): candidate scoring, contrastive identity test, merged-box guard, hold and decay limits, reacquisition verification. | F1, F2, F6, F8, F10 |
| 9 | `player_tracker_rb/output_schema.py` | Writes `{base}_mo_tracking_results.json` (historical-compatible) with the additive `v2` blocks (§4). Contains `validate(json)`, which implements acceptance check B6. | Interface map §2–4, §8 |
| 10 | `player_tracker_rb/run_rb.py` | `run(video_path, start_frame, init_box, jersey_number=None, max_frames=450, out_dir=None, debug=False, config_overrides=None)`. This is the historical signature **minus `resume`**. It verifies the seed and writes provenance. Also usable from the CLI. | Interface map §1 |
| 11 | `tests_rb/test_config_units.py` | Frame-rate independence (the same scenario at 30 and 60 fps gives the same decisions); scale normalisation. | F16 |
| 12 | `tests_rb/test_target_state_synthetic.py` | Synthetic detection streams (no video) for each failure mode: fast-path swap, merged box, competitor present from birth, AMBIGUOUS decay, held-box honesty, wrong reacquisition refused, leaving frame leads to LOST. | F1–F10 |
| 13 | `tests_rb/test_output_contract.py` | Validates R1 output against the 29-field, 7-key and summary contract, and against the two **frozen historical JSONs** as schema exemplars (GOLD 88246efe, PC 40fa0df1). Also checks that the NEW block is additive. | B6 |
| 14 | `tests_rb/test_surviving_consumers.py` | Runs the **unmodified** surviving `actor_attribution.frame_identity` and `event_adapter.build_integrated_event_snapshots` on R1 output. Their missing imports (`new_tracker_snapshots`, `football_analysis.adapter`) are provided by **test-only `sys.modules` stubs** that live in `tests_rb/` and are labelled NEW / TEST-ONLY. No historical path is written. | Interface map §5 |
| 15 | `tests_rb/test_seed_and_provenance.py` | The seed must overlap a raw detection at the seed frame, otherwise the run is refused. The source hash must be recorded. | F12 |
| 16 | `tools_rb/make_gold_label_sheet.py` | Builds contact sheets of GOLD extract frames 250–290 for **human** confirmation of per-frame target identity. No model is drawn on the sheets. The human then promotes it to truth. | Tracker acceptance B4 |
| 17 | `R1/R1_SCHEMA.md`, `R1/R1_DESIGN_NOTES.md`, `R1/R1_REPORT.md` | Documentation. | — |

That is **10 source files, 5 test files and 1 tool**. Historical paths (`player_tracker/`, `player_tracker_mo/`, `new_tracker_snapshots.py`) are **not** written.

---

## 2. Target state machine and identity strategy (NEW behaviour)

### 2.1 States

R1 has four public states, as the charter asks. They are mapped onto the historical online vocabulary so that the historical adapter table produces the right legacy states.

| R1 state | Meaning | Historical `state` written | Historical `occlusion_status` / `position_source` | Legacy state via historical table |
|---|---|---|---|---|
| **TRACKING** | Target observed this frame, and identity verified contrastively | ACTIVE | NONE / CONFIRMED_DETECTION | TRACKING (VERIFIED_OUTPUT mapping) |
| **OCCLUDED** | Target hidden; identity propagated safely; box is **held or predicted** | OCCLUDED | PARTIAL / PREDICTED | OCCLUDED (VERIFIED_OUTPUT) |
| **AMBIGUOUS** | Two or more plausible candidates, or merged or contaminated evidence | AMBIGUOUS | AMBIGUOUS / AMBIGUOUS_DETECTION | OCCLUDED (VERIFIED_OUTPUT) |
| **LOST** | Identity not available | TEMPORARILY_LOST (or OUT_OF_FRAME when the predicted box left the image) | FULL / PREDICTED | LOST (DOCUMENTED for TEMPORARILY_LOST; OUT_OF_FRAME → LOST via the `.get` default is INFERRED, so R1 tests pin it) |
| *(sub-state)* REACQUIRING | A candidate is under verification after OCCLUDED or LOST | REIDENTIFYING | FULL / CONFIRMED_DETECTION | REACQUIRING (DOCUMENTED) |

**Invariant.** Every non-TRACKING R1 state maps to a **non-TRACKING** legacy state. Pass-2 consumers treat only TRACKING as "target present" (interface map §5.4).

### 2.2 Candidate scoring: identity over proximity

For each detection inside a gate (gate radius in **h**, scaled by time since the last observation, and **capped**):

| Signal | Use |
|---|---|
| **Appearance vs target bank** | ReID cosine plus LAB kit similarity |
| **Appearance vs every competitor** | Contrastive: the best competitor-bank similarity for that box |
| Motion | Camera-compensated predicted-position likelihood, in h |
| Scale | Box height ratio versus the expected height |
| Geometry | Overlap with competitors' predicted boxes |

**Decision rule** (no single signal dominates):

1. **Never confirm** a box unless:
   - (a) target-appearance ≥ an absolute floor;
   - **and** (b) target-appearance exceeds the best competitor appearance on that box by a contrastive margin;
   - **and** (c) motion is plausible;
   - **and** (d) the box is not merged-suspect (§2.3).
2. There is **no fast path**. The appearance check always runs, even when the box is the only candidate. This fixes F1.
3. If two or more candidates pass 1(a)–(c) and their combined score margin is below the ambiguity margin, the frame is **AMBIGUOUS**. There is no guess.
4. Motion and IoU-to-last-box can **veto** a candidate (implausible jump). They can **never on their own** overturn an appearance preference. This fixes F1(e).
5. The appearance bank updates **only** on TRACKING frames with no competitor overlap. It is never updated on AMBIGUOUS, OCCLUDED, merged or contaminated frames. This carries forward what worked historically: the clean 3M bank enabled recovery.

### 2.3 Merged-detection guard (F2)

A box is flagged `merged_suspected` when any of these holds:

- its area jumps more than X× versus the target's recent box;
- it overlaps both the target's prediction and a competitor's prediction;
- the competitor track that was adjacent loses its detection in the same frame.

A merged-suspected box can never produce TRACKING. The state becomes OCCLUDED if the target's prediction is inside the box, and AMBIGUOUS otherwise.

### 2.4 Occlusion and ambiguity handling (F6)

| Mechanism | Behaviour |
|---|---|
| Held box | `bbox` is the **held last-confirmed** box (historical semantics). `observed_position_box` is the predicted or contested position. `v2.bbox_is_held = true`. `v2.held_age_s` is explicit. |
| Hold limits (seconds) | OCCLUDED → LOST after `max_occluded_s`. AMBIGUOUS → LOST after `max_ambiguous_s`. **AMBIGUOUS can no longer persist indefinitely.** |
| Gate growth | The gate grows while unobserved but is **capped**. It does not grow while AMBIGUOUS. |

### 2.5 Reacquisition (F8, F10): a separate decision

A candidate is accepted after OCCLUDED or LOST only when all of these hold:

- (i) it passes the contrastive test for at least K frames within a window of W seconds;
- (ii) no competitor is a better match to the target bank during those frames;
- (iii) for LOST longer than T seconds, the appearance bar **rises** and motion evidence is ignored;
- (iv) after a cut, reacquisition requires appearance only, and the bar is at its highest.

During verification the state is REACQUIRING (REIDENTIFYING). On acceptance the target keeps id **T001** and `v2.reidentified_this_frame = true`.

> **Design decision for your approval.** Historically, after a loss the online tracker moved `selected_player` to a **new** track id, and identity was left to the (now missing) offline resolver at that id boundary. R1 instead keeps T001 for the target and records the reacquisition evidence per frame. The trade-off: the historical resolver would see no boundary. That is safe only because R1 publishes per-frame `v2.identity_trusted`, and R3 must consume it.
>
> The alternative is to emulate the historical behaviour (new id after LOST). Please choose. **Default if no answer: keep T001, flag reacquisition.**

### 2.6 Seed initialisation (F12)

- Inputs: `seed_frame` plus `seed_bbox` `[x1, y1, x2, y2]` in pixels (the historical format, `BF:104-105`). The provenance tag remains `manual_bbox_hint`.
- Refuse the run if no raw detection overlaps the seed with IoU ≥ 0.5 at the seed frame. Record the best IoU.
- The target bank is initialised from the seed frame plus the next frames in which the target is unambiguous (no competitor overlap).
- **Event labels are never read.**

---

## 3. Detector and model dependencies (all NEW; approval needed)

None of the three historical models survives, and `yolo26n-reid.onnx` has no known origin (`R0_TESTS_AND_MODELS.md` §4).

| Need | Option | Recommendation |
|---|---|---|
| Person detector | **D1:** Ultralytics `yolov8s.pt` from the official release. Download in R1, then compare its hash with the historical `1f47a78b…`. If it matches, record "same bytes as historical file" (for that file only). **D2:** a newer Ultralytics model (YOLO11/26). | **D1**, for comparability. |
| ReID embedding | **A1:** an open person-ReID model (for example OSNet trained on Market-1501/MSMT17), exported to ONNX and pinned by hash. **A2:** search for the historical model. **A3:** no learned ReID; LAB kit descriptor plus jersey OCR only. A3 would make R1 abstain much more in same-kit crossings. | **A1**, with the A3 descriptor always on as a second cue. |
| OCR (optional) | EasyOCR | Off by default in R1. The historical OCR produced spurious numbers. |

**Licensing flag.** Ultralytics models are AGPL-3.0 unless you hold an enterprise licence. ReID weights carry their own dataset licences, and Market-1501 is research-only. **This matters for a commercial SPANA product. Please confirm before R1 downloads anything.**

---

## 4. Output schema (historical-compatible plus additive NEW)

- **File name:** `{video_basename}_mo_tracking_results.json`, with top-level keys `summary`, `frame_log` and `final_tracks` (historical).
- **summary:** all 15 historical keys, plus NEW keys:
  - `schema_version: "spana-v2rb-tracking/0.1"`
  - `tracker_version`, `code_sha256`, `config_sha256`
  - `model_sha256: {detector, reid}`
  - `source_video_sha256`
  - `seed: {frame, bbox, best_detection_iou}`
  - `rebuild_notice: "SPANA engine v2-rebuild — NEW CODE, not the 3A–3Q tracker"`
- **frame_log[i]:** the 7 historical keys. `overlap_contests` lists **every** competitor pair touching the target, not only "scorable" tracks. NEW `v2: {frame_content_hash, camera_motion_h: [dx, dy], cut_detected}`.
- **track record:** the 29 historical fields with historical semantics where they are known:
  - `detection_confidence` is quantised 1.0 / 0.5 / 0.0 as historically observed;
  - `source` takes the NEW value `"REBUILD_V2"`;
  - `identity_confidence` is NEW-computed from contrastive evidence and is designed to be **< 0.5 whenever identity is not verified**, so that the surviving `actor_attribution` (threshold 0.5) behaves correctly.

  Plus NEW `v2` on the selected track:

  ```json
  {
    "rb_state": "TRACKING|OCCLUDED|AMBIGUOUS|LOST|REACQUIRING",
    "identity_trusted": true,
    "identity_uncertainty_reasons": ["OCCLUDER_OVERLAP", "COMPETING_CANDIDATE", "MERGED_DETECTION_SUSPECTED", "BELOW_APPEARANCE_FLOOR", "REACQUISITION_UNVERIFIED", "POST_CUT", "HOLD_LIMIT"],
    "competitor_track_ids": [], "occluder_track_ids": [],
    "bbox_is_held": false, "held_age_s": 0.0, "last_confirmed_frame": 0,
    "observation_source": "DETECTION|PREDICTED|NONE",
    "raw_detection_conf": 0.0, "merged_detection_suspected": false,
    "appearance_target_sim": 0.0, "appearance_best_competitor_sim": 0.0, "appearance_margin": 0.0,
    "motion_score": 0.0, "association_score": 0.0,
    "reidentified_this_frame": false
  }
  ```

  Other tracks carry a minimal `v2` block (`population_only: true`).
- **Invariant tests:** stripping every `v2` key and the NEW summary keys leaves a document that validates against the historical exemplars.

---

## 5. Validation in R1 (R2 does the full benchmark)

| Check | Pass criterion |
|---|---|
| Synthetic unit suites (§1, files 11–15) | 100 % pass |
| Output contract (B6) | 100 %, against both historical exemplars |
| Surviving consumers run unmodified | `actor_attribution.frame_identity` and `event_adapter` succeed on R1 output; every non-TRACKING R1 frame produces a non-TRACKING legacy snapshot |
| **GOLD_DUEL_01 development run** (fresh from source 164541…, seed from fixture 7929bb28…) | Never CONFIDENT (TRACKING) on MAR #4 in extract 262–274, checked against **human-confirmed** per-frame labels (file 16). TRACKING on #23 outside that window for most frames (reported, not gated). |
| Determinism | Two runs give byte-identical JSON |
| Runtime | Reported (historically 912 s for 179 GOLD frames) |

**Explicitly not used:** GOLD's DUEL label, any event output, Task 3R and the holdout.

---

## 6. Failure behaviour

| Condition | Behaviour |
|---|---|
| Model file missing or hash mismatch | Stop with an error before reading video |
| Video unreadable or seek fails | Stop with an error |
| Seed not detection-backed | Refuse; write a `seed_rejected` report |
| Zero detections | Valid JSON; target LOST from frame 1 (matches historical degradation, TPI:326) |
| Target leaves frame | OUT_OF_FRAME / LOST with held box; never moves to another body |
| Any internal exception | Propagates; no partial JSON is marked valid |

---

## 7. Persistence checkpoints (charter §13–16, 51)

1. **Pre-task.** Confirm `~/Documents/SPANA_Engine_Rebuild/` is writable. List it, hash it, snapshot it to `~/Documents/SPANA_Engine_Archive/R1_PRE_<ts>/` and verify from the archive copy.
2. **Git.** `git init` in `~/Documents/SPANA_Engine_Rebuild/`. **Environment limitation found in R0:** this session's Mac shell cannot delete files in connected folders, and git needs to delete its own lock files. Either grant delete permission for the Documents folder when R1 asks, or run `git init/add/commit` yourself in Terminal. R0 did not initialise git for this reason.
3. **After each file.** Write to disk, record the SHA-256 in `CHANGED_FILES.txt`, generate a unified diff, run tests and save the raw test output.
4. **End of R1.** Archive snapshot `R1_<ts>/` with `FULL_FILE_MANIFEST_SHA256.txt`, `CHANGED_FILES.txt`, `CODE_DIFF.patch`, `TEST_RESULTS.txt` and `BUILD_STATUS.md`. Then archive-side hash verification and the persistence check block.
5. **Second backup.** Not available yet (see BUILD_STATUS). Proposed: a private Git remote or an external SSD that you nominate.

---

## 8. Open decisions for your approval

1. **Models and licence** (§3): D1 + A1? Download allowed in R1?
2. **Target id after reacquisition** (§2.5): keep T001 (default), or emulate the historical new-id behaviour?
3. **Git:** may R1 request delete permission for Documents, or will you run git locally?
4. **Human labelling:** will you confirm the GOLD 250–290 per-frame identity sheet (about 40 frames, about 10 minutes)?
5. **R2 material:** may R2 extract new windows from `videoplayback.mp4` and `videoplayback (1).mp4`? This excludes Task 3R and GENERAL_BENCHMARK_V2 windows and the holdout.

# R1_DESIGN_DECISIONS.md

SPANA engine v2-rebuild, R1 (selected-player tracker). Written at the **pre-coding gate** on 2026-10-24. No tracker source exists yet. Each decision below needs Nathan's approval before implementation.

Evidence labels follow the charter (VERIFIED_CODE, VERIFIED_OUTPUT, VERIFIED_TEST, DOCUMENTED, INFERRED, UNKNOWN). "HISTORICAL_REFERENCE" marks a recovered historical value that is shown for comparison only. It is **not** an R1 parameter.

---

### D01. Contract boundary
- **DESIGN DECISION:** R1 reproduces the Pass-1 **JSON contract**, not the old tracker's internals. Output is `{video_basename}_mo_tracking_results.json` with `summary`, `frame_log` and `final_tracks`.
- **EVIDENCE SOURCE:** R0_TRACKER_INTERFACE_MAP §0, §2–4; `analyze_target_player.py:269-271`; GOLD.json 88246efe; PC.json 40fa0df1.
- **EVIDENCE LEVEL:** VERIFIED_CODE + VERIFIED_OUTPUT.
- **HISTORICAL BEHAVIOUR:** Pass 2 only ever read this file.
- **KNOWN HISTORICAL DEFECT:** Useful Pass-1 context, such as `observed_position_box`, never reached Pass 2 (F5).
- **NEW R1 CHOICE:** The historical keys keep their historical semantics wherever those are known. All new information goes in additive `v2` blocks plus new summary keys.
- **WHY:** Surviving and rebuilt Pass-2 code can then be tested against R1 without coupling to tracker internals.
- **VALIDATION:** `tests_rb/test_output_contract.py` validates against both frozen exemplars. Stripping the v2 blocks must leave a historically valid document.

### D02. Package isolation
- **DESIGN DECISION:** All tracker code lives in one new package, `player_tracker_rb/`. It has **no imports** from `football_analysis/`, `player_tracker/`, `player_tracker_mo/` or `recovered/`.
- **EVIDENCE SOURCE:** F14 (a 3L TeamClassifier change altered Pass 1); TASK3M; R0_DEPENDENCY_GRAPH §3.
- **EVIDENCE LEVEL:** DOCUMENTED.
- **HISTORICAL BEHAVIOUR:** `team_track` imported Pass-2 `TeamClassifier`.
- **KNOWN HISTORICAL DEFECT:** Pass-2 edits silently changed tracking.
- **NEW R1 CHOICE:** The tracker owns every descriptor it uses. A test asserts that no forbidden import exists.
- **WHY:** Pass 1 must be reproducible independently of the event engine.
- **VALIDATION:** An import-scan test in `test_seed_and_provenance.py`.

### D03. Detector abstraction
- **DESIGN DECISION:** `detector.py` defines `DetectorBackend.detect(frame) -> list[Detection]`, where `Detection = (bbox_xyxy_px, raw_conf, cls, meta)`. Backends are pluggable. Every backend reports its model filename, SHA-256 and input size.
- **EVIDENCE SOURCE:** R1 prompt §15 and §17; R0_TESTS_AND_MODELS §4 (historical models are missing and their origin is partly unknown).
- **EVIDENCE LEVEL:** DOCUMENTED (requirement).
- **HISTORICAL BEHAVIOUR:** `player_tracker.detector.PlayerDetector` wrapped YOLO, with `classes=[0]` (VERIFIED_CODE in IPR:70).
- **KNOWN HISTORICAL DEFECT:** Which weights Pass 1 used is UNKNOWN; the reports conflict.
- **NEW R1 CHOICE:** The primary backend is **YOLOX (Apache-2.0) on ONNX Runtime**; see D17 and the model record. An optional Ultralytics backend may exist only as a DEVELOPMENT_ONLY comparison, isolated in its own class. No library-specific result object may leave `detector.py`.
- **WHY:** Licensing (commercial product); replaceability.
- **VALIDATION:** The synthetic tests use a `ScriptedDetector` backend. The contract test checks that provenance fields are present.

### D04. Appearance abstraction and composite descriptor
- **DESIGN DECISION:** `appearance.py` defines `AppearanceBackend.describe(frame, bbox) -> AppearanceFeature(vector, quality, valid, meta)`. Similarity is computed by the backend, so `target_state` never sees embedding dimensions. The default composite has two cues: **(a)** a NEW, tracker-owned LAB part-colour descriptor (upper/lower body), and **(b)** a learned general-purpose embedding (proposed: DINOv2 ViT-S/14, ONNX).
- **EVIDENCE SOURCE:** TASK3M_REID §4 (HSV colour inverts under contamination; LAB separates); R0.8 (yolo26n-reid origin UNKNOWN); R1 prompt §18–19.
- **EVIDENCE LEVEL:** DOCUMENTED.
- **HISTORICAL BEHAVIOUR:** The ReID embedding came from `yolo26n-reid.onnx`, plus an HSV histogram and EasyOCR.
- **KNOWN HISTORICAL DEFECT:** The HSV colour cue (weight 0.15) favoured the occluder during the GOLD swap (F1e).
- **NEW R1 CHOICE:** No HSV cue. OCR is off by default (PC.json showed spurious numbers). The learned embedding is **NEW / UNVERIFIED behaviour** and does not reproduce yolo26n-reid.
- **WHY:** LAB is licence-free and owned. A generic embedding gives body and texture cues beyond kit colour, which matters for same-kit cases.
- **VALIDATION:** Synthetic tests use a `ScriptedAppearance` backend. Real discrimination, especially same-kit, is measured in **R2**, not claimed in R1.

### D05. Public states and mapping to the historical vocabulary
- **DESIGN DECISION:** The public states are TRACKING, OCCLUDED, AMBIGUOUS and LOST, plus an internal REACQUIRING. The historical online `state` is written as follows:

  | R1 state | Historical `state` | Legacy state it maps to | Evidence for that mapping |
  |---|---|---|---|
  | TRACKING | ACTIVE | TRACKING | VERIFIED_OUTPUT |
  | OCCLUDED | OCCLUDED | OCCLUDED | VERIFIED_OUTPUT |
  | AMBIGUOUS | AMBIGUOUS | OCCLUDED | VERIFIED_OUTPUT |
  | LOST, predicted box inside the image | TEMPORARILY_LOST | LOST | DOCUMENTED |
  | LOST, predicted box outside the image | OUT_OF_FRAME | LOST | INFERRED via the `.get` default |
  | REACQUIRING | REIDENTIFYING | REACQUIRING | DOCUMENTED |

- **EVIDENCE SOURCE:** R0_TRACKER_INTERFACE_MAP §5.2 and §5.4.
- **EVIDENCE LEVEL:** As listed per mapping.
- **HISTORICAL BEHAVIOUR:** The 7 online states mapped to 4 legacy states. Pass 2 trusts only TRACKING.
- **KNOWN HISTORICAL DEFECT:** None in the table itself.
- **NEW R1 CHOICE:** Invariant: every non-TRACKING R1 state maps to non-TRACKING. R1 never emits PROVISIONAL for the target.
- **WHY:** Downstream safety. Possession, receiver and duel logic all gate on TRACKING.
- **VALIDATION:** `test_surviving_consumers.py` runs the unmodified `event_adapter` with a **test-only** `NEW_TO_LEGACY_STATE` stub that holds exactly the evidenced mappings above. The OUT_OF_FRAME → LOST row is pinned as INFERRED.

### D06. Observed vs held vs predicted
- **DESIGN DECISION:** Each selected record carries `v2.observation_source ∈ {DETECTION, HELD, PREDICTED, NONE}`, `bbox_is_observed`, `bbox_is_held`, `bbox_is_predicted`, `held_age_s`, `last_confirmed_frame`, `time_since_confirmed_s` and `observed_or_contested_box`. The historical `bbox` keeps its historical meaning (last confirmed box while not observed). `observed_position_box` and `position_source` keep their historical vocabulary.
- **EVIDENCE SOURCE:** SEG_D Stage B; GOLD.json frame 86 (bbox held, observed_position_box predicted); F5, F6.
- **EVIDENCE LEVEL:** VERIFIED_OUTPUT + DOCUMENTED.
- **HISTORICAL BEHAVIOUR:** The distinction existed in Pass 1 but was dropped by `event_adapter`.
- **KNOWN HISTORICAL DEFECT:** Pass 2 searched around a stale held box and admitted the target's own body as a receiver.
- **NEW R1 CHOICE:** Explicit, redundant flags, plus an invariant: `position_source == CONFIRMED_DETECTION` ⇔ `bbox_is_observed` ⇔ `rb_state == TRACKING`.
- **WHY:** R1 prompt §11 requires that Pass 2 never needs to infer staleness.
- **VALIDATION:** A synthetic test case: a held box is never reported as observed. A contract-level invariant check also runs on every output.

### D07. Logical target identity vs tracklet identity
- **DESIGN DECISION:** One **logical target** is kept for the whole run. The `track_id`, `identity_id` and `online_track_id` of the selected record stay `T001` (approved by Nathan). Separate fields carry the detection-level identity: `v2.observation_tracklet_id` (the population tracklet that supplied the box this frame) and `v2.target_episode` (it increments at each loss → reacquisition). The summary gains `reacquisition_events`, one entry per event: `{lost_from_frame, uncertain_from_frame, reacq_started_frame, reacq_accepted_frame, uncertain_duration_s, evidence: {...}}`.
- **EVIDENCE SOURCE:** R1 prompt §12; R0 plan §2.5; historical selected-id changes (blind_r2_009 T001→T064; shot_close_range T001→T076).
- **EVIDENCE LEVEL:** DOCUMENTED (historical behaviour); decision approved by Nathan.
- **HISTORICAL BEHAVIOUR:** After a loss, `selected_player` moved to a new online id, and the offline resolver judged the boundary.
- **KNOWN HISTORICAL DEFECT:** The resolver saw nothing within a run (F12a), and it is now missing.
- **NEW R1 CHOICE:** Keep T001, record discontinuities explicitly, and never hide them. `identity_trusted` is false on every frame from loss until acceptance.
- **WHY:** One logical target simplifies Pass 2. The explicit episode and event records preserve auditability.
- **VALIDATION:** Synthetic tests for correct reacquisition and for refusing a wrong candidate. The events list must match the state sequence.

### D08. Contrastive identity, with no fast path
- **DESIGN DECISION:** A detection can become the target's **observed** box only if:
  1. target-appearance similarity ≥ `appearance_floor`;
  2. target-similarity minus the best competitor-similarity on that same box ≥ `contrastive_margin`;
  3. the motion veto does not fire;
  4. it is not merged-suspect;
  5. the margin over the second-best target candidate ≥ `ambiguity_margin`.

  This applies **even when only one box is in the gate**.
- **EVIDENCE SOURCE:** F1; TASK3M_MOTION §2 (fast path); TASK3M_FIX_LAYER_OPTIONS (contrastive tests safe; absolute self-consistency unsafe, 65 false flags on duel_lost_01).
- **EVIDENCE LEVEL:** DOCUMENTED + VERIFIED_OUTPUT (IPR capture).
- **HISTORICAL BEHAVIOUR:** The identity gate was skipped when exactly one colour-plausible box was within 45 px.
- **KNOWN HISTORICAL DEFECT:** At frame 268 a box was confirmed with appearance 0.289, below the 0.30 floor.
- **NEW R1 CHOICE:** No fast path. The competitor bank for a box comes from the population tracklet that owns it. When no competitor is known, the contrast is against the **pre-contact appearance of any competitor that overlapped the target in the last `contact_memory_s`**, if one exists.
- **WHY:** In an occlusion, the lone visible box is often the occluder.
- **VALIDATION:** Synthetic tests: opponent crossing, merged box, competitor present from birth, two equally plausible identities.

### D09. Association weights
- **DESIGN DECISION:** R1 does **not** use a single fixed weighted sum as the decision rule. Appearance and contrast are gates (D08). Motion and scale act as **vetoes and tie-breakers only**, and can never on their own overturn an appearance preference.
- **EVIDENCE SOURCE:** R0_CONSTANTS. Fitted weights: appearance 0.40, motion 0.20, IoU 0.10, colour 0.15, size 0.05 (**HISTORICAL_REFERENCE**, VERIFIED_OUTPUT). Floor 0.30 and margin 0.08 (**HISTORICAL_REFERENCE**, DOCUMENTED).
- **EVIDENCE LEVEL:** VERIFIED_OUTPUT / DOCUMENTED.
- **HISTORICAL BEHAVIOUR:** A linear sum; geometry terms totalled 0.50 against appearance 0.40.
- **KNOWN HISTORICAL DEFECT:** After a wrong confirmation, the self-reinforcing geometry outvoted ReID at 270–272 (F1e).
- **NEW R1 CHOICE:** `association_score` is still reported for audit. It is computed with NEW weights in `config.py`, each marked NEW, and decisions follow D08. The historical numbers appear in `config.py` only as commented HISTORICAL_REFERENCE values.
- **WHY:** R1 prompt §21 and §29.
- **VALIDATION:** A synthetic test: after 3 wrong-geometry frames, appearance still wins or the state goes AMBIGUOUS.

### D10. Competitors: every detection-backed person counts
- **DESIGN DECISION:** `population.py` keeps lightweight tracklets for **all** non-target detections, associated by IoU and motion only. They make no identity claims, have no promotion stage and have no team gating. Each has a short appearance bank (for contrast) and a predicted box (for merged-box and occluder logic). Any tracklet whose predicted box overlaps the target gate is a competitor **from its first frame**.
- **EVIDENCE SOURCE:** F4; TASK3M Part 11 item 1; TASK3N Part 6.
- **EVIDENCE LEVEL:** DOCUMENTED.
- **HISTORICAL BEHAVIOUR:** PROVISIONAL tracks were invisible to association and contests.
- **KNOWN HISTORICAL DEFECT:** #4's PROVISIONAL track could not protect its own detection.
- **NEW R1 CHOICE:** As above. `overlap_contests` lists **every** pair that touches the target.
- **WHY:** R1 prompt §24.
- **VALIDATION:** A synthetic test: a competitor born the same frame it overlaps the target still blocks.

### D11. Merged-detection guard
- **DESIGN DECISION:** A box is `merged_suspected` if any of these hold:
  - its area exceeds `merged_area_ratio` × the target's recent median area;
  - its aspect ratio is outside `merged_aspect_band` relative to the recent median;
  - it overlaps the predicted boxes of the target **and** a competitor, each ≥ `merged_overlap_min`;
  - an adjacent competitor loses its detection in the same frame.

  Result: never TRACKING. The state is OCCLUDED if the target prediction lies inside the box, otherwise AMBIGUOUS. There is no bank update.
- **EVIDENCE SOURCE:** F2 (GOLD 266: area ×3.1, one box for two players).
- **EVIDENCE LEVEL:** DOCUMENTED + VERIFIED_OUTPUT (IPR detections).
- **HISTORICAL BEHAVIOUR:** The merged box was accepted as the target.
- **KNOWN HISTORICAL DEFECT:** F1 trigger.
- **NEW R1 CHOICE:** As above. All thresholds are NEW and live in `config.py`.
- **WHY:** R1 prompt §26.
- **VALIDATION:** A synthetic test for a merged target/opponent detection.

### D12. Appearance-bank hygiene
- **DESIGN DECISION:** The target bank is multi-view and capped at N samples (diversity-pruned). It is updated **only** when all of these hold:
  - state TRACKING;
  - not merged-suspect;
  - no competitor overlap ≥ `bank_contact_iou`;
  - not within `post_reacq_quarantine_s` of a reacquisition;
  - crop quality valid.
- **EVIDENCE SOURCE:** TASK3M_REID §6 (the clean Fix-C bank enabled recovery).
- **EVIDENCE LEVEL:** DOCUMENTED + VERIFIED_CODE (`IdentityBankV2.consider` signature).
- **HISTORICAL BEHAVIOUR:** A quality-gated bank refused samples from 262 to 273.
- **KNOWN HISTORICAL DEFECT:** None in the bank. It was the part that worked.
- **NEW R1 CHOICE:** The same principle, with explicit, testable conditions.
- **WHY:** R1 prompt §20.
- **VALIDATION:** A synthetic test: the bank is not updated from a contaminated frame.

### D13. Units: time in seconds, geometry in body heights
- **DESIGN DECISION:** Every time limit is in **seconds** and converted using the video's own fps. Every distance or gate is in units of **target height h** (the median height of recent observed boxes). The gate grows while the target is unobserved but is **capped**, and does **not** grow while AMBIGUOUS.
- **EVIDENCE SOURCE:** F16 (90/35/350 px gates across 640×360–2056×1144); F6 (unbounded gate growth during AMBIGUOUS).
- **EVIDENCE LEVEL:** DOCUMENTED.
- **HISTORICAL BEHAVIOUR:** Pixel gates and frame counts.
- **KNOWN HISTORICAL DEFECT:** Relocks 234–619 px away.
- **NEW R1 CHOICE:** As above.
- **WHY:** R1 prompt §29 and §31.
- **VALIDATION:** `test_config_units.py`: the same scenario at 30 and 60 fps, and at 1× and 2× scale, gives the same decisions.

### D14. Hold and ambiguity limits
- **DESIGN DECISION:** OCCLUDED becomes LOST after `max_occluded_s`. AMBIGUOUS becomes LOST after `max_ambiguous_s`. Uncertainty (`v2.identity_confidence_rb`) decays with `held_age_s`.
- **EVIDENCE SOURCE:** F6 (AMBIGUOUS for 77 frames, then relock 234.8 px away); `occlusion_to_lost_frames = 10` (HISTORICAL_REFERENCE).
- **EVIDENCE LEVEL:** DOCUMENTED.
- **HISTORICAL BEHAVIOUR:** AMBIGUOUS never aged out.
- **KNOWN HISTORICAL DEFECT:** As above.
- **NEW R1 CHOICE:** Proposed defaults, NEW and to be tuned **only** on R2 DEV tracking labels, never on events: `max_occluded_s = 1.0`, `max_ambiguous_s = 1.0`.
- **WHY:** R1 prompt §31–32.
- **VALIDATION:** A synthetic test: AMBIGUOUS ages to LOST.

### D15. Reacquisition
- **DESIGN DECISION:** Reacquisition is a separate verifier. A candidate must pass the D08 tests with the **reacquisition margins** (stricter than ordinary association) for K of the last W seconds. No competitor may be a better match to the target bank in that window. The positional prior weight decays with time since the last confirmation and is **zero after a cut** or after `lost_long_s`. During verification the state is REACQUIRING; on acceptance a `reacquisition_events` entry is written.
- **EVIDENCE SOURCE:** F8, F10; R1 prompt §33–34.
- **EVIDENCE LEVEL:** DOCUMENTED.
- **HISTORICAL BEHAVIOUR:** Fix D re-entry used a 0.08 margin (HISTORICAL_REFERENCE). The old single-target tracker accepted after 3 hits regardless of absence length.
- **KNOWN HISTORICAL DEFECT:** No long-absence gate; relocks without conflict logging.
- **NEW R1 CHOICE:** As above. **A missed reacquisition is preferred to a wrong one.**
- **WHY:** As above.
- **VALIDATION:** Synthetic tests: correct reacquisition, wrong candidate refused, camera cut.

### D16. Camera motion and cuts
- **DESIGN DECISION:** `motion.py` estimates global image motion with sparse optical flow on background features, masking person boxes, and subtracts it from track motion. A hard cut, detected by frame-histogram distance plus flow-failure, sets `cut_detected`. After a cut, all position continuity is invalidated: the target goes to LOST and reacquisition is appearance-only. No pitch calibration in R1.
- **EVIDENCE SOURCE:** F11; IDENTITY_FEASIBILITY_GATE camera-cut rule; frame-log `camera_shift` / `is_cut` fields (VERIFIED_OUTPUT types).
- **EVIDENCE LEVEL:** DOCUMENTED.
- **HISTORICAL BEHAVIOUR:** `CameraMotionEstimator` existed; no cut was ever observed in surviving outputs.
- **KNOWN HISTORICAL DEFECT:** A possible missed cut on shot_close_range_01 (unresolved).
- **NEW R1 CHOICE:** As above. The historical `camera_shift` is filled in pixels; the v2 block gives it in h.
- **WHY:** R1 prompt §30.
- **VALIDATION:** A synthetic camera-cut test. Real pan and zoom are checked in R2.

### D17. Detector input resolution for small players
- **DESIGN DECISION:** The detector input size is a config parameter. Two options are measured in R1 on GOLD **only for detection recall of persons near the seed** (no identity tuning): full-frame at 640 and full-frame at 960/1280, or 640 tiles.
- **EVIDENCE SOURCE:** The GOLD human label (players about 60–80 px tall at 1280×720); TASK3M (#23's partial body was detected at conf 0.47–0.59).
- **EVIDENCE LEVEL:** DOCUMENTED.
- **HISTORICAL BEHAVIOUR:** The Pass-1 `imgsz` is UNKNOWN. Pass-2 diagnostics used 640, plus 960/1280 crops.
- **KNOWN HISTORICAL DEFECT:** Partial bodies were missed or recovered only by the low-conf local passes.
- **NEW R1 CHOICE:** Choose the smallest setting whose recall on the seed player's unoccluded frames is ≥ 0.95, and report it. The threshold is NEW.
- **WHY:** Detection misses look like occlusion and drive needless AMBIGUOUS/LOST states.
- **VALIDATION:** Recorded in R1_GOLD_TRACKING_COMPARISON.md.

### D18. Seed contract and refusal
- **DESIGN DECISION:** The inputs are `seed_frame` and `seed_bbox` `[x1, y1, x2, y2]` in pixels (BF:104-105). The run is refused if no raw detection has IoU ≥ `seed_min_iou` (NEW 0.5) with the seed at the seed frame. The video SHA-256 must match the fixture when one is supplied. No event fields are ever read.
- **EVIDENCE SOURCE:** F12; R0_TRACKER_INTERFACE_MAP §1.5.
- **EVIDENCE LEVEL:** VERIFIED_CODE + DOCUMENTED.
- **HISTORICAL BEHAVIOUR:** Seeds were sometimes copied from tracker output (pass_completed_01 is circular) or came from the wrong video (shot_close_range_01).
- **KNOWN HISTORICAL DEFECT:** As above.
- **NEW R1 CHOICE:** As above.
- **WHY:** R1 prompt §13.
- **VALIDATION:** `test_seed_and_provenance.py`.

### D19. Historical confidence fields
- **DESIGN DECISION:**
  - `identity_confidence` is written as `v2.identity_confidence_rb` when the target is trusted, and **forced below 0.5** whenever `identity_trusted` is false. This makes the surviving `actor_attribution` threshold (0.5, VERIFIED_CODE) behave safely.
  - `overall_track_confidence` (which becomes snapshot `tracking_confidence`) is set from the v2 confidence.
  - `detection_confidence` keeps the **historical quantised semantics** (1.0 / 0.5 / 0.0 by position source). The raw detector score goes to `v2.raw_detection_conf`.
  - The other historical confidence fields (`motion_confidence`, `appearance_confidence`, `reid_confidence`, and so on) are filled from the nearest v2 quantity and documented as **NEW semantics, not the historical formula**.
- **EVIDENCE SOURCE:** R0_DISCREPANCIES D8; R0_TRACKER_INTERFACE_MAP §4, §5.5; R1 prompt §36–37.
- **EVIDENCE LEVEL:** VERIFIED_OUTPUT (quantisation) / VERIFIED_CODE (consumer thresholds) / UNKNOWN (historical formulas).
- **HISTORICAL BEHAVIOUR:** Formulas UNKNOWN.
- **KNOWN HISTORICAL DEFECT:** Confidence fields measured track health, not physical identity (F13).
- **NEW R1 CHOICE:** As above, documented in R1_SCHEMA.md.
- **WHY:** The narrowest behaviour that keeps surviving consumers safe.
- **VALIDATION:** `test_surviving_consumers.py`: `actor_attribution.frame_identity` returns untrusted for every non-trusted R1 frame.

### D20. Video I/O and determinism
- **DESIGN DECISION:**
  - One seek to `seed_frame`, then **sequential** decode. No checkpoint or resume.
  - A per-frame content hash (64×36 grey-thumbnail SHA-1) is stored in `frame_log[].v2`.
  - ONNX Runtime runs with a fixed thread count and sequential execution mode.
  - JSON is serialised with sorted keys and fixed float rounding.
  - Determinism is checked by two runs and a byte compare. Any differing field is reported, never silently accepted.
- **EVIDENCE SOURCE:** F15; TASK3F §2 (random seek drift); R1 prompt §53 and §77.
- **EVIDENCE LEVEL:** DOCUMENTED + VERIFIED_OUTPUT.
- **HISTORICAL BEHAVIOUR:** A resume on .mov landed +10 frames off.
- **KNOWN HISTORICAL DEFECT:** As above.
- **NEW R1 CHOICE:** As above.
- **WHY:** Registration integrity and reproducibility.
- **VALIDATION:** A determinism test (synthetic), plus a real GOLD double run.

### D21. Dependencies
- **DESIGN DECISION:** Runtime dependencies are limited to `numpy`, `opencv` (cv2) and `onnxruntime`. All three are already present on the Mac VM (numpy 2.2.6, cv2 5.0.0, onnxruntime 1.23.2). There is no PyTorch, SciPy or Ultralytics at runtime. Assignment uses a small in-package Hungarian or greedy-with-margin implementation.
- **EVIDENCE SOURCE:** Environment check at the pre-coding gate (4 CPU, about 3 GB RAM, no torch).
- **EVIDENCE LEVEL:** VERIFIED (environment).
- **HISTORICAL BEHAVIOUR:** Ultralytics plus ONNX Runtime plus EasyOCR.
- **KNOWN HISTORICAL DEFECT:** Unpinned and partly unknown model provenance.
- **NEW R1 CHOICE:** As above. Model export (D22) is a one-off build step. It is saved as a script and may use PyTorch in the cloud workspace, but the output ONNX files are hashed on the Mac.
- **WHY:** Licensing, footprint and determinism.
- **VALIDATION:** An import-scan test.

### D22. Model acquisition and export
- **DESIGN DECISION:** Model files are downloaded **only after approval**:
  - official YOLOX `.pth` from the Megvii-BaseDetection release (or its official ONNX, if published there);
  - DINOv2 ViT-S/14 from facebookresearch.

  Each is exported to ONNX by a saved script `tools_rb/export_models.py`. Source and ONNX SHA-256 are recorded in R1_MODEL_AND_LICENSE_RECORD.md and pinned in `config.py`. Models live in `models/` (git-ignored) and in every archive snapshot.
- **EVIDENCE SOURCE:** R1 prompt §16, §54, §65.
- **EVIDENCE LEVEL:** DOCUMENTED (requirement).
- **HISTORICAL BEHAVIOUR:** Model provenance unrecorded (yolo26n-reid origin UNKNOWN).
- **KNOWN HISTORICAL DEFECT:** As above.
- **NEW R1 CHOICE:** As above.
- **WHY:** Reproducibility and licence traceability.
- **VALIDATION:** The detector refuses to load on a hash mismatch.

---

## Changes recommended to the R0 R1 architecture

1. **`detector.py` and `appearance.py` become backend interfaces**, with YOLOX-ONNX and composite (LAB + DINOv2-ONNX) implementations, plus scripted test backends (D03, D04). R0 assumed one detector and a ReID model.
2. **Logical target vs tracklet identity** is made explicit with `observation_tracklet_id`, `target_episode` and `summary.reacquisition_events` (D07). R0 only proposed `reidentified_this_frame`.
3. **Association weights demoted from decision rule to audit value**; decisions are gate-based (D08, D09).
4. **One extra tool**, `tools_rb/export_models.py` (D22). There is no new tracker module. The module list is otherwise unchanged: 10 files.
5. **Runtime dependencies restricted** to numpy, cv2 and onnxruntime (D21).
6. **Detector input resolution measured** for small players (D17).
7. **R0 errata carried forward.** R0_TRACKER_INTERFACE_MAP §8 bullet 4 is superseded by §5.2 (REIDENTIFYING → REACQUIRING, DOCUMENTED).

## Observation recorded (no decision)

A **royal-blue-kitted ball boy** stands by the asics board on the right of the GOLD frames 266–269 (visible on the label sheets). His kit colour is close to #23's.

This is a live example of a non-player and same-colour distractor. It supports D04/D08: colour is supporting evidence only. He must be handled as a competitor or ignored, never as the target.

# R1-E stage record: generic candidate population + motion

**Stage:** R1-E. Approved by Nathan on 2026-10-24 at 20:14 Sydney (resent at 21:31 after a session limit interrupted the first run). `R1/R1_GENERIC_TRACKER_REQUIREMENT.md` is binding.

**Built** (generic only: TARGET / PERSON_A / PERSON_B; no team, number, colour, clip or benchmark logic):
- `player_tracker_rb/population.py`: short-lived tracklets for every detected person, their lifecycle, duplicates and nesting, the merged-detection guard, contested links, hidden tracklets, overlap context, competitors and the appearance-bank contamination guard.
- `player_tracker_rb/motion.py`: a bounded, camera-compensated constant-velocity model for one body. It is a veto only.
- `player_tracker_rb/camera.py`: global camera motion (pan / zoom / roll) from background optical flow, and hard-cut detection.

**Not built:** no target identity, no target state machine, no reacquisition, no ball or event logic. The population never touches the target's appearance bank.

**Contracts:**
- The output contract (`output_schema.py`) is unchanged.
- The config registry is now `r1e-1`: 154 parameters, 33 of them new, all additions. The default config hash is `8882a812bbfe8a0484c73e06145e616ea325bccd543f627073f9de292c986cb4`. The full list is in `R1E_CONFIG_REGISTRY.csv`.
- No new models are used.

## Population model

**What a tracklet is.** A chain of detections that motion and geometry say is ONE physical body, with no gap in which another body could have taken its place.
- It makes no identity claim and has no promotion stage.
- It ends whenever geometry cannot guarantee continuity. Breaks are recorded as `possible_predecessors` / `possible_successors`.
- Linking identities across breaks is left to R1-F / R1-G, which will use appearance and contrast.

**Per frame** (all times in seconds and sizes in body heights or ratios; every threshold is in `config.py`):

0. **Camera.**
   - Continuity is broken by: CUT; UNKNOWN (by default); or FIRST_FRAME after the first frame, meaning the estimator was reset.
   - When broken, every live tracklet and merged region is closed (CAMERA_CUT / CAMERA_UNKNOWN / CAMERA_RESET), and remembered positions become invalid.
   - Otherwise every remembered box, size and region is moved by the measured transform.
1. **Duplicates and nesting.**
   - Two boxes with IoU ≥ 0.85 are one body detected twice. The lower score is kept as a record but is never a body.
   - A box ≥ 0.75 inside a larger box makes the larger one CONTAINS_OTHER, which is never an appearance sample.
   - If that inner box is also ≥ 0.3 of the larger box's height, it is NESTED: a part-body or alternative box of the same person.
   - Nothing is deleted for overlapping. Nested boxes stay observations and competitors.
2. **Association** (geometry only).
   - A match needs IoU(prediction, detection) ≥ 0.3 and must pass the motion veto.
   - A HIDDEN tracklet may re-associate only within 0.2 s, and only if no other body touched its predicted box during the gap.
   - Matching is greedy by IoU with deterministic tie-breaks.
   - An unassigned nested box, or a box containing an assigned nested box, is an ALTERNATIVE: it starts no tracklet.
3. **Merged detections** (D11). A box is never a tracklet when any of these hold:
   - **MULTI_CLAIM:** it covers ≥ 0.3 of two or more physically distinct, recently observed bodies, and each of them fills ≥ 0.25 of the box;
   - **AREA_RATIO / ASPECT_WIDE:** it fails the size tests against the ESTABLISHED size of the body it continues;
   - **PERSISTENT_MERGE / UNRESOLVED_MERGE:** it lies in a merged region that still holds more bodies than have come out of it.
   - A size reference is ESTABLISHED only once non-partial sizes span 0.2 s.
   - Merged regions RESOLVE when the bodies are seen apart again. They LAPSE after 0.5 s without evidence, and are forgotten 2 s after their last evidence.
4. **Contested links.**
   - A link is contested when a rival detection, or a rival tracklet, matches within 0.1 IoU of it.
   - Every tracklet in the contested group ends (CONTACT_AMBIGUITY), and each detection starts a CONTESTED tracklet whose predecessors are the ended tracklets.
   - So a crossing never produces a silent swap.
5. **Births.**
   - Hidden tracklets covered by the new box become possible predecessors. The hidden tracklet itself is not closed, because it may be someone else.
   - **MERGE_LINEAGE:** a tracklet born in an unresolved or lapsed region, or descended from one, may hold two bodies. It is never an appearance sample, for its whole life. Only a clean split clears lineage.
6. **Unmatched tracklets** become HIDDEN.
   - The box is PREDICTED for up to 0.3 s, then HELD. It is never observed.
   - The tracklet closes on leaving the frame (LEFT_FRAME) or after 0.5 s unmatched (UNMATCHED_TIMEOUT).
7. **Overlap context:** every overlapping pair of bodies, with their IoU and both cover fractions.
8. **Bank eligibility.** Only one kind of box may feed a tracklet's appearance bank. It must be:
   - observed, uncontested, unmerged, non-partial and not a duplicate;
   - not nested, not containing another box, and not merge lineage;
   - isolated: IoU ≤ 0.05 with every other body and ≤ 5% covered;
   - camera-verified;
   - at most one sample per 0.25 s.

   `sample_appearance(rel_frame, image, …)` checks the frame number and samples a tracklet at most once per frame.

**Competitors** (`competitors(box, h)`). Every body that overlaps the box, or lies within 1.5 h of it, is returned:
- observations: tracked or not, provisional or not, merged or nested;
- hidden tracklets, with PREDICTED/HELD boxes;
- recently closed tracklets, kept 2 s, with `last_box_source` and their appearance banks;
- after a cut, every remembered body, with position invalid.

## Motion model (`motion.py`)

- **State:** the last observed box (camera-compensated), a velocity in px/s and a log-height rate in 1/s.
- **Update:** exponential smoothing with α = 1 − exp(−dt/0.2 s), which gives the same behaviour at any fps.
  - Speed is clipped to `motion_veto_max_speed_h_per_s`·h and the log-height rate to ln(1.8)/s.
  - These observations update neither the velocity nor the log-height rate:
    - a PARTIAL box;
    - the first box after a partial one;
    - the first box after an uncompensated camera frame.
- **Prediction:** extrapolate for at most 0.3 s (PREDICTED), then HELD. A prediction is never DETECTION.
- **Veto:**
  - Displacement must be ≤ (0.25 + `max_speed`·dt)·h.
  - Height ratio must lie within [1/k, k], with k = 1.25·1.8^dt. This test is skipped for partial boxes.
  - The veto produces no identity score.

## Camera handling (`camera.py`)

**Method:**
- The frame is converted to grey at a 640 px working width.
- Background features are found with goodFeaturesToTrack, with person boxes (padded 0.1 h) masked in both frames.
- They are tracked with pyramidal Lucas–Kanade flow.
- A similarity transform is fitted with RANSAC.

**Bounds.** An estimate is rejected (UNKNOWN) if it exceeds any of:
- a shift of 3 frame-widths per second;
- a zoom of 3× per second;
- a roll of 0.5 rad per second.

**Photometric check** (added after review finding H2):
- The previous frame is warped with the fitted transform and compared with this frame over the background, with person boxes of both frames masked.
- The residual is `mean|a′−b′| / (mean|a′|+mean|b′|)`, computed after blurring and mean-centring.
- A residual ≥ 0.25 rejects the estimate (HIGH_RESIDUAL).

**Cut:** a grey-histogram Bhattacharyya distance ≥ 0.5 together with a flow failure.

**Statuses:** FIRST_FRAME, ESTIMATED, UNKNOWN or CUT. UNKNOWN and CUT both break continuity (fail closed). Rejected transforms are kept in `measured_matrix` for audit only.

## Merged / overlap handling

- **Merged boxes** are flagged MERGED_SUSPECTED with their reasons. Their claimants are closed (MERGED_CONTACT). They remain observations and competitors.
- **Duplicates** are suppressed only at IoU ≥ 0.85, and kept in the record. Genuinely overlapping people (tested at IoU ≈ 0.4) both stay tracked.
- **Nested part-boxes and alternative boxes** of one person do not become extra tracklets and do not create merges.
- **A far person inside a near person's box:**
  - if too small to be a part-box (< 0.3 of the height), both are tracked;
  - in either case the containing box is never a bank sample.
- **Occlusion is not a merge.** A large near person passing in front of a small far person is not flagged as a merge.

## Development-clip behaviour (real footage)

**Run:** `tools_rb/population_eval.py` on the Mac, over 180 frames with the R1-C saved person candidates (2×2 tiles).
- From the fixture only `video_sha256` and `seed_frame` are read.
- No labels, target seed, identity or event output are used, and nothing here measures accuracy. It shows behaviour only.
- The report records the code hashes, config hash, video hash and candidate-file hashes.

### Final run (`population_eval/gold_population_run.json`)

| Measure | Value |
|---|---|
| Camera | 179 / 179 ESTIMATED; photometric residual median 0.018, P90 0.033, max 0.062 (threshold 0.25, about 4× margin) |
| Tracklets started | 121 |
| Tracklet lifetime | median 0.53 s, P90 1.17 s, max 4.67 s |
| Closures | MERGED_CONTACT 31, CONTACT_AMBIGUITY 25, UNMATCHED_TIMEOUT 43, LEFT_FRAME 9 |
| Merged-suspected detections | 138 (MULTI_CLAIM 12, AREA 27, ASPECT 11, PERSISTENT 26, UNRESOLVED 94 frames) |
| Nested / alternative boxes | 289 / 261 |
| Observed tracklet boxes | 1,390 |
| Bank-eligible share | 0.52 |
| Provisional share | 0.33 |
| Competitors per observed body | median 3, P90 13 |
| Invariant errors | 0 |
| Runtime (camera + population) | median 6.7 ms per frame |

The contact sheet (`gold_population_run_contact_sheet.jpg`) shows the remaining merge decisions at the real player contact. The earlier false merges on isolated players are gone.

### Negative results (recorded, project rule 12)

The intermediate runs are kept with their JSON reports and contact sheets.

| Run | Hypothesis / change | Observed | Decision |
|---|---|---|---|
| PRE_FIX | First implementation (synthetic tests all passed) | **1,106 of the clip's detections flagged merged**, median tracklet 1 frame. Isolated players and the referee were "merged" (contact sheet) | Synthetic same-size people hid three real-footage failure classes |
| FIX1_occluder | Size tests only against the body a box continues; MULTI_CLAIM claimants must fill ≥ 25% of the box | 887 merged | Near/far occlusion fixed; the cascade of merged regions remained |
| FIX2_nested090 | Part-body boxes inside a larger box (≥ 0.9) are NESTED; size references must be established | 310 merged, 158 tracklets | The detector's alternative boxes (75–90% inside) still made false MULTI_CLAIMs |
| FIX3_nested075 | Nesting at 0.75 (measured: alternatives 0.75–0.9; adjacent people 0.33–0.62) | 184 merged, 108 tracklets | An over-wide (person + shadow) box hijacked the tracklet |
| FIX4_alternatives | Association may use any box of a nested group; the other boxes are ALTERNATIVE | 128 merged, 120 tracklets | Merges only at real contact |
| FIX5_review | Code review fixes (lineage, residual, partial, part-box ratio) | 128 merged; bank share 0.60 → 0.52 | Fail closed on lineage and containment, as intended |
| final | Test-audit fixes (region membership by containment, inlier denominator, residual masks both frames) | 138 merged, 121 tracklets | Accepted |

**Caution:** `population_nested_containment` = 0.75 was set from development-clip measurements. It is in-sample and UNVALIDATED.

## Independent review (project subagent policy)

### Code reviewer: 8 findings, all validated and fixed

| # | Finding | Fix (general rule) |
|---|---|---|
| H1 | A box from a merge that outlived the region hold became a clean tracklet that fed its bank | MERGE_LINEAGE (born in an unresolved or lapsed region, or descended from one: never sampled); regions lapse instead of vanishing |
| H2 | A cut between two shots with the same static broadcast overlay was ESTIMATED as "no motion", so continuity was kept | Photometric-residual check (normalised, scale-free) |
| M1 | A short box without an established reference was matched as partial but stored as full | Partial = shorter than max(prediction, last full height) everywhere |
| M2 | A near person first seen containing a small far person never got a tracklet | A nested box must be ≥ 0.3 of the host's height; CONTAINS_OTHER applies to any containment |
| M3 | `sample_appearance` could sample twice per frame, or on the wrong frame | The frame must match; one sample per frame |
| L1–L7 | Successors of older closed tracklets; `last_box_source`; camera reset; explicit NESTED refusal; no velocity update after uncompensated frames; config wording; eval fail-closed and input hashes | All fixed |

### Test auditor: 70 mutants, 23 survived the first suite, 4 real defects

**Defects** (all fixed):
1. Merged-region membership used IoU only, so a body coming out of a 3-person box was missed. It now also uses containment.
2. The camera inlier fraction counted masked-out points as failures. The denominator is now the tracked features.
3. The residual check masked the current frame's person boxes only. It now masks both frames.
4. Rule 28 data: a hidden predecessor can sit beside its own successor. The links are recorded and tested, and the identity layer must exclude them (an R1-F item).

**Tests:** the auditor's probe tests became `tests_rb/test_population_audit.py`. Re-mutation of the key survivors (contact bar, contested split, CONTESTED refusal, region body count, cut-invalidated memory, region pan, motion veto, lineage, residual, residual masks, region membership, partial reference, double sampling): **13 of 13 are now caught.**

## Tests

- **Mac:** **310 tests OK, 0 skipped**, with `SPANA_REQUIRE_MODELS=1` and `SPANA_GOLD_VIDEO` set (Python 3.10.12, onnxruntime 1.23.2, cv2 5.0.0). The output is in `R1E_TEST_OUTPUT_mac.txt`.
- **Cloud:** 310 tests run; 4 model tests are skipped because the models are absent there.
- **New test files:** `test_population.py`, `test_population_audit.py`, `test_motion.py`, `test_camera.py`, plus the helper `_synth_people.py`.
- **Required scenarios:**

| Scenario | Tests |
|---|---|
| Crossings | 36-case matrix with no-swap and coverage checks; contested splits |
| Same-kit people | Identical colours: banks sample only isolated moments |
| Merged boxes | Multi-claim, area, aspect, persistent, unresolved, 3-person, lineage |
| Partial detections | Height and edge cases, young tracklet |
| Short / long disappearance | Open space, contact during the gap, long gap, leaving the frame |
| Pan / zoom | Synthetic transforms and rendered end-to-end |
| Hard cut | Including a cut with a static overlay; memory invalidation |
| Duplicates | Duplicates vs genuine overlap; input-order independence |
| Provisional competitor blocking | Blocks from its first frame |

- **Also covered:** observed vs inferred invariants on every frame; bank contamination; fps and resolution invariance; candidate-order permutation; a 3-minute bounded-state run; input validation.
- **Regressions:** none. All R1-A, R1-B, R1-C and R1-D tests pass unchanged.

## Determinism

- No randomness; every ordering is explicit.
- Tests confirm that repeated runs give identical records and that permuting the candidate order gives the same tracklet partition.

## Known limits

- **No accuracy measurement.** There are no human tracking labels for population behaviour; R2 must measure identity-switch and coverage rates.
- **Thresholds in-sample or unvalidated.** All R1-E thresholds are NEW and UNVALIDATED. The nested-containment value was set on the development clip.
- **One clip, no real cut.** The real-footage check is one 6 s clip with no camera cut, so real cut detection and the residual threshold are validated only synthetically and against this clip's non-cut distribution.
- **Merge lineage is permanent per tracklet** (fail closed). Coverage of competitor appearance banks drops: the bank-eligible share is 0.52 on the clip.
- **The SPEED veto rarely fires**, because IoU gating already limits displacement. In practice motion acts mainly through the SCALE veto and the HIDDEN-gap rules.
- **Rule 28:** competitors include the target's own hidden predecessor unless the identity layer excludes it (data is provided; to be done in R1-F).
- **No pitch calibration**, as per D16.

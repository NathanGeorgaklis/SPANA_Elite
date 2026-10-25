# R1-H stage record: real-footage tracker validation against frozen human truth

**Stage:** R1-H. Requested by Nathan on 2026-10-26 at 07:07 Sydney.

**Locked decisions (Nathan, R1-H):**
- Strict trusted-identity rules are kept. Look-alike protection is not relaxed for coverage.
- Individual people are not fabricated inside merged boxes.
- `identity_trusted` is the authority for target attribution downstream.

**Not done in this stage:** no events were evaluated and no Pass-2 / event-engine code was written. No tracker parameter or tracker code changed: `player_tracker_rb/` is identical to R1-G `30f2e39`, and the config registry is unchanged (r1f-2, hash `8e3b9d67…`). Task 3R and the holdout were not touched.

## 1. What frozen human truth exists

| Truth | Content | Status |
|---|---|---|
| **GOLD human labels v1** (`R1/gold_label_sheet/GOLD_HUMAN_LABELS_v1.csv`, sha256 `490a8af1…`, frozen 2026-10-24, matched against its freeze record at evaluation time) | 41 frames (rel 70–110) on one clip. Per frame: target visibility, identity resolvable, merged/hidden, first clean reappearance, location relative to the occluder. **No boxes**, neither for the target nor for anyone else. | Usable. **Development** footage: it was already seen in R1-F and R1-G. |
| Any other clip | None. The R0 benchmark design (Part C: 13 scenarios, EVAL split, two annotators, boxes at ≥ 5 Hz) has not been labelled. Holdout and Task 3R are excluded. | Does not exist. |

**Consequence.** Real-footage validation in R1-H rests on 41 labelled frames of one development clip, without box truth. Metrics that need boxes are **UNKNOWN** on real footage, not zero:
- false confident identity on visible frames (wrong person);
- swaps;
- reacquisition correctness;
- box quality.

Those metrics are measured only on synthetic clips with complete box truth.

## 2. Evaluation tooling (NEW)

`tools_rb/tracker_eval.py` implements R0 Part B against a generic per-frame truth model:
- Truth fields: visibility, resolvable, merged, optional target and other-person boxes, clean reappearance.
- Label formats plug in through adapters; GOLD v1 is `gold_v1`. Benchmark column names live only in the adapter.
- The claim per frame is read from `v2.identity_trusted` (the authority). An observed but untrusted frame counts as an abstention (`OBSERVED_UNVERIFIED`), never a confident claim.
- It fails closed when:
  - the labels' hash differs from the freeze record;
  - the run used another video, seed frame or seed box than the fixture;
  - a label frame lies outside the run;
  - the contract is violated;
  - the report file already exists.
- A metric the truth cannot support is reported as UNKNOWN with the reason.

| Priority | Definition |
|---|---|
| 1. False confident identity | Trusted on a frame where the human target is not available (hidden, out of frame, unresolvable, or merged), **or** trusted box IoU < 0.3 with the human target box. The count is a number only if every trusted available frame had box truth; otherwise it is UNKNOWN, with the verified part as a lower bound. |
| 2. Identity swaps | Maximal runs of trusted frames whose box has IoU ≥ 0.5 with **another** annotated person. UNKNOWN without other people's boxes. |
| 3. Wrong reacquisitions | Per span where the target is not available: a trusted frame inside the span (premature lock), or a first trusted frame after it (searched only up to the next span) whose box has IoU < 0.5 to the target. UNKNOWN if any first trusted frame lacks box truth. |
| 4. Trusted on hidden / merged / unresolvable | Trusted frames where the human says the target is not available. |
| Coverage, abstention | Over **available** frames (visible or partial, resolvable, not merged): trusted share; share of each abstaining claim (AMBIGUOUS / OCCLUDED / REACQUIRING / LOST / OBSERVED_UNVERIFIED). |
| Reacquisition delay | Seconds from the target becoming available after a gap to the first trusted frame. If there is none: censored ("at least", by the window end or the next gap). "Missed" is true when the target was continuously available for ≥ 2 s, false when the next gap came sooner, and UNKNOWN when the window ended sooner. |
| B5 contact integrity | No trusted claim on the contact partner within ±0.5 s of a human-marked contact. PASS when nothing is trusted in those windows; UNKNOWN when trusted frames exist but partner boxes do not. |
| Box quality | Mean IoU of trusted boxes with the human target box; UNKNOWN without boxes. |

The evaluation constants (0.3, 0.5, 2 s, 0.5 s) are R0 Part B definitions. They are evaluation-only, not tracker parameters.

## 3. Real footage: GOLD (development only)

Run evaluated (by the clean evaluator commit `0daba3a`): R1-G GOLD run A (`R1/R1G/gold_pipeline/run_a_mo_tracking_results.json`, sha256 `9dde1440…`). It was produced by the clean commit `30f2e39`, whose tracker code is identical to this stage's, and its determinism was already shown in R1-G. Report: `R1/R1H/gold_eval/GOLD_R1H_TRACKER_EVAL.json`.

| Priority | Result |
|---|---|
| 1. False confident identity | **UNKNOWN** overall, with **0 verified**. Nothing is trusted on the 11 unavailable frames. The 10 trusted available frames (rel 70–79) have no box truth (status PARTIAL_NO_BOX_TRUTH). Not human truth: a visual check by Claude of the R1-G contact sheet shows all 10 boxes on the blue target. |
| 2. Swaps | **UNKNOWN** (no boxes for anyone). Trusted frames exist only in rel 70–79, before the contact. |
| 3. Wrong reacquisitions | **0** (verified: no premature lock inside the gap rel 82–92, and no trusted frame after it within the window). |
| 4. Trusted on hidden / merged / unresolvable | **0 of 11** (6 merged, 5 hidden or unresolvable). |
| R0 B5 GOLD collision regression ("never confident on the occluder in extract 262–274", i.e. rel 82–94) | **PASS**: no trusted frame at all in rel 82–94 (`claims_per_frame`). |
| R0 B5 contact integrity (±0.5 s around the 12 human contact frames rel 81–92, i.e. rel 70–107 clipped) | **UNKNOWN**: trusted frames 70–79 fall in the window and there are no partner boxes. |

**Coverage and abstention, over the 30 available frames:**

| Claim | Frames | Share |
|---|---|---|
| Trusted (CONFIDENT) | 10 | 0.333 |
| REACQUIRING | 16 | 0.533 |
| AMBIGUOUS | 2 | 0.067 |
| OCCLUDED | 2 | 0.067 |
| LOST | 0 | 0 |
| OBSERVED_UNVERIFIED | 0 | 0 |

- Abstention on available frames: 0.667.
- By visibility: VISIBLE 8/26 trusted; PARTIAL 2/4 trusted.
- Uncertainty when unavailable: 1.0 (all 11 frames OCCLUDED or AMBIGUOUS). Over the R0 B1 subset (3 frames marked identity-unresolvable): 1.0.

**Reacquisition delay:**
- The target was human-available again from rel 93 (the human "first clean reappearance").
- It was not trusted again inside the window (≤ rel 110): delay ≥ 0.6 s, censored by the window end. Whether the reacquisition was "missed" is UNKNOWN, because the window is shorter than 2 s.
- Outside the label window the tracker accepted a reacquisition at rel 120, 0.9 s after rel 93. Its episode stays **untrusted to the end of the clip** under the locked strict rule (the only competitor was the opponent, at similarity 0.75, below the 0.80 confusable floor). Identity there is not labelled.

**Diagnosis of the low post-contact coverage (first failing layer):**
- **Hypothesis 1: the held box follows the occluder after the split. Rejected.** The held box moves about 9–10 px/frame. That equals the estimated camera translation (pan), so it is a world-static hold, correctly compensated.
- **Finding:** the real target body (tracklet T040, "left of #4" in the human labels) passes the reacquisition tests intermittently from rel 95. Merged or rival boxes near it keep the verification window from completing until rel 120.
- **Class:** abstention under contact and opponent-only contrast. This is the cost of the locked strict rule, not an identity error. **No code change** (a coverage change would relax look-alike protection, which Nathan ruled out).

## 4. Synthetic footage with complete box truth

`tools_rb/r1h_synthetic_eval.py` → `R1H_SYNTHETIC_TRACKER_EVAL.json`. All 15 R1-G scenarios, scored by the same `tracker_eval.evaluate`, with harness truth: the target box, every other person's box, and merged / hidden status.

| Scenario | FCI | Swaps | Wrong reacq | Trusted on unavailable | Trusted coverage of available | AMBIGUOUS | OCCLUDED | LOST | REACQUIRING | Observed unverified | After the gap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| full_occlusion | 0 | 0 | 0 | 0 | 0.384 | 0.018 | 0.036 | 0 | 0.134 | 0.429 | not re-trusted (≥ 2.233 s; missed = True) |
| gap_self_competition | 0 | 0 | 0 | 0 | 0.215 | 0 | 0 | 0 | 0.19 | 0.595 | not re-trusted (≥ 2.067 s; missed = True) |
| hard_cut_impostor | 0 | 0 | 0 | 0 | 0.291 | 0 | 0 | 0 | 0.118 | 0.591 | no gap in the truth |
| leave_and_return | 0 | 0 | 0 | 0 | 0.228 | 0.013 | 0 | 0 | 0.19 | 0.57 | not re-trusted (≥ 2.0 s; missed = True) |
| newcomer_in_contact | 0 | 0 | 0 | 0 | 0.195 | 0.345 | 0 | 0.46 | 0 | 0 | no gap in the truth |
| opponent_crossing | 0 | 0 | 0 | 0 | 0.474 | 0.018 | 0.026 | 0 | 0.132 | 0.351 | not re-trusted (≥ 1.933 s; missed = UNKNOWN) |
| opponent_crossing_hidden | 0 | 0 | 0 | 0 | 0.482 | 0.009 | 0.018 | 0 | 0.134 | 0.357 | not re-trusted (≥ 1.9 s; missed = UNKNOWN) |
| pan_crossing | 0 | 0 | 0 | 0 | 0.825 | 0.018 | 0.026 | 0 | 0.132 | 0 | re-trusted correctly after 0.6 s |
| posed_crossing | 0 | 0 | 0 | 0 | 0.48 | 0.013 | 0.013 | 0.118 | 0.237 | 0.138 | not re-trusted (≥ 2.567 s; missed = True) |
| same_kit_crossing | 0 | 0 | 0 | 0 | 0.825 | 0.018 | 0.026 | 0 | 0.132 | 0 | re-trusted correctly after 0.6 s |
| same_kit_crossing_hidden | 0 | 0 | 0 | 0 | 0.839 | 0.009 | 0.018 | 0 | 0.134 | 0 | re-trusted correctly after 0.567 s |
| same_kit_crossing_merged | 0 | 0 | 0 | 0 | 0.855 | 0 | 0.009 | 0 | 0.136 | 0 | re-trusted correctly after 0.533 s |
| twin_appears | 0 | 0 | 0 | 0 | 0.195 | 0.345 | 0 | 0.46 | 0 | 0 | no gap in the truth |
| uncontrasted_return | 0 | 0 | 0 | 0 | 0.123 | 0 | 0 | 0 | 0.263 | 0.614 | not re-trusted (≥ 1.667 s; missed = UNKNOWN) |
| wrong_reacquisition | 0 | 0 | 0 | 0 | 0.947 | 0.053 | 0 | 0 | 0 | 0 | target never returns |

**Summary:**
- **0** false confident identities, **0** swaps, **0** wrong reacquisitions and **0** trusted frames on unavailable truth in all 15 scenarios. Box truth is complete (status COMPLETE). The trusted-box IoU is 1.0 because synthetic detections are ideal, so it says nothing about real box quality.
- **Reacquisition delay** (trusted): 0.53–0.60 s in the 4 scenarios where a confusable same-kit competitor allows verification.
- **Coverage cost of the locked strict rule.** Where the regained episode can only be contrasted against an opponent (or no one), the target is never re-trusted:
  - 4 scenarios are **missed reacquisitions** by R0 B3: available ≥ 2 s and never re-trusted;
  - 3 more are censored before 2 s;
  - in those spans the tracker still follows the target as OBSERVED_UNVERIFIED (35–61 % of available frames).
- Scenario-level synthetic coverage: 0.12–0.95.


## 5. Downstream rule

`R1/R1H/R1H_DOWNSTREAM_IDENTITY_RULE.md`:
- Physical events may exist while identity is untrusted.
- Target attribution requires `identity_trusted = true` on every frame of the actor window; otherwise the actor is UNKNOWN.
- Target stats count only events credited to the target.

**Measured gap in the surviving recovered `actor_attribution.assess_actor`** (unmodified). It credits TARGET_PLAYER at a ≥ 75 % trusted window fraction. On R1 synthetic output without a clip-level conflict, 15 of 83 credited windows contained untrusted frames. R1 marks those frames correctly; the gap is in Pass-2 aggregation and is listed as a requirement for the Pass-2 rebuild.

## 6. Independent review

Both reviews followed the project subagent policy: report only, never editing. Each finding was validated by the main agent before any fix.

**Code reviewer: 10 findings, all validated, all fixed** (commit `a035b69`):

| # | Finding | Fix |
|---|---|---|
| 1 (H) | FCI `count: 0` beside "no box truth" reads as zero errors. | Count is UNKNOWN unless every trusted frame could be checked; `verified_count` is the lower bound. |
| 2 (H) | Unverified reacquisitions were silently left out of the wrong count. | Count is UNKNOWN when any reacquisition is unverified; correct / unverified / not-reacquired are reported. |
| 3 (M) | Back-to-back gaps shared one reacquisition (double count; inflated "missed"). | The search stops at the next gap; "missed" needs continuous availability. |
| 4 (M) | A trusted frame inside a gap was not a wrong reacquisition. | New outcome WRONG_PREMATURE. |
| 5 (L) | Sparse truth would split one swap into several. | Contiguous truth is required (fail closed). |
| 6 (M) | B5 contact checks were missing, and the contact column was ignored. | Contact mapped; B5 check added (PASS / FAIL / UNKNOWN). |
| 7 (M) | Fail-closed gaps: freeze status, file name, fps, video_time_s. | All are refused. |
| 8 (M) | The report could not be reproduced from what it recorded. | It now records the evaluator commit and dirty flag, the tool hash, the fixture / freeze / labels / results hashes, and the command. |
| 9 (M) | The legacy-gap figure (15) came from a setting the scenario itself does not produce (its conflict flag is True). | The rule doc states this, and the enrichment caveat. |
| 10 (L) | GOLD "0" lacked abstention context; the R0 B1 uncertainty subset was not reported; the clean-reappearance value was not validated; the B2 check was incomplete. | All addressed. |

The reviewer's GOLD collision window "rel 92–104" was an arithmetic slip. Extract 262–274 is rel 82–94 (seed 180), and that is the window used.

**Test auditor:** 40 mutants on the reviewed code; 21 caught, 19 survived; 4 possible bugs. All validated and fixed (commit `0daba3a`):
- **Bugs fixed:** duplicate frame-log records silently overwritten (now refused); labels hashed and then re-read (now parsed from the hashed bytes); non-bool `identity_trusted` (now refused); a gap at the window start (now flagged).
- **New tests (`AuditGaps`)** for every surviving mutant and the missing edge cases.
- **Re-run of the same 40 mutants: 36 caught, 4 equivalent, 0 surviving.** The 4 equivalents: duplicate labels are refused by the contiguity check; missing or inconsistent provenance (3 mutants) is refused first by contract validation, so the scorer's own checks are a second layer behind it.

## 7. Tests

**Mac** (source of truth; python 3.10.12, onnxruntime 1.23.2, cv2 5.0.0; `SPANA_REQUIRE_MODELS=1` and the GOLD video set, so nothing is skipped): **492 tests, all OK**, in 5 chunks covering all 19 test modules: 384 + 35 + 21 + 4 + 48. Output: `R1H_TEST_OUTPUT_mac.txt`.

**New in R1-H:** `tests_rb/test_tracker_eval.py`, 48 tests at three levels:
- **unit:** one failure class per test, plus the audit gaps;
- **synthetic integration:** 6 full-pipeline scenarios scored against complete harness box truth, plus a planted swap the scorer must catch;
- **real footage:** frozen GOLD labels v1 and the committed GOLD output, with the fail-closed checks.

No existing test changed. No tracker code changed (`player_tracker_rb/` identical to `30f2e39`).

## 8. Negative results (project rule 12)

| Hypothesis | Change | Observed | Decision |
|---|---|---|---|
| After the GOLD contact, the target's held box drifts with the occluder and pulls the positional prior to the wrong person. | None (diagnosis only). | The held box moves at the camera translation rate: a correct world-static hold. | Rejected. No change. |
| Low GOLD post-contact coverage is a defect. | None. | The real target passes intermittently; verification is blocked by merged or rival boxes; the episode is then unverified (opponent-only contrast). | It is abstention under the locked strict rules. Not fixed. |

## 9. Known limits

1. **Real-footage evidence is one development clip, 41 labelled frames, no boxes.** Swaps, wrong-person FCI on visible frames, reacquisition correctness and box quality are UNKNOWN on real footage. GOLD was seen during development, so it is not an unbiased evaluation.
2. **Coverage under the strict rule is low after any contact with an opponent.** GOLD: 10/30 available frames trusted, and none after the contact. Synthetic: 0.12–0.48 trusted coverage where the regained episode cannot be verified.
3. **The legacy Pass-2 attribution does not meet the downstream rule** (trusted-fraction rule). It is documented as a rebuild requirement. The recovered code is not modified.
4. **No long real-footage run.** Long-match stability is shown only synthetically (R1-G, 1,800 frames).
5. Identity confidence is UNCALIBRATED.

## 10. R1 TRACKER VALIDATED

**R1 TRACKER VALIDATED: NO.**

**What is established:**
- The four error priorities are 0 on 15 synthetic scenarios with complete box truth.
- On real development footage (GOLD), every check the frozen truth can support passes: 0 trusted frames on 11 hidden, merged or unresolvable frames; 0 wrong reacquisitions; the R0 GOLD collision regression passes.
- No identity failure was found, so no tracker change was needed or made.

**Why it is not "validated":**
1. **Real-footage truth is too thin.** It is one clip, already seen during development, with 41 labelled frames and no boxes. Swaps, wrong-person false confident identity on visible frames, reacquisition correctness, B5 contact integrity and box quality are UNKNOWN on real footage. Project rules 25 and 31 forbid calling that validation.
2. **There is no held-out EVAL set.** R0 Part C requires new DEV/EVAL clips across scenarios S1–S13, labelled with boxes by two annotators, frozen before inference. None exist yet.

**What would change the answer:** the R2 benchmark of R0 Part C. At minimum: several new non-holdout, non-Task-3R clips covering same-kit and opponent crossings, merges, exits/re-entries and cuts, with human target and other-person boxes, frozen and hashed before any run, then scored with `tools_rb/tracker_eval.py` (a new label adapter). GOLD labels v2 with boxes would complete GOLD's unknowns, but GOLD stays development evidence.

## Persistence

- Commits:
  - `fbf6385`: tooling and first results, before review;
  - `a035b69`: code-review fixes;
  - `0daba3a`: test-audit fixes;
  - a final records commit (this file, test output, regenerated evaluation outputs).
- The final commit hash, GitHub verification, snapshot path and manifest hash are recorded in READ FIRST and the stage report. A file cannot contain its own snapshot hash.
- The evaluation outputs record their own provenance: evaluator commit `0daba3a` (clean), tool sha256, labels / freeze / fixture / results hashes, and the command.

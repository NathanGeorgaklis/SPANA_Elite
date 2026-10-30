# R2 DEV inference results (2026-10-30)

**Instruction (Nathan, 2026-10-30 15:31 Sydney):**
- Run the frozen clean tracker commit on the 12 DEV clips; do not modify tracker logic first.
- Do not run or score EVAL; use frozen labels and fixtures only.
- Score the 11 identity-scorable DEV clips; keep C019 excluded.
- Report UNKNOWN where the truth cannot support a metric; do not tune.
- Classify failures; fix nothing.

**EVAL was not run, opened or scored. No tracker logic or setting was changed. Nothing was tuned.**

## What was run

| Item | Value |
|---|---|
| Tracker code | `player_tracker_rb/` identical to R1-G `30f2e39`: `git diff 30f2e39 ef6d635 -- player_tracker_rb` is empty |
| Commit the runs were made from | `ef6d635` (clean tree; every result records `git_dirty: false`). It adds only the runner `tools_rb/r2_dev_run.py`, which reads the fixture (seed, video sha) and never the labels before the run, and refuses EVAL. |
| Config | default production config (registry r1f-2) |
| Models | YOLOX-s `c5c2d13e…8063`, DINOv2 ViT-S/14 ONNX `0ab62713…efea` (hashes verified against `R1/R1C` and `R1/R1D` before the run) |
| Clips | the 11 identity-scorable DEV clips from `R2_CLIPS_2026-10-26_0821` (archive manifest PASS); every clip sha256 = registry = fixture |
| Truth | the frozen `labels_v1` files, FROZEN records and fixtures in `R2/labels/` (frozen 2026-10-27 and 2026-10-30) |
| Scorer | `tools_rb/tracker_eval.py`, format `r2_v1`, unchanged |

**Environment (disclosed).** The runs could not complete on the Mac. A Mac command is limited to about 3 minutes, a background process there is stopped when the command ends, the tracker has no resume (by design), and 600 frames take about 14 minutes. The first Mac attempt is kept as `_r2_work/dev_runs_f37446a_ABORTED_runner_stopped_on_refusal` and `_r2_work/dev_runs_ef6d635` (partial, killed).

The runs were therefore made in the cloud workspace from the same commit, models and clips:
- Linux x86_64, 2 cores, **Python 3.13.17** (the Mac has 3.10.12);
- onnxruntime 1.23.2, numpy 2.2.6 and opencv-python-headless 5.0.0.93, installed to match the Mac exactly. The full freeze is in `ENVIRONMENT_pip_freeze.txt`.

**Cross-check.** The Mac reproduced the first two seed refusals:
- C006: best IoU 0.422 on the Mac vs 0.421 in the cloud;
- C016: 0.455 vs 0.454.

So the two environments agree, apart from a third-decimal detector difference.

## Per-clip results (DEV)

**Seven of the eleven scorable DEV clips were refused at the seed.** The tracker failed closed and produced no output, so it makes no claim on them: no error and no coverage.

| Clip | Outcome | Reason (tracker's own) |
|---|---|---|
| R2_MCFC_MIT_C006 | RUN_REFUSED | no detection overlaps the seed box by IoU ≥ 0.5 (best 0.421) |
| R2_MCFC_MIT_C016 | RUN_REFUSED | IoU ≥ 0.5 not met (best 0.454) |
| R2_MCFC_MIT_C034 | RUN_REFUSED | seed crop cannot anchor the appearance bank: PARTIAL, TOUCHES_IMAGE_EDGE |
| R2_MCFC_MIT_C038 | RUN_REFUSED | seed crop cannot anchor the appearance bank: ALTERNATIVE, NESTED, NOT_ISOLATED |
| R2_SYO_MAR_C004 | RUN_REFUSED | IoU ≥ 0.5 not met (best 0.458) |
| R2_SYO_MAR_C016 | RUN_REFUSED | IoU ≥ 0.5 not met (best 0.420) |
| R2_SYO_MAR_C032 | RUN_REFUSED | seed crop cannot anchor the appearance bank: NOT_ISOLATED |
| R2_SYO_MAR_C019 | EXCLUDED | rejected by the annotator ("ball not in play yet"); not run |

The four clips that ran (metrics exactly as `tracker_eval` computed them; frames counted after the seed):

| Clip | False trusted identity (P1) | Swaps (P2) | Wrong reacq. (P3) | Trusted on unavailable (P4) | Available frames | Trusted coverage | Abstention | Box IoU (frames) |
|---|---|---|---|---|---|---|---|---|
| R2_MCFC_MIT_C020 | 0 | 0 † | 0 | 0 | 569 | **0.376** | 0.624 | 0.710 (57) |
| R2_MCFC_MIT_C033 | 1 (frame 105) | 0 † | 1 (frame 105) | 1 (frame 105) | 417 | **0.249** | 0.751 | 0.642 (49) |
| R2_SYO_MAR_C012 | 0 | 0 † | 0 | 0 | 254 | **0.004** | 0.996 | 0.485 (1) |
| R2_SYO_MAR_C021 | 1 (frame 36) | 0 † | 0 | 0 | 78 | **0.654** | 0.346 | 0.400 (9) |

† The swap count is a lower bound. Only one trusted grid frame in all four clips (C012) had another annotated person to compare against, so **identity swaps are effectively UNKNOWN on R2 DEV**.

**Claims on available frames:**
- C020: CONFIDENT 214, LOST 295, AMBIGUOUS 38, OCCLUDED 22.
- C033: CONFIDENT 104, LOST 235, REACQUIRING 70, AMBIGUOUS 8.
- C012: CONFIDENT 1, LOST 223, AMBIGUOUS 30.
- C021: CONFIDENT 51, OCCLUDED 18, AMBIGUOUS 9.

**On unavailable frames** (out of frame, hidden, unresolvable):
- C033: 1 CONFIDENT of 182 (frame 105).
- C021: 0 of 221.

**Contact integrity (B5):**
- C012: PASS. One trusted frame was in a contact window; it was not on the partner.
- The others had no trusted frame in a contact window.

**Grid checks.** Every trusted frame on the box grid was checked. The checked fraction of trusted available frames was C020 0.27, C033 0.47, C012 1.0 and C021 0.18. Trusted frames between grid samples are counted but not box-checked (limit 2 of the R2 stage record).

## Aggregate DEV metrics

| Metric | Value |
|---|---|
| Clips scored / refused at the seed / excluded | 4 / 7 / 1 (C019) |
| False trusted identity (P1) | **2** (C033 f105 on an unresolvable frame; C021 f36 wrong box, IoU 0.269). Diagnosis below: neither is shown to be on another person. |
| Identity swaps (P2) | 0 found; **UNKNOWN** in substance (checked on 1 frame) |
| Wrong reacquisition (P3) | **1** (C033 f105, WRONG_PREMATURE; the same frame as P1 and P4) |
| Trusted on hidden / occluded / unresolvable / out of frame (P4) | **1** (C033 f105) |
| Trusted coverage, pooled over the 4 run clips | **0.281** (370 / 1,318 available frames); abstention 0.719 |
| Trusted coverage over all 11 scorable DEV clips (refused = 0) | **0.127** (370 / 2,921); abstention 0.873 |
| Reacquisition | 0 correct, 3 not reacquired (all C033), 1 premature. **Delay: UNKNOWN** (no reacquisition happened) |
| Box quality | mean IoU 0.655 over 116 trusted boxed frames. **This mostly measures box convention, not box error** (see F7) |
| HUMAN_UNCERTAIN | 0 frames (none labelled) |

The single frame C033 f105 is counted under P1, P3 and P4. **Distinct identity-error events: at most 2 frames.** Neither has been shown to be a box on another person.

## Failure classes (first failing layer, rule 2)

| Class | Clips | First failing layer | Evidence |
|---|---|---|---|
| **F1 Seed box-convention mismatch** | C006, C016, SYO C004, SYO C016 | target identity seeding: the interface between the human box and the detector box | `r2_seed_diag` (read-only) on C006 and C016: the detector **did** find the player. Its box lies wholly inside the human box (inside_seed = 1.0), with height 62–82% and width 55–68% of the human box, so the IoU is 0.42–0.46, under `seed_min_iou` 0.5. Across the trusted frames of the run clips the same holds: the tracker box sits inside the human box (median containment 1.0) with 0.38–0.73 of its area. **The human boxes are systematically looser than the detector's body boxes.** |
| **F2 Seed crop not isolated / at the edge** | C034 (PARTIAL, TOUCHES_IMAGE_EDGE), C038 (ALTERNATIVE, NESTED, NOT_ISOLATED), SYO C032 (NOT_ISOLATED) | target identity seeding: the bank-anchor rule | The human seed frames were labelled VISIBLE, resolvable, no contact, but the tracker's isolation rule refuses them. Abstention by design, at the cost of the whole clip. |
| **F3 Merged-guard false positives leading to permanent loss** | C020, C033, SYO C021 | population: merged-detection guard | MERGED_DETECTION_SUSPECTED is a reason on almost every untrusted available frame: C020 355 of 355 (from frame 245), C033 308 of 313, SYO C021 24 of 27. In SYO C012 it is 5 of 253. **No frame in any R2 truth is labelled merged**, and C020 has no contact or occlusion labelled at all. After the guard fires, the target ages out to LOST and is never trusted again. |
| **F4 Competing-candidate abstention that never recovers** | SYO C012 | target association: contrastive margin | Trust is lost 2 frames after the seed (COMPETING_CANDIDATE plus BELOW_APPEARANCE_FLOOR, during contact with 3 labelled people). It stays LOST for 223 of 254 available frames: coverage 0.004. Abstaining during contact is by design; never re-trusting after it is the failure. |
| **F5 No reacquisition after the target returns** | C033 | target reacquisition | The target was back in frame from f295 to f599 (5.1 s, missed reacquisition). The tracker sat in REACQUIRING with REACQUISITION_UNVERIFIED and later LOST. Over all clips: 0 correct reacquisitions. |
| **F6 One-frame trust lag into an unresolvable frame** | C033 f105 | target state transition | Human truth: f105 PARTIAL, partly hidden by an object, resolvable NO, with clean frames either side. The tracker stayed TRACKING (trusted) on f105 and dropped to AMBIGUOUS (OCCLUDER_OVERLAP) at f106. Its IoU with the human box fell 0.415 → 0.281 over f102–f105. Whether the box was on the target at f105 cannot be decided from the truth (the frame is unresolvable). |
| **F7 Box convention in scoring** | C021 f36 (and box quality overall) | evaluation interface (human vs tracker box size), not tracker identity | The P1 frame C021 f36 has IoU 0.269 (< 0.3) but the tracker box is **100% inside** the human box, with 0.38 of its area and 72% of its height. This is consistent with the right player in a looser human box, but has not been verified by a human. The count stays 1, as scored. The same effect depresses the box-quality IoU, and could hide a swap (0.5 IoU needed) or label a correct reacquisition WRONG. |

## Coverage trade-offs

- **The tracker is strongly precision-first on DEV.** There are at most 2 frames of questionable trusted identity (one an unresolvable instant, one a probable box-convention artefact). The price is 0.281 trusted coverage where it ran and 0.127 over all scorable DEV clips.
- **Most lost coverage comes from mechanisms that fire without the human-labelled condition** they guard against:
  - seed gating: F1, F2, 7 clips;
  - the merged guard: F3, with no merged frame in any truth;
  - non-recovery after abstention: F4, F5.

  These are generic mechanisms, not clip-specific. A change to any of them trades directly against priority 1–4 safety and must be judged against identity errors, not coverage alone.
- **R2 DEV cannot test swaps, merged bodies or contact identity in substance.** Other people are annotated on almost no trusted frames, and no merged frame exists. Zero swaps here is not evidence of swap robustness.

## Clips excluded or UNKNOWN

- **Excluded:** SYO C019 (rejected by the annotator).
- **No tracker output (RUN_REFUSED, fail closed):** C006, C016, C034, C038, SYO C004, SYO C016, SYO C032. Every metric except coverage (0) is UNKNOWN for these.
- **UNKNOWN on the run clips:**
  - identity swaps (1 checked frame);
  - reacquisition delay (no reacquisition);
  - true box precision (convention);
  - P1 on trusted frames between grid samples (counted, not box-checked).

## Recommendation

**RECOMMEND TRACKER CHANGES BEFORE EVAL: YES.**

1. 7 of 11 DEV clips produce no output, and coverage is 0.127 overall. Running EVAL now would spend the evaluation set mostly on seed refusals and lost tracks, and would say little about identity safety.
2. **Decide first, before any change, whether F1 and F7 are tracker problems or truth-convention problems.** Either change the seed rule (for example accept a detection mostly *inside* the human seed box, a general rule), or ask for tight body boxes on seeds. Either way, decide whether the scorer should also report a containment-based box check alongside IoU. This is Nathan's call; it changes the protocol, not the tracker.
3. Candidate tracker work, each needing its own regression test and DEV-only diagnosis, one change at a time (rules 10 and 13):
   - F3: why the merged guard fires on isolated players (first failing layer: population);
   - F4 and F5: recovery after abstention and verified reacquisition;
   - F6: one-frame state lag on partial occlusion by an object.
4. Before EVAL, R2 also needs truth that can test swaps and merges: other people boxed on trusted frames, and merged scenes. Otherwise EVAL can only show the same limits.

Nothing has been changed. Every item above needs Nathan's approval.

## Files

**In Git** (`R2/DEV_RUN_ef6d635/`):
- per-clip `*.r2_dev_eval.json` and `DONE.json`;
- `R2_DEV_EVAL_ALL.json` and `R2_DEV_AGGREGATE.json`;
- `run.log`;
- `seed_diag.json`;
- `ENVIRONMENT.txt` and `ENVIRONMENT_pip_freeze.txt`;
- `RAW_OUTPUTS_SHA256.txt`, which hashes every raw output.

**Archive and snapshot only** (57 MB, not in Git): per-clip `*_mo_tracking_results.json`, `*_rb_decision_log.jsonl` and `*_rb_video_provenance.json`, under `SPANA_Engine_Archive/_r2_work/dev_runs_ef6d635_cloud/`.

**Tools, new:**
- `tools_rb/r2_dev_run.py` (committed before the run);
- `tools_rb/r2_seed_diag.py` (read-only seed diagnosis);
- `tools_rb/r2_dev_report.py` (aggregation and a box-containment diagnostic; recomputes no metric).

**Disclosed.** The first draft of the aggregation script read label boxes at the wrong frame offset and the wrong tracker box field. The scorer's metrics were never affected. It was fixed before any number in this report was taken, and the draft output is kept in the archive as `R2_DEV_AGGREGATE_draft_frame_offset_bug.json`.

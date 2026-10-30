# R2 DEV remediation F3b: duplicate same-body tracklets (2026-10-31)

**Instruction (Nathan, 2026-10-31 05:05 Sydney):**
- Fix the population-layer failure where several tracklets can represent different detections of one physical person
  and later trigger MULTI_CLAIM / false merge handling.
- Diagnose generically and fix the first failing layer.
- Keep genuine overlapping people, same-kit people and real second persons partly behind the target separate and
  representable. Ambiguity must fail closed.
- Safety gates: no new confirmed wrong-person trusted identity, no silent swaps, no bank contamination, genuine
  two-person overlaps stay distinct.
- DEV only. Do not modify F5. Do not run EVAL.

**Outcome: F3b is diagnosed but NOT fixed.**
- Five generic population-geometry fixes were built. Each one, under independent review, had a geometrically identical
  unsafe twin: a real person whose boxes look exactly like the F3b pattern. Every twin produced a silent population
  swap, a wrong-person trusted identity or bank contamination that the baseline does not.
- All five were rejected, so the production code is unchanged. `player_tracker_rb/` is byte-identical to `37a109c`,
  the commit whose DEV run (`R2/DEV_REMEDIATION_37a109c/`) remains the current DEV result.
- What remains is the diagnosis, the measurements, the tools, 19 safety regression tests (all pass on the baseline)
  and 5 F3b reproducers (`expectedFailure`, the acceptance tests for a future fix).

**EVAL was not run, opened or scored.** No frozen label, fixture, holdout or Task 3R material was touched. The
review subagents were told not to open `R2/_spana_r2_work/`.

## Diagnosis (first failing layer: population births)

Traced on DEV C033 at frame 106 (`37a109c` decision log), then measured across all 8 DEV clips that run.

1. **Why the secondary tracklet is born.**
   - Near the right image edge, the detector returned two boxes for the target: a body box that had lost the head
     (`[1248,335,1280,537]`) and an upper-body box (`[1249,313,1280,399]`).
   - The upper box lies **0.74** inside the body box. That is below `population_nested_containment` (0.75), so it is
     not NESTED; its IoU is 0.29, so it is not a duplicate. Under population rule 5 (births), any other unassigned
     detection starts a tracklet, so it became T026.
2. **Evidence at birth.**
   - Both boxes were PARTIAL and TOUCHES_IMAGE_EDGE, and neither was NESTED or ALTERNATIVE.
   - The detector box of the full body had been shrinking at the top over frames 104–106 (311 → 330 → 335).
3. **What prevented association with the existing body.**
   - The body's tracklet (T005) took the body box by IoU.
   - A second detection in the same frame is by definition another body unless nested or a duplicate. The population
     has no notion of "a part box of an already represented body" between 0.5 and 0.75 containment.
4. **Why MULTI_CLAIM read them as two people.**
   - T026 was never detected again (hidden, frames 107–111).
   - At frame 112 one detection was covered by both T005's and T026's predicted boxes, each strong, and
     `_distinct_bodies` counted 2. The result was MULTI_CLAIM, T005 closed with MERGED_CONTACT, and a merged region
     followed.
   - Two tracklets co-observed as separate boxes are, in the current population, always two physical bodies.

**Frequency on DEV (`37a109c`, `tools_rb/r2_f3b_metrics.py` and `r2_f3b_births.py`).**
- 262 of 1,558 tracklet births were born ≥ 0.5 inside a body observed in the same frame.
- 223 frames had a MULTI_CLAIM merge; 95 of them had a claimant born ≥ 0.5 inside another claimant.
- There were 722 MERGED_CONTACT closures; 334 involved a part-born claimant.
- The target's own tracklet was closed as a merge 7 times.
- A visual check of 24 random part-born births (DEV frames, archived contact sheet) found both classes in the same
  0.55–0.77 containment band:
  - same-body upper/lower/part boxes;
  - genuine second people overlapping, partly behind, or in crowds and on benches.

## Rejected fixes (negative results, rule 12)

Each version was committed, pushed and reviewed independently. The reviewer compared every scenario against
`a7cc9ac`. Every failure was reproduced by me before acting, and each repro is now a test in
`tests_rb/test_part_tracklets.py`. They all pass on the baseline.

| Version | Commit | Rule | Found unsafe (reproduced) |
|---|---|---|---|
| v1 | (not committed) | A tracklet born ≥ 0.5 inside an observed body is UNPROVEN-distinct; unproven pairs count as one body everywhere | Silent population swap in the existing `test_short_gap_with_contact_starts_new_tracklet`: a walker crossing a hidden target took its box when it reappeared |
| v2 | `bc1606d` | MULTI_CLAIM skipped only when the OLDER tracklet keeps the box and the others are younger unproven partners, or no claimant took it | Round 1 HIGH: a full-height person first seen overlapping steps quickly in front. 9/18 population swaps; same kit trusted on the wrong person 5/6 with a contaminated bank. A union of an unproven pair escaped merge handling. Tracklet age says nothing about which body a box belongs to. |
| v3 | `e351504` | Only part-SIZED partners (< 0.75 of the host's height) form pairs | Round 2 HIGH: a crouching, farther or short-box real person stepping in front gave 26/72 swaps, same kit trusted 7/8 with a contaminated bank. A union with a short partner (1.375× area) gave no merge. MEDIUM: a momentarily tall host box made a whole person "short". |
| v4 | `46fe988` | The kept box must be the older body's own whole box at its established (median) single size, and every partner's predicted box must lie wholly inside that body's extent | Round 3 HIGH: a crouching person born partly inside moves wholly inside the target's box and stands up in front, which is exactly the F3b geometry. 36/54 swaps; same kit trusted at 30/15/10 fps with a contaminated bank. |
| v5 | (not committed) | The same geometry, but failing closed as a contested link (claimants close, new tracklet, no merged region) | The new tracklet born on the target's box carries straight on onto the person standing up in the identical box one frame later, a population swap. Only the baseline's merged region ("may hold two bodies") prevents it. |

**Conclusion.**
- At the population layer, "the target's part box vanished and its whole box remains" and "a real person wholly
  inside the target's box stood up in front of it" are the same boxes over the same time.
- No geometric or temporal rule separates them, and the baseline's conservative merge is the only handling that stays
  safe in both cases. Fixing F3b needs evidence the population does not have, such as the appearance of the part box
  against the body, or appearance-verified recovery after the merge (target/reacquisition layer, the F5 stage).
- This follows rules 3, 7 and 20: no fix that cannot be shown safe.

The rejected v5 production diff is kept as `REJECTED_V5_production_diff_vs_a7cc9ac.patch.txt`. v2–v4 are in the Git
history. The review repro scripts, aborted DEV run logs, synthetic evaluations and the DEV contact sheet are archived
in `SPANA_Engine_Archive/_r2_work/R2_F3B_review_repros_aborted_runs_diag.tar.gz` (SHA-256 below).

## Does C033's F5 remain? Yes, and F3b is not upstream of it

- When the target returns (f296 onward), its box is the clean tracklet T060: no lineage, not merged.
- Reacquisition rejects it on appearance: embedding 0.587 < 0.70 floor, composite 0.707 < 0.80 floor. A neighbouring
  tracklet (T064) is even more similar (0.721).
- The MERGED_DETECTION_SUSPECTED reason the target carries while LOST is a stale reason from the f112 merge; it does
  not gate reacquisition.
- So F5 in C033 is an appearance/reacquisition failure. It would remain even with F3b fixed. Since the tracker is
  unchanged, C033's DEV result is unchanged (P1 1, P3 1, P4 1, coverage 0.249).

## DEV pre/post

The tracker is byte-identical to `37a109c` (`git diff 37a109c -- player_tracker_rb` is empty), so the DEV results in
`R2/DEV_REMEDIATION_37a109c/` stand unchanged. Every F3b quantity is as measured above. No rerun is needed.

The three DEV reruns started on the rejected commits (`bc1606d`, `e351504`, `46fe988`) were each stopped when review
found an unsafe case; none completed a clip. Their logs are archived and they are not results.

## Pre-existing safety limits found by the review (present at the baseline, also at `a7cc9ac`)

- **Long two-person union:** when a union box lasts longer than the merged-region hold, the region lapses and the box
  becomes a tracklet holding both people (review `s5` n=30, `r2d` union-short n=30).
- **Nested crouching person:** a crouching person wholly inside the target's box from their first frame is NESTED and
  never gets a tracklet. If they stand up in front, the target's tracklet carries on onto them (review `r3a`, 81/81 at
  `a7cc9ac`).

Both should become tickets. They are the same physical-identity class as F3b.

## Tests

`tests_rb/test_part_tracklets.py` (new; `tests_rb/_synth_scene.py` gains optional part-body detections).

**Safety regressions (19, all pass on the baseline):**
- **TestGenuineSecondPeople:** overlapping people; full-height person partly behind; real two-person merge;
  near/far occlusion; partial player entering beside another; younger partner stepping in front; younger partner
  taking the host's box.
- **TestTargetLevel:** same-kit person first seen behind the target.
- **TestReviewRound1:** full-height person stepping in front fast, fps 10/15/30; union with a full-height partner;
  same-kit target level.
- **TestReviewRound2:** short person stepping in front (72-case sweep); same-kit short person at target level;
  partner born while the target box is momentarily tall; union with a short partner.
- **TestReviewRound3:** crouching person standing up in front (54-case sweep); same-kit at target level.

**F3b reproducers (5, `expectedFailure` with the F3b reason):**
- full + part box of one person;
- part box alternating over 2 s;
- duplicate tracklets converging on one detection;
- host size change with a live part tracklet;
- target tracking through its own part boxes (ends LOST at the baseline).

## Known limits / new unknowns

- F3b (95 of 223 DEV MULTI_CLAIM frames) remains. It is a coverage problem and permanent target poisoning when it hits
  the target (7 target-merge closures on DEV). It is not a safety problem: the baseline abstains.
- UNKNOWN: whether an appearance-based part test (is the part box's crop consistent with the body's own appearance
  bank?) can separate same-body parts from same-kit people. Same kit is exactly where appearance is weakest (R1-D
  same-kit proxy 14.7%).

## Archive

`SPANA_Engine_Archive/_r2_work/R2_F3B_review_repros_aborted_runs_diag.tar.gz` sha256 `4b8d584c66a23d9dc50801a7429cfa3d30876caeca4294bdbf1e6bec974cc8e9`.

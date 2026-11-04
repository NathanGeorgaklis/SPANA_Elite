# Jersey + shorts number reader evaluation: pre-registration v1 (2026-11-04)

**Instructions:**
- Nathan, 2026-11-04 18:44, **refined at 18:52**:
  - **the jersey number is the primary number source**;
  - **the shorts number is optional supporting evidence only**, used only when genuinely present and readable;
  - many players have no shorts number, or it is not visible, and **its absence never counts against identity**.
- Also: four regions inspected independently; never assume shirt = shorts unless both are readable; temporal
  aggregation; one reading never proves identity; unreadable or absent regions abstain; non-player numbers never count.
- The 18:52 refinement supersedes the 18:44 "equal identity evidence" wording; this plan follows 18:52.

**Status: PLAN ONLY.** No reader has been run on any footage. This document is fixed before any reader output exists.
Changing a rule after seeing output requires a new version of this file and marks the DEV result as tuned.

**Scope:**
- Only the 15 DEV clips of the jersey dataset. Never EVAL, the frozen 25-clip R2 benchmark clips, Task 3R or the
  holdout.
- No tracker: every crop comes from a **human box**:
  - the frozen v1 target box;
  - the v2 competitor boxes.

## 1. Truth

- **Dataset v2** = frozen v1 (`R2/labels/jersey_dataset/*.jersey.v1.FROZEN.json`, dataset v1 sha `1d9669dd…4e15`)
  + the frozen v2 region layer (`*.regions.v2.FROZEN.json`, schema `r2_jersey_regions_v2`).
- v1 is never changed.
- The dataset v2 sha256 comes from `tools_rb/r2_jersey_regions.py record` and is quoted in every result.

**Truth units:**
- **Person-keyframe:** the selected player, or a competitor, on a 0.5 s keyframe where the person is VISIBLE / PARTIAL
  and not HUMAN_UNCERTAIN.
- **Region unit:** one of JERSEY_BACK, JERSEY_FRONT, SHORTS_LEFT, SHORTS_RIGHT on a person-keyframe.
  - States: READABLE n / UNREADABLE / NOT_VISIBLE / NO_NUMBER.
  - SHORTS_LEFT / RIGHT mean the **image** left / right half of the person, as seen on screen.
- **Numbers:** each person has a shirt number and a shorts number, recorded separately. They are never assumed equal.

## 2. Readers

- Each candidate reader is pinned before it runs: name, source, exact version, sha256 of every weight file, licence,
  runtime, preprocessing and output interpretation.
- **First candidate:** EasyOCR 1.7.2, Apache-2.0.
  - Detector `craft_mlt_25k.pth` sha256 `4a5efbfb…db17`.
  - Recogniser `english_g2.pth` sha256 `e2272681…29e8`; its training data is UNKNOWN.
  - Character allowlist: digits.
- **Interface:** `read(person_crop) -> [(text, confidence, text_box_in_crop)]`.
- Preprocessing is recorded with the run and never changed per clip.

## 3. From reader output to region readings (fixed rules)

1. **Crop:** the human person box, clipped to the image.
   - A box shorter than `jersey_min_box_h_px` (80 px, config) is not read: ABSTAIN by policy.
     - This is a documented pixel exception to rule 15: the limit is the OCR input resolution, the smallest digit
       strokes the model can resolve, not a scene scale.
     - Policy abstentions are reported separately and are **not** included in the ABSTAIN rate.
2. **Valid reading:** 1–2 digits, no leading zero, value 1–99, confidence ≥ `jersey_min_read_conf` (0.5, config,
   uncalibrated). Anything else is discarded.
3. **Region assignment** uses the text box centre (cx, cy), normalised by the crop width and height:
   - **JERSEY band:** 0.10 ≤ cy < 0.55.
   - **SHORTS band:** 0.55 ≤ cy < 0.80. **SHORTS_LEFT** if cx < 0.5, **SHORTS_RIGHT** otherwise.
   - **Anywhere else:** UNASSIGNED, discarded and counted.
   - Why these bands: standing-body proportions (torso in the upper half, shorts at about 55–80 % of height). They are
     ratios, so scale and resolution do not matter. They are uncalibrated, fixed here, and never tuned on output.
4. **Back or front:** the reader cannot tell them apart.
   - A JERSEY-band reading is CORRECT if it equals the number of any READABLE jersey region. Truth forces both to equal
     the shirt number, unless a target region carries a V1_CONFLICT note, which is reported.
   - If neither jersey region is READABLE, the reading is a FALSE_READING.
   - Results are also broken down by which side was readable.
5. **Several different valid readings in one region** of one person-keyframe: that region ABSTAINS, and the conflict
   is counted. Separate text boxes are never merged: OCR splitting "10" into "1" and "0" gives a conflict and an
   abstention, never a constructed number.
6. **Geometry caveat:**
   - The bands assume an upright full body. On PARTIAL person boxes the region assignment is unreliable, so
     PARTIAL person-keyframes are reported separately in every table.
   - For the identity decision (section 6), SHORTS_LEFT and SHORTS_RIGHT form one pool, so a left / right mix-up
     cannot change it.
7. **Non-player numbers:**
   - A reading whose text-box centre lies inside a labelled distractor box (board, bib, scoreboard, advertising, other)
     never counts. It is discarded and counted as `distractor_hits`.
   - An unlabelled non-player number that reaches a region is scored like any other reading (a wrong or false
     reading).

## 4. Per-unit outcomes

| Truth of the region | Reader gives n | Reader gives nothing |
|---|---|---|
| READABLE m | CORRECT if n = m, else WRONG | ABSTAIN (missed) |
| UNREADABLE / NOT_VISIBLE / NO_NUMBER | FALSE_READING | ABSTAIN (correct abstention) |

**Combined (jersey + shorts) per person-keyframe:**
- **CORRECT:** at least one reading, and every reading equals its own region's truth.
- **WRONG:** any WRONG or FALSE_READING among its regions.
- **ABSTAIN:** no readings.

## 5. Metrics, each reported separately for JERSEY-only, SHORTS-only and COMBINED

- **CORRECT, WRONG, ABSTAIN** counts and rates over truth-readable units.
- **FALSE_READING** count and rate over truth-unreadable or absent units (UNREADABLE, NOT_VISIBLE and NO_NUMBER, each
  separately).
- **Shorts accuracy** is computed **only over shorts regions where a number is present and readable** (truth
  READABLE). Shorts with NO_NUMBER, NOT_VISIBLE or UNREADABLE enter only the false-reading counts, never the
  accuracy denominator, and never count against identity.
- **Shorts availability (truth):**
  - share of person-keyframes with a READABLE shorts region;
  - share with NO_NUMBER;
  - share NOT_VISIBLE;
  - number of persons with any shorts number at all;
  - per clip.
- **Precision** = CORRECT / (CORRECT + WRONG + FALSE_READING).
- **Coverage:**
  - human-readable coverage: truth;
  - reader correct coverage: share of person-keyframes with at least one CORRECT reading.
- **Breakdowns:**
  - by unreadable reason, clip, day/night, 60 vs 30 fps;
  - selected player vs competitor;
  - policy abstentions (box too small), UNASSIGNED and distractor_hits.
- **Every WRONG / FALSE_READING:** listed with its crop saved for human review. Review never changes truth.

## 6. Temporal aggregation and identity evidence

**Counting:** each keyframe contributes **at most one reading per pool**.
- If a keyframe's readings within one pool all agree, they count as a single reading.
- If they disagree, that keyframe enters the pool as a conflict.
- So "3 readings" always means 3 different keyframes, at least 1.0 s apart on the 0.5 s grid. The combined evidence
  never adds jersey and shorts readings together to reach the count.

**Per person per clip**, valid readings are pooled in time order into two pools:
- **J:** JERSEY band only;
- **S:** SHORTS_LEFT and SHORTS_RIGHT only.

**Consensus of a pool:**
- **CONFIRMED n:** at least `identity_number_min_reads` (3, config) readings of n, spanning at least
  `jersey_confirm_span_s` (0.5 s, config), and no reading of any other number in that pool.
- **CONFLICT:** two or more different numbers.
- **UNKNOWN:** anything else. One reading never confirms.

**Evidence against the selected player's shirt number E (frozen v1). The jersey is primary, the shorts only support:**

| J | S | Evidence |
|---|---|---|
| CONFIRMED = E | anything | **SUPPORT** (a disagreeing S is reported as `shorts_disagrees`, never used against identity) |
| CONFIRMED ≠ E | anything | **VETO** (jersey only) |
| CONFLICT | anything | UNKNOWN |
| UNKNOWN | CONFIRMED = E | **SUPPORT (shorts-supported)** |
| UNKNOWN | anything else, including no shorts number, NOT_VISIBLE, UNREADABLE, CONFLICT, or CONFIRMED ≠ E | UNKNOWN |

Why:
- Shorts can legitimately carry another number, and many players have none. So shorts never VETO, and absent,
  unreadable or different shorts never count against identity.
- A shorts number equal to E is still only support, never proof.
- Jersey-only evidence is the same table with S ignored. Shorts-only evidence uses the S column alone: SUPPORT if
  CONFIRMED = E, never VETO.

**Separation (selected player vs same-kit competitor):**

| | SUPPORT | VETO | UNKNOWN / CONFLICT |
|---|---|---|---|
| Selected player | correct | **false veto** | abstain |
| Competitor, numbers ≠ target's | **FALSE SUPPORT (critical)** | correct | abstain |
| Competitor sharing the target's number | expected reading; reported as proof that support ≠ identity | — | abstain |

**Clip-level wrong CONFIRMED** is judged per pool against that pool's own truth:
- J against the person's shirt number;
- S against the person's shorts number.

A person with no shorts number has no S truth, so any S CONFIRMED for them is wrong.

"Sharing the target's number" means a competitor whose **shirt or shorts** number equals the selected player's shirt
number E. It is reported per number source.

## 7. Decision criteria (fixed now)

**Q0. How often are shorts numbers actually available?**
- From truth alone, before any reader runs: the shorts-availability figures in section 5.

**Q1. Do shorts add useful readable coverage when present?**

Measured over the persons who have a shorts number. YES only if, combined against jersey-only:
- a) reader correct coverage of their person-keyframes rises by **≥ 10 percentage points**, or **≥ 2 more of these
  person-clips reach SUPPORT** (selected players) or a correct CONFIRMED; and
- b) **no new** WRONG CONFIRMED, FALSE SUPPORT or false VETO appears.

Also reported:
- the same gain over the whole dataset;
- the truth ceiling: person-keyframes readable only on the shorts.

**Q2. Is combined jersey + shorts evidence safe enough for target identity support (support / veto only, never
proof)?**

**The decision uses the combined evidence of section 6 only.** This is fixed now and is not chosen after the results.
The J and S CONFIRMED results are checked for wrongness against their own truth.

**Minimum evidence:** Q2 is decided only if the run has:
- at least **10 selected-player person-clips** with a non-UNKNOWN combined result;
- at least **3 competitor person-clips** with a readable number;
- at least **100 emitted readings** in the combined pool.

Otherwise the answer is **NO (insufficient evidence)**. Competitor separation is tested on J04, J10 and J14 (plus any
optional competitors), not on all 15 clips.

YES only if, on the 15 DEV clips:
- **0** FALSE SUPPORT on competitors, from jersey or shorts;
- **0** shorts-originated VETO (true by construction; it is checked anyway);
- **0** WRONG CONFIRMED (any person, any pool used for the decision);
- **0** false VETO on selected players;
- **0** readings inside labelled distractor boxes reaching a pool. The rule discards them, so this checks that
  `distractor_hits` were all discarded. Unlabelled non-player numbers can only show up as WRONG / FALSE readings,
  which the other conditions already count.
- combined per-reading precision ≥ 0.95, **and** its Wilson 95 % lower bound ≥ 0.90.

Otherwise NO.
- YES means safe enough to *design* tracker integration as support / veto evidence for Nathan's review. It is not
  validation: 15 DEV clips, two matches, and no EVAL.
- Wilson 95 % intervals are reported for precision and for the false-reading rate.
- Q1's thresholds are small-sample rules of thumb, fixed in advance. The raw counts are always reported with them.

## 8. Reproducibility record per run

- Git commit, dataset v2 sha256, registry sha256, reader pins (section 2), config hash, the command, the environment
  (pip freeze) and the output JSON sha256.
- Raw per-unit outputs are archived with a SHA256SUMS.

# R2 DEV F5: reacquisition (2026-11-01)

**Instructions (Nathan, 2026-11-01):**
- **01:35.** Improve correct reacquisition after gaps, contact and anomalies without weakening identity safety. Focus
  on C006, C020 and C033. DEV only. Diagnose before changing thresholds; do not simply lower appearance floors.
  Untrusted frames never update the bank. No new wrong trusted identities, swaps or bank contamination. Keep F3c
  unchanged.
- **01:56.** E1 is diagnostic only; do not adopt global-frame competitors.
  - C020: allow re-verification after an anomaly using sustained strong appearance + physical continuity + no
    plausible local competitor.
  - C033: keep abstaining if competitors score better.
  - C006: classify as camera/motion upstream; do not fix in F5.
  - Test C020-style success plus same-kit replacement counterexamples. Stop before EVAL.

**EVAL was not run, opened or scored.** No frozen label, fixture, holdout or Task 3R material was touched.

**Outcome:**
- A post-anomaly re-verification rule was built, reviewed, hardened and committed. It is **OFF by default**
  (`target_reverify_enabled = False`), so the default tracker behaves exactly as before F5.
- With the rule ON, two same-kit replacement cases still put the wrong person on a trusted identity in synthetic
  tests. They are recorded as `expectedFailure`: the true target walks away while missed by the detector, or stays
  hidden behind the box in lockstep. This breaks the "no new wrong trusted identity" gate, so turning the rule on is
  Nathan's decision.
- With the final rule ON, DEV is unchanged on every official metric: on both C020 and SYO C021, a target look-alike
  in the frame cancelled it during the anomaly.

## Diagnosis (first failing layer per clip, c67c738 decision logs)

Tool: `tools_rb/r2_f5_reacq_diag.py` (read-only, DEV only). It classifies every available, human-boxed frame on
which the target is not trusted. Output: `F5_DIAG_c67c738.txt`.

| Clip | Untrusted available boxed frames | First failing layer | F5 decision |
|---|---|---|---|
| C020 | 56 UNVERIFIED_EPISODE (262 frames TRACKING-untrusted overall) | **Target layer, verification policy.** An uncorroborated SIZE_ANOMALY on the target's own tracklet T002 (f215) broke the verified identity. The target came back by reacquisition onto the **same** tracklet (f292) as an unverified episode. No confusable competitor was ever near it, so the R1-G rule could never verify it. The returning crop matches strongly (composite 0.88–0.92; local competitors ≤ 0.60). | Re-verification rule (below) |
| C033 | 44 FAILED composite + embedding below floor; 6 not detected | **Appearance discrimination.** After the 3 s gap (f116–294) the true target's embedding is 0.35–0.68. Other people often score higher (T066 up to 0.72 / composite 0.79; an untracked box 0.71). True target leads by ≥ ambiguity_margin on only 20 of 62 frames. | Keep abstaining: lowering floors would reacquire the wrong person |
| C006 | 68 detected but not evaluated, 49 not detected | **Camera layer (upstream).** The similarity-model camera scale drifts strongly (×0.996/frame; cumulative 0.69 by f250) while people's heights stay level or rise (median tracked-height chain 1.16). The held target height reference falls from 85 px to 44 px (true ~100 px), so the gate excludes the true box (5–6 h away in wrong-h units, ~2.8 true heights) and size tests mark true-size boxes MERGED. When evaluated (f276–288), the true box T104 scores composite 0.81–0.82, level with a neighbour (T095 0.79–0.82). | Classified upstream, not fixed in F5 (see F8) |

**Camera-scale probe (F8, new).** `cam_scale_probe.py.txt` compares, per frame, the camera similarity scale, a
homography's local scale at people's feet, and the median frame-to-frame height ratio of tracked people.

| Clip | Camera estimate | People heights | Note |
|---|---|---|---|
| C006 | 0.66 | 1.09 | homography local scale 0.99 |
| C033 | 0.34 | 0.51 | homography worse, 0.27 |
| C021 | 0.81 | 0.95 | |
| C012 | 0.88 | 1.18 | |

The camera similarity scale is biased against people's actual image heights. A homography does not fix it in
general. This is a new upstream finding (**F8: camera scale drift corrupts the held target size and gate**). It
affects size priors while the target is unconfirmed. It causes abstentions (fail-closed), not wrong identities.

## Experiment E1 (diagnostic only, NOT adopted; negative result)

- **Hypothesis:** a confusable banked body anywhere in the frame, beaten by the margin, verifies an episode.
- **Change:** `E1_global_contrast_DIAGNOSTIC_ONLY.patch.txt`; local branch only, never on `main`.
- **Observed:** C020 trusted 214 → 446 frames. C006 58 (vs 57). C033 not completed.
- **Decision (Nathan 01:56):** not adopted. A distant, irrelevant person must not grant verification.

## The rule (final, `2084627` + logging `7df8045`; `player_tracker_rb/target.py` docstring)

**Watch.**
- When a SIZE_ANOMALY starts on the tracklet of a VERIFIED identity (the origin), a watch opens (ANOMALY).
- When the anomaly ends and breaks the identity, the watch moves to WAIT.
- If the target is reacquired onto the very same tracklet as an unverified episode, a RUN starts.
- Any of these ends the watch: LOST, a return onto another tracklet, a cut, or 3 s after the anomaly ended
  (`target_reverify_max_s`).

**RUN.** Verified only after 1.0 s (`target_reverify_window_s`) in which **every** frame:
- is TRACKING on the origin;
- passes the ordinary tests with composite ≥ `reacq_appearance_floor` (0.85; no floor lowered);
- has a clean box and a measured camera (ESTIMATED).

The run must also end with:
- ≥ 0.5 h of camera-compensated travel (`target_reverify_min_travel_h`);
- both vacated spots wholly vacated and checked clear (where the anomaly ended and where the run started).

**Cancel for good** (from the anomaly's start), on any evidence that another body could be the target:
- an unknown or confusable body near the origin box;
- any other body anywhere that passes `appearance_floor` against the target bank;
- a target look-alike at a vacated spot.

Far bodies only ever withhold; nothing far away verifies. Until verified, frames stay untrusted and the bank is
never updated. The cancel record names the body (rule 19).

## Validation

**Tests.** `tests_rb/test_reacquisition_f5.py` (20 tests + 2 `expectedFailure`, at 10/15/30 fps), with the rule
enabled through `TrackerConfig.with_overrides`:
- C020-style success (fails on `304f27e`), with checks on the re-verifying frame;
- still target (not re-verified: a recorded cost);
- unknown, confusable and look-alike bodies withhold;
- same-kit replacements: target detected, missed, moving, out of frame;
- stale origin; uncompensated pan; lock-out; wrong look-alike; long gap; camera cut;
- the rule is off by default;
- audit gaps: look-alike at the vacated spot, look-alike only around the anomaly, an unverified episode never
  laundered, expiry, flicker restart;
- residual risk (`expectedFailure`): the undetected target walking away; lockstep.

**Suite and synthetic eval.**
- Full suite (cloud, `2084627`): 655 passed, 9 skipped (environment: GOLD video / Playwright), 7 xfailed (5 F3b
  + 2 F5 residual).
- Synthetic tracker evaluation identical to F3c at every commit.

**Independent code review (`d68b0cd`).** Real regressions found and reproduced: wrong trusted identity on the
same-kit replacement with a contaminated bank, which `304f27e` never produces:
- (b) the real target detected and leaving the frame;
- (h) the real target seen for seconds and then leaving (stale origin);
- (e) the real target hidden by a third body;
- (f2) an uncompensated pan counted as travel;
- (i) a wrong re-verification locking the real target out as a known other.

All were fixed in `6a1ff14`. The residual (a, c: no visible evidence of the true target) cannot be told apart by
geometry or generic appearance. Hence OFF by default.

**Test audit (59 mutants on `6a1ff14`).**
- 45 survived, mostly because the conditions overlap.
- One real gap: a look-alike seen during the anomaly itself was not watched. Fixed in `2084627` by opening the
  watch at the anomaly's start.
- 5 tests were added for the main gaps.
- Remaining survivors are listed in `audit/` (test gaps, documented).

## DEV results (EVAL not run; baseline `c67c738`)

| | c67c738 (F3c) | final rule ON (`a64bf54`, local experiment) | default `7df8045` (rule OFF) |
|---|---|---|---|
| Initialised / refused | 8 / 3 | 8 / 3 | 8 / 3 |
| P1 false confident identity | 5 | 5 | 5 |
| P2 swaps | 0 | 0 | 0 |
| P3 wrong reacquisition | 1 | 1 | 1 |
| P4 trusted on unavailable | 1 | 1 | 1 |
| Trusted / available (run clips) | 818 / 2105 (0.389) | 818 / 2105 (0.389) | 818 / 2105 (0.389) |
| Correct reacquisitions (P3 correct) | 0 | 0 | 0 |
| Bank accepted / IoU flags / unverified | 72 / 10 / 35 | 72 / 10 / 35 | 72 / 10 / 35 |
| C020 trusted coverage | 0.376 | 0.376 | 0.376 |

**Default `7df8045` (rule off): every target decision record in every DEV clip is identical to `c67c738`**
(0 differing records across 8 run clips; no `reverify` record anywhere). F5 does not change default behaviour.

**What the rule did on DEV:**
- **C020:** the watch was cancelled at f239 (during the anomaly) because T005 scored 0.8005 against the target bank.
  T005 is another player whose box contains a second detection (CONTAINS_OTHER).
- **SYO C021:** cancelled at f68 because T006 scored 0.82 (itself in a SIZE_ANOMALY).

**Reference only, a superseded rule (`9e545ff`).** That version only withheld while a look-alike was visible in
RUN. It gave C020 coverage 0.376 → 0.606 with 0 false confident identity per the frozen truth. It is the version the
review showed unsafe in synthetic same-kit replacements, so it is not adopted.

**Raw outputs.** `raw/` holds tarballs and per-file hashes for the default run (`7df8045`) and the rule-ON run
(`a64bf54`, plus the C020/C021 rerun `de0df86` with the withholding log). The ON experiment branches only flip
`target_reverify_enabled` to True on top of the named `main` commit; they were never pushed. The superseded run
(`9e545ff`) is kept only as scores.

## Remaining

- **F5 (reacquisition):** no DEV improvement under the safety constraints. C020 needs evidence beyond geometry and
  generic appearance, such as a re-identification model or a view-robust appearance bank. Any enabling of the rule
  is Nathan's decision on the residual risk. C033 needs better appearance discrimination. C006 waits on F8.
- **F8 (new, camera scale drift):** upstream of C006 reacquisition.
- **F3b:** 5 reproducers still `expectedFailure`.
- **F2:** seed refusals on C034, C038 and SYO C032.
- **F4:** SYO C012 coverage 0.004.
- **F6:** one-frame lag.
- **F7:** box convention.

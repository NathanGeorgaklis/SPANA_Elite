# R2 DEV F4 + F6: state-machine cleanup (2026-11-01)

**Instruction (Nathan, 2026-11-01 15:13):**
- **F4:** recover correctly after temporary competing-candidate ambiguity.
- **F6:** remove the one-frame trusted lag into unresolvable/occluded states.
- DEV only.
- Diagnose the first failing layer for each.
- Do not weaken appearance, contact, merge or reacquisition safety.
- F5 re-verification stays OFF. No untrusted bank updates.

**EVAL was not run, opened or scored.** No frozen label, fixture, holdout or Task 3R material was touched.

## Root causes (first failing layer)

**F4.** Three separate causes.

1. **SYO C012 (the F4 DEV clip): appearance layer, not the state machine.**
   - After the first contact (f2) the true box is evaluated on almost every frame, but scores below the floors:
     - composite 0.74–0.83 while AMBIGUOUS;
     - 0.55–0.79 while LOST;
     - the reacquisition floor is 0.85.
   - Other players score as high or higher: T064 scores 0.82–0.86 against the target's 0.64–0.67.
   - Recovering here would mean lowering floors, which is forbidden, so it keeps abstaining. This is the same class
     as the C033 appearance-discrimination finding from F5.
2. **Generic bug, fixed: frame-time boundary comparisons.**
   - Frame times are `frame / fps`. A duration that is an exact multiple of the frame interval can exceed the same
     limit by floating-point error.
   - At 10 fps, one missed frame gives a 0.2 s gap between observations, equal to `population_max_gap_s`. It
     computed as 0.20000000000000018 > 0.2, so:
     - the population closed the target's tracklet;
     - the return became a reacquisition and opened an UNVERIFIED episode;
     - **trust never came back.**
   - The same exact-gap case failed at 15 and 30 fps (a gap of exactly `population_max_gap_s`).
   - All DEV clips are 30 or 60 fps, so every limit that is a whole number of frames was exposed.
3. **Residual, not fixed: Nathan's decision.**
   - After contact the return is a reacquisition. Its episode is verified only by beating a *confusable* competitor
     (R1-G review L4: beating an obviously different player says nothing about an unseen look-alike).
   - So a crossing by a different-kit player never restores trust while no look-alike is near. A same-kit crossing
     does restore it (the look-alike is beaten by the margin).
   - Relaxing this weakens reacquisition safety, so it is recorded as `expectedFailure`.
   - On DEV, unverified episodes occur only in C020, the F5 case, where a confusable body cancelled re-verification.

**F6. No state-machine lag.** C033 f105 (truth: PARTIAL, resolvable NO) was trusted, then AMBIGUOUS at f106.
- At f104–105 a smaller box inside the target box (another player's upper body) is classed NESTED/ALTERNATIVE by
  the population layer: possibly a part of the same body, so not an occluder.
- At f106 it gets a tracklet and covers the narrowed target crop past `crop_max_covered_fraction`, so trust drops on
  that frame.
- Counting such nested boxes as occluders would not fix f105 (coverage stays ≤ 0.3). It would also have removed trust
  on 50 DEV frames that the truth labels available, with no unavailable frame gained (`diag/F6_nested_probe`).
- It is an evidence-threshold boundary (geometry cannot separate a part box from an overlapping player). No code
  change; a regression test pins the same-frame response.

## Fix (`a2555ef`, review fixes `15966c3`)

- `config.sec_gt / sec_ge / sec_lt / sec_le`: a duration within 1e-9 s of a limit is **equal** to it.
  - This is the only sanctioned way to compare a duration with a limit.
  - Nothing changes off the boundary: about 1e-12 s of float error after 90 minutes, against 4e-3 s for one frame at
    240 fps.
  - At non-integer fps (29.97 / 59.94) no limit is a whole number of frames, so nothing changes.
- Every duration-vs-limit comparison now uses them: population, target, motion, appearance, and camera (F8 windows;
  parameters renamed `win_s` / `min_span_s`).
- An AST hygiene test forbids raw comparisons (local-variable, parenthesised, `min`/`max` and multi-line forms), with
  a documented allow-list: the helpers themselves and timestamp ordering.
- Config values are unchanged. Version `0.12.1-r2f4f6`.

## Tests (`tests_rb/test_state_cleanup_f4_f6.py`)

| Test | Notes |
|---|---|
| One-frame visibility loss keeps trust at 10/15/25/30/50/60 fps | Fails on `1ab094a` at 10 fps |
| Gap of exactly `population_max_gap_s` continues | Fails at 10/15/30 fps on `1ab094a` |
| Gap one frame past the limit starts a new tracklet (untrusted) | Upper side of the gap rule |
| Undetected target ages out exactly one frame past its limit | Both sides pinned; a `sec_gt`→`sec_ge` mutant is killed |
| Non-integer fps unchanged | |
| Tolerance pinned | 1 µs past a limit is past it |
| Hygiene check sees hidden forms | |
| Temporary same-kit competitor, then clean separation | Untrusted in contact, trusted again; 30/15 fps |
| Persistent competitor stays untrusted | Same-kit and different-kit; 10/15/30 fps |
| Close look-alike (cos 0.9 / 0.97) and different-kit crossings | Never wrongly trusted; bank pure |
| Occluder crossing | No frame trusted while the crop is covered past the threshold; 10/15/30 fps (F6) |
| Partial target behind an occluder, then clear | |
| F4 residual: different-kit crossing re-trusts | `expectedFailure` |

Two existing tests were made frame-exact; they had absorbed the old float error:
- `test_population.py` `test_aspect_wide_side_by_side`: the merge is now confirmed on the frame the anomaly reaches
  exactly `merged_anomaly_max_s`, one frame earlier.
- `test_appearance.py` quarantine case: `nextafter(q, 0)` becomes `q - 1e-6`. Exactly-q acceptance is still tested.

**Suite (`15966c3`, cloud):** 688 passed, 9 skipped (environment), 8 xfailed (5 F3b, 2 F5 residual, 1 F4 residual).
**Synthetic evaluation:** identical to F3c.

**Known limit.** At 10 fps a 2.5 h/s walk-away still breaks population association: a new tracklet every frame
(the low-fps fast-mover limit, also recorded in F9). The separation tests therefore run at 30/15 fps.

## Review and test audit (on `a2555ef`)

- No identity-safety issue found: no wrong trusted identity, no untrusted bank update, no fail-open.
- Every changed site was classified by safety direction. The looser shifts are at most one frame and match the
  documented `>` / `>=` meaning:
  - aging: LOST one frame later at an exact limit;
  - quarantine and provisional periods end one frame earlier;
  - `max_unmatched_s` keeps a predicted tracklet one frame longer;
  - the gap rule, which is the intended fix.
- Several tightened:
  - a merge is confirmed one frame earlier;
  - contact and closed memory are remembered one frame longer;
  - the positional prior is dropped one frame earlier.
- **MEDIUM, fixed in `15966c3`:**
  - missed sites: merged-region hold; population bank sample interval; target `_height`; camera windows;
  - the regex hygiene test could be bypassed; replaced by the AST check;
  - no two-sided aging or gap boundary tests; added.
- **Mutants:** reverts to raw comparisons are now killed by the hygiene test, and the aging-boundary flip by the
  boundary test.
- **Not covered:** the `max_occluded_s` / `max_ambiguous_s` aging of a target whose tracklet stays alive has no
  boundary test. For an undetected target the shorter `population_max_unmatched_s` applies first.

## DEV results (EVAL not run; all 8 initialised clips rerun on `15966c3`; baseline `3793d7e`, F9)

| | 3793d7e | 15966c3 |
|---|---|---|
| Initialised / refused | 8 / 3 | 8 / 3 |
| P1 false confident identity | 5 | 5 |
| P2 swaps | 0 | 0 |
| P3 wrong reacquisition | 1 | 1 |
| P4 trusted on unavailable | 1 | 1 |
| Trusted / available | 818 / 2105 (0.389) | 818 / 2105 (0.389) |
| Bank accepted / contaminated / unverified | 72 / 10 / 35 | 72 / 10 / 35 |
| Merged-reason frames (diagnostic) | 882 | 881 |

- **Per clip:** every official metric is identical (`COMPARE_vs_3793d7e.json`).
- **Decision records:** 3 of 2,775 target records differ, and **none changes trust** (`diag/DECISION_DIFF_vs_3793d7e.txt`):
  - C020 f261: AMBIGUOUS → REACQUIRING one frame earlier (an exact merged-anomaly clear);
  - C020 f262: follows from f261;
  - SYO C004 f190: an added INSUFFICIENT_CONTRAST reason while LOST.
- **F4 on DEV:** SYO C012 coverage is unchanged at 0.004 (appearance-limited, see root cause 1).
- **F6 on DEV:** C033 f105 is still P4 = 1 (evidence boundary, see F6).

## Remaining

- **F4 residual:** a non-confusable crossing never re-verifies (R1-G L4); this needs Nathan's decision.
- **SYO C012 / C033 appearance discrimination:** needs stronger appearance evidence (re-identification, view-robust
  bank), not lower floors.
- **F6:** a part box versus an overlapping player cannot be separated geometrically.
- **Carried over:** F5 (OFF), F3b (5 reproducers), F2 (seed refusals), F7 (box convention).

**Raw:** `raw/R2_DEV_F4F6_raw_15966c3.tar.gz` (sha256 in `raw/SHA256SUMS.txt`).

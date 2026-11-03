# R2 DEV: suspended-target recovery (2026-11-03) — NEGATIVE RESULT, WITHDRAWN

**Instruction (Nathan, 2026-11-03 17:36):**
- Recover false handoff suspensions on the true target without reopening the same-frame substitution hole.
- Keep detection, suspension, no bank updates while suspended, and current thresholds.
- Add sustained, multi-frame, clean recovery that agrees strongly with trusted evidence and is clearly separated from
  every nearby rival. One strong frame must never restore trust.
- No NOT_TARGET memory, no lowered floors, no trust from kit alone, no bank update before recovery.
- Focus on C034, SYO C016, C006 and the synthetic substitutes. Stop before EVAL.

**EVAL was not run, opened or scored.**

**Outcome.** The recovery mechanism (`9acf2c5`) recovered **0** false suspensions on DEV, and the review showed it
**reopens a wrong-trust path** in synthetic scenes. It is **withdrawn**: the tracker is restored byte-for-byte to the
`58a3642` handoff rule. The reviewer's reproducer is kept as a permanent regression test, which fails on `9acf2c5`
and passes now.

## Recovery rule as built (`9acf2c5`; withdrawn)

While a suspended tracklet was followed, a clean frame passed only if all of these held:
- its trusted-bank match was at least `reacq_appearance_floor` (0.85);
- that match was within `ambiguity_margin` (0.06) of the frozen **pre-break bank level**: the median bank composite of
  the verified target over the last 0.5 s before the break;
- the crop was continuous with the run's previous crop (composite ≥ 0.85);
- every other body within 3 box heights was scorable and at least 0.06 below it.

Recovery needed a consecutive run of at least 0.5 s and at least 3 frames; any failing, unclean or hidden frame
restarted it. A followed suspension aged at 3 s instead of `max_ambiguous_s`. Nothing was banked on the resolving frame.

**Pre-registration measurement** (`tools_rb/r2_handoff_bank_level.py`, profile run at `800a117`, 4 clips): nearby real
bodies passed the per-frame test on 3 of 5,538 frames (longest run 2 frames). The true target passed on 99% of frames
on C020, but only 15% / 39% / 0% on C006 / C033 / SYO C012 (`BANK_LEVEL_DEV_measurement.txt`).

## DEV results (C006, C020, C034, SYO C016 rerun at `9acf2c5`; the other clips cannot change)

| Clip | Trusted: `27dedec` / `58a3642` / `9acf2c5` | Suspensions | Recovered by the new rule | Why it did not recover |
|---|---|---|---|---|
| C006 | 58 / 28 / 28 | 2 | 0 | bank 0.81–0.84 is below the 0.85 floor; 2–3 same-kit rivals within 3 h not separated |
| C020 | 215 / 215 / 215 | 3 | 0 (all 3 resolved by the existing glitch path) | — |
| C034 | 320 / 39 / 39 | 1 | 0 | rival not separated on 16 of 16 clean frames; below the floor or the pre-break level on 10 |
| SYO C016 | 137 / 21 / 21 | 1 | 0 | rival not separated on 75 of 78 clean frames; below the floor on 20 |

- **False suspensions recovered:** 0 of 4 unresolved. Recovery delay: not applicable. Wrong recoveries on DEV: 0.
- **P1 / P3 / P4, bank contamination and coverage:** identical to `58a3642`.
  - P1: 2. P2 / P3 / P4: 0 / 1 / 1.
  - Bank contaminated: 9.
  - Trusted coverage: 0.266 (`27dedec`: P1 5, contaminated 10, coverage 0.426).
- **Why recovery fails here:** the clips where the suspensions cost coverage are same-kit crowds. The true target is
  rarely separated from a nearby same-kit player by `ambiguity_margin` on its trusted-bank match, and often matches its
  own bank below the reacquisition floor. Appearance evidence cannot prove identity there, and the rule correctly
  abstains.

## Review (one round, on `9acf2c5`)

**HIGH-1 (confirmed by reproduction): the substitute recovers and becomes trusted.**
- **Scene:** the target shows two views for 2 s, both banked. A same-kit player then takes the box in one of those
  views.
- **Result:** cos 0.83–0.92 at pose_gain 0.9 gives 80 wrong-trusted frames at 30 fps (39 at 15 fps, 161 at 60 fps)
  and an impure bank. `58a3642` is clean on the same scenes.
- **Cause:** recovery compares with the best of a multi-view bank, which is more permissive than the single pre-break
  crop. In a real substitution the true target is gone, so no rival blocks it.

**Other findings:**
- MEDIUM: the suspension clock could extend AMBIGUOUS beyond `max_ambiguous_s` (D14).
- MEDIUM: `_bank_lvl` was recorded from unclean and unproven frames.
- MEDIUM: the run had no drop test.
- LOW: rival radius in the box's own height; hidden rivals ignored; `state_sizes` omissions; the twin residual figure
  depends on pose strength.
- Mutants: 9 of 22 survived on `9acf2c5`.

All findings are moot because the code is withdrawn.

## Decision and state

- **Production code:** identical to `58a3642`. Version `0.17.1-r2ho`, registry `r2ho-2`. `continuity_step_enabled`
  stays ON (fail closed).
- **New regression test:** `test_a_multi_view_target_replaced_by_a_same_kit_substitute_is_never_trusted` (15/30/60
  fps; cos 0.83 / 0.85 / 0.9; pose_gain 0.9). It passes now and fails on `9acf2c5`, with 8 of 9 cases failing.
- **Suite (this state):** 782 passed, 9 skipped, 11 xfailed (`TEST_OUTPUT_cloud_full_suite_withdrawn.txt`).
- **Synthetic evaluation:** identical to F3c (the tracker code equals `58a3642`, which matched).
- `TEST_OUTPUT_cloud_full_suite.txt` and `SYNTH_EVAL_cloud.log` are from the `9acf2c5` code: 788 passed, synthetic
  identical.

**RECOMMEND FREEZE TRACKER FOR EVAL: NO.**
- The handoff rule is safe but costs coverage: 0.426 → 0.266 trusted.
- Its false suspensions cannot be recovered safely from colour + DINOv2 evidence in same-kit crowds.

**What would change this (Nathan's decision):**
1. **Independent identity evidence that survives same-kit crowds:** a licensable jersey-number reader (the profile and
   identity-layer hooks exist), or a football-trained ReID with documented data. Re-measure before any recovery.
2. **Make the step test less trigger-happy at its source:** require the failed step to persist for 2+ clean frames
   before suspending. Measure on DEV how many of the 9 false suspensions it removes, and confirm the synthetic
   substitutes still break.
3. **Accept the coverage cost for EVAL**, or run EVAL with the rule off, accepting the substitution residual. Either
   is a policy decision for Nathan.

**Raw:** `raw/R2_DEV_HANDOFF_RECOVERY_raw_9acf2c5.tar.gz` (sha256 in `raw/SHA256SUMS.txt`).

# R2 DEV: same-frame identity handoff safety (2026-11-02/03)

**Instruction (Nathan, 2026-11-02 20:14):**
- Prevent trusted target identity from carrying onto a different same-kit player who takes the vanished target's
  box.
- Keep the architecture. No new ReID model, no NOT_TARGET memory, no profile integration, and no lowered thresholds.
- No bank update unless continuity is physically proven. Fail closed.
- The rule must distinguish genuine motion, a brief continuous occlusion, a box being replaced, a same-kit crossing,
  and camera motion.
- Stop before EVAL.

**EVAL was not run, opened or scored.**

## Root cause (first failing layer)

In the review reproducer of `b36c5cb` HIGH-1, the target vanishes and a same-kit player appears 2 px from its box in
the same frame.
1. **Population:** it continues tracklet T002 on geometry alone, with no flag. That is correct by its own rules, since
   no physical event is visible.
2. **Tracker:** the continuation test compares the crop only with the trusted bank. A same-kit composite of 0.87
   passes `appearance_floor` (0.80), so trust is kept.
3. **Bank:** its self-consistency check uses the same floor, so it takes the substitute.

The only evidence of the handoff is a single-frame **appearance step** between consecutive crops of the same tracklet.

## Continuity rule (final: v2, `58a3642`; module docstring "SAME-FRAME HANDOFF SAFETY")

1. **Reference.** The tracklet's last **clean** crop: usable, uncontaminated, and not overlapping, containing,
   nested, partial, at the edge, contested, merged, anomalous, a duplicate or an alternative. In other words, one
   whole isolated body.
   - It is set when a tracklet is first confirmed (seed, F2 anchor, reacquisition).
   - It is kept across hidden or unclean frames, and reset on a cut or a tracklet change.
2. **Step.** Every clean crop's composite similarity to the reference, at any gap.
3. **Fail.** The step fails if it is below `reacq_appearance_floor` (0.85), or if it is at least `ambiguity_margin`
   (0.06) below the median of the tracklet's passed steps over the last 0.5 s (with at least 1 step).
4. **Suspend.** A failed step suspends the tracklet:
   - every frame is AMBIGUOUS, untrusted, unbanked and not reacquirable, and no known others are recorded;
   - the reference and baseline are frozen.
5. **Resolve.** The suspension resolves only after 3 consecutive clean crops pass against the **pre-break**
   reference and baseline.
   - Otherwise it ages out (`max_ambiguous_s`), or ends when another body is accepted.
   - The tracklet is then **handoff-suspect** for its lifetime. It is never contrast-verified again, through confirm,
     reacquisition or F5.
6. **Bank gate.** On a continuation frame the bank takes a sample only after a passed step against a reference from
   the previous frame or at most 0.1 s old (otherwise CONTINUITY_UNPROVEN). The F2 anchor's proof is its seed-crop
   match.

**Thresholds.** Both reuse existing values (`reacq_appearance_floor`, `ambiguity_margin`), so no trust or
reacquisition threshold was lowered. The new parameters are timing and count parameters only (`continuity_step_*`,
registry `r2ho-2`, version `0.17.1-r2ho`).

**DEV measurement** (`tools_rb/r2_handoff_diag.py`, 9 clips; rates from `tools_rb/r2_handoff_rel_rule.py` →
`RULE_RATES_DEV_measurement.txt`):

| | All frames | Clean frames only |
|---|---|---|
| Same-kit neighbour substitutions caught | 100% of 4,656 | 100% of 2,981 |
| Genuine trusted steps that would fail | 1.6% | 1.3% |
| Genuine untrusted steps that would fail | 5.9% | 3.5% |

Same-kit neighbour substitutions: a nearby real player's crop compared with the target's previous crop.

## Development history (negative results recorded)

- **v1 (`94f3a1e`): hard break on a failed step, tested only on consecutive crops.**
  - Review found:
    - HIGH-1: a detection gap over 0.1 s skipped the test and moved the reference onto the substitute;
    - HIGH-2: the substitute regained trust by beating a third confusable body;
    - MEDIUM-1: short history;
    - MEDIUM-2: below 10 fps the step was never tested;
    - MEDIUM-3: the evidence could not be reproduced.
  - A partial DEV run showed both breaks were false (the true target with overlapping or edge boxes), and the hard
    break cost C006 51 trusted frames.
- **v2 fixed all review findings:** the regression tests fail on `94f3a1e` (21 failures) and pass on `58a3642`.

## Tests

`tests_rb/test_handoff_safety.py` has 16 tests and 1 expected failure, at 15/30 fps, plus 8/10/60 fps where
relevant:
- substitution at cos 0.7–0.9 (with the rule off as a control: there the substitute is trusted);
- genuine motion with pose change (no break; trusted frames identical with the rule off);
- a brief occluder passing in front (no break; identical trusted frames);
- a same-kit crossing (0.8, 0.9);
- a camera pan (no break; identical);
- cuts;
- substitution before any step history (caught by the floor);
- the bank gate;
- review HIGH-1 (gaps of 1–8 frames at 15/30/60 fps);
- a swap during an unclean overlap;
- HIGH-2 (a third confusable body at 0.765–0.8);
- MEDIUM-1 (2–5 history frames);
- MEDIUM-2 (8/10 fps);
- a one-frame glitch of the true target, which resolves after exactly 3 passes;
- suspect guards, with contrast verification forced to succeed.

Expected failure: the residual of a near-identical twin (cos 0.97).

**Mutants of v2:** 18 tested, 15 killed.
- Equivalent: no handoff reset on a cut (the general cleanup releases it).
- Benign survivors:
  - a max instead of median baseline (stricter);
  - testing unclean boxes (costs coverage, not safety).

**Suite (`58a3642`):** 781 passed, 9 skipped, 11 xfailed. **Synthetic evaluation:** identical to F3c.

## DEV results (11 clips, `58a3642` against `27dedec`; `COMPARE_58a3642_vs_27dedec.json`, `DEV_SUSPENSIONS_58a3642.json`)

| Metric | Pre (`27dedec`) | Post (`58a3642`) |
|---|---|---|
| P1 false-confident identity (wrong trusted frames) | **5** | **2** (C006 3 → 0; C033 1 and SYO C021 1 unchanged) |
| P2 swaps / P3 wrong reacquisitions / P4 | 0 / 1 / 1 | 0 / 1 / 1 |
| Bank contaminated / accepted / unverified | 10 / 86 / 46 | 9 / 56 / 30 |
| Trusted coverage (available frames) | 1138/2674 (**0.426**) | 711/2674 (**0.266**) |
| Initialised / refused | 9 / 2 | 9 / 2 |

**Per-clip trusted coverage:**
- C006: 0.201 → 0.095
- C034: 0.562 → 0.069
- SYO C016: 0.651 → 0.096
- C016, C020, C033, SYO C004, SYO C012, SYO C021: unchanged

**All 9 suspensions were on the true target's tracklet.** Their IoU to the human box is 0.38–0.72; two are UNCLEAR
(IoU 0.38 and 0.48).

| Clip | Suspension frames | Resolved | Unresolved, tracklet then handoff-suspect |
|---|---|---|---|
| C006 | 20, 43 | 20 | 43 |
| C020 | 4 | all 4 | — |
| C034 | 56 | — | 56 |
| SYO C016 | 21, 73 | — | both |

- **The C006 P1 gain is real.** In the baseline, frames 91–93 were trusted on a box drifting off the target (IoU
  about 0.28). The rule suspended the tracklet at frame 43, when its box was already at IoU 0.38, and kept it
  suspended through 91–93.
- **No real same-frame handoff occurs on DEV.** The coverage cost comes from false suspensions. Real broadcast
  appearance varies far more between consecutive frames than the synthetic scenes do: true-target steps fall to
  0.74–0.92 on quick turns and blur.

## Known limits

- A near-identical twin passes the step (synthetic cos ≥ about 0.92). Recorded as an expected failure.
- A substitution during an unclean stretch (overlap, edge) is caught only on the first clean crop after it.
- On real footage, genuine steps and same-kit impostor steps overlap in absolute terms. C006 has genuine steps as low
  as 0.74, and impostors reach 0.87 on SYO C016. The relative drop separates them on the measurement, but not without
  frequent false alarms on the true target.

## Decision

**RECOMMEND FREEZE TRACKER FOR EVAL: NO.**

- **Rule on (`58a3642`, current main):** the confirmed synthetic handoff hole is closed, and P1 falls from 5 to 2 and
  bank contamination from 10 to 9. But trusted coverage falls by 37%, to 0.266, entirely from 9 false suspensions on
  the true target, 4 of them unresolved. That is not freeze-ready under rule 26.
- **Rule off:** the synthetic same-kit substitution would be trusted and banked again.
- The flag stays **ON** in main (fail closed, per rule 3) until Nathan decides.

**Next options (Nathan's decision):**
1. **Recommended. Make unresolved suspensions recoverable instead of lifetime-suspect.** Let a suspect tracklet
   regain trust only through sustained positive evidence, such as F5-style travel plus spot-clear checks, or matching
   several distinct trusted bank views. Measure false-suspension recovery on DEV first.
2. **Resolve against the trusted bank's views** (any trusted view within margin over N frames) instead of the single
   pre-break crop. This needs a DEV measurement of impostor bank-match drops before adoption.
3. **Accept the coverage cost**, or switch the flag off and accept the residual. Either is Nathan's call.

**Raw:** `raw/R2_DEV_HANDOFF_raw_58a3642.tar.gz` (sha256 in `raw/SHA256SUMS.txt`). Measurement:
`HANDOFF_DIAG_measurement_4a3fd9e.json`.

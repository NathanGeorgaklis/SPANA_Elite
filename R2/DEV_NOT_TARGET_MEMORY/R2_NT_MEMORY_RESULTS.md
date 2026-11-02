# R2 DEV: NOT_TARGET memory (2026-11-02)

**Instruction (Nathan, 2026-11-02 16:12):**
- Remember previously confirmed non-target people so they can be excluded during target reacquisition.
- DEV only; target-centric; NOT_TARGET may only veto, never prove TARGET; colour + DINOv2 evidence only.
- Never learn from ambiguous or untrusted identity. Invalidate across unsafe cuts, merges and identity breaks.
- Preserve F3c size-anomaly safety. Focus on the C020 narrowing gain. Stop before EVAL.

**EVAL was not run, opened or scored.** The memory is behind `not_target_memory_enabled`, **default False**. With the
flag off the tracker behaves exactly as before (DEV and synthetic outputs are unchanged).

## Design (`player_tracker_rb/target.py`, docstring "NOT_TARGET MEMORY" is normative)

- **What is remembered.** Only KNOWN OTHERS: bodies seen as a separate box at the seed, or beside the trusted,
  contrast-verified target settled on one tracklet for `target_known_other_min_s`. Each entry holds a snapshot of that
  body's own population bank, refreshed on each settled frame. A body is recorded only with no break flag and no merge
  lineage. Nothing is learned from an untrusted or ambiguous frame.
- **What changes.** Known others are reset when a reacquisition is accepted. The memory is not, so a remembered body
  stays excluded in later episodes while its own tracklet remains one unbroken body.
- **Invalidation.** An entry is dropped when its tracklet:
  - is no longer live;
  - becomes one of the target's own tracklets;
  - gains MERGE_LINEAGE;
  - or its box is CONTESTED, DUPLICATE_ABSORBED, IN_MERGED_REGION, MERGED_SUSPECTED, SIZE_ANOMALY (F3c), PARTIAL,
    TOUCHES_IMAGE_EDGE or REASSOCIATED_AFTER_GAP.

  Every entry is dropped on a camera continuity break.
- **Veto rule (positive evidence only).** The crop must match its own snapshot at `appearance_floor` **and** better
  than the target bank by `ambiguity_margin`. There is no veto without a usable crop or snapshot, and none for a box
  that looks like both.
- **A vetoed box is never accepted, but it remains a RIVAL.** A veto can therefore never clear the way for a worse
  candidate.
- **No new tunable thresholds.** The rule reuses `appearance_floor` and `ambiguity_margin`. Version `0.15.1-r2nt`,
  registry `r2nt-1`.

## Review and test audit (one round, on `d5efd63`)

| Finding | Status |
|---|---|
| HIGH-1: a same-kit near twin (cos 0.97), carried by the population onto the true target at the image edge, kept the target vetoed until the clip ended (t_sim 1.0 vs snapshot 0.98) | Fixed in `567047f`: positive-evidence rule plus PARTIAL/EDGE invalidation. Reproducer is a regression test |
| HIGH-2: the vetoed true target was dropped from the rivals, so a worse look-alike (cos 0.8) was reacquired (40 untrusted wrong frames) | Fixed: vetoed boxes stay rivals. Test forces a wrong veto: 0 wrong frames |
| MEDIUM-3: partial, edge and reassociated boxes did not drop entries | Fixed |
| LOW-4: a stale entry remained on the target's own tracklet (rule 28) | Fixed: dropped on confirm and prune |
| LOW-5: an entry dropped this frame could be recreated in the same frame | Fixed: break flags and merge lineage are checked at recording |
| LOW-6: the debug decision log always gains `not_target_memory: []` | Noted. It is additive and debug-only; the V2 output schema is unchanged |
| Mutants M4 (no MERGE_LINEAGE prune) and M8 (veto on an empty snapshot) survived; the off test was a tautology | Fixed. **All 7 mutants of the fixes are killed**, and the off test is replaced |

No wrong trusted identity and no bank contamination were found in any review or test scenario.

**Tests:** `tests_rb/test_not_target_memory.py`, 15 tests at 15/30 fps. They cover:
- a known non-target returning (also with F5 on);
- same-kit players and look-alikes (0.75, 0.9, 0.97);
- an edge carry onto the true target (different kit, same kit 0.9 and 0.97);
- a forced wrong veto with the rival kept;
- empty and look-alike snapshots;
- camera cut and continuity break;
- merge, size anomaly and merge lineage;
- partial, edge and gap-reassociated boxes;
- the target's own tracklet;
- nothing learned while the target is untrusted;
- bounded memory;
- off behaviour.

**Suite (`567047f`, cloud):** 741 passed, 9 skipped (environment), 9 xfailed (`TEST_OUTPUT_cloud_full_suite_fixes.txt`).
**Synthetic evaluation:** identical to F3c (`SYNTH_EVAL_cloud_fixes.log`).

## DEV results (memory ON, experiment commit `97434ad` = main `567047f` with the flag set True; never pushed)

- **Only C020 was rerun.** It is the only DEV clip with an accepted reacquisition.
- On every other clip the memory is a subset of the known others by construction, because they are never reset there.
- The comparison uses the `27dedec` baseline for the other clips.

| | Baseline (flag off) | Memory ON |
|---|---|---|
| Exclusions (vetoes) | — | **32 on 16 frames** (T005, T006, after the episode-2 reset) |
| True-target vetoes | — | **0** (6 checked on labelled frames, 26 on interpolated frames; neither vetoed tracklet ever overlaps the human box) |
| Withdrawn | — | 0 |
| Differing decision records, C020 | — | **0 of 570** (no state or trust change) |
| P1 / P2 / P3 / P4 (all clips) | 5 / 0 / 1 / 1 | 5 / 0 / 1 / 1 |
| Trusted / available | 1138 / 2674 (0.426) | 1138 / 2674 (0.426) |
| Bank accepted / contaminated / unverified | 86 / 10 / 46 | 86 / 10 / 46 |

The real C020 vetoes have positive margins of about 0.15–0.28 (snapshot 0.91–0.93 against target 0.64–0.76). They
survive the stricter rule unchanged.

## Findings

1. **The memory narrows safely on DEV, but changes no outcome.** On C020 it adds 32 exclusions that known others
   no longer provide. Those bodies were never the reacquisition winner, so neither reacquisition nor coverage changes.
2. **The ON-mode failures were real and are fixed.** Before the fixes, the synthetic reproducers produced one
   permanent true-target exclusion and one wrong (untrusted) reacquisition. After the fixes, near twins are never
   vetoed. This gives up the twin benefit: the memory only helps where appearance already separates the two bodies.
3. **Removing the reset removes a safety bound.** Known others were bounded by the episode reset. The memory is
   bounded only by the invalidation rules and by appearance. That is acceptable only with the positive-evidence rule
   and the rival rule, and it is still not validated on more than one clip.
4. **New unknown (pre-existing, not changed):** a box excluded as a KNOWN OTHER is also removed from the rivals in
   `_reacquire`. If a known other were ever wrong (for example, its tracklet carried onto the target), a worse
   look-alike could be accepted. This is the same class as HIGH-2, but in the baseline.
   - It was seen only in a synthetic scene that also forced a veto: the edge-carried target tracklet stayed a known
     other, and a look-alike Z (cos 0.8) was reacquired untrusted.
   - The same scene with the flag off gives 0 wrong frames.
   - It is not reproduced in the baseline alone. It is recorded for a later stage.

## Decision

**RECOMMEND INTEGRATION: NO (keep the flag OFF).**
- There is no reacquisition or coverage gain on DEV (0 decisions changed), and only one clip exercises it.
- The ON-mode risk class is real (HIGH-1 and HIGH-2), even though both are fixed and tested.
- It needs DEV clips with repeated reacquisitions beside returning known non-targets before it can show value.

**Next options (Nathan's decision):**
1. Keep rival status for KNOWN-OTHER exclusions too (new unknown 4). This is a baseline safety hardening.
2. Re-evaluate the memory on DEV footage with repeated reacquisitions and returning non-targets.

**Raw:** `raw/R2_DEV_NT_MEMORY_raw_97434ad.tar.gz` (sha256 in `raw/SHA256SUMS.txt`); `VETO_COUNT_C020.json`;
`COMPARE_vs_27dedec.json`. Veto counter: `tools_rb/r2_nt_veto_count.py`.

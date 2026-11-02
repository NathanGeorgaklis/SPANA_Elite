# R2 DEV: positive target evidence — target reference profile (diagnostic; 2026-11-02)

**Instruction (Nathan, 2026-11-02 17:46):**
- Keep the current tracker unchanged. Strengthen how the selected player is represented.
- Use multiple trusted reference crops, varied-pose anchors, colour + DINOv2, kit and number as support only, and
  temporal consistency.
- Report best match and consistency across anchors.
- DEV only (C006, C020, C033, SYO C012); no NOT_TARGET memory; no threshold weakened. Stop before EVAL.

**EVAL was not run, opened or scored.** **The tracker is unchanged:** 0 of 1,725 DEV frames differ from the `27dedec`
baseline in state, trust or tracklet, and tests confirm identical output with and without the layer.

## Target-profile design (`player_tracker_rb/target_profile.py`; module docstring is normative)

`TargetProfile` is a read-only diagnostic layer. It is not called by `run_rb`.

**Anchors.**
- Anchors start from the user-selected seed, which is never evicted.
- After that, an anchor is added only from the exact sample the tracker's **trusted** bank ACCEPTED on that frame.
  That sample has passed every bank hygiene rule: quality, isolation, no merge, contest, edge or lineage, verified
  camera, quarantine, self-consistency.
- Diversity: a new anchor's embedding similarity to every existing anchor must stay below 0.95
  (`profile_anchor_min_novelty`). Up to 8 anchors are kept; above that the most redundant non-seed anchor is evicted.
- There is no pose model, so each anchor's view is recorded as UNKNOWN, with the box aspect ratio as a pose proxy.

**Match against the profile.** The production composite (0.35 LAB + 0.65 DINOv2) is computed per anchor. From it:
- **best**: the highest anchor composite;
- **topk**: the mean of the 3 best composites (consistency);
- **support**: the fraction of anchors at or above `appearance_floor`;
- **kit**: the median LAB similarity, support or veto only.

**Temporal.**
- For each tracklet, the median of its clean matches over the last 0.5 s is kept, and counts once it spans at least
  0.3 s.
- The history is cleared on cuts, break flags (including gap reassociation) and when the tracklet ends.
- Temporal evidence exists only on frames where the box is clean now.

**Verdict** (counterfactual, never used by the tracker): SUPPORTED requires all of the following.
- topk and support both reach their floors, using the **lower of the temporal and the current value** (past evidence
  never outvotes the present box). The floors are `reacq_appearance_floor` and 0.5 respectively.
- kit is at or above `lab_floor`.
- No number veto.
- A margin of `ambiguity_margin` over every rival.
- Every other body in view can be scored.

Otherwise the verdict is AMBIGUOUS or UNKNOWN. No single cue can make SUPPORTED.

**Number.** An optional reader hook; none exists in R2, so the number is always UNKNOWN. A confirmed different number
vetoes; an equal number never proves.

**Parameters.** Six `profile_*` parameters (NEW_R1, UNVALIDATED, registry `r2tp-1`; version `0.16.1-r2tp`), all set
before any DEV result was seen.

## Review and test audit (one round, on `b36c5cb`)

| Finding | Status |
|---|---|
| **HIGH-1 (in the UNCHANGED tracker).** The target vanishes and a same-kit player (embedding cosine ≥ ~0.7, so composite ≥ `appearance_floor`) takes its exact box in the same frame. The population continues the tracklet, the tracker stays TRACKING **and trusted** (45/45 frames at 30 fps, 22/22 at 15 fps) with no uncertainty reason, and its bank takes the substitute. The profile inherits it as an anchor. | **Confirmed and not fixed: this stage may not change tracker state logic.** Recorded as `expectedFailure` in `TestTrackerResidualSameKitSubstitution`. At cosine ≤ 0.65 nothing is trusted. |
| HIGH-2. A look-alike was SUPPORTED while the true target was in view but unscorable (a tiny crop after a cut) | Fixed: an unscored body caps the verdict at AMBIGUOUS. Regression test added. |
| MEDIUM-1. Stale temporal history was read on unclean frames | Fixed: pruned every frame; temporal evidence only for a box that is clean now. |
| MEDIUM-2. A carried tracklet's history supported the new body | Fixed: gap reassociation clears the history, and the lower of temporal and current is used. A same-kit substitution with no break flag is the tracker's HIGH-1 residual. |
| MEDIUM-3/4/5. Evaluation-tool bias: PRE on bank-accept frames, rival scoring, the target-not-detected bucket | Fixed: PRE is causal, competitors are scored like verdict rivals, and the common-frame comparison and undetected bucket are added. |
| LOW. Kit taken from temporal only; `n_anchors` not recorded; number readings unbounded; literal duplicated | Fixed |

- **Mutants:** every reviewed mutant is killed, after direct tests for novelty, seed eviction, cut, break flags,
  contamination, edge/partial boxes, unscored bodies, stale history and rival temporal.
  - The trust and ACCEPTED-gate mutants are equivalent in practice: the bank grows only on accepted frames.
- **Tests:** `tests_rb/test_target_profile.py` has 22 tests and 3 expected failures, at 15/30 fps. They cover:
  - pose/view change;
  - near-duplicate views;
  - same-kit look-alikes (0.8, 0.9, 0.97);
  - occlusion and return;
  - the image edge;
  - drift while untrusted;
  - a wrong candidate matching only one anchor;
  - the true target matching all anchors;
  - every verdict cue;
  - kit as support only;
  - the number veto;
  - cuts and bounded memory;
  - a carried tracklet;
  - the review regressions.
  - Expected failures: the open-world residual and the two tracker HIGH-1 residuals.
- **Suite (`800a117` code):** 763 passed, 9 skipped, 12 xfailed.
- **Synthetic tracker evaluation:** identical to F3c. The tracker is untouched.

## DEV results (`tools_rb/r2_profile_eval.py`, code `800a117`; truth fixed before results)

**Anchors collected per clip**

| Clip | Anchors (bank samples) | Anchor frames | Aspect range (pose proxy) |
|---|---|---|---|
| C006 | 5 (5) | 0, 15, 30, 46, 77 | 0.42–0.72 |
| C020 | 8 (15) | 0, 90, …, 210 | 0.24–0.42 |
| C033 | 7 (7) | 0, 15, …, 90 | 0.31–0.40 |
| SYO C012 | **1** (1) | 0 (seed only; the tracker is trusted on 2 frames) | — |

**True target against the hardest same-kit competitor**
- Values are the median separation, rank-1 rate and margin rate (separation ≥ `ambiguity_margin`).
- They are computed on **common frames**: every method has a value for the true box and for a same-kit competitor.
- PRE is the production trusted bank (single best match, causal).

| Clip (common frames) | PRE (bank best) | PROFILE best | PROFILE topk | PROFILE joint (lower of temporal and current) |
|---|---|---|---|---|
| C006, all (38) | 0.037 / 0.82 / 0.26 | 0.037 / 0.82 / 0.26 | 0.049 / 0.79 / 0.29 | 0.040 / 0.79 / 0.21 |
| C006, untrusted (18) | 0.010 / 0.67 / 0.00 | 0.010 / 0.67 / 0.00 | 0.017 / 0.61 / 0.11 | 0.017 / 0.61 / 0.00 |
| C020, all (7) | 0.239 / 1.00 / 1.00 | same | 0.251 / 1.00 / 1.00 | 0.251 / 1.00 / 1.00 |
| C033, all (43) | 0.136 / 0.86 / 0.79 | same | 0.140 / 0.84 / 0.79 | 0.132 / 0.79 / 0.79 |
| C033, untrusted (11) | −0.000 / 0.45 / 0.18 | same | −0.018 / 0.36 / 0.18 | −0.020 / 0.18 / 0.18 |
| SYO C012, untrusted (12) | −0.062 / 0.00 / 0.00 | same | same | same |

**Counterfactual reacquisition evidence** (frames where the true box is detected). PRE is the production analogue: a
production score ≥ the reacquisition floor, with a margin over every other body.

| Clip | PRE true, untrusted | PROFILE true, untrusted | PRE wrong / PROFILE wrong (true box detected) | Target in view but not scored: PRE / PROFILE SUPPORTED |
|---|---|---|---|---|
| C006 | 0 | 0 | 0 / 0 | 4 / 0 |
| C020 | 60 | 31 | 0 / 0 | 5 / 4\* |
| C033 | 2 | 0 | 0 / 0 | 7 / 0 |
| SYO C012 | 0 | 0 | 0 / 0 | 1 / 0 |

\* All 4 are on the tracker's own target tracklet. The box has IoU 0.42–0.49 with the wider human box, so it is the
true target just under the 0.5 match threshold, not a wrong body.

**Actual reacquisition and coverage:** unchanged (the tracker is untouched; 0 frames differ). Trusted coverage is
1138/2674 (0.426), and P1–P4 and bank metrics are as in `27dedec`.

## Findings

1. **No material separation gain.**
   - PROFILE best equals PRE on every clip. The production bank is already a small, diverse set of trusted views
     (5, 7 and 1 samples), so the profile mostly is the bank.
   - Top-k consistency adds +0.007 to +0.012 median separation on C006 and C020, but is worse on C033 untrusted frames
     (rank-1 0.45 → 0.36, joint 0.18).
   - SYO C012 is unchanged: still negative separation.
2. **The profile cannot grow where it is needed.** Anchors come only from trusted frames. SYO C012 is trusted on 2
   frames, so its profile is the seed alone. Without new trusted evidence, positive target evidence does not
   accumulate. This is the inherent cost of the trusted-only rule, which must not be relaxed.
3. **Consistency and temporal rules make the verdict stricter, not more accurate.**
   - With the true box detected, wrong SUPPORTED is 0 for both methods on every clip.
   - The profile supports the true target on untrusted frames less often (C020: 31 against 60; C033: 0 against 2).
   - Where the target is in view but not scored, it is more conservative (0 against 4, 7 and 1).
4. **A tracker-level safety residual was found (review HIGH-1).** A same-kit body taking the vanished target's exact
   box keeps trusted identity and contaminates the bank. It is reproduced synthetically; this stage did not
   measure whether it occurs on DEV footage. It is the most important result of this stage.
5. **Open-world residual (expectedFailure).** With the target out of view, a never-seen same-kit look-alike above the
   floors has no rival in an appearance-only verdict.

## Decision

**RECOMMEND INTEGRATION: NO.**
- No meaningful separation gain on the four focus clips.
- No reacquisition gain without weakening thresholds.
- The profile duplicates the trusted bank on short clips and cannot grow on the clip that needs it (SYO C012).

The layer stays as a diagnostic, unused by the tracker.

**Next options (Nathan's decision):**
1. **(Recommended first) A tracker stage for review HIGH-1:** an appearance-continuity check on CONTINUATION frames,
   comparing each crop with the tracklet's previous clean crops and the trusted bank. That would stop a same-kit
   substitute from staying trusted. The `expectedFailure` tests already reproduce it.
2. Re-evaluate the profile on longer DEV footage with repeated exits and returns, where many trusted views accumulate.
3. Add independent evidence: a licensable jersey-number reader (the hook exists) or a pose/orientation model for true
   view anchors.

**Raw:** `raw/R2_DEV_TARGET_PROFILE_raw_800a117.tar.gz` (sha256 in `raw/SHA256SUMS.txt`); `PROFILE_EVAL.json`.

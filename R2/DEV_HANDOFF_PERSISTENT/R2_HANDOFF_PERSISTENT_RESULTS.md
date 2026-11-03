# R2 DEV: persistent handoff break (2026-11-03/04)

**Instruction (Nathan, 2026-11-03 19:51):**
- Reduce false handoff suspensions without reopening the same-frame substitution hole.
- Change only the trigger: do not suspend from one failed clean crop. Require the break to persist for 2 consecutive
  clean frames.
- Never bank the first failed frame. If the next clean frame restores continuity, continue normally; if it also
  fails, suspend exactly as `58a3642`.
- No recovery logic, no threshold change, no new identity system, no bank update during an unresolved break.
- Preserve cut, merge and anomaly safety. Stop before EVAL.

**EVAL was not run, opened or scored.**

## Rule (final: `f5b64cd`; module docstring "SAME-FRAME HANDOFF SAFETY")

**First failed clean step: PENDING, not suspended.**
- The frame is AMBIGUOUS (COMPETING_CANDIDATE): untrusted and never banked.
- No confirm bookkeeping runs: no contrast verification, no known others, no motion, size or confirmation-time
  update. It ages by `max_ambiguous_s`, and the legacy output is OCCLUDED.
- The tracklet is kept and is not reacquirable. The reference and baseline stay frozen.
- Unclean frames keep it pending; hidden frames follow the ordinary hidden logic.

**The next clean crop decides,** tested against the same pre-failure reference and baseline:
- pass: continue normally on the tracklet;
- fail: suspend exactly as `58a3642`.

**Repeat failure.** A failure within `continuity_step_window_s` (0.5 s) of the previous failure on the same tracklet
suspends at once: an unstable box is not a one-frame glitch.

No parameter or threshold changed (registry `r2ho-2`); version `0.17.3-r2ho`.

## Development history (negative results recorded)

**First version (`059387d`):** the pending frame was TRACKING but untrusted. The review found:

| Finding | Problem |
|---|---|
| HIGH-1 (reproduced) | `_confirm` ran on the pending frame, so its failed crop could contrast-verify an unverified substitute episode: 60 wrong-trusted frames at 30 fps (120 at 60 fps, 16 at 8 fps), cosine 0.8, impure bank, a wrong known other. `dddfb1c` was clean. |
| MEDIUM-2 | An alternating substitute cleared the pending state on every pass. |
| MEDIUM-3 | An unclean pending stretch never aged and fed motion and sizes. |
| LOW-4 | Legacy output showed the substitute's first frame as ACTIVE. |

All of these are fixed in `f5b64cd`, with regression tests that fail on `059387d`.

## Tests

`tests_rb/test_handoff_safety.py` has 26 tests and 1 expected failure.

Critical cases:
- **Same-kit substitute taking the box,** cos 0.7–0.9 at 15/30/60 fps: pending, then suspended; never trusted, bank
  pure. The control with the rule off is trusted.
- **One-frame blur or turn:** pending, then cleared on the next clean frame and trusted again.
- **Two-frame dip of the true target:** suspended, then resolved by the existing glitch path.
- **Crossing and occlusion; camera pan:** no break, identical to the rule off.
- **Edge or partial crop:** no break.
- **Unclean frame between two failures:** never trusted.
- **Review HIGH-1 probe** at 8/15/30/60 fps.
- **Alternating substitute:** suspended.
- **Unclean pending stretch:** ages out.
- **Pending never crosses a tracklet change.**
- **No known other recorded on a pending frame.**
- **Existing guards:** gaps, third confusable body, short history, low fps, multi-view substitute, suspect guards, bank
  gate.

Mutants: 7 of 7 trigger mutants killed at `059387d`. The review's HIGH-1 scenario is now a test.

**Suite (`f5b64cd`):** 790 passed, 9 skipped, 11 xfailed. **Synthetic evaluation:** identical to F3c.

## DEV results (C006, C020, C034, SYO C016 rerun at `f5b64cd`; the other clips never failed a step, so they cannot change)

The 9 DEV suspensions under `58a3642` (all on the true target's tracklet; see `R2/DEV_HANDOFF_SAFETY`):

| Clip | `58a3642` suspensions | `f5b64cd` |
|---|---|---|
| C006 | 20, 43 | suspended at 21 (frame 20 pending) and 43 (repeat failure 0.37 s after 21; the next clean frame failed too under `58a3642`) |
| C020 | 293, 331, 399, 508 | 294, 400, 509 suspended (each pending one frame); **331 removed** (pending, then cleared) |
| C034 | 56 | suspended at 58 (pending 56) |
| SYO C016 | 21, 73 | suspended at 22, 74 (each pending one frame) |

**False suspensions removed: 1 of 9** (C020 frame 331). It had already resolved within 3 frames under `58a3642`, so
there is no coverage gain. The other 8 broke on two consecutive clean frames. These are persistent appearance changes
of the true target (turns, blur, partial overlap), not one-frame glitches.

| Metric (11 DEV clips) | `27dedec` | `58a3642` | `f5b64cd` |
|---|---|---|---|
| P1 false-confident identity | 5 | 2 | **2** |
| P2 / P3 / P4 | 0 / 1 / 1 | 0 / 1 / 1 | 0 / 1 / 1 |
| Bank contaminated / accepted | 10 / 86 | 9 / 56 | **9 / 56** |
| Trusted coverage | 0.426 | 0.266 | **0.266** |
| Wrong trusted frames on the 4 clips (P1) | 3 | 0 | **0** |

**Substitution safety:**
- `58a3642`: 0 wrong-trusted frames on every synthetic substitute case.
- `f5b64cd`: the same, including the review HIGH-1 probe, the alternating substitute, the multi-view substitute and
  all fps.
- Residual unchanged: a near-identical twin (cosine ≥ ~0.92, or lower under strong pose similarity) passes the step,
  as before.

## Decision

**RECOMMEND FREEZE TRACKER FOR EVAL: NO.**
- The 2-frame trigger is safe: it reopens nothing.
- But it removes only 1 of the 9 false suspensions on DEV, and trusted coverage stays at 0.266 (against 0.426 in the
  baseline).
- The DEV false suspensions are persistent appearance changes that colour + DINOv2 continuity cannot tell apart from a
  same-kit substitution.

The trigger stays in main: it is no worse on any metric and spares the coverage of genuine one-frame glitches.

**Options (Nathan's decision):**
1. **Independent identity evidence** (a jersey-number reader, or a football-trained ReID with documented data). This is
   the only route the DEV data supports for telling these cases apart.
2. **A policy choice for EVAL:** freeze with the handoff rule ON (safer, coverage 0.266), or OFF (coverage 0.426,
   with the same-frame substitution residual).

**Raw:** `raw/R2_DEV_HANDOFF_PERSISTENT_raw_f5b64cd.tar.gz` (sha256 in `raw/SHA256SUMS.txt`); `COMPARE_f5b64cd_vs_27dedec.json`.

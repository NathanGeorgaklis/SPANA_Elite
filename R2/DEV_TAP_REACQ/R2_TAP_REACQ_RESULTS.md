# R2 DEV: real-user (tap) reacquisition protocol (2026-11-06)

**Instruction:** Nathan, 2026-11-06 00:39. Validate human-assisted reacquisition with the interaction the product will
give users: pause on the checkpoint, the user taps the player, snap to exactly one detection, normal re-seed
validation, refuse ambiguity, "I can't identify" continues automatically. Do not relax crowded-scene rules.
**EVAL was not run, opened or scored.**

## Implementation (`checkpoint.snap_tap`, TAP answers; commit `88768bb`)

A tap `(x, y)` snaps to a detection only if:
- it lies inside at least one body detection (else **TAP_OFF_PLAYER**);
- every detection containing it belongs to one nested group, i.e. one body with its part / alternative boxes (two or
  more groups = **TAP_AMBIGUOUS**); the snapped box is the group's host;
- the host is not MERGED_SUSPECTED / IN_MERGED_REGION (else **TAP_MERGED**).

The snapped **detector box** then goes through the unchanged re-seed validation: the ambiguity pre-check (other body
≥ 0.25 inside, own track, no other track nested, no merge lineage), `TargetTracker.seed()`, and an immediate bank
anchor. Nothing was relaxed. A refused tap leaves the checkpoint open; NOT_VISIBLE declines it and the automatic
tracker continues.

## Fresh clips and simulated users (`tools_rb/r2_tap_dev.py`)

- **Clips:** the 15 jersey-dataset DEV clips (8 s; never used for tracker development; frozen human target boxes
  every 0.5 s, v1 `1d9669dd…`, v2 `b322b8f7…`). They were chosen for same-kit neighbours, crowds and turns.
- **Seed:** the first keyframe with a certain visible human box. **6 of 15 seeds were refused** by the normal seed
  rules (J01, J04, J14: another body inside the box; J08: crop partly covered; J09, J12: no detection matches the loose
  box). 9 clips ran.
- **Users** answer only on keyframes (≤ 0.5 s after the pause): CENTRE (box centre), SHOULDER (25 % in, 22 % down),
  JITTER (centre + Gaussian finger error, σ 15 % of the box), WRONG (nearest other player, same-kit competitor when
  labelled), WRONG_HIDDEN (taps another player while the target is hidden), SILENT (automatic baseline).
- **Scoring** on the keyframe grid: a trusted keyframe is correct if the tracker box matches the human box (IoU ≥ 0.5,
  or fully contained, ≥ half its height, centre inside — the human boxes are loose; the two strict-IoU misses, J02 f150
  and J05 f450, are containment 1.0 and were verified as the right player by eye). Both counts are reported.

## Results (9 clips, 116 visible keyframes)

| Protocol | Coverage | Correct / wrong keyframes | Checkpoints | Re-seeds | Taps accepted / refused | Wrong accepted (strict) |
|---|---|---|---|---|---|---|
| **SILENT (automatic)** | **0.155** | 18 / 0 | 9 | 0 | – | – |
| CENTRE | **0.181** | 21 / 0 | 13 | 3 | 3 / 57 | 0 (2*) |
| SHOULDER | 0.172 | 20 / 0 | 13 | 2 | 2 / 58 | 0 (1*) |
| JITTER | 0.181 | 21 / 0 | 13 | 3 | 3 / 57 | 0 (2*) |
| WRONG | 0.155 | 18 / **1** | 12 | 1 | 1 / 62 | **1** |
| WRONG_HIDDEN | 0.181 | 21 / 0 | 11 | 3 | 3 / 65 | 0 (2*) |

\* strict-IoU artefacts (the right player in a tighter box).

**Refusal reasons (CENTRE):** re-seed pre-check ambiguous 34, TAP_MERGED 11, TAP_AMBIGUOUS 6, deferred bank anchor 5,
seed validation 1. SHOULDER adds 10 TAP_OFF_PLAYER (the shoulder point fell outside the tighter detector box).

**Safety:**
- Correct taps: 0 wrong trusted keyframes, 0 wrong acceptances (containment-aware).
- WRONG: 63 deliberate wrong taps, **1 accepted** (J07 f360: an isolated, clean other player; followed as the user's
  decision, 1 wrong keyframe, then lost). Geometry cannot refuse a clean wrong tap: that is the user's authority.
- WRONG_HIDDEN: 8 taps on another player while the target was hidden, **0 accepted**. All 8 were refused by the
  crowd rules, not because the system knew the target was hidden; an isolated wrong tap while hidden would be accepted
  (as WRONG shows).
- The automatic path is unchanged (SILENT = no-resolver output; synthetic evaluation identical).

**Checkpoints per clip:** 1.0 automatic (every scorable clip but J13 lost trust by ~1.6 s and never recovered); 1.4
with taps. Correct taps succeeded 3 times in 60 (5 %).

## Usability problems found

1. **Most correct taps are refused in these scenes (90–95 %).** The fresh clips are crowded; the detector boxes are
   merged, nested or contain another body, so the (unrelaxed) ambiguity rules refuse. The user would tap repeatedly
   without success unless the UI says why and asks for a moment when the players separate.
2. **Seed refusal (6 of 15).** A loose initial box over a nearby body is refused; the product's initial selection
   should be a tap too (same snapping), not a drawn box.
3. **Shoulder / edge taps miss the detector box (10 of 68).** Detector boxes are tighter than a person's sense of the
   body. The paused frame should show the candidate boxes so the tap lands inside one; a tolerance would be a relaxation
   and was not added.
4. **Re-seeds do not hold.** Each accepted re-seed restored trust for about one keyframe before the same crowd broke it
   again: +3 keyframes in total.

## Proposed user protocol

1. Initial selection by tap on a paused frame with candidate boxes shown; refused taps explain why.
2. When identity stays unresolved for 1.5 s, pause with the candidate boxes highlighted and the message "Tap your
   player, or choose 'Can't tell'".
3. A tap inside exactly one box is validated as a new seed; a refusal shows the reason (overlapping players, merged
   box, player cut off) and offers "try a later moment" (resume, re-pause after 1.5 s) or "Can't tell".
4. "Can't tell" resumes automatic tracking; the checkpoint re-asks after another 1.5 s of unresolved identity.
5. Every answer is logged (frame, tap, snapped box, outcome, reason) in the checkpoint file.

## Decision

**Assisted mode is safe enough on this evidence** (no wrong identity from a correct or ambiguous tap; a clean wrong tap
is the user's decision and is recorded), **but it is not useful enough on fresh crowded footage** (+0.026 coverage,
5 % tap success) and 40 % of seeds were refused.

**RECOMMEND FREEZE TRACKER FOR EVAL: NO.**
- Automatic coverage on fresh clips 0.155; assisted 0.181. On DEV (previous stage, ideal boxes) 0.266 → 0.360.
- The blocker is upstream of the checkpoint: merged / nested detections in crowds make both automatic reacquisition and
  tap validation refuse. That is a detector-population problem, not a re-seed rule to relax.

**Open for Nathan:** whether to accept the assisted protocol as the product interaction now (it is safe) while the
detector / population crowd behaviour is worked on; and whether the seed box rules should also take a tap.

**Raw:** `raw/R2_DEV_TAP_REACQ_raw_7ddd3e9.tar.gz` (checkpoint files, DONE records, logs; results documents excluded
for size; sha256 in `raw/SHA256SUMS.txt`), `TAP_SCORES.json`.

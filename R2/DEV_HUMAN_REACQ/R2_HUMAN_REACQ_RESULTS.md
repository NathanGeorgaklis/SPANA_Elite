# R2 DEV: human-assisted target reacquisition (2026-11-05)

**Instruction:** Nathan, 2026-11-05 20:22.
- Keep handoff safety ON, all states and thresholds, and automatic reacquisition.
- No jersey / OCR, no NOT_TARGET or identity layer.
- When identity stays unresolved, emit an IDENTITY_CHECKPOINT. A user selection becomes a new trusted seed only after
  normal seed validation. Ambiguous selections are refused, and no contaminated bank sample crosses the re-seed.
- Stop before EVAL. **EVAL was not run, opened or scored.**

## Design (`player_tracker_rb/checkpoint.py`, opt-in via `run_rb.run(checkpoint_resolver=...)`)

**Wrapper.** The unchanged TargetTracker decides every frame first. The wrapper only watches its result.

**Checkpoint.**
- Opens after `checkpoint_unresolved_s` = 1.5 s (= max_ambiguous_s + reacq_window_s) of continuous untrusted
  identity, and never while the initial seed anchor is pending.
- Carries the frame and time, the trust-loss frame, tracker context (state, reasons, episode, last confirmed frame,
  reported box, tracklet) and every body detection.
- The resolver is asked each frame while the checkpoint is open. It closes as:
  - AUTO_RESOLVED: the tracker regained trust on its own;
  - DECLINED: the user cannot identify the target; the checkpoint re-arms after another 1.5 s;
  - RESEEDED.
- A refused, malformed or crashing answer leaves the checkpoint open and changes nothing.

**Re-seed = a new human seed.** It must pass:
1. a pre-check, refusing the selection if:
   - more than one detection matches the box;
   - another body lies ≥ 0.25 inside the box;
   - the matched detection has no own track, or its nested group holds another track;
   - its track has merge lineage;
2. a fresh `TargetTracker.seed()` (the run's own seed rules);
3. an immediate bank anchor. A deferred anchor is refused.

**On acceptance:**
- The old TargetTracker (bank, contrast history, known others, handoff suspicion, motion) is discarded.
- The old target tracklets stay the target's physical body chains (rule 28): only ids carry across, never appearance.
- The re-seed frame is the next episode, with a `USER_CONFIRMED_RESEED` reacquisition event (document contract
  unchanged).
- Output: `<video>_rb_identity_checkpoints.json`, with its sha256 in the provenance sidecar.

**With no resolver** the run is unchanged:
- tests show frame records identical to a run without the wrapper;
- the synthetic evaluation is identical to F3c;
- DEV C006 with the resolver is identical to a no-resolver run on the same commit up to the re-seed.

## Tests

- `tests_rb/test_identity_checkpoint.py` (27 tests) and `tests_rb/test_identity_checkpoint_run.py` cover:
  - false suspension on the true target;
  - genuine same-kit substitution (declined, never trusted);
  - target leaves and re-enters;
  - wrong user selection (followed as the user's decision, clean restarted bank);
  - ambiguous box; selection on no detection;
  - camera cut;
  - multiple checkpoints; auto-recovery;
  - malformed answers and a raising resolver;
  - a deferred-anchor refusal;
  - the pre-check rules;
  - episode offsets;
  - 15 fps;
  - the run-level contract, checkpoint file and continuity.
- **Full suite (`1978a6c`):** 913 passed, 9 skipped, 11 xfailed.
- **Correction:** commit `4eaf500` wrongly said "Suite green". Its run had 1 failure (the publish test did not know
  the checkpoint file is optional), fixed in `1978a6c`.

## DEV (11 clips; simulated correct user from the frozen DEV labels; `tools_rb/r2_checkpoint_dev.py`)

**Simulated user:**
- SELECTs the human box when the target is visible, resolvable and certain at that frame;
- gives no answer if no human box lies on that frame (5 Hz grid);
- otherwise answers NOT_VISIBLE.

The tracker never reads labels.

**Before = current main without checkpoints** (handoff rule ON). For C006 a fresh no-resolver rerun on `1978a6c` is
identical in metrics to the baseline. That baseline differs only by a 0.1 px environmental rounding drift, which
appears before any checkpoint.

| Metric (9 scorable clips; C032, C038 seed-refused as before) | before | after |
|---|---|---|
| Trusted coverage | 0.266 (711 frames) | **0.360 (963)** |
| P1 false-confident identity | 2 | 2 (same frames: C033 f105, C021 f36) |
| P2 / P3 / P4 | 0 / 1 / 1 | 0 / 1 / 1 (unchanged) |
| Bank contaminated / accepted | 9 / 56 | **9 / 79** (no new contaminated sample) |

Pre-handoff-rule baseline: 0.426.

**Per clip:**

| Clip | Coverage before → after |
|---|---|
| C020 | 0.376 → 0.670 |
| SYO C016 | 0.096 → 0.407 |
| C006 | 0.095 → 0.137 |
| C033 | 0.249 → 0.269 |
| Others | unchanged |

**Checkpoints:**
- 20 opened on 8 of 9 clips (C016: none): 5 RESEEDED, 12 DECLINED (target not identifiable), 3 still OPEN at clip
  end.
- Averages: 2.2 checkpoints per clip; 0.56 re-seeds per clip; 4 checkpoints per successful re-seed.
- The 3 still open are C012, C034 and one in C033. Every correct answer there was refused by validation:
  - more than one detection matches;
  - a nested box without its own track;
  - another body inside the box;
  - a nested group holding another track;
  - merge lineage;
  - deferred anchor.
- 337 refused answers in total, almost all in crowded set-piece / contact scenes. This is the ambiguity rule working:
  coverage is lost rather than a contaminated seed accepted.

**Continuity:**
- Every run passed the document checks (episode increments = events).
- One logical target T001 throughout.
- Pre-re-seed target records are identical to the no-resolver run.

## Decision

**RECOMMEND FREEZE TRACKER FOR EVAL: NO.**
- The checkpoint path is safe on DEV: 0 new wrong-trusted frames, 0 new bank contamination, and 0 frames changed
  without a validated user answer. It recovers +0.094 coverage (252 frames).
- But coverage stays below the pre-handoff baseline (0.360 vs 0.426), with 3 checkpoints never resolvable in crowds.
- The 2 pre-existing P1 frames and 9 contaminated bank samples remain.
- The DEV numbers assume an ideal user (exact box, instant answer). A real user's looser boxes would be refused more
  often; that is untested on real footage.

**Open for Nathan:**
- (a) whether EVAL should include human-assisted mode, and with what user protocol;
- (b) whether to relax the re-seed ambiguity rules for crowded scenes (not recommended without new evidence).

**Raw:** `raw/R2_DEV_HUMAN_REACQ_raw_1978a6c.tar.gz` (sha256 in `raw/SHA256SUMS.txt`), `COMPARE_cp_vs_pb.json`,
`CHECKPOINT_SUMMARY.json`.

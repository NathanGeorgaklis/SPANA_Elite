# R2 PRODUCTION TRACKER — CONSOLIDATION RECORD

Instruction: Nathan, 2026-11-06 10:12 — "Stop adding tracker complexity and freeze the strongest simple SPANA Elite
target tracker … choose the production default [for the handoff-safety rule] based on real DEV performance, not
synthetic perfection … Stop before EVAL."

Frozen commit: `4cc95cf5806727b51bb034f737e36f9b552bbbf9` (main). EVAL never opened, run or scored. Task 3R untouched.

## 1. Exact production configuration

| item | value |
|---|---|
| tracker version | `player_tracker_rb.__version__` = `1.0.0-r2prod` |
| config registry | `CONFIG_REGISTRY_VERSION` = `r2prod-1` |
| default config hash | `f193342f6287ad473cc02ed2b2382f9a07eb0b52858359ea504c25d18baaca13` |
| output schema | `2.0.0-r1a` (unchanged) |
| person detector | `yolox_s.onnx` sha256 `c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063` (status LICENCE_UNCLEAR, as pinned in R1) |
| appearance | `dinov2_vits14_cls_patchmean_224x112.onnx` sha256 `0ab62713cf7759dd820b53576b39010cb0fe8dcfbe22af444c945c519bf9efea` (LICENCE_UNCLEAR) + colour signature |
| entry point | `player_tracker_rb.run_rb.run(video, seed_frame, seed_bbox, out_dir, ...)` with `TrackerConfig.default()` |

Only change to the config registry in this stage: `continuity_step_enabled` default `True -> False`. No parameter
added, no threshold value changed. The default tracker is therefore behaviourally the `27dedec` tracker (best
historical DEV baseline) — verified below, per clip, frame for frame.

## 2. Features kept / disabled

Kept (in the default path, all pre-existing):
- safe seed handling (seed anchor validation, nested/part-box refusal, NOT_ISOLATED refusal);
- population / merge safeguards (merged-detection reasons, physical-entity exclusion);
- camera compensation (CUT / continuity-break handling);
- LOST search expansion;
- colour signature + DINOv2 appearance; strict reacquisition (`reacq_appearance_floor` 0.85, bounded reacq window);
- ambiguity / fail-closed behaviour (AMBIGUOUS, LOST, UNKNOWN; bank refusals).

Optional modes (code present, OFF by default, enabled only by explicit override):
- same-frame handoff safety rule `continuity_step_enabled` (see §3);
- identity checkpoints — only active when a `checkpoint_resolver` is passed to `run_rb.run`; never a requirement.

Not integrated (not imported by the production path; `tests_rb/test_production_defaults.py` pins this):
- persistent identity layer (`identity.py`), NOT_TARGET memory (`not_target_memory_enabled=False`),
  target-reference profile (`target_profile.py`, `target_reverify_enabled=False`), jersey/shorts number reader
  (`jersey.py`; EasyOCR eval RECOMMEND NO), experimental ReID models, handoff recovery heuristics
  (no DEV gain), human-assisted / tap re-seeding as a requirement (RECOMMEND FREEZE NO twice).

## 3. Handoff-safety rule: ON vs OFF on the complete DEV benchmark

Evidence: `R2/DEV_HANDOFF_PERSISTENT/COMPARE_f5b64cd_vs_27dedec.json` (both complete 11-clip runs, same labels v1).

| | rule ON (f5b64cd) | rule OFF (27dedec) |
|---|---|---|
| trusted frames / available | 711 / 2674 | 1138 / 2674 |
| coverage (run clips) | 0.266 | 0.426 |
| P1 wrong-identity trusted frames | 2 | 5 |
| P2 / P3 / P4 | 0 / 1 / 1 | 0 / 1 / 1 |
| bank contaminated / accepted | 9 / 56 | 10 / 86 |
| seeds refused | 2 | 2 |

Per clip the rule changed only three clips, all false suspensions of the true target:
C006 0.201→0.095 (removes the 3-frame P1 flicker at f91–93 = 50 ms, costs 30 true frames), C034 0.562→0.069,
SYO C016 0.651→0.096. The two remaining P1 frames (C033 f105, C021 f36) are not touched by the rule.

Decision: production default OFF. Removing one 50 ms flicker is not worth 427 trusted true-target frames (−38% of
coverage); the rule stays available as an opt-in mode and keeps its synthetic test coverage with explicit overrides.

## 4. Production DEV result (this run, commit 4cc95cf, default config)

Run: `tools_rb/r2_dev_run.py run` on all 11 DEV fixtures, clean tree, pinned video hashes, labels v1
(sha `9b657a59…`), scored with `tools_rb/r2_dev_run.py score`. Raw outputs: `raw/` (tarball + SHA256SUMS).
Comparison files: `COMPARE_27dedec_vs_prod.json`, `COMPARE_pb_ON_vs_prod.json`.

| clip | outcome | coverage | trusted | P1 frames | P4 frames | bank contam./accepted |
|---|---|---|---|---|---|---|
| MCFC C006 | SCORED | 0.201 | 57 | 91, 92, 93 | – | 1 / 5 |
| MCFC C016 | SCORED | 0.946 | 158 | – | – | 2 / 9 |
| MCFC C020 | SCORED | 0.376 | 214 | – | – | 0 / 15 |
| MCFC C033 | SCORED | 0.249 | 104 | 105 | 105 | 0 / 7 |
| MCFC C034 | SCORED | 0.562 | 320 | – | – | 0 / 14 |
| MCFC C038 | RUN_REFUSED (seed matches nested/part detection) | – | – | – | – | – |
| SYO C004 | SCORED | 0.764 | 97 | – | – | 3 / 11 |
| SYO C012 | SCORED | 0.004 | 1 | – | – | 0 / 1 |
| SYO C016 | SCORED | 0.651 | 136 | – | – | 2 / 17 |
| SYO C021 | SCORED | 0.654 | 51 | 36 | – | 2 / 7 |
| SYO C032 | RUN_REFUSED (seed crop drift 0.829 < 0.85; NOT_ISOLATED) | – | – | – | – | – |

Aggregate: **trusted coverage 0.426 (1138 / 2674 frames), P1 = 5 frames (3 clips), P2 = 0, P3 = 1, P4 = 1,
bank contamination 10 / 86 accepted samples (all `BOX_NOT_ON_HUMAN_TARGET_BOX`), 2 / 11 seeds refused.**

Identical, clip by clip and frame by frame, to the best historical baseline `27dedec_combined`
(`COMPARE_27dedec_vs_prod.json`: every metric SAME) — the consolidation reproduced the baseline exactly rather than
re-tuning it. Synthetic eval (`SYNTH_EVAL_r2prod.log`) identical to the F3C reference.

## 5. Known limitations (unchanged from the baseline; now the production limits)

1. P1: C033 f105 and C021 f36 are single wrong-identity trusted frames at merged/contact moments; C006 f91–93 is a
   3-frame (50 ms) flicker on a same-kit crossing. 5 / 1138 trusted frames = 0.44% wrong, all ≤ 3 consecutive frames.
2. Coverage is low on crowded / merged scenes (C012 0.004, C006 0.201, C033 0.249): the tracker abstains rather than
   guesses; 1065 target frames carry a merged-detection reason.
3. Seed refusals: C038 (seed box on a nested/part detection) and C032 (seed body crop drifts below the reacquisition
   floor, not isolated). The user must re-seed on an isolated frame; no automatic recovery.
4. Bank contamination 10 / 86 samples, all boxes off the human target box at merge moments; it has not produced a
   wrong trusted identity on DEV but is the main long-match drift risk (rule 32).
5. No long-match (≥ 90 min) evidence; DEV clips are 5–20 s. Memory growth, camera cuts over long periods and repeated
   re-entry are tested synthetically only.
6. Model licences LICENCE_UNCLEAR (unchanged since R1 pinning).
7. Single camera style per club (MCFC broadcast, SYO fixed wide); 2 clubs, 2 venues.

## 6. Test record

- Full suite (`tests_rb`, commit 4cc95cf): 928 passed, 9 skipped, 11 xfailed, 1765 subtests — `SUITE_OUTPUT_r2prod.txt`.
- New: `tests_rb/test_production_defaults.py` (defaults, experimental flags OFF, no experimental imports on the
  production path, version tags). Scenario tests that exist to exercise the handoff rule now enable it explicitly.
- Synthetic eval: identical to reference (`SYNTH_EVAL_DIFF.txt`).

## 7. TRACKER READY FOR USER TESTING: **YES** (DEV scope, with the limits in §5)

Meaning: passes the complete 11-clip DEV benchmark at the best historical level with 0.44% wrong trusted frames, no
wrong reacquisition, no experimental layer in the path, simple predictable behaviour. Not a product-quality claim;
EVAL has not been run. The user-testing stage should collect: seed refusals, wrong-player episodes (any length),
and coverage, on new footage, without changing thresholds.

## 8. Persistence

See commit messages on main and `00_READ_FIRST` for the Git / push / fresh-clone verification of this stage.

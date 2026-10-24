# BUILD_STATUS.md: SPANA engine v2-rebuild, R0

Phase: **R0, recovery, tracker interface map, tracking evidence and tracker rebuild plan.** R0 was read-only with respect to engine code.

Completed 2026-10-24, Australia/Sydney. Author: Claude, working for Nathan Georgaklis.

## What R0 did

- Read the three project docs and the full archive: `SPANA_recovered_originals.zip` (sha256 1e99c82f…, which matches the project record) and `SPANA_specs_and_reports.zip`.
- **Also read the original task 3A–3Q zips in `~/Downloads`.** They hold frozen outputs that are not in the recovered-originals zip; see R0_DISCREPANCIES D5.
- Re-verified every one of the 82 manifest entries by SHA-256.
- Dry-ran the original 3P/3Q patches on the 3N bases, in scratch only.
- Mapped the Pass-1 → Pass-2 interface from surviving code and frozen hash-verified outputs.
- Collected the tracker behaviour and failure evidence.
- Designed the tracker acceptance targets and the R2 benchmark (design only, not run).
- Wrote the R1 plan.

## What R0 did not do

- No engine or tracker Python source was created or modified.
- No model was downloaded and no inference was run.
- No benchmark was run.
- Task 3R stayed untouched.
- **The holdout was not opened.**

**Scratch-only runs.** Two things ran in the cloud scratch workspace and were discarded:
- `pytest` collection of the surviving tests, which ran no tracker or model; the raw output is in `R0_TEST_COLLECTION_RAW.txt`;
- a `git apply` patch dry run; the raw output is in `R0_PATCH_DRYRUN.txt`.

## Deliverables (`~/Documents/SPANA_Engine_Rebuild/R0/`)

The 11 charter deliverables, all present:
- `R0_RECOVERY_LEDGER.csv`
- `R0_TRACKER_INTERFACE_MAP.md`
- `R0_TRACKER_BEHAVIOUR_EVIDENCE.md`
- `R0_TRACKER_FAILURE_CATALOGUE.md`
- `R0_TRACKER_ACCEPTANCE_TARGETS.md`, which includes the R0.10 benchmark design
- `R0_R1_TRACKER_REBUILD_PLAN.md`
- `R0_DEPENDENCY_GRAPH.md`
- `R0_BEHAVIOUR_EVIDENCE.md`
- `R0_CONSTANTS_AND_THRESHOLDS.csv`, 211 rows
- `R0_TESTS_AND_MODELS.md`
- `R0_ACCEPTANCE_TARGETS.md`

Supporting files: `R0_DISCREPANCIES.md`, `R0_EVIDENCE_SOURCES_SHA256.txt`, `R0_PATCH_DRYRUN.txt`, `R0_TEST_COLLECTION_RAW.txt`, `FULL_FILE_MANIFEST_SHA256.txt`, this file, and `.write_test`, a 5-byte persistence probe created at the start of R0. It cannot be deleted without delete permission.

## Known errata (left in place; the snapshot was already verified)

- **`R0_TRACKER_INTERFACE_MAP.md` §8, bullet 4,** says REIDENTIFYING has "an unknown legacy mapping". §5.2 of the same file is authoritative and records REIDENTIFYING → REACQUIRING as DOCUMENTED, from the live validation trace.

## Persistence and backup facts

- **Mac source of truth.** `~/Documents/SPANA_Engine_Rebuild/R0/` was created in R0. A write test passed. All 15 deliverable hashes on the Mac matched the workspace copies byte-for-byte.
- **Archive snapshot.** `~/Documents/SPANA_Engine_Archive/R0_2026-10-24_1237/`. The manifest is `FULL_FILE_MANIFEST_SHA256.txt`, sha256 5b98410e…4955. Re-hashed **from the archive copy**: 16/16 OK, and the archive manifest is identical to the source manifest. **MAC_ARCHIVE_VERIFICATION = PASS.** This file and `ARCHIVE_VERIFICATION.txt` were added afterwards and are verified separately in `ARCHIVE_VERIFICATION.txt`.
- **Prior archive items untouched.** `SPANA_recovered_originals.zip`, `SPANA_specs_and_reports.zip` and the 3A spec were re-hashed after R0; see `ARCHIVE_VERIFICATION.txt`.
- **Git: NOT INITIALISED.** This session's Mac shell cannot delete files in connected folders, and git needs to delete its own `.lock` files. Initialising git here would leave a broken repository. R1 needs either your delete permission for Documents or a local `git init` that you run yourself (see R1 plan §7).
- **Second independent backup: NOT VERIFIED.** The 12 R0 documents were also uploaded to the claude.ai project "SPANA Elite" under `claude/R0/`, which is off the Mac's disk. That is a convenience copy only and was **not** hash-verified, because the read-back channel returns text rather than bytes. A Git remote or an external SSD is still needed (charter §18).
- **Temp-only copies.** None. Every R0 deliverable is on the Mac and in the verified archive snapshot.

## Final verdict

```
SPANA ENGINE v2-REBUILD — R0

RECOVERY INVENTORY VERIFIED:
YES

HISTORICAL MANIFEST ENTRIES:
82

EXACT RECOVERED:
14   (+1 outside the 3N manifest: football_analysis/controlled_duel.py, new in 3Q)

HASH VERIFIED:
1    (event_engine_v1/possession_events.py 3Q; +1 outside the manifest: football_analysis/receiver_eligibility.py, new in 3P)

PATCHABLE:
7    (player_stats, confidence, event_arbitration, event_detector, event_qualifier, possession_detector, shot_candidate — 3N base hash-verified, original patches apply cleanly; resulting 3Q bytes not hash-provable)

OLDER VERSION ONLY:
0    (the inventory's "older version only (7)" are reclassified PATCHABLE; see R0_DISCREPANCIES D2)

PARTIAL:
0

MISSING:
60   (incl. event_engine_v1/__init__.py, which is provably a 0-byte file, and the 3 models)

UNKNOWN:
0

TRACKER FILES RECOVERED:
3    (player_tracker_mo/event_adapter.py, pipeline.py, tracking_health.py; of 34 tracker manifest entries = player_tracker/ 12 + player_tracker_mo/ 21 + new_tracker_snapshots.py)

TRACKER FILES MISSING:
31   (player_tracker/ 12/12, player_tracker_mo/ 18/21, new_tracker_snapshots.py) + 3 model files

TRACKER INTERFACE:

FULLY SPECIFIED:
13

PARTIALLY SPECIFIED:
4

UNKNOWN:
1

TRACKER TESTS SURVIVING:
4    (test_pipeline_integration.py, test_actor_attribution.py, test_benchmark_fixture_guard.py, test_temporal_coverage_guard.py; the 155-test player_tracker_mo/tests/ suite is missing)

TRACKER TESTS RUNNABLE NOW:
0

TRACKER FIXTURES AVAILABLE:
1    (GOLD_DUEL_01: source extract re-hashed on the Mac 164541…, fixture/seed 7929bb28…, human-signed clip-level identity 05fe67c6…/423ad2ba…, frozen Pass-1 JSON 88246efe…). Plus 1 output-contract-only reference (pass_completed_01 JSON 40fa0df1…; its source video is not on the Mac). No human per-frame tracking annotations exist anywhere.

TRACKER MODELS EXACTLY RECOVERED:
0    (historical SHA-256 known for yolov8s, yolov8x and yolo26n-reid; none of the files survive)

BIGGEST TRACKER UNKNOWNS:
- origin of yolo26n-reid.onnx and the ReID embedding (undocumented; no public match)
- which detector weights Pass 1 actually used (reports conflict: YOLOv8s vs yolo26n-reid)
- how identity_confidence / overall_track_confidence were computed (consumers and thresholds known, formula not)
- legacy state mapping for PROVISIONAL and OUT_OF_FRAME
- tracker behaviour in same-kit crossings (never examined on a surviving run)
- no human per-frame tracking truth for any clip
- location of the pass_completed_01 and duel_lost_01 source videos

R1 TRACKER ARCHITECTURE:
One new package, player_tracker_rb/, with no imports from football_analysis or historical tracker paths. It reimplements the historical contract (the {base}_mo_tracking_results.json schema: 15 summary keys, 7 frame keys, 29 track fields) plus an additive, labelled "v2" block. Public states are TRACKING / OCCLUDED / AMBIGUOUS / LOST (+ REACQUIRING sub-state), each mapped onto the historical online vocabulary so every non-TRACKING state reaches Pass 2 as non-TRACKING. Contrastive appearance identity: target bank vs every competitor, with no appearance-free fast path. Every detection-backed body counts as a competitor (no PROVISIONAL blind spot). Merged-box guard. Held/observed boxes explicit. Hold and ambiguity limits in seconds. Gates in body heights and capped. Reacquisition is a separate multi-frame verification whose bar rises with absence time and after cuts. Sequential decode, no resume, provenance and hashes in every output.

FILES PROPOSED FOR R1:
player_tracker_rb/{__init__, config, video_io, detector, appearance, motion, population, target_state, output_schema, run_rb}.py
tests_rb/{test_config_units, test_target_state_synthetic, test_output_contract, test_surviving_consumers, test_seed_and_provenance}.py
tools_rb/make_gold_label_sheet.py
R1/{R1_SCHEMA, R1_DESIGN_NOTES, R1_REPORT}.md

TRACKER ACCEPTANCE STRATEGY:
Judge tracking only on human-labelled physical identity, never on event results. Hard gates: 0 false-confident-identity frames on event-critical windows, 0 identity swaps, 0 wrong reacquisitions, 100% output-contract and held-box honesty, determinism, and never TRACKING on MAR #4 in GOLD extract 262–274. Abstention (coverage) is reported, not hidden. R1 checks synthetic suites + contract + surviving consumers + GOLD collision on human-confirmed labels. The full 13-scenario, two-source benchmark (holdout, Task 3R and GENERAL_BENCHMARK_V2 windows excluded) is labelled and frozen before inference in R2.

EVENT ENGINE MODIFICATIONS PROPOSED IN R1:
NONE

MY APPROVAL REQUIRED BEFORE R1:
YES

ENGINE CODE WRITTEN DURING R0:
NO

PERSISTENCE CHECK

SOURCE TREE SAVED:
YES   (R0 documentation tree on the Mac; no engine source tree exists yet)

SNAPSHOT CREATED:
YES

SNAPSHOT PATH:
~/Documents/SPANA_Engine_Archive/R0_2026-10-24_1237/

SHA256 MANIFEST CREATED:
YES   (FULL_FILE_MANIFEST_SHA256.txt, sha256 5b98410edd52b263bddb481b60a49caaafc26b6c51bd9cc592839855342a4955)

MAC ARCHIVE HASH VERIFICATION:
PASS

ONLY COPY EXISTS IN TEMP WORKSPACE:
NO

SAFE TO CONTINUE:
YES   (to R1 planning approval only; git and a second backup must be set up at R1 start)
```

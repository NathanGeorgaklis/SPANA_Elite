# R0_DISCREPANCIES.md

SPANA engine v2-rebuild, R0. This file records where the R0 evidence differs from the project docs (`SPANA_RECOVERY_INVENTORY.md` and `00_READ_FIRST…`) or from other reports.

**No historical file was edited.**

## D1. Recovery counts: same facts, different denominator

The inventory says 82 manifest entries made up of 15 exact, 2 hash-verified, 7 older and 60 missing. Those categories add up to 84, not 82.

**Resolution (verified in R0).** Two of the counted files were created after 3N, so they are not in the 82-entry 3N manifest:

- `receiver_eligibility.py`, new in 3P;
- `controlled_duel.py`, new in 3Q.

Within the 82 manifest entries:

| Status | Count | Files |
|---|---|---|
| EXACT | 14 | 15 minus `controlled_duel` |
| HASH_VERIFIED | 1 | `possession_events`, 3Q bytes |
| PATCHABLE_FROM_VERIFIED_BASE | 7 | see D2 |
| MISSING | 60 | — |

Outside the manifest there is one more EXACT file (`controlled_duel`) and one more HASH_VERIFIED file (`receiver_eligibility`).

## D2. The "older version only (7)" files are patchable from a verified base

The inventory says their 3Q versions "can be regenerated from the 3P/3Q patches but cannot be hash-proven". R0 confirms that statement and checks the charter's three conditions. All three are met:

1. The original patch text survives, with identical SHA-256 in the archive and in the task zips.
2. The 3N base is unambiguous and hash-verified.
3. `git apply` applies 3P and then 3Q cleanly. This was run as a scratch dry run (`R0_PATCH_DRYRUN.txt`).

The same patch chain also reproduces all three hashes that were recorded at the time:

- `receiver_eligibility` f1ccd2bc…
- `possession_events` f0ac5dc9…
- `controlled_duel` c5705cef…

They are therefore classified **PATCHABLE_FROM_VERIFIED_BASE**. They remain unprovable at the byte level because no 3Q hash was recorded for these 7 files. No patch has been applied to any real file.

## D3. `event_engine_v1/__init__.py` is provably empty

Its manifest hash, e3b0c442…, is the SHA-256 of zero bytes. The file is "missing", but its exact content is known, so it can be reconstructed with hash proof.

The other three `__init__.py` files are different. `football_analysis`, `player_tracker` and `player_tracker_mo` have non-empty hashes, so their contents are unknown.

## D4. Model hashes do exist

The inventory lists the three models as missing, which R0 confirms. The 3N manifest still records the SHA-256 of each one, so any future copy can be checked against it.

- yolov8s: 1f47a78b…
- yolov8x: 3df4ada6…
- yolo26n-reid: 8529c383…

## D5. Additional surviving evidence the inventory did not list

The **original 3A–3Q task zips** are all in `~/Downloads`. `SPANA_recovered_originals.zip` keeps only their `.py`, `.sh` and `.patch` members. The zips themselves contain much more, including:

- the frozen 3N Pass-1 tracking JSONs, which match the recorded hashes (88246efe…, 40fa0df1…);
- frozen Pass-2 outputs;
- per-frame traces;
- the Task 3M association-capture replay, from which R0 recovered the historical association score weights;
- fixtures and human labels.

None of this is engine code, so the inventory's "nothing new was found" statement about **code** still holds.

## D6. pass_completed_01 and duel_lost_01 source videos are not on the Mac

- `f5bd8ce0-pass_completed_01.mov` (a2e45703…) was not found in Documents or Downloads, searched by name, by size and inside zip listings.
- `8a710009-dual_lost_01.mov` was not found either.

Both originally lived in a cloud upload path (`/root/.claude/uploads/…`). The consequences:

- pass_completed_01 cannot be re-run from Pass 1;
- the surviving `test_benchmark_fixture_guard.py` and `test_temporal_coverage_guard.py` cannot execute as written.

## D7. Conflicting statements within historical reports

These are recorded and not resolved:

1. **Frame numbering prose.** `event_adapter.py:39-41` says "1-indexed", while `analyze_target_player.py:285-287` says "0-based relative". The data settles it: frame_log starts at 1, and frame 0 is the synthetic seed/bootstrap frame.
2. **Role of `yolo26n-reid.onnx`.** It is called the "Detector" in END_TO_END_TRACKING_PIPELINE_REPORT.md, the "tracking model version" in pipeline.py, and the "person-ReID embedding" in earlier_frame_reid_report.md. Separately, the Pass-1 PlayerDetector is described as YOLOv8s in one report and "YOLOv8" in another.
3. **The `DEFAULT_CANDIDATE_CONFIGURATION` comment in `pipeline.py:113-120`** says the ablation is "still incomplete". `TRACKING_ABLATION_FINAL_REPORT.md` and `test_pipeline_integration.py` say it is complete. So the surviving test file is newer than that comment, or the comment was never updated.
4. **Test counts.** Reports cite 155 tracker tests (SEG_D), 148 = 132 + 16 (END_TO_END) and 180/180 (status after 3Q). These are different suites at different times. None survive in runnable form.
5. **`overlap_contests` type.** Test fixtures use `0` (an int), while real outputs use a list of pairs.
6. **The UNKNOWN_RELEASE time in the status doc** is given as "11.43". The frozen output says 11.43–11.47, followed by POSSESSION_LOST at 11.47.
7. **The pass_completed_01 fixture** `tracking_artifact_sha256` (aba3464b…) points at the stale, checkpoint-misregistered Pass-1 JSON, not at the 3N fresh JSON (40fa0df1…).
8. **The pass_completed_01 seed** is circular. It was copied from a Pass-1 output and not independently human-selected.

## D8. Historical `detection_confidence` is not a detector score

In every one of the 1,713 GOLD frame-log records, the value is quantised by `position_source`:

| position_source | detection_confidence |
|---|---|
| CONFIRMED | 1.0 |
| AMBIGUOUS or WEAK_SPATIAL_RECOVERY | 0.5 |
| PREDICTED | 0.0 |

There is one exception. Any reading of this field as a YOLO confidence would be wrong.

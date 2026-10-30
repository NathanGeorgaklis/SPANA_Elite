# R2 label freeze record: second intake (2026-10-30)

**Instruction (Nathan, 2026-10-30 15:21 Sydney):** revalidate the corrected files and freeze only. Do not modify labels; skip clips already frozen; preserve HUMAN_UNCERTAIN; freeze only files that pass; do not run or score the tracker.

**Tool:** `tools_rb/r2_label_intake.py` (r2-intake-1), unchanged since `32d0eec`, run with `--freeze --hash-clips`. The full ledger is `R2/R2_LABEL_INTAKE_2026-10-30.json`. The first intake is in `R2/R2_LABEL_FREEZE_RECORD.md`.

## Inputs

- **Incoming folder:** Nathan's corrected files are in `~/Desktop/_spana_r2_work/labels_incoming/` (23 files, saved 2026-10-29 23:31 to 2026-10-30 04:19 UTC). The work folder moved from Downloads to Desktop.
- **Clip records:** read from the verified archive copy `SPANA_Engine_Archive/R2_CLIPS_2026-10-26_0821/`.
  - Its manifest (`5f66732e…376b`) was re-verified first: `sha256sum -c` PASS.
  - With `--hash-clips`, all 25 clips were re-hashed against the registry.
- **First attempt, disclosed.** The first check-only run pointed at the old Downloads clip folder, which no longer exists, so it failed closed for every file ("clip extract record missing"). That was not a label problem. Its ledger is kept as `_r2_work/freeze/intake2_check_FAILED_clipsdir_missing.json` in the archive. The re-run against the archive copy found every file valid.

## Nathan's confirmation before freezing (2026-10-30)

Long OUT_OF_FRAME stretches, and in three clips resolvable NO, are the only post-seed states that need no box. Claude asked whether those frames were genuine. **Nathan answered: "Genuinely off-screen."** The frames are frozen as labelled. Affected clips (OUT_OF_FRAME frames after the seed):
- MCFC_MIT: C002 (244, also resolvable NO), C014 (263), C016 (70), C032 (227), C033 (179), C037 (393);
- SYO_MAR: C004 (169, also NO), C008 (27), C021 (221, also NO), C024 (85), C035 (17).

Nathan also said that if he ever wants to label only part of a clip, the method is a documented pre-inference **labelled end frame** (scoring from the seed to that frame). This is not implemented; nothing frozen depends on it.

## Result

| | Count |
|---|---|
| Incoming files | 23 (C019 and C027 were not resent; they were frozen on 2026-10-27) |
| Valid | 23 / 23 |
| Newly frozen | **22** |
| Already frozen, skipped | 1: R2_MCFC_MIT_C009. The resent file differs from the frozen copy **only** in `saved_utc` (2026-10-27T06:58:58Z → 2026-10-30T04:13:19Z). The frozen copy stands. |
| **Frozen in total** | **25 / 25: DEV 12 / 12, EVAL 13 / 13** |
| Identity-scorable (has a run fixture) | **23: DEV 11, EVAL 12** |
| Excluded from identity scoring | R2_SYO_MAR_C019 (DEV, rejected: "ball not in play yet"); R2_SYO_MAR_C027 (EVAL, rejected: "celebration") |
| HUMAN_UNCERTAIN frames | **0** in all 25 frozen files |

Every newly frozen copy is byte-identical to its incoming file. Each fixture's `labels_sha256` equals its FROZEN record's sha256. The incoming files were not changed or moved.

## Frozen labels (sha256 of the frozen `labels_v1.json`)

| Clip | Split | sha256 | Frozen (UTC) |
|---|---|---|---|
| R2_MCFC_MIT_C002 | EVAL | `9dc91813733992d60a3951c86963894e862046d7e868cb38f127b2de5b0210d2` | 2026-10-30 |
| R2_MCFC_MIT_C006 | DEV | `81e3577cf5f9ee751efaa59b1465fbdf44c1c0a46c301e13d544c73b1a11a488` | 2026-10-30 |
| R2_MCFC_MIT_C009 | EVAL | `05ed11033c0d3735ae1a88ed88afe1b10f836ad106885718030807d332035c4e` | 2026-10-27 |
| R2_MCFC_MIT_C014 | EVAL | `a6f4b558c93267edb4f557efa722333840200841adc1e14c558d7cf961cf61ff` | 2026-10-30 |
| R2_MCFC_MIT_C016 | DEV | `9b657a5990a3a78983e4aa779e273a1e16fdfc5896b0653d21c8df6db08e2628` | 2026-10-30 |
| R2_MCFC_MIT_C020 | DEV | `0951ffc023083d8ab45eacaf17bacc98c03f88da9ea01fadbdb84293eed97e48` | 2026-10-30 |
| R2_MCFC_MIT_C026 | EVAL | `350f447893c4322e5345d9eef80513bac5d29b70d6b83e8810300c37a55f5bfa` | 2026-10-30 |
| R2_MCFC_MIT_C032 | EVAL | `77289def22b12c155ac8799ecc33c71adaa90fb410b376e0fcd1647d19ce5882` | 2026-10-30 |
| R2_MCFC_MIT_C033 | DEV | `fb0f2f03f5b71ad1835dcd6c9b9d85f9e22776350c6d12590da0333d8d633ca4` | 2026-10-30 |
| R2_MCFC_MIT_C034 | DEV | `8c0d71600689cf81cbdfd3cf74c0c3dc6d13667874daea11c3b31a4e4e8bcff9` | 2026-10-30 |
| R2_MCFC_MIT_C037 | EVAL | `fa463d53709ed8bec342e680eaf0f55ce5d5d9088191f648d16eaa5ca05a672c` | 2026-10-30 |
| R2_MCFC_MIT_C038 | DEV | `c5e9a36f9dfd39373358314f3f96fc10cabcb58a1f40d89580b411da2bbf56f4` | 2026-10-30 |
| R2_SYO_MAR_C004 | DEV | `0b3364d8189c43b980492858a4b79d55579ee0e2e1b31868b3bb6ffc33777fa3` | 2026-10-30 |
| R2_SYO_MAR_C008 | EVAL | `c2df6e79ae87fa16fbb024fdc1dc830fb254769c9189188296ae5595a59b81e5` | 2026-10-30 |
| R2_SYO_MAR_C012 | DEV | `e1261fdcb2618a111d2050c5b8d8260d417e5a6e4edce18f972549d1a3ba94aa` | 2026-10-30 |
| R2_SYO_MAR_C016 | DEV | `2d27274958217fb3efe3cec430d73854d1c017e7ae65f867a43728f54f9e5278` | 2026-10-30 |
| R2_SYO_MAR_C017 | EVAL | `a7e93ed55cf349c79812e9614d3c9de96b4dbcfc19e8f62809c2d012a62a0b29` | 2026-10-30 |
| R2_SYO_MAR_C019 | DEV | `a7815c15b3bf1e1ba6ffc4af6197b2a6796b014520b9a85e486e5b15ab9edbf2` | 2026-10-27 (rejected) |
| R2_SYO_MAR_C021 | DEV | `dbf27d41cb548b0c1d13b060b642ce2193b92a3c3c4a41e1280c502ed0262c51` | 2026-10-30 |
| R2_SYO_MAR_C023 | EVAL | `a5d877d007621b86cbbfb60e314d328989feec36e619fac34b135bf25b553a46` | 2026-10-30 |
| R2_SYO_MAR_C024 | EVAL | `927903d666f27fdec142f7576498f76f9535c8a0b1087cd2c7d74a7c57456262` | 2026-10-30 |
| R2_SYO_MAR_C027 | EVAL | `a0fa8ffbcbc55cb1e19d308d1161cc8ba665eb2b56d3ef874f39a0297e604565` | 2026-10-27 (rejected) |
| R2_SYO_MAR_C032 | DEV | `eaec0a2d73a25624a445159ceec05be162745ca9da8b3e61f3d0c994f11d80f7` | 2026-10-30 |
| R2_SYO_MAR_C034 | EVAL | `e3639a6c91ef29fe583f3c8230c240288fd738f31f945a2c616757ba9e7e9da0` | 2026-10-30 |
| R2_SYO_MAR_C035 | EVAL | `6f4cd2d9e9218c045de62aaa24b289a48757ce9bd8b3302d8a037a7c807225a4` | 2026-10-30 |

## What the frozen truth contains (from the labels; post-seed frames)

**Identity-challenge content is concentrated in a few clips:**
- Contact with named people:
  - DEV: MCFC C006 (101 frames, 2 people), SYO C012 (169 frames, 3 people), SYO C032 (76 frames, 3 people).
  - EVAL: SYO C017 (16 frames, 1 person).
- Occlusion by a person or object: C006 (92 frames), C012 (82), SYO C032 (33), MCFC C016 (19), C033 (2).
- First clean reappearances: C006 [503], C014 [535], MCFC C032 [314], C033 [106, 109, 115, 295], SYO C032 [235].

**Classes not present in the truth:**
- **No frame is labelled merged** in any clip.
- Camera cuts are labelled only in MCFC C002 (2).
- Several clips are clean continuous tracking with no other person labelled: MCFC C020, C026, C034, C038 and SYO C016, C023, C034. They test continuity and coverage, not identity under contact.
- Three clips have long resolvable-NO stretches, all coinciding with OUT_OF_FRAME, so they add no hidden-but-on-screen identity truth: MCFC C002, SYO C004, SYO C021.

The classes in `R2_CLIP_SELECTION.json` were Claude's thumbnail proposals; the frozen labels above are the truth. **Merged-body and ambiguous-reacquisition coverage is therefore absent or minimal in both DEV and EVAL**, which limits what R2 can show about priorities 2–4.

## No inference output before freeze (verified)

- `player_tracker_rb/` is identical to R1-G `30f2e39`: `git diff 30f2e39 HEAD -- player_tracker_rb` is empty.
- No tracker output exists for any R2 clip in the repo, the R2 work folders or the incoming folder:
  - no Pass-1 / tracking-results / provenance sidecar / decision log / jsonl output;
  - the only historical outputs are the GOLD and pass_completed_01 contract fixtures and the R1-G GOLD run.
- The intake tool imports no tracker, detector or model module; a test checks this.
- All 23 incoming files declare `tool: tools_rb/r2_annotator.html v2` and annotator "Nathan Georgaklis".

## Tests

After freezing: `test_r2_label_intake`, `test_r2_labels`, `test_r2_protocol`, `test_r2_audit`, `test_tracker_eval` and `test_package_hygiene` on the Mac: 114 tests, OK.

## Rules kept

- No label was edited, re-saved or moved, and frozen files are never replaced; a correction needs `labels_v2` with the reason recorded.
- Nothing was run or scored.
- Running the tracker on DEV needs Nathan's explicit approval.
- EVAL scoring stays refused without Nathan's approval record naming the clean tracker commit. Do not tune on EVAL.

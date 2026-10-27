# R2 label freeze record: intake of 2026-10-27

**Instruction:** Nathan, 2026-10-27 18:01 Sydney: validate and freeze only; no tracker run, no scoring; never repair a label; stop for any invalid file and report the exact issue.

**Tool:** `tools_rb/r2_label_intake.py` (r2-intake-1). It calls `tools_rb/r2_labels.py` `validate` / `freeze` unchanged. The full per-clip ledger is `R2/R2_LABEL_INTAKE_2026-10-27.json`, and it includes every validator problem for every file.

## Result

| | Count |
|---|---|
| Label files found | **25 / 25** (no unknown files) |
| Valid and frozen | **3 / 25** |
| Invalid, not frozen | **22 / 25** |
| DEV frozen | **1 / 12** (R2_SYO_MAR_C019, rejected by the annotator) |
| EVAL frozen | **2 / 13** (R2_MCFC_MIT_C009; R2_SYO_MAR_C027, rejected) |
| Frozen and scorable for identity | **DEV 0, EVAL 1** (R2_MCFC_MIT_C009) |
| HUMAN_UNCERTAIN frames | **0** in all 25 files |

Checks that passed on all 25 files:
- file name = `clip_id`;
- clip sha256: label file = extract record = registry = clip bytes (all 25 lossless clips re-hashed);
- DEV / EVAL recomputed from `r2_protocol` = extract record = registry: 12 DEV and 13 EVAL, as expected.

## Frozen labels (byte-identical to the incoming files)

| Clip | Split | Labels sha256 | FROZEN record sha256 | Identity scoring |
|---|---|---|---|---|
| R2_MCFC_MIT_C009 | EVAL | `05ed11033c0d3735ae1a88ed88afe1b10f836ad106885718030807d332035c4e` | `753eff117a3a9af5abd23b9123c825d7f416e7feca7d80212421945f26e99244` | INCLUDED |
| R2_SYO_MAR_C019 | DEV | `a7815c15b3bf1e1ba6ffc4af6197b2a6796b014520b9a85e486e5b15ab9edbf2` | `e2867e0c523d9e8db2b130de0ac6c977fcd4b17dbbed427eb32789018a7a621f` | EXCLUDED: rejected, "ball not in play yet" |
| R2_SYO_MAR_C027 | EVAL | `a0fa8ffbcbc55cb1e19d308d1161cc8ba665eb2b56d3ef874f39a0297e604565` | `137ce3a42326c1f686afbc1b2315bdc5b598078f33c06885c304c402c640c76a` | EXCLUDED: rejected, "celebration" |

Run fixture `R2_MCFC_MIT_C009.fixture.json` (sha256 `15b71fcf988854f1dfd1dcb21cdc789e88a9fcce562e39ef830f7db45218e07f`) has:
- the human seed: frame 459, box [90.2, 320.6, 248.2, 643.4];
- video sha256 `e064c60d…6496`.

Rejected clips get no fixture.

R2_MCFC_MIT_C009 has little identity information:
- the seed is at frame 459 of 600, so 140 frames (2.3 s) are scored;
- all 140 frames are clean;
- there are no other people, no contact and no occlusion.

It tests continuity only, not identity under contact.

## Invalid files (not frozen, not repaired)

All are unchanged in `~/Downloads/_spana_r2_work/labels_incoming/`. The hashes as received are in the ledger.

**Type A: frames after the seed left UNLABELLED** (labelling stops partway). 20 files:

| Clip | Split | Seed | Unlabelled frames |
|---|---|---|---|
| R2_MCFC_MIT_C002 | EVAL | 300 | 339–599 (261) |
| R2_MCFC_MIT_C006 | DEV | 300 | 373–599 (227) |
| R2_MCFC_MIT_C014 | EVAL | 136 | 245–599 (355) |
| R2_MCFC_MIT_C016 | DEV | 360 | 512–599 (88) |
| R2_MCFC_MIT_C020 | DEV | 30 | 73–599 (527) |
| R2_MCFC_MIT_C032 | EVAL | 30 | 361–599 (239) |
| R2_MCFC_MIT_C033 | DEV | 0 | 91–599 (509) |
| R2_MCFC_MIT_C034 | DEV | 30 | 211–599 (389) |
| R2_MCFC_MIT_C037 | EVAL | 147 | 209–599 (391) |
| R2_MCFC_MIT_C038 | DEV | 537 | 582–599 (18) |
| R2_SYO_MAR_C004 | DEV | 0 | 0–299 (300): occlusion never set on any frame; frames 128–299 have no state at all |
| R2_SYO_MAR_C008 | EVAL | 15 | 82–299 (218) |
| R2_SYO_MAR_C012 | DEV | 45 | 200–299 (100) |
| R2_SYO_MAR_C016 | DEV | 90 | 210–299 (90) |
| R2_SYO_MAR_C021 | DEV | 0 | 78–299 (222) |
| R2_SYO_MAR_C023 | EVAL | 60 | 79–299 (221) |
| R2_SYO_MAR_C024 | EVAL | 15 | 76–299 (224) |
| R2_SYO_MAR_C032 | DEV | 104 | 147–299 (153) |
| R2_SYO_MAR_C034 | EVAL | 15 | 49–299 (251) |
| R2_SYO_MAR_C035 | EVAL | 15 | 138–299 (162) |

In every Type A file, the target boxes stop where the states stop. This rules out the page losing states. The 4 files with one box past the last state (C016 512, C038 582, C012 200, C021 78) have a box drawn on the next frame before its state was applied. The page's range apply includes both end frames, so this is not an off-by-one.

**Type B: complete, but a rule is broken.** 2 files:
- **R2_MCFC_MIT_C026** (EVAL): `first_clean_reappearance 270 out of range`. 270 is the seed frame, and a first clean reappearance must come after the seed. This is its only problem.
- **R2_SYO_MAR_C017** (EVAL): there are two problems.
  - The seed frame (249) has `contact_with: [P1]`, and a seed must have no contact.
  - `first_clean_reappearance` includes the seed frame 249.

**Ticking the seed frame as a first clean reappearance** also appears in 13 of the Type A files: MCFC_MIT C006, C032, C033, C034, C038 and SYO_MAR C004, C012, C016, C021, C023, C024, C032, C034. The validator will report it there once their frames are complete. The page's **Check** reports it too, but only after every frame is labelled.

**Further checks still pending.** The validator only checks required boxes, contact-partner boxes and first-clean-reappearance completeness once every frame has a valid state. Type A files may show more problems once their states are complete. The page's **Check labels** button shows the same list.

## No inference output before freeze (verified)

- No tracker, detector or model was run on any R2 clip. `player_tracker_rb/` is identical to R1-G `30f2e39`; `git diff 30f2e39 HEAD -- player_tracker_rb` is empty.
- No file named after an R2 clip exists outside `_spana_r2_work/clips/` and `labels_incoming/` and the verified archive copies. No JSON in the repo refers to an R2 clip apart from the registry and protocol records.
- The annotation pack contains only:
  - per-frame JPEGs (for example, 300 `f_*.jpg` for C004);
  - the clip manifest, which mentions no model or tracker;
  - the page.

  The intake tool imports no tracker, detector or model module; a test checks this.
- All 25 files declare `tool: tools_rb/r2_annotator.html v2`, annotator "Nathan Georgaklis", and were saved 2026-10-27 06:57–06:59 UTC.

**Tool provenance (disclosed).** The labelling page in the pack (`annotate/index.html`, sha256 `c2a3761c…1596`) is `r2_annotator.html` from commit `fe8b3a1`, the reviewed v2 page. It does not include the two later page-side additions from `5ea06b9`:
- the fps check when a label file is loaded;
- the page check that the seed box equals the target box on the seed frame.

Neither affects label content or validity. The Python validator, which is the freeze authority, enforces the seed-box rule, and no file fails on it.

## Rules kept

- No label was edited, re-saved or moved. The incoming hashes were verified unchanged after the freeze.
- A frozen file is never replaced. A correction to C009, C019 or C027 needs `labels_v2` with the reason recorded.
- Invalid files stay unfrozen until Nathan corrects them in the page and downloads them again. The next intake run freezes them then, and already-frozen clips are skipped.
- Nothing was scored. DEV and EVAL are both untouched by inference.

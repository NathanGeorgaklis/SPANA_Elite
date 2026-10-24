# R1-C stage record: generic person detector backend

**Stage:** R1-C, approved by Nathan on 2026-10-24 at 18:39 Sydney. `R1/R1_GENERIC_TRACKER_REQUIREMENT.md` is binding.

**Built:** the generic detector interface and the YOLOX-s ONNX Runtime backend, covering preprocessing, inference, decoding, confidence filtering, NMS, tile merging, coordinate restoration and clipping, and detector provenance.

**Not built:** no appearance or ReID, motion, population, target association, state machine, ball or event logic.

**Contract:** the R1-A output contract is unchanged; `output_schema.py` was not modified. The config registry is now `r1c-1`, with 106 parameters (16 added, all detector or tiling settings). The default config hash is now `7a314e056049817dcc1656df62bcec97df1d9dfd7e66475f64705a0e947a5868`.

## Model

The full record is in `R1C_MODEL_RECORD.md`.
- **Model:** YOLOX-s, the official ONNX file from Megvii release `0.1.1rc0` (tag commit `e1052df7…`).
- **SHA-256:** `c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063`. This was confirmed by three independent copies: the cloud download, a separate direct download on the Mac, and the earlier third-party Hugging Face copy.
- **Pinned:** in `config.py`. A mismatch refuses to load.
- **YOLOX-m** (the fallback): `21ff6cfd…5b3f`. It was downloaded, hashed and evaluated, but **not selected**.
- **Licence:** code and weights are Apache-2.0. The status is **LICENCE_UNCLEAR** because of the COCO training-data caveat. It is approved for DEVELOPMENT / EVALUATION only and is **not** PRODUCTION_APPROVED.
- **Preprocessing:** verified on real footage. The model expects BGR pixels at 0–255, a top-left letterbox padded with 114, and no normalisation. The normalised variant scores at most 0.001.

## Files

| Path | Status | Purpose |
|---|---|---|
| `player_tracker_rb/detector.py` | NEW | Generic interface: `PersonCandidate` (bbox, score, touches_image_edge, detector_pass, kind=PERSON_CANDIDATE; no identity, team, role, colour or number), `DetectorBackend`, letterbox and its inverse, clipping, IoU, containment, NMS, tile layout, tile-merge rules, `ScriptedDetector` |
| `player_tracker_rb/detector_yolox.py` | NEW | The only module that imports onnxruntime or knows the YOLOX layout. It verifies the model hash and checks input and output shapes. |
| `player_tracker_rb/config.py` | CHANGED (additive) | 16 detector parameters. Model-convention values are CONTRACT/HISTORICAL_REFERENCE; the selected tiling is NEW_R1 and UNVALIDATED. Added a `tuple` parameter type. |
| `tests_rb/test_detector.py` | NEW | 33 tests: geometry, letterbox reversal, clipping, filtering, NMS, empty and multiple detections, tile merging, no leakage, hash, missing and invalid model, preprocessing, determinism, partial people at the border |
| `tests_rb/test_package_hygiene.py` | CHANGED | The structural literal 3 (box index, image ndim) is allowed |
| `tools_rb/detector_eval.py` | NEW | Input-mode evaluation on detection quality only, with the selection rule declared in the code before running |
| `R1/R1C/yolox_reference/` | NEW | The official YOLOX reference files read (Apache-2.0, with `LICENSE`), with hashes |
| `R1/R1C/detector_eval/` | NEW | Raw per-chunk outputs and `DETECTOR_EVAL_SUMMARY.json` (Mac, authoritative) |
| `R1/R1C/R1C_TEST_OUTPUT_mac.txt`, `R1C_CONFIG_REGISTRY.csv`, `R1C_MODEL_RECORD.md`, `MODEL_HASHES.sha256` | NEW | Records |
| `R1/R1_MODEL_AND_LICENSE_RECORD.md` | CHANGED | A1 now carries the downloaded hashes and confirms the official ONNX release |
| `models/yolox_s.onnx`, `models/yolox_m.onnx` | NEW (git-ignored) | On the Mac and in the snapshot; hashes as above |

## Input sizes evaluated

The official model has a **fixed 640×640 input**, so 960 and 1280 cannot be run without a re-export, which was not built. Settings compared on the development clip:

| Setting | Seed-box IoU / score | Pool recall | Small (<80 px) pool recall | Candidates per frame | Time per frame (Mac) | Peak RSS |
|---|---|---|---|---|---|---|
| s_full (one 640 pass) | 0.867 / 0.754 | 0.967 | **0.819** | 8.0 | 0.52 s | 702 MB |
| **s_grid2x2 (2×2 tiles + full frame)** | 0.867 / 0.754 | 0.996 | **0.993** | 9.9 | 3.13 s* | 702 MB |
| s_grid3x2 (3×2 tiles + full frame) | 0.867 / 0.754 | 0.996 | 0.987 | 10.0 | 4.26 s* | 702 MB |
| m_full (YOLOX-m, one 640 pass) | 0.904 / 0.888 | 0.973 | 0.879 | 7.6 | 1.56 s* | 808 MB |

\* These settings were run as three chunks in parallel on the Mac's 4 CPUs (1 ONNX thread each), so they include some CPU contention. s_full was run alone.

- **Selection rule** (declared in `tools_rb/detector_eval.py` before running): the first setting in cost order that has seed IoU ≥ 0.5, pool recall ≥ 0.95 and small-person pool recall ≥ 0.90.
- **Selected: s_grid2x2.** It is now the default (`detector_tiling = grid`, `detector_tile_grid = (2, 2)`, overlap 0.2, full-frame pass included).
- **What "pool recall" means:** the pool is a proxy, **not ground truth**. It is every candidate with score ≥ 0.5 from any setting, clustered at IoU 0.5, giving 1,351 people-frames, 149 of them small. No identity, tracking or event result was used. From the fixture, only `video_sha256`, `seed_frame` and `seed_bbox` were read.

## Thresholds in use

| Setting | Value |
|---|---|
| Person score (objectness × person class) | ≥ `detector_min_score` 0.10 |
| NMS | greedy, IoU > `detector_nms_iou` 0.65 suppresses (kept high so two overlapping people both survive) |
| Tiles | 2×2, overlap 0.2, plus a full-frame pass |
| Tile-edge margin | 0.02 of the tile size |
| Fragment rule | containment ≥ 0.8 **and** span ≥ 0.7 across the cut |
| Edge-touch margin | 2 px |

## Failures found during development and the general rules that fixed them

Each fix is general, tested with neutral synthetic boxes, and contains no clip, frame or person knowledge.

1. **Tile-cut fragments beat whole boxes.** A higher-scoring piece of a person cut by a tile edge suppressed that person's whole full-frame box. **Rule:** a box touching an internal tile edge can never suppress an uncut box. It is dropped if it overlaps a kept whole box above the NMS IoU, or is a fragment of one. Tests: `test_cut_fragment_of_whole_box_dropped_even_with_higher_score`, `test_two_cut_pieces_of_one_person`.
2. **Containment alone removed a partly hidden second person.** **Rule:** a cut box is a fragment of a larger box only if it is ≥ 0.8 inside it **and** spans ≥ 0.7 of it across the cut. A narrower box inside someone's box is a separate, partly occluded person and is kept. Test: `test_partly_hidden_second_person_kept`.
3. **Slivers at tile boundaries.** The first cloud evaluation showed many small or wide boxes starting exactly at a tile edge. **Rule:** a cut box lying wholly inside a neighbouring tile that extends past the cut is dropped, because that tile saw the same region uncut. Test: `test_cut_box_seen_uncut_by_neighbour_tile`. After this fix, wide boxes for 2×2 tiles fell from 287 to 110 (on the cloud and Mac runs respectively), and most of the remainder are people cut by the real bottom image border.

## Test results

- **Mac:** 135 tests OK (Python 3.10.12, onnxruntime 1.23.2, cv2 5.0.0). This includes the model-backed and real-footage detector tests (`SPANA_GOLD_VIDEO` set) and the R1-A and R1-B suites.
- **Cloud:** 135 tests OK (onnxruntime 1.29.0).

## Process note

One transfer to the Mac reported success but left three files at their previous version. The fresh hash comparison caught it before anything ran. The files were resent under new transfer paths and **all 17 R1-C files were re-verified identical** before the authoritative tests and evaluation.

## Known limits

- **The proxy pool is not ground truth.** Real per-person recall and precision need box labels (R2).
- **2×2 tiling costs about 6× the full-frame runtime** (about 3 s per frame on the Mac VM with one ONNX thread). That is acceptable for development but must be revisited before production; options are more threads, a re-exported larger-input model, or a GPU.
- **Scores differ slightly between onnxruntime 1.23.2 (Mac) and 1.29.0 (cloud)**: for example, seed IoU 0.867 against 0.872. Determinism holds within one environment; it is tested.
- **Tiling can leave two boxes for one partly occluded person** (a whole low-score box plus a tile box). Later layers must treat competitors as possibly duplicated.
- **YOLOX remains LICENCE_UNCLEAR** for production because of the COCO caveat.

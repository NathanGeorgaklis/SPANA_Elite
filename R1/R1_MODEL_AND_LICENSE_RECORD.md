# R1_MODEL_AND_LICENSE_RECORD.md

SPANA engine v2-rebuild, R1. Pre-coding gate, 2026-10-24.

**No model has been downloaded.** Every SHA-256 below that is marked *NOT YET DOWNLOADED* will be filled in from the downloaded file, and pinned in `player_tracker_rb/config.py`, only after Nathan approves.

**Status vocabulary** (R1 prompt §16):
- PRODUCTION_APPROVED
- DEVELOPMENT_ONLY
- LICENCE_UNCLEAR
- NOT_APPROVED

**Verification markers.** Licence facts marked **[verified 2026-10-24]** were read today from the primary repository or licence page. Facts marked **[general knowledge]** were not re-verified today and must be checked when the file is downloaded.

---

## A. Proposed for R1

### A1. Person detector: YOLOX-s (primary), with YOLOX-m as the fallback if recall is insufficient (D17)

| Field | Value |
|---|---|
| Name | YOLOX-s (and YOLOX-m) |
| Version | Megvii-BaseDetection/YOLOX release tag `0.1.1rc0` (current models) **[verified 2026-10-24, repo README]** |
| Source | https://github.com/Megvii-BaseDetection/YOLOX (official `.pth` in GitHub releases). The README links ONNX export via `demo/ONNXRuntime`; whether official `.onnx` files are attached to the release is **not confirmed**. A third-party ONNX conversion exists at https://huggingface.co/skillsafe-ai/yolox-s (yolox_s.onnx, 34.20 MB, sha256 c5c2d13e…8063). That copy is a conversion and will **not** be used unless it is verified against our own export. |
| SHA-256 | NOT YET DOWNLOADED |
| Code licence | Apache-2.0 **[verified 2026-10-24]** |
| Weight licence | Apache-2.0 (weights distributed in the same repository's releases; the HF model card also states the weights remain under the upstream Apache-2.0 licence) **[verified 2026-10-24]** |
| Training data | COCO. INFERRED from the repo's "Reproduce our results on COCO" instructions and its COCO mAP tables. COCO annotations are CC BY 4.0 and images are Flickr with mixed licences **[general knowledge]**. |
| Known restrictions | No field-of-use restriction in Apache-2.0. The only open question is any legal view on COCO-trained weights in a commercial product. |
| Commercial-use assessment | Strong production candidate. Apache-2.0 permits proprietary commercial use with attribution and NOTICE retention. |
| **Production status** | **LICENCE_UNCLEAR** (pending Nathan's or legal sign-off on the COCO training-data question). Used as the R1 development detector. |
| Role in R1 | `DetectorBackend` "yolox_onnx": person class only, raw confidence kept, input size per D17. |
| Runtime | ONNX Runtime 1.23.2, CPU, fixed threads (installed on the Mac VM). |

### A2. Appearance cue 1: tracker-owned LAB part-colour descriptor (NEW code, no model)

| Field | Value |
|---|---|
| Name / version | `player_tracker_rb.appearance.LabPartColour` v0.1 (to be written in R1) |
| Source | Own code |
| SHA-256 | the code file's hash, recorded at commit |
| Licence | Proprietary (SPANA); no third-party licence |
| Training data | none |
| **Production status** | **PRODUCTION_APPROVED** with respect to licensing, because it is own code. Identity performance is unvalidated until R2. |
| Role | Kit and part colour as **supporting** evidence only (D04). Never sufficient on its own for identity. |

### A3. Appearance cue 2: DINOv2 ViT-S/14 general-purpose embedding (ONNX export)

| Field | Value |
|---|---|
| Name | DINOv2 ViT-S/14 (`dinov2_vits14`) |
| Version | Checkpoint from facebookresearch/dinov2. The exact file and commit will be recorded at download. |
| Source | https://github.com/facebookresearch/dinov2 |
| SHA-256 | NOT YET DOWNLOADED (source `.pth` and our exported `.onnx` will both be recorded) |
| Code licence | Apache-2.0. The README states "DINOv2 code and model weights are released under the Apache License 2.0" **[verified 2026-10-24]**. |
| Weight licence | Apache-2.0 **[verified 2026-10-24]**. The same README puts Cell-DINO and XRay-DINO under **non-commercial** licences; those models are **not** proposed. An older GitHub issue (#128) records the weights as CC-BY-NC at the original release, so the licence **changed**. The LICENSE file shipped with the exact checkpoint must be re-checked at download time. |
| Training data | LVD-142M, curated web images (from the DINOv2 paper) **[general knowledge]** |
| Known restrictions | None under Apache-2.0. The licence history means the downloaded checkpoint must be checked against the current licence. |
| Commercial-use assessment | Production candidate. |
| **Production status** | **LICENCE_UNCLEAR** (pending download-time licence confirmation and Nathan's sign-off). Used as the R1 development embedding. |
| Role in R1 | General visual embedding of player crops, used **contrastively** (target bank vs competitor banks, D08). **It is not a person-ReID model and does not reproduce `yolo26n-reid.onnx`. This is NEW / UNVERIFIED behaviour.** Its same-kit discrimination is unmeasured until R2. |
| Export | A one-off export with `tools_rb/export_models.py`. PyTorch is used only in the cloud workspace for the export; the ONNX output is hashed on the Mac. |

### A4. Runtime libraries (not models; recorded for completeness)

| Library | Version on the Mac VM | Licence | Status |
|---|---|---|---|
| numpy | 2.2.6 | BSD-3-Clause **[general knowledge]** | PRODUCTION_APPROVED |
| opencv-python (cv2) | 5.0.0 | Apache-2.0 (OpenCV ≥ 4.5) **[general knowledge]** | PRODUCTION_APPROVED |
| onnxruntime | 1.23.2 | MIT **[general knowledge]** | PRODUCTION_APPROVED |

---

## B. Considered and not proposed for production

| Model | Code licence | Weight licence / data | Status | Note |
|---|---|---|---|---|
| Ultralytics YOLOv8s / YOLOv8x / YOLO11 / YOLO26 (`*.pt`) | AGPL-3.0 | "All Ultralytics YOLO trained models fall under the AGPL-3.0 License by default". Closed-source or commercial use, including SaaS and internal use, requires a paid **Enterprise License** **[verified 2026-10-24, ultralytics.com/license]**. | **DEVELOPMENT_ONLY** | Optional *comparison only*. For example, a downloaded `yolov8s.pt` could be hashed against the historical 1f47a78b… to learn whether it is byte-identical to the lost file. Not proposed, and not downloaded. It would be isolated in its own backend class. |
| OSNet and other person-ReID models (torchreid, MIT code) trained on Market-1501 / DukeMTMC / MSMT17 | MIT | Research datasets. Commercial terms are unclear or restrictive, and DukeMTMC was withdrawn **[general knowledge]**. | **NOT_APPROVED** (production); DEVELOPMENT_ONLY at most | Could serve as an R2 *comparison* embedding with approval. It is not part of R1. |
| Historical `yolo26n-reid.onnx` (sha256 8529c383…) | UNKNOWN | UNKNOWN; origin undocumented, and the file is missing | **NOT_APPROVED** | Must not be recreated or approximated (R1 prompt §18). |
| Historical `yolov8s.pt` / `yolov8x.pt` (1f47a78b… / 3df4ada6…) | AGPL-3.0 (if Ultralytics) | as above | **DEVELOPMENT_ONLY** | Missing; hashes recorded for future identification only. |
| EasyOCR (jersey OCR) | Apache-2.0 (code) **[general knowledge]** | weights unverified | **LICENCE_UNCLEAR** | Off by default in R1 (spurious numbers historically). |

---

## C. Rules this record enforces

- A model file is loaded only if its SHA-256 equals the value pinned in `config.py`. Otherwise the run is refused.
- Every tracking JSON records each model's filename, SHA-256 and the status listed here.
- No DEVELOPMENT_ONLY or LICENCE_UNCLEAR model may be described as "the production model".
- Changing a model is a recorded parameter change (R1 prompt §68): old value, new value, reason, evidence, and the tests and results affected.

## Sources consulted at the gate

- [YOLOX repository](https://github.com/Megvii-BaseDetection/YOLOX)
- [YOLOX-s ONNX model card (third-party conversion)](https://huggingface.co/skillsafe-ai/yolox-s)
- [DINOv2 repository](https://github.com/facebookresearch/dinov2)
- [DINOv2 licence issue #128 (historical CC-BY-NC)](https://github.com/facebookresearch/dinov2/issues/128)
- [Ultralytics licensing](https://ultralytics.com/license)

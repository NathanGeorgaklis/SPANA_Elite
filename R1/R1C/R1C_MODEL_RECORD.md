# R1-C model record: person detector

Recorded 2026-10-24 (Sydney), before any detector code used a model.

| Item | YOLOX-s (primary) | YOLOX-m (approved fallback) |
|---|---|---|
| Authoritative source | `https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx` | `.../0.1.1rc0/yolox_m.onnx` |
| Where the link is published | Official repository, `demo/ONNXRuntime/README.md` ("Download ONNX models" table) | same |
| Release / version | tag `0.1.1rc0`, commit `e1052df71842031413f6030723c3607b839c80ce` (`git ls-remote`, 2026-10-24) | same |
| Repository HEAD at download | `6ddff4824372906469a7fae2dc3206c7aa4bbaee` | same |
| File size | 35,858,002 bytes | 101,259,744 bytes |
| **SHA-256** | **`c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063`** | **`21ff6cfdeb53b013bac2249599e55f00bff3cfdfdab37ed7a4620818c1d15b3f`** |
| Input / output | `images` [1, 3, 640, 640] float32 → `output` [1, 8400, 85] | same |
| Classes | 80 (COCO); person = index 0 | same |
| Code licence | Apache-2.0 (repository `LICENSE`, sha256 `0ec3668d3274bcf29e8a29e9576d5a2cd96fc78d3c5bec4387355a796e5d9088`) | same |
| Weight licence | Apache-2.0: the official release assets of the Apache-2.0 repository | same |
| Training data | COCO (caveat below) | same |
| **Status** | **LICENCE_UNCLEAR**: approved for DEVELOPMENT / EVALUATION (R1 approval §2). Not PRODUCTION_APPROVED until the COCO training-data question is resolved for SPANA's commercial use. | same |

**Corroboration.** The `yolox_s.onnx` SHA-256 is byte-identical to the third-party Hugging Face copy noted at the R1 gate (`skillsafe-ai/yolox-s`, `c5c2d13e…8063`). Two independent sources therefore give the same bytes.

## Preprocessing convention (verified, not assumed)

The convention is taken from the official reference code, copied with hashes in `R1/R1C/yolox_reference/`:
- `yolox/data/data_augment.py preproc()`:
  - letterbox scale `r = min(S/h, S/w)`;
  - `cv2.resize` with INTER_LINEAR to `(int(w*r), int(h*r))`;
  - placed top-left on an S×S canvas filled with 114;
  - BGR, 0–255, float32, CHW, **no mean/std normalisation**.
- `demo_postprocess()`: strides 8, 16, 32; `xy = (raw + grid) * stride`; `wh = exp(raw) * stride`.
- `onnx_inference.py`: score = objectness × class score; boxes `/= r`.

This was checked on one real frame. The documented convention gives person scores up to 0.89, while the legacy normalised (RGB, mean/std) variant gives at most 0.001. So the model was exported without normalisation, and the convention is unambiguous.

## Consequence for input size

The official ONNX files have a **fixed 640×640 input**. Larger network inputs (960, 1280) would need a re-export from the `.pth` weights with PyTorch (approved tooling: `tools_rb/export_models.py`, not built). Instead, R1-C evaluates **overlapping tiles at 640**, which gives the same effect: more network pixels per small player, with the official model unchanged.

## Storage and backup

- `models/yolox_s.onnx` and `models/yolox_m.onnx` are git-ignored, so they are not on GitHub.
- They are on the Mac in `SPANA_Engine_Rebuild/models/`, and in every full archive snapshot.
- Their hashes are pinned in `player_tracker_rb/config.py` (`detector_model_sha256`, for YOLOX-s) and listed in `R1/R1C/MODEL_HASHES.sha256`.
- The detector refuses to load a file whose hash differs.

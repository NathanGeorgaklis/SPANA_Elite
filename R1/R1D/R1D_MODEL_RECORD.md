# R1-D model record: generic appearance embedding (DINOv2 ViT-S/14)

Recorded 2026-10-24 (Sydney) **before** any appearance code used the model. The licence was checked first, as required by R1 approval §5.

## What this model is and is not
- DINOv2 ViT-S/14 is a **generic self-supervised visual embedding** (Meta AI, *DINOv2: Learning Robust Visual Features without Supervision*, Oquab et al., 2023).
- It is **not** a football-player or person re-identification model.
- It does **not** reproduce the historical `yolo26n-reid.onnx`.
- Whether it can tell people apart is measured, not assumed (see `R1D_STAGE_RECORD.md`).

## Licence (captured before download)

| Source | Statement | Verified |
|---|---|---|
| github.com/facebookresearch/dinov2 `README.md` ("License") | "DINOv2 code and model weights are released under the Apache License 2.0." Only Cell-DINO and XRay-DINO are non-commercial; neither is used. | 2026-10-24 (sha256 `d1bc2e96…`, copy in `dinov2_reference/repo_README.md`) |
| github.com/facebookresearch/dinov2 `LICENSE` | Apache License, Version 2.0, January 2004 | 2026-10-24 (sha256 `600cc67c…`, `dinov2_reference/repo_LICENSE`) |
| Hugging Face `facebook/dinov2-small` model card | `license: apache-2.0` (card metadata and API `cardData`). The card notes it was written by the Hugging Face team, not the DINOv2 authors. | 2026-10-24 (sha256 `4c20dca4…`, `dinov2_reference/hf_model_card_README.md`) |

- **Agreement:** the model-card licence agrees with the repository basis recorded at the gate (A5). This closes the open item.
- **Training data:** LVD-142M, curated web images (from the paper). This is a training-data caveat comparable to YOLOX's COCO caveat.
- **Status: LICENCE_UNCLEAR.** Apache-2.0 is confirmed by three official texts. It is approved for DEVELOPMENT / EVALUATION (R1 approval §4) and is not PRODUCTION_APPROVED until Nathan's sign-off and the training-data question are resolved.

## Source files (pinned)

| File | Source | SHA-256 |
|---|---|---|
| `model.safetensors` | HF `facebook/dinov2-small` at commit `ed25f3a31f01632728cabb09d1542f84ab7b0056` | `ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1` |
| `config.json` | same | `1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d` |
| `preprocessor_config.json` | same | `14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828` |
| `dinov2_vits14_pretrain.pth` | official Meta checkpoint, `dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth` (linked from the repository README); used for cross-checking | `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9` |

- The DINOv2 repository HEAD at download was `7764ea0f912e53c92e82eb78a2a1631e92725fc8`.
- **Weights cross-check:** the Hugging Face weights are bit-for-bit identical to the official Meta checkpoint. The maximum absolute difference was 0.0 on `cls_token`, `pos_embed`, `patch_embed`, blocks 0 and 11 qkv, block 11 mlp, and the final norm (`DINOV2_EXPORT_RECORD.json`).
- **Independent downloads:** the cloud and the Mac downloaded all five source files independently and got identical hashes.

## Exported model

| Item | Value |
|---|---|
| File | `models/dinov2_vits14_cls_patchmean_224x112.onnx` (git-ignored; on the Mac and in snapshots) |
| **SHA-256** | **`0ab62713cf7759dd820b53576b39010cb0fe8dcfbe22af444c945c519bf9efea`** (88,392,522 bytes) |
| Tool | `tools_rb/export_models.py`. Inputs are pinned by hash and the export refuses on any mismatch. |
| Exporter | torch 2.14.1+cpu, transformers 5.20.2, onnx 1.23.2, `torch.onnx.export` (TorchScript, opset 17) |
| Reproducibility | Two exports gave a byte-identical file (same SHA-256) |
| Accuracy | ONNX vs PyTorch maximum absolute difference: 1.6e-05 (cls) and 1.7e-05 (patch_mean) |
| Input | `pixel_values` float32 [1, 3, 224, 112]: RGB, x/255, ImageNet mean/std, 16 × 8 patches of 14 px. Position embeddings are interpolated in-graph by transformers (bicubic). |
| Outputs | `cls` [1, 384] (final-layernorm CLS token); `patch_mean` [1, 384] (mean of the final-layernorm patch tokens) |
| Transfer to the Mac | The 30 MB file limit meant four parts were joined on the Mac. The joined file's SHA-256 equals the export's. |

The 224 × 112 input (2:1) suits a standing person crop. The crop is letterboxed with the normalisation-mean colour, so the padding is exactly 0 after normalisation.

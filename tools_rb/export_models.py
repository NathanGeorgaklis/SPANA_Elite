# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R1 approval §5 and §7 (export_models.py: pinned inputs, hashed outputs, recorded settings);
#                 R1/R1D/R1D_MODEL_RECORD.md; facebook/dinov2-small (HF, Apache-2.0) config/preprocessor files
"""One-off export of the DINOv2 ViT-S/14 backbone to ONNX for player_tracker_rb's generic appearance embedding.

This is BUILD tooling, not tracker code. It needs PyTorch + transformers + onnx (cloud export environment only);
the tracker itself only ever loads the resulting .onnx file with ONNX Runtime, and only if its SHA-256 matches the
pinned value in player_tracker_rb/config.py.

Inputs (all pinned by SHA-256; the export refuses to run on any mismatch):
  * HF facebook/dinov2-small at commit ed25f3a31f01632728cabb09d1542f84ab7b0056: model.safetensors, config.json
  * official Meta checkpoint dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth (cross-check only:
    the HF weights must equal the official ones)

Output: dinov2_vits14_cls_patchmean_224x112.onnx
  input  "pixel_values" float32 [1, 3, 224, 112]  (RGB, ImageNet mean/std normalised; 16 x 8 patches of 14 px)
  output "cls"        float32 [1, 384]  final-layernorm CLS token
  output "patch_mean" float32 [1, 384]  mean of the final-layernorm patch tokens
The 224 x 112 (2:1) input matches the shape of a standing person crop; DINOv2's position embeddings are interpolated
to the 16 x 8 patch grid inside the exported graph (transformers' own interpolation, bicubic).

Usage: PYTHON_WITH_TORCH tools_rb/export_models.py SRC_DIR OUT_DIR
  SRC_DIR holds model.safetensors, config.json, dinov2_vits14_pretrain.pth.
"""
import hashlib
import json
import os
import platform
import sys

PINNED = {
    "model.safetensors": "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1",
    "config.json": "1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d",
    "dinov2_vits14_pretrain.pth": "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9",
}
HF_COMMIT = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
INPUT_HW = (224, 112)
OPSET = 17
OUT_NAME = "dinov2_vits14_cls_patchmean_224x112.onnx"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(src, out_dir):
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    import transformers
    from safetensors.torch import load_file
    from transformers import Dinov2Config, Dinov2Model

    for name, want in PINNED.items():
        got = sha256(os.path.join(src, name))
        if got != want:
            raise SystemExit(f"{name}: sha256 {got} != pinned {want}")
    torch.manual_seed(0)
    cfg = Dinov2Config.from_json_file(os.path.join(src, "config.json"))
    # transformers' own loader (it maps checkpoint key names across transformers versions).
    model, info = Dinov2Model.from_pretrained(src, local_files_only=True, output_loading_info=True)
    bad = {k: v for k, v in info.items() if v and k in ("missing_keys", "unexpected_keys", "mismatched_keys",
                                                         "error_msgs")}
    if bad:
        raise SystemExit(f"loading problems: {bad}")
    model.eval()
    raw = load_file(os.path.join(src, "model.safetensors"))

    # Cross-check against the official Meta checkpoint (same weights, different key names).
    off = torch.load(os.path.join(src, "dinov2_vits14_pretrain.pth"), map_location="cpu", weights_only=True)
    sd = model.state_dict()

    def qkv(layer):
        for names in (("q_proj", "k_proj", "v_proj"), ("attention.query", "attention.key", "attention.value")):
            keys = [f"encoder.layer.{layer}.attention.{n}.weight" for n in names]
            if all(k in sd for k in keys):
                return torch.cat([sd[k] for k in keys], 0)
        raise SystemExit("cannot locate q/k/v weights in the loaded model")
    pairs = [("cls_token", "embeddings.cls_token"), ("pos_embed", "embeddings.position_embeddings"),
             ("patch_embed.proj.weight", "embeddings.patch_embeddings.projection.weight"),
             ("norm.weight", "layernorm.weight"), ("norm.bias", "layernorm.bias"),
             ("blocks.11.mlp.fc2.weight", "encoder.layer.11.mlp.fc2.weight")]
    cross = {a: float((off[a] - sd[b]).abs().max()) for a, b in pairs}
    cross["blocks.0.attn.qkv.weight"] = float((off["blocks.0.attn.qkv.weight"] - qkv(0)).abs().max())
    cross["blocks.11.attn.qkv.weight"] = float((off["blocks.11.attn.qkv.weight"] - qkv(11)).abs().max())
    # and the raw safetensors file itself (independent of transformers' renaming)
    cross["raw_safetensors.cls_token"] = float((off["cls_token"] - raw["embeddings.cls_token"]).abs().max())
    if max(cross.values()) != 0.0:
        raise SystemExit(f"HF weights differ from the official checkpoint: {cross}")

    class Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, pixel_values):
            h = self.m(pixel_values=pixel_values).last_hidden_state      # final layernorm applied
            return h[:, 0], h[:, 1:].mean(dim=1)

    w = Wrapper(model).eval()
    x = torch.randn(1, 3, *INPUT_HW, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        ref_cls, ref_pm = w(x)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, OUT_NAME)
    torch.onnx.export(w, (x,), out, input_names=["pixel_values"], output_names=["cls", "patch_mean"],
                      opset_version=OPSET, dynamo=False, do_constant_folding=True)
    onnx.checker.check_model(onnx.load(out))
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess = ort.InferenceSession(out, sess_options=so, providers=["CPUExecutionProvider"])
    o_cls, o_pm = sess.run(None, {"pixel_values": x.numpy()})
    diff = {"cls": float(np.abs(o_cls - ref_cls.numpy()).max()), "patch_mean": float(np.abs(o_pm - ref_pm.numpy()).max())}
    if max(diff.values()) > 1e-3:
        raise SystemExit(f"ONNX output differs from PyTorch: {diff}")
    rec = {"tool": "tools_rb/export_models.py", "hf_repo": "facebook/dinov2-small", "hf_commit": HF_COMMIT,
           "inputs_sha256": PINNED, "official_checkpoint_crosscheck_max_abs_diff": cross,
           "output_file": OUT_NAME, "output_sha256": sha256(out), "output_bytes": os.path.getsize(out),
           "input": {"name": "pixel_values", "shape": [1, 3, *INPUT_HW], "dtype": "float32",
                     "convention": "RGB, x/255, (x - mean)/std with ImageNet mean [0.485, 0.456, 0.406] and std "
                                   "[0.229, 0.224, 0.225] (preprocessor_config.json)"},
           "outputs": {"cls": [1, cfg.hidden_size], "patch_mean": [1, cfg.hidden_size]},
           "opset": OPSET, "exporter": "torch.onnx.export (TorchScript, dynamo=False)",
           "onnx_vs_torch_max_abs_diff": diff,
           "environment": {"python": platform.python_version(), "torch": torch.__version__,
                           "transformers": transformers.__version__, "onnx": onnx.__version__,
                           "onnxruntime": ort.__version__, "numpy": np.__version__, "platform": platform.platform()}}
    with open(os.path.join(out_dir, "EXPORT_RECORD.json"), "w") as f:
        json.dump(rec, f, indent=2, sort_keys=True)
    print(json.dumps(rec, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R1/R1D/R1D_MODEL_RECORD.md (DINOv2 ViT-S/14, facebook/dinov2-small @ ed25f3a3, Apache-2.0, ONNX export
#                 0ab62713...); facebook/dinov2-small preprocessor_config.json; R1 approval §4 (generic embedding,
#                 not player ReID truth)
"""
player_tracker_rb.embedding_dinov2: DINOv2 ViT-S/14 ONNX backend for the generic appearance embedding.

DINOv2 is a GENERIC self-supervised visual embedding. It is not a football player re-identification model and it
does not reproduce the historical yolo26n-reid.onnx. Its usefulness for telling people apart must be measured.

Preprocessing, preprocess_crop() (one person crop -> [1, 3, H, W], H x W = embedding_input_hw = 224 x 112):
  1. BGR uint8 -> RGB.
  2. Resize preserving aspect ratio, scale = min(H / h, W / w); INTER_AREA when shrinking (anti-aliased),
     INTER_LINEAR when enlarging; centre it on an H x W canvas filled with the normalisation-mean colour rounded to
     uint8 (so the padding is approximately 0 after normalisation: within 0.01 per channel).
  3. x / 255, then (x - mean) / std per channel (ImageNet mean/std from the model's preprocessor_config).
  (The model's own HF preprocessing uses a bicubic resize and a square centre crop; a square crop would cut a standing
  person's head and feet, so this backend letterboxes into a 2:1 input instead. Documented, deliberate difference.)
Outputs are read BY NAME: "cls" [1, D] and "patch_mean" [1, D]. embedding_pooling picks cls, or cls + patch_mean
(each L2-normalised, concatenated, L2-normalised). The returned vector is always unit length.

The model is loaded only if its SHA-256 equals embedding_model_sha256, its input is float32 [1, 3, H, W] and its two
outputs are [1, D] with a fixed positive D; otherwise EmbeddingModelError. ONNX Runtime runs with fixed threads and
sequential execution (deterministic within one environment).
"""
from __future__ import annotations

import hashlib
import os
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from .appearance import AppearanceError, EmbeddingBackend, EmbeddingModelError, l2_normalise
from .config import TrackerConfig

BACKEND_NAME = "dinov2_onnx"
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_NAMES = ("cls", "patch_mean")


def _sha256(path: str, chunk: int) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def preprocess_crop(crop: np.ndarray, cfg: TrackerConfig) -> np.ndarray:
    """BGR uint8 crop -> float32 [1, 3, H, W] network input (see the module docstring). No model needed."""
    mean = np.asarray(cfg.get("embedding_mean_rgb"), dtype=np.float64)
    std = np.asarray(cfg.get("embedding_std_rgb"), dtype=np.float64)
    if (crop is None or not isinstance(crop, np.ndarray) or crop.ndim != 3 or crop.dtype != np.uint8
            or crop.shape[2] != len(mean) or crop.shape[0] < 1 or crop.shape[1] < 1):
        raise AppearanceError("embedding needs a non-empty BGR uint8 crop of shape (h, w, 3)")
    hh, ww = (int(v) for v in cfg.get("embedding_input_hw"))
    h, w = crop.shape[:2]
    s = min(hh / h, ww / w)
    nh, nw = max(1, min(hh, int(round(h * s)))), max(1, min(ww, int(round(w * s))))
    rgb = cv2.cvtColor(np.ascontiguousarray(crop), cv2.COLOR_BGR2RGB)
    interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(rgb, (nw, nh), interpolation=interp)
    levels = float(np.iinfo(np.uint8).max)
    canvas = np.empty((hh, ww, len(mean)), dtype=np.uint8)
    canvas[:] = np.round(mean * levels).astype(np.uint8)
    y0, x0 = (hh - nh) // 2, (ww - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    x = (canvas.astype(np.float64) / levels - mean) / std
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=np.float32)


def pool(cls: np.ndarray, patch_mean: np.ndarray, pooling: str) -> np.ndarray:
    if pooling == "cls":
        return l2_normalise(cls)
    if pooling == "cls_patchmean":
        return l2_normalise(np.concatenate([l2_normalise(cls), l2_normalise(patch_mean)]))
    raise AppearanceError(f"unknown pooling {pooling!r}")


class Dinov2OnnxEmbedding(EmbeddingBackend):
    def __init__(self, cfg: TrackerConfig, model_path: Optional[str] = None):
        self._cfg = cfg
        path = model_path or cfg.get("embedding_model_file")
        path = path if os.path.isabs(path) else os.path.join(PACKAGE_ROOT, path)
        self._path = path
        if not os.path.isfile(path):
            raise EmbeddingModelError(f"embedding model not found: {path}")
        digest = _sha256(path, cfg.get("hash_chunk_bytes"))
        if digest != cfg.get("embedding_model_sha256"):
            raise EmbeddingModelError(f"embedding model sha256 {digest} != pinned {cfg.get('embedding_model_sha256')}")
        self._sha = digest
        so = ort.SessionOptions()
        so.intra_op_num_threads = cfg.get("onnx_intra_op_threads")
        so.inter_op_num_threads = cfg.get("onnx_inter_op_threads")
        if cfg.get("onnx_sequential_execution"):
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        try:
            self._sess = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
        except Exception as e:  # noqa: BLE001
            raise EmbeddingModelError(f"cannot load embedding model {path}: {e}") from e
        self._dim = self._check_signature()
        self._hw = tuple(int(v) for v in cfg.get("embedding_input_hw"))

    def _check_signature(self) -> int:
        c = self._cfg
        hh, ww = (int(v) for v in c.get("embedding_input_hw"))
        ins = self._sess.get_inputs()
        if len(ins) != 1 or list(ins[0].shape) != [1, len(c.get("embedding_mean_rgb")), hh, ww] \
                or ins[0].type != "tensor(float)":
            raise EmbeddingModelError(f"model input {[(i.name, i.shape, i.type) for i in ins]} != float32 "
                                      f"[1, 3, {hh}, {ww}]")
        self._in = ins[0].name
        outs = {o.name: o.shape for o in self._sess.get_outputs()}
        if set(outs) != set(OUTPUT_NAMES):
            raise EmbeddingModelError(f"model outputs {sorted(outs)} != {list(OUTPUT_NAMES)}")
        dims = set()
        for name in OUTPUT_NAMES:
            shp = outs[name]
            if len(shp) != 2 or shp[0] != 1 or not isinstance(shp[1], int) or shp[1] < 1:
                raise EmbeddingModelError(f"output {name} shape {shp} is not [1, D] with a fixed D")
            dims.add(shp[1])
        if len(dims) != 1:
            raise EmbeddingModelError(f"cls and patch_mean dimensions differ: {outs}")
        return dims.pop()

    @property
    def dim(self) -> int:
        return self._dim * (2 if self._cfg.get("embedding_pooling") == "cls_patchmean" else 1)

    def preprocess(self, crop: np.ndarray) -> np.ndarray:
        return preprocess_crop(crop, self._cfg)

    def raw_outputs(self, crop: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        cls, pm = self._sess.run(list(OUTPUT_NAMES), {self._in: self.preprocess(crop)})
        return np.asarray(cls[0], dtype=np.float64), np.asarray(pm[0], dtype=np.float64)

    def embed(self, crop: np.ndarray) -> np.ndarray:
        cls, pm = self.raw_outputs(crop)
        return pool(cls, pm, self._cfg.get("embedding_pooling"))

    def provenance(self) -> Dict[str, Any]:
        c = self._cfg
        return {"backend": BACKEND_NAME, "model_file": os.path.basename(self._path), "model_sha256": self._sha,
                "model_status": c.get("embedding_model_status"), "input_hw": list(self._hw),
                "pooling": c.get("embedding_pooling"), "dim": self.dim, "onnxruntime_version": ort.__version__,
                "note": "generic visual embedding (DINOv2 ViT-S/14), not a person/football re-identification model"}

    def model_record(self) -> Dict[str, Any]:
        return {"role": "appearance_embedding", "filename": os.path.basename(self._path), "sha256": self._sha,
                "status": self._cfg.get("embedding_model_status")}

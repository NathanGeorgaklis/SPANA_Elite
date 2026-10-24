# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: Megvii YOLOX release 0.1.1rc0 (Apache-2.0): yolox/data/data_augment.py preproc(),
#                 yolox/utils/demo_utils.py demo_postprocess(), demo/ONNXRuntime/onnx_inference.py (read 2026-10-24,
#                 copies and hashes in R1/R1C/yolox_reference/); R1/R1_DESIGN_DECISIONS.md D03, D17, D20, D22
"""
player_tracker_rb.detector_yolox: YOLOX ONNX Runtime person-detector backend.

Everything YOLOX-specific lives here, and nothing YOLOX-specific leaves this module (detect() returns generic
PersonCandidate records). This is an independent re-implementation of the YOLOX inference conventions, written
from the Apache-2.0 reference code (see the evidence line); no YOLOX source file is imported or copied.

Pipeline per pass (full frame, or one tile):
  1. letterbox: scale = min(S/h, S/w); resize with INTER_LINEAR to (int(w*scale), int(h*scale)); place at the
     top-left of an S x S canvas filled with the pad value (114); BGR, 0-255, float32, CHW; no normalisation.
  2. inference: ONNX Runtime CPU, fixed thread counts, sequential execution (deterministic).
  3. decode: for strides (8, 16, 32): xy = (raw_xy + grid) * stride, wh = exp(raw_wh) * stride (cx, cy, w, h).
  4. person score = objectness * class_score[person]; keep score >= detector_min_score.
  5. restore: network coordinates -> source image via the exact letterbox inverse.
Then all passes are merged (global NMS at detector_nms_iou; tile-cut fragment removal) and clipped to the image.

The model file is loaded only if its SHA-256 equals the pinned config value; its input must be [1, 3, S, S] with
S == detector_input_size, and its output [1, N, 5 + C] with N matching the strides. Otherwise: DetectorModelError.
"""
from __future__ import annotations

import hashlib
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from .config import TrackerConfig
from .detector import (CUT_BOTTOM, CUT_LEFT, CUT_RIGHT, CUT_TOP, DetectorBackend, DetectorError,
                       DetectorModelError, PersonCandidate, Tile, finalize, letterbox_for, merge_passes,
                       resized_extent, seen_uncut_elsewhere, tile_layout)

BACKEND_NAME = "yolox_onnx"
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(PACKAGE_ROOT, path)


def _sha256(path: str, chunk: int) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


class YoloxOnnxDetector(DetectorBackend):
    def __init__(self, cfg: TrackerConfig, model_path: Optional[str] = None):
        self._cfg = cfg
        path = _resolve(model_path or cfg.get("detector_model_file"))
        self._path = path
        if not os.path.isfile(path):
            raise DetectorModelError(f"detector model not found: {path}")
        digest = _sha256(path, cfg.get("hash_chunk_bytes"))
        pinned = cfg.get("detector_model_sha256")
        if digest != pinned:
            raise DetectorModelError(f"detector model sha256 {digest} != pinned {pinned} ({path})")
        self._sha = digest
        so = ort.SessionOptions()
        so.intra_op_num_threads = cfg.get("onnx_intra_op_threads")
        so.inter_op_num_threads = cfg.get("onnx_inter_op_threads")
        if cfg.get("onnx_sequential_execution"):
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        try:
            self._sess = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
        except Exception as e:  # noqa: BLE001 - any loader failure is a model failure
            raise DetectorModelError(f"cannot load detector model {path}: {e}") from e
        inp, out = self._sess.get_inputs()[0], self._sess.get_outputs()[0]
        size = cfg.get("detector_input_size")
        self._head = cfg.get("detector_box_fields")
        if list(inp.shape) != [1, cfg.get("detector_input_channels"), size, size]:
            raise DetectorModelError(f"model input {inp.shape} does not match detector_input_size {size}")
        strides = cfg.get("detector_strides")
        n_cells = sum((size // s) * (size // s) for s in strides)
        if len(out.shape) != 3 or out.shape[1] != n_cells or not isinstance(out.shape[2], int) or out.shape[2] <= self._head:
            raise DetectorModelError(f"model output {out.shape} does not match strides {strides} at {size}")
        self._n_classes = int(out.shape[2]) - self._head
        self._person = cfg.get("detector_person_class_index")
        if self._person >= self._n_classes:
            raise DetectorModelError(f"person class index {self._person} >= model classes {self._n_classes}")
        self._in_name = inp.name
        self._size = size
        self._grid, self._stride = self._make_grid(size, strides)
        self.last_pass_count = 0
        self.last_dropped_seen_uncut = 0

    @staticmethod
    def _make_grid(size: int, strides) -> Tuple[np.ndarray, np.ndarray]:
        grids, sts = [], []
        for s in strides:
            n = size // s
            xv, yv = np.meshgrid(np.arange(n), np.arange(n))
            g = np.stack((xv, yv), 2).reshape(-1, 2)
            grids.append(g)
            sts.append(np.full((g.shape[0], 1), s))
        return np.concatenate(grids, 0).astype(np.float64), np.concatenate(sts, 0).astype(np.float64)

    # ---- one pass -------------------------------------------------------------------------------
    def _preprocess(self, region: np.ndarray, origin_x: int, origin_y: int):
        h, w = int(region.shape[0]), int(region.shape[1])
        lb = letterbox_for(w, h, self._size, origin_x, origin_y)
        rw, rh = resized_extent(lb)
        canvas = np.full((self._size, self._size, region.shape[2]), self._cfg.get("detector_letterbox_pad_value"),
                         dtype=np.uint8)
        canvas[:rh, :rw] = cv2.resize(region, (rw, rh), interpolation=cv2.INTER_LINEAR)
        tensor = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32)
        return tensor, lb

    def _run_pass(self, region: np.ndarray, origin_x: int, origin_y: int):
        tensor, lb = self._preprocess(region, origin_x, origin_y)
        raw = self._sess.run(None, {self._in_name: tensor})[0][0].astype(np.float64)
        score = raw[:, self._head - 1] * raw[:, self._head + self._person]
        sel = np.nonzero(score >= self._cfg.get("detector_min_score"))[0]
        if sel.size == 0:
            return np.zeros((0, 4)), np.zeros(0), lb
        r = raw[sel]
        cxy = (r[:, 0:2] + self._grid[sel]) * self._stride[sel]
        wh = np.exp(r[:, 2:self._head - 1]) * self._stride[sel]
        half = wh / 2
        boxes_net = np.concatenate([cxy - half, cxy + half], 1)
        return lb.to_image(boxes_net), score[sel], lb

    # ---- public API -----------------------------------------------------------------------------
    def detect(self, image: np.ndarray) -> Tuple[PersonCandidate, ...]:
        if (image is None or image.ndim != 3 or image.dtype != np.uint8
                or image.shape[2] != self._cfg.get("detector_input_channels")):
            raise DetectorError("detect() needs a BGR uint8 image of shape (H, W, 3)")
        height, width = int(image.shape[0]), int(image.shape[1])
        jobs: List[Tile] = []
        tiles: List[Tile] = []
        tiled = self._cfg.get("detector_tiling") == "grid"
        if not tiled or self._cfg.get("detector_tile_include_full_frame"):
            jobs.append(Tile("full_frame", 0, 0, width, height))
        if tiled:
            cols, rows = (int(v) for v in self._cfg.get("detector_tile_grid"))
            tiles = tile_layout(width, height, cols, rows, self._cfg.get("detector_tile_overlap_fraction"))
            jobs += tiles
        all_boxes, all_scores, passes, internal = [], [], [], []
        self.last_dropped_seen_uncut = 0
        for job in jobs:
            name, x0, y0, x1, y1 = job.name, job.x0, job.y0, job.x1, job.y1
            boxes, scores, _ = self._run_pass(image[y0:y1, x0:x1], x0, y0)
            if boxes.shape[0] == 0:
                continue
            # Touching an internal tile edge (an edge that is not the image border) marks a possible cut fragment.
            cut = np.zeros(boxes.shape[0], dtype=np.int64)
            if name != "full_frame":
                frac = self._cfg.get("detector_tile_edge_margin_fraction")
                mx, my = frac * (x1 - x0), frac * (y1 - y0)
                cut |= np.where((boxes[:, 0] <= x0 + mx) & (x0 > 0), CUT_LEFT, 0)
                cut |= np.where((boxes[:, 1] <= y0 + my) & (y0 > 0), CUT_TOP, 0)
                cut |= np.where((boxes[:, 2] >= x1 - mx) & (x1 < width), CUT_RIGHT, 0)
                cut |= np.where((boxes[:, 3] >= y1 - my) & (y1 < height), CUT_BOTTOM, 0)
                # A cut box lying wholly inside a neighbouring tile that extends past the cut was seen uncut there.
                redundant = np.array([bool(c) and seen_uncut_elsewhere(b, int(c), job, tiles)
                                      for b, c in zip(boxes, cut)], dtype=bool)
                self.last_dropped_seen_uncut += int(redundant.sum())
                boxes, scores, cut = boxes[~redundant], scores[~redundant], cut[~redundant]
                if boxes.shape[0] == 0:
                    continue
            all_boxes.append(boxes)
            all_scores.append(scores)
            passes += [name] * boxes.shape[0]
            internal.append(cut)
        self.last_pass_count = len(jobs)
        if not all_boxes:
            return ()
        boxes = np.concatenate(all_boxes, 0)
        scores = np.concatenate(all_scores, 0)
        cut_sides = np.concatenate(internal, 0)
        keep = merge_passes(boxes, scores, passes, cut_sides, self._cfg)
        return finalize(boxes, scores, passes, width, height, keep, self._cfg.get("detector_edge_touch_margin_px"))

    def provenance(self) -> Dict[str, Any]:
        c = self._cfg
        return {"backend": BACKEND_NAME, "model_file": os.path.basename(self._path), "model_sha256": self._sha,
                "model_status": c.get("detector_model_status"), "input_size": self._size,
                "n_classes": self._n_classes, "person_class_index": self._person,
                "min_score": c.get("detector_min_score"), "nms_iou": c.get("detector_nms_iou"),
                "tiling": c.get("detector_tiling"), "tile_grid": list(c.get("detector_tile_grid")),
                "tile_overlap_fraction": c.get("detector_tile_overlap_fraction"),
                "tile_include_full_frame": c.get("detector_tile_include_full_frame"),
                "onnxruntime_version": ort.__version__, "providers": list(self._sess.get_providers())}

    def model_record(self) -> Dict[str, Any]:
        return {"role": "person_detector", "filename": os.path.basename(self._path), "sha256": self._sha,
                "status": self._cfg.get("detector_model_status")}

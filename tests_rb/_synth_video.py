# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R1-B instruction (frame-numbering and video-time tests); test helper only, no tracking logic
"""Test helper: writes small synthetic videos whose frames carry their own video frame index as a binary code,
so a test can read back exactly which source frame was decoded (independent of the fingerprint under test)."""
import os

import cv2
import numpy as np

W, H = 160, 96
BITS = 10                      # up to 1023 frames
BLOCK = W // BITS


def make_frame(index: int, width: int = W, height: int = H) -> np.ndarray:
    img = np.zeros((height, width, 3), np.uint8)
    block = width // BITS
    for b in range(BITS):
        if (index >> b) & 1:
            img[: height // 2, b * block:(b + 1) * block] = 255
    img[height // 2:, :] = (37 * index) % 200 + 20      # varies per frame, keeps fingerprints distinct
    return img


def read_index(img: np.ndarray) -> int:
    height, width = img.shape[:2]
    block = width // BITS
    top = img[: height // 2]
    v = 0
    for b in range(BITS):
        if top[:, b * block:(b + 1) * block].mean() > 127:
            v |= 1 << b
    return v


def write_video(path: str, n_frames: int, fps: float, fourcc: str = "MJPG", size=(W, H)) -> str:
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), fps, size)
    if not w.isOpened():
        raise RuntimeError(f"cannot write test video {path} with {fourcc}")
    for i in range(n_frames):
        w.write(make_frame(i, *size))
    w.release()
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        raise RuntimeError(f"test video {path} was not written")
    return path

# Frame-content fingerprint (R1-B)

## What it is
Each decoded frame gets a fingerprint, `thumb_sha1`, computed as follows:

1. Convert the decoded BGR frame to greyscale with `cv2.COLOR_BGR2GRAY`.
2. Resize it to `frame_thumbnail_size` (default 64×36 px) using `fingerprint_interpolation` (default `INTER_AREA`).
3. Store the thumbnail as row-major `uint8` bytes.
4. `thumb_sha1 = sha1(b"64x36:" + thumbnail_bytes)`.

The settings live in `player_tracker_rb/config.py`, so they also feed into the config hash. The method is also written into every provenance sidecar under `frame_alignment.fingerprint_method`.

A run additionally records `frame_thumbs_sha256`: one SHA-256 over the ordered list of per-frame fingerprints.

## What it is for
It is for **frame-alignment debugging only**. For example:
- checking that two runs decoded the same pixels at the same frame index;
- checking that a container seek and a decode-from-zero land on the same frame (the historical failure F15 was a resume landing +10 frames late);
- checking later that a frame_log record belongs to the expected video frame (`frame_log[].v2.frame_thumb_sha1`).

## What it is not
- It is **not** an appearance, ReID or identity feature. It summarises the whole frame, not a player.
- Detection, appearance, population, motion and state code must never read it. `tests_rb/test_package_hygiene.py::TestFingerprintIsNotAnIdentityFeature` enforces that only `video_io.py`, `provenance.py` and `output_schema.py` refer to it.

## Reproducibility limits
- The fingerprint is a hash of **decoded** pixels, so it is reproducible for the same file, the same OpenCV/FFmpeg build and the same settings.
- A different decoder build may decode a few pixels differently, which changes the hash. That is why the sidecar records `cv2_version` and the backend.
- Comparisons are meaningful between runs on the same machine and toolchain. Across machines, a mismatch means "investigate", not necessarily "misaligned".

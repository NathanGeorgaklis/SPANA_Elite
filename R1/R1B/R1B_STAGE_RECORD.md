# R1-B stage record: sequential video I/O and provenance

**Stage:** R1-B. Approved by Nathan on 2026-10-24 under the R1 implementation approval, with the R1-B scope message of 17:13 Sydney.

**Not built:** no detector, appearance, population, motion or state-machine logic. No event-engine code. No model downloaded. Task 3R and the holdout were not touched.

**R1-A contract:** the semantics are unchanged.
- `output_schema.py` is unmodified.
- `config.py` gained only 4 new RUNTIME parameters and a unit constant. The registry version is now `r1b-1`, with 90 parameters. The default config hash is now `35cd8fce48e15cff124d1409401d25b9fb6f5282db2da058c36f0eb15c1ee932`; R1-A's was `7fc5276e…`.

## Files

| Path | Status | Purpose |
|---|---|---|
| `player_tracker_rb/video_io.py` | NEW | `SequentialVideoReader`, frame fingerprint, fail-closed errors |
| `player_tracker_rb/provenance.py` | NEW | Contract provenance block (R1-A `PROVENANCE_FIELDS`), video provenance sidecar, git state, deterministic digest |
| `player_tracker_rb/config.py` | CHANGED (additive) | Added `seek_mode`, `hash_chunk_bytes`, `fingerprint_interpolation`, `timing_drift_tolerance_frames` and `MS_PER_SECOND` |
| `player_tracker_rb/__init__.py` | CHANGED | `__version__` set to `0.2.0-r1b`; stage note |
| `tests_rb/test_video_io.py` | NEW | Start frame, sequence, numbering against the frozen exemplars, time mapping at 4 frame rates, failure modes, single positioning, source change, no resume, fingerprint, and the GOLD alignment test |
| `tests_rb/test_provenance.py` | NEW | Contract block, sidecar, config hash, git, determinism, fail-closed |
| `tests_rb/_synth_video.py` | NEW | Synthetic videos whose frames encode their own index (an independent content check) |
| `tests_rb/test_package_hygiene.py` | CHANGED (additive) | The frame fingerprint may only be referenced by the video and provenance layer |
| `tools_rb/video_provenance_check.py` | NEW | Runs on a real clip three times and checks determinism and alignment |
| `R1/R1B/FRAME_FINGERPRINT.md` | NEW | Fingerprint method and limits |
| `R1/R1B/R1B_CONFIG_REGISTRY.csv` | NEW | Registry export (90 parameters) |
| `R1/R1B/R1B_TEST_OUTPUT_mac.txt` | NEW | Raw Mac test output |
| `R1/R1B/R1B_DIFF_code.patch` | NEW | Exact diff of the code commit `ff3c2d1` (sha256 `5d63cf0d…81d4a6`) |
| `R1/R1B/gold_video_check/` | NEW | GOLD provenance sidecar and check report |

## Behaviour

- **Positioning.** The reader reaches `start_frame` once. The default, `decode_from_zero`, decodes and discards frames 0..start−1 with no container seek. The `container_seek` option does exactly one verified `CAP_PROP_POS_FRAMES` seek. After that the reader only calls `read()`; a test wraps the capture object and counts its calls to confirm this.
- **Numbering.** The seed frame is the initialization frame, relative frame 0, and is not logged. Frame_log frame *f* is video frame `start_frame + f`, so the first logged step is the next frame. After every read, the decoder's reported position is checked against this.
- **Time.** `video_time_s = video_frame_index / fps` is the absolute time and becomes the v2 `timestamp_s`. `rel_time_s = frame / fps` is the time since the seed and matches the surviving `event_adapter` snapshot timestamp. The container timestamp is recorded and its drift is checked against a tolerance of 0.5 frame.
- **Fail closed.** The reader raises on:
  - a missing path, an empty path or a directory;
  - an unreadable or empty file, or no usable fps;
  - a start frame below 0, non-integer, or beyond the stream;
  - a sha256 mismatch with the expected hash;
  - a decoder position mismatch or a frame-size change;
  - the source being changed or deleted during the run (stat-checked before every frame; full re-hash at close);
  - an unknown git commit;
  - a seed box outside the frame, or an incomplete model record;
  - a sidecar requested without verification at close.
- **No resume.** No API name contains resume or checkpoint. `resume_allowed=True` is rejected by config, and passing `resume=` raises `TypeError`. A reader can be iterated only once, and it writes no files.

## Where provenance is recorded

- **R1-A contract block (`summary.v2.provenance`):** video path, video sha256, fps, frame count, decode mode `SEQUENTIAL_NO_RESUME`, seed frame, seed bbox, models (empty in R1-B), config hash, git commit, git dirty, schema version, tracker version and run start time.
- **Sidecar `{video}_rb_video_provenance.json`:** the items the contract block has no slot for. These are frame width and height, file size, decoder backend, OpenCV version, seek mode, the frame-index and time rules, the seed frame's fingerprint, per-run frame-fingerprint digests, processed and last frame, EOF, timing drift, container seek calls, end-of-run source re-verification and `resume_supported: false`.

## GOLD alignment and provenance check (video fixture only; no event fields read)

The run is `R1/R1B/gold_video_check/VIDEO_PROVENANCE_CHECK.json`, made at git `ff3c2d1`, clean, with config hash `35cd8fce…`. It read only `video_sha256`, `seed_frame`, `seed_bbox`, `fps`, `frame_count`, `width` and `height` from the fixture.

- **All 16 checks pass.**
- Video sha256 `164541108ca7…` matches the fixture. The clip is 1280×720 at 30 fps, with 360 frames.
- The seed is frame 180. Frames 1..179 map to video frames 181..359, and EOF was reached. **179 processed frames equals the frozen 3N GOLD Pass-1 output** (`start_frame` 180, `processed_frames` 179).
- Two `decode_from_zero` runs gave the same deterministic digest, `cbc9e9d4…8b26`, and the same per-frame fingerprints, `c97e7ace…238a`.
- `container_seek` decoded identical fingerprints for the seed and all 179 frames. No seek misregistration was found on this H.264 MP4.
- Maximum timing drift was 5e-14 frames; there were 0 warnings.

## Known limits

- **Constant frame rate is assumed.** Variable-frame-rate files are only detected (as drift warnings), not handled.
- **The fingerprint is decoder-build specific**; see `FRAME_FINGERPRINT.md`.
- **The frame dimensions and the other sidecar-only fields are not in the tracking JSON.** Adding them would need an additive R1-A schema change, which has not been made and needs approval.
- **`decode_from_zero` costs a decode of every frame before the seed.** That is negligible for extracts but linear in the seed position for long sources.
- **Expected sha256 is optional in the reader.** The run entry point (R1-G) should require it when a fixture exists (D18).
- **The sidecar `abs_path` is the Mac VM mount path.** It is excluded from the deterministic digest.
- **Tests leave unclosed-file `ResourceWarning`s.** These are cosmetic, from `json.load(open(...))` in the tests.

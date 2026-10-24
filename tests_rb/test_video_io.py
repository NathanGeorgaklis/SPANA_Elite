# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R1-B instruction (Nathan, 2026-10-24, required tests); R0/R0_TRACKER_INTERFACE_MAP.md §1.10-§1.11;
#                 fixtures/historical_contract (frozen 3N Pass-1 JSONs: frame_log numbering)
"""Tests for player_tracker_rb.video_io: start frame, sequential progression, numbering, time mapping, fps,
failure modes, source hashing, single positioning, source-change detection and absence of resume."""
import hashlib
import inspect
import json
import os
import shutil
import tempfile
import unittest

import cv2

from player_tracker_rb import output_schema as S
from player_tracker_rb import video_io as V
from player_tracker_rb.config import ConfigError, TrackerConfig
from tests_rb import _synth_video as SV

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIST = os.path.join(ROOT, "fixtures", "historical_contract")
GOLD_JSON = os.path.join(HIST, "GOLD_DUEL_01_syo_mar_3036_3048_mo_tracking_results.json")
PC_JSON = os.path.join(HIST, "f5bd8ce0-pass_completed_01_mo_tracking_results.json")


def cfg(**over):
    return TrackerConfig.with_overrides(over)


class _Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="rb_video_")
        cls.v30 = SV.write_video(os.path.join(cls.tmp, "v30.avi"), 60, 30.0)
        cls.v25 = SV.write_video(os.path.join(cls.tmp, "v25.avi"), 40, 25.0)
        cls.v2997 = SV.write_video(os.path.join(cls.tmp, "v2997.avi"), 40, 29.97)
        cls.v56 = SV.write_video(os.path.join(cls.tmp, "v56.avi"), 40, 56.502242152466366)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def read_all(self, path, start, c=None, **kw):
        with V.SequentialVideoReader(path, start, c or cfg(), **kw) as r:
            frames = list(r)
            seed = r.seed
            info = r.info
            summary = r.alignment_summary()
        return seed, frames, info, summary, r


class TestStartAndSequence(_Base):
    def test_seed_is_start_frame_both_seek_modes(self):
        for mode in ("decode_from_zero", "container_seek"):
            for start in (0, 1, 7, 30, 59):
                with self.subTest(mode=mode, start=start):
                    with V.SequentialVideoReader(self.v30, start, cfg(seek_mode=mode)) as r:
                        self.assertEqual(r.seed.rel_frame, 0)
                        self.assertEqual(r.seed.video_frame_index, start)
                        self.assertEqual(SV.read_index(r.seed.image), start)

    def test_first_logged_step_is_next_frame_and_sequence_is_contiguous(self):
        seed, frames, info, summ, _ = self.read_all(self.v30, 10)
        self.assertEqual([f.rel_frame for f in frames], list(range(1, 60 - 10)))
        self.assertEqual([f.video_frame_index for f in frames], list(range(11, 60)))
        self.assertEqual([SV.read_index(f.image) for f in frames], list(range(11, 60)))   # decoded content
        self.assertEqual(summ["first_logged_frame"], 1)
        self.assertEqual(summ["processed_frames"], 49)
        self.assertEqual(summ["last_video_frame_index"], 59)
        self.assertTrue(summ["eof_reached"])

    def test_no_duplicate_or_skipped_frames(self):
        _, frames, _, _, _ = self.read_all(self.v30, 0)
        rel = [f.rel_frame for f in frames]
        self.assertEqual(len(rel), len(set(rel)))
        self.assertEqual(rel, list(range(1, len(rel) + 1)))
        content = [SV.read_index(f.image) for f in frames]
        self.assertEqual(content, [f.video_frame_index for f in frames])

    def test_seek_modes_decode_identical_frames(self):
        a = self.read_all(self.v30, 23, cfg(seek_mode="decode_from_zero"))
        b = self.read_all(self.v30, 23, cfg(seek_mode="container_seek"))
        self.assertEqual(a[0].thumb_sha1, b[0].thumb_sha1)
        self.assertEqual([f.thumb_sha1 for f in a[1]], [f.thumb_sha1 for f in b[1]])

    def test_max_frames_cap(self):
        _, frames, _, summ, _ = self.read_all(self.v30, 5, cfg(max_frames=7))
        self.assertEqual([f.rel_frame for f in frames], list(range(1, 8)))
        self.assertFalse(summ["eof_reached"])

    def test_iterate_only_once(self):
        with V.SequentialVideoReader(self.v30, 0, cfg(max_frames=3)) as r:
            list(r)
            with self.assertRaises(V.VideoError):
                iter(r)


class _CountingCapture:
    """Wraps cv2.VideoCapture and records every set() call (positioning)."""
    instances = []

    def __init__(self, path):
        self._c = cv2.VideoCapture(path)
        self.set_calls = []
        _CountingCapture.instances.append(self)

    def set(self, prop, value):
        self.set_calls.append((prop, value))
        return self._c.set(prop, value)

    def __getattr__(self, name):
        return getattr(self._c, name)


class TestSinglePositioning(_Base):
    def test_no_seek_during_run(self):
        for mode, expected_sets in (("decode_from_zero", 0), ("container_seek", 1)):
            with self.subTest(mode=mode):
                _CountingCapture.instances.clear()
                _, frames, _, summ, r = self.read_all(self.v30, 12, cfg(seek_mode=mode),
                                                      capture_factory=_CountingCapture)
                cap = _CountingCapture.instances[-1]
                self.assertEqual(len(cap.set_calls), expected_sets)
                self.assertEqual(summ["container_seek_calls"], expected_sets)
                self.assertEqual(len(frames), 60 - 13)


class TestTimeMapping(_Base):
    def test_time_mapping_30_and_non_30_fps(self):
        for path, fps in ((self.v30, 30.0), (self.v25, 25.0), (self.v2997, 29.97), (self.v56, 56.502)):
            with self.subTest(fps=fps):
                seed, frames, info, summ, _ = self.read_all(path, 4)
                self.assertAlmostEqual(info.fps, fps, places=3)
                for f in [seed] + frames:
                    self.assertEqual(f.video_frame_index, 4 + f.rel_frame)
                    self.assertEqual(f.video_time_s, f.video_frame_index / info.fps)
                    self.assertEqual(f.rel_time_s, f.rel_frame / info.fps)
                    if f.pos_msec_reported is not None:
                        self.assertAlmostEqual(f.pos_msec_reported / 1000.0, f.video_time_s, delta=0.5 / info.fps)
                self.assertEqual(summ["timing_drift_warnings"], 0)

    def test_legacy_snapshot_timestamp_is_relative(self):
        # event_adapter writes timestamp = round(frame_number / fps, 4) with frame_number = frame_log frame.
        seed, frames, info, _, _ = self.read_all(self.v25, 10)
        dp = cfg().get("timestamp_round_dp")
        for f in frames:
            self.assertEqual(round(f.rel_time_s, dp), round(f.rel_frame / info.fps, dp))


class TestFailureModes(_Base):
    def test_invalid_paths(self):
        for p in ("", os.path.join(self.tmp, "missing.mp4"), self.tmp):
            with self.subTest(path=p):
                with self.assertRaises(V.VideoNotFoundError):
                    V.SequentialVideoReader(p, 0, cfg())

    def test_unreadable_files(self):
        junk = os.path.join(self.tmp, "junk.mp4")
        with open(junk, "wb") as f:
            f.write(os.urandom(4096))
        empty = os.path.join(self.tmp, "empty.mp4")
        open(empty, "wb").close()
        for p in (junk, empty):
            with self.subTest(path=os.path.basename(p)):
                with self.assertRaises(V.VideoUnreadableError):
                    V.SequentialVideoReader(p, 0, cfg())

    def test_start_frame_out_of_range(self):
        for mode in ("decode_from_zero", "container_seek"):
            for start in (60, 61, 10_000):
                with self.subTest(mode=mode, start=start):
                    with self.assertRaises(V.StartFrameOutOfRangeError):
                        V.SequentialVideoReader(self.v30, start, cfg(seek_mode=mode))
        for bad in (-1, 1.5, "3", True, None):
            with self.subTest(bad=bad):
                with self.assertRaises(V.StartFrameOutOfRangeError):
                    V.SequentialVideoReader(self.v30, bad, cfg())

    def test_start_frame_beyond_stream_when_count_unknown(self):
        class NoCount(_CountingCapture):
            def get(self, prop):
                if prop == cv2.CAP_PROP_FRAME_COUNT:
                    return -1.0
                return self._c.get(prop)
        with self.assertRaises(V.StartFrameOutOfRangeError):
            V.SequentialVideoReader(self.v30, 75, cfg(), capture_factory=NoCount)

    def test_hash_mismatch_refused(self):
        with self.assertRaises(V.SourceHashMismatchError):
            V.SequentialVideoReader(self.v30, 0, cfg(), expected_sha256="0" * 64)


class TestSourceHashing(_Base):
    def test_sha256_recorded(self):
        want = hashlib.sha256(open(self.v30, "rb").read()).hexdigest()
        with V.SequentialVideoReader(self.v30, 0, cfg(), expected_sha256=want) as r:
            self.assertEqual(r.info.sha256, want)
            self.assertEqual(r.info.size_bytes, os.path.getsize(self.v30))
        self.assertTrue(r.source_verified_at_close)

    def test_chunk_size_does_not_change_hash(self):
        a = V.sha256_file(self.v30, 4096)
        b = V.sha256_file(self.v30, 1048576)
        self.assertEqual(a, b)


class TestSourceChangeFailsClosed(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="rb_change_")
        self.v = SV.write_video(os.path.join(self.tmp, "v.avi"), 30, 30.0)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_change_during_iteration(self):
        r = V.SequentialVideoReader(self.v, 0, cfg())
        it = iter(r)
        next(it)
        with open(self.v, "ab") as f:
            f.write(b"x")
        with self.assertRaises(V.SourceChangedError):
            next(it)
        r.close(verify=False)

    def test_same_size_content_change_detected_at_close(self):
        r = V.SequentialVideoReader(self.v, 0, cfg(max_frames=2))
        list(r)
        st = os.stat(self.v)
        data = bytearray(open(self.v, "rb").read())
        data[-1] ^= 0xFF
        with open(self.v, "wb") as f:
            f.write(data)
        os.utime(self.v, ns=(st.st_atime_ns, st.st_mtime_ns))     # same size and mtime: only the hash can tell
        with self.assertRaises(V.SourceChangedError):
            r.close()
        self.assertFalse(r.source_verified_at_close)

    def test_deleted_source_detected(self):
        r = V.SequentialVideoReader(self.v, 0, cfg())
        it = iter(r)
        next(it)
        os.remove(self.v)
        with self.assertRaises(V.SourceChangedError):
            next(it)
        r.close(verify=False)


class TestNoResume(_Base):
    def test_no_resume_or_checkpoint_api(self):
        sig = inspect.signature(V.SequentialVideoReader.__init__)
        for name in sig.parameters:
            self.assertNotIn("resume", name.lower())
            self.assertNotIn("checkpoint", name.lower())
        for mod in (V,):
            for name, _ in inspect.getmembers(mod):
                self.assertNotIn("resume", name.lower())
                self.assertNotIn("checkpoint", name.lower())
        for name, _ in inspect.getmembers(V.SequentialVideoReader):
            self.assertNotIn("resume", name.lower())
            self.assertNotIn("checkpoint", name.lower())

    def test_resume_cannot_be_enabled(self):
        with self.assertRaises(ConfigError):
            TrackerConfig.with_overrides({"resume_allowed": True})
        with self.assertRaises(TypeError):
            V.SequentialVideoReader(self.v30, 0, cfg(), resume=True)

    def test_reader_writes_nothing(self):
        before = sorted(os.listdir(self.tmp))
        self.read_all(self.v30, 3)
        self.assertEqual(sorted(os.listdir(self.tmp)), before)


class TestFrameLogNumberingContract(_Base):
    """Relative numbering against the recovered Pass-1 schema (frozen 3N exemplars)."""

    def test_numbering_matches_frozen_exemplars(self):
        for path in (GOLD_JSON, PC_JSON):
            with self.subTest(os.path.basename(path)):
                d = json.load(open(path))
                start, n = d["summary"]["start_frame"], d["summary"]["processed_frames"]
                hist_frames = [fr["frame"] for fr in d["frame_log"]]
                self.assertEqual(hist_frames, list(range(1, n + 1)))
                # A synthetic video long enough to contain the same span, read by R1-B, must produce the same
                # frame numbers: seed = start_frame (not logged), then 1..n.
                vid = SV.write_video(os.path.join(self.tmp, f"span_{start}_{n}.avi"), start + n + 1, 30.0)
                with V.SequentialVideoReader(vid, start, cfg(max_frames=n)) as r:
                    frames = list(r)
                    self.assertEqual(r.seed.video_frame_index, start)
                self.assertEqual([f.rel_frame for f in frames], hist_frames)
                self.assertEqual([SV.read_index(f.image) for f in frames], [start + f for f in hist_frames])

    def test_frames_build_valid_frame_records(self):
        c = cfg()
        _, frames, info, _, _ = self.read_all(self.v30, 20, c)
        for f in frames:
            rec = S.build_frame_record(c, frame=f.rel_frame, is_cut=False, camera_shift_px=[0.0, 0.0],
                                       n_detections=0, overlap_contests=[], tracks=[],
                                       v2={"video_frame_index": f.video_frame_index, "timestamp_s": f.video_time_s,
                                           "frame_thumb_sha1": f.thumb_sha1, "cut_detected": False,
                                           "camera_shift_h": None, "target_height_px": None})
            self.assertEqual(S.validate_frame_record(rec, require_v2=False), [])
            self.assertEqual(rec["v2"]["video_frame_index"], 20 + rec["frame"])


class TestFingerprint(_Base):
    def test_fingerprint_reproducible_and_distinct(self):
        a = self.read_all(self.v30, 0)[1]
        b = self.read_all(self.v30, 0)[1]
        self.assertEqual([f.thumb_sha1 for f in a], [f.thumb_sha1 for f in b])
        self.assertEqual(len({f.thumb_sha1 for f in a}), len(a))   # synthetic frames all differ

    def test_fingerprint_depends_on_documented_settings(self):
        img = SV.make_frame(5)
        x = V.frame_fingerprint(img, cfg())
        y = V.frame_fingerprint(img, cfg(frame_thumbnail_size=(32, 18)))
        self.assertNotEqual(x, y)
        self.assertEqual(x, V.frame_fingerprint(img.copy(), cfg()))
        m = V.fingerprint_method(cfg())
        self.assertIn("alignment", m["purpose"])
        self.assertIn("not an identity", m["purpose"])


GOLD_FIXTURE = os.path.join(ROOT, "fixtures", "GOLD_DUEL_01", "GOLD_DUEL_01_BENCHMARK_FIXTURE.json")


@unittest.skipUnless(os.environ.get("SPANA_GOLD_VIDEO") and os.path.isfile(os.environ.get("SPANA_GOLD_VIDEO", "")),
                     "set SPANA_GOLD_VIDEO to the GOLD extract to run the real-video alignment test")
class TestGoldVideoAlignment(unittest.TestCase):
    """GOLD used only as a video/frame-alignment fixture: sha256, fps, size, seed frame. No event fields read."""

    def test_gold_seed_and_first_frames(self):
        fx = json.load(open(GOLD_FIXTURE))
        path = os.environ["SPANA_GOLD_VIDEO"]
        out = {}
        for mode in ("decode_from_zero", "container_seek"):
            with V.SequentialVideoReader(path, fx["seed_frame"], cfg(seek_mode=mode, max_frames=30),
                                         expected_sha256=fx["video_sha256"]) as r:
                frames = list(r)
                out[mode] = (r.seed.thumb_sha1, [f.thumb_sha1 for f in frames], r.info)
        info = out["decode_from_zero"][2]
        self.assertEqual(info.fps, fx["fps"])
        self.assertEqual((info.frame_width, info.frame_height), (fx["width"], fx["height"]))
        self.assertEqual(info.frame_count_reported, fx["frame_count"])
        self.assertEqual(out["decode_from_zero"][0], out["container_seek"][0])
        self.assertEqual(out["decode_from_zero"][1], out["container_seek"][1])


if __name__ == "__main__":
    unittest.main()

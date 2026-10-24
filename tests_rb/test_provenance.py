# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R1-B instruction (provenance fields, fail closed, determinism); player_tracker_rb/output_schema.py
#                 PROVENANCE_FIELDS (R1-A contract)
"""Tests for player_tracker_rb.provenance: contract block, sidecar, config hash, git, determinism, fail-closed."""
import json
import os
import shutil
import tempfile
import unittest

from player_tracker_rb import __version__
from player_tracker_rb import output_schema as S
from player_tracker_rb import provenance as P
from player_tracker_rb import video_io as V
from player_tracker_rb.config import TrackerConfig
from tests_rb import _synth_docs as SD
from tests_rb import _synth_video as SV

FIXED_GIT = lambda: ("a" * 40, False)                      # noqa: E731
FIXED_CLOCK = lambda: "2026-10-24T00:00:00Z"               # noqa: E731
SEED_BOX = [10.0, 5.0, 60.0, 90.0]


def run_once(path, start, cfg, git=FIXED_GIT, clock=FIXED_CLOCK, seed_box=SEED_BOX):
    with V.SequentialVideoReader(path, start, cfg) as r:
        thumbs = [f.thumb_sha1 for f in r]
        contract = P.contract_provenance(r, seed_box, cfg, git=git, clock=clock)
    side = P.video_provenance_sidecar(r, contract, cfg, frame_thumbs=thumbs)
    return contract, side


class TestProvenance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="rb_prov_")
        cls.v = SV.write_video(os.path.join(cls.tmp, "clip.avi"), 50, 25.0)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_contract_block_fields(self):
        cfg = TrackerConfig.default()
        contract, side = run_once(self.v, 8, cfg)
        self.assertEqual(set(contract), set(S.PROVENANCE_FIELDS))
        self.assertEqual(contract["video_sha256"], V.sha256_file(self.v, 4096))
        self.assertEqual(contract["seed_frame"], 8)
        self.assertEqual(contract["fps"], 25.0)
        self.assertEqual(contract["frame_count"], 50)
        self.assertEqual(contract["decode_mode"], S.DECODE_MODE)
        self.assertEqual(contract["config_hash"], cfg.config_hash())
        self.assertEqual(contract["git_commit"], "a" * 40)
        self.assertEqual(contract["schema_version"], S.V2_SCHEMA_VERSION)
        self.assertEqual(contract["tracker_version"], __version__)
        self.assertEqual(contract["models"], [])

    def test_contract_block_fits_r1a_summary(self):
        cfg = TrackerConfig.default()
        contract, _ = run_once(self.v, 8, cfg)
        doc = SD.build_doc(cfg)
        doc["summary"]["v2"]["provenance"] = dict(contract, fps=doc["summary"]["fps"],
                                                  seed_frame=doc["summary"]["start_frame"])
        doc["summary"]["v2"]["config_hash"] = contract["config_hash"]
        self.assertEqual(S.validate_summary(doc["summary"], require_v2=True), [])

    def test_sidecar_fields(self):
        cfg = TrackerConfig.default()
        contract, side = run_once(self.v, 8, cfg)
        vid = side["video"]
        for k in ("path", "sha256", "size_bytes", "fps", "frame_width", "frame_height", "frame_count_reported",
                  "backend", "cv2_version", "seek_mode", "start_frame"):
            self.assertIn(k, vid)
        self.assertEqual((vid["frame_width"], vid["frame_height"]), (SV.W, SV.H))
        al = side["frame_alignment"]
        self.assertEqual(al["seed_video_frame_index"], 8)
        self.assertEqual(al["first_logged_frame"], 1)
        self.assertEqual(al["processed_frames"], 50 - 9)
        self.assertEqual(al["last_video_frame_index"], 49)
        self.assertIn("video_frame_index = start_frame + frame", al["frame_index_rule"])
        self.assertIn("video_time_s = video_frame_index / fps", al["time_rule"])
        self.assertTrue(side["source_reverified_at_close"])
        self.assertFalse(side["resume_supported"])
        self.assertEqual(side["config_hash"], cfg.config_hash())

    def test_config_hash_recorded_and_sensitive(self):
        a, _ = run_once(self.v, 8, TrackerConfig.default())
        b, _ = run_once(self.v, 8, TrackerConfig.with_overrides({"seed_min_iou": 0.6}))
        self.assertNotEqual(a["config_hash"], b["config_hash"])
        self.assertEqual(b["config_hash"], TrackerConfig.with_overrides({"seed_min_iou": 0.6}).config_hash())

    def test_deterministic_across_runs(self):
        cfg = TrackerConfig.default()
        c1, s1 = run_once(self.v, 8, cfg, clock=lambda: "2026-10-24T01:00:00Z")
        c2, s2 = run_once(self.v, 8, cfg, clock=lambda: "2026-10-24T02:00:00Z")
        self.assertNotEqual(c1["run_started_utc"], c2["run_started_utc"])
        self.assertEqual(P.deterministic_digest(s1), P.deterministic_digest(s2))
        self.assertEqual(s1["frame_alignment"]["frame_thumbs_sha256"], s2["frame_alignment"]["frame_thumbs_sha256"])
        self.assertEqual(S.dumps_canonical(P._strip_nondeterministic(s1)),
                         S.dumps_canonical(P._strip_nondeterministic(s2)))

    def test_digest_detects_real_differences(self):
        cfg = TrackerConfig.default()
        _, s1 = run_once(self.v, 8, cfg)
        _, s2 = run_once(self.v, 9, cfg)
        self.assertNotEqual(P.deterministic_digest(s1), P.deterministic_digest(s2))

    def test_git_failure_fails_closed(self):
        def broken():
            raise P.ProvenanceError("no git")
        with self.assertRaises(P.ProvenanceError):
            run_once(self.v, 8, TrackerConfig.default(), git=broken)

    def test_real_git_state_of_this_tree(self):
        head, dirty = P.git_state()
        self.assertEqual(len(head), 40)
        self.assertIsInstance(dirty, bool)

    def test_git_state_outside_repo_fails_closed(self):
        with self.assertRaises(P.ProvenanceError):
            P.git_state(self.tmp)

    def test_dirty_tree_recorded(self):
        contract, _ = run_once(self.v, 8, TrackerConfig.default(), git=lambda: ("b" * 40, True))
        self.assertTrue(contract["git_dirty"])

    def test_bad_seed_bbox_fails_closed(self):
        for box in ([0, 0, 0, 10], [50, 5, 10, 90], [-1, 0, 10, 10], [0, 0, SV.W + 1, 10], [1, 2, 3], ["a", 0, 1, 1]):
            with self.subTest(box=box):
                with self.assertRaises(P.ProvenanceError):
                    run_once(self.v, 8, TrackerConfig.default(), seed_box=box)

    def test_model_records_must_be_complete(self):
        cfg = TrackerConfig.default()
        with V.SequentialVideoReader(self.v, 0, cfg) as r:
            with self.assertRaises(P.ProvenanceError):
                P.contract_provenance(r, SEED_BOX, cfg, models=[{"role": "detector", "filename": "x.onnx"}],
                                      git=FIXED_GIT, clock=FIXED_CLOCK)

    def test_sidecar_refused_without_close_verification(self):
        cfg = TrackerConfig.default()
        r = V.SequentialVideoReader(self.v, 0, cfg)
        contract = P.contract_provenance(r, SEED_BOX, cfg, git=FIXED_GIT, clock=FIXED_CLOCK)
        with self.assertRaises(P.ProvenanceError):
            P.video_provenance_sidecar(r, contract, cfg)
        r.close(verify=False)
        with self.assertRaises(P.ProvenanceError):
            P.video_provenance_sidecar(r, contract, cfg)

    def test_write_sidecar_canonical_and_no_overwrite(self):
        cfg = TrackerConfig.default()
        _, side = run_once(self.v, 8, cfg)
        out = tempfile.mkdtemp(dir=self.tmp)
        path = P.sidecar_path(out, self.v)
        self.assertTrue(path.endswith("clip" + P.SIDECAR_SUFFIX))
        digest = P.write_sidecar(side, path)
        self.assertEqual(json.load(open(path)), json.loads(S.dumps_canonical(side)))
        self.assertEqual(len(digest), 64)
        with self.assertRaises(P.ProvenanceError):
            P.write_sidecar(side, path)


if __name__ == "__main__":
    unittest.main()

# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R1-G instruction (Nathan, 2026-10-25: deterministic reruns, provenance/schema validation, fail
#                 closed); R1/R1_DESIGN_DECISIONS.md D18, D20; R0/R0_R1_TRACKER_REBUILD_PLAN.md item 10 (run signature)
"""R1-G pipeline mechanics: determinism, frame/time mapping, provenance and sidecar, streaming assembly, and every
fail-closed refusal of run_rb.run (no output file may survive a refused or failed run)."""
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import unittest

import cv2

from player_tracker_rb import output_schema as S
from player_tracker_rb import provenance as PV
from player_tracker_rb import run_rb as R
from player_tracker_rb.config import TrackerConfig
from player_tracker_rb.detector import ScriptedDetector
from player_tracker_rb.video_io import SourceHashMismatchError, frame_fingerprint, sha256_file
from tests_rb._synth_pipeline import SCRIPTED, box, clip, detector_script, run_clip, walk, write_clip
from tests_rb._synth_scene import SignatureEmbedding, make_people

KIT_A = (200, 60, 40)
KIT_B = (40, 40, 210)


def _people():
    return make_people([("TARGET", KIT_A, 1.0), ("PERSON_A", KIT_A, 0.75), ("PERSON_B", KIT_B, 0.1)])


def _frames(n=70):
    return clip({"TARGET": walk(n, (150, 200), (3, 0)), "PERSON_A": walk(n, (150 + 3 * n, 206), (-3, 0)),
                 "PERSON_B": [None] * 10 + walk(n - 10, (500, 120), (-2, 1))},
                drop=lambda k: ["TARGET"] if 40 <= k < 44 else [])


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class PipelineBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="r1g_run_")
        cls.people = _people()
        cls.frames = _frames()
        cls.start = 3
        cls.video = write_clip(os.path.join(cls.tmp, "clip.avi"), cls.frames, cls.people)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def run_once(self, tag, **kw):
        return run_clip(self.frames, self.people, os.path.join(self.tmp, tag), start_frame=self.start, name=tag,
                        video_path=self.video, **kw)

    def scripted(self, ov=None):
        cfg = TrackerConfig.with_overrides(dict(SCRIPTED, **(ov or {})))
        script, _ = detector_script(self.frames, self.start, 0.35)
        return ScriptedDetector(script, cfg), SignatureEmbedding(self.people)

    def out_dir(self, tag):
        d = os.path.join(self.tmp, tag)
        os.makedirs(d)
        return d


class Determinism(PipelineBase):
    def test_two_runs_are_identical_except_wall_clock(self):
        a, b = self.run_once("det_a"), self.run_once("det_b")
        cmp = R.compare_documents(a.doc, b.doc)
        self.assertTrue(cmp["identical_except_wall_clock"], cmp["first_difference"])
        self.assertEqual(_sha(a.paths["decision_log"]), _sha(b.paths["decision_log"]))
        # byte level: with the wall-clock fields set equal, the files are byte-identical
        texts = []
        for e in (a, b):
            with open(e.paths["results"], encoding="ascii") as fh:
                d = json.load(fh)
            d["summary"]["processing_time_s"] = 0.0
            d["summary"]["v2"]["provenance"]["run_started_utc"] = "X"
            texts.append(S.dumps_canonical(d))
        self.assertEqual(texts[0], texts[1])
        # sidecars agree on everything that is not wall-clock or the (wall-clock dependent) results hash
        sides = []
        for e in (a, b):
            with open(e.paths["provenance"], encoding="ascii") as fh:
                s = json.load(fh)
            s.pop("results_sha256")
            sides.append(PV.deterministic_digest(s))
        self.assertEqual(sides[0], sides[1])

    def test_streamed_file_is_the_canonical_serialisation(self):
        e = self.run_once("canon")
        with open(e.paths["results"], encoding="ascii") as fh:
            text = fh.read()
        self.assertEqual(text, S.dumps_canonical(json.loads(text)))
        self.assertFalse([p for p in os.listdir(os.path.dirname(e.paths["results"])) if R.PARTIAL_SUFFIX in p])


class FrameMappingAndProvenance(PipelineBase):
    def test_frame_index_time_and_fingerprint(self):
        e = self.run_once("map")
        cfg = TrackerConfig.default()
        s = e.doc["summary"]
        self.assertEqual(s["start_frame"], self.start)
        self.assertEqual(s["processed_frames"], len(self.frames) - self.start - 1)
        cap = cv2.VideoCapture(self.video)
        imgs = []
        while True:
            ok, img = cap.read()
            if not ok:
                break
            imgs.append(img)
        for fr in e.doc["frame_log"]:
            vfi = fr["v2"]["video_frame_index"]
            self.assertEqual(vfi, self.start + fr["frame"])
            self.assertAlmostEqual(fr["v2"]["timestamp_s"], round(vfi / s["fps"], 4))
            self.assertEqual(fr["v2"]["frame_thumb_sha1"], frame_fingerprint(imgs[vfi], cfg))

    def test_provenance_and_sidecar(self):
        e = self.run_once("prov")
        v2 = e.doc["summary"]["v2"]
        p = v2["provenance"]
        self.assertEqual(p["video_sha256"], sha256_file(self.video, 1 << 20))
        self.assertEqual(p["seed_frame"], self.start)
        self.assertEqual(p["seed_bbox"], [round(v, 1) for v in self.frames[self.start].scene.boxes["TARGET"]])
        self.assertEqual(p["config_hash"], v2["config_hash"])
        self.assertEqual(p["config_hash"], TrackerConfig.with_overrides(SCRIPTED).config_hash())
        head = subprocess.run(["git", "-C", PV.PACKAGE_ROOT, "rev-parse", "HEAD"], capture_output=True,
                              text=True).stdout.strip()
        self.assertEqual(p["git_commit"], head)
        self.assertIsInstance(p["git_dirty"], bool)
        self.assertEqual(p["models"], [])                      # scripted backends use no model file
        self.assertEqual(p["decode_mode"], S.DECODE_MODE)
        self.assertEqual(v2["schema_version"], S.V2_SCHEMA_VERSION)
        with open(e.paths["provenance"], encoding="ascii") as fh:
            side = json.load(fh)
        self.assertEqual(side["results_sha256"], _sha(e.paths["results"]))
        self.assertTrue(side["source_reverified_at_close"])
        self.assertFalse(side["resume_supported"])
        self.assertEqual(side["backends"]["detector"]["backend"], "scripted")
        self.assertEqual(side["backends"]["embedding"]["backend"], "test_signature")
        self.assertEqual(side["frame_alignment"]["processed_frames"], e.doc["summary"]["processed_frames"])
        self.assertTrue(any("contrast_available" in n for n in v2["notes"]))

    def test_max_frames_caps_the_run(self):
        e = self.run_once("cap", overrides={"max_frames": 20})
        self.assertEqual(e.doc["summary"]["processed_frames"], 20)
        self.assertEqual(S.validate_document(e.doc, require_v2=True), [])

    def test_no_decision_log_without_debug(self):
        e = self.run_once("nodebug", debug=False)
        self.assertFalse(os.path.exists(e.paths["decision_log"]))

    def test_final_tracks(self):
        e = self.run_once("finals")
        ft = e.doc["final_tracks"]
        self.assertEqual(ft[0]["track_id"], "T001")
        self.assertTrue(ft[0]["selected_player"])
        last_ids = {t["track_id"] for t in e.doc["frame_log"][-1]["tracks"]}
        written = {t["track_id"] for fr in e.doc["frame_log"] for t in fr["tracks"]}
        self.assertEqual({t["track_id"] for t in ft}, written)
        for t in ft:
            self.assertNotIn("v2", t)
            self.assertEqual(t["state"] == "TERMINATED", t["track_id"] not in last_ids)
        s = e.doc["summary"]
        self.assertEqual(s["n_tracks_ever_created"], len(written))
        self.assertEqual(s["n_tracks_final_active"], len(last_ids))
        self.assertEqual(s["final_confirmed_count"] + s["final_provisional_count"], len(last_ids))


class FailClosed(PipelineBase):
    def assertNothingWritten(self, d):
        self.assertEqual(os.listdir(d), [], "a refused or failed run must leave no output")

    def test_seed_without_detection_is_refused(self):
        d = self.out_dir("refuse_seed")
        det, emb = self.scripted()
        with self.assertRaises(R.RunRefusedError):
            R.run(self.video, self.start, [10.0, 10.0, 40.0, 80.0], out_dir=d, config_overrides=SCRIPTED,
                  detector=det, embedder=emb)
        self.assertNothingWritten(d)

    def test_jersey_number_and_missing_out_dir_are_refused(self):
        det, emb = self.scripted()
        seed = list(self.frames[self.start].scene.boxes["TARGET"])
        d = self.out_dir("refuse_jersey")
        with self.assertRaises(R.RunRefusedError):
            R.run(self.video, self.start, seed, jersey_number="23", out_dir=d, config_overrides=SCRIPTED,
                  detector=det, embedder=emb)
        with self.assertRaises(R.RunRefusedError):
            R.run(self.video, self.start, seed, out_dir=None, config_overrides=SCRIPTED, detector=det, embedder=emb)
        self.assertNothingWritten(d)

    def test_source_hash_mismatch_is_refused(self):
        d = self.out_dir("refuse_hash")
        det, emb = self.scripted()
        with self.assertRaises(SourceHashMismatchError):
            R.run(self.video, self.start, list(self.frames[self.start].scene.boxes["TARGET"]), out_dir=d,
                  config_overrides=SCRIPTED, detector=det, embedder=emb, expected_sha256="0" * 64)
        self.assertNothingWritten(d)

    def test_existing_output_is_never_overwritten(self):
        e = self.run_once("exists")
        before = _sha(e.paths["results"])
        det, emb = self.scripted()
        with self.assertRaises(R.RunRefusedError):
            R.run(self.video, self.start, list(self.frames[self.start].scene.boxes["TARGET"]),
                  out_dir=os.path.dirname(e.paths["results"]), config_overrides=SCRIPTED, detector=det, embedder=emb)
        self.assertEqual(_sha(e.paths["results"]), before)

    def test_backend_substitution_is_refused(self):
        d = self.out_dir("refuse_backend")
        det, emb = self.scripted()
        seed = list(self.frames[self.start].scene.boxes["TARGET"])
        with self.assertRaises(R.RunRefusedError):     # production config, scripted detector passed in
            R.run(self.video, self.start, seed, out_dir=d, detector=det, embedder=emb)
        with self.assertRaises(R.RunRefusedError):     # scripted config but no detector
            R.run(self.video, self.start, seed, out_dir=d, config_overrides=SCRIPTED, embedder=emb)
        with self.assertRaises(R.RunRefusedError):     # scripted config but no embedder
            R.run(self.video, self.start, seed, out_dir=d, config_overrides=SCRIPTED, detector=det)
        self.assertNothingWritten(d)

    def test_contract_violation_stops_the_run_and_leaves_nothing(self):
        d = self.out_dir("violation")
        det, emb = self.scripted()
        orig = R._RecordBuilder.target_record

        def broken(self_, f, tf, excluded):
            rec = orig(self_, f, tf, excluded)
            if f == 10:
                rec["state"] = "ACTIVE" if rec["state"] != "ACTIVE" else "OCCLUDED"   # state/v2 disagree
            return rec
        R._RecordBuilder.target_record = broken
        try:
            with self.assertRaises(R.ContractViolationError):
                R.run(self.video, self.start, list(self.frames[self.start].scene.boxes["TARGET"]), out_dir=d,
                      debug=True, config_overrides=SCRIPTED, detector=det, embedder=emb)
        finally:
            R._RecordBuilder.target_record = orig
        self.assertNothingWritten(d)


class FileSafety(PipelineBase):
    """R1-G review M3 / L2 / L3."""

    def test_a_file_appearing_mid_run_is_never_deleted_or_replaced(self):
        for which in ("results", "decision_log", "provenance"):
            with self.subTest(which=which):
                d = self.out_dir("race_" + which)
                planted = R.output_paths(d, self.video)[which]

                def hook(f, sizes, planted=planted):
                    if f == 5 and not os.path.exists(planted):
                        with open(planted, "w") as fh:
                            fh.write("SOMEONE ELSE'S FILE\n")
                det, emb = self.scripted()
                with self.assertRaises(R.RunRefusedError):
                    R.run(self.video, self.start, list(self.frames[self.start].scene.boxes["TARGET"]), out_dir=d,
                          debug=True, config_overrides=SCRIPTED, detector=det, embedder=emb, on_frame=hook)
                with open(planted) as fh:
                    self.assertEqual(fh.read(), "SOMEONE ELSE'S FILE\n")
                self.assertEqual(os.listdir(d), [os.path.basename(planted)], "only the foreign file may remain")

    def test_outputs_appear_under_final_names_only_when_complete(self):
        d = self.out_dir("publish")
        paths = R.output_paths(d, self.video)
        seen = []

        def hook(f, sizes):
            seen.append(sorted(os.listdir(d)))
        det, emb = self.scripted()
        R.run(self.video, self.start, list(self.frames[self.start].scene.boxes["TARGET"]), out_dir=d, debug=True,
              config_overrides=SCRIPTED, detector=det, embedder=emb, on_frame=hook)
        finals = {os.path.basename(p) for p in paths.values()}
        for listing in seen:
            self.assertFalse(set(listing) & finals, f"a final name existed mid-run: {listing}")
            self.assertTrue(all(name.endswith(R.PARTIAL_SUFFIX) for name in listing))
        self.assertEqual(set(os.listdir(d)), finals)

    def test_backend_built_with_other_settings_is_refused(self):
        d = self.out_dir("refuse_settings")
        other = TrackerConfig.with_overrides(dict(SCRIPTED, detector_min_score=0.9))
        script, _ = detector_script(self.frames, self.start, 0.35)
        det = ScriptedDetector(script, other)
        with self.assertRaises(R.RunRefusedError):
            R.run(self.video, self.start, list(self.frames[self.start].scene.boxes["TARGET"]), out_dir=d,
                  config_overrides=SCRIPTED, detector=det, embedder=SignatureEmbedding(self.people))
        self.assertNothingWritten(d)

    def test_production_named_backend_without_a_model_record_is_refused(self):
        class FakeYolox(ScriptedDetector):
            def provenance(self):
                return dict(super().provenance(), backend="yolox_onnx")
        cfg = TrackerConfig.with_overrides({"appearance_backend": "scripted"})
        script, _ = detector_script(self.frames, self.start, 0.35)
        with self.assertRaises(R.RunRefusedError):
            R.build_backends(cfg, FakeYolox(script, cfg), SignatureEmbedding(self.people))

    def assertNothingWritten(self, d):
        self.assertEqual(os.listdir(d), [])


if __name__ == "__main__":
    unittest.main()

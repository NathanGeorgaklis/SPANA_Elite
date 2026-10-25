# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R1-G independent code review + test audit (2026-10-25) of commit c6d118b; R1-G instruction
"""R1-G audit tests: regressions for the defects the independent review and test audit found, and the tests the
audit showed were missing (mutants that survived the first suite). Neutral TARGET / PERSON_A / TWIN names only."""
import copy
import json
import os
import shutil
import tempfile
import unittest

from player_tracker_rb import output_schema as S
from player_tracker_rb import run_rb as R
from player_tracker_rb import target as T
from player_tracker_rb.config import TrackerConfig
from player_tracker_rb.detector import ScriptedDetector
from tests_rb._synth_pipeline import (PW, SCRIPTED, box, clip, detector_script, iou, run_clip, walk, write_clip)
from tests_rb._synth_scene import SceneFrame, SignatureEmbedding, box_at, linear, make_people
from tests_rb.test_target import run as run_target

KIT_A = (200, 60, 40)
KIT_B = (40, 40, 210)
CFG = TrackerConfig.default()


def _people(*others):
    return make_people([("TARGET", KIT_A, 1.0)] + list(others))


def _universal(tc, e):
    tc.assertEqual(S.validate_document(e.doc, require_v2=True, cfg=CFG), [])
    tc.assertEqual(S.validate_document(S.strip_v2(e.doc)), [])
    tc.assertEqual(e.swaps(), [])
    tc.assertEqual(e.body_written_twice(), [])
    for f in range(1, e.n + 1):
        if e.target_det_box(f) is None:
            tc.assertFalse(e.target(f)["v2"]["identity_trusted"], f)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="r1g_aud_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def d(self, tag):
        p = os.path.join(self.tmp, tag)
        os.makedirs(p)
        return p

    def inputs(self, frames, ppl, start=2, ov=None):
        video = write_clip(os.path.join(self.tmp, "v.avi"), frames, ppl)
        cfg = TrackerConfig.with_overrides(dict(SCRIPTED, **(ov or {})))
        script, _ = detector_script(frames, start, 0.35)
        return video, cfg, script

    def run_direct(self, frames, ppl, out, start=2, ov=None, **kw):
        video, cfg, script = self.inputs(frames, ppl, start, ov)
        return video, R.run(video, start, list(frames[start].scene.boxes["TARGET"]), out_dir=out,
                            config_overrides=dict(SCRIPTED, **(ov or {})), detector=ScriptedDetector(script, cfg),
                            embedder=SignatureEmbedding(ppl), **kw)


class Rule28(Base):
    """Test audit C1: the target's re-emerging body is never written as another person while T001 is on it."""

    def test_crossings_and_occlusion_never_write_the_target_body_twice(self):
        n = 120
        scenes = {
            "xing": clip({"TARGET": walk(n, (150, 200), (3, 0)), "PERSON_A": walk(n, (150 + 3 * n, 206), (-3, 0))}),
            "occ": clip({"TARGET": [box(320, 180)] * n, "PERSON_A": walk(n, (120, 186), (4, 0))}),
        }
        for name, fr in scenes.items():
            with self.subTest(scene=name):
                e = run_clip(fr, _people(("PERSON_A", KIT_B, 0.2)), self.tmp, name=name, start_frame=2)
                _universal(self, e)

    def test_lone_target_after_a_gap_has_no_ghost(self):
        """Only the target exists: any population record would be the target written twice (coasting ghost)."""
        fr = clip({"TARGET": [box(320, 180)] * 90}, drop=lambda k: ["TARGET"] if 20 <= k < 28 else [])
        e = run_clip(fr, _people(), self.tmp, name="ghost", start_frame=2)
        for f in range(1, e.n + 1):
            self.assertEqual(e.population(f), [], f"frame {f}: a population record exists with only the target")


class TargetRecordContract(Base):
    def test_overlap_contest_names_the_target(self):
        n = 120
        fr = clip({"TARGET": walk(n, (150, 200), (3, 0)), "PERSON_A": walk(n, (150 + 3 * n, 206), (-3, 0))})
        e = run_clip(fr, _people(("PERSON_A", KIT_B, 0.2)), self.tmp, name="ov", start_frame=2)
        with_t = [r["frame"] for r in e.doc["frame_log"] if any("T001" in p for p in r["overlap_contests"])]
        self.assertTrue(with_t, "no frame names T001 in overlap_contests despite a body crossing")

    def test_population_coasting_record_semantics(self):
        n = 60
        fr = clip({"TARGET": [box(150, 200)] * n, "PERSON_A": walk(n, (420, 200), (1, 0))},
                  drop=lambda k: ["PERSON_A"] if 30 <= k < 36 else [])
        e = run_clip(fr, _people(("PERSON_A", KIT_B, 0.2)), self.tmp, name="coast", start_frame=2)
        coasting = [(f, p) for f in range(1, e.n + 1) for p in e.population(f) if not p["v2"]["detected_this_frame"]]
        self.assertTrue(coasting, "the scene must make a population record coast")
        for f, p in coasting:
            self.assertEqual((p["state"], p["occlusion_status"], p["position_source"]),
                             ("TEMPORARILY_LOST", "FULL", "PREDICTED"))
            self.assertEqual(p["frames_since_observed_position"], f - p["last_seen_frame"])
            self.assertGreater(p["frames_since_observed_position"], 0)
            last = [q for q in e.population(p["last_seen_frame"]) if q["track_id"] == p["track_id"]]
            self.assertEqual(p["bbox"], last[0]["observed_position_box"], f"frame {f}: bbox = last observed box")
            self.assertEqual(p["identity_confidence"], 0.0)


class EpisodeVerification(unittest.TestCase):
    def test_verified_episode_stays_trusted_after_the_competitor_leaves(self):
        """Test audit A9 / M22: once verified, later uncontrasted continuation frames stay trusted (> 0.5)."""
        ppl = _people(("PERSON_A", KIT_A, 0.75))
        frames = [SceneFrame({"TARGET": box_at(500, 400, 40, 100)}) for _ in range(10)]
        frames += [SceneFrame({}) for _ in range(40)]
        frames += [SceneFrame({"TARGET": box_at(500, 400, 40, 100)}) for _ in range(30)]
        frames += [SceneFrame({"TARGET": box_at(500, 400, 40, 100), "PERSON_A": box_at(580, 400, 40, 100)})
                   for _ in range(30)]
        frames += [SceneFrame({"TARGET": box_at(500, 400, 40, 100)}) for _ in range(30)]
        r = run_target(frames, ppl)
        trusted = [k for k, f in enumerate(r.frames) if k >= 50 and f.identity_trusted]
        self.assertTrue(trusted)
        first = min(trusted)
        self.assertLess(first, 110)
        for f in r.frames[first:]:
            self.assertEqual(f.rb_state, T.TRACKING)
            self.assertTrue(f.identity_trusted, f.rel_frame)
            self.assertGreater(f.identity_confidence_rb, 0.5)


class Conflicts(Base):
    def test_a_lone_returning_target_logs_no_conflict(self):
        fr = clip({"TARGET": [box(320, 180)] * 100}, drop=lambda k: ["TARGET"] if 10 <= k < 50 else [])
        e = run_clip(fr, _people(), self.tmp, name="lone", start_frame=2)
        self.assertEqual(e.summary["identity_conflict_counts"], {})
        self.assertEqual(e.summary["identity_conflicts_logged"], 0)

    def test_conflict_counts_are_frame_counts(self):
        ppl = _people(("TWIN", KIT_A, 1.0))
        n = 90
        fr = clip({"TARGET": [box(320, 180)] * n, "TWIN": [None] * 20 + [box(320 + PW * 2.5, 180)] * (n - 20)})
        e = run_clip(fr, ppl, self.tmp, name="cc", start_frame=2)
        logged = e.summary["identity_conflicts_logged"]
        counts = e.summary["identity_conflict_counts"]
        untrusted = sum(1 for f in range(1, e.n + 1) if not e.target(f)["v2"]["identity_trusted"])
        self.assertGreater(logged, 0)
        self.assertLessEqual(logged, untrusted)
        self.assertEqual(max(counts.values()), logged)          # one rival: every conflict frame names it
        self.assertTrue(all(k.startswith("T001|") for k in counts))

    def test_reacquisition_rival_is_logged(self):
        """An identical person (edge crop, no bank) appears during the target's verification and blocks it."""
        ppl = _people(("TWIN", KIT_A, 0.97))
        tgt = [box(110, 180)] * 10 + [None] * 40 + [box(170, 180)] * 45
        twin = [None] * 55 + [box(PW / 2, 180)] * 40
        e = run_clip(clip({"TARGET": tgt, "TWIN": twin}), ppl, self.tmp, name="rival", start_frame=2)
        twin_ids = {p["track_id"] for f in range(1, e.n + 1) for p in e.population(f)
                    if e.target_det_box(f, "TWIN") is not None
                    and iou(e.target_det_box(f, "TWIN"), p["observed_position_box"]) > 0.5}
        self.assertTrue(twin_ids)
        self.assertTrue(any(k.split("|")[1] in twin_ids for k in e.summary["identity_conflict_counts"]))


class ConflictScope(unittest.TestCase):
    """Unit level: which decision entries count as identity conflicts (post-fix mutants K15 / K16)."""

    class _Obs:
        def __init__(self, tracklet_id, claimants=()):
            self.tracklet_id, self.claimants = tracklet_id, tuple(claimants)

    class _Res:
        def __init__(self, obs):
            self.observations = obs

    class _TF:
        def __init__(self, decision, trusted=False):
            self.decision, self.identity_trusted = decision, trusted

    def _builder(self):
        return R._RecordBuilder(CFG, [0.0, 0.0, 10.0, 20.0])

    def test_merged_box_stands_for_its_written_claimants(self):
        b = self._builder()
        res = self._Res([self._Obs(None, ("T005", "T006")), self._Obs("T007")])
        self.assertEqual(b.written_ids("D0", res, set()), {"T005", "T006"})
        self.assertEqual(b.written_ids("D0", res, {"T006"}), {"T005", CFG.get("target_track_id")})
        tf = self._TF({"candidates": [{"role": T.CONTINUATION, "passed": False, "best_competitor": "D0",
                                       "unknown_contacts": []}]})
        b.record_conflicts(tf, res, set(), {"T005", "T006", "T007"})
        self.assertEqual(b.conflicts, {"T001|T005": 1, "T001|T006": 1})

    def test_only_the_targets_own_competitors_and_rivals_count(self):
        b = self._builder()
        res = self._Res([self._Obs("T002"), self._Obs("T003"), self._Obs("T004")])
        written = {"T002", "T003", "T004", "T009"}
        tf = self._TF({"candidates": [
            # a rival box judged against its own competitor: not about the target
            {"role": T.RIVAL, "passed": False, "best_competitor": "T009", "unknown_contacts": ["T009"]},
            # the continuing box passed: its competitor was beaten, not a conflict
            {"role": T.CONTINUATION, "passed": True, "best_competitor": "T002", "unknown_contacts": []}],
            "continuation_rivals": [1]})
        b.record_conflicts(tf, res, set(), written)
        self.assertEqual(b.conflicts, {"T001|T003": 1})
        tf2 = self._TF({"candidates": [], "reacquisition": {"rivals": [2]}})
        b.record_conflicts(tf2, res, set(), written)
        self.assertEqual(b.conflicts, {"T001|T003": 1, "T001|T004": 1})
        self.assertEqual(b.conflict_frames, 2)
        b.record_conflicts(self._TF({"candidates": [], "continuation_rivals": [1]}, trusted=True), res, set(), written)
        self.assertEqual(b.conflict_frames, 2)                  # a trusted frame has no conflict


class RunArguments(Base):
    def test_max_frames_argument(self):
        frames = clip({"TARGET": walk(40, (150, 200), (3, 0))})
        _, s = self.run_direct(frames, _people(), self.d("mf"), max_frames=7)
        self.assertEqual(s["processed_frames"], 7)

    def test_bad_max_frames_is_refused_and_nothing_written(self):
        frames = clip({"TARGET": walk(20, (150, 200), (3, 0))})
        for bad in (0, -3):
            with self.subTest(max_frames=bad):
                out = self.d(f"mf{bad}")
                with self.assertRaises(R.RunRefusedError):
                    self.run_direct(frames, _people(), out, max_frames=bad)
                self.assertEqual(os.listdir(out), [])

    def test_seed_on_the_last_frame_is_refused(self):
        frames = clip({"TARGET": walk(6, (150, 200), (3, 0))})
        out = self.d("last")
        with self.assertRaises(R.RunRefusedError):
            self.run_direct(frames, _people(), out, start=5)
        self.assertEqual(os.listdir(out), [])

    def test_start_frame_zero_and_other_frame_rates(self):
        for fps in (25.0, 50.0):
            with self.subTest(fps=fps):
                n = 120
                fr = clip({"TARGET": walk(n, (150, 200), (3, 0)), "PERSON_A": walk(n, (150 + 3 * n, 206), (-3, 0))})
                e = run_clip(fr, _people(("PERSON_A", KIT_A, 0.75)), self.tmp, name=f"fps{int(fps)}",
                             start_frame=0, fps=fps)
                _universal(self, e)
                self.assertEqual(e.doc["summary"]["fps"], fps)
                self.assertEqual(e.doc["frame_log"][0]["v2"]["video_frame_index"], 1)
                for r in e.doc["frame_log"]:
                    self.assertAlmostEqual(r["v2"]["timestamp_s"], round(r["v2"]["video_frame_index"] / fps, 4))

    def test_existing_decision_log_does_not_block_a_run_without_debug(self):
        frames = clip({"TARGET": walk(20, (150, 200), (3, 0))})
        out = self.d("nodebug")
        video = write_clip(os.path.join(self.tmp, "v.avi"), frames, _people())
        log = R.output_paths(out, video)["decision_log"]
        with open(log, "w") as fh:
            fh.write("an earlier debug run\n")
        self.run_direct(frames, _people(), out, debug=False)
        with open(log) as fh:
            self.assertEqual(fh.read(), "an earlier debug run\n")


class BackendRefusals(unittest.TestCase):
    """Test audit A11: each substitution refused on its own (no other wrong backend masking it)."""

    @classmethod
    def setUpClass(cls):
        cls.ppl = _people()

    def _script_det(self, cfg, name=None):
        class Named(ScriptedDetector):
            def provenance(self_):
                p = super().provenance()
                return dict(p, backend=name) if name else p

            def model_record(self_):
                return {"role": "person_detector", "filename": "x.onnx", "sha256": "0" * 64, "status": "TEST"} \
                    if name else None
        return Named([[]], cfg)

    def _emb(self, name=None):
        class Named(SignatureEmbedding):
            def provenance(self_):
                return {"backend": name} if name else super().provenance()

            def model_record(self_):
                return {"role": "appearance_embedding", "filename": "e.onnx", "sha256": "0" * 64,
                        "status": "TEST"} if name else None
        return Named(self.ppl)

    def test_each_wrong_backend_alone_is_refused(self):
        prod = TrackerConfig.default()
        scripted = TrackerConfig.with_overrides(SCRIPTED)
        cases = {
            "production config, scripted detector": (prod, self._script_det(prod), self._emb("dinov2_onnx")),
            "production config, scripted embedder": (prod, self._script_det(prod, "yolox_onnx"), self._emb()),
            "scripted config, yolox detector": (scripted, self._script_det(scripted, "yolox_onnx"), self._emb()),
            "scripted config, dinov2 embedder": (scripted, self._script_det(scripted), self._emb("dinov2_onnx")),
        }
        for name, (cfg, det, emb) in cases.items():
            with self.subTest(case=name):
                with self.assertRaises(R.RunRefusedError):
                    R.build_backends(cfg, det, emb)


class DeterminismTooling(unittest.TestCase):
    def test_compare_documents_reports_any_non_wall_clock_difference(self):
        e = None
        doc = {"summary": {"processing_time_s": 1.0, "v2": {"provenance": {"run_started_utc": "a"}}, "fps": 30.0},
               "frame_log": [{"frame": 1, "tracks": [{"bbox": [1.0, 2.0, 3.0, 4.0]}]}], "final_tracks": []}
        other = copy.deepcopy(doc)
        other["summary"]["processing_time_s"] = 9.0
        other["summary"]["v2"]["provenance"]["run_started_utc"] = "b"
        self.assertTrue(R.compare_documents(doc, other)["identical_except_wall_clock"])
        other["frame_log"][0]["tracks"][0]["bbox"][2] = 3.1
        cmp = R.compare_documents(doc, other)
        self.assertFalse(cmp["identical_except_wall_clock"])
        self.assertEqual(cmp["first_difference"], ".frame_log[0].tracks[0].bbox[2]")


class FailClosedEdges(Base):
    def test_stale_partial_files_are_refused_up_front_and_preserved(self):
        frames = clip({"TARGET": walk(30, (150, 200), (3, 0))})
        for suffix in (R.PARTIAL_SUFFIX, R.PARTIAL_SUFFIX + ".doc"):
            with self.subTest(suffix=suffix):
                out = self.d("stale" + str(len(suffix)))
                video = write_clip(os.path.join(self.tmp, "v.avi"), frames, _people())
                stale = R.output_paths(out, video)["results"] + suffix
                with open(stale, "w") as fh:
                    fh.write("earlier run")
                calls = []
                with self.assertRaises(R.RunRefusedError):
                    self.run_direct(frames, _people(), out, on_frame=lambda f, s: calls.append(f))
                self.assertEqual(calls, [], "must refuse before processing any frame")
                with open(stale) as fh:
                    self.assertEqual(fh.read(), "earlier run")

    def test_document_level_violation_stops_the_run(self):
        frames = clip({"TARGET": walk(30, (150, 200), (3, 0))})
        out = self.d("finish")
        orig = R._summary

        def bad(*a, **k):
            s, finals = orig(*a, **k)
            s["processed_frames"] += 1
            return s, finals
        R._summary = bad
        try:
            with self.assertRaises(R.ContractViolationError):
                self.run_direct(frames, _people(), out, debug=True)
        finally:
            R._summary = orig
        self.assertEqual(os.listdir(out), [])

    def test_decision_log_starts_with_the_seed(self):
        frames = clip({"TARGET": walk(30, (150, 200), (3, 0))})
        out = self.d("seedline")
        video, _ = self.run_direct(frames, _people(), out, debug=True)
        with open(R.output_paths(out, video)["decision_log"]) as fh:
            first = json.loads(fh.readline())
        self.assertEqual(first["frame"], 0)
        self.assertTrue(first["target"]["decision"]["seed"])
        self.assertEqual(first["target"]["rb_state"], "TRACKING")


class ImpostorAfterCut(Base):
    def test_above_floor_impostor_at_the_old_place_is_never_trusted(self):
        """After a cut, a look-alike above the reacquisition floor stands where the target was and the target is
        elsewhere; whatever the tracker accepts, the impostor is never a trusted target."""
        ppl = _people(("PERSON_A", KIT_A, 0.9))
        n1, n2 = 40, 90
        tgt = [box(300, 180)] * n1 + [box(520, 200)] * n2
        imp = [None] * n1 + [box(300, 180)] * n2
        e = run_clip(clip({"TARGET": tgt, "PERSON_A": imp}, shots=[0] * n1 + [1] * n2), ppl, self.tmp, name="imp",
                     start_frame=2)
        _universal(self, e)
        for f in range(1, e.n + 1):
            t = e.target(f)
            imp_box = e.target_det_box(f, "PERSON_A")
            if t["v2"]["identity_trusted"] and imp_box is not None:
                self.assertLess(iou(imp_box, t["observed_position_box"]), 0.5, f"frame {f}: impostor trusted")


if __name__ == "__main__":
    unittest.main()

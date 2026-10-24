# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: player_tracker_rb/output_schema.py; R0/R0_TRACKER_INTERFACE_MAP.md §2-§5;
#                 fixtures/historical_contract (frozen GOLD.json 88246efe, PC.json 40fa0df1)
"""Tests for player_tracker_rb.output_schema.

1. Both frozen historical Pass-1 JSONs validate as the historical contract (hash-checked first).
2. R1 documents with v2 blocks validate, and stripping v2 leaves a historically valid document.
3. Every safety invariant fails loudly when violated (held box reported as observed, untrusted identity above the
   actor_attribution threshold, merged TRACKING, non-TRACKING reaching legacy TRACKING, and so on).
"""
import copy
import hashlib
import json
import os
import unittest

from player_tracker_rb import output_schema as S
from player_tracker_rb.config import TrackerConfig
from tests_rb import _synth_docs as SD

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(os.path.dirname(HERE), "fixtures", "historical_contract")
EXEMPLARS = {
    "GOLD_DUEL_01_syo_mar_3036_3048_mo_tracking_results.json":
        "88246efea738ffe6f7203ae63c2ba17c56a595e8d9a3f74cfde40d11143c1dc5",
    "f5bd8ce0-pass_completed_01_mo_tracking_results.json":
        "40fa0df1f5e639ebd457eefb9e814fbad8096a580c79157b272f4a0f933f30fc",
}


def _load_exemplar(name):
    path = os.path.join(FIX, name)
    with open(path, "rb") as f:
        blob = f.read()
    return hashlib.sha256(blob).hexdigest(), json.loads(blob)


class TestHistoricalExemplars(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docs = {}
        for name, want in EXEMPLARS.items():
            got, doc = _load_exemplar(name)
            if got != want:
                raise AssertionError(f"frozen exemplar {name} hash {got} != recorded {want}")
            cls.docs[name] = doc

    def test_exemplars_validate_as_historical_contract(self):
        for name, doc in self.docs.items():
            with self.subTest(name):
                self.assertEqual(S.validate_document(doc, require_v2=False), [])

    def test_exemplars_are_not_r1_documents(self):
        for name, doc in self.docs.items():
            with self.subTest(name):
                self.assertTrue(S.validate_document(doc, require_v2=True))

    def test_strip_v2_is_identity_on_historical(self):
        for name, doc in self.docs.items():
            with self.subTest(name):
                self.assertEqual(S.strip_v2(doc), doc)

    def test_field_lists_match_exemplars_exactly(self):
        for name, doc in self.docs.items():
            with self.subTest(name):
                self.assertEqual(tuple(doc.keys()), S.TOP_LEVEL_KEYS)
                self.assertEqual(tuple(doc["summary"].keys()), S.SUMMARY_KEYS)
                self.assertEqual(tuple(doc["summary"]["health_metrics"].keys()), S.HEALTH_METRIC_KEYS)
                for fr in doc["frame_log"]:
                    self.assertEqual(tuple(fr.keys()), S.FRAME_KEYS)
                    for t in fr["tracks"]:
                        self.assertEqual(tuple(t.keys()), S.TRACK_FIELD_NAMES)
                for t in doc["final_tracks"]:
                    self.assertEqual(tuple(t.keys()), S.TRACK_FIELD_NAMES)

    def test_detection_confidence_is_quantised_historically(self):
        cfg = TrackerConfig.default()
        n_reid = 0
        for name, doc in self.docs.items():
            with self.subTest(name):
                for fr in doc["frame_log"]:
                    for t in fr["tracks"]:
                        self.assertEqual(t["detection_confidence"],
                                         S.historical_detection_confidence(cfg, t["state"], t["position_source"]))
                        n_reid += t["state"] == "REIDENTIFYING"
        self.assertEqual(n_reid, 11)  # R0 erratum E1: these are CONFIRMED_DETECTION yet 0.0

    def test_target_triples_were_all_observed_historically(self):
        seen = set()
        for doc in self.docs.values():
            for fr in doc["frame_log"]:
                for t in fr["tracks"]:
                    seen.add((t["state"], t["occlusion_status"], t["position_source"]))
        for triple in S.TARGET_TRIPLE_BY_RB_STATE.values():
            self.assertIn(triple, S.OBSERVED_HISTORICAL_TRIPLES)
            self.assertIn(triple, seen)
        self.assertIn(S.POPULATION_TRIPLE_DETECTED, seen)
        self.assertIn(S.POPULATION_TRIPLE_COASTING, seen)

    def test_selected_confidence_rule_matches_exemplars(self):
        for doc in self.docs.values():
            for fr in doc["frame_log"]:
                for t in fr["tracks"]:
                    if t["selected_player"] and t["state"] == "ACTIVE":
                        self.assertEqual(t["selected_player_confidence"], t["identity_confidence"])
                    elif t["selected_player"]:  # R0 erratum E2: lower than identity_confidence when not ACTIVE
                        self.assertLessEqual(t["selected_player_confidence"], t["identity_confidence"])
                    else:
                        self.assertEqual(t["selected_player_confidence"], 0.0)

    def test_historical_rejections(self):
        doc = copy.deepcopy(self.docs["GOLD_DUEL_01_syo_mar_3036_3048_mo_tracking_results.json"])
        doc["frame_log"][0]["overlap_contests"] = 0  # TPI stub form: not the real contract
        doc["frame_log"][1]["tracks"][0]["extra_field"] = 1  # new info outside v2
        doc["frame_log"][2]["tracks"][0]["state"] = "TERMINATED"  # only legal in final_tracks
        doc["frame_log"][3]["tracks"][0]["merged_into"] = "T002"  # only legal in final_tracks
        del doc["frame_log"][4]["tracks"][0]["identity_confidence"]
        errs = "\n".join(S.validate_document(doc))
        for needle in ("overlap_contests", "extra_field", "TERMINATED", "merged_into", "identity_confidence"):
            self.assertIn(needle, errs)


class TestStateMapping(unittest.TestCase):
    def test_r1_state_mapping(self):
        self.assertEqual(S.historical_state_for("TRACKING"), "ACTIVE")
        self.assertEqual(S.historical_state_for("OCCLUDED"), "OCCLUDED")
        self.assertEqual(S.historical_state_for("AMBIGUOUS"), "AMBIGUOUS")
        self.assertEqual(S.historical_state_for("REACQUIRING"), "REIDENTIFYING")
        self.assertEqual(S.historical_state_for("LOST", True), "TEMPORARILY_LOST")
        self.assertEqual(S.historical_state_for("LOST", False), "OUT_OF_FRAME")
        with self.assertRaises(ValueError):
            S.historical_state_for("PROVISIONAL")

    def test_only_tracking_reaches_legacy_tracking(self):
        for rb in S.RB_STATES:
            for inside in (True, False):
                legacy = S.legacy_state_for(S.historical_state_for(rb, inside))
                self.assertIn(legacy, S.LEGACY_STATES)
                self.assertEqual(legacy == "TRACKING", rb == "TRACKING", (rb, inside, legacy))

    def test_mapping_table_consistent(self):
        for (rb, inside), triple in S.TARGET_TRIPLE_BY_RB_STATE.items():
            self.assertEqual(triple[0], S.historical_state_for(rb, inside))
        self.assertEqual(S.legacy_state_for("PROVISIONAL"), "LOST")  # unknown key -> event_adapter default
        self.assertEqual(S.LEGACY_STATE_EVIDENCE["OUT_OF_FRAME"][1], "INFERRED")


class TestR1Documents(unittest.TestCase):
    def setUp(self):
        self.cfg = TrackerConfig.default()

    def test_synthetic_document_valid(self):
        doc = SD.build_doc(self.cfg)
        self.assertEqual(S.validate_document(doc, require_v2=True, cfg=self.cfg), [])

    def test_all_states_valid(self):
        states = [("TRACKING", True), ("AMBIGUOUS", True), ("OCCLUDED", True), ("LOST", True), ("LOST", False),
                  ("REACQUIRING", True), ("TRACKING", True)]
        episodes = [1, 1, 1, 1, 1, 1, 2]
        doc = SD.build_doc(self.cfg, states=states, episodes=episodes)
        self.assertEqual(S.validate_document(doc, require_v2=True, cfg=self.cfg), [])

    def test_strip_v2_leaves_valid_historical_document(self):
        doc = SD.build_doc(self.cfg)
        stripped = S.strip_v2(doc)
        self.assertEqual(S.validate_document(stripped, require_v2=False, cfg=self.cfg), [])
        self.assertNotIn("v2", json.dumps(stripped).replace('"video"', ""))
        # stripping removes only v2: every historical value is untouched
        for fr_a, fr_b in zip(doc["frame_log"], stripped["frame_log"]):
            for ta, tb in zip(fr_a["tracks"], fr_b["tracks"]):
                self.assertEqual({k: v for k, v in ta.items() if k != "v2"}, tb)

    def test_canonical_dump_deterministic(self):
        a = S.dumps_canonical(SD.build_doc(self.cfg))
        b = S.dumps_canonical(SD.build_doc(self.cfg))
        self.assertEqual(a, b)
        self.assertTrue(a.endswith("\n"))
        bad = SD.build_doc(self.cfg)
        bad["summary"]["processing_time_s"] = float("nan")
        with self.assertRaises(ValueError):
            S.dumps_canonical(bad)

    def test_rounding_matches_historical_precision(self):
        rec = S.build_track_record(
            self.cfg, track_id="T001", state="ACTIVE", age=1, last_seen_frame=1, bbox=[1.234, 2.25, 3.0, 4.96],
            observed_position_box=[1, 2, 3, 4], occlusion_status="NONE", position_source="CONFIRMED_DETECTION",
            selected_player=True, confidences=SD.confidences(0.85714), frames_since_observed_position=0,
            frames_since_confirmed_identity=0)
        self.assertEqual(rec["bbox"], [1.2, 2.2, 3.0, 5.0])
        self.assertEqual(rec["identity_confidence"], 0.857)
        self.assertEqual(rec["selected_player_confidence"], 0.857)
        self.assertEqual(rec["detection_confidence"], 1.0)
        self.assertEqual(tuple(rec.keys()), S.TRACK_FIELD_NAMES)

    def test_builder_refuses_bad_input(self):
        kw = dict(track_id="T001", state="ACTIVE", age=1, last_seen_frame=1, bbox=[0, 0, 1, 1],
                  observed_position_box=[0, 0, 1, 1], occlusion_status="NONE",
                  position_source="CONFIRMED_DETECTION", selected_player=True,
                  frames_since_observed_position=0, frames_since_confirmed_identity=0)
        conf = SD.confidences(0.8)
        with self.assertRaises(ValueError):
            S.build_track_record(self.cfg, confidences={**conf, "detection_confidence": 0.9}, **kw)
        missing = dict(conf)
        del missing["reid_confidence"]
        with self.assertRaises(ValueError):
            S.build_track_record(self.cfg, confidences=missing, **kw)
        with self.assertRaises(ValueError):
            S.build_track_record(self.cfg, confidences={**conf, "identity_confidence": 1.2}, **kw)
        with self.assertRaises(ValueError):
            S.build_track_record(self.cfg, confidences=conf, **{**kw, "bbox": [0, 0, 1]})


class TestInvariantViolations(unittest.TestCase):
    """Each mutation breaks one safety rule; the validator must name it."""

    def setUp(self):
        self.cfg = TrackerConfig.default()

    def _errs(self, mutate, states=None, episodes=None):
        doc = SD.build_doc(self.cfg, states=states, episodes=episodes)
        mutate(doc)
        return "\n".join(S.validate_document(doc, require_v2=True, cfg=self.cfg))

    @staticmethod
    def _target(doc, i):
        return doc["frame_log"][i]["tracks"][0]

    def test_held_box_reported_as_observed(self):
        def m(doc):
            t = self._target(doc, 3)  # OCCLUDED frame
            t["v2"]["bbox_is_observed"] = True
            t["v2"]["bbox_is_held"] = False
        self.assertIn("bbox_is_observed", self._errs(m))

    def test_held_box_written_as_confirmed_detection(self):
        def m(doc):
            t = self._target(doc, 3)
            t["state"], t["occlusion_status"], t["position_source"] = "ACTIVE", "NONE", "CONFIRMED_DETECTION"
            t["position_is_confirmed"] = True
            t["detection_confidence"] = 1.0
        errs = self._errs(m)
        self.assertIn("rb_state OCCLUDED written as", errs)
        self.assertIn("maps to legacy TRACKING", errs)

    def test_untrusted_identity_above_actor_attribution_threshold(self):
        def m(doc):
            t = self._target(doc, 3)
            t["identity_confidence"] = t["selected_player_confidence"] = 0.5
        self.assertIn("untrusted identity but identity_confidence", self._errs(m))

    def test_trusted_identity_requires_tracking(self):
        def m(doc):
            t = self._target(doc, 3)
            t["v2"]["identity_trusted"] = True
            t["v2"]["identity_uncertainty_reasons"] = []
        self.assertIn("identity_trusted requires rb_state TRACKING", self._errs(m))

    def test_untrusted_needs_a_reason(self):
        def m(doc):
            self._target(doc, 3)["v2"]["identity_uncertainty_reasons"] = []
        self.assertIn("must name at least one uncertainty reason", self._errs(m))

    def test_unknown_reason_rejected(self):
        def m(doc):
            self._target(doc, 3)["v2"]["identity_uncertainty_reasons"] = ["LOOKS_FINE"]
        self.assertIn("identity_uncertainty_reasons", self._errs(m))

    def test_merged_detection_cannot_be_tracking(self):
        def m(doc):
            t = self._target(doc, 0)
            t["v2"]["merged_suspected"] = True
        self.assertIn("merged-suspected detection cannot establish TRACKING", self._errs(m))

    def test_target_id_must_stay_t001(self):
        def m(doc):
            t = self._target(doc, 5)
            t["track_id"] = t["identity_id"] = t["online_track_id"] = "T064"
        self.assertIn("logical target id", self._errs(m))

    def test_detection_confidence_must_stay_quantised(self):
        def m(doc):
            self._target(doc, 0)["detection_confidence"] = 0.81  # a raw detector score
        self.assertIn("historical quantised value", self._errs(m))

    def test_one_selected_record_per_frame(self):
        def m(doc):
            doc["frame_log"][1]["tracks"][0]["selected_player"] = False
        self.assertIn("exactly one selected", self._errs(m))

        def m2(doc):
            doc["frame_log"][1]["tracks"][1]["selected_player"] = True
        self.assertIn("selected_player records", self._errs(m2))

    def test_non_historical_key_outside_v2(self):
        def m(doc):
            self._target(doc, 0)["bbox_is_held"] = False
        self.assertIn("non-historical keys outside 'v2'", self._errs(m))

        def m2(doc):
            doc["summary"]["schema_version"] = "2"
        self.assertIn("summary: non-historical keys", self._errs(m2))

        def m3(doc):
            doc["v2"] = {}
        self.assertIn("top-level keys", self._errs(m3))

    def test_frame_numbering(self):
        def m(doc):
            doc["frame_log"][2]["frame"] = 4
        self.assertIn("frames must run 1, 2, ..., N", self._errs(m))

        def m2(doc):
            doc["frame_log"][2]["v2"]["video_frame_index"] += 1
        self.assertIn("video_frame_index must equal start_frame + frame", self._errs(m2))

    def test_processed_frames_must_match(self):
        def m(doc):
            doc["summary"]["processed_frames"] += 1
        self.assertIn("processed_frames", self._errs(m))

    def test_reacquisition_events_must_match_episodes(self):
        states = [("TRACKING", True), ("LOST", True), ("REACQUIRING", True), ("TRACKING", True)]
        doc = SD.build_doc(self.cfg, states=states, episodes=[1, 1, 1, 2], n_events=0)
        errs = "\n".join(S.validate_document(doc, require_v2=True, cfg=self.cfg))
        self.assertIn("episode increments but 0 reacquisition_events", errs)
        doc = SD.build_doc(self.cfg, states=states, episodes=[1, 1, 1, 3])
        errs = "\n".join(S.validate_document(doc, require_v2=True, cfg=self.cfg))
        self.assertIn("may only stay the same or increase by 1", errs)

    def test_population_record_rules(self):
        def m(doc):
            doc["frame_log"][0]["tracks"][1]["state"] = "PROVISIONAL"
        self.assertIn("population triple", self._errs(m))

    def test_provenance_rules(self):
        def m(doc):
            doc["summary"]["v2"]["provenance"]["decode_mode"] = "RESUMED"
        self.assertIn("decode_mode", self._errs(m))

        def m2(doc):
            doc["summary"]["v2"]["provenance"]["config_hash"] = "x" * 64
        self.assertIn("config_hash must equal", self._errs(m2))

        def m3(doc):
            doc["summary"]["v2"]["provenance"]["seed_frame"] = 181
        self.assertIn("seed_frame must equal summary.start_frame", self._errs(m3))

    def test_final_tracks_carry_no_v2(self):
        def m(doc):
            doc["final_tracks"][0]["v2"] = {"record_kind": "TARGET"}
        self.assertIn("final_tracks records carry no v2", self._errs(m))


if __name__ == "__main__":
    unittest.main()

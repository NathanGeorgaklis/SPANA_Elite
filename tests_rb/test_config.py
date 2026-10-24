# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: player_tracker_rb/config.py; R0/R0_CONSTANTS_AND_THRESHOLDS.csv; R1/R1_APPROVALS.md (configuration rule)
"""Tests for player_tracker_rb.config: registry completeness, provenance, validation, units and hashing."""
import json
import math
import unittest

from player_tracker_rb import config as C
from player_tracker_rb.config import ConfigError, TrackerConfig


class TestRegistryCompleteness(unittest.TestCase):
    def test_every_param_documents_itself(self):
        for p in C.PARAMS:
            with self.subTest(p.name):
                self.assertTrue(p.name and p.name == p.name.strip())
                self.assertTrue(p.units.strip(), "units required")
                self.assertTrue(len(p.meaning.strip()) >= 10, "meaning required")
                self.assertTrue(p.evidence.strip(), "evidence/provenance required")
                self.assertIn(p.source, C.SOURCES)
                self.assertIn(p.evidence_level, C.EVIDENCE_LEVELS)
                self.assertIn(p.validation_status, C.VALIDATION_STATUSES)
                self.assertIn(p.role, C.ROLES)
                self.assertTrue(p.group)

    def test_names_unique(self):
        names = [p.name for p in C.PARAMS]
        self.assertEqual(len(names), len(set(names)))

    def test_defaults_pass_their_own_checks(self):
        for p in C.PARAMS:
            with self.subTest(p.name):
                self.assertIsNone(C._check_value(p, p.default))

    def test_new_behavioural_values_are_not_claimed_validated(self):
        # Nothing NEW_R1 that drives a decision may claim validation before synthetic/R2 evidence exists.
        for p in C.params_by(source="NEW_R1", role="BEHAVIOURAL"):
            with self.subTest(p.name):
                self.assertIn(p.validation_status, ("UNVALIDATED", "NOT_APPLICABLE"))

    def test_reference_only_params_are_historical(self):
        refs = C.params_by(role="REFERENCE_ONLY")
        self.assertTrue(refs)
        for p in refs:
            with self.subTest(p.name):
                self.assertEqual(p.source, "HISTORICAL_REFERENCE")
                self.assertTrue(p.name.startswith("hist_"))

    def test_historical_values_match_r0_constants_registry(self):
        # Spot-check against R0_CONSTANTS_AND_THRESHOLDS.csv (recovered values must not drift).
        cfg = TrackerConfig.default()
        expect_ref = {"hist_assoc_weight_appearance": 0.40, "hist_assoc_weight_motion": 0.20,
                      "hist_assoc_weight_iou_last_box": 0.10, "hist_assoc_weight_colour_hsv": 0.15,
                      "hist_assoc_weight_size": 0.05, "hist_min_appearance_floor": 0.30,
                      "hist_ambiguity_margin": 0.08, "hist_base_gate_px": 90.0,
                      "hist_gate_growth_px_per_frame": 35.0, "hist_max_gate_px": 350.0,
                      "hist_fast_path_radius_px": 45.0, "hist_occlusion_to_lost_frames": 10,
                      "hist_max_frames_default": 450}
        for k, v in expect_ref.items():
            self.assertEqual(cfg.reference_value(k), v, k)
        self.assertEqual(cfg.get("legacy_identity_trust_threshold"), 0.5)  # confidence.py MEDIUM_THRESH
        self.assertEqual(dict(cfg.get("detection_confidence_by_position_source")),
                         {"CONFIRMED_DETECTION": 1.0, "AMBIGUOUS_DETECTION": 0.5,
                          "WEAK_SPATIAL_RECOVERY": 0.5, "PREDICTED": 0.0})

    def test_new_params_with_historical_counterparts_record_them(self):
        p = C.REGISTRY
        self.assertEqual(p["appearance_floor"].historical_value, 0.30)
        self.assertEqual(p["ambiguity_margin"].historical_value, 0.08)
        self.assertEqual(p["gate_base_h"].historical_value, 90.0)
        self.assertEqual(p["max_occluded_s"].historical_value, 10)
        self.assertEqual(p["reacq_contrastive_margin"].historical_value, 0.08)

    def test_no_historical_fast_path_parameter_is_usable(self):
        cfg = TrackerConfig.default()
        with self.assertRaises(ConfigError):
            cfg.get("hist_fast_path_radius_px")
        self.assertFalse(any("fast_path" in p.name and p.role != "REFERENCE_ONLY" for p in C.PARAMS))

    def test_time_params_are_seconds_and_geometry_in_h(self):
        # D13: behavioural limits are in seconds (never frame counts) and geometry in target heights h.
        # Pixels are allowed only where the quantity is genuinely about image resolution or the detector input.
        for p in C.params_by(role="BEHAVIOURAL"):
            with self.subTest(p.name):
                if p.name.endswith("_per_s"):
                    self.assertIn("per s", p.units)
                elif p.name.endswith("_s"):
                    self.assertEqual(p.units, "s")
                if p.name.endswith("_h"):
                    self.assertEqual(p.units, "h")
                self.assertFalse(p.units.startswith("frames"), "behavioural limits must not be frame counts")
                if "px" in p.units:
                    self.assertTrue("image resolution" in p.units or "network input" in p.units, p.units)


class TestValidation(unittest.TestCase):
    def test_default_builds(self):
        cfg = TrackerConfig.default()
        self.assertEqual(cfg.get("target_track_id"), "T001")
        self.assertFalse(cfg.get("resume_allowed"))
        self.assertFalse(cfg.get("gate_grows_while_ambiguous"))

    def test_unknown_override_rejected(self):
        with self.assertRaises(ConfigError):
            TrackerConfig.with_overrides({"no_such_param": 1})

    def test_reference_only_cannot_be_overridden(self):
        with self.assertRaises(ConfigError):
            TrackerConfig.with_overrides({"hist_base_gate_px": 10.0})

    def test_type_range_allowed_checks(self):
        bad = [{"appearance_floor": "0.5"}, {"appearance_floor": 1.5}, {"appearance_floor": float("nan")},
               {"bank_max_samples": 2.5}, {"bank_max_samples": True}, {"resume_allowed": True},
               {"gate_grows_while_ambiguous": True}, {"detector_input_size": 800},
               {"lab_upper_band": (0.5,)}, {"onnx_sequential_execution": False}]
        for ov in bad:
            with self.subTest(ov):
                with self.assertRaises(ConfigError):
                    TrackerConfig.with_overrides(ov)

    def test_cross_checks(self):
        bad = [{"composite_weight_lab": 0.5},  # weights no longer sum to 1
               {"untrusted_identity_confidence_ceiling": 0.5},
               {"reacq_appearance_floor": 0.5},  # below ordinary floor 0.55
               {"reacq_contrastive_margin": 0.05},
               {"gate_base_h": 4.0},
               {"lab_lower_band": (0.8, 0.5)},
               {"merged_aspect_band": (1.1, 1.6)},
               {"lost_long_s": 0.5}]
        for ov in bad:
            with self.subTest(ov):
                with self.assertRaises(ConfigError):
                    TrackerConfig.with_overrides(ov)

    def test_valid_override_recorded(self):
        cfg = TrackerConfig.with_overrides({"composite_weight_lab": 0.4, "composite_weight_embedding": 0.6})
        self.assertEqual(cfg.get("composite_weight_lab"), 0.4)
        rec = cfg.as_record()
        self.assertEqual(rec["overrides"], [["composite_weight_embedding", 0.6], ["composite_weight_lab", 0.4]])

    def test_immutable(self):
        cfg = TrackerConfig.default()
        with self.assertRaises(TypeError):
            cfg.values["appearance_floor"] = 0.1
        with self.assertRaises(Exception):
            cfg.values = {}


class TestHashing(unittest.TestCase):
    def test_hash_deterministic_and_sensitive(self):
        a, b = TrackerConfig.default(), TrackerConfig.default()
        self.assertEqual(a.config_hash(), b.config_hash())
        self.assertEqual(len(a.config_hash()), 64)
        c = TrackerConfig.with_overrides({"max_ambiguous_s": 0.8})
        self.assertNotEqual(a.config_hash(), c.config_hash())

    def test_record_is_json_safe(self):
        rec = TrackerConfig.default().as_record()
        json.dumps(rec, allow_nan=False)
        json.dumps(C.registry_records(), allow_nan=False)
        self.assertEqual(rec["registry_version"], C.CONFIG_REGISTRY_VERSION)


class TestUnits(unittest.TestCase):
    def test_seconds_to_frames(self):
        self.assertEqual(C.seconds_to_frames(0, 30), 0)
        self.assertEqual(C.seconds_to_frames(1.0, 30), 30)
        self.assertEqual(C.seconds_to_frames(1.0, 60), 60)
        self.assertEqual(C.seconds_to_frames(0.001, 30), 1)
        self.assertEqual(C.seconds_to_frames(0.5, 29.97), 15)
        self.assertEqual(C.seconds_to_frames(1.0, 56.502242152466366), 57)  # PC.json fps
        for bad in [(-1, 30), (1, 0), (1, -30), (float("inf"), 30), (1, float("nan"))]:
            with self.assertRaises(ValueError):
                C.seconds_to_frames(*bad)

    def test_same_duration_at_different_fps(self):
        # D13: a limit in seconds covers the same real time at any frame rate (to within one frame).
        for s in (0.2, 0.5, 1.0, 3.0):
            for fps in (25.0, 30.0, 50.0, 56.502242152466366, 60.0):
                n = C.seconds_to_frames(s, fps)
                self.assertGreaterEqual(n / fps, s - 1e-9)
                self.assertLess((n - 1) / fps, s)

    def test_h_conversions(self):
        self.assertAlmostEqual(C.h_to_px(1.5, 80.0), 120.0)
        self.assertAlmostEqual(C.px_to_h(120.0, 80.0), 1.5)
        # 2x scale: same h distance, double the pixels
        self.assertAlmostEqual(C.h_to_px(0.75, 160.0), 2 * C.h_to_px(0.75, 80.0))
        with self.assertRaises(ValueError):
            C.h_to_px(1.0, 0.0)
        self.assertAlmostEqual(C.per_second_to_per_frame(2.0, 30.0), 2.0 / 30.0)
        self.assertTrue(math.isclose(C.frames_to_seconds(45, 30.0), 1.5))


if __name__ == "__main__":
    unittest.main()

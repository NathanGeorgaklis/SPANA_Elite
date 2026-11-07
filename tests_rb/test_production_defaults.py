# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2 production tracker consolidation (Nathan, 2026-11-06 10:12): freeze the strongest simple
#                 tracker; experimental layers stay out of the production path; the same-frame handoff rule is an
#                 optional mode whose production default was chosen on the complete DEV benchmark (OFF).
"""Pins the R2 production configuration so a later change to a default is a visible, deliberate act."""
import unittest

from player_tracker_rb import __version__
from player_tracker_rb.config import CONFIG_REGISTRY_VERSION, TrackerConfig


class ProductionDefaults(unittest.TestCase):
    def test_experimental_layers_are_off_by_default(self):
        c = TrackerConfig.default()
        self.assertFalse(c.get("not_target_memory_enabled"))
        self.assertFalse(c.get("target_reverify_enabled"))

    def test_handoff_safety_is_an_optional_mode(self):
        self.assertFalse(TrackerConfig.default().get("continuity_step_enabled"))
        self.assertTrue(TrackerConfig.with_overrides({"continuity_step_enabled": True}).get("continuity_step_enabled"))

    def test_production_path_imports_no_experimental_module(self):
        # static check: other tests may already have imported the experimental modules into sys.modules
        import inspect
        import re
        from player_tracker_rb import run_rb, target, population
        for mod in (run_rb, target, population):
            src = inspect.getsource(mod)
            for name in ("identity", "target_profile", "jersey"):
                self.assertIsNone(re.search(r"^\s*(from \.|import player_tracker_rb\.)\s*%s\b" % name, src, re.M),
                                  "%s imports experimental module %s" % (mod.__name__, name))

    def test_version_tags(self):
        self.assertEqual(CONFIG_REGISTRY_VERSION, "r2str-1")
        self.assertTrue(__version__.startswith("1.3.0-r2str"), __version__)


if __name__ == "__main__":
    unittest.main()

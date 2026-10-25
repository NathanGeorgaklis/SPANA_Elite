# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2 instruction (Nathan, 2026-10-26); tools_rb/r2_protocol.py
"""R2 pre-registration: exclusions, split determinism and clip-window rules."""
import unittest

from tools_rb import r2_protocol as P


class Exclusions(unittest.TestCase):
    def test_task3r_gold_and_gb_v2_windows_and_their_buffers_are_refused(self):
        for src, a, b, why in P.EXCLUDED_WINDOWS:
            with self.subTest(why=why):
                with self.assertRaises(ValueError):
                    P.check_window(src, a, a + P.CLIP_MIN_S)
                # just inside the buffer before the window
                start = a - P.EXCLUSION_BUFFER_S - P.CLIP_MIN_S + 1.0
                if start >= 0:
                    with self.assertRaises(ValueError):
                        P.check_window(src, start, start + P.CLIP_MIN_S)

    def test_holdout_is_not_a_source(self):
        for m in P.SOURCES.values():
            self.assertNotEqual(m["file"], "videoplayback (2).mp4")
            self.assertNotIn("esfc", m["file"].lower())
            self.assertNotIn("esfc", m["match"].lower())
        with self.assertRaises(ValueError):
            P.check_window("ESFC_BCFC", 100.0, 108.0)

    def test_task3r_ranges_match_the_extract_logs(self):
        r3 = {w: (a, b) for (s, a, b, w) in P.EXCLUDED_WINDOWS if w.startswith("Task 3R")}
        self.assertEqual(sorted(r3), sorted(f"Task 3R R{n:02d}" for n in (1, 2, 3, 4, 5, 10)))
        self.assertAlmostEqual(r3["Task 3R R10"][1], 3135.0)
        self.assertAlmostEqual(r3["Task 3R R04"][1], 5927.933333, places=5)


class Split(unittest.TestCase):
    def test_split_is_deterministic_balanced_and_by_block(self):
        rows = P.block_table()
        self.assertEqual(rows, P.block_table())
        for src in P.SOURCES:
            s = [r["split"] for r in rows if r["source"] == src]
            self.assertTrue(0.35 <= s.count("EVAL") / len(s) <= 0.65, (src, s.count("EVAL"), len(s)))
        self.assertEqual(P.split_of("SYO_MAR", 10.0, 18.0), P.split_of_block("SYO_MAR", 0))

    def test_window_crossing_a_block_boundary_is_refused(self):
        with self.assertRaises(ValueError):
            P.check_window("SYO_MAR", P.BLOCK_S - 3.0, P.BLOCK_S + 3.0)

    def test_clip_length_rules(self):
        with self.assertRaises(ValueError):
            P.check_window("SYO_MAR", 10.0, 10.0 + P.CLIP_MIN_S - 0.5)
        with self.assertRaises(ValueError):
            P.check_window("SYO_MAR", 10.0, 10.0 + P.CLIP_MAX_S + 0.5)
        with self.assertRaises(ValueError):
            P.check_window("SYO_MAR", 8520.0, 8530.0)            # past the end of the source


if __name__ == "__main__":
    unittest.main()

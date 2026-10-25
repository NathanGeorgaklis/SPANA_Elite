# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R1/R1_DESIGN_DECISIONS.md D02, D21; R1/R1_APPROVALS.md (rebuild header; no hidden magic thresholds)
"""Package hygiene for player_tracker_rb:

* every new Python file (player_tracker_rb/, tests_rb/, tools_rb/) carries the rebuild header;
* no imports from the historical packages, the event engine or recovered/ (D02), and no runtime dependency
  beyond the standard library, numpy, cv2 and onnxruntime (D21);
* no numeric literal in tracker code outside config.py, apart from a small structural allow-list (box length,
  0/1, percentages). A behavioural threshold must be a config parameter.
"""
import ast
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "player_tracker_rb")
HEADER = ("# SPANA engine v2-rebuild", "# REBUILT 2026-10 — NOT ORIGINAL SOURCE", "# Evidence basis:")
FORBIDDEN_TOP = {"football_analysis", "player_tracker", "player_tracker_mo", "event_engine_v1", "recovered",
                 "ultralytics", "torch", "torchvision", "scipy", "sklearn", "easyocr", "new_tracker_snapshots"}
ALLOWED_THIRD_PARTY = {"numpy", "cv2", "onnxruntime"}
# Structural numbers only: identity values, box/pair lengths, percent scale, and 3 (the last index of an
# [x1, y1, x2, y2] box, and the ndim of an H x W x C image).
ALLOWED_LITERALS = {0, 1, -1, 2, 3, 4, 0.0, 1.0, -1.0, 100.0}


def _py_files(*dirs):
    for d in dirs:
        full = os.path.join(ROOT, d)
        for dirpath, _, files in os.walk(full):
            if "__pycache__" in dirpath:
                continue
            for f in sorted(files):
                if f.endswith(".py"):
                    yield os.path.join(dirpath, f)


def _stdlib():
    names = getattr(sys, "stdlib_module_names", None)
    if names:
        return set(names)
    return {"abc", "argparse", "ast", "collections", "copy", "csv", "dataclasses", "datetime", "functools",
            "hashlib", "io", "itertools", "json", "math", "os", "pathlib", "platform", "re", "shutil", "statistics",
            "subprocess", "sys", "tempfile", "time", "types", "typing", "unittest", "__future__", "warnings"}


class TestHeaders(unittest.TestCase):
    def test_rebuild_header_everywhere(self):
        files = list(_py_files("player_tracker_rb", "tests_rb", "tools_rb"))
        self.assertTrue(files)
        for path in files:
            with self.subTest(os.path.relpath(path, ROOT)):
                with open(path, encoding="utf-8") as f:
                    head = [f.readline().rstrip("\n") for _ in range(3)]
                self.assertEqual(head[0], HEADER[0])
                self.assertEqual(head[1], HEADER[1])
                self.assertTrue(head[2].startswith(HEADER[2]) and len(head[2]) > len(HEADER[2]) + 3)


class TestImports(unittest.TestCase):
    def test_package_imports_only_allowed_modules(self):
        std = _stdlib()
        for path in _py_files("player_tracker_rb"):
            tree = ast.parse(open(path, encoding="utf-8").read())
            for node in ast.walk(tree):
                mods = []
                if isinstance(node, ast.Import):
                    mods = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    if node.level and node.level > 0:
                        continue  # relative import inside the package
                    mods = [node.module or ""]
                for m in mods:
                    top = m.split(".")[0]
                    with self.subTest(file=os.path.basename(path), module=m):
                        self.assertNotIn(top, FORBIDDEN_TOP)
                        self.assertTrue(top in std or top in ALLOWED_THIRD_PARTY or top == "player_tracker_rb",
                                        f"unexpected dependency {m}")


class TestNoMagicNumbers(unittest.TestCase):
    def test_no_numeric_thresholds_outside_config(self):
        offenders = []
        for path in _py_files("player_tracker_rb"):
            if os.path.basename(path) == "config.py":
                continue
            tree = ast.parse(open(path, encoding="utf-8").read())
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                        and not isinstance(node.value, bool) and node.value not in ALLOWED_LITERALS:
                    offenders.append(f"{os.path.basename(path)}:{node.lineno}: {node.value!r}")
        self.assertEqual(offenders, [], "numeric literals belong in config.py:\n" + "\n".join(offenders))


class TestFingerprintIsNotAnIdentityFeature(unittest.TestCase):
    """R1-B: the frame-content fingerprint is for alignment debugging only. Only the video layer, provenance and the
    output schema may refer to it; detection, appearance and state code must never read it. run_rb.py (R1-G) is the
    output writer: it copies the fingerprint into the contract's frame v2 field and the provenance sidecar and never
    passes it to a decision layer."""
    ALLOWED = {"video_io.py", "provenance.py", "output_schema.py", "config.py", "__init__.py", "run_rb.py"}
    TOKENS = ("thumb_sha1", "frame_fingerprint", "frame_thumb", "thumbs_digest")

    def test_fingerprint_confined_to_video_layer(self):
        offenders = []
        for path in _py_files("player_tracker_rb"):
            name = os.path.basename(path)
            if name in self.ALLOWED:
                continue
            src = open(path, encoding="utf-8").read()
            offenders += [f"{name}: {t}" for t in self.TOKENS if t in src]
        self.assertEqual(offenders, [])


class TestNoBenchmarkKnowledgeInTracker(unittest.TestCase):
    """R1/R1_GENERIC_TRACKER_REQUIREMENT.md: the production package knows only TARGET_PLAYER and other physical
    people. Benchmark subjects, clips, distractors and club names may appear only in labels, fixtures, tests,
    evaluation reports and notes, never in player_tracker_rb/ (code, comments or docstrings)."""
    import re as _re
    PATTERNS = (r"\bGOLD\b", r"gold_", r"\bSOL\b", r"\bMAR\b", r"#\s?23\b", r"#\s?4\b", r"\bsyo\b",
                r"syo_mar", r"duel_01", r"pass_completed", r"ball[ _-]?boy", r"royal[ -]?blue", r"\bPC\.json\b",
                r"\bGOLD\.json\b", r"3036_3048")

    def test_no_benchmark_terms_in_package(self):
        offenders = []
        for path in _py_files("player_tracker_rb"):
            with open(path, encoding="utf-8") as fh:
                lines = fh.readlines()
            for n, line in enumerate(lines, start=1):
                for pat in self.PATTERNS:
                    if self._re.search(pat, line, flags=self._re.IGNORECASE if pat.islower() else 0):
                        offenders.append(f"{os.path.basename(path)}:{n}: {pat}")
        self.assertEqual(offenders, [], "benchmark-specific terms in the production package:\n" + "\n".join(offenders))

    def test_patterns_catch_examples(self):
        bad = ["GOLD 266", "SOL #23", "MAR #4", "make_gold_label_sheet", "royal-blue ball boy", "syo_mar_3036_3048"]
        for text in bad:
            with self.subTest(text=text):
                self.assertTrue(any(self._re.search(p, text, flags=self._re.IGNORECASE if p.islower() else 0)
                                    for p in self.PATTERNS))
        for ok in ("resolution", "marked", "Solver", "golden-section"):
            with self.subTest(text=ok):
                self.assertFalse(any(self._re.search(p, ok, flags=self._re.IGNORECASE if p.islower() else 0)
                                     for p in self.PATTERNS))


if __name__ == "__main__":
    unittest.main()

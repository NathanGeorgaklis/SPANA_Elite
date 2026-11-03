# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2 DEV jersey-number validation dataset instruction (Nathan, 2026-11-04 08:24);
#                 tools_rb/r2_jersey_labels.py
"""Jersey truth schema: synthetic label documents only (no footage, no real labels)."""
import copy
import json
import os
import shutil
import tempfile
import unittest

from tools_rb import r2_jersey_labels as L

FPS, FRAMES, W, H = 60.0, 480, 1280, 720
GRID = L.keyframe_grid(FPS, FRAMES)
BOX = [100, 100, 160, 260]


def kf(vis="VISIBLE", num="READABLE", reason=None, box=BOX, unc=False):
    return {"visibility": vis, "box": box if vis in ("VISIBLE", "PARTIAL") else None, "number": num, "reason": reason,
            "human_uncertain": unc, "note": ""}


def base(cid="R2_JERSEY_X_J01"):
    k = {str(g): kf() for g in GRID}
    k[str(GRID[1])] = kf(num="UNREADABLE", reason="FRONT_ON")
    k[str(GRID[2])] = kf(vis="OUT_OF_FRAME", num="NOT_APPLICABLE")
    return {"schema": L.SCHEMA, "clip_id": cid, "clip_sha256": "a" * 64, "fps": FPS, "frames": FRAMES, "width": W,
            "height": H, "annotator": "test", "tool": "t", "saved_utc": "x", "clip_rejected": None,
            "target": {"description": "dark kit, left", "team": "A", "expected_number": 7,
                       "number_visibility": "READABLE_IN_SOME_FRAMES"},
            "keyframes": k,
            "competitors": {"P1": {"description": "same kit right", "relation": "SAME_TEAM", "number": 8,
                                   "readable_keyframes": [GRID[0]], "boxes": {str(GRID[0]): [300, 100, 360, 260]}}},
            "distractors": []}


DEV_WIN = ("MCFC_MIT", 2734.5, 2742.5)       # protocol DEV block, 60 fps, 1280x720
EVAL_WIN = ("MCFC_MIT", 600.0, 608.0)         # protocol EVAL block


def reg(cids, fps=FPS, frames=FRAMES, win=DEV_WIN, split="DEV"):
    return {"clips": [{"clip_id": c, "clip_sha256": "a" * 64, "fps": fps, "frames": frames, "split": split,
                       "source": win[0], "start_s": win[1], "end_s": win[2]} for c in cids]}


class Grid(unittest.TestCase):
    def test_half_second_grid(self):
        self.assertEqual(L.keyframe_grid(60.0, 480), list(range(0, 480, 30)))
        self.assertEqual(L.keyframe_grid(30.0, 240), list(range(0, 240, 15)))
        self.assertEqual(len(L.keyframe_grid(30.0, 240)), 16)


class Validate(unittest.TestCase):
    def bad(self, lab, frag):
        p = L.validate(lab)
        self.assertTrue(any(frag in x for x in p), f"expected {frag!r} in {p}")

    def test_valid(self):
        self.assertEqual(L.validate(base()), [])

    def test_unlabelled_keyframe(self):
        lab = base(); lab["keyframes"][str(GRID[3])]["visibility"] = None
        self.bad(lab, "visibility unlabelled")
        lab = base(); lab["keyframes"][str(GRID[3])]["number"] = None
        self.bad(lab, "number state unlabelled")

    def test_grid_must_be_exact(self):
        lab = base(); del lab["keyframes"][str(GRID[-1])]
        self.bad(lab, "exactly the 0.5 s grid")
        lab = base(); lab["keyframes"]["31"] = kf()
        self.bad(lab, "exactly the 0.5 s grid")

    def test_readable_needs_number_box_and_certainty(self):
        lab = base(); lab["target"]["expected_number"] = None; lab["target"]["number_visibility"] = "NEVER_VISIBLE"
        self.bad(lab, "READABLE but no expected_number")
        lab = base(); lab["keyframes"][str(GRID[0])]["box"] = None
        self.bad(lab, "needs a box")
        lab = base(); lab["keyframes"][str(GRID[0])]["human_uncertain"] = True
        self.bad(lab, "cannot be HUMAN_UNCERTAIN")

    def test_never_visible_consistency(self):
        lab = base()
        for g in GRID:
            if lab["keyframes"][str(g)]["number"] == "READABLE":
                lab["keyframes"][str(g)] = kf(num="UNREADABLE", reason="SIDE_ON")
        self.bad(lab, "no READABLE keyframe")
        lab["target"]["number_visibility"] = "NEVER_VISIBLE"
        self.bad(lab, "null exactly when")
        lab["target"]["expected_number"] = None
        self.assertEqual(L.validate(lab), [])

    def test_hidden_target(self):
        lab = base(); lab["keyframes"][str(GRID[2])]["box"] = BOX
        self.bad(lab, "must have no box")
        lab = base(); lab["keyframes"][str(GRID[2])]["number"] = "UNREADABLE"; lab["keyframes"][str(GRID[2])]["reason"] = "OTHER"
        self.bad(lab, "NOT_APPLICABLE exactly when")
        lab = base(); lab["keyframes"][str(GRID[0])]["number"] = "NOT_APPLICABLE"
        self.bad(lab, "NOT_APPLICABLE exactly when")

    def test_unreadable_reason(self):
        lab = base(); lab["keyframes"][str(GRID[1])]["reason"] = None
        self.bad(lab, "needs a reason")
        lab = base(); lab["keyframes"][str(GRID[0])]["reason"] = "FRONT_ON"
        self.bad(lab, "only allowed when UNREADABLE")
        lab = base(); lab["keyframes"][str(GRID[1])]["reason"] = "BANANA"
        self.bad(lab, "needs a reason")

    def test_numbers_and_boxes(self):
        for n in (0, 100, "7", True, 7.0):
            lab = base(); lab["target"]["expected_number"] = n
            self.bad(lab, "expected_number must be")
        lab = base(); lab["keyframes"][str(GRID[0])]["box"] = [0, 0, W + 1, 10]
        self.bad(lab, "invalid")
        lab = base(); lab["keyframes"][str(GRID[0])]["box"] = [50, 50, 40, 60]
        self.bad(lab, "invalid")

    def test_competitors(self):
        lab = base(); lab["competitors"]["P1"]["readable_keyframes"] = []
        self.bad(lab, "SAME_TEAM number given")
        lab = base(); lab["competitors"]["P1"]["readable_keyframes"] = [31]
        self.bad(lab, "must be keyframes")
        lab = base(); lab["competitors"]["P1"]["number"] = None
        self.bad(lab, "number is null")
        lab = base(); lab["competitors"]["P1"]["boxes"] = {"31": [1, 1, 5, 5]}
        self.bad(lab, "not a keyframe")
        lab = base(); lab["competitors"]["P1"]["relation"] = "FRIEND"
        self.bad(lab, "relation")

    def test_registry_mismatch_and_rejection(self):
        r = reg(["R2_JERSEY_X_J01"])["clips"][0]
        lab = base(); lab["clip_sha256"] = "b" * 64
        self.assertTrue(any("differs from the registry" in x for x in L.validate(lab, r)))
        lab = base(); lab["clip_rejected"] = "no suitable player"; lab["keyframes"] = {}
        self.assertEqual(L.validate(lab, r), [])
        lab["clip_rejected"] = "  "
        self.assertTrue(L.validate(lab, r))

    def test_summary_duplicate_derived(self):
        s = L.summary(base())
        self.assertEqual((s["readable"], s["unreadable"], s["not_visible"]), (len(GRID) - 2, 1, 1))
        self.assertEqual(s["same_team_competitor_numbers"], [8])
        self.assertFalse(s["duplicate_number"])
        lab = base(); lab["competitors"]["P2"] = {"description": "opponent", "relation": "OPPONENT", "number": 7,
                                                  "readable_keyframes": [GRID[0]], "boxes": {}}
        self.assertTrue(L.summary(lab)["duplicate_number"])


class ValidateMore(unittest.TestCase):
    """Rules found untested by the test audit (2026-11-04)."""

    def ok(self, lab):
        self.assertEqual(L.validate(lab), [])

    def bad(self, lab, frag):
        p = L.validate(lab)
        self.assertTrue(any(frag in x for x in p), f"expected {frag!r} in {p}")

    def test_partial_and_hidden(self):
        lab = base(); lab["keyframes"][str(GRID[3])] = kf(vis="PARTIAL")
        self.ok(lab)
        lab["keyframes"][str(GRID[3])]["box"] = None
        self.bad(lab, "needs a box")
        lab = base(); lab["keyframes"][str(GRID[3])] = kf(vis="HIDDEN", num="NOT_APPLICABLE")
        self.ok(lab)
        lab["keyframes"][str(GRID[3])]["box"] = BOX
        self.bad(lab, "must have no box")
        lab = base(); lab["keyframes"][str(GRID[3])] = kf(vis="HIDDEN", num="READABLE")
        self.bad(lab, "NOT_APPLICABLE exactly when")

    def test_human_uncertain(self):
        lab = base(); lab["keyframes"][str(GRID[3])] = kf(num="UNREADABLE", reason="OTHER", box=None, unc=True)
        self.ok(lab)
        lab["keyframes"][str(GRID[3])]["number"] = "READABLE"; lab["keyframes"][str(GRID[3])]["reason"] = None
        self.bad(lab, "cannot be HUMAN_UNCERTAIN")
        lab = base(); lab["keyframes"][str(GRID[3])]["human_uncertain"] = "yes"
        self.bad(lab, "human_uncertain must be")

    def test_reason_on_not_applicable(self):
        lab = base(); lab["keyframes"][str(GRID[2])]["reason"] = "OTHER"
        self.bad(lab, "only allowed when UNREADABLE")

    def test_empty_or_null_keyframes(self):
        lab = base(); lab["keyframes"] = {str(g): None for g in GRID}
        self.bad(lab, "missing or not an object")
        lab = base(); lab["keyframes"] = {g: kf() for g in GRID}         # int keys (not JSON) are not the grid
        self.bad(lab, "exactly the 0.5 s grid")

    def test_grid_other_fps(self):
        self.assertEqual(L.keyframe_grid(29.97, 240)[:3], [0, 15, 30])
        self.assertEqual(L.keyframe_grid(25.0, 200)[:3], [0, 12, 24])     # 12.5 rounds half to even
        self.assertEqual(L.keyframe_grid(50.0, 400)[:3], [0, 25, 50])
        lab = base(); lab["fps"], lab["frames"] = 30.0, 240
        lab["keyframes"] = {str(g): kf() for g in L.keyframe_grid(30.0, 240)}
        lab["competitors"]["P1"]["readable_keyframes"] = [0]; lab["competitors"]["P1"]["boxes"] = {}
        self.ok(lab)
        lab["keyframes"]["30"] = lab["keyframes"].pop("15")
        self.bad(lab, "exactly the 0.5 s grid")

    def test_registry_size_and_fields(self):
        c = reg(["R2_JERSEY_X_J01"])["clips"][0]
        self.assertEqual(L.validate(base(), c), [])
        for k, v in (("fps", 30.0), ("frames", 470), ("width", 1920), ("height", 1080)):
            lab = base(); lab[k] = v
            self.assertTrue(L.validate(lab, c), k)
        c2 = dict(c, source="NOT_A_SOURCE")
        self.assertTrue(L.validate(base(), c2))

    def test_required_fields(self):
        for path, v, frag in ((("annotator",), "", "annotator missing"), (("target", "description"), " ", "description"),
                              (("target", "team"), None, "target.team"),
                              (("target", "number_visibility"), "SOMETIMES", "number_visibility must")):
            lab = base(); d = lab
            for k in path[:-1]:
                d = d[k]
            d[path[-1]] = v
            self.bad(lab, frag)
        lab = base(); lab["distractors"] = "board shows 9"
        self.bad(lab, "distractors must be a list")

    def test_box_bounds(self):
        lab = base(); lab["keyframes"][str(GRID[0])]["box"] = [0, 0, W, H]
        self.ok(lab)
        for b in ([0, 0, W, H + 1], [-1, 0, 10, 10], [0, 10, 10, 10], [0, 0, 10], [0, 0, "10", 10], [0, 0, True, 10]):
            lab = base(); lab["keyframes"][str(GRID[0])]["box"] = b
            self.bad(lab, "invalid")

    def test_competitor_variants(self):
        lab = base(); lab["competitors"]["P1"]["boxes"] = {}
        self.ok(lab)
        lab = base(); lab["competitors"]["P1"]["boxes"] = {str(GRID[0]): [0, 0, W + 5, 10]}
        self.bad(lab, "invalid")
        for n in (0, 100, "8", True):
            lab = base(); lab["competitors"]["P1"]["number"] = n
            self.bad(lab, "number must be 1-99")
        lab = base(); lab["competitors"]["P1"]["readable_keyframes"] = ["0"]
        self.bad(lab, "must be keyframes")
        lab = base(); lab["competitors"]["P1"].update(relation="OPPONENT", readable_keyframes=[])
        self.ok(lab)
        lab = base(); lab["competitors"]["P1"]["description"] = ""
        self.bad(lab, "description missing")

    def test_same_team_clash_derived(self):
        lab = base(); lab["competitors"]["P2"] = dict(lab["competitors"]["P1"], description="another")
        self.assertTrue(L.summary(lab)["same_team_number_clash"])
        self.assertFalse(L.summary(base())["same_team_number_clash"])


def _readable_to(lab, reason="FRONT_ON"):
    for g in GRID:
        if lab["keyframes"][str(g)]["number"] == "READABLE":
            lab["keyframes"][str(g)] = kf(num="UNREADABLE", reason=reason)
    return lab


class FreezeAndRecord(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.mkdtemp(prefix="r2_jersey_")
        self.reg = os.path.join(self.t, "reg.json")
        with open(self.reg, "w") as f:
            json.dump(reg(["R2_JERSEY_X_J01", "R2_JERSEY_X_J02"]), f)
        self.out = os.path.join(self.t, "frozen")

    def tearDown(self):
        for root, _, files in os.walk(self.t):
            for n in files:
                os.chmod(os.path.join(root, n), 0o644)
        shutil.rmtree(self.t)

    def _write(self, lab, name=None):
        p = os.path.join(self.t, name or (lab["clip_id"] + ".json"))
        with open(p, "w") as f:
            json.dump(lab, f, indent=2)
        return p

    def _sha(self, p):
        import hashlib
        return hashlib.sha256(open(p, "rb").read()).hexdigest()

    def test_freeze_is_byte_copy_with_record(self):
        src = self._write(base())
        r = L.freeze(src, self.reg, self.out)
        fp = L.frozen_path(self.out, "R2_JERSEY_X_J01", 1)
        self.assertEqual(open(fp, "rb").read(), open(src, "rb").read())
        self.assertEqual(r["sha256"], self._sha(fp))
        self.assertEqual(r["sha256"], self._sha(src))
        self.assertEqual((r["version"], r["supersedes"], r["reason"]), (1, None, None))
        self.assertEqual(r["registry_sha256"], self._sha(self.reg))
        self.assertEqual(os.stat(fp).st_mode & 0o222, 0)

    def test_freeze_refuses_invalid_unknown_eval_and_registry_mismatch(self):
        bad = base(); bad["keyframes"][str(GRID[0])]["number"] = None
        with self.assertRaises(SystemExit):
            L.freeze(self._write(bad), self.reg, self.out)
        self.assertEqual(L._versions(self.out, "R2_JERSEY_X_J01"), [])
        with self.assertRaises(SystemExit):
            L.freeze(self._write(base("R2_JERSEY_NOT_REGISTERED")), self.reg, self.out)
        for k, v in (("fps", 30.0), ("frames", 479), ("clip_sha256", "b" * 64), ("width", 1920)):
            lab = base(); lab[k] = v
            with self.assertRaises(SystemExit, msg=k):
                L.freeze(self._write(lab), self.reg, self.out)
        for split, win in (("EVAL", DEV_WIN), ("DEV", EVAL_WIN)):
            rp = os.path.join(self.t, f"reg_{split}.json")
            with open(rp, "w") as f:
                json.dump(reg(["R2_JERSEY_X_J01"], win=win, split=split), f)
            with self.assertRaises(SystemExit):
                L.freeze(self._write(base()), rp, self.out)
        self.assertEqual(L._versions(self.out, "R2_JERSEY_X_J01"), [])

    def test_new_version_needs_reason_and_keeps_old(self):
        L.freeze(self._write(base()), self.reg, self.out)
        lab2 = base(); lab2["target"]["description"] = "corrected"
        with self.assertRaises(SystemExit):
            L.freeze(self._write(lab2), self.reg, self.out)
        r = L.freeze(self._write(lab2), self.reg, self.out, reason="box at f30 was the wrong player")
        self.assertEqual((r["version"], r["supersedes"]), (2, "R2_JERSEY_X_J01.jersey.v1.FROZEN.json"))
        self.assertTrue(os.path.exists(L.frozen_path(self.out, "R2_JERSEY_X_J01", 1)))
        L.freeze(self._write(self._never("R2_JERSEY_X_J02")), self.reg, self.out)
        rec = L.dataset_record(self.out, self.reg)
        self.assertIn("R2_JERSEY_X_J01.jersey.v2.FROZEN.json", [f["file"] for f in rec["files"]])
        self.assertEqual([s["file"] for s in rec["superseded"]], ["R2_JERSEY_X_J01.jersey.v1.FROZEN.json"])

    def _never(self, cid):
        b = _readable_to(base(cid))
        b["target"]["expected_number"] = None
        b["target"]["number_visibility"] = "NEVER_VISIBLE"
        return b

    def test_record_fails_closed_until_complete_then_stable(self):
        L.freeze(self._write(base()), self.reg, self.out)
        with self.assertRaises(SystemExit):
            L.dataset_record(self.out, self.reg)
        L.freeze(self._write(self._never("R2_JERSEY_X_J02")), self.reg, self.out)
        r1 = L.dataset_record(self.out, self.reg)
        self.assertEqual(r1["dataset_sha256"], L.dataset_record(self.out, self.reg)["dataset_sha256"])
        cov = r1["coverage"]
        self.assertEqual((cov["clips_never_visible"], cov["readable"], cov["unreadable"], cov["not_visible"]),
                         (1, len(GRID) - 2, 1 + len(GRID) - 1, 2))
        self.assertEqual(cov["clips_with_duplicate_number"], 0)
        self.assertEqual(cov["clips_with_same_team_competitor_numbers"], 2)

    def test_tampered_or_unrecorded_frozen_file_fails_closed(self):
        L.freeze(self._write(base()), self.reg, self.out)
        L.freeze(self._write(self._never("R2_JERSEY_X_J02")), self.reg, self.out)
        p = L.frozen_path(self.out, "R2_JERSEY_X_J01", 1)
        os.chmod(p, 0o644)
        d = json.load(open(p)); d["annotator"] = "someone else"
        with open(p, "w") as f:
            json.dump(d, f)
        with self.assertRaises(SystemExit):
            L.dataset_record(self.out, self.reg)
        # a hand-made "frozen" file without a freeze record
        shutil.copyfile(p, L.frozen_path(self.out, "R2_JERSEY_X_J01", 7))
        with self.assertRaises(SystemExit):
            L.dataset_record(self.out, self.reg)

    def test_unrecorded_extra_version_fails_closed(self):
        L.freeze(self._write(base()), self.reg, self.out)
        L.freeze(self._write(self._never("R2_JERSEY_X_J02")), self.reg, self.out)
        shutil.copyfile(L.frozen_path(self.out, "R2_JERSEY_X_J01", 1), L.frozen_path(self.out, "R2_JERSEY_X_J01", 7))
        with self.assertRaises(SystemExit):
            L.dataset_record(self.out, self.reg)

    def test_record_checks_registry_and_digest_formula(self):
        import hashlib
        L.freeze(self._write(base()), self.reg, self.out)
        L.freeze(self._write(self._never("R2_JERSEY_X_J02")), self.reg, self.out)
        rec = L.dataset_record(self.out, self.reg)
        want = hashlib.sha256("\n".join(f"{f['file']}  {f['sha256']}" for f in rec["files"]).encode()).hexdigest()
        self.assertEqual(rec["dataset_sha256"], want)
        r = json.load(open(self.reg)); r["clips"][0]["clip_sha256"] = "c" * 64
        with open(self.reg, "w") as f:
            json.dump(r, f)
        with self.assertRaises(SystemExit):
            L.dataset_record(self.out, self.reg)

    def test_rejected_label_still_bound_to_registry(self):
        rj = base(); rj["clip_rejected"] = "no suitable player"; rj["clip_sha256"] = "d" * 64
        with self.assertRaises(SystemExit):
            L.freeze(self._write(rj), self.reg, self.out)

    def test_rejected_clip_counted_not_covered(self):
        L.freeze(self._write(base()), self.reg, self.out)
        rj = base("R2_JERSEY_X_J02"); rj["clip_rejected"] = "no suitable player"
        L.freeze(self._write(rj), self.reg, self.out)
        rec = L.dataset_record(self.out, self.reg)
        self.assertEqual(rec["n_rejected"], 1)
        self.assertEqual(rec["coverage"]["keyframes"], len(GRID))

    def test_duplicate_counted(self):
        d = base(); d["competitors"]["P2"] = {"description": "opponent", "relation": "OPPONENT", "number": 7,
                                             "readable_keyframes": [], "boxes": {}}
        L.freeze(self._write(d), self.reg, self.out)
        L.freeze(self._write(self._never("R2_JERSEY_X_J02")), self.reg, self.out)
        self.assertEqual(L.dataset_record(self.out, self.reg)["coverage"]["clips_with_duplicate_number"], 1)


class AnnotatorPage(unittest.TestCase):
    def test_page_emits_the_schema_names(self):
        here = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools_rb",
                            "r2_jersey_annotator.html")
        s = open(here).read()
        self.assertIn(f'SCHEMA="{L.SCHEMA}"', s)
        for v in L.VIS + L.UNREADABLE_REASONS + L.NUMBER_VISIBILITY + L.RELATIONS + ("READABLE", "UNREADABLE"):
            self.assertTrue(f'"{v}"' in s or f"<option>{v}</option>" in s, v)
        self.assertNotIn("ocr", s.lower().replace("no tracker, detector or ocr output", ""))


if __name__ == "__main__":
    unittest.main()

# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2 label-freeze instruction (Nathan, 2026-10-27); tools_rb/r2_label_intake.py
"""R2 label intake: synthetic registry, clips and label documents only (no footage, no real labels)."""
import copy
import hashlib
import json
import os
import shutil
import tempfile
import unittest

from tools_rb import r2_label_intake as I
from tools_rb import r2_labels as L
from tests_rb.test_r2_labels import base, frame

DEV_WIN = ("SYO_MAR", 1482.5, 1492.5)      # protocol DEV
EVAL_WIN = ("SYO_MAR", 650.0, 660.0)       # protocol EVAL


def _sha(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class Intake(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.mkdtemp(prefix="r2_intake_")
        self.inc, self.clips, self.out = (os.path.join(self.t, d) for d in ("incoming", "clips", "frozen"))
        os.makedirs(self.inc)
        os.makedirs(self.clips)
        self.reg = {"clips": []}
        # A: valid DEV; B: invalid EVAL (unlabelled tail); C: missing; D: rejected EVAL; E: valid DEV with HU frames
        for cid, win in (("R2_A", DEV_WIN), ("R2_B", EVAL_WIN), ("R2_C", DEV_WIN), ("R2_D", EVAL_WIN),
                         ("R2_E", DEV_WIN)):
            sha = hashlib.sha256(cid.encode()).hexdigest()
            rec = {"clip_id": cid, "clip_sha256": sha, "frames": 30, "fps": 10.0, "source": win[0],
                   "start_s": win[1], "end_s": win[2], "split": I.P.split_of(*win)}
            self.reg["clips"].append(dict(rec))
            with open(os.path.join(self.clips, cid + ".extract.json"), "w") as f:
                json.dump(rec, f)
            with open(os.path.join(self.clips, cid + ".mp4"), "wb") as f:
                f.write(cid.encode())                          # bytes whose sha256 equals the registered sha
        self.regp = os.path.join(self.t, "reg.json")
        with open(self.regp, "w") as f:
            json.dump(self.reg, f)
        self._write("R2_A", self._lab("R2_A"))
        b = self._lab("R2_B")
        for i in range(20, 30):
            b["per_frame"][i] = dict(frame(), visibility=L.UNLABELLED, resolvable=L.UNLABELLED, occlusion=L.UNLABELLED)
        self._write("R2_B", b)
        d = self._lab("R2_D")
        d["clip_rejected"] = "celebration"
        self._write("R2_D", d)
        e = self._lab("R2_E")
        for i in (5, 6, 7, 12):
            e["per_frame"][i]["human_uncertain"] = True
        e["first_clean_reappearance"] = [8, 13]              # every return to a clean target is marked
        self._write("R2_E", e)

    def tearDown(self):
        shutil.rmtree(self.t, ignore_errors=True)

    def _lab(self, cid):
        lab = base()
        lab["clip_id"] = cid
        lab["clip_sha256"] = hashlib.sha256(cid.encode()).hexdigest()
        return lab

    def _write(self, cid, lab, raw=None):
        p = os.path.join(self.inc, cid + I.SUFFIX)
        with open(p, "w") as f:
            f.write(raw if raw is not None else json.dumps(lab, indent=1))
        return p

    def _snapshot_incoming(self):
        return {n: _sha(os.path.join(self.inc, n)) for n in sorted(os.listdir(self.inc))}

    def _by(self, led):
        return {e["clip_id"]: e for e in led["clips"]}

    def test_check_only_writes_nothing_and_classifies_every_clip(self):
        before = self._snapshot_incoming()
        led = I.intake(self.inc, self.clips, self.regp, self.out, do_freeze=False, hash_clips=True)
        e = self._by(led)
        self.assertEqual({k: v["outcome"] for k, v in e.items()},
                         {"R2_A": "VALID", "R2_B": "INVALID", "R2_C": "MISSING", "R2_D": "VALID", "R2_E": "VALID"})
        self.assertFalse(os.path.exists(self.out))
        self.assertEqual(self._snapshot_incoming(), before)
        self.assertTrue(any("frame 20: not labelled" in p for p in e["R2_B"]["problems"]))
        self.assertEqual(e["R2_B"]["unlabelled_after_seed"], 10)
        self.assertEqual((led["found"], led["registered"], led["frozen"]), (4, 5, 0))

    def test_freeze_only_valid_files_unchanged_and_record_splits(self):
        before = self._snapshot_incoming()
        led = I.intake(self.inc, self.clips, self.regp, self.out, do_freeze=True)
        e = self._by(led)
        self.assertEqual(sorted(k for k, v in e.items() if v["outcome"] == "FROZEN"), ["R2_A", "R2_D", "R2_E"])
        self.assertFalse(os.path.exists(os.path.join(self.out, "R2_B" + I.SUFFIX)))   # invalid: never frozen
        for cid in ("R2_A", "R2_D", "R2_E"):
            self.assertEqual(_sha(os.path.join(self.out, cid + I.SUFFIX)), before[cid + I.SUFFIX])
            self.assertEqual(e[cid]["frozen_sha256"], before[cid + I.SUFFIX])
        self.assertEqual(self._snapshot_incoming(), before)                           # incoming untouched
        self.assertEqual((led["frozen_dev"], led["frozen_eval"]), (2, 1))
        self.assertFalse(os.path.exists(os.path.join(self.out, "R2_D.fixture.json")))  # rejected: no run fixture
        self.assertTrue(os.path.exists(os.path.join(self.out, "R2_A.fixture.json")))

    def test_human_uncertain_is_preserved_and_reported(self):
        led = I.intake(self.inc, self.clips, self.regp, self.out, do_freeze=True)
        e = self._by(led)["R2_E"]
        self.assertEqual(e["human_uncertain_frames"], 4)
        self.assertEqual(e["human_uncertain_ranges"], [[5, 7], [12, 12]])
        with open(os.path.join(self.out, "R2_E" + I.SUFFIX)) as f:
            frozen = json.load(f)
        self.assertEqual([i for i, x in enumerate(frozen["per_frame"]) if x["human_uncertain"]], [5, 6, 7, 12])

    def test_identity_scoring_exclusions(self):
        e = self._by(I.intake(self.inc, self.clips, self.regp, self.out))
        self.assertEqual(e["R2_D"]["identity_scoring"]["status"], "EXCLUDED")
        self.assertIn("celebration", e["R2_D"]["identity_scoring"]["reason"])
        self.assertEqual(e["R2_A"]["identity_scoring"]["status"], "INCLUDED")
        self.assertEqual(e["R2_B"]["identity_scoring"]["status"], "NOT_ASSESSED")
        lab = self._lab("R2_A")
        for i in range(3, 30):                               # nothing identifiable after the seed
            lab["per_frame"][i] = frame(res="NO")
        self.assertEqual(I.identity_scoring(lab, True)["status"], "EXCLUDED")

    def test_rerun_never_overwrites_a_frozen_label(self):
        I.intake(self.inc, self.clips, self.regp, self.out, do_freeze=True)
        frozen_sha = _sha(os.path.join(self.out, "R2_A" + I.SUFFIX))
        changed = self._lab("R2_A")
        changed["annotator"] = "someone else"
        self._write("R2_A", changed)
        led = I.intake(self.inc, self.clips, self.regp, self.out, do_freeze=True)
        e = self._by(led)["R2_A"]
        self.assertEqual(e["outcome"], "ALREADY_FROZEN")
        self.assertFalse(e["frozen_matches_incoming"])
        self.assertEqual(_sha(os.path.join(self.out, "R2_A" + I.SUFFIX)), frozen_sha)

    def test_identity_and_membership_mismatches_fail_closed(self):
        lab = self._lab("R2_A")
        lab["clip_id"] = "R2_E"                              # file for one clip claims to be another
        self._write("R2_A", lab)
        reg = copy.deepcopy(self.reg)
        reg["clips"][4]["split"] = "EVAL"                    # registry disagrees with the protocol for R2_E
        with open(self.regp, "w") as f:
            json.dump(reg, f)
        e = self._by(I.intake(self.inc, self.clips, self.regp, self.out, do_freeze=True))
        self.assertEqual(e["R2_A"]["outcome"], "INVALID")
        self.assertTrue(any("file name says R2_A" in p for p in e["R2_A"]["problems"]))
        self.assertEqual(e["R2_E"]["outcome"], "INVALID")
        self.assertTrue(any("split mismatch" in p for p in e["R2_E"]["problems"]))
        self.assertFalse(os.path.exists(os.path.join(self.out, "R2_E" + I.SUFFIX)))

    def test_clip_bytes_mismatch_and_bad_json_and_unknown_files(self):
        with open(os.path.join(self.clips, "R2_A.mp4"), "wb") as f:
            f.write(b"not the registered clip")
        self._write("R2_E", None, raw="{not json")
        with open(os.path.join(self.inc, "stray.json"), "w") as f:
            f.write("{}")
        led = I.intake(self.inc, self.clips, self.regp, self.out, do_freeze=True, hash_clips=True)
        e = self._by(led)
        self.assertEqual(e["R2_A"]["outcome"], "INVALID")
        self.assertTrue(any("clip file bytes" in p for p in e["R2_A"]["problems"]))
        self.assertEqual(e["R2_E"]["outcome"], "INVALID")
        self.assertEqual(led["unknown_incoming_files"], ["stray.json"])

    def test_ranges(self):
        self.assertEqual(I.ranges([3, 1, 2, 7, 9, 8]), [[1, 3], [7, 9]])
        self.assertEqual(I.ranges([]), [])

    def test_main_refuses_to_overwrite_a_ledger(self):
        ledger = os.path.join(self.t, "ledger.json")
        I.main([self.inc, self.clips, self.regp, self.out, ledger])
        with self.assertRaises(SystemExit):
            I.main([self.inc, self.clips, self.regp, self.out, ledger])

    def test_no_model_or_tracker_import(self):
        with open(I.__file__) as f:
            imports = [ln for ln in f if ln.lstrip().startswith(("import ", "from "))]
        for bad in ("player_tracker_rb", "onnxruntime", "cv2", "detector", "embedding", "run_rb"):
            self.assertFalse([ln for ln in imports if bad in ln], bad)


if __name__ == "__main__":
    unittest.main()

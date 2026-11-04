# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: Nathan, 2026-11-04 18:44 (shorts numbers as equal evidence; v2 additive layer over frozen v1);
#                 tools_rb/r2_jersey_regions.py
"""Jersey region layer v2: synthetic documents only (no footage, no real labels)."""
import copy
import hashlib
import json
import os
import shutil
import tempfile
import unittest

from tools_rb import r2_jersey_labels as L1
from tools_rb import r2_jersey_regions as R
from tests_rb.test_r2_jersey_labels import GRID, base, reg

CID = "R2_JERSEY_MCFC_MIT_J04"          # a REQUIRED_COMPETITOR clip id (synthetic content)


def rg(back=("READABLE", 7), front=("NOT_VISIBLE", None), sl=("READABLE", 7), sr=("NOT_VISIBLE", None)):
    out = {}
    for name, (st, n) in zip(R.REGIONS, (back, front, sl, sr)):
        out[name] = {"state": st, "number": n, "reason": "MOTION_BLUR" if st == "UNREADABLE" else None, "note": ""}
    return out


def unread_jersey():
    return rg(back=("UNREADABLE", None), sl=("NOT_VISIBLE", None))


def layer(v1, v1_sha):
    seen = [g for g in GRID if v1["keyframes"][str(g)]["visibility"] in ("VISIBLE", "PARTIAL")]
    regions = {}
    for g in seen:
        regions[str(g)] = rg() if v1["keyframes"][str(g)]["number"] == "READABLE" else unread_jersey()
    comp = {"description": "same kit right", "relation": "SAME_TEAM", "jersey_number": 8, "shorts_number": None,
            "keyframes": {str(GRID[0]): {"visibility": "VISIBLE", "box": [300, 100, 360, 260], "human_uncertain": False,
                                         "regions": rg(back=("READABLE", 8), sl=("UNREADABLE", None))},
                          str(GRID[1]): {"visibility": "OUT_OF_FRAME", "box": None, "human_uncertain": False,
                                         "regions": None}}}
    return {"schema": R.SCHEMA, "clip_id": v1["clip_id"], "clip_sha256": v1["clip_sha256"], "fps": v1["fps"],
            "frames": v1["frames"], "width": v1["width"], "height": v1["height"], "annotator": "t", "tool": "t",
            "saved_utc": "x", "base_v1": {"file": f"{v1['clip_id']}.jersey.v1.FROZEN.json", "sha256": v1_sha},
            "target": {"shorts_number": 7, "regions": regions}, "competitors": {"C1": comp}, "competitors_note": "",
            "distractors": [{"keyframe": GRID[2], "box": [10, 10, 50, 40], "kind": "BOARD", "text": "9"}]}


class Base(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.mkdtemp(prefix="r2_jreg_")
        self.reg = os.path.join(self.t, "reg.json")
        with open(self.reg, "w") as f:
            json.dump(reg([CID]), f)
        self.v1dir, self.v2dir = os.path.join(self.t, "v1"), os.path.join(self.t, "v2")
        self.v1 = base(CID)
        p = os.path.join(self.t, "v1src.json")
        with open(p, "w") as f:
            json.dump(self.v1, f)
        L1.freeze(p, self.reg, self.v1dir)
        self.v1_sha = L1._sha(L1.frozen_path(self.v1dir, CID, 1))
        self.c = L1._load_registry(self.reg)[CID]

    def tearDown(self):
        for root, _, files in os.walk(self.t):
            for n in files:
                os.chmod(os.path.join(root, n), 0o644)
        shutil.rmtree(self.t)

    def lay(self):
        return layer(self.v1, self.v1_sha)

    def probs(self, lab):
        return R.validate(lab, self.v1, self.v1_sha, self.c)

    def bad(self, lab, frag):
        p = self.probs(lab)
        self.assertTrue(any(frag in x for x in p), f"expected {frag!r} in {p}")

    def write(self, lab, name="v2src.json"):
        p = os.path.join(self.t, name)
        with open(p, "w") as f:
            json.dump(lab, f)
        return p


class Validate(Base):
    def test_valid(self):
        self.assertEqual(self.probs(self.lay()), [])

    def test_bound_to_frozen_v1(self):
        lab = self.lay(); lab["base_v1"]["sha256"] = "0" * 64
        self.bad(lab, "base_v1")
        lab = self.lay(); lab["fps"] = 30.0
        self.bad(lab, "differs from frozen v1")

    def test_target_regions_cover_exactly_v1_visible(self):
        lab = self.lay(); lab["target"]["regions"][str(GRID[2])] = rg()      # GRID[2] is OUT_OF_FRAME in v1
        self.bad(lab, "exactly v1's visible keyframes")
        lab = self.lay(); del lab["target"]["regions"][str(GRID[0])]
        self.bad(lab, "exactly v1's visible keyframes")

    def test_all_four_regions_and_states(self):
        lab = self.lay(); del lab["target"]["regions"][str(GRID[0])]["SHORTS_RIGHT"]
        self.bad(lab, "regions must be exactly")
        lab = self.lay(); lab["target"]["regions"][str(GRID[0])]["SHORTS_RIGHT"]["state"] = None
        self.bad(lab, "state unlabelled")
        lab = self.lay(); lab["target"]["regions"][str(GRID[0])]["JERSEY_BACK"]["number"] = None
        self.bad(lab, "READABLE needs a number")
        lab = self.lay(); lab["target"]["regions"][str(GRID[0])]["JERSEY_FRONT"]["number"] = 7
        self.bad(lab, "number only allowed when READABLE")
        lab = self.lay(); lab["target"]["regions"][str(GRID[0])]["SHORTS_RIGHT"].update(state="UNREADABLE")
        self.bad(lab, "UNREADABLE needs a reason")
        lab = self.lay(); lab["target"]["regions"][str(GRID[0])]["SHORTS_RIGHT"].update(reason="FOLDED")
        self.bad(lab, "reason only allowed")
        lab = self.lay(); lab["target"]["regions"][str(GRID[0])]["SHORTS_RIGHT"].update(state="NO_NUMBER")
        self.assertEqual(self.probs(lab), [])

    def test_shorts_independent_of_shirt(self):
        lab = self.lay(); lab["target"]["shorts_number"] = None
        self.bad(lab, "shorts number is not set")              # shorts READABLE needs its own number
        lab = self.lay(); lab["target"]["shorts_number"] = 4
        self.bad(lab, "target shorts number is 4")             # never assumed equal to the shirt
        for g in GRID:                                          # consistent when every shorts reading says 4
            regs = lab["target"]["regions"].get(str(g))
            if regs and regs["SHORTS_LEFT"]["state"] == "READABLE":
                regs["SHORTS_LEFT"]["number"] = 4
        self.assertEqual(self.probs(lab), [])
        self.assertTrue(R.summary(lab, self.v1)["target_shorts_differs_from_jersey"])

    def test_v1_consistency_and_conflict_note(self):
        lab = self.lay(); lab["target"]["regions"][str(GRID[0])] = unread_jersey()      # v1 READABLE there
        self.bad(lab, "v1 says shirt READABLE")
        lab["target"]["regions"][str(GRID[0])]["JERSEY_BACK"]["note"] = R.CONFLICT_TAG + " blur, cannot read now"
        self.assertEqual(self.probs(lab), [])
        self.assertEqual(R.summary(lab, self.v1)["v1_conflict_notes"], 1)
        lab = self.lay(); lab["target"]["regions"][str(GRID[1])] = rg()                  # v1 UNREADABLE there
        self.bad(lab, "v1 says shirt not readable")
        lab = self.lay(); lab["target"]["regions"][str(GRID[0])]["JERSEY_BACK"]["number"] = 1
        self.bad(lab, "target jersey number is 7")

    def test_competitors(self):
        lab = self.lay(); lab["competitors"] = {}
        self.bad(lab, "needs its nearby competitors")
        lab["competitors_note"] = "no same-kit player near the target"
        self.assertEqual(self.probs(lab), [])
        lab = self.lay(); c = lab["competitors"]["C1"]
        c["keyframes"][str(GRID[0])]["regions"]["JERSEY_BACK"]["number"] = 9
        self.bad(lab, "competitor C1 jersey number is 8")
        lab = self.lay(); lab["competitors"]["C1"]["jersey_number"] = None
        self.bad(lab, "competitor C1 jersey number is not set")
        lab = self.lay(); lab["competitors"]["C1"]["keyframes"][str(GRID[0])]["box"] = None
        self.bad(lab, "needs a box")
        lab = self.lay(); lab["competitors"]["C1"]["keyframes"][str(GRID[1])]["box"] = [1, 1, 5, 5]
        self.bad(lab, "must have no box and no regions")
        lab = self.lay(); lab["competitors"]["C1"]["keyframes"]["31"] = lab["competitors"]["C1"]["keyframes"][str(GRID[1])]
        self.bad(lab, "is not a keyframe")
        lab = self.lay(); lab["competitors"]["C1"]["relation"] = "OTHER"
        self.bad(lab, "relation")
        lab = self.lay(); x = lab["competitors"]["C1"]["keyframes"][str(GRID[0])]
        x["human_uncertain"] = True
        self.bad(lab, "HUMAN_UNCERTAIN keyframes carry no region truth")
        x["regions"] = None
        self.bad(lab, "jersey_number is set but no keyframe has a READABLE jersey region")
        lab = self.lay(); lab["competitors"]["C1"]["keyframes"] = {}
        self.bad(lab, "no keyframes labelled")

    def test_distractors(self):
        lab = self.lay(); lab["distractors"][0]["kind"] = "SHIRT"
        self.bad(lab, "distractor 0")
        lab = self.lay(); lab["distractors"][0]["box"] = None
        self.bad(lab, "distractor 0")
        lab = self.lay(); lab["distractors"][0]["keyframe"] = 31
        self.bad(lab, "distractor 0")

    def test_summary_shorts_only_coverage(self):
        lab = self.lay()
        g = str(GRID[1])                    # v1 shirt UNREADABLE; shorts readable here
        lab["target"]["regions"][g]["SHORTS_RIGHT"] = {"state": "READABLE", "number": 7, "reason": None, "note": ""}
        self.assertEqual(self.probs(lab), [])
        s = R.summary(lab, self.v1)["target"]
        self.assertEqual(s["shorts_only"], 1)
        self.assertEqual(s["either"], s["jersey_readable"] + 1)


class FreezeRecord(Base):
    def test_freeze_and_record(self):
        src = self.write(self.lay())
        r = R.freeze(src, self.reg, self.v1dir, self.v2dir)
        out = R.frozen_path(self.v2dir, CID, 2)
        self.assertEqual(open(out, "rb").read(), open(src, "rb").read())
        self.assertEqual((r["version"], r["base_v1_sha256"]), (2, self.v1_sha))
        self.assertEqual(os.stat(out).st_mode & 0o222, 0)
        rec = R.dataset_record(self.v1dir, self.v2dir, self.reg)
        names = [f["file"] for f in rec["files"]]
        self.assertEqual(names, sorted([f"{CID}.jersey.v1.FROZEN.json", f"{CID}.regions.v2.FROZEN.json"]))
        want = hashlib.sha256("\n".join(f"{f['file']}  {f['sha256']}" for f in rec["files"]).encode()).hexdigest()
        self.assertEqual(rec["dataset_sha256"], want)
        self.assertEqual(rec["coverage"]["n_competitors"], 1)
        # v1 is untouched by all of this
        self.assertEqual(L1._sha(L1.frozen_path(self.v1dir, CID, 1)), self.v1_sha)

    def test_refusals(self):
        bad = self.lay(); bad["target"]["regions"][str(GRID[0])]["JERSEY_BACK"]["state"] = None
        with self.assertRaises(SystemExit):
            R.freeze(self.write(bad), self.reg, self.v1dir, self.v2dir)
        R.freeze(self.write(self.lay()), self.reg, self.v1dir, self.v2dir)
        with self.assertRaises(SystemExit):                         # new version needs a reason
            R.freeze(self.write(self.lay(), "b.json"), self.reg, self.v1dir, self.v2dir)
        r = R.freeze(self.write(self.lay(), "b.json"), self.reg, self.v1dir, self.v2dir, reason="added C2")
        self.assertEqual((r["version"], r["supersedes"]), (3, f"{CID}.regions.v2.FROZEN.json"))

    def test_record_fails_closed(self):
        with self.assertRaises(SystemExit):                         # no layer yet
            R.dataset_record(self.v1dir, self.v2dir, self.reg)
        R.freeze(self.write(self.lay()), self.reg, self.v1dir, self.v2dir)
        p = L1.frozen_path(self.v1dir, CID, 1)                       # tampered v1
        os.chmod(p, 0o644)
        d = json.load(open(p)); d["annotator"] = "x"
        with open(p, "w") as f:
            json.dump(d, f)
        with self.assertRaises(SystemExit):
            R.dataset_record(self.v1dir, self.v2dir, self.reg)

    def test_tampered_layer_fails_closed(self):
        R.freeze(self.write(self.lay()), self.reg, self.v1dir, self.v2dir)
        p = R.frozen_path(self.v2dir, CID, 2)
        os.chmod(p, 0o644)
        with open(p, "a") as f:
            f.write(" ")
        with self.assertRaises(SystemExit):
            R.dataset_record(self.v1dir, self.v2dir, self.reg)


if __name__ == "__main__":
    unittest.main()


class AuditGaps(Base):
    """Rules found untested by the test audit and the code review (2026-11-04)."""

    def kf0(self, lab):
        return lab["target"]["regions"][str(GRID[0])]

    def test_conflict_tag_exact_and_jersey_only(self):
        lab = self.lay(); self.kf0(lab)["JERSEY_BACK"].update(number=1, note="blurred")       # untagged note
        self.bad(lab, "target jersey number is 7")
        lab = self.lay(); self.kf0(lab)["JERSEY_BACK"]["number"] = 1
        self.kf0(lab)["SHORTS_LEFT"]["note"] = R.CONFLICT_TAG + " x"                            # tag on the wrong region
        self.bad(lab, "target jersey number is 7")
        lab = self.lay(); self.kf0(lab)["JERSEY_BACK"].update(number=1, note=R.CONFLICT_TAG + " v1 misread")
        self.assertEqual(self.probs(lab), [])
        lab = self.lay(); self.kf0(lab)["SHORTS_LEFT"].update(number=1, note=R.CONFLICT_TAG + " x")   # never for shorts
        self.bad(lab, "target shorts number is 7")
        lab = self.lay(); self.kf0(lab).update(unread_jersey())
        self.kf0(lab)["SHORTS_LEFT"]["note"] = R.CONFLICT_TAG + " x"
        self.bad(lab, "v1 says shirt READABLE")
        self.kf0(lab)["SHORTS_LEFT"]["note"] = ""; self.kf0(lab)["JERSEY_BACK"]["note"] = "plain note"
        self.bad(lab, "v1 says shirt READABLE")

    def test_competitor_never_gets_conflict_override(self):
        lab = self.lay(); x = lab["competitors"]["C1"]["keyframes"][str(GRID[0])]["regions"]["JERSEY_BACK"]
        x.update(number=9, note=R.CONFLICT_TAG + " two people?")
        self.bad(lab, "competitor C1 jersey number is 8")

    def test_front_counts_as_jersey(self):
        lab = self.lay(); self.kf0(lab)["JERSEY_BACK"] = {"state": "NOT_VISIBLE", "number": None, "reason": None, "note": ""}
        self.kf0(lab)["JERSEY_FRONT"] = {"state": "READABLE", "number": 7, "reason": None, "note": ""}
        self.assertEqual(self.probs(lab), [])
        lab = self.lay(); self.kf0(lab)["SHORTS_LEFT"]["state"] = "READABLE"     # shorts never satisfy v1 READABLE
        self.kf0(lab)["JERSEY_BACK"] = {"state": "NOT_VISIBLE", "number": None, "reason": None, "note": ""}
        self.bad(lab, "v1 says shirt READABLE")

    def test_reason_and_ranges(self):
        lab = self.lay(); self.kf0(lab)["SHORTS_RIGHT"].update(state="UNREADABLE", reason="FRONT_ON")   # v1 reason, not v2
        self.bad(lab, "UNREADABLE needs a reason")
        for v in (0, 100, "7", True):
            lab = self.lay(); lab["target"]["shorts_number"] = v
            self.bad(lab, "shorts_number must be 1-99")

    def test_shorts_number_needs_a_readable_shorts_region(self):
        lab = self.lay()
        for regs in lab["target"]["regions"].values():
            for r in R.SHORTS:
                regs[r] = {"state": "NO_NUMBER", "number": None, "reason": None, "note": ""}
        self.bad(lab, "target.shorts_number is set but no keyframe")
        lab["target"]["shorts_number"] = None
        self.assertEqual(self.probs(lab), [])            # no shorts number at all is fine (absence never counts)

    def test_required_competitor_clips(self):
        self.assertEqual(set(R.REQUIRED_COMPETITOR_CLIPS),
                         {"R2_JERSEY_MCFC_MIT_J04", "R2_JERSEY_MCFC_MIT_J10", "R2_JERSEY_SYO_MAR_J14"})
        lab = self.lay(); lab["competitors"] = {}; lab["competitors_note"] = "   "
        self.bad(lab, "needs its nearby competitors")
        lab = self.lay(); lab["competitors"] = []
        self.bad(lab, "competitors must be an object")

    def test_optional_competitors_elsewhere(self):
        cid = "R2_JERSEY_MCFC_MIT_J01"
        v1 = base(cid); lab = layer(v1, "s" * 64); lab["competitors"] = {}
        self.assertFalse(any("nearby competitors" in x for x in R.validate(lab, v1, "s" * 64)))

    def test_competitor_rules(self):
        k0, k1 = str(GRID[0]), str(GRID[1])
        cases = [(lambda c: c.update(description=" "), "description missing"),
                 (lambda c: c.update(jersey_number=100), "jersey_number must be 1-99"),
                 (lambda c: c.update(shorts_number="8"), "shorts_number must be 1-99"),
                 (lambda c: c["keyframes"][k0].update(human_uncertain="no"), "human_uncertain must be"),
                 (lambda c: c["keyframes"][k0].update(visibility="SEEN"), "visibility unlabelled"),
                 (lambda c: c["keyframes"][k0].update(box=[0, 0, 99999, 10]), "box invalid"),
                 (lambda c: c["keyframes"][k1].update(regions=rg()), "must have no box and no regions"),
                 (lambda c: c["keyframes"][k0].update(visibility="PARTIAL", regions=None), "regions must be exactly"),
                 (lambda c: c["keyframes"][k0]["regions"]["SHORTS_LEFT"].update(state="READABLE", number=8, reason=None)
                  or c.update(shorts_number=5), "competitor C1 shorts number is 5"),
                 (lambda c: c["keyframes"][k0]["regions"]["JERSEY_BACK"].update(state="NOT_VISIBLE", number=None),
                  "jersey_number is set but no keyframe")]
        for f, frag in cases:
            lab = self.lay(); f(lab["competitors"]["C1"])
            self.bad(lab, frag)
        lab = self.lay(); lab["competitors"]["C1"].update(shorts_number=8)          # set, but shorts never readable
        self.bad(lab, "shorts_number is set but no keyframe")
        lab = self.lay(); x = lab["competitors"]["C1"]["keyframes"][k0]          # uncertain: box optional, no regions
        x.update(human_uncertain=True, box=None, regions=None)
        lab["competitors"]["C1"]["keyframes"][str(GRID[3])] = {"visibility": "VISIBLE", "box": [300, 100, 360, 260],
                                                              "human_uncertain": False, "regions": rg(back=("READABLE", 8),
                                                                                                        sl=("NOT_VISIBLE", None))}
        lab["competitors"]["C1"]["shorts_number"] = None
        self.assertEqual(self.probs(lab), [])

    def test_distractor_and_list(self):
        lab = self.lay(); lab["distractors"][0]["keyframe"] = False          # bool is not keyframe 0
        lab["distractors"][0]["keyframe"] = False
        self.bad(lab, "distractor 0")
        lab = self.lay(); lab["distractors"] = {"x": 1}
        self.bad(lab, "distractors must be a list")

    def test_v1_binding_fields(self):
        for mut, frag in ((lambda l: l["base_v1"].update(file="other.json"), "base_v1"),
                          (lambda l: l.update(clip_sha256="b" * 64), "differs from frozen v1"),
                          (lambda l: l.update(annotator=""), "annotator missing")):
            lab = self.lay(); mut(lab)
            self.bad(lab, frag)
        v1 = copy.deepcopy(self.v1); v1["clip_rejected"] = "x"; v1["keyframes"] = {}
        self.assertTrue(any("rejected" in x for x in R.validate(self.lay(), v1, self.v1_sha, self.c)))
        v1 = copy.deepcopy(self.v1); v1["keyframes"][str(GRID[0])]["number"] = None
        self.assertTrue(any("base v1 does not validate" in x for x in R.validate(self.lay(), v1, self.v1_sha, self.c)))

    def test_uncertain_v1_keyframes_carry_no_target_regions(self):
        v1 = copy.deepcopy(self.v1)
        v1["keyframes"][str(GRID[4])].update(human_uncertain=True, number="UNREADABLE", reason="OTHER", box=None)
        lab = self.lay()
        self.assertTrue(any("exactly v1's visible keyframes" in x for x in R.validate(lab, v1, self.v1_sha, self.c)))
        del lab["target"]["regions"][str(GRID[4])]
        self.assertEqual(R.validate(lab, v1, self.v1_sha, self.c), [])


class AuditFreeze(Base):
    def test_freeze_copies_bytes_of_an_indented_source(self):
        p = os.path.join(self.t, "ind.json")
        with open(p, "w") as f:
            json.dump(self.lay(), f, indent=3)
            f.write("\n\n")
        R.freeze(p, self.reg, self.v1dir, self.v2dir)
        out = R.frozen_path(self.v2dir, CID, 2)
        self.assertEqual(open(out, "rb").read(), open(p, "rb").read())
        self.assertEqual(os.stat(out[:-len(".json")] + ".record.json").st_mode & 0o222, 0)

    def test_freeze_refusals(self):
        lab = self.lay(); lab["clip_id"] = "R2_JERSEY_NOPE"
        with self.assertRaises(SystemExit):
            R.freeze(self.write(lab), self.reg, self.v1dir, self.v2dir)
        os.makedirs(self.v2dir)
        open(R.frozen_path(self.v2dir, CID, 2)[:-len(".json")] + ".record.json", "w").close()    # stray record
        with self.assertRaises(SystemExit):
            R.freeze(self.write(self.lay()), self.reg, self.v1dir, self.v2dir)

    def test_whitespace_reason_and_extra_v1(self):
        R.freeze(self.write(self.lay()), self.reg, self.v1dir, self.v2dir)
        with self.assertRaises(SystemExit):
            R.freeze(self.write(self.lay(), "b.json"), self.reg, self.v1dir, self.v2dir, reason="   ")
        shutil.copyfile(L1.frozen_path(self.v1dir, CID, 1), L1.frozen_path(self.v1dir, CID, 2))
        with self.assertRaises(SystemExit):                                  # v1 must be the single frozen v1
            R.freeze(self.write(self.lay(), "c.json"), self.reg, self.v1dir, self.v2dir, reason="x")


class AuditRecord(unittest.TestCase):
    """dataset_record over two clips, superseded versions and tampering."""

    def setUp(self):
        self.t = tempfile.mkdtemp(prefix="r2_jrec_")
        self.cids = ["R2_JERSEY_MCFC_MIT_J04", "R2_JERSEY_MCFC_MIT_J01"]
        self.reg = os.path.join(self.t, "reg.json")
        with open(self.reg, "w") as f:
            json.dump(reg(self.cids), f)
        self.v1dir, self.v2dir = os.path.join(self.t, "v1"), os.path.join(self.t, "v2")
        for cid in self.cids:
            v1 = base(cid); p = os.path.join(self.t, cid + ".v1.json")
            with open(p, "w") as f:
                json.dump(v1, f)
            L1.freeze(p, self.reg, self.v1dir)
            sha = L1._sha(L1.frozen_path(self.v1dir, cid, 1))
            q = os.path.join(self.t, cid + ".v2.json")
            with open(q, "w") as f:
                json.dump(layer(v1, sha), f)
            R.freeze(q, self.reg, self.v1dir, self.v2dir)

    def tearDown(self):
        for root, _, files in os.walk(self.t):
            for n in files:
                os.chmod(os.path.join(root, n), 0o644)
        shutil.rmtree(self.t)

    def test_digest_sorted_and_latest_version(self):
        rec = R.dataset_record(self.v1dir, self.v2dir, self.reg)
        self.assertEqual(len(rec["files"]), 4)
        self.assertEqual([f["file"] for f in rec["files"]], sorted(f["file"] for f in rec["files"]))
        want = hashlib.sha256("\n".join(f"{f['file']}  {f['sha256']}" for f in rec["files"]).encode()).hexdigest()
        self.assertEqual(rec["dataset_sha256"], want)
        cid = self.cids[0]
        lab = json.load(open(R.frozen_path(self.v2dir, cid, 2))); lab["competitors_note"] = "v3"
        q = os.path.join(self.t, "v3.json")
        with open(q, "w") as f:
            json.dump(lab, f)
        R.freeze(q, self.reg, self.v1dir, self.v2dir, reason="note")
        rec2 = R.dataset_record(self.v1dir, self.v2dir, self.reg)
        self.assertIn(f"{cid}.regions.v3.FROZEN.json", [f["file"] for f in rec2["files"]])
        self.assertEqual([s["file"] for s in rec2["superseded"]], [f"{cid}.regions.v2.FROZEN.json"])

    def _tamper_record(self, key, value):
        rp = R.frozen_path(self.v2dir, self.cids[0], 2)[:-len(".json")] + ".record.json"
        os.chmod(rp, 0o644)
        d = json.load(open(rp)); d[key] = value
        with open(rp, "w") as f:
            json.dump(d, f)

    def test_record_mismatches_fail_closed(self):
        for key, value in (("version", 9), ("base_v1_sha256", "0" * 64), ("sha256", "0" * 64)):
            self.setUp()
            self._tamper_record(key, value)
            with self.assertRaises(SystemExit, msg=key):
                R.dataset_record(self.v1dir, self.v2dir, self.reg)

    def test_missing_record_and_tampered_superseded(self):
        rp = R.frozen_path(self.v2dir, self.cids[0], 2)[:-len(".json")] + ".record.json"
        os.chmod(rp, 0o644); os.remove(rp)
        with self.assertRaises(SystemExit):
            R.dataset_record(self.v1dir, self.v2dir, self.reg)

    def test_tampered_superseded_version(self):
        cid = self.cids[0]
        q = os.path.join(self.t, "v3.json")
        shutil.copyfile(R.frozen_path(self.v2dir, cid, 2), q)
        os.chmod(q, 0o644)
        R.freeze(q, self.reg, self.v1dir, self.v2dir, reason="same content")
        p = R.frozen_path(self.v2dir, cid, 2)
        os.chmod(p, 0o644)
        with open(p, "a") as f:
            f.write(" ")
        with self.assertRaises(SystemExit):
            R.dataset_record(self.v1dir, self.v2dir, self.reg)


class AuditFailClosed(Base):
    """Fail-closed checks that other guards normally shadow, exercised on their own."""

    def _rewrite_with_record(self, path, data):
        """Rewrite a frozen file AND its freeze record consistently (a forged but self-consistent freeze)."""
        os.chmod(path, 0o644)
        with open(path, "w") as f:
            json.dump(data, f)
        rp = path[:-len(".json")] + ".record.json"
        os.chmod(rp, 0o644)
        rec = json.load(open(rp)); rec["sha256"] = L1._sha(path)
        with open(rp, "w") as f:
            json.dump(rec, f)

    def test_v1_loader_checks_record_split_and_validity(self):
        reg_ = L1._load_registry(self.reg)
        p = L1.frozen_path(self.v1dir, CID, 1)
        os.chmod(p, 0o644)
        with open(p, "a") as f:
            f.write(" ")
        with self.assertRaises(SystemExit):                       # differs from its freeze record
            R._v1(self.v1dir, reg_, CID)
        bad = copy.deepcopy(self.v1); bad["keyframes"][str(GRID[0])]["number"] = None
        self._rewrite_with_record(p, bad)
        with self.assertRaises(SystemExit):                       # consistent record but invalid v1
            R._v1(self.v1dir, reg_, CID)

    def test_v1_loader_refuses_non_dev(self):
        r = json.load(open(self.reg)); r["clips"][0]["split"] = "EVAL"
        with open(self.reg, "w") as f:
            json.dump(r, f)
        with self.assertRaises(SystemExit):
            R._v1(self.v1dir, L1._load_registry(self.reg), CID)

    def test_record_revalidates_layer(self):
        R.freeze(self.write(self.lay()), self.reg, self.v1dir, self.v2dir)
        p = R.frozen_path(self.v2dir, CID, 2)
        bad = json.load(open(p)); bad["target"]["regions"][str(GRID[0])]["JERSEY_BACK"]["state"] = None
        self._rewrite_with_record(p, bad)
        with self.assertRaises(SystemExit):
            R.dataset_record(self.v1dir, self.v2dir, self.reg)

    def test_summary_shares_target_number(self):
        lab = self.lay()
        self.assertFalse(R.summary(lab, self.v1)["competitor_shares_target_number"])
        lab["competitors"]["C1"]["jersey_number"] = 7
        self.assertTrue(R.summary(lab, self.v1)["competitor_shares_target_number"])

# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2/JERSEY_DATASET/READER_EVAL_PREREGISTRATION.md; tools_rb/r2_jersey_reader_eval.py
"""Reader-evaluation scoring on synthetic units and a scripted reader (no footage, no model)."""
import json
import os
import shutil
import tempfile
import unittest

from tools_rb import r2_jersey_reader_eval as E

BOX = [100, 100, 200, 400]          # 100 x 300 person box; crop coordinates below are inside it
JERSEY_TXT = [40, 60, 60, 90]       # centre (50, 75)  -> cy_n 0.25 -> JERSEY
SH_L_TXT = [10, 190, 30, 210]       # centre (20, 200) -> cy_n 0.67, cx_n 0.2 -> SHORTS_LEFT
SH_R_TXT = [70, 190, 90, 210]       # cx_n 0.8 -> SHORTS_RIGHT
HEAD_TXT = [40, 5, 60, 20]          # cy_n 0.04 -> UNASSIGNED


def reg(state, n=None):
    return {"state": state, "number": n, "reason": "MOTION_BLUR" if state == "UNREADABLE" else None, "note": ""}


def truth(back=("READABLE", 7), front=("NO_NUMBER", None), sl=("NOT_VISIBLE", None), sr=("NOT_VISIBLE", None)):
    return {"JERSEY_BACK": reg(*back), "JERSEY_FRONT": reg(*front), "SHORTS_LEFT": reg(*sl), "SHORTS_RIGHT": reg(*sr)}


def unit(clip="C", person="TARGET", kf=0, fps=2.0, t=None, jn=7, sn=None, e=7, rel="TARGET", vis="VISIBLE",
         policy=False, dis=None):
    return {"unit_id": f"{clip}|{person}|{kf}", "clip_id": clip, "person": person, "relation": rel, "keyframe": kf,
            "t_s": kf / fps, "visibility": vis, "box": BOX, "crop": "x.png", "policy_abstain": policy,
            "jersey_number": jn, "shorts_number": sn, "expected_number": e, "truth": t or truth(),
            "distractor_boxes": dis or [], "source": "SYN"}


class Rules(unittest.TestCase):
    def test_valid_number(self):
        for txt, ok in (("7", 7), (" 17 ", 17), ("07", None), ("0", None), ("123", None), ("1a", None), ("", None)):
            self.assertEqual(E.valid_number(txt, 0.9, 0.5), ok, txt)
        self.assertIsNone(E.valid_number("7", 0.49, 0.5))
        self.assertEqual(E.valid_number("7", 0.5, 0.5), 7)

    def test_bands(self):
        self.assertEqual(E.assign_region(0.5, 0.10), "JERSEY")
        self.assertEqual(E.assign_region(0.5, 0.549), "JERSEY")
        self.assertEqual(E.assign_region(0.49, 0.55), "SHORTS_LEFT")
        self.assertEqual(E.assign_region(0.5, 0.79), "SHORTS_RIGHT")
        self.assertIsNone(E.assign_region(0.5, 0.80))
        self.assertIsNone(E.assign_region(0.5, 0.09))

    def test_region_readings_conflict_unassigned_distractor(self):
        u = unit(dis=[[140, 150, 170, 200]])                       # image coords covering the jersey text centre
        rr = E.region_readings(u, [["7", 0.9, JERSEY_TXT], ["9", 0.9, SH_L_TXT], ["4", 0.9, HEAD_TXT],
                                   ["x", 0.9, SH_R_TXT]], 0.5)
        self.assertIsNone(rr["JERSEY"]["number"])                   # the jersey text sat in a distractor box
        self.assertEqual(rr["counts"], {"invalid": 1, "unassigned": 1, "distractor_hits": 1, "distractor_leaks": 0})
        self.assertEqual(rr["SHORTS_LEFT"]["number"], 9)
        rr = E.region_readings(unit(), [["1", 0.9, JERSEY_TXT], ["0", 0.9, JERSEY_TXT]], 0.5)
        self.assertTrue(rr["JERSEY"]["conflict"])                   # §3.5 ruling: "10" split = conflict, abstain
        self.assertIsNone(rr["JERSEY"]["number"])
        rr = E.region_readings(unit(), [["0", 0.9, JERSEY_TXT]], 0.5)
        self.assertIsNone(rr["JERSEY"]["number"])                   # a lone invalid token is discarded
        self.assertFalse(rr["JERSEY"]["conflict"])
        rr = E.region_readings(unit(), [["1", 0.9, JERSEY_TXT], ["8", 0.9, JERSEY_TXT]], 0.5)
        self.assertTrue(rr["JERSEY"]["conflict"])                   # two numbers in one region: abstain
        self.assertIsNone(rr["JERSEY"]["number"])

    def test_unit_outcomes(self):
        u = unit(t=truth(sl=("NO_NUMBER", None), sr=("READABLE", 7)))
        rr = E.region_readings(u, [["7", 0.9, JERSEY_TXT], ["7", 0.9, SH_R_TXT], ["3", 0.9, SH_L_TXT]], 0.5)
        o = E.score_unit(u, rr)
        self.assertEqual((o["JERSEY"], o["SHORTS_RIGHT"], o["SHORTS_LEFT"], o["COMBINED"]),
                         (E.CORRECT, E.CORRECT, E.FALSE, E.WRONG))
        u = unit(t=truth(back=("UNREADABLE", None)))
        o = E.score_unit(u, E.region_readings(u, [], 0.5))
        self.assertEqual((o["JERSEY"], o["COMBINED"]), (E.OK_ABSTAIN, E.OK_ABSTAIN))
        o = E.score_unit(u, E.region_readings(u, [["5", 0.9, JERSEY_TXT]], 0.5))
        self.assertEqual(o["JERSEY"], E.FALSE)
        u = unit()
        o = E.score_unit(u, E.region_readings(u, [["1", 0.9, JERSEY_TXT]], 0.5))
        self.assertEqual((o["JERSEY"], o["COMBINED"]), (E.WRONG, E.WRONG))
        o = E.score_unit(u, E.region_readings(u, [], 0.5))
        self.assertEqual((o["JERSEY"], o["COMBINED"]), (E.ABSTAIN, E.ABSTAIN))
        self.assertEqual(o["jersey_truth_side"], ["JERSEY_BACK"])

    def test_consensus(self):
        self.assertEqual(E.consensus([(0.0, 7), (0.5, 7), (1.0, 7)], 3, 0.5)["status"], E.CONFIRMED)
        self.assertEqual(E.consensus([(0.0, 7), (0.5, 7)], 3, 0.5)["status"], E.UNKNOWN)       # one/two never decide
        self.assertEqual(E.consensus([(0.0, 7), (0.0, 7), (0.1, 7)], 3, 0.5)["status"], E.UNKNOWN)  # span
        self.assertEqual(E.consensus([(0.0, 7), (0.5, 7), (1.0, 7), (1.5, 9)], 3, 0.5)["status"], E.CONFLICT)
        self.assertEqual(E.consensus([(0.0, 7), (0.5, E.CONFLICT), (1.0, 7), (1.5, 7)], 3, 0.5)["status"], E.CONFLICT)

    def test_evidence_table(self):
        c = lambda n: {"status": E.CONFIRMED, "number": n}
        unk, conf = {"status": E.UNKNOWN, "number": None}, {"status": E.CONFLICT, "number": None}
        self.assertEqual(E.evidence(c(7), c(3), 7, "combined"), E.SUPPORT)          # shorts disagreement ignored
        self.assertEqual(E.evidence(c(8), c(7), 7, "combined"), E.VETO)             # jersey decides
        self.assertEqual(E.evidence(conf, c(7), 7, "combined"), E.UNKNOWN)
        self.assertEqual(E.evidence(unk, c(7), 7, "combined"), E.SUPPORT)           # shorts-supported
        self.assertEqual(E.evidence(unk, c(8), 7, "combined"), E.UNKNOWN)           # shorts never VETO
        self.assertEqual(E.evidence(unk, unk, 7, "combined"), E.UNKNOWN)            # absence never counts
        self.assertEqual(E.evidence(unk, c(7), 7, "jersey"), E.UNKNOWN)
        self.assertEqual(E.evidence(c(8), c(7), 7, "shorts"), E.SUPPORT)
        self.assertEqual(E.evidence(unk, c(8), 7, "shorts"), E.UNKNOWN)
        self.assertEqual(E.evidence(c(7), unk, None, "combined"), E.UNKNOWN)

    def test_pool_one_reading_per_keyframe(self):
        u = unit(t=truth(sl=("READABLE", 7), sr=("READABLE", 7)), sn=7)
        rr = E.region_readings(u, [["7", 0.9, SH_L_TXT], ["7", 0.9, SH_R_TXT]], 0.5)
        self.assertEqual(E.pool_entries([dict(u, rr=rr)], "S"), [(0.0, 7)])        # L + R = one reading
        rr = E.region_readings(u, [["7", 0.9, SH_L_TXT], ["1", 0.9, SH_R_TXT]], 0.5)
        self.assertEqual(E.pool_entries([dict(u, rr=rr)], "S"), [(0.0, E.CONFLICT)])

    def test_wilson(self):
        lo, hi = E.wilson(19, 20)
        self.assertAlmostEqual(lo, 0.7639, places=3)
        self.assertEqual(E.wilson(0, 0), (None, None))


class ScoreRun(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.mkdtemp(prefix="r2_jre_")

    def tearDown(self):
        shutil.rmtree(self.t)

    def run_score(self, units, raw):
        cfg = {"jersey_min_box_h_px": 80.0, "jersey_min_read_conf": 0.5, "identity_number_min_reads": 3,
               "jersey_confirm_span_s": 0.5, "config_hash": "x"}
        with open(os.path.join(self.t, "units.json"), "w") as f:
            json.dump({"dataset_v2_sha256": "d", "config": cfg, "units": units}, f)
        raw = dict(raw, __reader__={"name": "scripted"})
        with open(os.path.join(self.t, "raw_reads.json"), "w") as f:
            json.dump(raw, f)
        return E.score(self.t)

    def test_target_support_competitor_veto_and_false_support(self):
        U, raw = [], {}
        for k in range(4):                                  # target #7 read 4x -> SUPPORT
            u = unit(kf=k); U.append(u); raw[u["unit_id"]] = {"results": [["7", 0.9, JERSEY_TXT]]}
            c = unit(person="C1", rel="SAME_TEAM", kf=k, jn=9, e=7, t=truth(back=("READABLE", 9)))
            U.append(c); raw[c["unit_id"]] = {"results": [["9", 0.9, JERSEY_TXT]]}           # competitor -> VETO
            d = unit(person="C2", rel="SAME_TEAM", kf=k, jn=4, e=7, t=truth(back=("READABLE", 4)))
            U.append(d); raw[d["unit_id"]] = {"results": [["7", 0.9, JERSEY_TXT]]}           # misread as 7 -> FALSE SUPPORT
        U.append(unit(kf=9, policy=True))                                                    # never read
        r = self.run_score(U, raw)
        sep = r["separation"]
        self.assertEqual(sep["selected"]["SUPPORT"], 1)
        self.assertEqual(sep["competitors"]["VETO"], 1)
        self.assertEqual(sep["false_support"], ["C|C2"])
        self.assertIn("C|C2|J", sep["wrong_confirmed"])
        self.assertFalse(r["Q2"]["conditions"]["false_support_0"])
        self.assertTrue(r["Q2"]["answer"].startswith("NO"))
        self.assertEqual(r["counts"]["policy_abstain_units"], 1)
        self.assertEqual(r["tables"]["all"]["JERSEY (shirt-only)"]["CORRECT"], 8)
        self.assertEqual(r["tables"]["all"]["JERSEY (shirt-only)"]["WRONG"], 4)

    def test_absent_shorts_never_count_and_shorts_only_support(self):
        U, raw = [], {}
        for k in range(3):
            u = unit(kf=k, sn=7, t=truth(back=("UNREADABLE", None), sr=("READABLE", 7)))
            U.append(u); raw[u["unit_id"]] = {"results": [["7", 0.9, SH_R_TXT]]}
            v = unit(clip="D", kf=k, t=truth(sl=("NO_NUMBER", None), sr=("NO_NUMBER", None)))
            U.append(v); raw[v["unit_id"]] = {"results": [["7", 0.9, JERSEY_TXT]]}
        r = self.run_score(U, raw)
        P = {(p["clip_id"], p["person"]): p for p in r["persons"]}
        self.assertEqual(P[("C", "TARGET")]["evidence_combined"], E.SUPPORT)    # shorts-supported
        self.assertEqual(P[("C", "TARGET")]["evidence_jersey"], E.UNKNOWN)
        self.assertEqual(P[("D", "TARGET")]["evidence_combined"], E.SUPPORT)    # NO_NUMBER shorts: no penalty
        self.assertEqual(r["Q1_with_shorts_number"]["gain_pp"], 100.0)
        self.assertEqual(r["Q1_answer"], "YES")
        self.assertEqual(r["Q2"]["answer"], "NO (insufficient evidence)")

    def test_missing_reader_output_fails_closed(self):
        with self.assertRaises(SystemExit):
            self.run_score([unit()], {})


if __name__ == "__main__":
    unittest.main()


class ReviewFindings(ScoreRun):
    """Regression tests for the pre-run review of the scorer (2026-11-05)."""

    def test_pool_counting_one_reading_per_pool_per_keyframe(self):
        U, raw = [], {}
        for k in range(3):
            u = unit(kf=k, sn=7, t=truth(sl=("READABLE", 7), sr=("READABLE", 7)))
            U.append(u); raw[u["unit_id"]] = {"results": [["7", 0.9, JERSEY_TXT], ["7", 0.9, SH_L_TXT],
                                                           ["7", 0.9, SH_R_TXT]]}
        r = self.run_score(U, raw)
        self.assertEqual(r["readings_combined_pool"]["emitted"], 6)     # 3 keyframes x (J + S), not 9

    def test_q1_coverage_counts_any_correct_reading(self):
        U, raw = [], {}
        for k in range(3):                                   # jersey correct + a false left-shorts token
            u = unit(kf=k, sn=7, t=truth(back=("UNREADABLE", None), sr=("READABLE", 7), sl=("NO_NUMBER", None)))
            U.append(u); raw[u["unit_id"]] = {"results": [["7", 0.9, SH_R_TXT], ["3", 0.9, SH_L_TXT]]}
        r = self.run_score(U, raw)
        self.assertEqual(r["Q1_with_shorts_number"]["combined_correct_cov"], 1.0)   # >= 1 CORRECT reading
        self.assertEqual(r["tables"]["all"]["COMBINED (person-keyframe)"]["WRONG"], 3)

    def test_any_wrong_shorts_confirmation_blocks_q1(self):
        U, raw = [], {}
        for k in range(3):
            u = unit(kf=k, sn=None, t=truth(sr=("NO_NUMBER", None)))
            U.append(u); raw[u["unit_id"]] = {"results": [["7", 0.9, JERSEY_TXT], ["4", 0.9, SH_R_TXT]]}
            v = unit(clip="D", kf=k, sn=7, t=truth(back=("UNREADABLE", None), sr=("READABLE", 7)))
            U.append(v); raw[v["unit_id"]] = {"results": [["7", 0.9, SH_R_TXT]]}
        r = self.run_score(U, raw)
        self.assertIn("C|TARGET|S", r["separation"]["wrong_confirmed"])
        self.assertTrue(r["Q1_new_unsafe"])
        self.assertEqual(r["Q1_answer"], "NO")

    def test_competitor_support_never_counts_as_q1_gain(self):
        U, raw = [], {}
        for k in range(3):
            c = unit(person="C1", rel="SAME_TEAM", kf=k, jn=9, sn=7, e=7,
                     t=truth(back=("UNREADABLE", None), sr=("READABLE", 7)))
            U.append(c); raw[c["unit_id"]] = {"results": [["7", 0.9, SH_R_TXT]]}
        r = self.run_score(U, raw)
        p = r["persons"][0]
        self.assertEqual(p["evidence_combined"], E.SUPPORT)          # shares the target number via shorts
        self.assertTrue(p["shares_target_shorts"])
        self.assertEqual(r["separation"]["shares_target_shorts"], ["C|C1"])
        self.assertTrue(p["established_combined"])                   # correct CONFIRMED of its own shorts number

    def test_distractor_leak_check_and_v1_note_reporting(self):
        t = truth(); t["JERSEY_BACK"]["note"] = "V1_CONFLICT: x"
        u = unit(t=t, dis=[[0, 0, 5, 5]])
        r = self.run_score([u], {u["unit_id"]: {"results": [["7", 0.9, JERSEY_TXT]]}})
        self.assertEqual(r["v1_conflict_note_units"], ["C|TARGET|0"])
        self.assertTrue(r["Q2"]["conditions"]["distractor_readings_discarded"])

    def test_results_md_and_run_record(self):
        u = unit()
        self.run_score([u], {u["unit_id"]: {"results": [["7", 0.9, JERSEY_TXT]]}})
        self.assertTrue(open(os.path.join(self.t, "RESULTS.md")).read().startswith("# Jersey reader evaluation"))
        rec = E.run_record(self.t, os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ["x"])
        self.assertEqual(len(rec["results_sha256"]), 64)
        self.assertIn("results.json", open(os.path.join(self.t, "SHA256SUMS.txt")).read())

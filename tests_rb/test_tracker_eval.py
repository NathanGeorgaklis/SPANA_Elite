# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R1-H instruction (Nathan, 2026-10-26); R0_TRACKER_ACCEPTANCE_TARGETS.md Part B
"""Tests for tools_rb/tracker_eval.py (R1-H evaluation against frozen human truth).

Unit level: hand-built tracker documents and truth, one failure class per test, so a scorer that silently misses a
false confident identity, a swap, a wrong reacquisition or a trusted hidden frame fails here. Integration level: the
frozen GOLD labels v1 and the committed R1-G GOLD output (fail-closed checks)."""
import copy
import json
import os
import shutil
import tempfile
import unittest

from tools_rb import tracker_eval as E

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD_LABELS = os.path.join(ROOT, "R1", "gold_label_sheet", "GOLD_HUMAN_LABELS_v1.csv")
GOLD_FREEZE = os.path.join(ROOT, "R1", "gold_label_sheet", "GOLD_HUMAN_LABELS_v1.FROZEN.json")
GOLD_FIXTURE = os.path.join(ROOT, "fixtures", "GOLD_DUEL_01", "GOLD_DUEL_01_BENCHMARK_FIXTURE.json")
GOLD_RESULTS = os.path.join(ROOT, "R1", "R1G", "gold_pipeline", "run_a_mo_tracking_results.json")

A = (100.0, 100.0, 140.0, 200.0)        # the target
B = (300.0, 100.0, 340.0, 200.0)        # another person
FPS = 10.0


def rec(trusted, state="TRACKING", box=A):
    observed = state == "TRACKING"
    return {"selected_player": True, "observed_position_box": list(box), "state": "ACTIVE" if observed else "OCCLUDED",
            "v2": {"identity_trusted": trusted, "rb_state": state, "bbox_is_observed": observed}}


def doc(records):
    return {"summary": {"fps": FPS},
            "frame_log": [{"frame": i + 1, "tracks": [r]} for i, r in enumerate(records)]}


def tf(f, vis=E.VISIBLE, res=True, merged=False, target=A, others=None, clean=False):
    return E.TruthFrame(rel_frame=f, visibility=vis, identity_resolvable=res, merged=merged,
                        target_box=target, other_boxes={"P2": B} if others is None else others,
                        clean_reappearance=clean)


class ClaimAuthority(unittest.TestCase):
    def test_identity_trusted_is_the_authority(self):
        r = rec(False)                               # legacy ACTIVE, observed, but untrusted
        self.assertEqual(E.claim(r), "OBSERVED_UNVERIFIED")
        self.assertEqual(E.claim(rec(True)), "CONFIDENT")
        for st in ("OCCLUDED", "AMBIGUOUS", "REACQUIRING", "LOST"):
            self.assertEqual(E.claim(rec(False, st)), st)


class Priorities(unittest.TestCase):
    def test_clean_run_scores_zero_errors_and_full_coverage(self):
        r = E.evaluate(doc([rec(True)] * 5), [tf(f) for f in range(1, 6)])
        self.assertEqual(r["priority_1_false_confident_identity"]["count"], 0)
        self.assertEqual(r["priority_1_false_confident_identity"]["status"], "COMPLETE")
        self.assertEqual(r["priority_2_identity_swaps"]["count"], 0)
        self.assertEqual(r["priority_3_wrong_reacquisitions"]["count"], 0)
        self.assertEqual(r["priority_4_trusted_on_hidden_merged_unresolvable"]["count"], 0)
        self.assertEqual(r["coverage_and_abstention"]["trusted_coverage_of_available"], 1.0)
        self.assertEqual(r["box_quality"]["mean_iou_confident"], 1.0)

    def test_trusted_on_hidden_merged_or_unresolvable_is_counted(self):
        truth = [tf(1), tf(2, vis=E.HIDDEN, target=None), tf(3, merged=True), tf(4, res=False),
                 tf(5, vis=E.OUT_OF_FRAME, target=None)]
        r = E.evaluate(doc([rec(True)] * 5), truth)
        self.assertEqual(r["priority_4_trusted_on_hidden_merged_unresolvable"]["frames"], [2, 3, 4, 5])
        self.assertEqual(r["priority_1_false_confident_identity"]["frames"], [2, 3, 4, 5])
        self.assertEqual(r["coverage_and_abstention"]["uncertainty_when_unavailable"], 0.0)

    def test_trusted_on_the_wrong_box_is_a_false_confident_identity_and_a_swap(self):
        recs = [rec(True), rec(True, box=B), rec(True, box=B), rec(True), rec(True, box=B)]
        r = E.evaluate(doc(recs), [tf(f) for f in range(1, 6)])
        fci = r["priority_1_false_confident_identity"]
        self.assertEqual(fci["frames"], [2, 3, 5])
        self.assertEqual(r["priority_2_identity_swaps"]["count"], 2)       # two maximal runs: 2-3 and 5
        self.assertEqual(r["priority_2_identity_swaps"]["runs"], [(2, 3), (5, 5)])

    def test_partial_overlap_between_thresholds_is_neither_fci_nor_swap(self):
        half = (110.0, 100.0, 150.0, 200.0)          # IoU with A = 30/50 = 0.6 ... shifted further below
        shifted = (125.0, 100.0, 165.0, 200.0)       # IoU with A = 15/65 ≈ 0.23 -> FCI; with B = 0 -> no swap
        r = E.evaluate(doc([rec(True, box=half), rec(True, box=shifted)]), [tf(1), tf(2)])
        self.assertEqual(r["priority_1_false_confident_identity"]["frames"], [2])
        self.assertEqual(r["priority_2_identity_swaps"]["count"], 0)

    def test_untrusted_frames_never_count_as_errors(self):
        recs = [rec(False, box=B), rec(False, "AMBIGUOUS", box=B), rec(False, "OCCLUDED", box=B)]
        truth = [tf(1), tf(2, vis=E.HIDDEN, target=None), tf(3, merged=True)]
        r = E.evaluate(doc(recs), truth)
        self.assertEqual(r["priority_1_false_confident_identity"]["count"], 0)
        self.assertEqual(r["priority_2_identity_swaps"]["count"], 0)
        self.assertEqual(r["priority_4_trusted_on_hidden_merged_unresolvable"]["count"], 0)
        self.assertEqual(r["coverage_and_abstention"]["claims_on_available"], {"OBSERVED_UNVERIFIED": 1})

    def test_missing_box_truth_is_unknown_not_zero(self):
        truth = [tf(f, target=None, others=None) for f in range(1, 4)]
        truth = [E.TruthFrame(rel_frame=t.rel_frame, visibility=t.visibility, identity_resolvable=True,
                              merged=False) for t in truth]
        r = E.evaluate(doc([rec(True)] * 3), truth)
        self.assertEqual(r["priority_1_false_confident_identity"]["status"], "PARTIAL_NO_BOX_TRUTH")
        self.assertEqual(r["priority_1_false_confident_identity"]["confident_frames_without_box_truth"], [1, 2, 3])
        self.assertEqual(r["priority_2_identity_swaps"]["count"], E.UNKNOWN)
        self.assertEqual(r["box_quality"]["mean_iou_confident"], E.UNKNOWN)


class Reacquisition(unittest.TestCase):
    def _truth(self, n, gap):
        return [tf(f, vis=E.HIDDEN, target=None) if f in gap else tf(f, clean=f == max(gap) + 1)
                for f in range(1, n + 1)]

    def test_correct_reacquisition_and_its_delay_in_seconds(self):
        gap = range(3, 6)
        recs = [rec(True), rec(True)] + [rec(False, "OCCLUDED")] * 3 + [rec(False, "REACQUIRING")] * 2 + \
            [rec(True)] * 3
        r = E.evaluate(doc(recs), self._truth(10, gap))
        (item,) = r["reacquisition"]
        self.assertEqual(item["outcome"], "CORRECT")
        self.assertEqual(item["target_available_from"], 6)
        self.assertEqual(item["first_confident_frame"], 8)
        self.assertAlmostEqual(item["delay_s"], 0.2)          # 2 frames at 10 fps
        self.assertEqual(r["priority_3_wrong_reacquisitions"]["count"], 0)

    def test_first_trusted_frame_on_another_person_is_a_wrong_reacquisition(self):
        gap = range(3, 6)
        recs = [rec(True), rec(True)] + [rec(False, "OCCLUDED")] * 3 + [rec(True, box=B)] + [rec(True)] * 4
        r = E.evaluate(doc(recs), self._truth(10, gap))
        self.assertEqual(r["priority_3_wrong_reacquisitions"]["count"], 1)
        self.assertEqual(r["reacquisition"][0]["outcome"], "WRONG")

    def test_trusted_inside_the_gap_is_wrong_reacquisition_and_priority_4(self):
        gap = range(3, 6)
        recs = [rec(True), rec(True), rec(False, "OCCLUDED"), rec(False, "OCCLUDED"), rec(True)] + [rec(True)] * 5
        r = E.evaluate(doc(recs), self._truth(10, gap))
        self.assertEqual(r["priority_4_trusted_on_hidden_merged_unresolvable"]["frames"], [5])
        # the gap's own trusted frame is reported under priority 4; the first trusted frame AFTER the gap is correct
        self.assertEqual(r["reacquisition"][0]["outcome"], "CORRECT")

    def test_not_reacquired_is_censored_and_missed_only_when_long_enough(self):
        gap = range(3, 6)
        short = [rec(True), rec(True)] + [rec(False, "OCCLUDED")] * 3 + [rec(False, "REACQUIRING")] * 5
        r = E.evaluate(doc(short), self._truth(10, gap))
        item = r["reacquisition"][0]
        self.assertEqual(item["outcome"], "NOT_REACQUIRED_IN_WINDOW")
        self.assertAlmostEqual(item["delay_s_at_least"], 0.5)
        self.assertEqual(item["missed_reacquisition"], E.UNKNOWN)     # 0.5 s < 2 s: window cannot tell
        n = 5 + int(E.MISSED_REACQ_S * FPS)
        long = [rec(True), rec(True)] + [rec(False, "OCCLUDED")] * 3 + [rec(False, "LOST")] * (n - 5)
        r = E.evaluate(doc(long), self._truth(n, gap))
        self.assertIs(r["reacquisition"][0]["missed_reacquisition"], True)
        self.assertEqual(r["priority_3_wrong_reacquisitions"]["count"], 0)    # missing is not wrong

    def test_reacquisition_without_box_truth_is_unverified_not_correct(self):
        truth = [E.TruthFrame(rel_frame=f, visibility=E.HIDDEN if f == 2 else E.VISIBLE, identity_resolvable=True,
                              merged=False) for f in range(1, 5)]
        r = E.evaluate(doc([rec(True), rec(False, "OCCLUDED"), rec(True), rec(True)]), truth)
        self.assertEqual(r["reacquisition"][0]["outcome"], "UNVERIFIED_NO_BOX_TRUTH")


class Refusals(unittest.TestCase):
    def test_label_frames_outside_the_run_are_refused(self):
        with self.assertRaises(E.EvalRefused):
            E.evaluate(doc([rec(True)] * 2), [tf(1), tf(3)])

    def test_two_target_records_in_a_frame_are_refused(self):
        d = doc([rec(True)])
        d["frame_log"][0]["tracks"].append(rec(True, box=B))
        with self.assertRaises(E.EvalRefused):
            E.evaluate(d, [tf(1)])

    def test_unknown_label_values_and_formats_are_refused(self):
        row = {"frame": "10", "t23_visibility": "blurry", "t23_identity_resolvable": "yes",
               "merged_or_hidden": "none", "first_clean_reappearance": ""}
        with self.assertRaises(E.EvalRefused):
            E.ADAPTERS["gold_v1"]([row], 0)
        with self.assertRaises(E.EvalRefused):
            E.read_truth(GOLD_LABELS, "no_such_format", 180)


class GoldLabelsV1(unittest.TestCase):
    """The frozen GOLD labels v1 through the adapter (identity / visibility only, no boxes)."""

    def test_adapter_classes(self):
        truth = E.read_truth(GOLD_LABELS, "gold_v1", 180)
        self.assertEqual([truth[0].rel_frame, truth[-1].rel_frame], [70, 110])
        self.assertEqual(len(truth), 41)
        unavailable = [t.rel_frame for t in truth if not t.available]
        self.assertEqual(len(unavailable), 11)                  # 5 hidden / unresolvable + 6 merged
        self.assertTrue(all(t.target_box is None and t.other_boxes is None for t in truth))
        self.assertEqual([t.rel_frame for t in truth if t.clean_reappearance], [93])


@unittest.skipUnless(os.path.exists(GOLD_RESULTS), "committed R1-G GOLD output not present")
class FailClosedMain(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="r1h_eval_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _main(self, results=GOLD_RESULTS, fixture=GOLD_FIXTURE, labels=GOLD_LABELS, freeze=GOLD_FREEZE):
        return E.main(results, fixture, labels, freeze, "gold_v1", os.path.join(self.tmp, "out.json"))

    def test_committed_gold_output_scores_and_records_provenance(self):
        r = self._main()
        self.assertEqual(r["priority_4_trusted_on_hidden_merged_unresolvable"]["count"], 0)
        self.assertEqual(r["priority_1_false_confident_identity"]["status"], "PARTIAL_NO_BOX_TRUTH")
        self.assertFalse(r["events_evaluated"])
        self.assertEqual(len(r["results_git_commit"]), 40)
        with self.assertRaises(FileExistsError):                # never overwrites a report
            self._main()

    def test_tampered_labels_are_refused(self):
        bad = os.path.join(self.tmp, "labels.csv")
        with open(GOLD_LABELS) as f:
            text = f.read()
        with open(bad, "w") as f:
            f.write(text.replace("hidden", "visible", 1))
        with self.assertRaises(E.EvalRefused):
            self._main(labels=bad)

    def test_results_from_another_video_or_seed_are_refused(self):
        with open(GOLD_FIXTURE) as f:
            fx = json.load(f)
        for key, value in (("video_sha256", "0" * 64), ("seed_frame", fx["seed_frame"] + 1),
                           ("seed_bbox", [v + 5.0 for v in fx["seed_bbox"]])):
            with self.subTest(key=key):
                other = copy.deepcopy(fx)
                other[key] = value
                p = os.path.join(self.tmp, f"fx_{key}.json")
                with open(p, "w") as f:
                    json.dump(other, f)
                with self.assertRaises(E.EvalRefused):
                    self._main(fixture=p)

    def test_contract_violation_is_refused(self):
        with open(GOLD_RESULTS, encoding="ascii") as f:
            d = json.load(f)
        d["frame_log"][0]["tracks"][0]["v2"]["identity_trusted"] = "yes"
        p = os.path.join(self.tmp, "bad.json")
        with open(p, "w", encoding="ascii") as f:
            json.dump(d, f)
        with self.assertRaises(E.EvalRefused):
            self._main(results=p)


def truth_from_synthetic(e):
    """Ground truth from the synthetic harness: the target's own ideal detection box, every other person's box, and
    merged / hidden status (a frame where the target has no own detection counts as not available)."""
    out = []
    for f in range(1, e.n + 1):
        own = [d.box for d in e.dets[f] if d.owners == ("TARGET",)]
        merged = any("TARGET" in d.owners and len(d.owners) > 1 for d in e.dets[f])
        others = {d.owners[0]: tuple(d.box) for d in e.dets[f] if len(d.owners) == 1 and d.owners != ("TARGET",)}
        out.append(E.TruthFrame(rel_frame=f, visibility=E.VISIBLE if own else E.HIDDEN, identity_resolvable=True,
                                merged=merged and not own, target_box=tuple(own[0]) if own else None,
                                other_boxes=others))
    return out


class SyntheticIntegration(unittest.TestCase):
    """Full pipeline runs (tests_rb/test_pipeline_e2e scenarios) scored against the harness's complete box truth."""
    SCENARIOS = ("same_kit_crossing_merged", "opponent_crossing_hidden", "wrong_reacquisition", "hard_cut_impostor",
                 "leave_and_return", "twin_appears")

    @classmethod
    def setUpClass(cls):
        from tests_rb import test_pipeline_e2e as P
        cls.P = P
        P.setUpModule()

    @classmethod
    def tearDownClass(cls):
        cls.P.tearDownModule()

    def test_pipeline_scores_zero_on_every_priority_with_complete_truth(self):
        for name in self.SCENARIOS:
            with self.subTest(scenario=name):
                e = self.P.result(name)
                r = E.evaluate(e.doc, truth_from_synthetic(e))
                fci = r["priority_1_false_confident_identity"]
                self.assertEqual(fci["status"], "COMPLETE")
                self.assertEqual(fci["count"], 0)
                self.assertEqual(r["priority_2_identity_swaps"]["count"], 0)
                self.assertEqual(r["priority_3_wrong_reacquisitions"]["count"], 0)
                self.assertEqual(r["priority_4_trusted_on_hidden_merged_unresolvable"]["count"], 0)
                self.assertEqual(r["confident_not_grounded_violations"], [])
                if r["box_quality"]["mean_iou_confident"] != E.UNKNOWN:
                    self.assertGreaterEqual(r["box_quality"]["mean_iou_confident"], 0.95)

    def test_scorer_catches_a_swap_planted_into_a_real_pipeline_output(self):
        e = self.P.result("same_kit_crossing")
        truth = truth_from_synthetic(e)
        d = copy.deepcopy(e.doc)
        planted = []
        for t in truth:
            fr = d["frame_log"][t.rel_frame - 1]
            tgt = [x for x in fr["tracks"] if x["selected_player"]][0]
            if tgt["v2"]["identity_trusted"] and t.other_boxes:
                tgt["observed_position_box"] = list(next(iter(t.other_boxes.values())))
                planted.append(t.rel_frame)
            if len(planted) == 3:
                break
        self.assertEqual(len(planted), 3)
        r = E.evaluate(d, truth)
        self.assertEqual(r["priority_1_false_confident_identity"]["frames"], planted)
        self.assertGreaterEqual(r["priority_2_identity_swaps"]["count"], 1)


if __name__ == "__main__":
    unittest.main()

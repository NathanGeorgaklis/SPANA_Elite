# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2 independent test audit (2026-10-26): mutants that survived on d8170c9 and missing edge cases
"""R2 test-audit regressions: pinned protocol (split table, exclusions, edges), exact event padding per event type,
validator gaps, r2_v1 adapter / reacquisition judging, freeze atomicity, and annotator UI behaviours."""
import json
import os
import shutil
import tempfile
import unittest

from tools_rb import r2_labels as L
from tools_rb import r2_protocol as P
from tools_rb import tracker_eval as E
from tests_rb.test_r2_labels import BOX, CLIP, OTHER, base, frame


class PinnedProtocol(unittest.TestCase):
    def test_split_table_is_pinned(self):
        ev = {s: [r["block"] for r in P.block_table() if r["source"] == s and r["split"] == "EVAL"] for s in P.SOURCES}
        self.assertEqual(ev["SYO_MAR"], [0, 2, 3, 7, 8, 9, 11, 16, 17, 20, 21, 23, 24, 25, 27])
        self.assertEqual(ev["MCFC_MIT"], [0, 1, 2, 3, 5, 7, 8, 10, 11, 15, 16, 17, 19, 20, 25, 26, 28, 29])
        self.assertEqual((P.BLOCK_S, P.SPLIT_SALT), (300.0, "SPANA-R2-split-v1"))

    def test_exclusions_are_pinned(self):
        self.assertEqual(sorted(w for (_, _, _, w) in P.EXCLUDED_WINDOWS), sorted([
            "GENERAL_BENCHMARK_V2 MCFC/MIT 2400 s", "GENERAL_BENCHMARK_V2 MCFC/MIT 4200 s",
            "GENERAL_BENCHMARK_V2 SYO/MAR 2400 s", "GENERAL_BENCHMARK_V2 SYO/MAR 6600 s",
            "GOLD_DUEL_01 (development clip; never R2 evaluation truth)", "Task 3R R01", "Task 3R R02",
            "Task 3R R03", "Task 3R R04", "Task 3R R05", "Task 3R R10"]))
        self.assertEqual((P.EXCLUSION_BUFFER_S, P.GB_V2_ASSUMED_LENGTH_S), (60.0, 120.0))
        gold = [(a, b) for (s, a, b, w) in P.EXCLUDED_WINDOWS if w.startswith("GOLD")]
        self.assertEqual(gold, [(3036.0, 3048.0)])

    def test_buffer_edges_on_both_sides_of_every_window(self):
        for src, a, b, why in P.EXCLUDED_WINDOWS:
            with self.subTest(why=why):
                lo, hi = a - P.EXCLUSION_BUFFER_S, b + P.EXCLUSION_BUFFER_S
                self.assertEqual(P.excluded_overlaps(src, lo - 5.0, lo), [] if not self._other(src, lo - 5.0, lo, why)
                                 else P.excluded_overlaps(src, lo - 5.0, lo))
                self.assertIn(why, P.excluded_overlaps(src, lo - 5.0 + 0.01, lo + 0.01))
                self.assertIn(why, P.excluded_overlaps(src, hi - 0.01, hi + 5.0 - 0.01))
                self.assertNotIn(why, P.excluded_overlaps(src, hi, hi + 5.0))

    @staticmethod
    def _other(src, a, b, why):
        return [w for w in P.excluded_overlaps(src, a, b) if w != why]

    def test_clip_length_and_block_edges(self):
        self.assertEqual(P.check_window("SYO_MAR", 900.0, 905.0), P.split_of_block("SYO_MAR", 3))      # exactly 5 s
        self.assertEqual(P.check_window("SYO_MAR", 905.0, 920.0), P.split_of_block("SYO_MAR", 3))      # exactly 15 s
        for a, b in ((905.0, 920.01), (905.0, 909.99)):
            with self.assertRaises(ValueError):
                P.check_window("SYO_MAR", a, b)
        self.assertEqual(P.check_window("SYO_MAR", 1190.0, 1200.0), P.split_of_block("SYO_MAR", 3))   # ends on edge


def _lab60(event_at=30, **kw):
    """60-frame clip at 10 fps (step 2, pad 5), seed 2, every frame boxed."""
    lab = base()
    n = 60
    lab.update(frames=n, per_frame=[frame() for _ in range(n)], target_boxes={str(i): list(BOX) for i in range(n)})
    lab["people"]["P1"]["boxes"] = {str(i): list(OTHER) for i in range(n)}
    return lab


class EventPadding(unittest.TestCase):
    """Required box frames: exactly e-pad .. e+pad around each event type (pad = 5 at 10 fps), never one more."""
    CASES = {
        "contact": {30: frame(contact=["P1"])},
        "occlusion": {30: frame("PARTIAL", occ="PARTIAL_BY_OTHER")},
        "merged": {30: frame(merged=True, contact=["P1"])},
        "camera_cut": {30: frame(cut=True)},
        "visibility_change": {30: frame("PARTIAL")},                 # changes at 30 and back at 31
        "resolvable_change": {30: frame(res="NO")},
        "out_of_frame": {30: frame("OUT_OF_FRAME")},
    }

    def test_padding_is_exact_for_every_event_type(self):
        for name, change in self.CASES.items():
            with self.subTest(event=name):
                lab = _lab60()
                for i, f in change.items():
                    lab["per_frame"][i] = f
                    if f["visibility"] in ("HIDDEN", "OUT_OF_FRAME"):
                        lab["target_boxes"].pop(str(i), None)
                req = set(L.box_frames(lab))
                last_event = 31 if name in ("visibility_change", "resolvable_change", "out_of_frame") else 30
                self.assertIn(25, req)
                self.assertIn(last_event + 5 if (last_event + 5) % 2 else last_event + 4, req)
                self.assertNotIn(23, req)
                self.assertNotIn(last_event + 6 if (last_event + 6) % 2 else last_event + 7, req)

    def test_step_rounding_is_half_even_like_the_page(self):
        lab = _lab60()
        lab["fps"] = 12.5                                            # 12.5 / 5 = 2.5 -> 2 (half to even)
        self.assertEqual(L.box_frames(lab)[:3], [2, 4, 6])
        lab["fps"] = 25.0                                            # pad 0.5 * 25 = 12.5 -> 12
        self.assertEqual(L.pad_frames(25.0), 12)
        lab["fps"] = 29.97                                           # 29.97 / 5 = 5.994 -> 6 (round, not truncate)
        self.assertEqual(L.box_frames(lab)[:3], [2, 8, 14])


class ValidatorGaps(unittest.TestCase):
    def _errs(self, mutate):
        lab = base()
        mutate(lab)
        return L.validate(lab, CLIP)

    def test_each_gap_is_refused(self):
        cases = {
            "HIDDEN needs a FULL_* occlusion": lambda l: (l["per_frame"].__setitem__(9, frame("HIDDEN", occ="PARTIAL_BY_OTHER")),
                                                         l["target_boxes"].pop("9")),
            "HIDDEN needs a FULL_* occlusion ": lambda l: (l["per_frame"].__setitem__(9, frame("HIDDEN", occ="NONE")),
                                                          l["target_boxes"].pop("9")),
            "OUT_OF_FRAME cannot": lambda l: (l["per_frame"].__setitem__(9, frame("OUT_OF_FRAME", contact=["P1"])),
                                              l["target_boxes"].pop("9")),
            "seed frame must be": lambda l: (l["per_frame"].__setitem__(2, frame(contact=["P1"])),
                                             l["people"]["P1"]["boxes"].update({str(i): list(OTHER) for i in range(30)})),
            "relation": lambda l: l["people"]["P1"].update(relation="FRIEND"),
            "fps": lambda l: l.update(fps=25.0),
            "target not cleanly identifiable": lambda l: (l["per_frame"].__setitem__(5, frame(res="NO")),
                                                          l["first_clean_reappearance"].extend([5, 6])),
            "duplicates": lambda l: (l["per_frame"].__setitem__(9, frame("HIDDEN", occ="FULL_BY_OTHER")),
                                     l["target_boxes"].pop("9"), l["first_clean_reappearance"].extend([10, 10])),
            "target box at 05 invalid": lambda l: l["target_boxes"].update({"05": list(BOX)}),
            "bad box at '07'": lambda l: l["people"]["P1"]["boxes"].update({"07": list(OTHER)}),
        }
        for msg, mutate in cases.items():
            with self.subTest(msg=msg):
                errs = self._errs(mutate)
                self.assertTrue(any(msg.strip() in e for e in errs), errs[:4])


def _doc(boxes, trusted=None):
    trusted = trusted or [True] * len(boxes)
    recs = [{"selected_player": True, "observed_position_box": list(b),
             "v2": {"identity_trusted": t, "rb_state": "TRACKING" if t else "AMBIGUOUS", "bbox_is_observed": t,
                    "observation_source": "DETECTION" if t else "HELD", "bbox_is_held": not t,
                    "bbox_is_predicted": False}} for b, t in zip(boxes, trusted)]
    return {"summary": {"fps": 10.0}, "frame_log": [{"frame": i + 1, "tracks": [r]} for i, r in enumerate(recs)]}


class AdapterAndReacquisition(unittest.TestCase):
    def _truth(self, lab):
        return E.parse_truth(json.dumps(lab).encode(), "r2_v1", lab["seed"]["frame"], float(lab["fps"]))

    def test_trusted_on_a_merged_frame_is_priority_4(self):
        lab = base()
        lab["per_frame"][10] = frame(merged=True, contact=["P1"])
        lab["people"]["P1"]["boxes"] = {str(i): list(OTHER) for i in range(30)}
        lab["first_clean_reappearance"] = [11]
        r = E.evaluate(_doc([BOX] * 27), self._truth(lab))
        self.assertEqual(r["priority_4_trusted_on_hidden_merged_unresolvable"]["frames"], [8])

    def test_human_clean_reappearance_is_carried_into_the_report(self):
        lab = base()
        lab["per_frame"][9] = frame("HIDDEN", occ="FULL_BY_OTHER")
        del lab["target_boxes"]["9"]
        lab["first_clean_reappearance"] = [10]
        tr = [True] * 6 + [False] + [True] * 20                     # rel 7 = clip frame 9 untrusted
        r = E.evaluate(_doc([BOX] * 27, tr), self._truth(lab))
        self.assertEqual(r["reacquisition"][0]["human_clean_reappearance"], 8)

    def test_fps_mismatch_is_refused(self):
        with self.assertRaises(E.EvalRefused):
            E.parse_truth(json.dumps(base()).encode(), "r2_v1", 2, 25.0)

    def test_an_extra_drawn_box_off_the_grid_is_box_checked(self):
        lab = base()
        lab["target_boxes"] = {str(i): list(BOX) for i in L.box_frames(lab)}
        lab["target_boxes"]["5"] = list(BOX)                       # off the 5 Hz grid, drawn anyway
        t = {x.rel_frame: x for x in self._truth(lab)}
        self.assertTrue(t[3].box_annotated)
        self.assertFalse(t[5].box_annotated)

    def test_uncertain_available_frame_is_excluded_from_coverage(self):
        lab = base()
        lab["per_frame"][9] = frame(unc=True)
        lab["first_clean_reappearance"] = [10]
        r = E.evaluate(_doc([BOX] * 27), self._truth(lab))
        self.assertEqual(r["coverage_and_abstention"]["available_frames"], 26)

    def _gap_truth(self, first_trusted_box_annotated, judge_uncertain=False):
        tf = []
        for i in range(1, 13):
            vis = E.HIDDEN if i in (3, 4) else E.VISIBLE
            tf.append(E.TruthFrame(rel_frame=i, visibility=vis, identity_resolvable=True, merged=False,
                                   target_box=None if vis == E.HIDDEN else tuple(BOX), other_boxes={},
                                   box_annotated=(i != 6 or first_trusted_box_annotated),
                                   uncertain=(i == 7 and judge_uncertain)))
        return tf

    def test_reacquisition_judged_on_the_next_grid_frame_of_the_trusted_run(self):
        tr = [True, True, False, False, False, True, True, True, True, True, True, True]
        r = E.evaluate(_doc([BOX] * 12, tr), self._gap_truth(False))
        item = r["reacquisition"][0]
        self.assertEqual((item["first_confident_frame"], item["judged_on_frame"], item["outcome"]), (6, 7, "CORRECT"))
        boxes = [BOX] * 12
        boxes[6] = OTHER                                             # rel 7 = the judged grid frame
        r = E.evaluate(_doc(boxes, tr), self._gap_truth(False))
        self.assertEqual(r["reacquisition"][0]["outcome"], "WRONG")

    def test_uncertain_judge_frame_is_skipped_never_judged(self):
        tr = [True, True, False, False, False, True, True, True, True, True, True, True]
        r = E.evaluate(_doc([BOX] * 12, tr), self._gap_truth(False, judge_uncertain=True))
        item = r["reacquisition"][0]
        self.assertEqual((item["judged_on_frame"], item["outcome"]), (8, "CORRECT"))


class FreezeAtomicity(unittest.TestCase):
    def test_a_stale_fixture_blocks_the_freeze_before_anything_is_written(self):
        tmp = tempfile.mkdtemp(prefix="r2_atomic_")
        try:
            lab_p, clip_p, out = (os.path.join(tmp, x) for x in ("l.json", "c.json", "out"))
            with open(lab_p, "w") as f:
                json.dump(base(), f)
            with open(clip_p, "w") as f:
                json.dump(CLIP, f)
            os.makedirs(out)
            with open(os.path.join(out, "R2_TEST.fixture.json"), "w") as f:
                f.write("{}")
            with self.assertRaises(L.LabelError):
                L.freeze(lab_p, clip_p, out)
            self.assertEqual(os.listdir(out), ["R2_TEST.fixture.json"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

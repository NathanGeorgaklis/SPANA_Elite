# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R1-G instruction (Nathan, 2026-10-25): full Pass-1 tracker end to end; output matches the approved
#                 R1 contract; zero silent swaps across synthetic identity scenarios; wrong reacquisition rejected;
#                 camera cuts reset continuity; the target never competes with itself; uncontaminated bank; no
#                 hidden/merged frame reported as trusted TRACKING. R1/R1_DESIGN_DECISIONS.md D05-D07, D16, D19
"""R1-G synthetic END-TO-END tests: real video files through the production pipeline (run_rb.run).

Each scenario is rendered to a lossless video (see _synth_pipeline), run once, and checked for the invariants that
must hold in EVERY scenario (UniversalChecks) plus its own expectations. Names are neutral (TARGET / PERSON_A /
TWIN); nothing here is benchmark knowledge.
"""
import os
import shutil
import tempfile
import unittest
from collections import Counter

from player_tracker_rb import output_schema as S
from player_tracker_rb.config import TrackerConfig
from tests_rb._synth_pipeline import PH, PW, ClipFrame, box, clip, gait_walk, iou, run_clip, walk
from tests_rb._synth_scene import SceneFrame, covered, make_people

KIT_A = (200, 60, 40)
KIT_B = (40, 40, 210)
SAME_KIT_COS = 0.75
CFG = TrackerConfig.default()
_TMP = None
_CACHE = {}


def setUpModule():
    global _TMP
    _TMP = tempfile.mkdtemp(prefix="r1g_e2e_")


def tearDownModule():
    shutil.rmtree(_TMP, ignore_errors=True)


def people(*others):
    return make_people([("TARGET", KIT_A, 1.0)] + list(others))


def hide_far(frames, a="TARGET", b="PERSON_A"):
    """Drop whichever of a/b is behind the other while it is at least half covered (a real detector misses it)."""
    for cf in frames:
        bx = cf.scene.boxes
        if a in bx and b in bx:
            far = a if bx[a][3] < bx[b][3] else b
            near = b if far == a else a
            if covered(bx[far], bx[near]) >= 1.0 / 2:
                cf.scene.drop = [far]
    return frames


def merge_when_close(frames, a="TARGET", b="PERSON_A", min_overlap_px=8):
    for cf in frames:
        bx = cf.scene.boxes
        if a in bx and b in bx:
            ov = min(bx[a][2], bx[b][2]) - max(bx[a][0], bx[b][0])
            if ov > min_overlap_px:
                cf.scene.merge = [(a, b)]
    return frames


# ---- scenarios (name -> (frames, people, kwargs)) ---------------------------------------------------
def _crossing(kit, cos, n=120, dy=6, mode="visible", pans=None):
    ppl = people(("PERSON_A", kit, cos))
    fr = clip({"TARGET": walk(n, (150, 200), (3, 0)), "PERSON_A": walk(n, (150 + 3 * n, 200 + dy), (-3, 0))},
              pans=pans)
    if mode == "hide":
        hide_far(fr)
    if mode == "merge":
        merge_when_close(fr)
    return fr, ppl, {}


def _occlusion():
    ppl = people(("PERSON_A", KIT_B, 0.2))
    n = 120
    return clip({"TARGET": [box(320, 180)] * n, "PERSON_A": walk(n, (120, 186), (4, 0))}), ppl, {}


def _leave_and_return():
    ppl = people()
    out = walk(40, (560, 200), (4, 0)) + [None] * 30 + walk(60, (640, 200), (-3, 0))
    return clip({"TARGET": out}), ppl, {}


def _wrong_reacquisition():
    ppl = people(("PERSON_A", KIT_A, SAME_KIT_COS))
    tgt = walk(40, (560, 200), (4, 0)) + [None] * 120
    imp = [None] * 50 + walk(110, (650, 200), (-3, 0))
    return clip({"TARGET": tgt, "PERSON_A": imp}), ppl, {}


def _hard_cut():
    ppl = people(("PERSON_A", KIT_A, SAME_KIT_COS))
    n1, n2 = 40, 90
    tgt = [box(300, 180)] * n1 + [box(500, 200)] * n2
    imp = [None] * n1 + [box(300, 180)] * n2               # an impostor exactly where the target was
    shots = [0] * n1 + [1] * n2
    return clip({"TARGET": tgt, "PERSON_A": imp}, shots=shots), ppl, {}


def _pan_crossing():
    n = 120
    return _crossing(KIT_A, SAME_KIT_COS, n=n, pans=[4 * k for k in range(n)])


def _gap_self_competition():
    ppl = people()
    fr = clip({"TARGET": [box(320, 180)] * 90}, drop=lambda k: ["TARGET"] if 20 <= k < 28 else [])
    return fr, ppl, {}


def _newcomer_in_contact():
    ppl = people(("PERSON_A", KIT_A, 0.9))
    n = 90
    return clip({"TARGET": [box(320, 180)] * n,
                 "PERSON_A": [None] * 20 + [box(320 + PW * 0.85, 180)] * (n - 20)}), ppl, {}


def _twin_appears():
    ppl = people(("TWIN", KIT_A, 1.0))
    n = 90
    return clip({"TARGET": [box(320, 180)] * n, "TWIN": [None] * 20 + [box(320 + PW * 2.5, 180)] * (n - 20)}), \
        ppl, {}


def _uncontrasted_return():
    ppl = people()
    return clip({"TARGET": [box(320, 180)] * 100}, drop=lambda k: ["TARGET"] if 10 <= k < 50 else []), ppl, {}


def _posed_crossing():
    """Same-kit crossing with stride-varying boxes and a pose-aware embedding: clean non-seed samples are novel, so
    the bank really accepts samples and bank_violations() is exercised beyond the seed (test audit A1)."""
    ppl = people(("PERSON_A", KIT_A, SAME_KIT_COS))
    n = 160
    fr = clip({"TARGET": gait_walk(n, (120, 200), (2.5, 0)), "PERSON_A": gait_walk(n, (120 + 2.5 * n, 205), (-2.5, 0))})
    return fr, ppl, {"pose_gain": 0.5}


SCENARIOS = {
    "same_kit_crossing": lambda: _crossing(KIT_A, SAME_KIT_COS),
    "same_kit_crossing_hidden": lambda: _crossing(KIT_A, SAME_KIT_COS, mode="hide"),
    "same_kit_crossing_merged": lambda: _crossing(KIT_A, SAME_KIT_COS, mode="merge"),
    "opponent_crossing": lambda: _crossing(KIT_B, 0.2),
    "opponent_crossing_hidden": lambda: _crossing(KIT_B, 0.2, mode="hide"),
    "full_occlusion": _occlusion,
    "leave_and_return": _leave_and_return,
    "wrong_reacquisition": _wrong_reacquisition,
    "hard_cut_impostor": _hard_cut,
    "pan_crossing": _pan_crossing,
    "gap_self_competition": _gap_self_competition,
    "newcomer_in_contact": _newcomer_in_contact,
    "twin_appears": _twin_appears,
    "uncontrasted_return": _uncontrasted_return,
    "posed_crossing": _posed_crossing,
}


def result(name):
    if name not in _CACHE:
        frames, ppl, kw = SCENARIOS[name]()
        _CACHE[name] = run_clip(frames, ppl, os.path.join(_TMP, name), name=name, start_frame=kw.get("start", 2),
                                pose_gain=kw.get("pose_gain", 0.0))
    return _CACHE[name]


class UniversalChecks(unittest.TestCase):
    """Invariants that must hold in every synthetic scenario."""

    def test_every_scenario(self):
        for name in SCENARIOS:
            with self.subTest(scenario=name):
                e = result(name)
                self.assertEqual(S.validate_document(e.doc, require_v2=True, cfg=CFG), [], "R1 contract")
                self.assertEqual(S.validate_document(S.strip_v2(e.doc)), [], "historical contract after strip_v2")
                self.assertEqual(e.swaps(), [], "silent identity swap")
                self.assertEqual(e.target_duplicated(), [], "target written twice (rule 28)")
                self.assertEqual(e.body_written_twice(), [], "target body also written as another person (rule 28)")
                self.assertEqual(e.bank_violations(), [], "target bank contaminated")
                for f in range(1, e.n + 1):
                    t = e.target(f)
                    trusted = t["v2"]["identity_trusted"]
                    if e.target_det_box(f) is None:      # hidden, merged, dropped or absent: never trusted
                        self.assertFalse(trusted, f"frame {f}: trusted without the target's own detection")
                    if trusted:
                        self.assertEqual((t["state"], t["position_source"]), ("ACTIVE", "CONFIRMED_DETECTION"))
                    else:
                        self.assertLess(t["identity_confidence"], CFG.get("legacy_identity_trust_threshold"))
                        self.assertLess(t["overall_track_confidence"], CFG.get("legacy_identity_trust_threshold"))
                    # observed vs predicted/held stays explicit
                    v2 = t["v2"]
                    self.assertEqual(v2["bbox_is_observed"], v2["rb_state"] == "TRACKING")
                    self.assertEqual(v2["observation_source"] == "DETECTION", v2["bbox_is_observed"])
                    # the target's tracklet is never its own competitor, and never written as another person
                    for p in e.population(f):
                        self.assertNotEqual(p["track_id"], CFG.get("target_track_id"))
                    for pair in e.doc["frame_log"][f - 1]["overlap_contests"]:
                        self.assertNotEqual(pair[0], pair[1])
                    ids = {x["track_id"] for x in e.doc["frame_log"][f - 1]["tracks"]}
                    self.assertTrue(set(t["v2"]["occluder_track_ids"]) <= ids, f"frame {f}: occluder not written")
                # D06: historical bbox = last confirmed (observed) box; last_seen_frame = last confirmation
                last_box = e.target(1)["bbox"] if e.target(1)["v2"]["rb_state"] != "TRACKING" else None
                for f in range(1, e.n + 1):
                    t = e.target(f)
                    self.assertEqual(t["last_seen_frame"], t["v2"]["last_confirmed_frame"], f)
                    if t["v2"]["rb_state"] == "TRACKING":
                        self.assertEqual(t["bbox"], t["observed_position_box"], f)
                        last_box = t["bbox"]
                    elif last_box is not None:
                        self.assertEqual(t["bbox"], last_box, f"frame {f}: bbox must be the last confirmed box")


class Scenarios(unittest.TestCase):
    def _states(self, e, lo=1, hi=None):
        return [e.target(f)["v2"]["rb_state"] for f in range(lo, (hi or e.n) + 1)]

    def test_crossings_end_tracking(self):
        for name in ("same_kit_crossing", "same_kit_crossing_hidden", "same_kit_crossing_merged",
                     "opponent_crossing", "opponent_crossing_hidden", "pan_crossing"):
            with self.subTest(scenario=name):
                e = result(name)
                self.assertEqual(e.target(e.n)["v2"]["rb_state"], "TRACKING")

    def test_merged_frames_are_never_trusted(self):
        e = result("same_kit_crossing_merged")
        merged = [f for f in range(1, e.n + 1) if any(len(d.owners) == 2 and "TARGET" in d.owners
                                                       for d in e.dets[f])]
        self.assertTrue(merged)
        for f in merged:
            self.assertFalse(e.target(f)["v2"]["identity_trusted"], f)

    def test_occluder_never_inherits_identity(self):
        e = result("full_occlusion")
        hidden = [f for f in range(1, e.n + 1) if e.target_det_box(f) is None]
        self.assertTrue(hidden)
        for f in hidden:
            t = e.target(f)
            self.assertIn(t["v2"]["rb_state"], ("OCCLUDED", "AMBIGUOUS", "LOST", "REACQUIRING"))
            self.assertFalse(t["v2"]["bbox_is_observed"])
        self.assertEqual(e.target(e.n)["v2"]["rb_state"], "TRACKING")

    def test_leaving_frame_is_lost_out_of_frame_then_reacquired(self):
        e = result("leave_and_return")
        states = Counter(e.target(f)["state"] for f in range(1, e.n + 1))
        self.assertIn("OUT_OF_FRAME", states)
        self.assertEqual(len(e.summary["v2"]["reacquisition_events"]), 1)
        self.assertEqual(e.target(e.n)["v2"]["rb_state"], "TRACKING")

    def test_wrong_reacquisition_rejected(self):
        e = result("wrong_reacquisition")
        gone = [f for f in range(1, e.n + 1) if e.target_det_box(f) is None]
        self.assertTrue(gone)
        self.assertEqual([f for f in gone if e.target(f)["v2"]["identity_trusted"]], [])
        self.assertEqual(e.summary["v2"]["reacquisition_events"], [])

    def test_rejected_lookalike_is_still_written_as_a_person(self):
        """R1-G review M1: a person the tracker has rejected (not the target, not under verification) is a real
        body and must appear as a population record; only what may physically be the target is withheld."""
        e = result("wrong_reacquisition")
        checked = 0
        for f in range(1, e.n + 1):
            other = e.target_det_box(f, "PERSON_A")
            if other is None:
                continue
            t = e.target(f)
            v2 = t["v2"]
            if v2["rb_state"] == "REACQUIRING":
                continue                                  # the verifying candidate may be withheld
            if iou(other, t["observed_position_box"]) >= 0.3:
                continue                                  # standing where the target would re-emerge (rule 28)
            written = [p for p in e.population(f) if iou(other, p["observed_position_box"]) > 0.9]
            self.assertEqual(len(written), 1, f"frame {f}: rejected look-alike missing from the output")
            checked += 1
        self.assertGreater(checked, 20)

    def test_hard_cut_resets_continuity(self):
        e = result("hard_cut_impostor")
        cut = [fr["frame"] for fr in e.doc["frame_log"] if fr["is_cut"]]
        self.assertEqual(len(cut), 1, "exactly one hard cut")
        c = cut[0]
        t = e.target(c)
        # continuity is gone: LOST, or already verifying a candidate (appearance only) in the same frame
        self.assertIn(t["v2"]["rb_state"], ("LOST", "REACQUIRING"))
        self.assertFalse(t["v2"]["identity_trusted"])
        self.assertIn("CAMERA_CUT", t["v2"]["identity_uncertainty_reasons"])
        self.assertTrue(e.doc["frame_log"][c - 1]["v2"]["cut_detected"])
        # position means nothing after the cut: the impostor standing exactly at the old place is never accepted
        imp = [f for f in range(c, e.n + 1) if e.target(f)["v2"]["identity_trusted"]
               and iou(e.target(f)["observed_position_box"], e.target_det_box(f, "PERSON_A") or (0, 0, 1, 1)) > 0.5]
        self.assertEqual(imp, [])
        # the true target is reacquired by appearance, as a new episode
        self.assertEqual(e.target(e.n)["v2"]["rb_state"], "TRACKING")
        self.assertGreaterEqual(e.target(e.n)["v2"]["target_episode"], 2)

    def test_bank_check_sees_real_non_seed_samples(self):
        e = result("posed_crossing")
        accepted = [f for f in e.accepted_bank_frames() if f > 0]
        self.assertGreaterEqual(len(accepted), 2, f"only {accepted}: bank_violations() would be vacuous")
        self.assertEqual(e.bank_violations(), [])

    def test_pan_is_measured(self):
        e = result("pan_crossing")
        shifts = [fr["camera_shift"][0] for fr in e.doc["frame_log"][2:]]
        self.assertTrue(all(abs(s - (-4.0)) <= 1.0 for s in shifts), shifts[:10])

    def test_gap_does_not_make_the_target_its_own_competitor(self):
        e = result("gap_self_competition")
        self.assertEqual(e.target(e.n)["v2"]["rb_state"], "TRACKING")
        for f in range(1, e.n + 1):
            own = e.target_det_box(f)
            for p in e.population(f):
                if own is not None and p["v2"]["detected_this_frame"]:
                    self.assertLess(iou(own, p["observed_position_box"]), 0.5, f"frame {f}: target also a person")

    def test_newcomer_in_contact_and_twin_block_without_swap(self):
        for name in ("newcomer_in_contact", "twin_appears"):
            with self.subTest(scenario=name):
                e = result(name)
                after = self._states(e, 20)
                self.assertNotIn("TRACKING", after[2:], "an unverifiable rival must block trust")
                self.assertTrue(any(k.startswith("T001|") for k in e.summary["identity_conflict_counts"]))
                self.assertGreater(e.summary["identity_conflicts_logged"], 0)

    def test_uncontrasted_reacquisition_stays_capped_and_flagged(self):
        """Nobody else is ever banked: the returning target cannot be contrast-verified. Its box is reported as
        observed (rb TRACKING) but untrusted, flagged INSUFFICIENT_CONTRAST, below the consumers' threshold."""
        e = result("uncontrasted_return")
        ev = e.summary["v2"]["reacquisition_events"]
        self.assertEqual(len(ev), 1)
        self.assertFalse(ev[0]["evidence"]["contrast_available"])
        self.assertFalse(ev[0]["evidence"]["contrast_verified"])
        acc = ev[0]["reacq_accepted_frame"]
        observed = [f for f in range(acc, e.n + 1) if e.target(f)["v2"]["rb_state"] == "TRACKING"]
        self.assertTrue(observed)
        for f in observed:
            t = e.target(f)
            self.assertFalse(t["v2"]["identity_trusted"], f)
            self.assertEqual(t["v2"]["identity_uncertainty_reasons"], ["INSUFFICIENT_CONTRAST"])
            self.assertLess(t["identity_confidence"], 0.5)
            self.assertLess(t["overall_track_confidence"], 0.5)
            self.assertEqual(t["position_source"], "CONFIRMED_DETECTION")       # the box itself is observed


if __name__ == "__main__":
    unittest.main()

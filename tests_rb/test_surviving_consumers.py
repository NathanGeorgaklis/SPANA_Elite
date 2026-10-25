# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R0/R0_R1_TRACKER_REBUILD_PLAN.md item 14 (run the UNMODIFIED surviving consumers on R1 output with
#                 test-only stubs for their missing imports); R0/R0_TRACKER_INTERFACE_MAP.md §5; R1/R1_DESIGN_DECISIONS.md
#                 D05, D19; R1-G instruction (Nathan, 2026-10-25)
"""The surviving Pass-2 consumers read R1 output safely.

Loaded read-only from recovered/ (hash-pinned, never modified, never imported by player_tracker_rb):
  * player_tracker_mo/event_adapter.py   build_integrated_event_snapshots (frame_log -> TrackingSnapshot list)
  * event_engine_v1/actor_attribution.py frame_identity, selected_ids, conflicts_involve
  * football_analysis/confidence.py      (3N version) HIGH_THRESH / MEDIUM_THRESH, imported by actor_attribution
Their other missing imports are TEST-ONLY STUBS defined here (NEW, never written to a historical path):
  * football_analysis.adapter: TrackingSnapshot / TrackingAdapter (plain records);
  * new_tracker_snapshots.NEW_TO_LEGACY_STATE: exactly the evidenced mapping rows of output_schema
    (LEGACY_STATE_EVIDENCE); unknown states fall to the adapter's own .get(..., "LOST") default.

Safety properties checked on full-pipeline R1 output (synthetic scenarios with crossings, merges, cuts and rivals):
  * a snapshot is TRACKING only on an R1 TRACKING frame, and every non-TRACKING R1 frame is non-TRACKING downstream;
  * the snapshot tracking_confidence of every untrusted frame is below the consumers' 0.5 threshold;
  * an episode opened by an uncontrasted reacquisition (possibly a look-alike) never yields actor TARGET_PLAYER
    (R1-G review H1);
  * actor_attribution.frame_identity trusts a frame only if R1 trusted it;
  * logged identity conflicts reach actor_attribution (conflicts_involve) when R1 recorded a competing hypothesis.
"""
import dataclasses
import hashlib
import importlib.util
import os
import shutil
import sys
import tempfile
import types
import unittest

from player_tracker_rb import output_schema as S
from tests_rb._synth_pipeline import PW, box, clip, run_clip, walk
from tests_rb._synth_scene import make_people

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REC = os.path.join(ROOT, "recovered", "SPANA_recovered_originals")
CONSUMERS = {
    "event_adapter": ("engine_current_3Q_state_exact/player_tracker_mo/event_adapter.py",
                      "418d54219ff18bb62b87502fbe14c9beff1ae1fc4a8d9c4de80ec4f948ca90dd"),
    "actor_attribution": ("engine_current_3Q_state_exact/event_engine_v1/actor_attribution.py",
                          "ac2003830a664790d91d3bea770b0acd42777503cb438ca4e58965ba28c65793"),
    "confidence": ("engine_3N_versions_of_files_changed_in_3P_3Q/football_analysis/confidence.py",
                   "41a94f8e628de1d9cc2c7821573a637ae7fda0148208d626a5f21fd03845b431"),
}
KIT_A = (200, 60, 40)
KIT_B = (40, 40, 210)


@dataclasses.dataclass
class TrackingSnapshot:                     # TEST-ONLY STUB of football_analysis.adapter.TrackingSnapshot
    player_id: str
    frame_number: int
    timestamp: float
    bbox: list
    centre_x: float
    centre_y: float
    tracking_confidence: float
    tracking_state: str


class TrackingAdapter:                      # TEST-ONLY STUB (never used by the functions under test)
    pass


def _load(name, rel, sha):
    path = os.path.join(REC, rel)
    with open(path, "rb") as f:
        got = hashlib.sha256(f.read()).hexdigest()
    if got != sha:
        raise AssertionError(f"recovered consumer {rel} hash {got} != pinned {sha}")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_SAVED = {}
_MODS = {}


def setUpModule():
    stubs = {
        "football_analysis": types.ModuleType("football_analysis"),
        "football_analysis.adapter": types.ModuleType("football_analysis.adapter"),
        "new_tracker_snapshots": types.ModuleType("new_tracker_snapshots"),
    }
    stubs["football_analysis.adapter"].TrackingSnapshot = TrackingSnapshot
    stubs["football_analysis.adapter"].TrackingAdapter = TrackingAdapter
    stubs["new_tracker_snapshots"].NEW_TO_LEGACY_STATE = {k: v[0] for k, v in S.LEGACY_STATE_EVIDENCE.items()}
    for k in list(stubs) + ["football_analysis.confidence"]:
        _SAVED[k] = sys.modules.get(k)
    sys.modules.update(stubs)
    conf = _load("football_analysis.confidence", *CONSUMERS["confidence"])
    sys.modules["football_analysis.confidence"] = conf
    stubs["football_analysis"].confidence = conf
    stubs["football_analysis"].adapter = stubs["football_analysis.adapter"]
    _MODS["event_adapter"] = _load("rb_test_event_adapter", *CONSUMERS["event_adapter"])
    _MODS["actor_attribution"] = _load("rb_test_actor_attribution", *CONSUMERS["actor_attribution"])
    _MODS["tmp"] = tempfile.mkdtemp(prefix="r1g_consumers_")


def tearDownModule():
    for k, v in _SAVED.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v
    shutil.rmtree(_MODS["tmp"], ignore_errors=True)


def _scenarios():
    people = make_people([("TARGET", KIT_A, 1.0), ("PERSON_A", KIT_A, 0.75), ("TWIN", KIT_A, 1.0)])
    n = 150
    tgt = walk(70, (150, 200), (3, 0)) + [box(360, 200)] * (n - 70)
    other = walk(70, (150 + 3 * 70, 204), (-3, 0)) + [None] * (n - 70)
    twin = [None] * 100 + [box(360 + PW * 2.5, 200)] * (n - 100)
    shots = [0] * 85 + [1] * (n - 85)
    fr = clip({"TARGET": tgt, "PERSON_A": other, "TWIN": twin}, shots=shots,
              merge=lambda k: [("TARGET", "PERSON_A")] if 62 <= k < 68 else [])
    return run_clip(fr, people, os.path.join(_MODS["tmp"], "mix"), name="mix", start_frame=2)


class SurvivingConsumers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.e = _scenarios()
        cls.ea = _MODS["event_adapter"]
        cls.aa = _MODS["actor_attribution"]

    def test_scenario_exercises_untrusted_states(self):
        rb = {self.e.target(f)["v2"]["rb_state"] for f in range(1, self.e.n + 1)}
        self.assertIn("TRACKING", rb)
        self.assertTrue(rb & {"OCCLUDED", "AMBIGUOUS", "LOST", "REACQUIRING"}, rb)

    def test_event_adapter_never_promotes_an_untrusted_frame(self):
        fl = self.e.doc["frame_log"]
        snaps = self.ea.build_integrated_event_snapshots(fl, self.e.doc["summary"]["fps"])
        self.assertEqual(len(snaps), len(fl) + 1)            # frame-0 bootstrap + one per frame
        for snap, fr in zip(snaps[1:], fl):
            t = [x for x in fr["tracks"] if x["selected_player"]][0]
            self.assertEqual(snap.frame_number, fr["frame"])
            trusted = t["v2"]["identity_trusted"]
            # the adapter sees TRACKING exactly when R1 observed the target's box (rb TRACKING); every frame that is
            # not R1-trusted (including an observed box of an unverified episode) carries confidence < 0.5
            self.assertEqual(snap.tracking_state == "TRACKING", t["v2"]["rb_state"] == "TRACKING",
                             f"frame {fr['frame']}")
            if trusted:
                self.assertEqual(snap.tracking_state, "TRACKING")
            else:
                self.assertLess(snap.tracking_confidence, 0.5)
            self.assertEqual(snap.bbox, t["bbox"])

    def test_actor_attribution_trusts_only_r1_trusted_frames(self):
        for fr in self.e.doc["frame_log"]:
            t = [x for x in fr["tracks"] if x["selected_player"]][0]
            verdict = self.aa.frame_identity(fr)
            if verdict["trusted"]:
                self.assertTrue(t["v2"]["identity_trusted"], f"frame {fr['frame']}: {verdict}")
            if not t["v2"]["identity_trusted"]:
                self.assertFalse(verdict["trusted"])

    def test_identity_conflicts_reach_actor_attribution(self):
        s = self.e.doc["summary"]
        self.assertTrue(s["identity_conflict_counts"], "the twin must be logged as a competing hypothesis")
        ids = self.aa.selected_ids(self.e.doc["frame_log"])
        self.assertEqual(ids, ["T001"])
        self.assertTrue(self.aa.conflicts_involve(ids, s["identity_conflict_counts"]))


class UncontrastedEpisodeDownstream(unittest.TestCase):
    """R1-G review H1: the target leaves for good; a near-identical person enters later with nobody else around, so
    the reacquisition has nothing to contrast. Whatever happens next must never be attributed to the target."""

    def test_lookalike_after_uncontrasted_reacquisition_is_never_the_actor(self):
        aa = _MODS["actor_attribution"]
        people = make_people([("TARGET", KIT_A, 1.0), ("TWIN", KIT_A, 0.97)])
        tgt = walk(40, (560, 200), (4, 0)) + [None] * 140
        twin = [None] * 70 + walk(110, (650, 200), (-3, 0))
        e = run_clip(clip({"TARGET": tgt, "TWIN": twin}), people, os.path.join(_MODS["tmp"], "unc"), name="unc",
                     start_frame=2)
        ev = e.summary["v2"]["reacquisition_events"]
        self.assertTrue(ev, "the scene must produce an uncontrasted reacquisition of the look-alike")
        self.assertFalse(ev[-1]["evidence"]["contrast_verified"])
        acc = ev[-1]["reacq_accepted_frame"]
        idx = aa.build_frame_index(e.doc["frame_log"])
        for start in range(acc, e.n - 5, 5):
            raw = {"type": "CARRY", "start_frame": start, "end_frame": min(start + 15, e.n)}
            sel = aa.selected_ids(e.doc["frame_log"])
            a = aa.assess_actor(raw, [raw], idx, "RESOLVED",
                                aa.conflicts_involve(sel, e.summary["identity_conflict_counts"]))
            self.assertNotEqual(a["actor"], "TARGET_PLAYER", f"window from {start}: {a['actor_evidence']}")
        for f in range(acc, e.n + 1):
            t = e.target(f)
            self.assertFalse(t["v2"]["identity_trusted"], f)
            self.assertLess(t["identity_confidence"], 0.5)


if __name__ == "__main__":
    unittest.main()

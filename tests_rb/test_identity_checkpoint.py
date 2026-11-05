# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2 DEV human-assisted target reacquisition instruction (Nathan, 2026-11-05 20:22): tests for false
#                 suspension on the true target, genuine same-kit substitution, target leaves / re-enters, wrong user
#                 selection, ambiguous box, camera cut, multiple checkpoints in one clip.
"""Identity checkpoints + user-confirmed re-seeds (checkpoint.py) on synthetic scenes with scripted resolvers."""
import dataclasses
import unittest

import numpy as np

from player_tracker_rb import checkpoint as CK
from player_tracker_rb import population as P
from player_tracker_rb import target as T
from player_tracker_rb.appearance import AppearanceExtractor
from player_tracker_rb.camera import FIRST_FRAME
from player_tracker_rb.config import TrackerConfig
from player_tracker_rb.detector import PersonCandidate
from tests_rb import _synth_scene as SS

KIT_A, KIT_B = (200, 60, 40), (40, 60, 200)
H = 100.0
FPS = 30.0
CFG = TrackerConfig.default()
WIN = CFG.get("checkpoint_unresolved_s")


def people(*others):
    return SS.make_people([("TARGET", KIT_A, 1.0)] + list(others))


class Run:
    def __init__(self, frames, out, dets, tracker, pp):
        self.frames, self.out, self.dets, self.tracker, self.people = frames, out, dets, tracker, pp

    def owners(self, k):
        tf = self.out[k]
        return None if tf.detection_index is None else self.dets[k][tf.detection_index].owners

    def wrong_trusted(self, allowed=("TARGET",)):
        return [k for k, tf in enumerate(self.out)
                if tf.identity_trusted and tf.rb_state == T.TRACKING and self.owners(k) not in [(a,) for a in allowed]]

    def trusted(self):
        return [k for k, tf in enumerate(self.out) if tf.identity_trusted]

    def bank_of(self, names):
        """Every current bank sample's identity part is one of the named people's identity vectors."""
        vecs = [self.people[n].vec for n in names]
        bank = (self.tracker.tracker if isinstance(self.tracker, CK.IdentityCheckpointer) else self.tracker).bank
        for smp in bank.samples:
            e = np.array(smp.embedding, float)
            e[list(SS.POSE_DIMS)] = 0.0
            e = SS.l2_normalise(e)
            if not any(float(e @ v) > 1.0 - 1e-9 for v in vecs):
                return False
        return True

    def episodes_consistent(self):
        inc = sum(1 for a, b in zip(self.out, self.out[1:]) if b.target_episode == a.target_episode + 1)
        ok = all(b.target_episode in (a.target_episode, a.target_episode + 1) for a, b in zip(self.out, self.out[1:]))
        return ok and inc == len(self.tracker.reacquisition_events)


def run(frames, pp, resolver=None, cfg=CFG, fps=FPS):
    pop = P.Population(cfg, SS.W, SS.H)
    ext = AppearanceExtractor(cfg, SS.SignatureEmbedding(pp))
    tr = T.TargetTracker(cfg, pop, ext) if resolver is None else CK.IdentityCheckpointer(cfg, pop, ext, resolver)
    out, dets = [], []
    for k, fr in enumerate(frames):
        img = SS.render(fr, pp)
        ds = SS.detect(fr)
        cands = [PersonCandidate(d.box, 0.9, d.box[0] <= 2 or d.box[2] >= SS.W - 2, "synthetic") for d in ds]
        cam = SS.identity_motion(FIRST_FRAME) if k == 0 else (
            fr.camera if fr.camera is not None else SS.motion_from_matrix([[1, 0, 0], [0, 1, 0]], SS.W, SS.H))
        res = pop.update(k, k / fps, cands, cam)
        if resolver is not None:
            resolver.scene = (fr, ds)
        out.append(tr.seed(res, img, fr.boxes["TARGET"]) if k == 0 else tr.update(res, img))
        dets.append(ds)
    return Run(frames, out, dets, tr, pp)


class Oracle:
    """A scripted user. pick(scene_frame, detections) -> box to SELECT, 'NOT_VISIBLE', or None (no answer)."""

    def __init__(self, pick):
        self.pick, self.scene, self.calls = pick, None, []

    def __call__(self, view, res):
        self.calls.append(view)
        a = self.pick(*self.scene)
        if a is None:
            return None
        if a == CK.NOT_VISIBLE:
            return {"decision": CK.NOT_VISIBLE}
        return {"decision": CK.SELECT, "box": list(a)}


def true_box(names):
    """The detection box of the first named person if detected alone, else NOT_VISIBLE."""
    def pick(fr, ds):
        for d in ds:
            if len(d.owners) == 1 and d.owners[0] in names:
                return d.box
        return CK.NOT_VISIBLE
    return pick


def strip(tf):
    return dataclasses.replace(tf, decision={})


def persistent_look_change(n1_s=1.5, n2_s=4.0, cos=0.7):
    """The TRUE target's look changes persistently (e.g. a turn / lighting): a false suspension on the true target."""
    n1, n2 = int(n1_s * FPS), int(n2_s * FPS)
    fr = [SS.SceneFrame({"TARGET": SS.box_at(400, 400, 40, H)}) for _ in range(n1)]
    # TARGET_ALT is the SAME physical person as TARGET (the truth), rendered with a changed look
    fr += [SS.SceneFrame({"TARGET_ALT": SS.box_at(401, 400, 40, H)}) for _ in range(n2)]
    return fr, people(("TARGET_ALT", KIT_A, cos)), n1


class TestIdentityCheckpoint(unittest.TestCase):
    def test_no_answer_changes_nothing(self):
        fr, pp, n1 = persistent_look_change()
        base = run(fr, pp)
        silent = run(fr, pp, Oracle(lambda f, d: None))
        declined = run(fr, pp, Oracle(lambda f, d: CK.NOT_VISIBLE))
        for r in (silent, declined):
            self.assertEqual([strip(a) for a in base.out], [strip(b) for b in r.out],
                             "no automatic candidate is ever promoted by a checkpoint")
        self.assertTrue(silent.tracker.checkpoints, "a checkpoint was emitted")
        cp = silent.tracker.checkpoints[0]
        self.assertEqual(cp["kind"], "IDENTITY_CHECKPOINT")
        self.assertGreaterEqual(cp["opened_time_s"] - cp["unresolved_since_s"], WIN - 1e-9)
        self.assertTrue(cp["candidates"] and all("box" in c for c in cp["candidates"]))
        self.assertIn("rb_state", cp["context"])

    def test_false_suspension_on_true_target_recovers_with_correct_confirmation(self):
        fr, pp, n1 = persistent_look_change()
        base = run(fr, pp)
        self.assertFalse(any(base.out[k].identity_trusted for k in range(n1 + 2, len(fr))),
                         "control: the automatic tracker never recovers the changed look (false suspension)")
        r = run(fr, pp, Oracle(true_box(("TARGET", "TARGET_ALT"))))
        cps = r.tracker.checkpoints
        self.assertEqual([c["status"] for c in cps], [CK.RESEEDED])
        k_re = cps[0]["closed_frame"]
        self.assertTrue(all(r.out[k].identity_trusted for k in range(k_re, len(fr))), "trust restored")
        self.assertEqual(r.wrong_trusted(("TARGET", "TARGET_ALT")), [])
        self.assertTrue(r.bank_of(("TARGET_ALT",)), "the new bank holds only post-re-seed samples")
        self.assertEqual([strip(a) for a in base.out[:k_re]], [strip(b) for b in r.out[:k_re]],
                         "pre-checkpoint output is preserved")
        self.assertTrue(r.episodes_consistent())
        self.assertTrue(r.out[k_re].reidentified_this_frame)
        self.assertEqual(r.out[k_re].target_episode, r.out[k_re - 1].target_episode + 1)
        ev = r.tracker.reacquisition_events[-1]
        self.assertEqual(ev.evidence["source"], CK.USER_RESEED)
        self.assertEqual(ev.reacq_accepted_frame, k_re)
        self.assertGreater(len(r.trusted()), len(base.trusted()))

    def test_genuine_same_kit_substitution_is_declined_and_never_trusted(self):
        n1, n2 = int(1.5 * FPS), int(5.0 * FPS)
        fr = [SS.SceneFrame({"TARGET": SS.box_at(400, 400, 40, H)}) for _ in range(n1)]
        fr += [SS.SceneFrame({"LOOK": SS.box_at(402, 400, 40, H)}) for _ in range(n2)]
        pp = people(("LOOK", KIT_A, 0.85))
        r = run(fr, pp, Oracle(true_box(("TARGET",))))
        self.assertEqual(r.wrong_trusted(), [])
        self.assertTrue(r.tracker.checkpoints)
        self.assertTrue(all(c["status"] == CK.DECLINED for c in r.tracker.checkpoints))
        self.assertGreaterEqual(len(r.tracker.checkpoints), 2, "re-armed after each decline")
        gaps = [b["opened_time_s"] - a["closed_time_s"] for a, b in zip(r.tracker.checkpoints, r.tracker.checkpoints[1:])]
        self.assertTrue(all(g >= WIN - 1e-9 for g in gaps))
        self.assertTrue(r.bank_of(("TARGET",)))

    def test_target_leaves_and_reenters(self):
        n1, n2, n3 = int(1.5 * FPS), int(3.0 * FPS), int(3.0 * FPS)
        fr = [SS.SceneFrame({"TARGET": SS.box_at(400, 400, 40, H), "OTHER": SS.box_at(900, 400, 40, H)})
              for _ in range(n1)]
        fr += [SS.SceneFrame({"OTHER": SS.box_at(900, 400, 40, H)}) for _ in range(n2)]
        fr += [SS.SceneFrame({"TARGET": SS.box_at(150, 420, 40, H), "OTHER": SS.box_at(900, 400, 40, H)})
               for _ in range(n3)]
        pp = people(("OTHER", KIT_A, 0.75))
        r = run(fr, pp, Oracle(true_box(("TARGET",))))
        self.assertEqual(r.wrong_trusted(), [])
        st = [c["status"] for c in r.tracker.checkpoints]
        self.assertIn(CK.DECLINED, st, "while the target is out of view the user declines")
        self.assertTrue(any(r.out[k].identity_trusted for k in range(n1 + n2, len(fr))), "recovered after re-entry")
        self.assertTrue(r.episodes_consistent())

    def test_wrong_user_selection_is_the_users_authority_but_restarts_a_clean_bank(self):
        fr, pp, n1 = persistent_look_change()
        for f in fr:
            f.boxes["OTHER"] = SS.box_at(900, 400, 40, H)
        pp = people(("TARGET_ALT", KIT_A, 0.7), ("OTHER", KIT_A, 0.75))
        base = run(fr, pp)
        r = run(fr, pp, Oracle(true_box(("OTHER",))))
        cp = r.tracker.checkpoints[0]
        self.assertEqual(cp["status"], CK.RESEEDED)
        k_re = cp["closed_frame"]
        self.assertEqual([strip(a) for a in base.out[:k_re]], [strip(b) for b in r.out[:k_re]])
        self.assertTrue(r.bank_of(("OTHER",)), "the bank restarted from the (wrong) confirmed player only")
        self.assertTrue(all(r.owners(k) == ("OTHER",) for k in range(k_re, len(fr)) if r.out[k].identity_trusted),
                        "a wrong confirmation is followed faithfully: recorded as the user's decision")

    def test_selection_on_no_detection_is_refused(self):
        fr, pp, n1 = persistent_look_change()
        r = run(fr, pp, Oracle(lambda f, d: SS.box_at(1100, 150, 40, H)))
        ans = [a for c in r.tracker.checkpoints for a in c["answers"]]
        self.assertTrue(ans and all(a["outcome"] == "REFUSED" for a in ans))
        self.assertEqual(r.tracker.checkpoints[0]["status"], CK.OPEN)
        base = run(fr, pp)
        self.assertEqual([strip(a) for a in base.out], [strip(b) for b in r.out])

    def test_ambiguous_box_is_refused_then_a_clear_answer_is_accepted(self):
        n1, n2 = int(1.5 * FPS), int(5.0 * FPS)
        fr = [SS.SceneFrame({"TARGET": SS.box_at(400, 400, 40, H), "MATE": SS.box_at(520, 400, 40, H)})
              for _ in range(n1)]
        fr += [SS.SceneFrame({"TARGET_ALT": SS.box_at(401, 400, 40, H), "MATE": SS.box_at(445, 400, 40, H)})
               for _ in range(n2)]
        pp = people(("TARGET_ALT", KIT_A, 0.7), ("MATE", KIT_A, 0.8))
        state = {"n": 0}

        def pick(f, ds):
            state["n"] += 1
            a, b = f.boxes["TARGET_ALT"], f.boxes["MATE"]
            if state["n"] <= 3:                              # one loose box around both players
                return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
            return true_box(("TARGET_ALT",))(f, ds)
        r = run(fr, pp, Oracle(pick))
        ans = r.tracker.checkpoints[0]["answers"]
        self.assertTrue(all(a["outcome"] == "REFUSED" for a in ans[:3]), ans[:3])
        self.assertTrue(any("ambiguous" in a.get("reason", "") or "refused" in a.get("reason", "") for a in ans[:3]))
        self.assertEqual(r.tracker.checkpoints[0]["status"], CK.RESEEDED)
        self.assertEqual(r.wrong_trusted(("TARGET", "TARGET_ALT")), [])

    def test_camera_cut_while_a_checkpoint_is_open(self):
        from player_tracker_rb import camera as C
        fr, pp, n1 = persistent_look_change(n2_s=5.0)
        kc = n1 + int(2.0 * FPS)                             # the cut lands after the checkpoint opened
        fr[kc] = SS.SceneFrame({"TARGET_ALT": SS.box_at(700, 380, 50, 120)},
                               camera=SS.identity_motion(C.CUT, continuity_broken=True))
        for k in range(kc + 1, len(fr)):
            fr[k] = SS.SceneFrame({"TARGET_ALT": SS.box_at(700, 380, 50, 120)})
        state = {"after_cut": False}

        def pick(f, ds):
            if f.camera is not None:
                state["after_cut"] = True
            return true_box(("TARGET_ALT",))(f, ds) if state["after_cut"] else None
        r = run(fr, pp, Oracle(pick))
        cp = r.tracker.checkpoints[0]
        self.assertLess(cp["opened_frame"], kc)
        self.assertEqual(cp["status"], CK.RESEEDED)
        self.assertGreaterEqual(cp["closed_frame"], kc)
        self.assertEqual(r.wrong_trusted(("TARGET", "TARGET_ALT")), [])
        self.assertTrue(r.bank_of(("TARGET_ALT",)))
        self.assertTrue(r.episodes_consistent())

    def test_multiple_checkpoints_in_one_clip(self):
        n = int(1.5 * FPS)
        seq = [("TARGET", n), ("TARGET_ALT", int(3.0 * FPS)), ("TARGET_ALT2", int(3.0 * FPS))]
        fr = []
        for name, m in seq:
            fr += [SS.SceneFrame({name: SS.box_at(400, 400, 40, H)}) for _ in range(m)]
        pp = people(("TARGET_ALT", KIT_A, 0.7), ("TARGET_ALT2", KIT_A, 0.6))
        r = run(fr, pp, Oracle(true_box(("TARGET", "TARGET_ALT", "TARGET_ALT2"))))
        st = [c["status"] for c in r.tracker.checkpoints]
        self.assertEqual(st.count(CK.RESEEDED), 2, st)
        self.assertEqual(r.wrong_trusted(("TARGET", "TARGET_ALT", "TARGET_ALT2")), [])
        self.assertTrue(r.bank_of(("TARGET_ALT2",)), "only the latest re-seed's samples are in the bank")
        self.assertTrue(r.episodes_consistent())
        src = [e.evidence.get("source") for e in r.tracker.reacquisition_events]
        self.assertEqual(src.count(CK.USER_RESEED), 2)

    def test_auto_recovery_closes_an_open_checkpoint(self):
        # the dip resolves automatically after the checkpoint opened: AUTO_RESOLVED, nothing re-seeded
        cfg = TrackerConfig.with_overrides({"checkpoint_unresolved_s": 0.0})
        n1 = int(1.5 * FPS)
        fr = [SS.SceneFrame({"TARGET": SS.box_at(400, 400, 40, H)}) for _ in range(n1)]
        fr += [SS.SceneFrame({"GLITCH": SS.box_at(401, 400, 40, H)}) for _ in range(2)]
        fr += [SS.SceneFrame({"TARGET": SS.box_at(400, 400, 40, H)}) for _ in range(n1)]
        pp = people(("GLITCH", KIT_A, 0.7))
        r = run(fr, pp, Oracle(lambda f, d: None), cfg=cfg)
        self.assertTrue(r.tracker.checkpoints)
        self.assertEqual(r.tracker.checkpoints[-1]["status"], CK.AUTO_RESOLVED)
        base = run(fr, pp, cfg=cfg)
        self.assertEqual([strip(a) for a in base.out], [strip(b) for b in r.out])


if __name__ == "__main__":
    unittest.main()


class TestSimulatedCorrectUser(unittest.TestCase):
    def test_answers_follow_the_frozen_human_labels(self):
        import hashlib, json, os, tempfile
        from tools_rb import r2_checkpoint_dev as CD
        d = tempfile.mkdtemp(prefix="r2_cu_")
        os.makedirs(os.path.join(d, "R2", "labels"))
        pf = [{"visibility": "VISIBLE", "resolvable": "YES", "human_uncertain": False}] * 3 + \
             [{"visibility": "HIDDEN", "resolvable": "NO", "human_uncertain": False},
              {"visibility": "PARTIAL", "resolvable": "NO", "human_uncertain": False},
              {"visibility": "VISIBLE", "resolvable": "YES", "human_uncertain": True}]
        lab = {"per_frame": pf, "target_boxes": {"2": [1, 2, 30, 90]}}
        p = os.path.join(d, "R2", "labels", "x.json")
        with open(p, "w") as f:
            json.dump(lab, f)
        fx = {"labels_file": "x.json", "labels_sha256": hashlib.sha256(open(p, "rb").read()).hexdigest(),
              "seed_frame": 1}
        u = CD.correct_user(d, fx)
        self.assertIsNone(u({"frame": 0}, None), "visible but no human box at this frame: no answer yet")
        self.assertEqual(u({"frame": 1}, None), {"decision": "SELECT", "box": [1, 2, 30, 90]})
        for f in (2, 3, 4):
            self.assertEqual(u({"frame": f}, None), {"decision": "NOT_VISIBLE"})
        self.assertIsNone(u({"frame": 99}, None))
        fx["labels_sha256"] = "0" * 64
        with self.assertRaises(SystemExit):
            CD.correct_user(d, fx)


class TestReviewFindings(unittest.TestCase):
    """Regression tests for the focused review of checkpoint.py (2026-11-05)."""

    def test_config_window_is_the_automatic_paths_window(self):
        self.assertAlmostEqual(CFG.get("checkpoint_unresolved_s"),
                               CFG.get("max_ambiguous_s") + CFG.get("reacq_window_s"))
        self.assertGreater(CFG.get("checkpoint_unresolved_s"), CFG.get("seed_anchor_max_s"))

    def test_former_target_tracklets_stay_the_target_after_a_reseed(self):
        # review #1 (rule 28): the old target tracklets stay in the target's physical-entity set after the re-seed
        fr, pp, n1 = persistent_look_change()
        r = run(fr, pp, Oracle(true_box(("TARGET", "TARGET_ALT"))))
        base = run(fr, pp)
        self.assertTrue(set(base.tracker.target_tracklets) <= set(r.tracker.target_tracklets))

    def test_malformed_answers_are_refused_not_fatal(self):
        fr, pp, n1 = persistent_look_change()
        bad = iter([{"decision": CK.SELECT, "box": [1, 2, 3]}, {"decision": CK.SELECT, "box": [1, 2, float("nan"), 4]},
                    {"decision": CK.SELECT, "box": [5, 5, 1, 1]}, {"decision": "MAYBE"}, "junk",
                    {"decision": CK.SELECT, "box": ["a", 1, 2, 3]}])

        class R:
            scene = None
            def __call__(self, view, res):
                return next(bad, None)
        r = run(fr, pp, R())
        ans = r.tracker.checkpoints[0]["answers"]
        self.assertEqual(len(ans), 6)
        self.assertTrue(all(a["outcome"] == "REFUSED" for a in ans), ans)
        self.assertEqual(r.tracker.checkpoints[0]["status"], CK.OPEN, "a refused answer never closes the checkpoint")
        self.assertEqual([strip(a) for a in run(fr, pp).out], [strip(b) for b in r.out])

    def _problem(self, obs, groups=None, lineage=()):
        from types import SimpleNamespace as NS
        observations = [NS(index=k, is_body=True, bbox=b, tracklet_id=t) for k, (b, t) in enumerate(obs)]
        res = NS(observations=observations,
                 tracklet=lambda tid: NS(flags=(P.MERGE_LINEAGE,) if tid in lineage else ()))
        probe = NS(_group=lambda r, i: set((groups or {}).get(i, {i})))
        ck = CK.IdentityCheckpointer.__new__(CK.IdentityCheckpointer)
        ck._cfg = CFG
        return lambda box: ck._selection_problem(res, box, probe)

    def test_selection_precheck_rules(self):
        a = SS.box_at(400, 400, 40, H)
        far = SS.box_at(800, 400, 40, H)
        self.assertIsNone(self._problem([(a, "T1"), (far, "T2")])(a))
        # review #3: nested group carrying another track
        kid = SS.box_at(405, 420, 24, 50)
        why = self._problem([(a, "T1"), (kid, "T2")], groups={0: {0, 1}, 1: {0, 1}})(a)
        self.assertIn("other tracks", why)
        # matched detection without its own track (a part / alternative box)
        self.assertIn("without its own track", self._problem([(a, None)])(a))
        # merge lineage
        self.assertIn("merge lineage", self._problem([(a, "T1")], lineage=("T1",))(a))
        # another body well inside the selected box (outside the nested group)
        why = self._problem([(a, "T1"), (kid, "T2")])(a)
        self.assertIn("inside the box", why)
        # two detections matching the box
        a2 = (a[0] + 1, a[1], a[2] + 1, a[3])
        self.assertIn("more than one detection", self._problem([(a, "T1"), (a2, "T2")])(a))
        self.assertIn("no person detection", self._problem([])(a))

    def test_no_checkpoint_while_the_initial_seed_anchor_is_pending(self):
        # the initial seed touches the image edge (deferrable): its anchor is pending for a few frames
        cfg = TrackerConfig.with_overrides({"checkpoint_unresolved_s": 0.0})
        n = int(1.0 * FPS)
        fr = [SS.SceneFrame({"TARGET": SS.box_at(19 + 3 * k, 400, 40, H)}) for k in range(n)]
        r = run(fr, people(), Oracle(lambda f, d: None), cfg=cfg)
        pending = [k for k, tf in enumerate(r.out)
                   if k == 0 or (tf.decision.get("seed_anchor") or {}).get("status") == "PENDING"]
        self.assertGreater(len(pending), 1, "precondition: the seed anchor was deferred")
        for c in r.tracker.checkpoints:
            self.assertGreater(c["opened_frame"], max(pending))

    def test_reseed_event_evidence(self):
        fr, pp, n1 = persistent_look_change()
        r = run(fr, pp, Oracle(true_box(("TARGET", "TARGET_ALT"))))
        ev = [e for e in r.tracker.reacquisition_events if e.evidence.get("source") == CK.USER_RESEED][0]
        self.assertTrue(ev.evidence["contrast_verified"])
        self.assertIn("identity_basis", ev.evidence)
        self.assertLessEqual(ev.uncertain_from_frame, ev.reacq_started_frame)
        self.assertLessEqual(ev.reacq_started_frame, ev.reacq_accepted_frame)


class TestAuditGaps(unittest.TestCase):
    """Gaps found by the test audit (2026-11-05)."""

    def test_a_selection_that_would_defer_the_anchor_is_refused(self):
        # the true target re-appears touching the image edge: its crop cannot anchor a bank at once
        n1, n2 = int(1.5 * FPS), int(4.0 * FPS)
        fr = [SS.SceneFrame({"TARGET": SS.box_at(400, 400, 40, H)}) for _ in range(n1)]
        fr += [SS.SceneFrame({"TARGET_ALT": SS.box_at(19, 400, 40, H)}) for _ in range(n2)]
        pp = people(("TARGET_ALT", KIT_A, 0.7))
        r = run(fr, pp, Oracle(true_box(("TARGET_ALT",))))
        ans = [a for c in r.tracker.checkpoints for a in c["answers"] if a["decision"] == CK.SELECT]
        self.assertTrue(ans)
        self.assertTrue(all(a["outcome"] == "REFUSED" for a in ans), ans[:2])
        self.assertTrue(any("anchor" in a["reason"] or "refused" in a["reason"] for a in ans))
        self.assertEqual(r.wrong_trusted(("TARGET", "TARGET_ALT")), [])
        self.assertFalse(any(e.evidence.get("source") == CK.USER_RESEED for e in r.tracker.reacquisition_events))

    def test_a_resolver_that_raises_changes_nothing(self):
        fr, pp, n1 = persistent_look_change()

        class Boom:
            scene = None
            def __call__(self, view, res):
                raise RuntimeError("ui crashed")
        r = run(fr, pp, Boom())
        self.assertTrue(r.tracker.checkpoints[0]["answers"])
        self.assertTrue(all(a["outcome"] == "REFUSED" for a in r.tracker.checkpoints[0]["answers"]))
        self.assertEqual([strip(a) for a in run(fr, pp).out], [strip(b) for b in r.out])

    def test_episode_offsets_of_the_current_tracker(self):
        ck = CK.IdentityCheckpointer.__new__(CK.IdentityCheckpointer)
        ev = T.ReacquisitionEvent(1, 5, 4, 6, 7, 0.1, {})

        class Stub:
            reacquisition_events = (ev,)
        ck.tracker, ck._offset, ck._archived = Stub(), 2, [dataclasses.replace(ev, episode=2)]
        self.assertEqual([e.episode for e in ck.reacquisition_events], [2, 3])

    def test_false_suspension_recovers_at_15_fps(self):
        global FPS
        old = FPS
        try:
            FPS = 15.0
            fr, pp, n1 = persistent_look_change()
            r = run(fr, pp, Oracle(true_box(("TARGET", "TARGET_ALT"))), fps=15.0)
            cp = r.tracker.checkpoints[0]
            self.assertEqual(cp["status"], CK.RESEEDED)
            self.assertGreaterEqual(cp["opened_time_s"] - cp["unresolved_since_s"], WIN - 1e-9)
            self.assertEqual(r.wrong_trusted(("TARGET", "TARGET_ALT")), [])
            self.assertTrue(r.episodes_consistent())
        finally:
            FPS = old

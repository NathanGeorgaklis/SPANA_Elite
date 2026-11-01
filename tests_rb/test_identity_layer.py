# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2 DEV persistent football identity layer instruction (Nathan, 2026-11-01 21:39): persistent
#                 PERSON_IDs across tracklet breaks; ReID / team / number are supporting evidence, never proof alone;
#                 UNKNOWN remains valid; do not weaken identity_trusted; untrusted target frames never update trusted
#                 appearance history; test same-kit crossing, tracklet break/restart, re-entry after occlusion,
#                 different-team crossing, duplicate/unreadable jersey numbers.
"""Persistent identity layer (player_tracker_rb/identity.py) on synthetic scenes (neutral TARGET / OTHER)."""
import unittest

from player_tracker_rb import identity as I
from player_tracker_rb import population as P
from player_tracker_rb import target as T
from player_tracker_rb.appearance import AppearanceExtractor
from player_tracker_rb.camera import FIRST_FRAME
from player_tracker_rb.config import TrackerConfig
from player_tracker_rb.detector import PersonCandidate
from tests_rb import _synth_scene as SS

CFG = TrackerConfig.default()
KIT_A, KIT_B = (200, 60, 40), (40, 60, 200)
H = 100.0
FPS_SET = (30.0, 15.0)


def people(*others):
    return SS.make_people([("TARGET", KIT_A, 1.0)] + list(others))


class Run:
    """Population + the unchanged target tracker + the identity layer, frame by frame."""

    def __init__(self, frames, ppl, fps=30.0, reader=None, with_layer=True):
        cfg = CFG
        pop = P.Population(cfg, SS.W, SS.H)
        tt = T.TargetTracker(cfg, pop, AppearanceExtractor(cfg, SS.SignatureEmbedding(ppl, 0.0)))
        lay = I.PersonIdentityLayer(cfg, pop, tt._ext, number_reader=reader) if with_layer else None
        self.frames, self.ids, self.dets, self.pops = [], [], [], []
        self.tt, self.layer = tt, lay
        for k, fr in enumerate(frames):
            img = SS.render(fr, ppl, SS.W, SS.H)
            ds = SS.detect(fr, 0.35, SS.W, SS.H)
            cands = [PersonCandidate(d.box, 0.9, d.box[0] <= 2 or d.box[2] >= SS.W - 2, "synthetic") for d in ds]
            cam = SS.identity_motion(FIRST_FRAME) if k == 0 else \
                fr.camera if fr.camera is not None else SS.motion_from_matrix([[1, 0, 0], [0, 1, 0]], SS.W, SS.H)
            res = pop.update(k, k / fps, cands, cam)
            tf = tt.seed(res, img, fr.boxes["TARGET"]) if k == 0 else tt.update(res, img)
            if reader is not None:
                reader.frame = (fr, ds)
            if lay is not None:
                on = tf.observation_tracklet_id if tf.rb_state == T.TRACKING else None
                self.ids.append(lay.update(res, img, target_tracklet=on, target_trusted=tf.identity_trusted,
                                           target_bank=tt.bank))
            self.frames.append(tf)
            self.dets.append(ds)
            self.pops.append(res)

    def owner(self, k, tid):
        """The scene person a tracklet's observation at frame k belongs to, or None."""
        for o in self.pops[k].observations:
            if o.tracklet_id == tid:
                ow = self.dets[k][o.index].owners
                return ow[0] if len(ow) == 1 else None
        return None

    def person_owners(self):
        """person id -> set of scene people its tracklets were observed on (one element if the person is pure)."""
        out = {}
        for k, idf in enumerate(self.ids):
            for x in idf.tracklets:
                if x.person_id is None:
                    continue
                ow = self.owner(k, x.tracklet_id)
                if ow is not None:
                    out.setdefault(x.person_id, set()).add(ow)
        return out

    def links(self):
        return [d for idf in self.ids for d in idf.decisions]

    def check_safe(self, tc):
        # no person mixes two scene people (a wrong link), and the true target is never marked not-target
        for pid, ows in self.person_owners().items():
            tc.assertEqual(len(ows), 1, f"person {pid} mixes {ows}")
        for k, idf in enumerate(self.ids):
            for x in idf.tracklets:
                if self.owner(k, x.tracklet_id) == "TARGET":
                    tc.assertFalse(x.not_target_person, f"frame {k}: the true target is marked not-target")
        # the target person's history is the tracker's own (trusted) bank, nothing else
        for idf in self.ids[-1:]:
            if idf.target_person is not None:
                p = self.layer._persons[idf.target_person]
                tc.assertEqual(len(self.layer._history(p, self.tt.bank)), len(self.tt.bank.samples))


def status_of_owner(r, owner, after):
    """Link statuses of tracklets of `owner` born after frame `after`."""
    out = []
    for d in r.links():
        tid = d["tracklet"]
        born = next((k for k, idf in enumerate(r.ids) if idf.of(tid) is not None), None)
        if born is not None and born > after and r.owner(born, tid) == owner:
            out.append(d["status"])
    return out


class TestIdentityLayer(unittest.TestCase):
    def test_tracklet_break_and_restart_relinks_the_same_person(self):
        for fps in FPS_SET:
            with self.subTest(fps=fps):
                n0, gap = int(1.5 * fps), int(0.5 * fps)          # missed > population_max_gap_s: new tracklet
                fr = [SS.SceneFrame({"TARGET": SS.box_at(400, 400, 40, H), "OTHER": SS.box_at(900, 400, 40, H)},
                                    drop=("TARGET",) if n0 <= k < n0 + gap else ()) for k in range(int(4 * fps))]
                r = Run(fr, people(("OTHER", KIT_B, 0.2)), fps)
                tids = {x.tracklet_id for idf in r.ids for x in idf.tracklets if r.owner(0, x.tracklet_id) == "TARGET"}
                self.assertEqual(status_of_owner(r, "TARGET", n0), [I.LINKED])
                pids = {x.person_id for idf in r.ids[n0 + gap + int(fps):] for x in idf.tracklets
                        if r.owner(r.ids.index(idf), x.tracklet_id) == "TARGET"}
                first = r.ids[int(0.5 * fps)].of(next(iter(tids))).person_id
                self.assertEqual(pids, {first}, "the same PERSON_ID across the break")
                self.assertTrue(r.ids[-1].of(r.frames[-1].observation_tracklet_id).is_target_person)
                r.check_safe(self)

    def test_re_entry_after_occlusion_relinks_when_unambiguous(self):
        # the target walks behind a large static occluder of another kit and re-emerges on the other side
        for fps in FPS_SET:
            with self.subTest(fps=fps):
                fr = []
                for k in range(int(5 * fps)):
                    x = 300 + 60.0 * k / fps
                    fr.append(SS.SceneFrame({"TARGET": SS.box_at(x, 400, 40, H), "OTHER": SS.box_at(400, 410, 90, 120)},
                                            order=["TARGET", "OTHER"]))
                r = Run(fr, people(("OTHER", KIT_B, 0.2)), fps)
                st = status_of_owner(r, "TARGET", 0)
                self.assertTrue(st, "the target was re-detected after the occluder")
                self.assertIn(st[-1], (I.LINKED, I.UNKNOWN))
                r.check_safe(self)

    def test_different_team_crossing_relinks_both_without_mixing(self):
        for fps in FPS_SET:
            with self.subTest(fps=fps):
                fr = []
                for k in range(int(5 * fps)):
                    s = k / fps
                    fr.append(SS.SceneFrame({"TARGET": SS.box_at(300 + 80 * s, 400, 40, H),
                                             "OTHER": SS.box_at(700 - 80 * s, 400, 40, H)},
                                            order=["TARGET", "OTHER"] if s < 2.5 else ["OTHER", "TARGET"]))
                r = Run(fr, people(("OTHER", KIT_B, 0.2)), fps)
                r.check_safe(self)
                self.assertIn(I.LINKED, status_of_owner(r, "TARGET", 0) + status_of_owner(r, "OTHER", 0))

    def test_same_kit_crossing_never_links_the_wrong_person(self):
        # same kit; look-alike embeddings from clearly different (0.75) to near twins (0.97): UNKNOWN is fine, a
        # link to the other player never is
        for fps in FPS_SET:
            for cos in (0.75, 0.9, 0.97):
                with self.subTest(fps=fps, cos=cos):
                    fr = []
                    for k in range(int(5 * fps)):
                        s = k / fps
                        fr.append(SS.SceneFrame({"TARGET": SS.box_at(300 + 80 * s, 400, 40, H),
                                                 "OTHER": SS.box_at(700 - 80 * s, 400, 40, H)},
                                                order=["TARGET", "OTHER"] if s < 2.5 else ["OTHER", "TARGET"]))
                    r = Run(fr, people(("OTHER", KIT_A, cos)), fps)
                    r.check_safe(self)

    def test_twins_crossing_is_unknown_not_linked(self):
        fps = 30.0
        fr = []
        for k in range(int(5 * fps)):
            s = k / fps
            fr.append(SS.SceneFrame({"TARGET": SS.box_at(300 + 80 * s, 400, 40, H),
                                     "OTHER": SS.box_at(700 - 80 * s, 400, 40, H)},
                                    order=["TARGET", "OTHER"] if s < 2.5 else ["OTHER", "TARGET"]))
        r = Run(fr, people(("OTHER", KIT_A, 0.995)), fps)
        r.check_safe(self)
        after = [d for d in r.links() if d["scores"]]
        self.assertTrue(after)
        self.assertTrue(all(d["status"] != I.LINKED for d in after), [d["reason"] for d in after])

    def test_layer_never_changes_the_tracker(self):
        for fps in FPS_SET:
            with self.subTest(fps=fps):
                fr = []
                for k in range(int(4 * fps)):
                    s = k / fps
                    fr.append(SS.SceneFrame({"TARGET": SS.box_at(300 + 80 * s, 400, 40, H),
                                             "OTHER": SS.box_at(620 - 80 * s, 400, 40, H)},
                                            drop=("TARGET",) if 1.0 <= s < 1.5 else ()))
                a = Run(fr, people(("OTHER", KIT_A, 0.75)), fps, with_layer=True)
                b = Run(fr, people(("OTHER", KIT_A, 0.75)), fps, with_layer=False)
                self.assertEqual([f.as_record() for f in a.frames], [f.as_record() for f in b.frames])
                self.assertEqual(len(a.tt.bank.samples), len(b.tt.bank.samples))

    def test_not_target_persists_across_the_other_players_break(self):
        # OTHER is seen beside the trusted target, is missed long enough to restart its tracklet, and comes back:
        # its new tracklet is linked and stays NOT_TARGET; the target never is
        for fps in FPS_SET:
            with self.subTest(fps=fps):
                n0, gap = int(1.5 * fps), int(0.5 * fps)
                fr = [SS.SceneFrame({"TARGET": SS.box_at(400, 400, 40, H), "OTHER": SS.box_at(800, 400, 40, H)},
                                    drop=("OTHER",) if n0 <= k < n0 + gap else ()) for k in range(int(4 * fps))]
                r = Run(fr, people(("OTHER", KIT_B, 0.2)), fps)
                last = r.ids[-1]
                oth = [x for x in last.tracklets if r.owner(len(r.ids) - 1, x.tracklet_id) == "OTHER"]
                self.assertEqual(len(oth), 1)
                self.assertEqual(oth[0].status, I.LINKED)
                self.assertTrue(oth[0].not_target_person)
                r.check_safe(self)

    def test_unknown_when_evidence_is_too_short(self):
        # a body is seen for a few frames only, then never again: it is not forced into a link
        fps = 30.0
        fr = [SS.SceneFrame({"TARGET": SS.box_at(400, 400, 40, H)}) for _ in range(30)]
        fr += [SS.SceneFrame({"TARGET": SS.box_at(400, 400, 40, H), "OTHER": SS.box_at(900, 400, 40, H)})
               for _ in range(2)]
        fr += [SS.SceneFrame({"TARGET": SS.box_at(400, 400, 40, H)}) for _ in range(60)]
        r = Run(fr, people(("OTHER", KIT_A, 0.75)), fps)
        self.assertNotIn(I.LINKED, status_of_owner(r, "OTHER", 0))
        r.check_safe(self)

    def test_memory_is_bounded(self):
        # many short-lived passers-by over a long run: persons older than the link window are forgotten
        fps = 15.0
        fr = []
        for k in range(int(40 * fps)):
            boxes = {"TARGET": SS.box_at(400, 400, 40, H)}
            if (k // int(fps)) % 2 == 0:
                boxes["OTHER"] = SS.box_at(900, 400, 40, H)
            fr.append(SS.SceneFrame(boxes))
        r = Run(fr, people(("OTHER", KIT_B, 0.2)), fps)
        win = CFG.get("identity_link_window_s")
        self.assertLessEqual(len(r.layer._persons), 2 + int(win) + 2)
        r.check_safe(self)

    def test_a_candidate_is_not_forgotten_while_a_newborn_still_considers_it(self):
        # regression (DEV crash): a person last seen just inside the link window when a newborn is born passes the
        # window before the newborn's decision; it must stay available until that decision
        fps = 30.0
        win = CFG.get("identity_link_window_s")
        # the person's last time is its tracklet's closure (population_max_unmatched_s after its last observation):
        # the newborn appears 0.1 s before that person passes the window, and is decided >= identity_evidence_s later
        n_gone = int((win + CFG.get("population_max_unmatched_s") - 0.1) * fps)
        fr = [SS.SceneFrame({"TARGET": SS.box_at(400, 400, 40, H), "OTHER": SS.box_at(900, 400, 40, H)})
              for _ in range(30)]
        fr += [SS.SceneFrame({"TARGET": SS.box_at(400, 400, 40, H)}) for _ in range(n_gone)]
        fr += [SS.SceneFrame({"TARGET": SS.box_at(400, 400, 40, H), "OTHER": SS.box_at(905, 400, 40, H)})
               for _ in range(int(2 * fps))]
        r = Run(fr, people(("OTHER", KIT_B, 0.2)), fps)
        r.check_safe(self)


class FakeReader:
    """A jersey-number reader keyed on the synthetic scene: numbers[owner] -> number, None for unreadable."""

    def __init__(self, numbers):
        self.numbers = numbers
        self.frame = None

    def __call__(self, image, box):
        fr, ds = self.frame
        for d in ds:
            if SS.iou(d.box, box) > 0.9 and len(d.owners) == 1:
                return (self.numbers[d.owners[0]], 1.0) if self.numbers.get(d.owners[0]) is not None else None
        return None


class TestJerseyNumbers(unittest.TestCase):
    def _break_restart(self, fps, reader):
        n0, gap = int(1.5 * fps), int(0.5 * fps)
        fr = [SS.SceneFrame({"TARGET": SS.box_at(400, 400, 40, H), "OTHER": SS.box_at(900, 400, 40, H)},
                            drop=("TARGET",) if n0 <= k < n0 + gap else ()) for k in range(int(4 * fps))]
        return Run(fr, people(("OTHER", KIT_B, 0.2)), fps, reader=reader), n0

    def test_unreadable_numbers_change_nothing(self):
        r, n0 = self._break_restart(30.0, FakeReader({"TARGET": None, "OTHER": None}))
        self.assertEqual(status_of_owner(r, "TARGET", n0), [I.LINKED])
        r.check_safe(self)

    def test_duplicate_numbers_on_two_players_are_not_proof(self):
        # both players read the same number: it neither links them nor vetoes the correct link
        r, n0 = self._break_restart(30.0, FakeReader({"TARGET": "7", "OTHER": "7"}))
        self.assertEqual(status_of_owner(r, "TARGET", n0), [I.LINKED])
        r.check_safe(self)

    def test_confirmed_different_numbers_veto_a_link(self):
        # the reader reads a different number after the break (a different player in the same spot): no link
        class Switch(FakeReader):
            def __call__(self, image, box):
                fr, ds = self.frame
                for d in ds:
                    if SS.iou(d.box, box) > 0.9 and d.owners == ("TARGET",):
                        return ("9" if self.after else "7", 1.0)
                return None
        rd = Switch({})
        rd.after = False
        fps, n0, gap = 30.0, 45, 15
        fr = [SS.SceneFrame({"TARGET": SS.box_at(400, 400, 40, H)}, drop=("TARGET",) if n0 <= k < n0 + gap else ())
              for k in range(120)]

        class Frames(list):
            pass
        r = None
        # run manually so the reader can switch at the break
        cfg = CFG
        ppl = people()
        pop = P.Population(cfg, SS.W, SS.H)
        tt = T.TargetTracker(cfg, pop, AppearanceExtractor(cfg, SS.SignatureEmbedding(ppl, 0.0)))
        lay = I.PersonIdentityLayer(cfg, pop, tt._ext, number_reader=rd)
        decisions = []
        for k, f in enumerate(fr):
            rd.after = k >= n0
            img = SS.render(f, ppl, SS.W, SS.H)
            ds = SS.detect(f, 0.35, SS.W, SS.H)
            cands = [PersonCandidate(d.box, 0.9, False, "synthetic") for d in ds]
            cam = SS.identity_motion(FIRST_FRAME) if k == 0 else SS.motion_from_matrix([[1, 0, 0], [0, 1, 0]], SS.W, SS.H)
            res = pop.update(k, k / fps, cands, cam)
            tf = tt.seed(res, img, f.boxes["TARGET"]) if k == 0 else tt.update(res, img)
            rd.frame = (f, ds)
            idf = lay.update(res, img, target_tracklet=tf.observation_tracklet_id if tf.rb_state == T.TRACKING
                             else None, target_trusted=tf.identity_trusted, target_bank=tt.bank)
            decisions += [d for d in idf.decisions if d["scores"]]
        self.assertTrue(decisions)
        self.assertNotEqual(decisions[-1]["status"], I.LINKED)
        self.assertIn(I.VETO_NUMBER, decisions[-1]["scores"][0]["vetoes"])

    def test_conflicting_readings_on_one_tracklet_confirm_nothing(self):
        lay = I.PersonIdentityLayer(CFG, P.Population(CFG, SS.W, SS.H), None)
        self.assertIsNone(lay._confirmed_number({"7": 5, "1": 1}))
        self.assertIsNone(lay._confirmed_number({"7": CFG.get("identity_number_min_reads") - 1}))
        self.assertEqual(lay._confirmed_number({"7": CFG.get("identity_number_min_reads")}), "7")


if __name__ == "__main__":
    unittest.main()

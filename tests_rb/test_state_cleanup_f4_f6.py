# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2 DEV F4 + F6 instruction (Nathan, 2026-11-01 15:13): recover correctly after temporary
#                 competing-candidate ambiguity; remove one-frame trusted lag into unresolvable/occluded states; do not
#                 weaken appearance, contact, merge or reacquisition safety; test temporary competitor then clean
#                 separation, persistent competitor, partial/occluded transition, one-frame visibility loss, same kit.
"""F4 / F6 state-machine cleanup (synthetic, neutral TARGET / OTHER; no benchmark knowledge).

Failure reproduced: frame times are frame / fps, so a gap that is an exact multiple of the frame interval can exceed
the same limit by floating-point error. At 10 fps a single missed frame (a 0.2 s gap = population_max_gap_s) computed
as 0.20000000000000018 > 0.2: the population closed the target's tracklet, the return became a reacquisition and opened
an UNVERIFIED episode, so trust never came back (at 15 / 30 fps the same miss was harmless).
"""
import re
import os
import unittest

from player_tracker_rb import config as CF
from player_tracker_rb import target as T
from player_tracker_rb.appearance import covered_fraction
from player_tracker_rb.config import TrackerConfig
from tests_rb import _synth_scene as SS

CFG = TrackerConfig.default()
KIT_A, KIT_B = (200, 60, 40), (40, 60, 200)
SAME_KIT_COS = 0.75
H = 100.0


def people(*others):
    return SS.make_people([("TARGET", KIT_A, 1.0)] + list(others))


def wrong_trusted(r):
    return [k for k, f in enumerate(r.frames) if f.identity_trusted
            and r.dets[k][f.detection_index].owners != ("TARGET",)]


def bank_only_when_trusted(tc, r):
    for f in r.frames:
        if (f.decision.get("bank") or {}).get("status") == "ACCEPTED":
            tc.assertTrue(f.identity_trusted)


def crossing(fps, *, cos, kit, contact_s, overlap=25.0, walk_s=0.5, end=150.0, after_s=3.0):
    """OTHER walks up to the standing target, stays in contact (boxes overlapping, OTHER in front) for contact_s,
    walks clear and stays visible beside it. Returns (frames, first contact frame, first separated frame)."""
    tgt = SS.box_at(400, 400, 40, H)
    fr = [SS.SceneFrame({"TARGET": tgt, "OTHER": SS.box_at(700, 400, 40, H)}) for _ in range(int(fps))]
    n = int(0.5 * fps)
    for k in range(n):
        fr.append(SS.SceneFrame({"TARGET": tgt, "OTHER": SS.box_at(700 - (300 - overlap) * (k + 1) / n, 400, 40, H)},
                                order=["TARGET", "OTHER"]))
    c0 = len(fr)
    fr += [SS.SceneFrame({"TARGET": tgt, "OTHER": SS.box_at(400 + overlap, 400, 40, H)}, order=["TARGET", "OTHER"])
           for _ in range(int(contact_s * fps))]
    m = int(walk_s * fps)
    for k in range(m):
        fr.append(SS.SceneFrame({"TARGET": tgt, "OTHER": SS.box_at(400 + overlap + (end - overlap) * (k + 1) / m, 400,
                                                                   40, H)}, order=["TARGET", "OTHER"]))
    sep = len(fr)
    fr += [SS.SceneFrame({"TARGET": tgt, "OTHER": SS.box_at(400 + end, 400, 40, H)}) for _ in range(int(after_s * fps))]
    return fr, c0, sep


class TestTimeBoundaries(unittest.TestCase):
    def test_sanctioned_comparisons_treat_float_error_at_a_limit_as_equal(self):
        g = 16 / 10 - 14 / 10                                # one missed frame at 10 fps, as frame times compute it
        self.assertGreater(g, 0.2)                           # the raw comparison is wrong
        self.assertFalse(CF.sec_gt(g, 0.2))
        self.assertTrue(CF.sec_le(g, 0.2))
        self.assertTrue(CF.sec_ge(0.2 - 1e-12, 0.2))
        self.assertFalse(CF.sec_lt(0.2 - 1e-12, 0.2))
        # off the boundary nothing changes (one 60 fps frame is far above the tolerance)
        self.assertTrue(CF.sec_gt(0.2 + 1 / 60, 0.2))
        self.assertTrue(CF.sec_lt(0.2 - 1 / 60, 0.2))

    def test_no_raw_comparison_with_a_seconds_limit_in_the_package(self):
        pkg = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "player_tracker_rb")
        pat = re.compile(r'(?:[<>]=?)\s*(?:c|cfg|self\._cfg)\.get\("[a-z0-9_]+_s"\)|'
                         r'(?:c|cfg|self\._cfg)\.get\("[a-z0-9_]+_s"\)\s*(?:[<>]=?)')
        bad = []
        for name in sorted(os.listdir(pkg)):
            if name.endswith(".py"):
                with open(os.path.join(pkg, name)) as f:
                    for i, line in enumerate(f, 1):
                        if pat.search(line) and not re.search(r'_per_s"\)', line):     # rates, not durations
                            bad.append(f"{name}:{i}: {line.strip()}")
        self.assertEqual(bad, [], "compare durations with config.sec_gt / sec_ge / sec_lt / sec_le")

    def test_one_frame_visibility_loss_keeps_trust_at_any_fps(self):
        # FAILS ON 1ab094a at 10 fps: a single missed frame ended trust for good
        for fps in (10.0, 15.0, 25.0, 30.0, 60.0):
            with self.subTest(fps=fps):
                n, kk = int(3 * fps), int(1.5 * fps)
                fr = [SS.SceneFrame({"TARGET": SS.box_at(500, 400, 40, H)}, drop=("TARGET",) if k == kk else ())
                      for k in range(n)]
                r = SS.run(fr, people(), fps=fps)
                self.assertFalse(r.frames[kk].identity_trusted, "the missed frame itself is not observed")
                self.assertEqual(r.frames[kk].rb_state, T.OCCLUDED)
                self.assertTrue(all(f.identity_trusted for f in r.frames[kk + 1:]))
                self.assertEqual(r.frames[-1].target_episode, 1, "continuation, not a reacquisition")

    def test_gap_of_exactly_the_population_limit_continues_at_any_fps(self):
        # the target is missed for (population_max_gap_s * fps - 1) frames: the gap between its observations is
        # exactly population_max_gap_s, which "at most this long may continue" allows
        gap_s = CFG.get("population_max_gap_s")
        for fps in (10.0, 15.0, 25.0, 30.0, 60.0):
            with self.subTest(fps=fps):
                miss = round(gap_s * fps) - 1
                if miss < 1:
                    continue
                n, kk = int(3 * fps), int(1.5 * fps)
                fr = [SS.SceneFrame({"TARGET": SS.box_at(500, 400, 40, H)},
                                    drop=("TARGET",) if kk <= k < kk + miss else ()) for k in range(n)]
                r = SS.run(fr, people(), fps=fps)
                self.assertEqual(r.frames[-1].target_episode, 1)
                self.assertTrue(r.frames[-1].identity_trusted)


class TestTemporaryCompetitor(unittest.TestCase):
    def test_same_kit_competitor_then_clean_separation_recovers_trust(self):
        # (at 10 fps a 2.5 h/s walk-away breaks the population's frame-to-frame association: a known separate limit)
        for fps in (30.0, 15.0):
            for contact_s in (0.3, 1.5):
                with self.subTest(fps=fps, contact_s=contact_s):
                    fr, c0, sep = crossing(fps, cos=SAME_KIT_COS, kit=KIT_A, contact_s=contact_s)
                    r = SS.run(fr, people(("OTHER", KIT_A, SAME_KIT_COS)), fps=fps)
                    self.assertFalse(any(f.identity_trusted for f in r.frames[c0:sep - int(0.5 * fps)]),
                                     "untrusted while in contact")
                    after = [f.identity_trusted for f in r.frames[sep:]]
                    self.assertTrue(all(after[int(CFG.get("reacq_window_s") * fps) + 1:]),
                                    "trusted again once clearly separated")
                    self.assertEqual(wrong_trusted(r), [])
                    self.assertEqual(r.swaps(), [])
                    self.assertTrue(r.bank_is_pure())
                    bank_only_when_trusted(self, r)

    def test_persistent_competitor_stays_untrusted(self):
        for fps in (30.0, 15.0, 10.0):
            for cos, kit in ((SAME_KIT_COS, KIT_A), (0.2, KIT_B)):
                with self.subTest(fps=fps, cos=cos):
                    fr, c0, _ = crossing(fps, cos=cos, kit=kit, contact_s=5.0, after_s=0.0, walk_s=0.0)
                    r = SS.run(fr, people(("OTHER", kit, cos)), fps=fps)
                    self.assertFalse(any(f.identity_trusted for f in r.frames[c0:]))
                    self.assertEqual(wrong_trusted(r), [])
                    self.assertTrue(r.bank_is_pure())

    def test_same_kit_close_look_alike_crossing_is_never_trusted_wrongly(self):
        for fps in (30.0, 15.0, 10.0):
            for cos in (0.9, 0.97):
                with self.subTest(fps=fps, cos=cos):
                    fr, _, _ = crossing(fps, cos=cos, kit=KIT_A, contact_s=0.5)
                    r = SS.run(fr, people(("OTHER", KIT_A, cos)), fps=fps)
                    self.assertEqual(wrong_trusted(r), [])
                    self.assertEqual(r.swaps(), [])
                    self.assertTrue(r.bank_is_pure())
                    bank_only_when_trusted(self, r)

    def test_different_kit_crossing_is_never_trusted_wrongly(self):
        for fps in (30.0, 15.0, 10.0):
            with self.subTest(fps=fps):
                fr, _, _ = crossing(fps, cos=0.2, kit=KIT_B, contact_s=0.5)
                r = SS.run(fr, people(("OTHER", KIT_B, 0.2)), fps=fps)
                self.assertEqual(wrong_trusted(r), [])
                self.assertTrue(r.bank_is_pure())

    @unittest.expectedFailure
    def test_different_kit_crossing_then_separation_recovers_trust(self):
        # F4 RESIDUAL (not fixed, Nathan's decision): the return after the contact is a reacquisition; its episode is
        # verified only by beating a CONFUSABLE competitor (R1-G review L4). An obviously different player beaten by
        # a wide margin never verifies it, so trust does not come back while no look-alike is near.
        fps = 30.0
        fr, _, sep = crossing(fps, cos=0.2, kit=KIT_B, contact_s=0.5)
        r = SS.run(fr, people(("OTHER", KIT_B, 0.2)), fps=fps)
        self.assertTrue(r.frames[-1].identity_trusted)


class TestOcclusionTransition(unittest.TestCase):
    def test_trust_drops_on_the_same_frame_the_crop_becomes_occluded(self):
        # F6: an occluder walks across in front of the standing target; no frame is trusted while the target's crop
        # is covered beyond crop_max_covered_fraction (no one-frame lag in the state machine)
        thr = CFG.get("crop_max_covered_fraction")
        for fps in (30.0, 15.0, 10.0):
            for cos, kit in ((0.2, KIT_B), (SAME_KIT_COS, KIT_A)):
                with self.subTest(fps=fps, cos=cos):
                    speed = 150.0
                    tgt = SS.box_at(500, 400, 40, H)
                    fr = [SS.SceneFrame({"TARGET": tgt, "OTHER": SS.box_at(200 + speed * k / fps, 410, 40, H)},
                                        order=["TARGET", "OTHER"]) for k in range(int(4 * fps))]
                    r = SS.run(fr, people(("OTHER", kit, cos)), fps=fps)
                    covered = [k for k in range(len(fr))
                               if covered_fraction(tgt, [SS.box_at(200 + speed * k / fps, 410, 40, H)]) > thr]
                    self.assertTrue(covered)
                    self.assertEqual([k for k in covered if r.frames[k].identity_trusted], [])
                    self.assertEqual(wrong_trusted(r), [])
                    self.assertTrue(r.bank_is_pure())

    def test_partial_target_behind_an_occluder_then_clear(self):
        # the target is partly hidden (an occluder covers its lower body) for 1 s, then the occluder leaves
        thr = CFG.get("crop_max_covered_fraction")
        for fps in (30.0, 15.0):
            with self.subTest(fps=fps):
                tgt = SS.box_at(500, 400, 40, H)
                occ = SS.box_at(500, 430, 50, 60)
                n0, n1, n2 = int(fps), int(fps), int(2 * fps)
                fr = [SS.SceneFrame({"TARGET": tgt}) for _ in range(n0)]
                fr += [SS.SceneFrame({"TARGET": tgt, "OTHER": occ}, order=["TARGET", "OTHER"]) for _ in range(n1)]
                fr += [SS.SceneFrame({"TARGET": tgt, "OTHER": SS.box_at(900, 400, 40, H)}) for _ in range(n2)]
                r = SS.run(fr, people(("OTHER", KIT_B, 0.2)), fps=fps)
                self.assertGreater(covered_fraction(tgt, [occ]), thr)
                self.assertFalse(any(f.identity_trusted for f in r.frames[n0:n0 + n1]))
                self.assertEqual(wrong_trusted(r), [])
                self.assertTrue(r.bank_is_pure())
                bank_only_when_trusted(self, r)


if __name__ == "__main__":
    unittest.main()

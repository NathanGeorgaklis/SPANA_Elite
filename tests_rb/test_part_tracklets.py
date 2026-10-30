# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2 DEV remediation instruction (Nathan, 2026-10-31 05:05): F3b duplicate same-body tracklets. A
#                 nested / alternative / partial detection of an already represented body must not become an
#                 independent person unless there is evidence it is a second person; genuine overlapping people stay
#                 separate; same-kit people are not collapsed; a real second person partly behind stays representable;
#                 provisional competitors do not disappear for overlapping; ambiguity fails closed.
"""F3b regression tests (synthetic, neutral TARGET / PERSON_A / OTHER; no benchmark knowledge).

Failure class reproduced (R2 DEV F3b): a detector returns, for ONE body, its main box plus a part box (here the upper
body, sticking out above a main box that lost the head) that is substantially inside the main box but below the
NESTED containment. The part box starts its own tracklet. When the part box is no longer detected, its hidden tracklet
and the body's own tracklet both cover the body's next detection: MULTI_CLAIM counted two physically distinct bodies,
the body's tracklet was closed as a merge and a merged region (and lineage) followed.

Rule under test: a tracklet born substantially inside a body observed in the same frame has UNPROVEN distinctness from
it until the two are seen as clearly separate boxes in one frame. An unproven pair never counts as two bodies (no
MULTI_CLAIM between them, no size corroboration of one by the other). Both remain tracklets and competitors.
"""
import math
import unittest

from player_tracker_rb import population as P
from player_tracker_rb.config import TrackerConfig
from tests_rb import _synth_scene as SS
from tests_rb._synth_people import Frame, box_at, cand, run

CFG = TrackerConfig.default()
FPS = 30.0
FULL = box_at(400, 360, 32, 80)                 # (384, 320, 416, 400)
CLIPPED = (384.0, 330.0, 416.0, 400.0)          # the same body's box without the head
UPPER = (386.0, 312.0, 414.0, 356.0)            # its upper-body box, sticking out above CLIPPED


def part_overlap(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    sa = (a[2] - a[0]) * (a[3] - a[1])
    sb = (b[2] - b[0]) * (b[3] - b[1])
    return ix * iy / min(sa, sb)


def frames_full(n):
    return [Frame([cand(FULL)], [("TARGET",)]) for _ in range(n)]


def no_merge_evidence(tc, rn, start=0):
    for r in rn.results[start:]:
        for o in r.observations:
            tc.assertNotIn(P.MERGED_SUSPECTED, o.flags, f"frame {r.rel_frame}: {o.merge_reasons}")
            tc.assertNotIn(P.IN_MERGED_REGION, o.flags, f"frame {r.rel_frame}")
        for v in r.tracklets:
            tc.assertNotIn(P.MERGE_LINEAGE, v.flags, f"frame {r.rel_frame}")
        for cl in r.closed:
            tc.assertNotEqual(cl.reason, P.MERGED_CONTACT, f"frame {r.rel_frame}: {cl.tracklet_id}")


def owner_tracklets(rn, person):
    return {tid for tid, own in rn.tracklet_owners().items() if person in own}


class TestGeometry(unittest.TestCase):
    def test_fixture_geometry_is_the_failure_class(self):
        # substantially inside, but neither NESTED nor a duplicate
        c = part_overlap(UPPER, CLIPPED)
        self.assertGreaterEqual(c, CFG.get("population_part_birth_containment"))
        self.assertLess(c, CFG.get("population_nested_containment"))


class TestSameBodyPartBoxes(unittest.TestCase):
    def test_full_and_part_box_of_one_person_never_merge(self):
        # frame 30: main box loses the head and an upper-body box is detected; afterwards only the full box
        fr = frames_full(30) + [Frame([cand(CLIPPED), cand(UPPER)], [("TARGET",), ("TARGET",)])] + frames_full(30)
        rn = run(fr, fps=FPS)
        no_merge_evidence(self, rn)
        main = {r.observations[0].tracklet_id for r in rn.results if r.observations[0].bbox != UPPER}
        self.assertEqual(len(main), 1, "the body's own tracklet is never closed by its part box")
        born = rn.results[30].tracklet(rn.results[30].observations[1].tracklet_id)
        self.assertIsNotNone(born, "the part box is still represented (a competitor), not deleted")
        self.assertEqual(born.unproven_distinct_from, (rn.results[30].observations[0].tracklet_id,))

    def test_nested_alternatives_over_many_frames_never_merge(self):
        # 2 s of a body whose detector alternates: full box / main box plus a part box (upper body, or lower legs)
        # lower-leg box under a foot-clipped main box (sticks out below it; inside the full body)
        foot_clipped = (384.0, 320.0, 416.0, 388.0)
        lower = (386.0, 360.0, 414.0, 404.0)
        self.assertGreaterEqual(part_overlap(lower, foot_clipped), CFG.get("population_part_birth_containment"))
        self.assertLess(part_overlap(lower, foot_clipped), CFG.get("population_nested_containment"))
        for part, main in ((UPPER, CLIPPED), (lower, foot_clipped)):
            # KNOWN LIMIT (recorded): if the body's own box is clipped on most frames its established height shrinks
            # and a part box may no longer fit the body extent: the convergence is then a conservative merge
            fr = frames_full(30)
            for k in range(60):
                fr.append(Frame([cand(main), cand(part)], [("TARGET",), ("TARGET",)]) if k % 3 == 0
                          else Frame([cand(FULL)], [("TARGET",)]))
            rn = run(fr, fps=FPS)
            no_merge_evidence(self, rn)
            mains = {r.observations[0].tracklet_id for r in rn.results}
            self.assertEqual(len(mains), 1, part)

    def test_full_height_alternative_box_keeps_the_conservative_handling(self):
        # KNOWN LIMIT (recorded): a shifted alternative box as tall as the body is geometrically a second person
        # beside it; it is treated as one (review round 1), so a convergence may still be a MULTI_CLAIM merge.
        # Safety holds: no tracklet is ever observed on two people.
        alt = (396.0, 322.0, 434.0, 398.0)
        fr = frames_full(30)
        for k in range(60):
            fr.append(Frame([cand(FULL), cand(alt)], [("TARGET",), ("TARGET",)]) if k % 2 == 0
                      else Frame([cand(FULL)], [("TARGET",)]))
        rn = run(fr, fps=FPS)
        self.assertEqual(rn.results[30].tracklet(rn.results[30].observations[1].tracklet_id).unproven_distinct_from,
                         ())

    def test_duplicate_tracklets_converging_on_one_detection(self):
        # both the body's tracklet and its part tracklet are hidden, then one detection covers both: one body,
        # not a merge; the detection's tracklet records both as possible predecessors and carries no lineage
        fr = frames_full(30) + [Frame([cand(CLIPPED), cand(UPPER)], [("TARGET",), ("TARGET",)])]
        fr += [Frame([], []) for _ in range(2)] + frames_full(20)
        rn = run(fr, fps=FPS)
        no_merge_evidence(self, rn)
        r = rn.results[33]
        self.assertNotIn(P.MULTI_CLAIM, r.observations[0].merge_reasons)
        host, part = (rn.results[30].observations[k].tracklet_id for k in (0, 1))
        self.assertEqual(set(r.observations[0].possible_predecessors), {host, part},
                         "no identity continues: the new tracklet records both as possible predecessors")

    def test_part_box_does_not_corroborate_its_hosts_size_change(self):
        # the body's own pose widening while its unproven part tracklet is still live (hidden, no longer a recent
        # claimant) and touching it: a SIZE_ANOMALY (abstain), not a merge
        fr = frames_full(30) + [Frame([cand(CLIPPED), cand(UPPER)], [("TARGET",), ("TARGET",)])]
        fr += frames_full(int(math.ceil(CFG.get("population_max_gap_s") * FPS)) + 2)
        for k in range(5):
            w = 32 + 40 * (k + 1) / 5
            fr.append(Frame([cand(box_at(400, 360, w, 80))], [("TARGET",)]))
        rn = run(fr, fps=FPS)
        part = rn.results[30].observations[1].tracklet_id
        self.assertIsNotNone(rn.results[-1].tracklet(part), "the part tracklet is still live")
        codes = ["M" if P.MERGED_SUSPECTED in r.observations[0].flags else
                 ("A" if P.SIZE_ANOMALY in r.observations[0].flags else ".") for r in rn.results[31:]]
        self.assertIn("A", codes)
        self.assertNotIn("M", codes)


class TestGenuineSecondPeople(unittest.TestCase):
    def _pair(self, a_boxes, b_boxes, drop_b=()):
        fr = []
        for k, (a, b) in enumerate(zip(a_boxes, b_boxes)):
            if b is None or k in drop_b:
                fr.append(Frame([cand(a)], [("TARGET",)]))
            else:
                fr.append(Frame([cand(a), cand(b)], [("TARGET",), ("PERSON_A",)]))
        return run(fr, fps=FPS)

    def test_overlapping_people_stay_separate_tracklets(self):
        # seen apart, then side by side overlapping up to 0.6 of a body: both always detected, two tracklets,
        # no tracklet ever holds both people, no merge
        a = [FULL] * 90
        b = [box_at(445, 360, 32, 80)] * 30 + [box_at(445 - 32 * (k + 1) / 30, 360, 32, 80) for k in range(30)]
        b += [box_at(413, 360, 32, 80)] * 30
        self.assertGreaterEqual(part_overlap(b[-1], FULL), 0.55)
        rn = self._pair(a, b)
        no_merge_evidence(self, rn)
        for tid, own in rn.tracklet_owners().items():
            self.assertEqual(len(own), 1, f"{tid} holds {own}")
        self.assertEqual(len(owner_tracklets(rn, "TARGET")), 1)
        self.assertEqual(len(owner_tracklets(rn, "PERSON_A")), 1)

    def test_full_height_person_first_seen_partly_behind_stays_a_person(self):
        # PERSON_A is first detected 0.6 inside the target's box at full height: a person, not an unproven part; it
        # walks out beside the target and a later convergence into one box is a merge (conservative)
        a = [FULL] * 120
        b = [None] * 30 + [box_at(413 + 1.5 * k, 360, 32, 80) for k in range(30)]   # walks out to x=456
        b += [box_at(456, 360, 32, 80)] * 20
        b += [None] * 40
        rn = self._pair(a, b)
        tb = rn.results[30].observations[1].tracklet_id
        # full height: a person from its first frame, never an unproven part (review round 1)
        self.assertEqual(rn.results[30].tracklet(tb).unproven_distinct_from, ())
        # now the two converge into one union box
        fr = [Frame([cand(x), cand(y)], [("TARGET",), ("PERSON_A",)]) if y else Frame([cand(x)], [("TARGET",)])
              for x, y in zip(a[:80], b[:80])]
        fr += [Frame([cand((384.0, 320.0, 472.0, 400.0))], [("TARGET", "PERSON_A")]) for _ in range(15)]
        rn2 = run(fr, fps=FPS)
        self.assertTrue(any(P.MULTI_CLAIM in r.observations[0].merge_reasons for r in rn2.results[80:]),
                        "two people seen apart that converge into one box are a merge")

    def test_short_partner_is_proven_distinct_when_seen_apart_or_seen_at_full_height(self):
        # a short box born inside the body (unproven) that then moves clearly apart, or grows as tall as a whole
        # body, is a person: the pair is no longer unproven
        apart = [(386.0 + 6 * k, 312.0, 414.0 + 6 * k, 356.0) for k in range(1, 6)]    # ends 0.0 inside
        grows = [(386.0, 312.0, 414.0, 356.0 + 10 * k) for k in range(1, 4)]           # ends 74 px of 80
        for seq in (apart, grows):
            fr = frames_full(30) + [Frame([cand(CLIPPED), cand(UPPER)], [("TARGET",), ("PERSON_A",)])]
            fr += [Frame([cand(FULL), cand(b)], [("TARGET",), ("PERSON_A",)]) for b in seq]
            rn = run(fr, fps=FPS)
            tb = rn.results[30].observations[1].tracklet_id
            self.assertEqual(rn.results[30].tracklet(tb).unproven_distinct_from,
                             (rn.results[30].observations[0].tracklet_id,))
            self.assertEqual(rn.results[-1].observations[1].tracklet_id, tb, "the same tracklet followed it")
            self.assertEqual(rn.results[-1].tracklet(tb).unproven_distinct_from, ())

    def test_younger_partner_stepping_in_front_of_its_host_is_never_a_silent_swap(self):
        # PERSON_A is first seen 0.6 inside the target's box (unproven), stays there, then moves in front of the
        # target (the target is no longer detected) and walks away: no tracklet ever holds both people
        tgt = FULL
        fr = [Frame([cand(tgt)], [("TARGET",)]) for _ in range(30)]
        a0 = box_at(413, 362, 33, 82)
        self.assertGreaterEqual(part_overlap(a0, tgt), CFG.get("population_part_birth_containment"))
        fr += [Frame([cand(tgt), cand(a0)], [("TARGET",), ("PERSON_A",)]) for _ in range(10)]
        for k in range(6):                                      # moves over the target, which disappears behind it
            x = 413 - 13 * (k + 1) / 6
            fr.append(Frame([cand(box_at(x, 362, 33, 82))], [("PERSON_A",)]))
        for k in range(30):                                     # walks away; the target stays hidden
            fr.append(Frame([cand(box_at(400 + 2 * (k + 1), 362, 33, 82))], [("PERSON_A",)]))
        rn = run(fr, fps=FPS)
        for tid, own in rn.tracklet_owners().items():
            self.assertEqual(len(own), 1, f"{tid} holds {own}")

    def test_younger_partner_taking_its_hosts_detection_is_still_a_merge(self):
        # the host (older) is hidden and the younger unproven tracklet continues on the one detection the host also
        # claims: conservative MULTI_CLAIM, as before F3b
        tgt = FULL
        a0 = box_at(413, 360, 32, 80)
        fr = [Frame([cand(tgt)], [("TARGET",)]) for _ in range(30)]
        fr += [Frame([cand(tgt), cand(a0)], [("TARGET",), ("PERSON_A",)]) for _ in range(5)]
        fr += [Frame([cand(box_at(413 - 13 * (k + 1) / 3, 360, 32, 80))], [("TARGET", "PERSON_A")])
               for k in range(3)]
        rn = run(fr, fps=FPS)
        self.assertTrue(any(P.MULTI_CLAIM in r.observations[0].merge_reasons for r in rn.results[35:]))

    def test_real_two_person_merge_keeps_conservative_handling(self):
        # two people seen apart, walking into one box: MULTI_CLAIM, claimants closed, merged region, lineage
        fr = [Frame([cand(FULL), cand(box_at(445, 360, 32, 80))], [("TARGET",), ("PERSON_A",)]) for _ in range(30)]
        fr += [Frame([cand((384.0, 320.0, 461.0, 400.0))], [("TARGET", "PERSON_A")]) for _ in range(20)]
        rn = run(fr, fps=FPS)
        self.assertTrue(any(P.MULTI_CLAIM in o.merge_reasons for r in rn.results for o in r.observations))
        self.assertIn(P.MERGED_CONTACT, set(rn.closed().values()))
        last = rn.results[-1].observations[0]
        self.assertIn(P.MERGED_SUSPECTED, last.flags)
        self.assertIn(P.IN_MERGED_REGION, last.flags)
        self.assertIsNone(last.tracklet_id, "a merged box is never a tracklet")

    def test_near_far_occlusion(self):
        # a far, smaller person passes behind a near one: hidden while covered, re-detected after; never a merge,
        # and no tracklet holds both people. Also when the far person is first seen while partly behind.
        near = box_at(400, 360, 32, 80)
        for start_x in (360.0, 392.0):
            fr = []
            for k in range(90):
                x = start_x + 1.0 * k
                far = box_at(x, 330, 16, 40)
                hidden = 392 <= x <= 408
                fr.append(Frame([cand(near)], [("TARGET",)]) if hidden else
                          Frame([cand(near), cand(far)], [("TARGET",), ("PERSON_A",)]))
            rn = run(fr, fps=FPS)
            no_merge_evidence(self, rn)
            for tid, own in rn.tracklet_owners().items():
                self.assertEqual(len(own), 1, f"{start_x}: {tid} holds {own}")
            self.assertEqual(len(owner_tracklets(rn, "TARGET")), 1, start_x)

    def test_partial_player_entering_beside_another(self):
        # PERSON_A enters from the image edge, first seen as a partial edge box about 0.6 inside the target's box
        # (unproven), passes behind the target (not detected while mostly covered) and comes out on the other side:
        # its own tracklet(s) throughout, the target's tracklet unbroken, no merge, and proven distinct at the end
        tgt = box_at(26, 360, 32, 80)                           # (10, 320, 42, 400)
        first = (0.0, 320.0, 24.0, 400.0)
        self.assertGreaterEqual(part_overlap(first, tgt), CFG.get("population_part_birth_containment"))
        self.assertLess(part_overlap(first, tgt), CFG.get("population_nested_containment"))
        fr = [Frame([cand(tgt)], [("TARGET",)]) for _ in range(30)]
        for k in range(90):
            right = 24 + 1.0 * k
            b = (max(0.0, right - 32), 320.0, right, 400.0)
            behind = part_overlap(b, tgt) >= CFG.get("population_nested_containment")
            fr.append(Frame([cand(tgt)], [("TARGET",)]) if behind else
                      Frame([cand(tgt), cand(b)], [("TARGET",), ("PERSON_A",)]))
        rn = run(fr, fps=FPS)
        # a full-height entering player is a person from its first frame (never an unproven part): the old,
        # conservative handling applies while it is behind the target (merges allowed), never a silent swap
        self.assertEqual(rn.results[30].tracklet(rn.results[30].observations[1].tracklet_id).unproven_distinct_from,
                         ())
        for tid, own in rn.tracklet_owners().items():
            self.assertEqual(len(own), 1, f"{tid} holds {own}")
        last = rn.results[-1]
        tb = last.observations[1].tracklet_id
        self.assertIsNotNone(tb, "the entering player is represented")
        self.assertEqual(last.tracklet(tb).unproven_distinct_from, ())


class TestTargetLevel(unittest.TestCase):
    PPL = SS.make_people([("TARGET", (200, 60, 40), 1.0), ("OTHER", (200, 60, 40), 0.95)])

    def test_target_keeps_tracking_through_its_own_part_boxes(self):
        tb = SS.box_at(300, 400, 40, 100)                      # (280, 350, 320, 450)
        clipped = (280.0, 362.0, 320.0, 450.0)
        upper = (282.0, 340.0, 318.0, 395.0)
        fr = [SS.SceneFrame({"TARGET": tb}) for _ in range(30)]
        for k in range(60):
            if k % 3 == 0:
                fr.append(SS.SceneFrame({"TARGET": clipped}, parts=(("TARGET", upper),)))
            else:
                fr.append(SS.SceneFrame({"TARGET": tb}))
        r = SS.run(fr, self.PPL)
        self.assertEqual(r.swaps(), [])
        self.assertTrue(r.bank_is_pure())
        self.assertEqual({f.target_episode for f in r.frames}, {1}, "never broken by its own part boxes")
        self.assertNotIn("LOST", r.states())
        # it may abstain on a frame where its part box is detected (a part box not wholly inside its box is a
        # possible person in contact), and is followed on every other frame
        for k in range(30, 90):
            if (k - 30) % 3 != 0:
                self.assertEqual(r.states()[k], "TRACKING", k)
        for p in r.pops:
            for o in p.observations:
                self.assertNotIn(P.MERGED_SUSPECTED, o.flags)

    def test_same_kit_person_first_seen_behind_the_target_is_never_trusted_as_it(self):
        # OTHER (same kit) is first detected mostly behind the target (unproven distinct), then steps fully in front
        # of it and walks off with the target hidden: no trusted frame on OTHER, bank pure
        tb = SS.box_at(300, 400, 40, 100)
        fr = [SS.SceneFrame({"TARGET": tb}) for _ in range(30)]
        fr += [SS.SceneFrame({"TARGET": tb, "OTHER": SS.box_at(318, 396, 40, 100)}, order=("OTHER", "TARGET"))
               for _ in range(10)]
        fr += [SS.SceneFrame({"TARGET": tb, "OTHER": SS.box_at(318 - 18 * (k + 1) / 6, 401, 41, 101)},
                             order=("TARGET", "OTHER")) for k in range(6)]
        fr += [SS.SceneFrame({"TARGET": tb, "OTHER": SS.box_at(300 + 3 * (k + 1), 401, 41, 101)},
                             order=("TARGET", "OTHER"), drop=("TARGET",)) for k in range(40)]
        r = SS.run(fr, self.PPL)
        bad = [k for k, (f, ds) in enumerate(zip(r.frames, r.dets))
               if f.identity_trusted and ds[f.detection_index].owners != ("TARGET",)]
        self.assertEqual(bad, [])
        self.assertTrue(r.bank_is_pure())



class TestReviewRound1(unittest.TestCase):
    """Independent review of bc1606d: a real person first seen overlapping the target (unproven) who steps in front of
    it quickly, or who forms one union box with it, must keep the conservative handling (both failed on bc1606d)."""
    TGT = box_at(400, 360, 32, 80)

    @staticmethod
    def _cover(a, b):
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        return ix * iy / ((a[2] - a[0]) * (a[3] - a[1]))

    def test_full_height_person_stepping_in_front_is_never_a_silent_swap(self):
        for fps in (30.0, 15.0, 10.0):
            for v_hs in (1.0, 2.0, 3.0):                       # body heights per second
                for hide in (0.75, 0.9):
                    v = v_hs * 80 / fps
                    fr = [Frame([cand(self.TGT)], [("TARGET",)]) for _ in range(int(fps))]
                    x = 412.0
                    fr += [Frame([cand(self.TGT), cand(box_at(x, 360, 32, 80))], [("TARGET",), ("PERSON_A",)])
                           for _ in range(int(fps / 3))]
                    while x > 401:
                        x = max(400.0, x - v)
                        b = box_at(x, 360, 32, 80)
                        fr.append(Frame([cand(b)], [("PERSON_A",)]) if self._cover(self.TGT, b) >= hide else
                                  Frame([cand(self.TGT), cand(b)], [("TARGET",), ("PERSON_A",)]))
                    fr += [Frame([cand(box_at(x, 360, 32, 80))], [("PERSON_A",)]) for _ in range(int(fps / 2))]
                    fr += [Frame([cand(box_at(x - v * (k + 1), 360, 32, 80))], [("PERSON_A",)])
                           for k in range(int(fps))]
                    rn = run(fr, fps=fps)
                    bad = {t: o for t, o in rn.tracklet_owners().items() if len(o) > 1}
                    self.assertEqual(bad, {}, f"fps={fps} v={v_hs} hide={hide}")

    def test_union_box_with_a_full_height_unproven_partner_is_a_merge(self):
        for off in (12, 14):
            fr = [Frame([cand(self.TGT)], [("TARGET",)]) for _ in range(30)]
            fr += [Frame([cand(self.TGT), cand(box_at(400 + off, 360, 32, 80))], [("TARGET",), ("PERSON_A",)])
                   for _ in range(10)]
            fr += [Frame([cand((384.0, 320.0, 416.0 + off, 400.0))], [("TARGET", "PERSON_A")]) for _ in range(10)]
            fr += [Frame([cand(self.TGT), cand(box_at(400 + off + 1.5 * (k + 1), 360, 32, 80))],
                         [("TARGET",), ("PERSON_A",)]) for k in range(30)]
            rn = run(fr, fps=FPS)
            self.assertEqual({t: o for t, o in rn.tracklet_owners().items() if len(o) > 1}, {}, off)
            self.assertTrue(all(P.MERGED_SUSPECTED in r.observations[0].flags for r in rn.results[40:50]), off)

    def test_same_kit_person_stepping_in_front_fast_is_never_trusted(self):
        ppl = SS.make_people([("TARGET", (200, 60, 40), 1.0), ("OTHER", (200, 60, 40), 0.95)])
        for fps in (30.0, 15.0, 10.0):
            for v_hs in (2.0, 3.0):
                v = v_hs * 100 / fps
                tb = SS.box_at(300, 400, 40, 100)
                fr = [SS.SceneFrame({"TARGET": tb}) for _ in range(int(2 * fps))]
                x = 315.0
                fr += [SS.SceneFrame({"TARGET": tb, "OTHER": SS.box_at(x, 400, 40, 100)}, order=("OTHER", "TARGET"))
                       for _ in range(int(fps / 3))]
                while x > 300.5:
                    x = max(300.0, x - v)
                    fr.append(SS.SceneFrame({"TARGET": tb, "OTHER": SS.box_at(x, 400, 40, 100)},
                                            order=("TARGET", "OTHER")))
                fr += [SS.SceneFrame({"TARGET": tb, "OTHER": SS.box_at(x, 400, 40, 100)}, order=("TARGET", "OTHER"))
                       for _ in range(int(fps / 2))]
                fr += [SS.SceneFrame({"TARGET": tb, "OTHER": SS.box_at(x - v * (k + 1), 400, 40, 100)},
                                     order=("TARGET", "OTHER"), drop=("TARGET",)) for k in range(int(2 * fps))]
                r = SS.run(fr, ppl, fps=fps)
                bad = [k for k, (f, ds) in enumerate(zip(r.frames, r.dets))
                       if f.identity_trusted and f.detection_index is not None
                       and ds[f.detection_index].owners != ("TARGET",)]
                self.assertEqual(bad, [], f"fps={fps} v={v_hs}")
                self.assertTrue(r.bank_is_pure(), f"fps={fps} v={v_hs}")


class TestReviewRound2(unittest.TestCase):
    """Independent review of e351504: a SHORT real person (crouching, farther, short detector box) first seen inside
    the target's box, who then steps in front of it or forms one union box with it, and a partner born while the
    target's own box is momentarily tall, must keep the conservative handling (all failed on e351504)."""
    TGT = box_at(400, 360, 32, 80)

    @staticmethod
    def _cover(a, b):
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        return ix * iy / ((a[2] - a[0]) * (a[3] - a[1]))

    def test_short_person_stepping_in_front_is_never_a_silent_swap(self):
        for fps in (30.0, 15.0, 10.0):
            for v_hs in (1.0, 2.0, 3.0):
                for hr in (0.6, 0.7):
                    for pose in ("crouch", "far"):
                        for hide in ("miss", "cover"):
                            bh, bw = 80 * hr, 32 * (1.0 if pose == "crouch" else hr)
                            cy = 400 - bh / 2 if pose == "crouch" else 350.0
                            v = v_hs * 80 / fps
                            x = 412.0

                            def b_at(x_):
                                return box_at(x_, cy, bw, bh)
                            fr = [Frame([cand(self.TGT)], [("TARGET",)]) for _ in range(int(fps))]
                            fr += [Frame([cand(self.TGT), cand(b_at(x))], [("TARGET",), ("PERSON_A",)])
                                   for _ in range(int(fps / 3))]
                            while x > 400.5:
                                x = max(400.0, x - v)
                                gone = (hide == "miss" and x <= 404) or \
                                    (hide == "cover" and pose == "crouch" and self._cover(self.TGT, b_at(x)) > 0)
                                fr.append(Frame([cand(b_at(x))], [("PERSON_A",)]) if gone else
                                          Frame([cand(self.TGT), cand(b_at(x))], [("TARGET",), ("PERSON_A",)]))
                            fr += [Frame([cand(b_at(x))], [("PERSON_A",)]) for _ in range(int(fps / 2))]
                            fr += [Frame([cand(b_at(x - v * (k + 1)))], [("PERSON_A",)]) for k in range(int(fps))]
                            rn = run(fr, fps=fps)
                            bad = {t: o for t, o in rn.tracklet_owners().items() if len(o) > 1}
                            self.assertEqual(bad, {}, f"fps={fps} v={v_hs} hr={hr} {pose} {hide}")

    def test_same_kit_short_person_stepping_in_front_is_never_trusted(self):
        ppl = SS.make_people([("TARGET", (200, 60, 40), 1.0), ("OTHER", (200, 60, 40), 0.95)])
        for fps in (15.0, 10.0):
            for v_hs in (2.0, 3.0):
                for hr in (0.6, 0.7):
                    v = v_hs * 100 / fps
                    tb = SS.box_at(300, 400, 40, 100)
                    oh = 100 * hr

                    def ob(x_):
                        return SS.box_at(x_, 450 - oh / 2, 40, oh)
                    fr = [SS.SceneFrame({"TARGET": tb}) for _ in range(int(2 * fps))]
                    x = 315.0
                    fr += [SS.SceneFrame({"TARGET": tb, "OTHER": ob(x)}, order=("TARGET", "OTHER"))
                           for _ in range(int(fps / 3))]
                    while x > 300.5:
                        x = max(300.0, x - v)
                        fr.append(SS.SceneFrame({"TARGET": tb, "OTHER": ob(x)}, order=("TARGET", "OTHER"),
                                                drop=("TARGET",) if x <= 304 else ()))
                    fr += [SS.SceneFrame({"TARGET": tb, "OTHER": ob(x)}, order=("TARGET", "OTHER"),
                                         drop=("TARGET",)) for _ in range(int(fps / 2))]
                    fr += [SS.SceneFrame({"TARGET": tb, "OTHER": ob(x - v * (k + 1))}, order=("TARGET", "OTHER"),
                                         drop=("TARGET",)) for k in range(int(2 * fps))]
                    r = SS.run(fr, ppl, fps=fps)
                    bad = [k for k, (f, ds) in enumerate(zip(r.frames, r.dets))
                           if f.identity_trusted and f.detection_index is not None
                           and ds[f.detection_index].owners != ("TARGET",)]
                    self.assertEqual(bad, [], f"fps={fps} v={v_hs} hr={hr}")
                    self.assertTrue(r.bank_is_pure(), f"fps={fps} v={v_hs} hr={hr}")

    def test_partner_born_while_the_target_box_is_momentarily_tall(self):
        for fps in (15.0, 10.0):
            v = 2 * 80 / fps
            tall = (384.0, 290.0, 416.0, 400.0)
            fr = [Frame([cand(self.TGT)], [("TARGET",)]) for _ in range(int(fps))]
            fr += [Frame([cand(tall)], [("TARGET",)]) for _ in range(3)]
            fr += [Frame([cand(tall), cand(box_at(412, 360, 32, 80))], [("TARGET",), ("PERSON_A",)]) for _ in range(3)]
            x = 412.0
            while x > 400.5:
                x = max(400.0, x - v)
                fr.append(Frame([cand(box_at(x, 360, 32, 80))], [("PERSON_A",)]))
            fr += [Frame([cand(box_at(x, 360, 32, 80))], [("PERSON_A",)]) for _ in range(int(fps / 2))]
            fr += [Frame([cand(box_at(x - v * (k + 1), 360, 32, 80))], [("PERSON_A",)]) for k in range(int(fps))]
            rn = run(fr, fps=fps)
            self.assertEqual({t: o for t, o in rn.tracklet_owners().items() if len(o) > 1}, {}, fps)

    def test_union_with_a_short_partner_is_a_merge(self):
        b_at = (lambda x_: box_at(x_, 372, 32, 56))
        fr = [Frame([cand(self.TGT)], [("TARGET",)]) for _ in range(30)]
        fr += [Frame([cand(self.TGT), cand(b_at(412))], [("TARGET",), ("PERSON_A",)]) for _ in range(10)]
        fr += [Frame([cand((384.0, 320.0, 428.0, 400.0))], [("TARGET", "PERSON_A")]) for _ in range(10)]
        fr += [Frame([cand(self.TGT), cand(b_at(412 + 1.5 * (k + 1)))], [("TARGET",), ("PERSON_A",)])
               for k in range(30)]
        rn = run(fr, fps=FPS)
        self.assertEqual({t: o for t, o in rn.tracklet_owners().items() if len(o) > 1}, {})
        self.assertTrue(all(P.MERGED_SUSPECTED in r.observations[0].flags for r in rn.results[40:50]))

if __name__ == "__main__":
    unittest.main()

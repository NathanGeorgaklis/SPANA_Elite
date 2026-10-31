# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2 DEV remediation instruction (Nathan, 2026-10-31 07:09): F3c baseline population safety - (1) a
#                 genuine two-person union that persists must never become one ordinary tracklet; (2) a second person
#                 hidden / crouched wholly inside the target's box who stands up in front must not inherit the
#                 target's tracklet. Fail closed; bank pure; ordinary single-person poses stay trackable; multiple fps.
"""F3c regression tests (synthetic, neutral TARGET / OTHER / THIRD; no benchmark knowledge).

Failure 1 (confirmed at 8c798bc): a merged region whose two-person box is never merge-sized lapses, and when a third
person walks through it the region "resolves" (the union box counts as one single body, the passer-by as the second);
the union box's tracklet loses its lineage and becomes an ordinary tracklet holding two people (the target then sat
TRACKING on it).
Failure 2 (confirmed at 8c798bc, review r3a 81/81): a person crouching wholly inside the target's box (NESTED: no
tracklet) stands up in front of it; the target's own tracklet carries on onto that person.
"""
import unittest

from player_tracker_rb import population as P
from tests_rb import _synth_scene as SS
from tests_rb._synth_people import Frame, box_at, cand, run

SAME = (200, 60, 40)
OPP = (40, 60, 200)
FPS_SET = (30.0, 15.0, 10.0)


def people(kit):
    return SS.make_people([("TARGET", SAME, 1.0), ("OTHER", kit, 0.95), ("THIRD", (40, 200, 60), 0.0)])


def wrong_trusted(r):
    return [k for k, (f, ds) in enumerate(zip(r.frames, r.dets))
            if f.identity_trusted and f.detection_index is not None and ds[f.detection_index].owners != ("TARGET",)]


def tracking_on_union(r):
    return [k for k, (f, ds) in enumerate(zip(r.frames, r.dets))
            if f.rb_state == "TRACKING" and f.detection_index is not None and len(ds[f.detection_index].owners) > 1]


def clean_union_tracklet_frames(r):
    """Frames where a two-person box is an observed tracklet with neither MERGED_SUSPECTED nor MERGE_LINEAGE."""
    out = []
    for k, (p, ds) in enumerate(zip(r.pops, r.dets)):
        for v in p.tracklets:
            if v.observed and len(ds[v.detection_index].owners) > 1 and P.MERGE_LINEAGE not in v.flags \
                    and P.MERGED_SUSPECTED not in p.observations[v.detection_index].flags:
                out.append(k)
    return out


def mixed_single_tracklets(rn):
    """Population tracklets observed on two different people as SINGLE-person boxes (a silent swap)."""
    seen = {}
    for r, fr in zip(rn.results, rn.frames):
        for v in r.tracklets:
            if v.observed and len(fr.owners[v.detection_index]) == 1:
                seen.setdefault(v.tracklet_id, set()).update(fr.owners[v.detection_index])
    return {t: o for t, o in seen.items() if len(o) > 1}


class TestPersistentUnion(unittest.TestCase):
    TB = SS.box_at(300, 400, 40, 100)
    OB = SS.box_at(315, 400, 40, 100)

    def _union(self, fps, kit, seconds, third=False, gaps=()):
        fr = [SS.SceneFrame({"TARGET": self.TB}) for _ in range(int(2 * fps))]
        fr += [SS.SceneFrame({"TARGET": self.TB, "OTHER": self.OB}, order=("OTHER", "TARGET"))
               for _ in range(int(fps / 3))]
        for k in range(int(seconds * fps)):
            t = k / fps
            bx = {"TARGET": self.TB, "OTHER": self.OB}
            if third and 1.5 <= t <= 3.0:                     # a third person crosses the region after it formed
                bx["THIRD"] = SS.box_at(200 + (t - 1.5) / 1.5 * 220, 395, 40, 100)
            drop = ("TARGET", "OTHER") if any(a <= t < b for a, b in gaps) else ()
            fr.append(SS.SceneFrame(bx, merge=(("TARGET", "OTHER"),) if not drop else (),
                                    order=("THIRD", "TARGET", "OTHER"), drop=drop))
        return SS.run(fr, people(kit), fps=fps)

    def test_union_box_never_becomes_an_ordinary_tracklet(self):
        for fps in FPS_SET:
            for kit in (SAME, OPP):
                for third, gaps in ((False, ()), (True, ()), (True, ((4.0, 4.6),))):
                    # (audit A1: a third person crossing, then a detection gap of the union)
                    r = self._union(fps, kit, 6.0, third=third, gaps=gaps)
                    tag = f"fps={fps} kit={kit} third={third} gaps={gaps}"
                    self.assertEqual(clean_union_tracklet_frames(r), [], tag)
                    self.assertEqual(tracking_on_union(r), [], tag)
                    self.assertEqual(wrong_trusted(r), [], tag)
                    self.assertTrue(r.bank_is_pure(), tag)

    def test_union_box_reborn_after_long_persistence_keeps_lineage(self):
        # the union is undetected for a while (its tracklet ends) late in a long union: when it is detected again
        # the merge must still be remembered (it never ages out while the two-person box persists)
        for fps in FPS_SET:
            r = self._union(fps, SAME, 7.0, gaps=((4.0, 4.0 + 1.2),))
            self.assertEqual(clean_union_tracklet_frames(r), [], fps)
            self.assertEqual(tracking_on_union(r), [], fps)
            self.assertTrue(r.bank_is_pure(), fps)

    def test_real_split_after_a_union_gives_two_people_and_no_swap(self):
        # the two people come apart clearly: the population follows both again, with no mixed tracklet. Their
        # tracklets inherit MERGE_LINEAGE from the union box's tracklet (8c798bc behaviour, conservative: never a
        # bank sample); clearing it after a proven split is reacquisition/verification work, not asserted here.
        for fps in FPS_SET:
            # (audit C: the union box must really hold both people, so that a merged region forms)
            n_in = int(fps)
            fr = [Frame([cand(box_at(400, 360, 32, 80)), cand(box_at(445 - 33 * k / n_in, 360, 32, 80))],
                        [("TARGET",), ("OTHER",)]) for k in range(n_in)]
            fr += [Frame([cand((384.0, 320.0, 428.0, 400.0))], [("TARGET", "OTHER")]) for _ in range(int(fps))]
            fr += [Frame([cand(box_at(400 - 40 * (k + 1) / fps, 360, 32, 80)),
                          cand(box_at(412 + 40 * (k + 1) / fps, 360, 32, 80))], [("TARGET",), ("OTHER",)])
                   for k in range(int(2 * fps))]
            rn = run(fr, fps=fps)
            self.assertTrue(any(P.MERGED_SUSPECTED in o.flags for r in rn.results for o in r.observations), fps)
            last = rn.results[-1]
            self.assertEqual(len([v for v in last.tracklets if v.observed]), 2, fps)
            self.assertEqual(mixed_single_tracklets(rn), {}, fps)


class TestHiddenPersonEmergence(unittest.TestCase):
    TGT = box_at(400, 360, 32, 80)

    def test_crouched_person_standing_up_in_front_never_inherits_a_tracklet(self):
        for fps in FPS_SET:
            for sc in (1.0, 1.05, 1.1):
                for dx in (0.0, 2.0, 4.0):
                    for rise in (1, 2, 4):
                        bw, bh, cx = 32 * sc, 80 * sc, 400 + dx
                        crouch = box_at(cx, 376, 28, 48)                      # wholly inside the target's box
                        fr = [Frame([cand(self.TGT)], [("TARGET",)]) for _ in range(int(fps))]
                        fr += [Frame([cand(self.TGT), cand(crouch)], [("TARGET",), ("OTHER",)])
                               for _ in range(int(fps / 2))]
                        for k in range(rise):
                            f = (k + 1) / rise
                            h, w = 48 + (bh - 48) * f, 28 + (bw - 28) * f
                            b = box_at(cx, 400 - h / 2, w, h)
                            fr.append(Frame([cand(b)], [("OTHER",)]) if f >= 0.75 else
                                      Frame([cand(self.TGT), cand(b)], [("TARGET",), ("OTHER",)]))
                        fr += [Frame([cand(box_at(cx, 400 - bh / 2, bw, bh))], [("OTHER",)])
                               for _ in range(int(fps / 2))]
                        v = 2 * 80 / fps
                        fr += [Frame([cand(box_at(cx + v * (k + 1), 400 - bh / 2, bw, bh))], [("OTHER",)])
                               for k in range(int(fps))]
                        rn = run(fr, fps=fps)
                        self.assertEqual(mixed_single_tracklets(rn), {}, f"fps={fps} s={sc} dx={dx} rise={rise}")

    def _scene(self, fps, kit, rise_s):
        tb = SS.box_at(300, 400, 40, 100)
        fr = [SS.SceneFrame({"TARGET": tb}) for _ in range(int(2 * fps))]
        fr += [SS.SceneFrame({"TARGET": tb, "OTHER": SS.box_at(300, 425, 36, 50)}, order=("TARGET", "OTHER"))
               for _ in range(int(fps / 2))]                                   # crouched wholly inside, in front
        n = max(1, int(rise_s * fps))
        for k in range(n):
            f = (k + 1) / n
            h = 50 + 50 * f
            fr.append(SS.SceneFrame({"TARGET": tb, "OTHER": SS.box_at(300, 450 - h / 2, 36 + 4 * f, h)},
                                    order=("TARGET", "OTHER")))
        v = 2 * 100 / fps
        fr += [SS.SceneFrame({"TARGET": tb, "OTHER": SS.box_at(300 + v * (k + 1), 400, 40, 100)},
                             order=("TARGET", "OTHER"), drop=("TARGET",)) for k in range(int(2 * fps))]
        return SS.run(fr, people(kit), fps=fps)

    def test_target_is_never_trusted_on_a_person_who_stood_up_in_front(self):
        for fps in FPS_SET:
            for kit in (SAME, OPP):
                for rise_s in (0.1, 0.3):
                    r = self._scene(fps, kit, rise_s)
                    tag = f"fps={fps} kit={kit} rise={rise_s}"
                    self.assertEqual(wrong_trusted(r), [], tag)
                    self.assertTrue(r.bank_is_pure(), tag)
                    # no silent carry-over: the target's episode never continues onto the other person; any later
                    # TRACKING on it is a NEW (reacquired, unverified) episode and untrusted
                    for k in r.swaps():
                        self.assertGreater(r.frames[k].target_episode, 1, f"{tag} frame {k}")
                        self.assertFalse(r.frames[k].identity_trusted, f"{tag} frame {k}")


class TestOrdinaryCasesStayTrackable(unittest.TestCase):
    PPL = people(OPP)

    def test_isolated_player_is_followed_and_trusted(self):
        for fps in FPS_SET:
            fr = [SS.SceneFrame({"TARGET": SS.box_at(300 + 60 * k / fps, 400, 40, 100)}) for k in range(int(3 * fps))]
            r = SS.run(fr, self.PPL, fps=fps)
            self.assertEqual(r.states()[-5:], ["TRACKING"] * 5, fps)
            self.assertTrue(r.frames[-1].identity_trusted, fps)
            self.assertEqual({f.target_episode for f in r.frames}, {1}, fps)

    def test_short_pose_change_stays_trackable(self):
        for fps in FPS_SET:
            fr = [SS.SceneFrame({"TARGET": SS.box_at(300, 400, 40, 100)}) for _ in range(int(2 * fps))]
            n = int(0.3 * fps)
            fr += [SS.SceneFrame({"TARGET": SS.box_at(300, 400, 40 + 30 * (k + 1) / n, 100)}) for k in range(n)]
            fr += [SS.SceneFrame({"TARGET": SS.box_at(300, 400, 40, 100)}) for _ in range(int(2 * fps))]
            r = SS.run(fr, self.PPL, fps=fps)
            self.assertNotIn("LOST", r.states(), fps)
            self.assertEqual(r.states()[-5:], ["TRACKING"] * 5, fps)
            self.assertEqual(wrong_trusted(r), [], fps)
            self.assertTrue(r.bank_is_pure(), fps)

    def test_persistent_wide_single_person_pose_is_never_unsafe(self):
        # KNOWN LIMIT (F3 v4, recorded): a wide pose held longer than merged_anomaly_max_s escalates to a merge;
        # it must stay safe (never trusted on a merged box, bank pure, no swap)
        for fps in FPS_SET:
            fr = [SS.SceneFrame({"TARGET": SS.box_at(300, 400, 40, 100)}) for _ in range(int(2 * fps))]
            fr += [SS.SceneFrame({"TARGET": SS.box_at(300, 400, 90, 100)}) for _ in range(int(3 * fps))]
            r = SS.run(fr, self.PPL, fps=fps)
            self.assertEqual(wrong_trusted(r), [], fps)
            self.assertTrue(r.bank_is_pure(), fps)
            self.assertEqual(r.swaps(), [], fps)

    def test_near_far_occlusion_keeps_both_people(self):
        for fps in FPS_SET:
            near = box_at(400, 360, 32, 80)
            fr = []
            for k in range(int(3 * fps)):
                x = 360 + 30 * k / fps
                far = box_at(x, 330, 16, 40)
                fr.append(Frame([cand(near)], [("TARGET",)]) if 392 <= x <= 408 else
                          Frame([cand(near), cand(far)], [("TARGET",), ("OTHER",)]))
            rn = run(fr, fps=fps)
            self.assertEqual(mixed_single_tracklets(rn), {}, fps)
            self.assertEqual(len({r.observations[0].tracklet_id for r in rn.results}), 1, fps)

    def test_opponent_passing_in_front_and_behind(self):
        for fps in FPS_SET:
            ppl = people(OPP)
            tb = SS.box_at(300, 400, 40, 100)
            for order in (("TARGET", "OTHER"), ("OTHER", "TARGET")):
                fr = [SS.SceneFrame({"TARGET": tb}) for _ in range(int(2 * fps))]
                fr += [SS.SceneFrame({"TARGET": tb, "OTHER": SS.box_at(200 + 100 * k / fps, 402, 40, 102)},
                                     order=order) for k in range(int(2 * fps))]
                r = SS.run(fr, ppl, fps=fps)
                self.assertEqual(wrong_trusted(r), [], (fps, order))
                self.assertTrue(r.bank_is_pure(), (fps, order))


if __name__ == "__main__":
    unittest.main()


class TestReviewRound1(unittest.TestCase):
    """F3c independent review findings (reproduced, then fixed)."""

    def test_third_person_stepping_out_of_a_persisting_union_does_not_resolve_it(self):
        # H2: a third person hidden behind the pair since before the merge steps out sideways (born overlapping the
        # region: an emergence). The union box still fills the region and is still two bodies wide, so it must not
        # count as the region's remaining single body; after a detection gap the union must not come back clean.
        TB, OB = SS.box_at(300, 400, 40, 100), SS.box_at(315, 400, 40, 100)
        hidden3 = SS.box_at(307, 396, 36, 92)
        for fps in FPS_SET:
            for kit in (SAME, OPP):
                for gap in (0.0, 0.6, 1.2):
                    with self.subTest(fps=fps, kit=kit, gap=gap):
                        fr = [SS.SceneFrame({"TARGET": TB, "THIRD": hidden3}, order=("THIRD", "TARGET"))
                              for _ in range(int(2 * fps))]
                        fr += [SS.SceneFrame({"TARGET": TB, "OTHER": OB, "THIRD": hidden3},
                                             order=("THIRD", "OTHER", "TARGET")) for _ in range(int(fps / 3))]
                        for k in range(int(7 * fps)):
                            t = k / fps
                            bx = {"TARGET": TB, "OTHER": OB,
                                  "THIRD": SS.box_at(307 + min(1.0, max(0.0, t - 1.0)) * 120, 396, 36, 92)
                                  if t >= 1.0 else hidden3}
                            drop = ("TARGET", "OTHER") if 3.5 <= t < 3.5 + gap else ()
                            fr.append(SS.SceneFrame(bx, merge=(("TARGET", "OTHER"),) if not drop else (), drop=drop,
                                                    order=("THIRD", "OTHER", "TARGET")))
                        r = SS.run(fr, people(kit), fps=fps)
                        self.assertEqual(clean_union_tracklet_frames(r), [])
                        self.assertEqual(tracking_on_union(r), [])
                        self.assertEqual(wrong_trusted(r), [])
                        self.assertTrue(r.bank_is_pure())

    def test_persistent_part_box_on_a_running_player_breaks_nothing_at_any_frame_rate(self):
        # M1: a contained body is compared where it would be had it moved with its host, so a persistent lower-body
        # part box on a moving player never "vanishes" merely because the player moved between frames (low fps).
        for fps in FPS_SET:
            for speed_h in (1.0, 2.0):                         # body heights per second
                with self.subTest(fps=fps, speed_h=speed_h):
                    v = speed_h * 100 / fps
                    fr = []
                    for k in range(int(4 * fps)):
                        b = SS.box_at(200 + v * k, 400, 40, 100)
                        part = (b[0] + 4, b[1] + 45, b[2] - 4, b[3] - 1)
                        fr.append(SS.SceneFrame({"TARGET": b}, parts=(("TARGET", part),) if k >= int(fps) else ()))
                    r = SS.run(fr, people(OPP), fps=fps)
                    breaks = [o for p in r.pops for o in p.observations if o.continuity_break]
                    self.assertEqual(breaks, [])
                    self.assertEqual(r.states().count("TRACKING"), len(r.frames))
                    self.assertTrue(all(f.identity_trusted for f in r.frames[1:]))


class TestAuditRound1(unittest.TestCase):
    """F3c test-audit findings (reproduced, then fixed)."""

    def test_farther_person_overlapping_the_crouch_spot_does_not_mask_its_disappearance(self):
        # A2: a farther person behind (bottom off the host's feet line) whose box overlaps where the crouched
        # person was must not count as "the crouched person is still there" when that person stands up in front.
        TGT = box_at(300, 400, 40, 100)
        crouch = box_at(300, 425, 36, 50)                     # feet-aligned, nested in the target's box
        far = (285.0, 360.0, 315.0, 435.0)                    # bottom 15 px above the feet line
        for fps in FPS_SET:
            for third in (False, True):
                for rise in (1, 2):
                    with self.subTest(fps=fps, third=third, rise=rise):
                        def F(ds, os):
                            if third:
                                ds, os = ds + [far], os + [("THIRD",)]
                            return Frame([cand(b) for b in ds], os)
                        fr = [F([TGT], [("TARGET",)]) for _ in range(int(fps))]
                        fr += [F([TGT, crouch], [("TARGET",), ("OTHER",)]) for _ in range(int(fps / 2))]
                        for k in range(rise):
                            f = (k + 1) / rise
                            h = 50 + 52 * f
                            b = box_at(300, 450 - h / 2, 36 + 6 * f, h)
                            fr.append(F([b], [("OTHER",)]) if f >= 0.75 else F([TGT, b], [("TARGET",), ("OTHER",)]))
                        fr += [F([box_at(300 + 3 * (k + 1), 399, 42, 102)], [("OTHER",)]) for k in range(int(fps))]
                        rn = run(fr, fps=fps)
                        self.assertEqual(mixed_single_tracklets(rn), {})
                        self.assertTrue(any(o.continuity_break for r in rn.results for o in r.observations))

    def test_hidden_min_duration_is_measured_in_seconds_at_every_frame_rate(self):
        # population_hidden_min_s (0.2 s): a feet-aligned nested box seen for 0.1 s then gone is a flicker (no
        # break); one seen for 0.3 s then gone breaks continuity. Same in seconds at every frame rate.
        TGT = box_at(300, 400, 40, 100)
        crouch = box_at(300, 425, 36, 50)
        for fps in FPS_SET:
            for seen_s, expect_break in ((0.1, False), (0.3, True)):
                with self.subTest(fps=fps, seen_s=seen_s):
                    fr = [Frame([cand(TGT)], [("TARGET",)]) for _ in range(int(fps))]
                    fr += [Frame([cand(TGT), cand(crouch)], [("TARGET",), ("TARGET",)])
                           for _ in range(max(1, round(seen_s * fps)))]
                    fr += [Frame([cand(TGT)], [("TARGET",)]) for _ in range(int(fps))]
                    rn = run(fr, fps=fps)
                    breaks = [o for r in rn.results for o in r.observations if o.continuity_break]
                    self.assertEqual(bool(breaks), expect_break)

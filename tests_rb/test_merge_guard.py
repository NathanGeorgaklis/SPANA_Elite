# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2 DEV remediation instruction (Nathan, 2026-10-30 17:41, Part 2): diagnose why
#                 MERGED_DETECTION_SUSPECTED fires on isolated visible players; no relaxing the guard for coverage;
#                 distinguish true multi-person merge evidence from single-body box changes; overlapping people stay
#                 separate competitors; a detector artefact must not permanently poison the target tracklet; merge
#                 lineage stays conservative for real merges. R2 DEV finding F3 (R2/DEV_RUN_ef6d635). Independent code
#                 review of the first fix (H1 gradual unseen second body, H2 nested full-height second body, H3 frame-rate
#                 dependent abrupt test).
"""Merged-guard regression tests (synthetic, neutral TARGET / PERSON_A / OTHER; no benchmark knowledge).

Failure class reproduced (R2 DEV F3): a single isolated player's box widens with pose (limbs, stride) at constant
height. Against the 1 s median size reference that crossed the AREA_RATIO / ASPECT_WIDE tests although no second body
existed; the tracklet was closed, a merged region that could never resolve was created, and the body's next tracklet
carried MERGE_LINEAGE for life: the target was lost for the rest of the clip.

Geometry cannot tell pose from an unseen second body (review H1), so an uncorroborated size change is a SIZE_ANOMALY:
the target abstains on it (never trusted, never banked), the tracklet is NOT closed, the size reference is held, and an
anomaly lasting merged_anomaly_max_s becomes a merge. Corroborated size evidence (a second body could be in the box)
remains an immediate merge with the full conservative region / lineage logic.
"""
import math
import unittest

from player_tracker_rb import population as P
from player_tracker_rb import target as T
from player_tracker_rb.config import TrackerConfig
from tests_rb import _synth_scene as SS
from tests_rb._synth_people import Frame, box_at, cand, run

CFG = TrackerConfig.default()
FPS = 30.0
MAX_S = CFG.get("merged_anomaly_max_s")


def widening(n_still, n_grow, w0=32.0, w1=74.0, h=80.0, cx=400.0, cy=360.0, n_hold=10, n_back=0):
    """A single body: still at width w0, widening linearly to w1 over n_grow frames (pose), held, then back to w0.
    Defaults reproduce the measured failure: x2.3 width in 0.5 s at 30 fps (about x1.06 per frame)."""
    out = [box_at(cx, cy, w0, h) for _ in range(n_still)]
    out += [box_at(cx, cy, w0 + (w1 - w0) * (k + 1) / n_grow, h) for k in range(n_grow)]
    out += [box_at(cx, cy, w1, h) for _ in range(n_hold)]
    out += [box_at(cx, cy, w0, h) for _ in range(n_back)]
    return out


def code(o):
    return "M" if P.MERGED_SUSPECTED in o.flags else ("A" if P.SIZE_ANOMALY in o.flags else ".")


def widest(r):
    return max(r.observations, key=lambda o: (o.bbox[2] - o.bbox[0]) * (o.bbox[3] - o.bbox[1]))


class TestSingleBodySizeChange(unittest.TestCase):
    def test_short_pose_widening_does_not_close_the_tracklet(self):
        boxes = widening(30, 15, n_hold=8, n_back=20)            # anomaly < merged_anomaly_max_s, then back
        rn = run([Frame([cand(b)], [("TARGET",)]) for b in boxes], fps=FPS)
        tids = {r.observations[0].tracklet_id for r in rn.results}
        self.assertEqual(len(tids), 1, "the body's tracklet must not be closed by a short pose change")
        self.assertIsNotNone(next(iter(tids)))
        codes = "".join(code(r.observations[0]) for r in rn.results)
        self.assertNotIn("M", codes)
        self.assertIn("A", codes)
        self.assertTrue(codes.endswith("." * 5), "after the anomaly the body is a clean observation again")
        for r in rn.results:
            o = r.observations[0]
            v = r.tracklet(o.tracklet_id)
            self.assertNotIn(P.MERGE_LINEAGE, v.flags)
            self.assertNotIn(P.IN_MERGED_REGION, o.flags)
            if P.SIZE_ANOMALY in o.flags:
                self.assertFalse(v.bank_eligible, "a size-anomalous box is never an appearance sample")
                self.assertIn(P.SIZE_ANOMALY, v.bank_refusals)

    def test_a_persistent_wide_box_never_becomes_a_clean_single_body(self):
        # review H1: the reference must not adopt the wide size; a wide box that persists becomes a merge
        boxes = widening(30, 15, n_hold=int(2.0 * FPS))
        rn = run([Frame([cand(b)], [("TARGET",)]) for b in boxes], fps=FPS)
        codes = "".join(code(r.observations[0]) for r in rn.results[30:])
        first_a = codes.index("A")
        self.assertNotIn(".", codes[first_a:], "once anomalous, the wide box is never a clean single body again")
        self.assertTrue(codes.endswith("M" * 10), "a persistent anomaly becomes a merge")

    def test_escalation_time_is_frame_rate_free(self):
        # review H3: the same physical sequence classifies the same in seconds at any frame rate
        onsets = {}
        for fps in (25.0, 30.0, 60.0):
            k = fps / 30.0
            boxes = widening(int(30 * k), int(15 * k), n_hold=int(2.0 * fps))
            rn = run([Frame([cand(b)], [("TARGET",)]) for b in boxes], fps=fps)
            codes = [code(r.observations[0]) for r in rn.results]
            onsets[fps] = (codes.index("A") / fps, codes.index("M") / fps)
            self.assertAlmostEqual(onsets[fps][1] - onsets[fps][0], MAX_S, delta=2.0 / fps)
        a = [v[0] for v in onsets.values()]
        self.assertLess(max(a) - min(a), 0.1, onsets)

    def test_abrupt_jump_with_no_seen_second_body_is_abstained_then_merged(self):
        a = box_at(400, 360, 32, 80)
        frames = [Frame([cand(a)], [("TARGET",)]) for _ in range(12)]
        frames += [Frame([cand(box_at(410, 360, 58, 82))], [("TARGET", "PERSON_A")])
                   for _ in range(int(math.ceil(MAX_S * FPS)) + 3)]
        rn = run(frames, fps=FPS)
        self.assertNotEqual(code(rn.results[12].observations[0]), ".", "never a clean single body")
        self.assertEqual(code(rn.results[-1].observations[0]), "M")

    def test_growth_no_single_body_explains_is_a_merge_at_once(self):
        cap = CFG.get("merged_uncorroborated_area_ratio")
        boxes = widening(30, 12, w0=30.0, w1=30.0 * (cap + 0.5))   # x3 in 0.4 s
        rn = run([Frame([cand(b)], [("TARGET",)]) for b in boxes], fps=FPS)
        codes = "".join(code(r.observations[0]) for r in rn.results[30:])
        self.assertIn("M", codes)
        self.assertLess(codes.index("M"), int(MAX_S * FPS), "merged before any anomaly escalation")


class TestIntermittentAnomaly(unittest.TestCase):
    """Review round 2 H1: one near-size or partial frame must not end an anomaly (and reset escalation)."""

    def _union_run(self, pattern):
        # 1 s of a single body, then 3 s of a union whose width follows pattern(k) (in body widths); "P" = a partial
        # frame (lower body hidden)
        frames = [Frame([cand(box_at(400, 360, 32, 80))], [("TARGET",)]) for _ in range(30)]
        for k in range(90):
            p = pattern(k)
            if p == "P":
                frames.append(Frame([cand((384.0, 320.0, 448.0, 360.0))], [("TARGET", "PERSON_A")]))
            else:
                w = 32 * p
                frames.append(Frame([cand(box_at(400 + (w - 32) / 2, 360, w, 80))], [("TARGET", "PERSON_A")]))
        rn = run(frames, fps=FPS)
        return "".join(code(r.observations[0]) for r in rn.results[30:])

    def check(self, codes):
        first = codes.index("A")
        self.assertNotIn(".", codes[first:], "an anomaly never ends on intermittent frames of a union")
        self.assertTrue(codes.endswith("M" * 20), "a persistent union becomes a merge")
        self.assertLessEqual(codes.index("M") - first, int(math.ceil(MAX_S * FPS)) + 1)

    def test_union_alternating_wide_and_less_wide(self):
        self.check(self._union_run(lambda k: 2.0 if k % 2 == 0 else 1.45))

    def test_union_with_one_narrower_frame_every_half_second(self):
        self.check(self._union_run(lambda k: 1.45 if k % 15 == 14 else 2.0))

    def test_union_with_one_partial_frame_every_half_second(self):
        self.check(self._union_run(lambda k: "P" if k % 15 == 14 else 2.0))

    def test_partial_frames_during_an_anomaly_neither_end_nor_escalate_it(self):
        # review round 3 M1: a short pose anomaly, then the lower body hidden for 1.5 s, then normal again: a partial
        # box is not evidence of a second body, so it never becomes a merge (and the anomaly does not end on it)
        frames = [Frame([cand(b)], [("TARGET",)]) for b in widening(30, 15, n_hold=2)]
        part = (363.0, 320.0, 437.0, 368.0)                      # the same wide box, lower body hidden (0.6 h)
        frames += [Frame([cand(part)], [("TARGET",)]) for _ in range(45)]
        frames += [Frame([cand(box_at(400, 360, 32, 80))], [("TARGET",)]) for _ in range(30)]
        rn = run(frames, fps=FPS)
        codes = "".join(code(r.observations[0]) for r in rn.results)
        self.assertNotIn("M", codes)
        first = codes.index("A")
        self.assertNotIn(".", codes[first:first + 17 + 45], "partial frames do not end the anomaly")
        self.assertTrue(codes.endswith("." * 10))
        v = rn.results[-1].tracklet(rn.results[-1].observations[0].tracklet_id)
        self.assertNotIn(P.MERGE_LINEAGE, v.flags)

    def test_running_stride_returning_to_normal_width_never_escalates(self):
        # a single body's stride: width oscillates between x1.0 and x2.0 of its standing width, period 0.8 s; each
        # narrow phase (<= x1.25 for about 0.27 s) ends the anomaly, so it never becomes a merge
        frames = [Frame([cand(box_at(400, 360, 32, 80))], [("TARGET",)]) for _ in range(30)]
        for k in range(120):
            f = 1.5 - 0.5 * math.cos(2 * math.pi * k / (0.8 * FPS))
            w = 32 * f
            frames.append(Frame([cand(box_at(400, 360, w, 80))], [("TARGET",)]))
        rn = run(frames, fps=FPS)
        codes = "".join(code(r.observations[0]) for r in rn.results[30:])
        self.assertIn("A", codes)
        self.assertNotIn("M", codes)
        self.assertIn(".", codes[codes.index("A"):], "the anomaly ends between strides")
        self.assertEqual(len({r.observations[0].tracklet_id for r in rn.results}), 1)


class TestCorroboratedMerges(unittest.TestCase):
    def test_gradual_growth_over_a_second_tracked_body_is_a_merge_at_once(self):
        frames = []
        for k in range(30):
            frames.append(Frame([cand(box_at(400, 360, 32, 80)), cand(box_at(470 - 1.5 * k, 360, 32, 80))],
                                [("TARGET",), ("PERSON_A",)]))
        for k in range(20):
            w = 32 + 26 * (k + 1) / 20
            frames.append(Frame([cand(box_at(400 + (w - 32) / 2, 360, w, 80))], [("TARGET", "PERSON_A")]))
        rn = run(frames, fps=FPS)
        codes = "".join(code(r.observations[0]) for r in rn.results[30:])
        self.assertIn("M", codes)
        self.assertNotIn("A", codes, "corroborated size evidence is never a mere anomaly")

    def test_gradual_growth_over_a_recently_hidden_body_is_a_merge(self):
        frames = [Frame([cand(box_at(400, 360, 32, 80)), cand(box_at(440, 360, 32, 80))],
                        [("TARGET",), ("PERSON_A",)]) for _ in range(30)]
        frames += [Frame([cand(box_at(400, 360, 32, 80))], [("TARGET",)]) for _ in range(5)]
        for k in range(20):
            w = 32 + 30 * (k + 1) / 20
            frames.append(Frame([cand(box_at(400 + (w - 32) / 2, 360, w, 80))], [("TARGET", "PERSON_A")]))
        rn = run(frames, fps=FPS)
        self.assertTrue(any(P.MERGED_SUSPECTED in r.observations[0].flags for r in rn.results[35:]))

    def test_union_holding_a_second_persons_own_box_is_abstained_then_merged(self):
        # review H2: the union box M of A and B plus B's own full-height box nested inside M. Nested boxes are not
        # corroboration (tall same-body alternative boxes are common on real footage), so M is abstained on from the
        # first merge-sized frame (never a clean single body, never banked) and is a merge once it persists.
        frames = [Frame([cand(box_at(400, 360, 32, 80))], [("TARGET",)]) for _ in range(30)]
        for k in range(10):
            w = 32 + 32 * (k + 1) / 10
            m = box_at(400 + (w - 32) / 2, 360, w, 80)
            frames.append(Frame([cand(m), cand((m[2] - 30, m[1] + 2, m[2] - 1, m[3] - 1))],
                                [("TARGET", "PERSON_A"), ("PERSON_A",)]))
        m = box_at(416, 360, 64, 80)
        for _ in range(int(math.ceil(MAX_S * FPS)) + 3):
            frames.append(Frame([cand(m), cand((m[2] - 30, m[1] + 2, m[2] - 1, m[3] - 1))],
                                [("TARGET", "PERSON_A"), ("PERSON_A",)]))
        rn = run(frames, fps=FPS)
        codes = "".join(code(widest(r)) for r in rn.results[30:])
        first = next(k for k, ch in enumerate(codes) if ch != ".")
        self.assertNotIn(".", codes[first:], "once merge-sized, never a clean single body again")
        self.assertEqual(codes[-1], "M")
        for r in rn.results[30:]:
            o = widest(r)
            if o.tracklet_id:
                self.assertFalse(r.tracklet(o.tracklet_id).bank_eligible and code(o) != ".")

    def test_same_bodys_tight_alternative_box_does_not_turn_a_pose_change_into_a_merge(self):
        # review round 2 M1: the detector also returns the body's own tight box inside its pose-widened box
        frames = []
        for b in widening(30, 15, n_hold=8, n_back=20):
            cx = (b[0] + b[2]) / 2
            frames.append(Frame([cand(b), cand(box_at(cx, 361, 30, 76))], [("TARGET",), ("TARGET",)]))
        rn = run(frames, fps=FPS)
        codes = "".join(code(widest(r)) for r in rn.results)
        self.assertNotIn("M", codes)
        self.assertEqual(len({widest(r).tracklet_id for r in rn.results}), 1)

    def test_overlapping_genuine_people_stay_separate_competitors(self):
        frames = [Frame([cand(box_at(400, 360, 32, 80)), cand(box_at(420, 362, 32, 80))],
                        [("TARGET",), ("PERSON_A",)]) for _ in range(40)]
        rn = run(frames, fps=FPS)
        last = rn.results[-1]
        self.assertEqual(len({o.tracklet_id for o in last.observations}), 2)
        for o in last.observations:
            self.assertNotIn(P.MERGED_SUSPECTED, o.flags)
            self.assertTrue(o.second_body_possible)

    def test_real_merge_keeps_conservative_lineage(self):
        a, b = box_at(400, 360, 32, 80), box_at(440, 360, 32, 80)
        m = (a[0], a[1], b[2], b[3])
        frames = [Frame([cand(a), cand(b)], [("TARGET",), ("PERSON_A",)]) for _ in range(12)]
        frames += [Frame([cand(m)], [("TARGET", "PERSON_A")]) for _ in range(6)]
        frames += [Frame([cand(a)], [("TARGET",)]) for _ in range(6)]       # only one body re-emerges
        rn = run(frames, fps=FPS)
        for r in rn.results[12:18]:
            self.assertIn(P.MERGED_SUSPECTED, r.observations[0].flags)
        for r in rn.results[18:]:
            o = r.observations[0]
            v = r.tracklet(o.tracklet_id) if o.tracklet_id else None
            self.assertTrue(P.MERGED_SUSPECTED in o.flags or (v is not None and P.MERGE_LINEAGE in v.flags),
                            "a body from an unresolved real merge stays conservative")


class TestNestedAndPartial(unittest.TestCase):
    def test_nested_part_box_is_not_a_second_body(self):
        frames = []
        for b in widening(30, 15, n_hold=8, n_back=10):
            part = box_at((b[0] + b[2]) / 2, b[1] + 20, 20, 36)          # a torso box (0.45 of the height)
            frames.append(Frame([cand(b), cand(part)], [("TARGET",), ("TARGET",)]))
        rn = run(frames, fps=FPS)
        for r in rn.results:
            body = widest(r)
            self.assertNotIn(P.MERGED_SUSPECTED, body.flags)
        self.assertEqual(len({widest(r).tracklet_id for r in rn.results}), 1)

    def test_partial_box_recovering_to_full_is_not_a_merge_and_not_lineage(self):
        full = box_at(400, 360, 32, 80)
        part = (full[0], full[1], full[2], full[1] + 40)
        frames = [Frame([cand(full)], [("TARGET",)]) for _ in range(20)]
        frames += [Frame([cand(part)], [("TARGET",)]) for _ in range(8)]
        frames += [Frame([cand(full)], [("TARGET",)]) for _ in range(12)]
        rn = run(frames, fps=FPS)
        self.assertEqual(len({r.observations[0].tracklet_id for r in rn.results}), 1)
        for r in rn.results:
            self.assertNotIn(P.MERGED_SUSPECTED, r.observations[0].flags)


class TestTargetLevel(unittest.TestCase):
    PPL = SS.make_people([("TARGET", (200, 60, 40), 1.0), ("OTHER", (40, 60, 200), 1.0)])

    def test_target_abstains_on_its_pose_widening_and_recovers_without_poison(self):
        boxes = widening(30, 15, w0=40.0, w1=92.0, h=100.0, cx=300.0, cy=400.0, n_hold=6, n_back=60)
        r = SS.run([SS.SceneFrame({"TARGET": b}) for b in boxes], self.PPL)
        anomalous = [k for k, p in enumerate(r.pops) if P.SIZE_ANOMALY in p.observations[0].flags]
        self.assertTrue(anomalous)
        for k in anomalous:
            self.assertFalse(r.frames[k].identity_trusted, f"frame {k}: never trusted on a size anomaly")
            self.assertNotEqual(r.frames[k].decision.get("bank", {}).get("status"), "ACCEPTED")
        # review round 3 H1: after the anomaly the tracklet's continuity is not identity evidence; the target comes
        # back through reacquisition as a new episode (here unverified: no confusable competitor to contrast)
        self.assertEqual(r.states()[-10:], [T.TRACKING] * 10, "the body is followed again (not poisoned)")
        self.assertGreaterEqual(r.frames[-1].target_episode, 2)
        self.assertTrue(any(f.decision.get("continuity_broken") == "SIZE_ANOMALY_ENDED" for f in r.frames))
        self.assertTrue(r.bank_is_pure())
        self.assertEqual(r.swaps(), [])

    def test_unseen_same_kit_body_after_a_union_is_never_trusted(self):
        # review round 3 H1: an undetected same-kit person comes out from behind the target (a union for 0.5 s), then
        # steps in front of it and hides it. The anomaly evidence must not be thrown away: nothing after the union
        # phase may be trusted on the other person, and the bank stays pure.
        same = SS.make_people([("TARGET", (200, 60, 40), 1.0), ("OTHER", (200, 60, 40), 0.95)])
        tb = SS.box_at(300, 400, 40, 100)
        fr = [SS.SceneFrame({"TARGET": tb, "OTHER": SS.box_at(300, 395, 38, 95)}, order=("OTHER", "TARGET"),
                            drop=("OTHER",)) for _ in range(40)]
        fr += [SS.SceneFrame({"TARGET": tb, "OTHER": SS.box_at(300 + 40 * (k + 1) / 15, 395, 38, 95)},
                             order=("OTHER", "TARGET"), merge=(("TARGET", "OTHER"),)) for k in range(15)]
        fr += [SS.SceneFrame({"TARGET": tb, "OTHER": SS.box_at(302, 401, 41, 101)}, order=("TARGET", "OTHER"),
                             drop=("TARGET",)) for _ in range(30)]
        fr += [SS.SceneFrame({"TARGET": tb, "OTHER": SS.box_at(302 + 3 * (k + 1), 401, 41, 101)},
                             order=("TARGET", "OTHER"), drop=("TARGET",)) for k in range(30)]
        r = SS.run(fr, same)
        first_a = next(k for k, p in enumerate(r.pops) if p.observations and P.SIZE_ANOMALY in p.observations[0].flags)
        trusted_on_other = [k for k in range(first_a, len(fr)) if r.frames[k].identity_trusted
                            and r.dets[k][r.frames[k].detection_index].owners != ("TARGET",)]
        self.assertEqual(trusted_on_other, [])
        self.assertTrue(r.bank_is_pure())

    def _emerging(self, n_after):
        fr = []
        for _ in range(40):            # OTHER fully behind the target, never detected
            fr.append(SS.SceneFrame({"TARGET": SS.box_at(300, 400, 40, 100), "OTHER": SS.box_at(300, 395, 38, 95)},
                                    order=("OTHER", "TARGET"), drop=("OTHER",)))
        for k in range(15):            # OTHER slides out sideways inside one union box (gradual growth)
            x = 300 + 40 * (k + 1) / 15
            fr.append(SS.SceneFrame({"TARGET": SS.box_at(300, 400, 40, 100), "OTHER": SS.box_at(x, 395, 38, 95)},
                                    order=("OTHER", "TARGET"), merge=(("TARGET", "OTHER"),)))
        for _ in range(n_after):
            fr.append(SS.SceneFrame({"TARGET": SS.box_at(300, 400, 40, 100), "OTHER": SS.box_at(340, 395, 38, 95)},
                                    order=("OTHER", "TARGET"), merge=(("TARGET", "OTHER"),)))
        return fr

    @staticmethod
    def _sub_threshold(r, k):
        """The union box at frame k is still below the merge-size tests against the single target (area and width):
        no size-based guard can see a second body there (KNOWN LIMIT, identical at the parent commit 9ae0ce2)."""
        d = [x for x in r.dets[k] if len(x.owners) > 1]
        if not d:
            return True
        w = d[0].box[2] - d[0].box[0]
        h = d[0].box[3] - d[0].box[1]
        return (w * h) / (40.0 * 100.0) < CFG.get("merged_area_ratio") and \
            (w / h) / (40.0 / 100.0) <= CFG.get("merged_aspect_band")[1]

    def test_unseen_second_body_emerging_gradually_is_never_trusted_as_the_target(self):
        # review H1 at the target level: a union box of the target and a never-detected person. swaps() lists every
        # frame trusted on a detection that is not exactly the target: allowed only while the union is still below
        # the merge-size tests (pre-existing limit); never once it is merge-sized, however long it lasts.
        fr = self._emerging(60)
        r = SS.run(fr, self.PPL)
        self.assertTrue(any(len(d.owners) > 1 for k in range(len(fr)) for d in r.dets[k]))
        late = [k for k in r.swaps() if not self._sub_threshold(r, k)]
        self.assertEqual(late, [], "trusted on a merge-sized two-person box")
        self.assertTrue(r.bank_is_pure())

    def test_split_after_a_gradual_unseen_merge_makes_no_silent_swap(self):
        for mode in ("other_leaves_right", "target_leaves_left"):
            fr = self._emerging(30)
            for k in range(40):
                if mode == "other_leaves_right":
                    tb, ob = SS.box_at(300, 400, 40, 100), SS.box_at(340 + 3 * (k + 1), 395, 38, 95)
                else:
                    tb, ob = SS.box_at(300 - 3 * (k + 1), 400, 40, 100), SS.box_at(340, 395, 38, 95)
                fr.append(SS.SceneFrame({"TARGET": tb, "OTHER": ob}, order=("OTHER", "TARGET"),
                                        merge=(("TARGET", "OTHER"),) if k < 3 else ()))
            r = SS.run(fr, self.PPL)
            self.assertEqual([k for k in r.swaps() if not self._sub_threshold(r, k)], [], mode)
            self.assertEqual([k for k in r.swaps() if k >= 85], [], mode + ": no swap after the split")
            self.assertTrue(r.bank_is_pure(), mode)


class TestReviewRound4(unittest.TestCase):
    """Independent review round 4 (656a272): M1 pre-cut memory as corroboration, L1 stale target anomaly tracklet,
    L2 anomaly clearing across a detection gap. Each failed on 656a272."""
    PPL = SS.make_people([("TARGET", (200, 60, 40), 1.0), ("OTHER", (40, 60, 200), 1.0)])

    def _after_cut(self, with_precut_person):
        from player_tracker_rb.camera import CUT, identity_motion
        fr = []
        for _ in range(30):     # shot 1: a person where the target will stand in shot 2 (or far away)
            b = box_at(400, 360, 32, 80) if with_precut_person else box_at(1000, 360, 32, 80)
            fr.append(Frame([cand(b)], [("OTHER",)]))
        cut = identity_motion(CUT, continuity_broken=True)
        for k, b in enumerate(widening(8, 15, n_hold=4, n_back=20)):   # shot 2: one body, pose widening
            fr.append(Frame([cand(b)], [("TARGET",)], camera=cut if k == 0 else None))
        rn = run(fr, fps=FPS)
        codes = "".join(code(r.observations[0]) for r in rn.results[30:])
        lineage = any(P.MERGE_LINEAGE in v.flags for r in rn.results[30:] for v in r.tracklets)
        return codes, lineage

    def test_memory_from_before_a_cut_never_corroborates_a_merge(self):
        # pixel positions from the previous shot mean nothing in the new one (position_valid False)
        base = self._after_cut(False)
        self.assertNotIn("M", base[0])
        self.assertIn("A", base[0])
        self.assertEqual(self._after_cut(True), base)

    def test_target_lost_during_an_anomaly_returns_through_one_reacquisition(self):
        b = lambda w: SS.box_at(300, 400, w, 100)
        ws = [40.0] * 30 + [40 + 52 * (k + 1) / 15 for k in range(15)] + [92.0] * 3   # merge-sized union
        ws += [56.0] * 45                       # x1.4 held: neither clears nor escalates; target ages to LOST
        ws += [40.0] * 60
        r = SS.run([SS.SceneFrame({"TARGET": b(w)}) for w in ws], self.PPL)
        reacq = [k for k, f in enumerate(r.frames) if f.reidentified_this_frame]
        self.assertEqual(len(reacq), 1, f"one return, not a re-broken one: {reacq}")
        self.assertEqual([k for k, f in enumerate(r.frames) if k > reacq[0] and f.decision.get("continuity_broken")],
                         [])
        self.assertGreater(r.frames[reacq[0]].target_episode, r.frames[29].target_episode,
                           "the return is a new (unverified) episode")
        self.assertEqual(len({r.frames[k].target_episode for k in range(reacq[0], len(ws))}), 1)
        self.assertTrue(r.bank_is_pure())
        self.assertEqual(r.swaps(), [])

    def test_anomaly_clearing_must_be_observed_continuously(self):
        clear_s = CFG.get("merged_anomaly_clear_s")
        for fps in (25.0, 30.0, 60.0):
            gap = max(1, math.floor(0.6 * clear_s * fps))   # short enough that the same tracklet continues
            def frames(n_after):
                fr = [Frame([cand(b)], [("TARGET",)]) for b in widening(int(fps), int(fps / 2), n_hold=int(fps / 5))]
                fr.append(Frame([cand(box_at(400, 360, 32, 80))], [("TARGET",)]))      # one near-single frame
                fr += [Frame([], []) for _ in range(gap)]                                # unseen
                fr += [Frame([cand(box_at(400, 360, 32, 80))], [("TARGET",)]) for _ in range(n_after)]
                obs = [x.observations[0] for x in run(fr, fps=fps).results if x.observations]
                return [code(o) for o in obs], [o.tracklet_id for o in obs]
            # the pre-gap frame plus the gap plus `short` observed frames span more than clear_s, but the frames
            # observed continuously after the gap span less
            short = math.ceil(clear_s * fps) - gap + 1           # one frame clear of the boundary
            self.assertGreater((gap + short) / fps, clear_s * 1.001)
            self.assertLess((short - 1) / fps, clear_s)
            codes, tids = frames(short)
            self.assertEqual(len(set(tids[-short - 2:])), 1, f"{fps} fps: same tracklet across the gap")
            self.assertEqual(codes[-short:], ["A"] * short, f"{fps} fps: a gap is not part of the clearing")
            codes, tids = frames(math.ceil(2 * clear_s * fps))    # positive control: sustained observation clears
            self.assertEqual(codes[-1], ".", f"{fps} fps")
            self.assertEqual(len(set(tids[-20:])), 1, f"{fps} fps")



class TestAuditRound4(unittest.TestCase):
    """Test audit round 4 (656a272): each anomaly rule is decided by its own check, at more than one frame rate."""
    BODY = box_at(400, 360, 32, 80)

    def _union(self, fps=FPS, n_hold=None, cy=360.0):
        # a single body whose box becomes union-sized (x2.3 area, below the uncorroborated cap): a SIZE_ANOMALY
        n_hold = int(fps / 5) if n_hold is None else n_hold
        return [Frame([cand(b)], [("TARGET",)]) for b in widening(int(fps), int(fps / 2), n_hold=n_hold, cy=cy)]

    def _codes(self, frames, fps=FPS):
        rn = run(frames, fps=fps)
        return "".join(code(widest(r)) if r.observations else "_" for r in rn.results), rn

    def _tail_never_clears(self, tail_frame, body=None):
        # union, then 0.1 s of single-size boxes (re-associated, clearing started but not done), then 0.5 s of
        # tail boxes that only one check says are not near single size, then the full single body again
        body = body or self.BODY
        fr = self._union(cy=(body[1] + body[3]) / 2)
        fr += [Frame([cand(body)], [("TARGET",)]) for _ in range(int(0.1 * FPS))]
        fr += [tail_frame() for _ in range(int(0.5 * FPS))]
        n = len(fr)
        fr += [Frame([cand(body)], [("TARGET",)]) for _ in range(int(0.5 * FPS))]
        codes, rn = self._codes(fr)
        first = codes.index("A")
        self.assertEqual(set(codes[first:n]), {"A"}, codes)
        self.assertNotIn("M", codes)
        self.assertTrue(codes.endswith("." * 5), "a full, single-size body ends it")
        self.assertEqual(len({r.observations[0].tracklet_id for r in rn.results}), 1)

    def test_partial_near_size_boxes_never_end_an_anomaly(self):
        # lower body hidden: area and aspect are near single size, only the height says partial
        part = (387.0, 320.0, 413.0, 376.0)                    # 26 x 56: area x0.57, aspect x1.16, height x0.7
        self.assertLess(56 / 80, CFG.get("population_partial_height_ratio"))
        self._tail_never_clears(lambda: Frame([cand(part)], [("TARGET",)]))

    def test_edge_touching_boxes_never_end_an_anomaly(self):
        # a body just below the top of the image; the tail boxes are the same body moved up to touch the edge
        m = CFG.get("detector_edge_touch_margin_px")
        body = box_at(400, m + 6 + 40, 32, 80)
        at_edge = (body[0], m - 1, body[2], m - 1 + 80)
        self._tail_never_clears(lambda: Frame([cand(at_edge)], [("TARGET",)]), body=body)

    def test_same_area_wide_boxes_never_end_an_anomaly(self):
        wide = (380.0, 336.0, 420.0, 400.0)                    # 40 x 64: area x1.0, aspect x1.56, height x0.8
        lo, hi = CFG.get("merged_aspect_band")
        self.assertLess((40 / 64) / (32 / 80), hi)              # not itself merge-sized
        self.assertGreater((40 / 64) / (32 / 80), CFG.get("merged_anomaly_clear_area_ratio"))
        self._tail_never_clears(lambda: Frame([cand(wide)], [("TARGET",)]))

    def test_anomaly_ends_after_clear_s_in_seconds_at_any_frame_rate(self):
        clear_s = CFG.get("merged_anomaly_clear_s")
        for fps in (25.0, 30.0, 60.0):
            fr = self._union(fps)
            k0 = len(fr)
            fr += [Frame([cand(self.BODY)], [("TARGET",)]) for _ in range(int(fps))]
            codes, _ = self._codes(fr, fps)
            end = codes.index(".", k0)
            self.assertGreaterEqual((end - k0) / fps, clear_s - 1e-9, (fps, codes))
            self.assertLessEqual((end - k0) / fps, clear_s + 1.01 / fps, (fps, codes))

    def test_a_body_touching_an_active_anomaly_makes_it_a_merge_at_once(self):
        # a not-near-single box (same area, wide aspect: no size reason of its own) while the anomaly is active;
        # a second detection, born this frame, starts touching it: no 1 s wait
        wide = (380.0, 336.0, 420.0, 400.0)
        fr = self._union() + [Frame([cand(wide)], [("TARGET",)]) for _ in range(3)]
        k = len(fr)
        fr += [Frame([cand(wide), cand(box_at(434, 368, 32, 80))], [("TARGET",), ("PERSON_A",)]) for _ in range(3)]
        rn = run(fr, fps=FPS)
        first_a = [code(r.observations[0]) for r in rn.results].index("A")
        self.assertLess((k - first_a) / FPS, MAX_S)
        target_obs = [o for o in rn.results[k].observations if o.bbox == wide][0]
        self.assertEqual(code(target_obs), "M", "touching second body during an anomaly")

    def test_growth_touching_a_tracked_body_hidden_this_frame_is_a_merge(self):
        # the other body is tracked beside the target and undetected exactly when the target's box jumps to a
        # merge size covering it: its live predicted box corroborates the merge
        a = box_at(445, 360, 32, 80)
        fr = [Frame([cand(self.BODY), cand(a)], [("TARGET",), ("PERSON_A",)]) for _ in range(30)]
        fr.append(Frame([cand(box_at(400, 360, 64, 80))], [("TARGET", "PERSON_A")]))
        rn = run(fr, fps=FPS)
        r = rn.results[30]
        other = rn.results[29].observations[1].tracklet_id
        self.assertIn(other, [v.tracklet_id for v in r.tracklets], "the other body's tracklet is live, not memory")
        self.assertNotIn(other, [c.tracklet_id for c in r.closed])
        self.assertEqual(code(r.observations[0]), "M")

    def test_a_jump_right_after_an_anomaly_is_judged_against_the_held_reference(self):
        # the anomaly lasted long enough for every pre-anomaly size to leave the reference window; just after it
        # ends no new reference exists yet: a merge-sized jump must still be flagged (and anomalous sizes must not
        # have entered the reference)
        self.assertGreater((9 + 17 + 6) / FPS, CFG.get("population_size_ref_window_s") - 0.1)
        fr = self._union(n_hold=17)
        k0 = len(fr)
        fr += [Frame([cand(self.BODY)], [("TARGET",)]) for _ in range(8)]
        fr += [Frame([cand(box_at(400, 360, 64, 80))], [("TARGET",)]) for _ in range(3)]
        codes, _ = self._codes(fr)
        self.assertNotIn("M", codes[:k0 + 8])
        self.assertEqual(codes[k0 + 7], ".", codes)
        self.assertNotIn(".", codes[k0 + 8:], codes)

    def test_held_reference_follows_a_camera_zoom(self):
        from player_tracker_rb.camera import motion_from_matrix
        from tests_rb._synth_people import W as SW, H as SH
        cx, cy = SW / 2, SH / 2
        n_zoom, total = 15, 1.5
        z = total ** (1.0 / n_zoom)
        zoom = motion_from_matrix([[z, 0, (1 - z) * cx], [0, z, (1 - z) * cy]], SW, SH)
        fr = [Frame([cand(box_at(cx, cy, 32, 80))], [("TARGET",)]) for _ in range(30)]
        fr += [Frame([cand(box_at(cx, cy, 32 + 42 * (k + 1) / 10, 80))], [("TARGET",)]) for k in range(10)]
        for k in range(n_zoom):                                 # the union box zooms with the camera
            s = z ** (k + 1)
            fr.append(Frame([cand(box_at(cx, cy, 74 * s, 80 * s))], [("TARGET",)], camera=zoom))
        fr += [Frame([cand(box_at(cx, cy, 32 * total, 80 * total))], [("TARGET",)]) for _ in range(int(FPS))]
        codes, rn = self._codes(fr)
        self.assertIn("A", codes)
        self.assertNotIn("M", codes, "a single body at the new zoom is not merge-sized")
        self.assertTrue(codes.endswith("." * 10), codes)
        self.assertEqual(len({r.observations[0].tracklet_id for r in rn.results}), 1)

    def test_no_velocity_is_learnt_from_anomalous_boxes(self):
        # a union growing to one side moves its centre; that is not the body's motion. Undetected frames inside the
        # anomaly predict the held box where it was, not drifting with the union's growth
        fr = [Frame([cand(self.BODY)], [("TARGET",)]) for _ in range(30)]
        for k in range(6):                                      # abrupt: every box after the still body is anomalous
            w = 64 + 2 * k
            fr.append(Frame([cand(box_at(400 + (w - 32) / 2, 360, w, 80))], [("TARGET",)]))
        k = len(fr)
        fr += [Frame([], []) for _ in range(5)]
        rn = run(fr, fps=FPS)
        self.assertEqual([code(r.observations[0]) for r in rn.results[30:k]], ["A"] * 6)
        last = rn.results[k - 1].observations[0].bbox
        v = rn.results[k + 4].tracklets[0]
        self.assertIsNone(v.observed_box)
        self.assertAlmostEqual((v.box[0] + v.box[2]) / 2, (last[0] + last[2]) / 2, delta=1.0)

    def test_target_bank_refuses_size_anomaly_boxes_on_its_own(self):
        # defence in depth: the target abstains on a SIZE_ANOMALY box, and its bank refuses one independently
        self.assertIn(P.SIZE_ANOMALY, T._BANK_REFUSE_FLAGS)


if __name__ == "__main__":
    unittest.main()

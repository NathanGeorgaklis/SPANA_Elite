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
        boxes = widening(30, 15, n_hold=8, n_back=10)            # anomaly < merged_anomaly_max_s, then back
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

    def test_second_persons_own_full_height_box_inside_the_merged_box_corroborates(self):
        # review H2: the union box M of A and B plus B's own full-height box nested inside M
        frames = [Frame([cand(box_at(400, 360, 32, 80))], [("TARGET",)]) for _ in range(30)]
        for k in range(10):
            w = 32 + 32 * (k + 1) / 10
            m = box_at(400 + (w - 32) / 2, 360, w, 80)
            bb = (m[2] - 30, m[1] + 2, m[2] - 1, m[3] - 1)            # B: ~96 % of M's height, at M's right side
            frames.append(Frame([cand(m), cand(bb)], [("TARGET", "PERSON_A"), ("PERSON_A",)]))
        rn = run(frames, fps=FPS)
        for r in rn.results[31:]:
            m = widest(r)
            self.assertTrue(m.second_body_possible)
            self.assertNotIn(P.SIZE_ANOMALY, m.flags)
        self.assertTrue(any(P.MERGED_SUSPECTED in widest(r).flags for r in rn.results[30:]))

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
        boxes = widening(30, 15, w0=40.0, w1=92.0, h=100.0, cx=300.0, cy=400.0, n_hold=6, n_back=30)
        r = SS.run([SS.SceneFrame({"TARGET": b}) for b in boxes], self.PPL)
        anomalous = [k for k, p in enumerate(r.pops) if P.SIZE_ANOMALY in p.observations[0].flags]
        self.assertTrue(anomalous)
        for k in anomalous:
            self.assertFalse(r.frames[k].identity_trusted, f"frame {k}: never trusted on a size anomaly")
            self.assertNotEqual(r.frames[k].decision.get("bank", {}).get("status"), "ACCEPTED")
        self.assertTrue(all(f.identity_trusted for f in r.frames[-10:]), "trusted again once the box is normal")
        self.assertEqual(r.frames[-1].target_episode, 1, "recovered by continuation, not by reacquisition")
        self.assertTrue(r.bank_is_pure())
        self.assertEqual(r.swaps(), [])

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


if __name__ == "__main__":
    unittest.main()

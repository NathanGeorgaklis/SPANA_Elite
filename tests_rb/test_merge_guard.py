# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2 DEV remediation instruction (Nathan, 2026-10-30 17:41, Part 2): diagnose why
#                 MERGED_DETECTION_SUSPECTED fires on isolated visible players; no relaxing the guard for coverage;
#                 distinguish true multi-person merge evidence from single-body box changes; overlapping people stay
#                 separate competitors; a detector artefact must not permanently poison the target tracklet; merge
#                 lineage stays conservative for real merges. R2 DEV finding F3 (R2/DEV_RUN_ef6d635).
"""Merged-guard regression tests (synthetic, neutral TARGET / PERSON_A; no benchmark knowledge).

Failure class reproduced (R2 DEV F3): a single isolated player's box widens gradually with pose (limbs, stride) at a
constant height. Against the 1 s median size reference that crossed the AREA_RATIO / ASPECT_WIDE merge-size tests
although no second body existed; the tracklet was closed (MERGED_CONTACT), a merged region was created that could
never resolve (it needs two bodies), and the same body's next tracklet carried MERGE_LINEAGE for life.

Size-only merge evidence now needs corroboration: a second body that could be in the box (another detection
touching it outside its nested group, another tracklet's predicted box, or a recently closed tracklet's last box), or
an abrupt growth against the body's immediately previous size (an unseen second body merging appears abruptly), or a
growth no single body can explain (merged_uncorroborated_area_ratio). Otherwise the box is a SIZE_ANOMALY: not a
merge, never an appearance sample.
"""
import unittest

from player_tracker_rb import population as P
from player_tracker_rb import target as T
from player_tracker_rb.config import TrackerConfig
from tests_rb import _synth_scene as SS
from tests_rb._synth_people import Frame, box_at, cand, run

CFG = TrackerConfig.default()
FPS = 30.0


def widening(n_still, n_grow, w0=32.0, w1=74.0, h=80.0, cx=400.0, cy=360.0, n_hold=10):
    """A single body: still at width w0, then widening linearly to w1 over n_grow frames (pose), height constant.
    Defaults reproduce the measured failure: x2.3 width in 0.5 s at 30 fps (about x1.06 per frame)."""
    out = [box_at(cx, cy, w0, h) for _ in range(n_still)]
    out += [box_at(cx, cy, w0 + (w1 - w0) * (k + 1) / n_grow, h) for k in range(n_grow)]
    out += [box_at(cx, cy, w1, h) for _ in range(n_hold)]
    return out


def tracklet_of(res, i):
    return res.observations[i].tracklet_id


class TestSingleBodySizeChange(unittest.TestCase):
    def test_gradual_pose_widening_of_an_isolated_body_is_not_a_merge(self):
        boxes = widening(30, 15)                      # x2.3 width over 0.5 s, at constant height
        rn = run([Frame([cand(b)], [("TARGET",)]) for b in boxes], fps=FPS)
        tids = {r.observations[0].tracklet_id for r in rn.results}
        self.assertEqual(len(tids), 1, "the body's tracklet must not be closed by its own pose change")
        self.assertIsNotNone(next(iter(tids)))
        for r in rn.results:
            o = r.observations[0]
            self.assertNotIn(P.MERGED_SUSPECTED, o.flags)
            self.assertEqual(o.merge_reasons, ())
            self.assertNotIn(P.IN_MERGED_REGION, o.flags)
            v = r.tracklet(o.tracklet_id)
            self.assertNotIn(P.MERGE_LINEAGE, v.flags)
        self.assertTrue(any(P.SIZE_ANOMALY in r.observations[0].flags for r in rn.results),
                        "the size change is recorded as an anomaly")
        for r in rn.results:
            o = r.observations[0]
            if P.SIZE_ANOMALY in o.flags:
                v = r.tracklet(o.tracklet_id)
                self.assertFalse(v.bank_eligible, "a size-anomalous box is never an appearance sample")
                self.assertIn(P.SIZE_ANOMALY, v.bank_refusals)

    def test_rule_is_frame_rate_free(self):
        for fps in (25.0, 30.0, 60.0):
            k = fps / 30.0
            boxes = widening(int(30 * k), int(15 * k), n_hold=int(10 * k))
            rn = run([Frame([cand(b)], [("TARGET",)]) for b in boxes], fps=fps)
            self.assertFalse(any(P.MERGED_SUSPECTED in r.observations[0].flags for r in rn.results), fps)

    def test_abrupt_growth_without_a_seen_second_body_is_still_a_merge(self):
        # an unseen second body merging into the box appears as an abrupt growth between consecutive observations
        a = box_at(400, 360, 32, 80)
        frames = [Frame([cand(a)], [("TARGET",)]) for _ in range(12)]
        frames.append(Frame([cand(box_at(410, 360, 58, 82))], [("TARGET", "PERSON_A")]))
        rn = run(frames, fps=FPS)
        o = rn.results[-1].observations[0]
        self.assertIn(P.MERGED_SUSPECTED, o.flags)
        self.assertIsNone(o.tracklet_id)

    def test_growth_no_single_body_explains_is_a_merge_even_when_gradual(self):
        cap = CFG.get("merged_uncorroborated_area_ratio")
        boxes = widening(30, 12, w0=30.0, w1=30.0 * (cap + 0.5))   # x3 in 0.4 s, about x1.1 per frame
        rn = run([Frame([cand(b)], [("TARGET",)]) for b in boxes], fps=FPS)
        self.assertTrue(any(P.MERGED_SUSPECTED in r.observations[0].flags for r in rn.results))


class TestCorroboratedMerges(unittest.TestCase):
    def test_gradual_growth_over_a_second_tracked_body_is_still_a_merge(self):
        # real contact: a second tracked person walks behind the target; the target's box grows over it gradually
        frames = []
        for k in range(30):
            frames.append(Frame([cand(box_at(400, 360, 32, 80)), cand(box_at(470 - 1.5 * k, 360, 32, 80))],
                                [("TARGET",), ("PERSON_A",)]))
        for k in range(20):
            w = 32 + 26 * (k + 1) / 20                                  # PERSON_A now hidden inside a growing box
            frames.append(Frame([cand(box_at(400 + (w - 32) / 2, 360, w, 80))], [("TARGET", "PERSON_A")]))
        rn = run(frames, fps=FPS)
        merged = [r for r in rn.results[30:] if P.MERGED_SUSPECTED in r.observations[0].flags]
        self.assertTrue(merged, "a box growing over a second known body is merge evidence")

    def test_gradual_growth_over_a_recently_hidden_body_is_still_a_merge(self):
        frames = [Frame([cand(box_at(400, 360, 32, 80)), cand(box_at(440, 360, 32, 80))],
                        [("TARGET",), ("PERSON_A",)]) for _ in range(30)]
        frames += [Frame([cand(box_at(400, 360, 32, 80))], [("TARGET",)]) for _ in range(5)]   # PERSON_A hidden
        for k in range(20):
            w = 32 + 30 * (k + 1) / 20
            frames.append(Frame([cand(box_at(400 + (w - 32) / 2, 360, w, 80))], [("TARGET", "PERSON_A")]))
        rn = run(frames, fps=FPS)
        self.assertTrue(any(P.MERGED_SUSPECTED in r.observations[0].flags for r in rn.results[35:]))

    def test_overlapping_genuine_people_stay_separate_competitors(self):
        frames = [Frame([cand(box_at(400, 360, 32, 80)), cand(box_at(420, 362, 32, 80))],
                        [("TARGET",), ("PERSON_A",)]) for _ in range(40)]
        rn = run(frames, fps=FPS)
        last = rn.results[-1]
        self.assertEqual(len({o.tracklet_id for o in last.observations}), 2)
        for o in last.observations:
            self.assertNotIn(P.MERGED_SUSPECTED, o.flags)
            self.assertTrue(o.second_body_possible, "an overlapping body is a possible second body")

    def test_real_merge_keeps_conservative_lineage(self):
        a, b = box_at(400, 360, 32, 80), box_at(440, 360, 32, 80)
        m = (a[0], a[1], b[2], b[3])
        frames = [Frame([cand(a), cand(b)], [("TARGET",), ("PERSON_A",)]) for _ in range(12)]
        frames += [Frame([cand(m)], [("TARGET", "PERSON_A")]) for _ in range(6)]
        frames += [Frame([cand(a)], [("TARGET",)]) for _ in range(6)]       # only one body re-emerges
        rn = run(frames, fps=FPS)
        for r in rn.results[12:18]:
            self.assertIn(P.MERGED_SUSPECTED, r.observations[0].flags)
        for r in rn.results[18:]:                  # only one body came out: never a clean, unflagged tracklet
            o = r.observations[0]
            v = r.tracklet(o.tracklet_id) if o.tracklet_id else None
            self.assertTrue(P.MERGED_SUSPECTED in o.flags or (v is not None and P.MERGE_LINEAGE in v.flags),
                            "a body from an unresolved real merge stays conservative")


class TestNestedAndPartial(unittest.TestCase):
    def test_nested_part_box_is_not_a_second_body(self):
        frames = []
        for b in widening(30, 15):
            part = box_at((b[0] + b[2]) / 2, b[1] + 20, 20, 36)          # a torso box inside the body box
            frames.append(Frame([cand(b), cand(part)], [("TARGET",), ("TARGET",)]))
        rn = run(frames, fps=FPS)
        for r in rn.results:
            body = max(r.observations, key=lambda o: (o.bbox[2] - o.bbox[0]) * (o.bbox[3] - o.bbox[1]))
            self.assertNotIn(P.MERGED_SUSPECTED, body.flags)

    def test_partial_box_recovering_to_full_is_not_a_merge_and_not_lineage(self):
        full = box_at(400, 360, 32, 80)
        part = (full[0], full[1], full[2], full[1] + 40)                 # lower body hidden: half height
        frames = [Frame([cand(full)], [("TARGET",)]) for _ in range(20)]
        frames += [Frame([cand(part)], [("TARGET",)]) for _ in range(8)]
        frames += [Frame([cand(full)], [("TARGET",)]) for _ in range(12)]
        rn = run(frames, fps=FPS)
        tids = {r.observations[0].tracklet_id for r in rn.results}
        self.assertEqual(len(tids), 1)
        for r in rn.results:
            self.assertNotIn(P.MERGED_SUSPECTED, r.observations[0].flags)


class TestTargetNotPoisoned(unittest.TestCase):
    def test_target_stays_trusted_through_its_own_pose_widening(self):
        ppl = SS.make_people([("TARGET", (200, 60, 40), 1.0)])
        boxes = widening(30, 15, w0=40.0, w1=92.0, h=100.0, cx=300.0, cy=400.0, n_hold=20)
        r = SS.run([SS.SceneFrame({"TARGET": b}) for b in boxes], ppl)
        self.assertEqual(r.states(), [T.TRACKING] * len(boxes))
        self.assertTrue(all(f.identity_trusted for f in r.frames))
        self.assertTrue(r.bank_is_pure())
        self.assertEqual(r.swaps(), [])
        anomalous = [k for k, p in enumerate(r.pops) if P.SIZE_ANOMALY in p.observations[0].flags]
        self.assertTrue(anomalous)
        for k in anomalous:
            self.assertNotEqual(r.frames[k].decision.get("bank", {}).get("status"), "ACCEPTED",
                                f"frame {k}: a size-anomalous box must not enter the target bank")


if __name__ == "__main__":
    unittest.main()

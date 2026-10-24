# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R1-E independent test audit (mutation testing: 23 of 70 plausible mutants survived the first suite);
#                 R1/R1_DESIGN_DECISIONS.md D06, D10, D11, D16
"""R1-E tests added from the independent test audit. Each test kills one or more mutants that the first suite let
survive (contested splits, contact during a gap, camera compensation of remembered state, cut invalidation of memory,
bank refusals, MULTI_CLAIM recency, scale veto, PREDICTED/HELD in seconds, camera masks and bounds), plus the
regressions for the defects the audit found (merged-region membership, camera inlier denominator, residual mask)."""
import unittest

from player_tracker_rb import population as P
from player_tracker_rb.camera import CUT, UNKNOWN, identity_motion
from player_tracker_rb.config import TrackerConfig
from player_tracker_rb.motion import DETECTION, HELD, PREDICTED
from tests_rb._synth_people import Frame, box_at, cand, combine, pan, run, walker, zoom

FPS = 30.0


class Probe(unittest.TestCase):
    def test_contact_during_gap_bars_reassociation(self):
        """TARGET stands still and is undetected for 3 frames while PERSON_A walks across its position; PERSON_A is
        clearly past when TARGET is detected again. The re-detection must NOT continue the old tracklet."""
        frames = []
        for k in range(20):
            dets, owners = [], []
            if k not in (10, 11, 12):
                dets.append(cand(box_at(300, 360, 32, 80))); owners.append(("TARGET",))
            dets.append(cand(box_at(135 + 15 * k, 424, 32, 80))); owners.append(("PERSON_A",))  # covers ~20%
            frames.append(Frame(dets, owners))
        rn = run(frames)
        old = rn.results[9].observations[0].tracklet_id
        self.assertIn(P.CONTACT_DURING_GAP, rn.results[11].tracklet(old).flags)
        new = rn.results[13].observations[0]
        self.assertNotEqual(new.tracklet_id, old)
        self.assertIn(old, new.possible_predecessors)

    def test_open_space_gap_reassociates_without_contact_flag(self):
        rn = run(walker(20, FPS, drop=(10, 11, 12)))
        v = rn.results[11].tracklets[0]
        self.assertNotIn(P.CONTACT_DURING_GAP, v.flags)

    def test_cut_invalidates_previously_closed_memory(self):
        """A tracklet closed BEFORE a cut (still in memory) must lose its position at the cut."""
        frames = walker(10, FPS, drop=tuple(range(5, 10)))                 # closes? no: hidden < 0.5 s
        frames = walker(30, FPS, drop=tuple(range(5, 30)))[:25]            # closes by UNMATCHED_TIMEOUT ~k=20
        cut = identity_motion(CUT, reasons=("HISTOGRAM_CHANGE",), continuity_broken=True)
        frames.append(Frame([], [], cut))
        frames.append(Frame([cand(box_at(300, 360, 32, 80))], [("PERSON_B",)]))
        rn = run(frames)
        mem = rn.population.recently_closed()
        self.assertTrue(mem)
        self.assertTrue(all(not m.position_valid for m in mem))
        comp = rn.population.competitors(box_at(300, 360, 32, 80), 80.0)
        rc = [c for c in comp if c.kind == P.RECENTLY_CLOSED]
        self.assertTrue(rc and all(c.distance_h is None for c in rc))

    def test_memory_follows_pan(self):
        frames = walker(25, FPS, vx_h_per_s=0.0, drop=tuple(range(5, 25)))   # closes by timeout
        for k in range(10):
            frames.append(Frame([], [], pan(-30.0)))
        rn = run(frames)
        mem = [m for m in rn.population.recently_closed() if m.position_valid]
        self.assertTrue(mem)
        cx = (mem[0].last_box[0] + mem[0].last_box[2]) / 2
        self.assertAlmostEqual(cx, 300 - 30.0 * 10, delta=1.0)
        comp = rn.population.competitors(box_at(0, 360, 32, 80), 80.0)
        self.assertTrue(any(c.kind == P.RECENTLY_CLOSED and c.position_valid for c in comp))

    def test_merged_region_follows_pan(self):
        """A merge seen under a pan: the region moves with the image, so a box at the merged bodies' new image
        position is still IN_MERGED_REGION."""
        a, b = box_at(400, 360, 32, 80), box_at(440, 360, 32, 80)
        frames = [Frame([cand(a), cand(b)], [("PERSON_A",), ("PERSON_B",)]) for _ in range(12)]
        m = (a[0], a[1], b[2], b[3])
        frames.append(Frame([cand(m)], [("PERSON_A", "PERSON_B")]))
        shift = 0.0
        for k in range(3):
            shift -= 40.0
            frames.append(Frame([cand((m[0] + shift, m[1], m[2] + shift, m[3]))], [("PERSON_A", "PERSON_B")],
                                pan(-40.0)))
        rn = run(frames)
        for r in rn.results[13:]:
            o = r.observations[0]
            self.assertIn(P.IN_MERGED_REGION, o.flags)
            self.assertIn(P.MERGED_SUSPECTED, o.flags)

    def test_zoom_keeps_size_reference(self):
        """A steady zoom-in with a person standing still: the growing box must not become AREA_RATIO merged."""
        frames, bx = [], box_at(640, 360, 32, 80)
        for k in range(40):
            cam = None
            if k > 0:
                cam = zoom(1.03)
                bx = cam.warp_box(bx)
            frames.append(Frame([cand(bx)], [("PERSON_A",)], cam))
        rn = run(frames)
        for r in rn.results:
            self.assertFalse(any(P.MERGED_SUSPECTED in o.flags for o in r.observations))
        self.assertEqual(len({v.tracklet_id for r in rn.results for v in r.tracklets}), 1)

    def test_crossing_contested_births_flagged_and_refused(self):
        frames = [Frame([cand(box_at(400, 360, 32, 80)), cand(box_at(424, 360, 32, 80))],
                        [("PERSON_A",), ("PERSON_B",)]) for _ in range(10)]
        frames.append(Frame([cand(box_at(411, 360, 32, 80)), cand(box_at(413, 372, 32, 80))],
                            [("PERSON_A",), ("PERSON_B",)]))   # both boxes fit both predictions about equally
        rn = run(frames)
        self.assertEqual(sorted(rn.closed().values()), [P.CONTACT_AMBIGUITY] * 2)
        evs = [(r, e) for r in rn.results for e in r.contact_events if e.kind == P.EVENT_AMBIGUOUS]
        self.assertTrue(evs)
        for r, e in evs:
            for tid in e.births:
                v = r.tracklet(tid)
                self.assertIn(P.CONTESTED, v.flags)
                self.assertIn(P.CONTESTED, v.bank_refusals)
                self.assertTrue(set(v.possible_predecessors) >= set(e.tracklets))

    def test_rival_tracklet_alone_contests(self):
        """Two tracklets whose predictions both match ONE detection about equally (one person hidden behind the
        other): the link is contested, not given silently to one."""
        a = walker(12, FPS, x0=400, vx_h_per_s=0.0, name="PERSON_A")
        b = walker(12, FPS, x0=404, vx_h_per_s=0.0, name="PERSON_B", drop=(11,))
        frames = combine(a, b)
        rn = run(frames, cfg=TrackerConfig.with_overrides({"population_duplicate_iou": 0.99}))
        r = rn.results[11]
        self.assertTrue(any(e.kind == P.EVENT_AMBIGUOUS for e in r.contact_events) or
                        any(P.MERGED_SUSPECTED in o.flags for o in r.observations))

    def test_unknown_camera_tracklet_not_bank_eligible(self):
        cfg = TrackerConfig.with_overrides({"camera_unknown_breaks_continuity": False})
        frames = walker(12, FPS)
        frames[11] = Frame(list(frames[11].dets), [("TARGET",)], identity_motion(UNKNOWN, continuity_broken=False))
        rn = run(frames, cfg=cfg)
        v = rn.results[11].tracklets[0]
        self.assertTrue(v.observed)
        self.assertIn(P.CAMERA_UNVERIFIED, v.bank_refusals)
        self.assertFalse(v.bank_eligible)

    def test_hidden_uncompensated_flag(self):
        cfg = TrackerConfig.with_overrides({"camera_unknown_breaks_continuity": False})
        frames = walker(12, FPS, drop=(11,))
        frames[11] = Frame([], [], identity_motion(UNKNOWN, continuity_broken=False))
        rn = run(frames, cfg=cfg)
        v = rn.results[11].tracklets[0]
        self.assertFalse(v.camera_compensated)
        self.assertIn(P.CAMERA_UNCOMPENSATED, v.flags)


class ProbeZoomMerge(unittest.TestCase):
    def test_persistent_merge_under_zoom_out_stays_merged(self):
        """Two people merged into one box while the camera zooms out: the merged region's size reference must
        shrink with the image, so the merge keeps its evidence and the merged box never becomes a tracklet."""
        a, b = box_at(400, 360, 32, 80), box_at(440, 360, 32, 80)
        frames = [Frame([cand(a), cand(b)], [("PERSON_A",), ("PERSON_B",)]) for _ in range(12)]
        m = (a[0], a[1], b[2], b[3])
        frames.append(Frame([cand(m)], [("PERSON_A", "PERSON_B")]))
        for k in range(40):
            cam = zoom(0.985)
            m = cam.warp_box(m)
            frames.append(Frame([cand(m)], [("PERSON_A", "PERSON_B")], cam))
        rn = run(frames)
        for r in rn.results[12:]:
            o = r.observations[0]
            self.assertIn(P.MERGED_SUSPECTED, o.flags, f"rel {r.rel_frame}")
            self.assertIsNone(o.tracklet_id)


class ProbeRivalTracklet(unittest.TestCase):
    def test_detection_fitting_two_tracklets_is_contested(self):
        """PERSON_B is clearly re-detected; a second box lies between PERSON_A's and PERSON_B's predictions and fits
        both about equally. PERSON_A must not silently continue on it."""
        frames = [Frame([cand(box_at(400, 360, 32, 80)), cand(box_at(430, 360, 32, 80))],
                        [("PERSON_A",), ("PERSON_B",)]) for _ in range(10)]
        frames.append(Frame([cand(box_at(414, 360, 32, 80)), cand(box_at(431, 360, 32, 80))],
                            [("PERSON_A",), ("PERSON_B",)]))
        rn = run(frames)
        a_id = rn.results[9].observations[0].tracklet_id
        r = rn.results[10]
        self.assertTrue(any(e.kind == P.EVENT_AMBIGUOUS and a_id in e.tracklets for e in r.contact_events))
        self.assertNotEqual(r.observations[0].tracklet_id, a_id)


class ProbeRivalDetection(unittest.TestCase):
    def test_two_boxes_fitting_one_tracklet_equally_is_contested(self):
        """PERSON_A is tracked alone; then two separate boxes appear that both fit its prediction about equally
        (a second person steps out from behind). Neither may silently inherit PERSON_A's tracklet."""
        frames = [Frame([cand(box_at(400, 360, 32, 80))], [("PERSON_A",)]) for _ in range(10)]
        frames.append(Frame([cand(box_at(393, 360, 32, 80)), cand(box_at(409, 360, 32, 80))],
                            [("PERSON_A",), ("PERSON_B",)]))
        rn = run(frames)
        a_id = rn.results[9].observations[0].tracklet_id
        r = rn.results[10]
        self.assertEqual(rn.closed().get(a_id), P.CONTACT_AMBIGUITY)
        self.assertTrue(all(o.tracklet_id != a_id for o in r.observations))
        for o in r.observations:
            self.assertIn(P.CONTESTED, o.flags)
            self.assertIn(a_id, o.possible_predecessors)


class ProbeBank(unittest.TestCase):
    def test_small_partly_occluded_person_refused_by_cover_not_iou(self):
        """A small far body half behind a large near one: IoU is tiny (the occluder is big) but half the far body
        is covered, so it must not be sampled."""
        near = box_at(400, 400, 90, 200)
        far = (near[2] - 8, 380.0, near[2] + 8, 420.0)                    # 16 x 40, half inside the near box
        rn = run([Frame([cand(near), cand(far)], [("PERSON_A",), ("PERSON_B",)]) for _ in range(3)])
        r = rn.results[-1]
        o = r.observations[1]
        self.assertLess(o.max_iou_other, TrackerConfig.default().get("bank_contact_iou"))
        v = r.tracklet(o.tracklet_id)
        self.assertFalse(v.bank_eligible)
        self.assertIn(P.NOT_ISOLATED, v.bank_refusals)

    def test_box_containing_tiny_nested_box_is_refused(self):
        near = box_at(400, 400, 90, 200)
        tiny = box_at(400, 330, 16, 40)                                    # nested, IoU and cover < 0.05
        rn = run([Frame([cand(near), cand(tiny)], [("PERSON_A",), ("PERSON_A",)]) for _ in range(3)])
        r = rn.results[-1]
        v = r.tracklet(r.observations[0].tracklet_id)
        self.assertIn(P.CONTAINS_OTHER, v.bank_refusals)
        self.assertFalse(v.bank_eligible)


class ProbeLongHiddenClaim(unittest.TestCase):
    def test_box_over_long_hidden_body_is_not_multi_claim(self):
        """PERSON_B vanished > population_max_gap_s ago (still alive as HIDDEN); PERSON_A then walks over its last
        position. One tracked person is not a merged box."""
        frames = []
        for k in range(20):
            dets, owners = [cand(box_at(330 + 5.33 * k, 360, 32, 80))], [("PERSON_A",)]
            if k < 5:
                dets.append(cand(box_at(430, 360, 32, 80))); owners.append(("PERSON_B",))
            frames.append(Frame(dets, owners))
        rn = run(frames)
        ids = {r.observations[0].tracklet_id for r in rn.results}
        self.assertEqual(len(ids), 1)
        for r in rn.results:
            self.assertNotIn(P.MERGED_SUSPECTED, r.observations[0].flags)


class ProbeCameraMasks(unittest.TestCase):
    def _two(self, prev_boxes, cur_boxes):
        from player_tracker_rb import camera as C
        from tests_rb.test_camera import WORLD, view, canvas, CFG, DT
        fg = canvas(9, 900, 600)
        imgs = []
        for cx, ox in ((1200, 300), (1210, 240)):
            img = view(WORLD, cx, 700, 1.0).copy()
            img[60:660, ox:ox + 900] = fg
            imgs.append(img)
        est = C.CameraMotionEstimator(CFG)
        est.update(imgs[0], prev_boxes, DT)
        return est.update(imgs[1], cur_boxes, DT)

    def test_prev_frame_person_mask_used(self):
        """The big foreground person is detected in the previous frame but missed in this one: the previous-frame
        mask alone must keep its features out of the background estimate."""
        m = self._two([(300, 60, 1200, 660)], [])
        self.assertEqual(m.status, "ESTIMATED")
        self.assertAlmostEqual(m.shift_px[0], -10.0, delta=0.5)

    def test_current_frame_person_mask_used(self):
        """A large person steps in front of the camera between two frames (absent before): background points it now
        covers have no valid flow; the current-frame mask must discard them."""
        from player_tracker_rb import camera as C
        from tests_rb.test_camera import WORLD, view, canvas, CFG, DT
        fg = canvas(9, 500, 600)
        a = view(WORLD, 1200, 700, 1.0)
        b = view(WORLD, 1210, 700, 1.0).copy()
        b[60:660, 240:740] = fg
        out = []
        for boxes in ([(240, 60, 740, 660)], []):
            est = C.CameraMotionEstimator(CFG)
            est.update(a, [], DT)
            out.append(est.update(b, boxes, DT))
        self.assertEqual(out[0].status, C.ESTIMATED, out[0].reasons)
        self.assertAlmostEqual(out[0].shift_px[0], -10.0, delta=0.5)
        self.assertLess(out[0].n_tracked, out[1].n_tracked)       # points landing on the person were discarded

    def test_rotation_bound(self):
        import cv2
        import numpy as np
        from player_tracker_rb import camera as C
        from tests_rb.test_camera import WORLD, view, DT
        cfg = TrackerConfig.with_overrides({"camera_max_rotation_rad_per_s": 0.1})   # 0.0033 rad per frame
        a = view(WORLD, 1200, 700, 1.0)
        rot = cv2.getRotationMatrix2D((640, 360), 1.0, 1.0)                           # 1 degree = 0.017 rad
        b = cv2.warpAffine(a, rot, (1280, 720))
        est = C.CameraMotionEstimator(cfg)
        est.update(a, [], DT)
        m = est.update(b, [], DT)
        self.assertIn(C.IMPLAUSIBLE_ROTATION, m.reasons)
        self.assertNotEqual(m.status, C.ESTIMATED)


class ProbeMisc(unittest.TestCase):
    def test_single_sample_is_not_an_established_size_reference(self):
        """A person first seen once with a smaller box (e.g. partly hidden), then fully: a one-frame size is not a
        reference, so the larger box is not AREA_RATIO-merged (growth kept inside the motion scale veto)."""
        frames = [Frame([cand(box_at(400, 360, 40, 80))], [("PERSON_A",)]),
                  Frame([cand(box_at(400, 370, 52, 100))], [("PERSON_A",)])]
        rn = run(frames)
        o = rn.results[1].observations[0]
        self.assertNotIn(P.MERGED_SUSPECTED, o.flags)
        self.assertEqual(o.tracklet_id, rn.results[0].observations[0].tracklet_id)

    def test_scale_veto_blocks_association(self):
        """A box whose height jumps 37% in one frame overlaps the prediction well (IoU ~0.7) but is not the same
        body by motion: the old tracklet must not continue on it."""
        frames = [Frame([cand(box_at(400, 360, 32, 80))], [("PERSON_A",)]) for _ in range(12)]
        frames.append(Frame([cand(box_at(400, 375, 30, 110))], [("PERSON_B",)]))
        rn = run(frames)
        self.assertNotEqual(rn.results[-1].observations[0].tracklet_id, rn.results[-2].observations[0].tracklet_id)

    def test_predicted_to_held_transition_in_seconds(self):
        from player_tracker_rb.motion import MotionModel
        cfg = TrackerConfig.default()
        mm = MotionModel(cfg)
        st = mm.start(box_at(300, 360, 32, 80), 0.0)
        hz = cfg.get("motion_max_extrapolation_s")
        self.assertEqual(mm.predict(st, hz).source, PREDICTED)
        self.assertEqual(mm.predict(st, hz + 0.005).source, HELD)

    def test_camera_shift_bound_scales_with_dt(self):
        from player_tracker_rb import camera as C
        from tests_rb.test_camera import WORLD, view
        cfg = TrackerConfig.with_overrides({"camera_max_shift_frac_per_s": 0.5})   # 21 px at 1/30 s, 10.7 at 1/60
        res = {}
        for dt in (1 / 30, 1 / 60):
            est = C.CameraMotionEstimator(cfg)
            est.update(view(WORLD, 1200, 700, 1.0), [], dt)
            res[dt] = est.update(view(WORLD, 1215, 700, 1.0), [], dt)
        self.assertEqual(res[1 / 30].status, C.ESTIMATED)
        self.assertIn(C.IMPLAUSIBLE_SHIFT, res[1 / 60].reasons)


class ProbeAuditDefects(unittest.TestCase):
    def test_three_person_merge_split_keeps_links_and_refuses_bank(self):
        """Defect 1: a body coming out of a 3-person merged box has IoU < 0.3 with the region, but lies inside it.
        It must count as IN the region: while fewer bodies than went in have come out, each box stays
        MERGED_SUSPECTED (UNRESOLVED); after the hold lapses, the tracklets started there inherit the claimants as
        possible predecessors and MERGE_LINEAGE (never sampled)."""
        boxes = [box_at(380, 360, 32, 80), box_at(420, 360, 32, 80), box_at(460, 360, 32, 80)]
        names = [("PERSON_A",), ("PERSON_B",), ("PERSON_C",)]
        frames = [Frame([cand(b) for b in boxes], names) for _ in range(12)]
        m = (boxes[0][0], boxes[0][1], boxes[2][2], boxes[2][3])
        frames += [Frame([cand(m)], [("PERSON_A", "PERSON_B", "PERSON_C")]) for _ in range(3)]
        frames += [Frame([cand(boxes[0]), cand(boxes[2])], [names[0], names[2]]) for _ in range(25)]
        rn = run(frames)
        before = {v.tracklet_id for v in rn.results[11].tracklets}
        split = rn.results[15]
        for o in split.observations:
            self.assertIn(P.IN_MERGED_REGION, o.flags)
            self.assertIn(P.MERGED_SUSPECTED, o.flags)
        born = [v for r in rn.results[15:] for v in r.tracklets if v.status == P.NEW]
        self.assertTrue(born, "coverage: tracked again once the hold has lapsed")
        for v in born:
            self.assertTrue(set(v.possible_predecessors) & before)
            self.assertIn(P.MERGE_LINEAGE, v.flags)
        for r in rn.results[15:]:
            for v in r.tracklets:
                self.assertFalse(v.bank_eligible)

    def test_three_person_region_counts_three_bodies(self):
        """A region formed from three distinct bodies stays unresolved while only two have come out."""
        boxes = [box_at(380, 360, 32, 80), box_at(420, 360, 32, 80), box_at(460, 360, 32, 80)]
        frames = [Frame([cand(b) for b in boxes], [("PERSON_A",), ("PERSON_B",), ("PERSON_C",)]) for _ in range(12)]
        m = (boxes[0][0], boxes[0][1], boxes[2][2], boxes[2][3])
        frames += [Frame([cand(m)], [("PERSON_A", "PERSON_B", "PERSON_C")]) for _ in range(2)]
        frames += [Frame([cand(boxes[0]), cand(boxes[2])], [("PERSON_A",), ("PERSON_C",)]) for _ in range(2)]
        rn = run(frames)
        for v in rn.results[-1].tracklets:
            if v.observed:
                self.assertIn(P.MERGE_LINEAGE, v.flags)          # a third body was never seen apart again


class ProbeCameraDenominator(unittest.TestCase):
    def test_large_masked_person_does_not_fail_the_estimate(self):
        """Defect 2: background features discarded by the person mask are not flow failures. A large person in
        front of a panning camera must not turn a good estimate into UNKNOWN (which would break continuity)."""
        from player_tracker_rb import camera as C
        from tests_rb.test_camera import CFG, DT, WORLD, canvas, view
        fg = canvas(9, 900, 600)
        imgs = []
        for cx, ox in ((1200, 300), (1210, 240)):
            img = view(WORLD, cx, 700, 1.0).copy()
            img[60:660, ox:ox + 900] = fg
            imgs.append(img)
        est = C.CameraMotionEstimator(CFG)
        est.update(imgs[0], [(300, 60, 1200, 660)], DT)
        m = est.update(imgs[1], [(240, 60, 1140, 660)], DT)
        self.assertEqual(m.status, C.ESTIMATED, m.reasons)
        self.assertAlmostEqual(m.shift_px[0], -10.0, delta=0.5)


class MergeResolutionByEmergence(unittest.TestCase):
    """R1-F finding (first failing layer: population). A person in front is undetected while overlapping the person
    behind, so the visible box 'absorbs' them (a correct MULTI_CLAIM merge). When the hidden person re-emerges, its
    box first appears partly behind the merged box. The region must resolve (bodies first seen after the merge and
    overlapping it are the bodies coming out of it), and tracklets born in the region lose MERGE_LINEAGE. A body
    merely beside the region, or a box still merge-sized, must not resolve it (review H2)."""

    def _scene(self, passer_by=False):
        frames = []
        for k in range(70):
            ax = 300 + 5 * k
            bx = 650 - 5 * k
            a = box_at(ax, 400, 40, 100)
            b = box_at(bx, 415, 40, 100)
            dets, owners = [cand(a)], [("PERSON_A",)]
            ov = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
            if ov <= 0:
                dets.append(cand(b))
                owners.append(("PERSON_B",))
            if passer_by:
                dets.append(cand(box_at(900, 400, 40, 100)))
                owners.append(("PERSON_C",))
            frames.append(Frame(dets, owners))
        return frames

    def test_emerging_body_resolves_the_merge_and_clears_lineage(self):
        rn = run(self._scene())
        merged = [r.rel_frame for r in rn.results if any(P.MERGED_SUSPECTED in o.flags for o in r.observations)]
        self.assertTrue(merged, "the absorbed person must first produce a merge")
        last = rn.results[-1]
        self.assertEqual(sum(v.observed for v in last.tracklets), 2)
        for v in last.tracklets:
            self.assertNotIn(P.MERGE_LINEAGE, v.flags)
        self.assertFalse(any(P.MERGED_SUSPECTED in o.flags for o in last.observations))

    def test_a_body_that_existed_before_the_merge_does_not_resolve_it(self):
        """A passer-by that was already tracked elsewhere when the merge formed is not a body coming out of it."""
        frames = []
        for k in range(40):
            a, b = box_at(400, 400, 40, 100), box_at(436, 400, 40, 100)
            c = box_at(700 - 6 * k, 380, 40, 100)              # walks across, touching the merged box later
            if k < 10:
                frames.append(Frame([cand(a), cand(b), cand(c)], [("PERSON_A",), ("PERSON_B",), ("PERSON_C",)]))
            else:
                m = (a[0], a[1], b[2], b[3])
                frames.append(Frame([cand(m), cand(c)], [("PERSON_A", "PERSON_B"), ("PERSON_C",)]))
        rn = run(frames, cfg=TrackerConfig.with_overrides({"population_duplicate_iou": 0.99}))
        for r in rn.results[10:]:
            o = r.observations[0]
            self.assertIn(P.MERGED_SUSPECTED, o.flags, f"rel {r.rel_frame}")

    def _region_then(self, after):
        """Two bodies, then 10 frames merged into one box, then `after(k)` -> (dets, owners) per frame."""
        frames = []
        for k in range(60):
            a = box_at(400, 400, 40, 100)
            if k < 10:
                frames.append(Frame([cand(a), cand(box_at(470, 400, 40, 100))], [("PERSON_A",), ("PERSON_B",)]))
            elif k < 20:
                frames.append(Frame([cand(box_at(418, 400, 76, 100))], [("PERSON_A", "PERSON_B")]))
            else:
                frames.append(Frame(*after(k)))
        return run(frames, cfg=TrackerConfig.with_overrides({"population_duplicate_iou": 0.99}))

    def test_body_emerging_partly_behind_the_region_resolves_it(self):
        """The re-emerging body is still partly inside the merged box when first detected."""
        rn = self._region_then(lambda k: ([cand(box_at(400, 400, 40, 100)), cand(box_at(441, 400, 40, 100))],
                                          [("PERSON_A",), ("PERSON_B",)]))
        last = rn.results[-1]
        self.assertEqual(sum(v.observed for v in last.tracklets), 2)
        for v in last.tracklets:
            self.assertNotIn(P.MERGE_LINEAGE, v.flags)

    def test_newcomer_merely_beside_the_region_does_not_resolve_it(self):
        """R1-F review H2: a body first seen just OUTSIDE the merged box (no overlap) is not a body coming out of
        it; the region must stay merged so the still-merged box cannot be trusted as one person."""
        for first in (box_at(418, 400, 76, 100), box_at(400, 400, 40, 100)):   # still merged / one body showing
            with self.subTest(first=first):
                rn = self._region_then(lambda k: ([cand(first), cand(box_at(478, 400, 40, 100))],
                                                  [("PERSON_A", "PERSON_B"), ("PERSON_C",)]))
                for r in rn.results[20:]:
                    lin = [v for v in r.tracklets if v.observed and P.MERGE_LINEAGE in v.flags]
                    self.assertTrue(P.MERGED_SUSPECTED in r.observations[0].flags or lin, f"rel {r.rel_frame}")

    def test_merge_sized_box_in_region_does_not_count_as_a_resolved_body(self):
        """R1-F review H2: one body emerging beside a box that is STILL two-person sized must not resolve a
        two-body region by counting the merged box as the second body."""
        rn = self._region_then(lambda k: ([cand(box_at(418, 400, 76, 100)), cand(box_at(450, 400, 40, 100))],
                                          [("PERSON_A", "PERSON_B"), ("PERSON_C",)]))
        for r in rn.results[20:]:
            self.assertIn(P.MERGED_SUSPECTED, r.observations[0].flags, f"rel {r.rel_frame}")

# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2 residual same-kit contrast diagnosis (Nathan, 2026-11-07 14:53). Real-footage mechanisms
#                 (native 1080p60): G1_A f179 the target's own leg box (0.13-0.37 x his score, on his feet line,
#                 inside his box for 0.3 s) vanished for one detector frame and rule 4b ended his tracklet as
#                 CONTAINED_BODY_VANISHED, so the teammate-alongside stretch ended on a reacquisition that same-kit
#                 contrast can never verify; G2_D f102-130 a merged pair of players whose feet line was 0.92-0.98
#                 body heights above the target's (behind him, near his head) overlapped his box by 5-7 % and
#                 counted as the size anomaly's second-body contact, breaking continuity (SIZE_ANOMALY_ENDED ->
#                 reacquisition -> unverified episode); G2_A f125 the same with a farther player (feet gap 0.91).
"""Residual same-kit contrast stage: both residuals were upstream of the contrast test. (1) Rule 4b counts only a
CREDIBLE contained body (score above population_weak_part_score_ratio x its host): a weak leg box vanishing no
longer ends the tracklet; a credible crouched body vanishing still does. (2) A touching box is a second-body
indication for an active size anomaly only on a compatible feet line (|feet gap| <= separation_min_gap_h x the
host's height): bodies at different depths overlap in the image, not on the ground; partial or edge-clipped boxes
still count (fail closed). The contrast margin, the separation certificate, ReID and the appearance floor are
untouched."""
import math
import unittest

from player_tracker_rb import population as P
from player_tracker_rb.config import TrackerConfig
from tests_rb._synth_people import Frame, box_at, cand, pan, run

FPS = 30.0
CFG = TrackerConfig.default()
MAX_S = CFG.get("merged_anomaly_max_s")
GAP = CFG.get("separation_min_gap_h")
RATIO = CFG.get("population_weak_part_score_ratio")
HIDDEN_S = CFG.get("population_hidden_min_s")


def code(o):
    return "M" if P.MERGED_SUSPECTED in o.flags else ("A" if P.SIZE_ANOMALY in o.flags else ".")


def widest(r):
    return max(r.observations, key=lambda o: (o.bbox[2] - o.bbox[0]))


def _host_with_feet_box(n, score, n_box, host_w=48.0, host_h=118.0, teammate=True):
    """A host walking right with a feet-line box (0.85 of its width, 0.6 of its height) wholly inside it for the
    first n_box frames, then gone; a teammate (own tracklet) alongside, boxes touching by a few px, throughout."""
    frames = []
    for k in range(n):
        cx = 300 + 1.5 * k
        host = box_at(cx, 400, host_w, host_h)
        dets, owners = [cand(host, 0.8)], [("TARGET",)]
        if k < n_box:
            w = 0.85 * host_w
            dets.append(cand((cx - w / 2, host[3] - 0.6 * host_h, cx + w / 2, host[3] - 1.0), score))
            owners.append(("TARGET",))
        if teammate:
            dets.append(cand(box_at(cx + host_w - 2.0 + 25.0, 410, 50.0, 130.0), 0.87))
            owners.append(("TEAMMATE",))
        frames.append(Frame(dets, owners))
    return frames


class VanishedContainedBody(unittest.TestCase):
    def test_weak_leg_box_vanishing_does_not_end_the_tracklet(self):
        # G1_A f179 geometry: a 0.13-0.37 x host score leg box on the feet line, inside for > hidden_min_s
        n_box = int(math.ceil(HIDDEN_S * FPS)) + 10
        for sc in (0.1, 0.3, 0.8 * RATIO * 0.9):
            with self.subTest(part_score=sc):
                rn = run(_host_with_feet_box(n_box + 20, sc, n_box), fps=FPS)
                tid0 = rn.results[0].observations[0].tracklet_id
                for k, r in enumerate(rn.results):
                    self.assertEqual(r.observations[0].tracklet_id, tid0, f"frame {k}: host tracklet changed")
                self.assertFalse(any(c.reason == P.CONTACT_AMBIGUITY for r in rn.results for c in r.closed))

    def test_credible_crouched_body_vanishing_still_ends_the_tracklet(self):
        # the F3c rule itself is unchanged for a credible contained body (fails closed)
        n_box = int(math.ceil(HIDDEN_S * FPS)) + 10
        for sc in (0.8 * RATIO + 1e-6, 0.6, 0.8):
            with self.subTest(body_score=sc):
                rn = run(_host_with_feet_box(n_box + 20, sc, n_box), fps=FPS)
                breaks = [o.continuity_break for r in rn.results for o in r.observations if o.continuity_break]
                self.assertIn("CONTAINED_BODY_VANISHED", breaks)

    def test_tall_contained_body_vanishing_ends_the_tracklet_whatever_its_score(self):
        # review R2-RC H1: a box as tall as the body (an occluded same-kit player scores low) stays credible
        n_box = int(math.ceil(HIDDEN_S * FPS)) + 10
        for sc in (0.1, 0.3):
            with self.subTest(body_score=sc):
                frames = []
                for k in range(n_box + 20):
                    cx = 300 + 1.5 * k
                    host = box_at(cx, 400, 48, 118)
                    dets, owners = [cand(host, 0.8)], [("TARGET",)]
                    if k < n_box:
                        dets.append(cand((cx - 14, host[3] - 0.9 * 118, cx + 14, host[3] - 1.0), sc))
                        owners.append(("HIDDEN",))
                    frames.append(Frame(dets, owners))
                rn = run(frames, fps=FPS)
                breaks = [o.continuity_break for r in rn.results for o in r.observations if o.continuity_break]
                self.assertIn("CONTAINED_BODY_VANISHED", breaks)

    def test_score_flicker_across_the_ratio_does_not_reset_the_stay(self):
        # review R2-RC H2: a contained body credible on some frames of a continuous stay still counts
        n_box = int(math.ceil(HIDDEN_S * FPS)) + 10
        frames = []
        for k in range(n_box + 20):
            cx = 300 + 1.5 * k
            host = box_at(cx, 400, 48, 118)
            dets, owners = [cand(host, 0.8)], [("TARGET",)]
            if k < n_box:
                sc = 0.8 * RATIO * 0.95 if k % 2 else 0.8 * RATIO * 1.4          # 0.475 x / 0.70 x host
                dets.append(cand((cx - 20, host[3] - 0.6 * 118, cx + 20, host[3] - 1.0), sc))
                owners.append(("HIDDEN",))
            frames.append(Frame(dets, owners))
        rn = run(frames, fps=FPS)
        breaks = [o.continuity_break for r in rn.results for o in r.observations if o.continuity_break]
        self.assertIn("CONTAINED_BODY_VANISHED", breaks)

    def test_score_boundary_is_exactly_the_part_ratio_of_the_host(self):
        n_box = int(math.ceil(HIDDEN_S * FPS)) + 10
        rn = run(_host_with_feet_box(n_box + 20, 0.8 * RATIO, n_box), fps=FPS)      # == ratio x host: a part
        self.assertFalse(any(o.continuity_break for r in rn.results for o in r.observations))


def _anomaly_with_touch(touch_fn, n_tail=None):
    """An isolated size anomaly (abrupt union held), then the box narrows to 1.4 x while touch_fn(b) boxes touch
    it for longer than MAX_S (the G2_D sequence)."""
    n_tail = int(math.ceil(MAX_S * FPS)) + 10 if n_tail is None else n_tail
    boxes = [box_at(300, 400, 40, 100)] * 30
    boxes += [box_at(300 + 16 * (k + 1) / 10, 400, 40 + 32 * (k + 1) / 10, 100) for k in range(10)]
    boxes += [box_at(316, 400, 72, 100)] * 6
    frames = [Frame([cand(b)], [("TARGET",)]) for b in boxes]
    for _ in range(n_tail):
        b = box_at(316, 400, 56, 100)
        dets, owners = [cand(b)], [("TARGET",)]
        for q, (bx, sc, edge) in enumerate(touch_fn(b)):
            dets.append(cand(bx, sc, edge))
            owners.append(("PERSON_%d" % q,))
        frames.append(Frame(dets, owners))
    return frames


def _codes(rn):
    return "".join(code(widest(r)) for r in rn.results)


class DepthApartTouch(unittest.TestCase):
    def _beside(self, b, feet_gap_h, w=36.0, h=100.0, edge=False, score=0.9, inside=0.1):
        # a body-shaped box overlapping the host's right edge by `inside` of its width, its feet feet_gap_h host
        # heights above the host's. Geometries below are chosen so the overlap is a real touch (>= 5 % of the
        # smaller box, merged_corroboration_min_cover) yet below population_contact_cover_min of the box (10 %):
        # the band in which the feet-line test decides
        hh = b[3] - b[1]
        bottom = b[3] - feet_gap_h * hh
        return [((b[2] - inside * w, bottom - h, b[2] + (1.0 - inside) * w, bottom), score, edge)]

    def test_body_behind_on_a_far_feet_line_is_not_a_contact(self):
        # G2_D: feet 0.92-0.98 h above the host's; G2_A: 0.91
        for gap, inside in ((0.5, 0.15), (0.8, 0.3), (0.92, 0.8)):          # overlaps 0.075 / 0.06 / 0.064
            with self.subTest(feet_gap=gap):
                codes = _codes(run(_anomaly_with_touch(lambda b: self._beside(b, gap, inside=inside)), fps=FPS))
                self.assertIn("A", codes)
                self.assertNotIn("M", codes, codes)

    def test_body_in_front_on_a_far_feet_line_is_not_a_contact_either(self):
        codes = _codes(run(_anomaly_with_touch(lambda b: self._beside(b, -0.5, inside=0.15)), fps=FPS))
        self.assertNotIn("M", codes, codes)

    def test_body_on_a_compatible_feet_line_is_a_contact(self):
        # a player stepping forward or back (0.1-0.2 h) is in reach; the boundary is separation_min_gap_h
        for gap in (0.0, 0.15, GAP - 0.01, -(GAP - 0.01)):
            with self.subTest(feet_gap=gap):
                self.assertIn("M", _codes(run(_anomaly_with_touch(lambda b: self._beside(b, gap)), fps=FPS)))

    def test_feet_line_boundary(self):
        self.assertNotIn("M", _codes(run(_anomaly_with_touch(lambda b: self._beside(b, GAP + 0.02)), fps=FPS)))
        self.assertIn("M", _codes(run(_anomaly_with_touch(lambda b: self._beside(b, GAP - 0.02)), fps=FPS)))

    def test_edge_clipped_box_on_a_far_feet_line_still_counts(self):
        # its feet are cut by the image edge: unknown depth, fails closed
        from tests_rb._synth_people import H as IMG_H
        margin = CFG.get("detector_edge_touch_margin_px")
        frames = _anomaly_with_touch(lambda b: self._beside(b, -0.5))             # feet 0.5 h below the host's
        shifted = []
        for fr in frames:
            dets = []
            for d in fr.dets:
                dy = (IMG_H - margin - 2.0) - 450.0                                  # host bottom just inside the edge
                bx = d.bbox
                dets.append(cand((bx[0], bx[1] + dy, bx[2], min(float(IMG_H), bx[3] + dy)), d.score,
                                 bx[3] + dy >= IMG_H - margin))
            shifted.append(Frame(dets, fr.owners))
        self.assertIn("M", _codes(run(shifted, fps=FPS)))

    def test_body_directly_behind_with_hidden_legs_still_counts(self):
        # G2_D twin: a box whose bottom is high because the host hides its legs lies mostly inside the host's box
        # (covered >= population_contact_cover_min): depth unknown, fails closed
        def extra(b):
            hh = b[3] - b[1]
            w = 30.0
            return [((b[2] - 0.8 * w, b[1] - 15.0, b[2] + 0.2 * w, b[3] - 0.45 * hh), 0.9, False)]
        self.assertIn("M", _codes(run(_anomaly_with_touch(extra), fps=FPS)))

    def test_far_body_barely_covered_is_not_a_contact(self):
        # the measured G2_D geometry scaled to the synthetic host: a 62 x 98 pair box with its feet 0.92 h above
        # the host's feet, overlapping 5.3 % of the host box (a touch) and 4.9 % of its own area (not covered)
        def extra(b):
            hh = b[3] - b[1]
            w, h = 62.0, 98.0
            bottom = b[3] - 0.92 * hh
            return [((b[2] - 0.6 * w, bottom - h, b[2] + 0.4 * w, bottom), 0.9, False)]
        codes = _codes(run(_anomaly_with_touch(extra), fps=FPS))
        self.assertIn("A", codes)
        self.assertNotIn("M", codes, codes)

    def test_same_pair_box_on_the_hosts_feet_line_is_a_contact(self):
        def extra(b):
            w, h = 50.0, 98.0                                                      # narrower than the host (widest())
            return [((b[2] - 0.08 * w, b[3] - h, b[2] + 0.92 * w, b[3]), 0.9, False)]   # overlap 0.08 of itself
        self.assertIn("M", _codes(run(_anomaly_with_touch(extra), fps=FPS)))

    def _behind_body_frames(self, feet_gap_h, mode):
        """A tracked body behind the target (feet feet_gap_h host heights above his), its box overlapping the
        anomaly box by ~7 % of its own area (a touch, not covered). mode 'flicker': undetected every 7th frame,
        so its PREDICTED box touches on those frames; mode 'gone': detected 20 frames then never again (closed ->
        MEMORY box, position valid) while the anomaly is active."""
        boxes = [box_at(300, 400, 40, 100)] * 30
        boxes += [box_at(300 + 16 * (k + 1) / 10, 400, 40 + 32 * (k + 1) / 10, 100) for k in range(10)]
        boxes += [box_at(316, 400, 72, 100)] * 6
        boxes += [box_at(316, 400, 56, 100)] * (int(math.ceil(MAX_S * FPS)) + 10)
        frames = []
        for k, b in enumerate(boxes):
            dets, owners = [cand(b)], [("TARGET",)]
            w, h = 40.0, 90.0
            right = b[0] + 5.0 if b[2] - b[0] <= 56.0 else b[0] + 5.0            # 5 px inside the host's left edge
            bottom = b[3] - feet_gap_h * (b[3] - b[1])
            other = (right - w, bottom - h, right, bottom)
            # present only once the anomaly is active and the box has narrowed (k >= 46): a touch on the frame
            # the box first becomes merge-sized would take the unchanged D11 branch (size + any touch)
            present = (mode == "flicker" and k >= 46 and k % 7 != 0) or (mode == "gone" and 46 <= k < 66)
            if present:
                dets.append(cand(other))
                owners.append(("BEHIND",))
            frames.append(Frame(dets, owners))
        return frames

    def test_tracked_body_behind_hidden_this_frame_is_not_a_contact(self):
        # predicted-box branch: on its undetected frames the body's predicted box touches the anomaly box
        codes = _codes(run(self._behind_body_frames(0.5, "flicker"), fps=FPS))
        self.assertIn("A", codes)
        self.assertNotIn("M", codes, codes)

    def test_tracked_body_beside_hidden_this_frame_is_a_contact(self):
        self.assertIn("M", _codes(run(self._behind_body_frames(0.1, "flicker"), fps=FPS)))

    def test_memory_box_of_a_body_behind_is_not_a_contact(self):
        # memory-box branch: the body behind disappeared for good; its last box (feet 0.5 h up) still touches
        codes = _codes(run(self._behind_body_frames(0.5, "gone"), fps=FPS))
        self.assertIn("A", codes)
        self.assertNotIn("M", codes, codes)

    def test_memory_box_of_a_body_beside_is_a_contact(self):
        self.assertIn("M", _codes(run(self._behind_body_frames(0.1, "gone"), fps=FPS)))

    def test_camera_pan_keeps_feet_lines_comparable(self):
        # under a pan both boxes move together: a body on the same feet line is a contact, a far one is not
        for gap, expect, inside in ((0.1, "M", 0.1), (0.8, "A", 0.3)):
            with self.subTest(feet_gap=gap):
                frames = _anomaly_with_touch(lambda b: self._beside(b, gap, inside=inside))
                shifted = []
                for k, fr in enumerate(frames):
                    dx = 1.5 * k
                    dets = [cand((d.bbox[0] + dx, d.bbox[1], d.bbox[2] + dx, d.bbox[3]), d.score) for d in fr.dets]
                    shifted.append(Frame(dets, fr.owners, camera=pan(1.5) if k > 0 else None))
                codes = _codes(run(shifted, fps=FPS))
                if expect == "M":
                    self.assertIn("M", codes, codes)
                else:
                    self.assertNotIn("M", codes, codes)
                    self.assertIn("A", codes)


if __name__ == "__main__":
    unittest.main()

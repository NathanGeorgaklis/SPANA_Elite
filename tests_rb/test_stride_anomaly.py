# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2 running-stride / size-anomaly instruction (Nathan, 2026-11-07 10:34). Real-footage mechanisms
#                 (native 1080p60): G2_D f102 a sprinting body's stride box (width 50 -> 87 px, height constant,
#                 area 1.6 x the 1 s median) corroborated into a merge by a 5 x 5 px corner contact with an unrelated
#                 two-player box; G2_A f77 the same stride on an isolated body held as a SIZE_ANOMALY for > 1 s and
#                 escalated to a merge by duration alone, then LOST; neither clip had a second body.
"""Stride / size-anomaly stage tests (population + target). Second-body corroboration needs a real overlap; an
anomaly with no second-person indication is held (never trusted, never banked, never a merge by duration, never aged
to LOST) and the tracklet continues after it as an unverified episode. Genuine overlaps, same-kit contact and
credible nested bodies still fail closed. Camera zoom/pan and clipped boxes are unchanged."""
import math
import unittest

from player_tracker_rb import population as P
from player_tracker_rb import target as T
from player_tracker_rb.config import TrackerConfig
from tests_rb import _synth_scene as SS
from tests_rb._synth_people import Frame, box_at, cand, run

FPS = 30.0
CFG = TrackerConfig.default()
MAX_S = CFG.get("merged_anomaly_max_s")
COVER = CFG.get("merged_corroboration_min_cover")
KIT_A = (200, 60, 40)
H = 100.0


def code(o):
    return "M" if P.MERGED_SUSPECTED in o.flags else ("A" if P.SIZE_ANOMALY in o.flags else ".")


def stride(n_still, n_cycles, w0=40.0, w1=72.0, h=100.0, cx=300.0, cy=400.0, period_s=0.7, fps=FPS):
    """A sprinting body: width oscillates w0 -> w1 -> w0 each stride with the height constant (the measured 1080p
    pattern: 1.6-1.8 x the narrow-phase width, 0.97 x height)."""
    out = [box_at(cx, cy, w0, h) for _ in range(n_still)]
    n = int(round(period_s * fps))
    for _ in range(n_cycles):
        for k in range(n):
            w = w0 + (w1 - w0) * 0.5 * (1 - math.cos(2 * math.pi * k / n))
            out.append(box_at(cx, cy, w, h))
    return out


class PopulationLevel(unittest.TestCase):
    def test_sprint_widening_is_an_anomaly_never_a_merge_and_the_tracklet_survives(self):
        for fps in (30.0, 60.0):
            with self.subTest(fps=fps):
                boxes = stride(int(fps), 4, fps=fps, period_s=1.4)   # slow stride: wide phase > merged_anomaly_max_s
                rn = run([Frame([cand(b)], [("TARGET",)]) for b in boxes], fps=fps)
                codes = "".join(code(r.observations[0]) for r in rn.results)
                self.assertIn("A", codes)
                self.assertNotIn("M", codes, "a stride on an isolated body is never a merge")
                self.assertEqual(len({r.observations[0].tracklet_id for r in rn.results}), 1)
                for r in rn.results:
                    if P.SIZE_ANOMALY in r.observations[0].flags:
                        self.assertFalse(r.tracklet(r.observations[0].tracklet_id).bank_eligible)

    def test_corner_touch_from_an_unrelated_box_does_not_corroborate(self):
        # the measured failure: a 5 x 5 px corner contact (0.6 % of the smaller box) corroborated a merge
        boxes = stride(30, 1, period_s=1.4)
        frames = []
        for b in boxes:
            other = (b[2] - 5.0, b[1] - 95.0, b[2] + 41.0, b[1] + 5.0)   # 46 x 100 box clipping the top-right corner
            frames.append(Frame([cand(b), cand(other)], [("TARGET",), ("PERSON_B",)]))
        rn = run(frames, fps=FPS)
        codes = "".join(code(r.observations[0]) for r in rn.results)
        self.assertIn("A", codes)
        self.assertNotIn("M", codes, "a corner clip is not a second body")

    def test_genuine_overlap_of_two_bodies_still_corroborates(self):
        boxes = stride(30, 1, period_s=1.4)
        frames = []
        for b in boxes:
            w = b[2] - b[0]
            other = (b[2] - 0.5 * w, b[1], b[2] + 0.5 * w, b[3])             # half of it inside the box
            frames.append(Frame([cand(b), cand(other)], [("TARGET",), ("PERSON_B",)]))
        rn = run(frames, fps=FPS)
        codes = "".join(code(r.observations[0]) for r in rn.results)
        self.assertIn("M", codes, "a real overlap is a second body: merge as before")

    def test_cover_boundary(self):
        """An overlap of exactly merged_corroboration_min_cover of the smaller box corroborates; just below does not."""
        for frac, expect_m in ((COVER * 1.05, True), (COVER, True), (COVER * 0.9, False), (COVER * 0.5, False)):
            with self.subTest(frac=frac):
                boxes = stride(30, 1, period_s=1.4)
                frames = []
                for b in boxes:
                    ow, oh = 40.0, 100.0
                    ix = math.sqrt(frac * ow * oh)                         # square overlap of the right area
                    other = (b[2] - ix, b[3] - ix, b[2] - ix + ow, b[3] - ix + oh)
                    frames.append(Frame([cand(b), cand(other)], [("TARGET",), ("PERSON_B",)]))
                rn = run(frames, fps=FPS)
                codes = "".join(code(r.observations[0]) for r in rn.results)
                self.assertEqual("M" in codes, expect_m, codes)

    def test_credible_nested_body_still_escalates_on_duration(self):
        # review H2 geometry: the union holds the other person's own credible tall box -> merge after MAX_S
        boxes = [box_at(300, 400, 40, 100)] * 30                            # abrupt union held for > MAX_S
        boxes += [box_at(300 + 16 * (k + 1) / 10, 400, 40 + 32 * (k + 1) / 10, 100) for k in range(10)]
        boxes += [box_at(316, 400, 72, 100)] * (int(math.ceil(MAX_S * FPS)) + 10)
        frames = []
        for b in boxes:
            dets, owners = [cand(b)], [("TARGET",)]
            if b[2] - b[0] > 56.0:
                dets.append(cand((b[2] - 32, b[1] + 2, b[2] - 1, b[3] - 1), 0.9))
                owners.append(("PERSON_A",))
            frames.append(Frame(dets, owners))
        rn = run(frames, fps=FPS)
        codes = "".join(code(max(r.observations, key=lambda o: (o.bbox[2] - o.bbox[0]))) for r in rn.results)
        self.assertIn("M", codes)

    def test_credible_crouched_nested_body_on_the_feet_line_still_escalates(self):
        # review R2-STR 1: a credible nested box shorter than 0.75 of the host but standing on its feet line (a
        # crouched person who could stand up) is a second-person indication: merge after MAX_S, as before
        for frac in (0.4, 0.55, 0.7):
            with self.subTest(height_fraction=frac):
                boxes = [box_at(300, 400, 40, 100)] * 30
                boxes += [box_at(300 + 16 * (k + 1) / 10, 400, 40 + 32 * (k + 1) / 10, 100) for k in range(10)]
                boxes += [box_at(316, 400, 72, 100)] * (int(math.ceil(MAX_S * FPS)) + 10)
                frames = []
                for b in boxes:
                    dets, owners = [cand(b)], [("TARGET",)]
                    if b[2] - b[0] > 56.0:
                        hh = b[3] - b[1]
                        dets.append(cand((b[2] - 30, b[3] - frac * hh, b[2] - 1, b[3] - 1), 0.9))
                        owners.append(("PERSON_A",))
                    frames.append(Frame(dets, owners))
                rn = run(frames, fps=FPS)
                codes = "".join(code(max(r.observations, key=lambda o: (o.bbox[2] - o.bbox[0]))) for r in rn.results)
                self.assertIn("M", codes, codes)

    def _union_frames(self, extra, n_extra_tail=10):
        """An abrupt union held for > MAX_S; extra(b) returns additional (box, score) pairs on the wide frames."""
        boxes = [box_at(300, 400, 40, 100)] * 30
        boxes += [box_at(300 + 16 * (k + 1) / 10, 400, 40 + 32 * (k + 1) / 10, 100) for k in range(10)]
        boxes += [box_at(316, 400, 72, 100)] * (int(math.ceil(MAX_S * FPS)) + n_extra_tail)
        frames = []
        for b in boxes:
            dets, owners = [cand(b)], [("TARGET",)]
            if b[2] - b[0] > 56.0:
                for k, (bx, sc) in enumerate(extra(b)):
                    dets.append(cand(bx, sc))
                    owners.append(("PERSON_%d" % k,))
            frames.append(Frame(dets, owners))
        return frames

    def _codes(self, rn):
        return "".join(code(max(r.observations, key=lambda o: (o.bbox[2] - o.bbox[0]))) for r in rn.results)

    def test_own_part_boxes_wholly_inside_on_the_feet_line_are_not_an_indication(self):
        # real 1080p G2_A f101-114: the sprinting target's own leg box (0.28-0.36 x its score, wholly inside,
        # standing on its feet line) and torso box counted as a credible nested body -> contact -> escalation by
        # duration. A part scored <= population_weak_part_score_ratio x the host is a part, inside or poking out.
        ratio = CFG.get("population_weak_part_score_ratio")
        for sc in (0.1, 0.3, ratio * 0.9 - 1e-9):
            with self.subTest(part_score=sc):
                frames = self._union_frames(lambda b: [((b[0] + 8, b[3] - 0.62 * (b[3] - b[1]), b[2] - 8, b[3] - 1), sc),
                                                       ((b[0] + 10, b[1] + 2, b[2] - 10, b[1] + 0.55 * (b[3] - b[1])), sc)])
                rn = run(frames, fps=FPS)
                codes = self._codes(rn)
                self.assertIn("A", codes)
                self.assertNotIn("M", codes, codes)
                self.assertTrue(all(v.anomaly_isolated for r in rn.results[-5:] for v in r.tracklets
                                    if v.tracklet_id == rn.results[-1].observations[0].tracklet_id))

    def test_tall_nested_box_with_room_beside_it_is_a_person_whatever_its_score(self):
        # review R2-STR 2 H1: an occluded same-kit player inside the union scores low; with a body's width of the
        # host left beside it (host 72 - nested 31 = 41 >= 0.6 x 40) it is a possible person -> fails closed
        for sc in (0.1, 0.3, 0.45):
            with self.subTest(nested_score=sc):
                frames = self._union_frames(lambda b: [((b[2] - 32, b[1] + 2, b[2] - 1, b[3] - 1), sc)])
                self.assertIn("M", self._codes(run(frames, fps=FPS)))

    def test_tall_same_body_alternative_box_leaving_no_room_beside_it_is_not_a_person(self):
        # real 1080p G2_A f101: a 51-wide tall alternative box inside the 67-wide stride box of a body whose
        # reference is ~50 wide: 16 px beside it cannot hold the target itself -> the same body, not a second one
        lo = CFG.get("merged_aspect_band")[0]
        for sc in (0.3, 0.9):
            with self.subTest(alt_score=sc):
                frames = self._union_frames(lambda b: [((b[0] + 10, b[1] + 2, b[2] - 10, b[3] - 1), sc)])   # 52 of 72: 20 px beside
                codes = self._codes(run(frames, fps=FPS))
                self.assertIn("A", codes)
                self.assertNotIn("M", codes, codes)

    def test_room_beside_boundary(self):
        lo = CFG.get("merged_aspect_band")[0]
        for room, expect in ((lo * 40.0 + 0.5, "M"), (lo * 40.0 - 0.5, "A")):
            with self.subTest(room=room):
                frames = self._union_frames(lambda b: [((b[0] + room, b[1] + 2, b[2], b[3] - 1), 0.9)])
                codes = self._codes(run(frames, fps=FPS))
                if expect == "M":
                    self.assertIn("M", codes, codes)
                else:
                    self.assertNotIn("M", codes, codes)

    def test_feet_line_box_at_exactly_the_part_score_ratio_is_a_part(self):
        ratio = CFG.get("population_weak_part_score_ratio")
        frames = self._union_frames(lambda b: [((b[2] - 30, b[3] - 0.55 * (b[3] - b[1]), b[2] - 1, b[3] - 1),
                                                0.9 * ratio)])
        codes = self._codes(run(frames, fps=FPS))
        self.assertIn("A", codes)
        self.assertNotIn("M", codes, codes)

    def test_edge_clipped_touching_box_is_never_a_sliver(self):
        # review R2-STR 2 M2: a second person entering at the image edge is clipped, not narrow -> indication
        lo = CFG.get("merged_aspect_band")[0]
        sliver_w = 0.5 * lo * 40.0
        margin = CFG.get("detector_edge_touch_margin_px")
        cx = 300.0

        def extra(b):
            return [((b[2] - 0.64 * sliver_w, b[1] - 1, b[2] + 0.36 * sliver_w, b[3] + 2), 0.5)]
        frames = self._mid_anomaly_frames(extra)
        # shift the whole scene so the sliver's right edge sits on the image edge
        from tests_rb._synth_people import W as IMG_W
        shifted = []
        for fr in frames:
            dets = []
            for d in fr.dets:
                bx = d.bbox
                dx = IMG_W - margin * 0.5 - (316.0 + 28.0 + 0.36 * sliver_w)
                dets.append(cand((bx[0] + dx, bx[1], bx[2] + dx, bx[3]), d.score))
            shifted.append(Frame(dets, fr.owners))
        self.assertIn("M", self._codes(run(shifted, fps=FPS)))

    def test_credible_nested_box_on_the_feet_line_at_the_score_boundary_still_escalates(self):
        ratio = CFG.get("population_weak_part_score_ratio")
        frames = self._union_frames(lambda b: [((b[2] - 30, b[3] - 0.55 * (b[3] - b[1]), b[2] - 1, b[3] - 1),
                                                0.9 * ratio + 1e-6)])
        self.assertIn("M", self._codes(run(frames, fps=FPS)))

    def _mid_anomaly_frames(self, extra, n_tail=None):
        """The measured G2_A sequence: an isolated anomaly starts (stride box 1.8 x), then the box narrows to 1.4 x
        (above merged_anomaly_clear_area_ratio, below merged_area_ratio: not near single, not merge-sized) while
        extra(b) boxes touch it for longer than MAX_S. Pre-fix any touch here escalated (SUSTAINED_SIZE_ANOMALY)."""
        n_tail = int(math.ceil(MAX_S * FPS)) + 10 if n_tail is None else n_tail
        boxes = [box_at(300, 400, 40, 100)] * 30
        boxes += [box_at(300 + 16 * (k + 1) / 10, 400, 40 + 32 * (k + 1) / 10, 100) for k in range(10)]
        boxes += [box_at(316, 400, 72, 100)] * 6
        boxes += [(box_at(316, 400, 56, 100), True)] * n_tail
        frames = []
        for b in boxes:
            touched = isinstance(b, tuple) and len(b) == 2 and b[1] is True
            if touched:
                b = b[0]
            dets, owners = [cand(b)], [("TARGET",)]
            if touched:
                for k, (bx, sc) in enumerate(extra(b)):
                    dets.append(cand(bx, sc))
                    owners.append(("PERSON_%d" % k,))
            frames.append(Frame(dets, owners))
        return frames

    def test_sliver_fragment_touching_the_anomaly_box_is_not_an_indication(self):
        # real 1080p G2_A f121: an 18 x 143 px box (w/h 0.13 against the body's 0.36; score 0.45) on the sprinting
        # target's trailing edge, 64 % inside its 1.5 x box, escalated the anomaly into a merge. No whole person is
        # that narrow: a touching box narrower than merged_aspect_band[0] x the held single-body aspect is a sliver.
        lo = CFG.get("merged_aspect_band")[0]
        sliver_w = 0.5 * lo * 40.0                                          # half the sliver threshold
        frames = self._mid_anomaly_frames(lambda b: [((b[2] - 0.64 * sliver_w, b[1] - 1, b[2] + 0.36 * sliver_w,
                                                        b[3] + 2), 0.5)])
        codes = self._codes(run(frames, fps=FPS))
        self.assertIn("A", codes)
        self.assertNotIn("M", codes, codes)

    def test_body_shaped_touching_box_still_escalates(self):
        # the twin: a touching box at least merged_aspect_band[0] x the single-body aspect is a possible person
        lo = CFG.get("merged_aspect_band")[0]
        w = 1.05 * lo * 40.0                                                # just above the sliver threshold
        frames = self._mid_anomaly_frames(lambda b: [((b[2] - 0.64 * w, b[1] - 1, b[2] + 0.36 * w, b[3] + 2), 0.5)])
        self.assertIn("M", self._codes(run(frames, fps=FPS)))

    def test_sliver_is_judged_against_the_single_body_reference_not_the_stride_box(self):
        # a narrow side-on second person (w/h 0.75 x the target's single-body aspect) touching the 1.4 x box must
        # not look like a sliver against that box's inflated aspect
        lo = CFG.get("merged_aspect_band")[0]
        w = 0.75 * 40.0
        self.assertGreater(0.75, lo)
        self.assertLess(w / 100.0, lo * 56.0 / 100.0)                       # a sliver against the 56-wide box
        frames = self._mid_anomaly_frames(lambda b: [((b[2] - 0.5 * w, b[1], b[2] + 0.5 * w, b[3]), 0.9)])
        self.assertIn("M", self._codes(run(frames, fps=FPS)))

    def test_sliver_tracklet_touching_for_several_frames_is_still_not_an_indication(self):
        # a persisting sliver carries a tracklet whose predicted box touches the anomaly box every frame
        lo = CFG.get("merged_aspect_band")[0]
        sliver_w = 0.5 * lo * 40.0
        frames = self._mid_anomaly_frames(lambda b: [((b[2] - 0.3 * sliver_w, b[1] - 1, b[2] + 0.7 * sliver_w,
                                                        b[3] + 2), 0.6)], n_tail=int(math.ceil(MAX_S * FPS)) + 40)
        codes = self._codes(run(frames, fps=FPS))
        self.assertIn("A", codes)
        self.assertNotIn("M", codes, codes)

    def test_sliver_corroboration_of_a_merge_sized_box_at_first_sight_is_unchanged(self):
        # known limit (documented): ordinary corroboration (size + any real overlap, D11) is not changed by this
        # stage; a sliver touching a box that is merge-sized on the frame it first appears still corroborates
        lo = CFG.get("merged_aspect_band")[0]
        sliver_w = 0.5 * lo * 40.0
        frames = self._union_frames(lambda b: [((b[2] - 0.64 * sliver_w, b[1] - 1, b[2] + 0.36 * sliver_w,
                                                 b[3] + 2), 0.5)])
        self.assertIn("M", self._codes(run(frames, fps=FPS)))

    def test_contact_through_a_hidden_tracklets_predicted_box_is_an_indication(self):
        # a second body walks up, is lost by the detector while its predicted box overlaps the anomaly box: the
        # anomaly is not isolated and still escalates on duration
        boxes = [box_at(300, 400, 40, 100)] * 30
        boxes += [box_at(300 + 16 * (k + 1) / 10, 400, 40 + 32 * (k + 1) / 10, 100) for k in range(10)]
        boxes += [box_at(316, 400, 72, 100)] * (int(math.ceil(MAX_S * FPS)) + 20)
        frames = []
        for k, b in enumerate(boxes):
            dets, owners = [cand(b)], [("TARGET",)]
            if k < 44:                                                       # a second body approaching, then undetected
                dets.append(cand(box_at(420 - 2.0 * k, 400, 40, 100)))
                owners.append(("PERSON_B",))
            frames.append(Frame(dets, owners))
        rn = run(frames, fps=FPS)
        last = rn.results[-1]
        self.assertTrue(any(P.MERGED_SUSPECTED in o.flags for o in last.observations)
                        or not any(v.anomaly_isolated for v in last.tracklets),
                        "contact via a hidden tracklet must not leave the anomaly isolated")

    def test_edge_clipped_target_box_neither_ends_nor_escalates(self):
        boxes = stride(30, 1, period_s=1.4, cx=40.0)                       # at the left image edge (clipped boxes)
        frames = [Frame([cand(b, edge=True)], [("TARGET",)]) for b in boxes]
        rn = run(frames, fps=FPS)
        codes = "".join(code(r.observations[0]) for r in rn.results)
        self.assertNotIn("M", codes)


class TargetLevel(unittest.TestCase):
    PPL = SS.make_people([("TARGET", KIT_A, 1.0)])

    def _run(self, boxes, people=None, cfg=None, extra=None):
        fr = []
        for k, b in enumerate(boxes):
            d = {"TARGET": b}
            if extra:
                d.update(extra(k, b) or {})
            fr.append(SS.SceneFrame(d))
        return SS.run(fr, people or self.PPL, cfg=cfg)

    def test_sprint_is_held_not_lost_and_continues_as_an_unverified_episode(self):
        boxes = stride(30, 2, period_s=1.4) + [box_at(300, 400, 40, 100)] * 40
        r = self._run(boxes)
        self.assertNotIn(T.LOST, r.states())
        self.assertEqual(r.swaps(), [])
        self.assertTrue(any(f.decision.get("size_anomaly_held") for f in r.frames))
        self.assertTrue(any(f.decision.get("size_anomaly_ended") == "ISOLATED_UNVERIFIED_EPISODE" for f in r.frames))
        self.assertFalse(any(f.reidentified_this_frame for f in r.frames))
        self.assertEqual(r.states()[-10:], [T.TRACKING] * 10)
        self.assertTrue(r.bank_is_pure())

    def test_after_an_isolated_anomaly_the_episode_is_unverified_and_untrusted(self):
        boxes = stride(30, 1, period_s=1.4) + [box_at(300, 400, 40, 100)] * 40
        r = self._run(boxes)
        ended = [k for k, f in enumerate(r.frames) if f.decision.get("size_anomaly_ended")]
        self.assertEqual(len(ended), 1)
        k0 = ended[0]
        self.assertEqual(r.frames[k0].target_episode, r.frames[0].target_episode)   # not a reacquisition
        self.assertFalse(r.frames[k0].decision.get("episode_contrast_verified"))
        for f in r.frames[k0:]:
            self.assertFalse(f.identity_trusted)
            self.assertNotEqual((f.decision.get("bank") or {}).get("status"), "ACCEPTED")

    def test_one_hidden_frame_during_a_hold_does_not_trust_anyone(self):
        boxes = stride(30, 1, period_s=1.4) + [box_at(300, 400, 40, 100)] * 40
        fr = [SS.SceneFrame({"TARGET": b}, drop=("TARGET",) if k == 45 else ()) for k, b in enumerate(boxes)]
        r = SS.run(fr, self.PPL)
        self.assertEqual(r.swaps(), [])
        self.assertTrue(r.bank_is_pure())

    def test_f5_flag_keeps_the_previous_path(self):
        cfg = TrackerConfig.with_overrides({"target_reverify_enabled": True})
        boxes = stride(30, 1, period_s=1.4) + [box_at(300, 400, 40, 100)] * 40
        r = self._run(boxes, cfg=cfg)
        self.assertFalse(any(f.decision.get("size_anomaly_held") for f in r.frames))
        self.assertFalse(any(f.decision.get("size_anomaly_ended") for f in r.frames))
        self.assertEqual(r.swaps(), [])

    def test_crouch_jump_turn_pose_changes_never_break_identity(self):
        # crouch (height halves, width constant), jump (box moves up), turn (width 0.6 x): none is merge-sized
        base = box_at(300, 400, 40, 100)
        seq = [base] * 30
        seq += [box_at(300, 425, 44, 50)] * 15 + [base] * 15                 # crouch
        seq += [box_at(300, 370, 40, 100)] * 10 + [base] * 15                # jump
        seq += [box_at(300, 400, 24, 100)] * 15 + [base] * 30                # turn (edge-on)
        r = self._run(seq)
        self.assertNotIn(T.LOST, r.states())
        self.assertEqual(r.swaps(), [])
        self.assertEqual(len({f.target_episode for f in r.frames}), 1, "no episode break on pose changes")

    def test_same_kit_contact_during_a_stride_still_fails_closed(self):
        same = SS.make_people([("TARGET", KIT_A, 1.0), ("OTHER", KIT_A, 0.9)])
        boxes = stride(30, 1, period_s=1.4)

        def other(k, b):
            if k >= 45:                                                      # steps into the wide box
                return {"OTHER": (b[2] - 20, b[1] + 2, b[2] + 20, b[3])}
            return {"OTHER": (b[2] + 60, b[1], b[2] + 100, b[3])}
        r = self._run(boxes + [box_at(300, 400, 40, 100)] * 30, people=same, extra=other)
        wrong = [k for k in r.swaps() if r.frames[k].identity_trusted]
        self.assertEqual(wrong, [])
        self.assertFalse(any(f.decision.get("size_anomaly_ended") == "ISOLATED_UNVERIFIED_EPISODE"
                             for f in r.frames), "a touched anomaly is not isolated")

    def test_camera_zoom_and_pan_do_not_create_anomalies(self):
        fr = []
        base = SS.box_at(300, 400, 40, 100)
        prev = (1.0, 0.0, 0.0)
        for k in range(90):
            s = 1.0 + 0.004 * k                                              # slow zoom in (x1.36 over 3 s) + pan
            tx, ty = SS.W / 2 * (1 - s) + 2.0 * k, SS.H / 2 * (1 - s)
            b = (base[0] * s + tx, base[1] * s + ty, base[2] * s + tx, base[3] * s + ty)
            # the camera motion is given per frame (previous -> current): inc = cur o prev^-1
            ps, ptx, pty = prev
            si = s / ps
            inc = [[si, 0, tx - si * ptx], [0, si, ty - si * pty]]
            prev = (s, tx, ty)
            fr.append(SS.SceneFrame({"TARGET": b}, camera=SS.motion_from_matrix(inc, SS.W, SS.H)))
        r = SS.run(fr, self.PPL)
        self.assertEqual(r.swaps(), [])
        self.assertNotIn(T.LOST, r.states())
        self.assertFalse(any(f.decision.get("size_anomaly_held") for f in r.frames))


if __name__ == "__main__":
    unittest.main()

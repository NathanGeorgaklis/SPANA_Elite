# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2 DEV remediation instruction (Nathan, 2026-10-30 17:41, Part 1): a tighter detector body box may
#                 legitimately sit inside a looser human seed box; replace the IoU-only seed association with a
#                 generic rule (IoU OR strong containment with plausible geometry); still reject plausible wrong
#                 people; fail closed when identity is ambiguous. R2 DEV finding F1 (R2/DEV_RUN_ef6d635).
"""Seed association tests on rendered synthetic scenes (neutral TARGET / PERSON_A; no benchmark knowledge).

Known limit (not tested here because neither rule can see it): on the seed frame a detector box that merges two
touching people is indistinguishable by geometry from one person (no history exists yet), under the IoU rule as
under the containment rule. The protection is the labelling rule that a seed frame has no contact.

The seed box is the human's box; the detector's box is the rendered body box. A human box is often looser than a
detector body box (R2 DEV F1): that must not refuse a clean, unambiguous seed, and must never let a seed box that
could mean two people, an off-centre person or a much smaller (farther) person anchor the target.
"""
import unittest

from player_tracker_rb import target as T
from player_tracker_rb.config import TrackerConfig
from tests_rb._synth_scene import SceneFrame, box_at, make_people, run

CFG = TrackerConfig.default()
KIT_A = (200, 60, 40)
KIT_B = (40, 40, 210)


def loose(box, fw, fh, dx=0.0, dy=0.0):
    """A human box around `box`: width x fw, height x fh, centre shifted by (dx, dy) of the box height."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    w, h = box[2] - box[0], box[3] - box[1]
    cx, cy = cx + dx * h, cy + dy * h
    return (cx - w * fw / 2, cy - h * fh / 2, cx + w * fw / 2, cy + h * fh / 2)


def frames(n, boxes, drop=()):
    return [SceneFrame(dict(boxes), drop=list(drop)) for _ in range(n)]


class TestSeedRule(unittest.TestCase):
    def ppl(self, *others):
        return make_people([("TARGET", KIT_A, 1.0)] + list(others))

    def test_tight_box_uses_the_iou_rule_unchanged(self):
        b = box_at(300, 400, 40, 100)
        r = run(frames(3, {"TARGET": b}), self.ppl())
        self.assertEqual(r.frames[0].decision["seed_rule"], "IOU")
        self.assertEqual(r.states(), [T.TRACKING] * 3)

    def test_looser_human_box_around_one_body_is_accepted_by_containment(self):
        # R2 DEV F1 class: detector box wholly inside the human box, IoU < 0.5 (here 1 / (1.5 * 1.4) = 0.48)
        b = box_at(300, 400, 40, 100)
        r = run(frames(5, {"TARGET": b}), self.ppl(), seed_box=loose(b, 1.5, 1.4))
        d = r.frames[0].decision
        self.assertEqual(d["seed_rule"], "CONTAINMENT")
        self.assertLess(d["seed_iou"], CFG.get("seed_min_iou"))
        self.assertGreaterEqual(d["seed_containment"], CFG.get("seed_contain_min"))
        self.assertEqual(r.states(), [T.TRACKING] * 5)
        self.assertTrue(r.bank_is_pure())

    def test_containment_rule_is_scale_free(self):
        for scale in (0.5, 1.0, 2.0):
            b = box_at(400, 400, 40 * scale, 100 * scale)
            r = run(frames(2, {"TARGET": b}), self.ppl(), seed_box=loose(b, 1.6, 1.5, dx=0.05))
            self.assertEqual(r.frames[0].decision["seed_rule"], "CONTAINMENT", scale)

    def test_one_box_around_two_people_is_refused(self):
        # a box drawn around two separate people: neither is centred in it, so neither may anchor the target
        t, a = box_at(300, 400, 40, 100), box_at(352, 400, 40, 100)
        sb = (t[0] - 6, t[1] - 8, a[2] + 6, a[3] + 8)
        with self.assertRaises(T.SeedError):
            run(frames(1, {"TARGET": t, "PERSON_A": a}), self.ppl(("PERSON_A", KIT_B, 0.0)), seed_box=sb)

    def test_second_body_inside_a_box_centred_on_the_target_is_ambiguous(self):
        # the target is centred, but a second (smaller, farther) person also lies wholly inside the human box
        t = box_at(300, 400, 40, 100)
        sb = loose(t, 2.2, 1.45)
        a = box_at(330, 380, 16, 40)
        with self.assertRaises(T.SeedError) as cm:
            run(frames(1, {"TARGET": t, "PERSON_A": a}), self.ppl(("PERSON_A", KIT_B, 0.0)), seed_box=sb)
        self.assertIn("ambiguous", str(cm.exception))

    def test_neighbour_substantially_inside_a_loose_seed_box_is_ambiguous(self):
        t = box_at(300, 400, 40, 100)
        sb = loose(t, 1.9, 1.4)                              # 76 px wide around a 40 px body: x 262..338
        a = box_at(340, 400, 40, 100)                        # x 320..360: 45 % of the neighbour inside the box
        with self.assertRaises(T.SeedError) as cm:
            run(frames(1, {"TARGET": t, "PERSON_A": a}), self.ppl(("PERSON_A", KIT_B, 0.0)), seed_box=sb)
        self.assertIn("ambiguous", str(cm.exception))

    def test_neighbour_barely_inside_a_box_centred_on_the_target_is_accepted(self):
        t = box_at(300, 400, 40, 100)
        sb = loose(t, 1.9, 1.4)
        a = box_at(352, 400, 40, 100)                        # x 332..372: 15 % of the neighbour inside the box
        r = run(frames(2, {"TARGET": t, "PERSON_A": a}), self.ppl(("PERSON_A", KIT_B, 0.0)), seed_box=sb)
        self.assertEqual(r.frames[0].decision["seed_rule"], "CONTAINMENT")
        self.assertEqual(tuple(r.frames[0].bbox), tuple(t))

    def test_off_centre_body_inside_the_seed_box_is_refused(self):
        # the boxed person is not detected; the only detection inside the box sits well off its centre
        sb = box_at(300, 400, 120, 110)
        a = box_at(300 + 38, 400, 40, 100)                   # centre offset 0.35 seed heights
        with self.assertRaises(T.SeedError):
            run(frames(1, {"PERSON_A": a}), self.ppl(("PERSON_A", KIT_B, 0.0)), seed_name="PERSON_A",
                seed_box=sb)

    def test_much_smaller_body_inside_the_seed_box_is_refused(self):
        # a far, much smaller person fully inside the human box is not the boxed body
        sb = box_at(300, 400, 60, 140)
        a = box_at(300, 400, 18, 45)                         # 0.32 of the seed height
        with self.assertRaises(T.SeedError):
            run(frames(1, {"PERSON_A": a}), self.ppl(("PERSON_A", KIT_B, 0.0)), seed_name="PERSON_A",
                seed_box=sb)

    def test_partly_contained_detection_is_refused(self):
        # the detection pokes well out of the human box: neither IoU nor containment evidence
        b = box_at(300, 400, 40, 100)
        sb = (b[0] + 14, b[1] - 40, b[2] + 30, b[3] + 40)      # 26 % of the body box outside, IoU < 0.5
        with self.assertRaises(T.SeedError):
            run(frames(1, {"TARGET": b}), self.ppl(), seed_box=sb)

    def test_box_drawn_over_several_people_with_one_detected_is_refused(self):
        # review M1: only the middle person is detected; the human box spans three bodies side by side
        t = box_at(300, 400, 40, 100)
        sb = (t[0] - 50, t[1] - 6, t[2] + 50, t[3] + 6)          # ~3.5 bodies wide
        with self.assertRaises(T.SeedError):
            run(frames(1, {"TARGET": t}), self.ppl(), seed_box=sb)

    def test_smaller_farther_person_centred_in_the_box_is_refused_by_the_feet_line(self):
        # review M2: the boxed person is not detected; a farther person, ~0.55 of the box height, stands near its
        # centre (passes centring and height ratio) but their feet line is well above the box bottom
        sb = box_at(300, 400, 60, 140)
        a = box_at(300, 400 - 10, 30, 78)                     # passes every other containment check
        g = T._seed_geometry(sb, a)
        self.assertGreaterEqual(g["w_ratio"], CFG.get("seed_contain_min_width_ratio"))
        self.assertGreaterEqual(g["h_ratio"], CFG.get("seed_contain_min_height_ratio"))
        self.assertGreater(g["bottom_gap_h"], CFG.get("seed_contain_max_bottom_gap"))
        with self.assertRaises(T.SeedError):
            run(frames(1, {"PERSON_A": a}), self.ppl(("PERSON_A", KIT_B, 0.0)), seed_name="PERSON_A", seed_box=sb)

    def test_looser_box_with_a_standing_margin_under_the_feet_is_still_accepted(self):
        b = box_at(300, 400, 40, 100)
        sb = (b[0] - 12, b[1] - 20, b[2] + 12, b[3] + 14)        # IoU 0.36; feet 0.11 seed heights above bottom
        r = run(frames(2, {"TARGET": b}), self.ppl(), seed_box=sb)
        self.assertEqual(r.frames[0].decision["seed_rule"], "CONTAINMENT")


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


class TestSeedChecksInIsolation(unittest.TestCase):
    """Test audit round 4: each containment check refuses on its own (the body passes every other check), so a
    regression that drops one check cannot hide behind another."""
    CHECKS = {
        "contain": lambda g: g["contain"] >= CFG.get("seed_contain_min"),
        "dx": lambda g: abs(g["dx_h"]) <= CFG.get("seed_contain_max_centre_offset"),
        "dy": lambda g: abs(g["dy_h"]) <= CFG.get("seed_contain_max_centre_offset"),
        "height": lambda g: g["h_ratio"] >= CFG.get("seed_contain_min_height_ratio"),
        "width": lambda g: g["w_ratio"] >= CFG.get("seed_contain_min_width_ratio"),
        "feet": lambda g: abs(g["bottom_gap_h"]) <= CFG.get("seed_contain_max_bottom_gap"),
    }

    def _refused_by_only(self, sb, b, check):
        g = T._seed_geometry(sb, b)
        self.assertLess(_iou(sb, b), CFG.get("seed_min_iou"), "containment path")
        self.assertEqual([k for k, ok in self.CHECKS.items() if not ok(g)], [check], g)
        with self.assertRaises(T.SeedError) as cm:
            run(frames(1, {"PERSON_A": b}), make_people([("PERSON_A", KIT_B, 1.0)]), seed_name="PERSON_A",
                seed_box=sb)
        self.assertIn("none is contained", str(cm.exception))

    def test_horizontal_centre_offset_alone_refuses(self):
        self._refused_by_only(box_at(300, 400, 85, 110), box_at(322, 400, 40, 100), "dx")

    def test_vertical_centre_offset_alone_refuses(self):
        # feet on the box's bottom line, but the body fills only its lower half
        self._refused_by_only(box_at(300, 400, 60, 140), box_at(300, 430, 34, 80), "dy")

    def test_height_ratio_alone_refuses(self):
        self._refused_by_only(box_at(300, 400, 80, 200), box_at(300, 425, 40, 90), "height")

    def test_containment_alone_refuses(self):
        # a centred, body-sized detection reaching well outside the human box (above and below)
        self._refused_by_only(box_at(300, 400, 85, 120), box_at(300, 400, 40, 150), "contain")

    def test_two_qualifying_detections_are_ambiguous_without_the_competitor_test(self):
        # both bodies pass every containment check; with the competitor test disabled (threshold above any partly
        # contained body) the multiple-candidate refusal must still fail closed
        sb = box_at(300, 400, 70, 140)
        a, b = box_at(285, 396, 34, 140), box_at(315, 396, 34, 140)
        for d in (a, b):
            g = T._seed_geometry(sb, d)
            self.assertEqual([k for k, ok in self.CHECKS.items() if not ok(g)], [], g)
            self.assertLess(g["contain"], 1.0)
            self.assertLess(_iou(sb, d), CFG.get("seed_min_iou"))
        cfg = TrackerConfig.with_overrides({"seed_contain_competitor_max": 1.0})
        with self.assertRaises(T.SeedError) as cm:
            run(frames(1, {"PERSON_A": a, "OTHER": b}),
                make_people([("PERSON_A", KIT_B, 1.0), ("OTHER", KIT_A, 0.0)]), seed_name="PERSON_A", seed_box=sb,
                cfg=cfg)
        self.assertIn("ambiguous", str(cm.exception))


if __name__ == "__main__":
    unittest.main()

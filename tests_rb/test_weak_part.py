# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2 nested-part vs real-second-player instruction (Nathan, 2026-11-07 03:24). Real-footage
#                 mechanism (native 1080p user test G1_A f166): a 0.12-score leg box 0.71 inside the 0.66-score
#                 target body started a phantom tracklet that closed the target as a MULTI_CLAIM merge.
"""WEAK PART rule tests (synthetic, neutral people). A weak, aligned part box poking out of its host is NESTED
(no tracklet, no second-body corroboration); a credible second person stays a body and merges still fail closed.
Cases: upper-body duplicate, lower-body part poking out, crouched/occluded second person, overlapping same-kit
player beside the host, a nested part appearing/disappearing, phantom-tracklet regression, 720p and 1080p scale.
"""
import unittest

from player_tracker_rb import population as P
from player_tracker_rb.config import TrackerConfig
from tests_rb._synth_people import H, W, Frame, box_at, cand, run

FPS = 30.0
CFG = TrackerConfig.default()


def _frames(n, host_box, extra=None, extra_from=0, extra_to=None, host_score=0.9):
    """A still host body for n frames; `extra` = (box, score) added on frames [extra_from, extra_to)."""
    out = []
    for k in range(n):
        dets, owners = [cand(host_box, host_score)], [("TARGET",)]
        if extra is not None and extra_from <= k < (n if extra_to is None else extra_to):
            dets.append(cand(extra[0], extra[1]))
            owners.append(("PART",))
        out.append(Frame(dets, owners))
    return out


def _last(r):
    return r.results[-1]


def _tracklets_on(res, idx):
    return [v.tracklet_id for v in res.tracklets if v.detection_index == idx]


class WeakPartIsNested(unittest.TestCase):
    """A weak part box poking out of its host (containment below population_nested_containment) is NESTED."""

    def _check_nested(self, host, part, score, scale=1.0):
        host = tuple(v * scale for v in host)
        part = tuple(v * scale for v in part)
        r = run(_frames(12, host, (part, score)), FPS, CFG, width=int(W * scale), height=int(H * scale))
        res = _last(r)
        self.assertIn(P.NESTED, res.observations[1].flags)
        self.assertIn(P.WEAK_PART, res.observations[1].flags)
        self.assertIn(P.CONTAINS_OTHER, res.observations[0].flags)
        self.assertEqual(res.observations[1].nested_in, 0)
        self.assertEqual(_tracklets_on(res, 1), [], "a weak part must start no tracklet")
        self.assertEqual(len([v for v in res.tracklets]), 1)
        self.assertNotIn(P.MERGED_SUSPECTED, res.observations[0].flags)
        return res

    def test_lower_body_part_poking_out_below_the_feet(self):
        host = box_at(300, 360, 32, 116)                     # 242..302 .. 418
        part = (286.0, 344.0, 321.0, 430.0)                  # 35 wide (1.09x), 0.74 inside, bottom 12 px below the feet
        self._check_nested(host, part, 0.12)

    def test_upper_body_duplicate_inside(self):
        host = box_at(300, 360, 32, 116)
        part = box_at(300, 330, 30, 60)                      # fully inside, nested by the plain rule already
        res = run(_frames(6, host, (part, 0.3)), FPS, CFG).results[-1]
        self.assertIn(P.NESTED, res.observations[1].flags)
        self.assertNotIn(P.WEAK_PART, res.observations[1].flags)   # plain nesting, not the weak-part rule
        self.assertEqual(_tracklets_on(res, 1), [])

    def test_host_is_not_second_body_corroborated_by_its_weak_part(self):
        host = box_at(300, 360, 32, 116)
        part = (286.0, 344.0, 321.0, 430.0)
        res = self._check_nested(host, part, 0.12)
        self.assertFalse(res.observations[0].second_body_possible)

    def test_boundaries(self):
        host = box_at(300, 360, 32, 116)                     # x 284..316, y 302..418, score 0.9
        part = (286.0, 344.0, 321.0, 430.0)                  # 0.74 inside, 1.09 x width, offset 0.11
        for score, nested in ((0.45, True), (0.46, False)):   # ratio 0.5 of 0.9 = 0.45
            res = run(_frames(6, host, (part, score)), FPS, CFG).results[-1]
            self.assertEqual(P.WEAK_PART in res.observations[1].flags, nested, f"score {score}")
        # width 1.25 x host (40 px) passes, 1.26 x (40.3) fails: same centre, 0.74 inside is kept by height overlap
        for w, nested in ((40.0, True), (40.4, False)):
            pb = (300 - w / 2, 344.0, 300 + w / 2, 430.0)
            res = run(_frames(6, host, (pb, 0.12)), FPS, CFG).results[-1]
            ok = P.WEAK_PART in res.observations[1].flags or P.NESTED in res.observations[1].flags
            self.assertEqual(ok and P.WEAK_PART in res.observations[1].flags, nested, f"width {w}")

    def test_scale_invariance_720p_and_1080p(self):
        host = box_at(300, 360, 32, 116)
        part = (286.0, 344.0, 321.0, 430.0)
        self._check_nested(host, part, 0.12, scale=1.0)       # 1280x720 synthetic frame
        self._check_nested(host, part, 0.12, scale=1.5)       # 1920x1080

    def test_weak_part_is_still_a_competitor_and_an_observation(self):
        host = box_at(300, 360, 32, 116)
        part = (286.0, 344.0, 321.0, 430.0)
        res = self._check_nested(host, part, 0.12)
        self.assertTrue(res.observations[1].is_body)            # not duplicate-suppressed: still a competitor
        self.assertNotIn(P.DUPLICATE_SUPPRESSED, res.observations[1].flags)


class CredibleSecondPersonStaysABody(unittest.TestCase):
    def _body(self, host, other, score, msg):
        r = run(_frames(12, host, (other, score)), FPS, CFG)
        res = _last(r)
        self.assertNotIn(P.WEAK_PART, res.observations[1].flags, msg)
        # unchanged pre-rule handling: a plain duplicate is suppressed, a plainly nested box is nested, anything
        # else is a tracked body
        if P.NESTED not in res.observations[1].flags and P.DUPLICATE_SUPPRESSED not in res.observations[1].flags:
            self.assertEqual(len(_tracklets_on(res, 1)), 1, msg)
        return res

    def test_crouched_second_person_with_comparable_score(self):
        host = box_at(300, 360, 32, 116)
        crouched = (286.0, 344.0, 321.0, 430.0)             # same geometry as the leg box, credible score
        self._body(host, crouched, 0.6, "a 0.6-score box beside a 0.9 host is a person")

    def test_same_kit_player_beside_the_host(self):
        host = box_at(300, 360, 32, 116)
        beside = box_at(322, 362, 32, 110)                   # centre offset 0.69 x host width, 0.3 containment
        self._body(host, beside, 0.2, "a weak box beside the host is not its part")

    def test_offset_weak_box_mostly_inside_is_not_a_part(self):
        """Containment 0.6 but the centre is off the host (0.44 x width): only the centre test rejects it."""
        host = box_at(300, 360, 32, 116)                     # x 284..316
        off = (304.0, 344.0, 324.0, 430.0)                   # x 304..324: 12/20 = 0.6 inside, offset 14/32 = 0.44
        self._body(host, off, 0.12, "a weak box off the host's centre is not its part")

    def test_taller_or_wider_box_is_not_a_part(self):
        host = box_at(300, 360, 32, 116)                     # 32 x 116 = 3712
        self._body(host, box_at(300, 362, 30, 124), 0.2, "taller than the host")
        # taller but SMALLER in area, 0.6-0.75 inside: only the height test can reject it
        self._body(host, (288.0, 318.0, 310.0, 442.0), 0.12, "taller, narrower, smaller area")
        self._body(host, box_at(300, 380, 44, 70), 0.2, "1.4 x the host width")

    def test_genuine_overlap_still_fails_closed(self):
        """Two credible bodies overlapping: unchanged (both tracked; the population may merge them as before)."""
        host = box_at(300, 360, 32, 116)
        other = box_at(318, 358, 32, 116)
        r = run(_frames(12, host, (other, 0.85)), FPS, CFG)
        res = _last(r)
        self.assertNotIn(P.WEAK_PART, res.observations[1].flags)
        self.assertNotIn(P.NESTED, res.observations[1].flags)
        self.assertEqual(len(res.tracklets), 2)


class PhantomTrackletRegression(unittest.TestCase):
    """The real-footage failure: a weak leg box appears for two frames, becomes nested, and must not have left a
    phantom tracklet that turns the host into a MULTI_CLAIM merge."""

    def test_part_appearing_and_disappearing_leaves_no_phantom(self):
        host = box_at(300, 360, 32, 116)
        part = (286.0, 344.0, 321.0, 430.0)
        frames = _frames(30, host, (part, 0.12), extra_from=10, extra_to=13)
        r = run(frames, FPS, CFG)
        for k, res in enumerate(r.results):
            self.assertEqual(len(res.tracklets), 1, f"frame {k}: a phantom tracklet exists")
            self.assertNotIn(P.MERGED_SUSPECTED, res.observations[0].flags, f"frame {k}")
            self.assertEqual(res.observations[0].tracklet_id, r.results[0].observations[0].tracklet_id)
        self.assertEqual(res.observations[0].tracklet_id, r.results[0].observations[0].tracklet_id)

    def test_persistent_part_vanishing_does_not_end_the_host(self):
        """Review R2-WP finding 1: a leg box present longer than population_hidden_min_s must not feed the
        hidden-body rule 4b and end the host as CONTAINED_BODY_VANISHED when it disappears."""
        host = box_at(300, 360, 32, 116)
        part = (286.0, 344.0, 321.0, 430.0)
        n_part = int(round(3 * CFG.get("population_hidden_min_s") * FPS)) + 2
        r = run(_frames(n_part + 30, host, (part, 0.12), extra_from=5, extra_to=5 + n_part), FPS, CFG)
        tid0 = r.results[0].observations[0].tracklet_id
        for k, res in enumerate(r.results):
            self.assertEqual(len(res.tracklets), 1, f"frame {k}")
            self.assertEqual(res.observations[0].tracklet_id, tid0, f"frame {k}: host tracklet changed")
        self.assertEqual(r.closed(), {})

    def test_same_scenario_was_a_false_merge_without_the_rule(self):
        """Documents the pre-rule behaviour: with the weak-part rule disabled (score ratio 0) the leg box starts a
        tracklet and the host is closed as merged. Guards against the rule being silently bypassed."""
        cfg = TrackerConfig.with_overrides({"population_weak_part_score_ratio": 0.0})
        host = box_at(300, 360, 32, 116)
        part = (286.0, 344.0, 321.0, 430.0)
        r = run(_frames(30, host, (part, 0.12), extra_from=10, extra_to=13), FPS, cfg)
        born = max(len(res.tracklets) for res in r.results)
        self.assertGreaterEqual(born, 2, "without the rule the leg box starts a tracklet")


class TargetBankNeverTakesAHostWithAWeakPartInside(unittest.TestCase):
    """Review R2-WP finding 3: the target keeps confirming on its box while a weak part sits inside it, but that
    box is never banked (WEAK_PART_INSIDE), so a weakly detected crouched person could not contaminate the bank."""

    def test_bank_refused_while_weak_part_inside(self):
        from player_tracker_rb import target as T
        from tests_rb import _synth_scene as SS
        host = SS.box_at(500, 400, 40, 120)                 # x 480..520, y 340..460
        leg = (482.0, 392.0, 524.0, 478.0)                  # 42 wide (1.05x), 0.72 inside, pokes 18 px below
        frames = [SS.SceneFrame({"TARGET": host}, parts=(("TARGET", leg),) if 20 <= k < 50 else (),
                                part_score=0.12) for k in range(80)]
        r = SS.run(frames, SS.make_people([("TARGET", (200, 40, 40), 1.0)]), fps=SS.FPS)
        with_part = [f for f in r.frames if 20 <= f.rel_frame < 50 and f.decision.get("bank")
                     and f.decision["bank"]["refusals"] != ["SAMPLE_NOT_DUE"]]
        self.assertTrue(with_part)
        self.assertTrue(any(f.rb_state == T.TRACKING for f in with_part), "identity still confirms on the host")
        for f in with_part:
            self.assertEqual(f.decision["bank"]["status"], "REFUSED", f.rel_frame)
            self.assertIn("WEAK_PART_INSIDE", f.decision["bank"]["refusals"], f.rel_frame)
        self.assertEqual(r.swaps(), [])
        after = [f for f in r.frames if f.rel_frame >= 55 and f.decision.get("bank")]
        self.assertTrue(any(f.decision["bank"]["status"] != "REFUSED" or
                            "WEAK_PART_INSIDE" not in f.decision["bank"]["refusals"] for f in after),
                        "banking resumes once the part is gone")


if __name__ == "__main__":
    unittest.main()

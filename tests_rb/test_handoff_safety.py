# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2 DEV same-frame identity handoff safety instruction (Nathan, 2026-11-02 20:14): prevent trusted
#                 target identity from carrying onto a different same-kit player who takes the vanished target's box;
#                 distinguish genuine continuous motion, brief physically continuous occlusion, a box replaced by
#                 another player, same-kit crossing / overlap, and camera motion; no bank update without proven
#                 continuity; fail closed. Review of b36c5cb HIGH-1 reproducer.
"""Same-frame identity handoff safety (target.py: continuity_step_*), synthetic scenes at 15 / 30 fps."""
import unittest

from player_tracker_rb import target as T
from player_tracker_rb.config import TrackerConfig
from tests_rb import _synth_scene as SS

KIT_A, KIT_B = (200, 60, 40), (40, 60, 200)
H = 100.0
ON = TrackerConfig.default()
OFF = TrackerConfig.with_overrides({"continuity_step_enabled": False})
FPS_SET = (30.0, 15.0)


def people(*others):
    return SS.make_people([("TARGET", KIT_A, 1.0)] + list(others))


def wrong_trusted(r):
    return [k for k in r.swaps() if r.frames[k].identity_trusted]


def step_breaks(r):
    return [k for k, f in enumerate(r.frames) if f.decision.get("continuity_broken") == "APPEARANCE_STEP"]


def trusted(r):
    return [k for k, f in enumerate(r.frames) if f.identity_trusted]


def substitution(fps, cos):
    n1, n2 = int(1.5 * fps), int(1.5 * fps)
    fr = [SS.SceneFrame({"TARGET": SS.box_at(400, 400, 40, H)}) for _ in range(n1)]
    fr += [SS.SceneFrame({"LOOK": SS.box_at(402, 400, 40, H)}) for _ in range(n2)]
    return fr, people(("LOOK", KIT_A, cos)), n1


def banked_only_with_proven_step(tc, r):
    """Every bank sample accepted on a continuation frame had a passed step on that frame (seed / F2 anchor aside)."""
    for k, f in enumerate(r.frames):
        b = f.decision.get("bank") if isinstance(f.decision, dict) else None
        if k == 0 or not b or b.get("status") != "ACCEPTED" or f.decision.get("seed_anchor"):
            continue
        st = f.decision.get("continuity_step") or {}
        tc.assertTrue(st.get("tested") and not st.get("broken"), (k, st))


class TestHandoffSafety(unittest.TestCase):
    def test_a_different_same_kit_player_replacing_the_box_is_never_trusted(self):
        for fps in (15.0, 30.0, 60.0):
            for cos in (0.7, 0.8, 0.85, 0.9):
                with self.subTest(fps=fps, cos=cos):
                    fr, pp, n1 = substitution(fps, cos)
                    off = SS.run(fr, pp, fps=fps, cfg=OFF)
                    self.assertTrue(wrong_trusted(off), "control: without the rule the substitute is trusted")
                    r = SS.run(fr, pp, fps=fps, cfg=ON)
                    self.assertEqual(wrong_trusted(r), [])
                    self.assertTrue(r.bank_is_pure(), "no bank contamination")
                    # persistent break: the first substitute frame is PENDING (TRACKING, untrusted, unbanked), the
                    # second clean failure suspends
                    self.assertTrue(r.frames[n1].decision.get("continuity_pending"), "first failure is pending")
                    self.assertFalse(r.frames[n1].identity_trusted)
                    self.assertNotEqual((r.frames[n1].decision.get("bank") or {}).get("status"), "ACCEPTED")
                    self.assertEqual(step_breaks(r), [n1 + 1], "the break persisted on the next clean frame")
                    self.assertIn(r.frames[n1 + 1].rb_state, (T.AMBIGUOUS, T.REACQUIRING), "fail closed")
                    self.assertTrue(all(not f.identity_trusted for f in r.frames[n1:]), "never trusted again")
                    banked_only_with_proven_step(self, r)

    @unittest.expectedFailure
    def test_residual_a_near_identical_twin_substitute_is_not_detected(self):
        # appearance cannot tell a twin (cosine 0.97) from the target in one step: recorded residual
        fr, pp, n1 = substitution(30.0, 0.97)
        self.assertEqual(wrong_trusted(SS.run(fr, pp, fps=30.0, cfg=ON)), [])

    def test_genuine_continuous_motion_with_pose_change_keeps_trust_and_banks(self):
        for fps in FPS_SET:
            with self.subTest(fps=fps):
                n = int(5.0 * fps)
                fr = [SS.SceneFrame({"TARGET": SS.box_at(300 + 200 * k / n, 400, 30 + 40 * k / n, H),
                                     "OTHER": SS.box_at(900, 400, 40, H)}) for k in range(n)]
                pp = people(("OTHER", KIT_B, 0.2))
                r = SS.run(fr, pp, fps=fps, cfg=ON, pose_gain=0.6)
                off = SS.run(fr, pp, fps=fps, cfg=OFF, pose_gain=0.6)
                self.assertEqual(step_breaks(r), [])
                self.assertEqual(trusted(r), trusted(off), "no coverage cost on genuine motion")
                self.assertGreater(len(r.tracker.bank.samples), 1, "the bank still learns new views")
                self.assertTrue(r.bank_is_pure())
                banked_only_with_proven_step(self, r)

    def test_brief_physically_continuous_occlusion_is_not_a_handoff(self):
        for fps in FPS_SET:
            with self.subTest(fps=fps):
                n = int(4.0 * fps)
                fr = []
                for k in range(n):
                    s = k / fps
                    ox = 250 + 150 * s                     # OCC passes in front of the target around s = 1
                    fr.append(SS.SceneFrame({"TARGET": SS.box_at(400, 400, 40, H), "OCC": SS.box_at(ox, 405, 40, H)},
                                            order=("TARGET", "OCC")))
                pp = people(("OCC", KIT_B, 0.2))
                r = SS.run(fr, pp, fps=fps, cfg=ON)
                off = SS.run(fr, pp, fps=fps, cfg=OFF)
                self.assertEqual(step_breaks(r), [], "an occluder passing in front is not a handoff")
                self.assertEqual(trusted(r), trusted(off), "no coverage cost (the unverified return is baseline)")
                self.assertEqual(wrong_trusted(r), [])
                self.assertTrue(r.bank_is_pure())
                banked_only_with_proven_step(self, r)

    def test_same_kit_crossing_and_overlap(self):
        for fps in FPS_SET:
            for cos in (0.8, 0.9):
                with self.subTest(fps=fps, cos=cos):
                    n = int(4.0 * fps)
                    fr = []
                    for k in range(n):
                        s = k / fps
                        fr.append(SS.SceneFrame({"TARGET": SS.box_at(300 + 60 * s, 400, 40, H),
                                                 "OTHER": SS.box_at(540 - 60 * s, 402, 40, H)}))
                    pp = people(("OTHER", KIT_A, cos))
                    r = SS.run(fr, pp, fps=fps, cfg=ON)
                    self.assertEqual(wrong_trusted(r), [])
                    self.assertTrue(r.bank_is_pure())
                    banked_only_with_proven_step(self, r)

    def test_camera_pan_is_not_a_handoff(self):
        for fps in FPS_SET:
            with self.subTest(fps=fps):
                n, pan = int(4.0 * fps), 120.0                # px per s
                fr = []
                for k in range(n):
                    s = k / fps
                    cam = None if k == 0 else SS.motion_from_matrix([[1, 0, -pan / fps], [0, 1, 0]], SS.W, SS.H)
                    fr.append(SS.SceneFrame({"TARGET": SS.box_at(700 + 30 * s - pan * s, 400, 40, H),
                                             "OTHER": SS.box_at(1000 - pan * s, 400, 40, H)}, camera=cam))
                pp = people(("OTHER", KIT_B, 0.2))
                r = SS.run(fr, pp, fps=fps, cfg=ON)
                off = SS.run(fr, pp, fps=fps, cfg=OFF)
                self.assertEqual(step_breaks(r), [])
                self.assertEqual(trusted(r), trusted(off))
                self.assertTrue(r.bank_is_pure())

    def test_a_substitution_before_any_step_history_is_caught_by_the_floor(self):
        # too few accepted steps for the relative test: the absolute reacq_appearance_floor test still breaks
        for fps in FPS_SET:
            with self.subTest(fps=fps):
                fr = [SS.SceneFrame({"TARGET": SS.box_at(400, 400, 40, H)}) for _ in range(2)]
                fr += [SS.SceneFrame({"LOOK": SS.box_at(402, 400, 40, H)}) for _ in range(int(1.5 * fps))]
                r = SS.run(fr, people(("LOOK", KIT_A, 0.7)), fps=fps, cfg=ON)
                self.assertEqual(wrong_trusted(r), [])
                self.assertTrue(r.bank_is_pure())
                st = r.frames[2].decision.get("continuity_step") or {}
                self.assertIn("BELOW_REACQ_FLOOR", st.get("why", []), st)

    def test_the_bank_refuses_every_continuation_sample_without_a_tested_step(self):
        # force every step to be untestable (no reference): the bank must take nothing after the seed
        def no_ref(k, res, img, tf, tt, ext):
            tt._step_ref = None
        fps = 30.0
        fr = [SS.SceneFrame({"TARGET": SS.box_at(400 + k, 400, 40, H)}) for k in range(int(3 * fps))]
        r = SS.run(fr, people(), fps=fps, cfg=ON, observer=no_ref)
        self.assertEqual(len(r.tracker.bank.samples), 1, "only the seed sample")
        refused = [f.decision["bank"]["refusals"] for f in r.frames[1:]
                   if (f.decision.get("bank") or {}).get("status") == "REFUSED"
                   and "SAMPLE_NOT_DUE" not in f.decision["bank"]["refusals"]]
        self.assertTrue(refused and all("CONTINUITY_UNPROVEN" in x for x in refused))
        self.assertTrue(all(f.identity_trusted for f in r.frames), "trust itself is not affected")

    def test_review_high1_a_short_detection_gap_before_the_substitute_does_not_evade(self):
        for fps in (15.0, 30.0, 60.0):
            for gap in (1, 2, 3, 4, 6, 8):
                for cos in (0.8, 0.9):
                    with self.subTest(fps=fps, gap=gap, cos=cos):
                        n1 = int(1.5 * fps)
                        fr = [SS.SceneFrame({"TARGET": SS.box_at(400, 400, 40, H)}) for _ in range(n1)]
                        fr += [SS.SceneFrame({"TARGET": SS.box_at(400, 400, 40, H)}, drop=("TARGET",))
                               for _ in range(gap)]
                        fr += [SS.SceneFrame({"LOOK": SS.box_at(402, 400, 40, H)}) for _ in range(int(1.5 * fps))]
                        r = SS.run(fr, people(("LOOK", KIT_A, cos)), fps=fps, cfg=ON)
                        self.assertEqual(wrong_trusted(r), [])
                        self.assertTrue(r.bank_is_pure())

    def test_a_substitution_during_an_unclean_overlap_is_caught_on_the_first_clean_crop(self):
        for fps in FPS_SET:
            with self.subTest(fps=fps):
                n1, n2, n3 = int(1.5 * fps), int(0.3 * fps), int(1.5 * fps)
                fr = [SS.SceneFrame({"TARGET": SS.box_at(400, 400, 40, H), "P": SS.box_at(800, 400, 40, H)})
                      for _ in range(n1)]
                fr += [SS.SceneFrame({"TARGET": SS.box_at(400, 400, 40, H), "P": SS.box_at(430, 400, 40, H)})
                       for _ in range(n2)]                     # the partner overlaps the target's box
                fr += [SS.SceneFrame({"LOOK": SS.box_at(402, 400, 40, H), "P": SS.box_at(430, 400, 40, H)})
                       for _ in range(n2)]                     # swap while still overlapped
                fr += [SS.SceneFrame({"LOOK": SS.box_at(402, 400, 40, H), "P": SS.box_at(800, 400, 40, H)})
                       for _ in range(n3)]
                r = SS.run(fr, people(("LOOK", KIT_A, 0.85), ("P", KIT_B, 0.2)), fps=fps, cfg=ON)
                self.assertEqual([k for k in wrong_trusted(r) if k >= n1 + 2 * n2], [])
                self.assertTrue(r.bank_is_pure())

    def test_review_high2_the_substitute_never_regains_trust_through_a_third_confusable_body(self):
        for fps in FPS_SET:
            for ccos in (0.765, 0.77, 0.775, 0.8):
                with self.subTest(fps=fps, ccos=ccos):
                    n1, n2 = int(2.0 * fps), int(4.0 * fps)
                    fr = [SS.SceneFrame({"TARGET": SS.box_at(400, 400, 40, H), "C": SS.box_at(520, 400, 40, H)})
                          for _ in range(n1)]
                    fr += [SS.SceneFrame({"LOOK": SS.box_at(402, 400, 40, H), "C": SS.box_at(520, 400, 40, H)})
                           for _ in range(n2)]
                    r = SS.run(fr, people(("LOOK", KIT_A, 0.9), ("C", KIT_A, ccos)), fps=fps, cfg=ON)
                    self.assertEqual(wrong_trusted(r), [])
                    self.assertTrue(r.bank_is_pure())

    def test_review_medium1_a_substitution_with_a_short_history_is_caught(self):
        for fps in FPS_SET:
            for nhist in (2, 3, 4, 5):
                with self.subTest(fps=fps, nhist=nhist):
                    fr = [SS.SceneFrame({"TARGET": SS.box_at(400, 400, 40, H)}) for _ in range(nhist)]
                    fr += [SS.SceneFrame({"LOOK": SS.box_at(402, 400, 40, H)}) for _ in range(int(1.5 * fps))]
                    r = SS.run(fr, people(("LOOK", KIT_A, 0.9)), fps=fps, cfg=ON)
                    self.assertEqual(wrong_trusted(r), [])
                    self.assertTrue(r.bank_is_pure())

    def test_review_medium2_low_frame_rates_are_still_tested_and_banked(self):
        for fps in (8.0, 10.0):
            with self.subTest(fps=fps):
                fr, pp, n1 = substitution(fps, 0.85)
                r = SS.run(fr, pp, fps=fps, cfg=ON)
                self.assertEqual(wrong_trusted(r), [])
                self.assertTrue(r.bank_is_pure())
                n = int(5.0 * fps)
                g = [SS.SceneFrame({"TARGET": SS.box_at(300 + 200 * k / n, 400, 30 + 40 * k / n, H)}) for k in range(n)]
                rg = SS.run(g, people(), fps=fps, cfg=ON, pose_gain=0.6)
                self.assertGreater(len(rg.tracker.bank.samples), 1, "consecutive frames still bank-prove")

    def test_a_one_frame_glitch_of_the_true_target_resolves_and_keeps_its_tracklet(self):
        for fps in FPS_SET:
            with self.subTest(fps=fps):
                n1, n3 = int(1.5 * fps), int(1.5 * fps)
                fr = [SS.SceneFrame({"TARGET": SS.box_at(400, 400, 40, H)}) for _ in range(n1)]
                fr.append(SS.SceneFrame({"GLITCH": SS.box_at(401, 400, 40, H)}))   # one odd crop at the target's box
                fr += [SS.SceneFrame({"TARGET": SS.box_at(400, 400, 40, H)}) for _ in range(n3)]
                r = SS.run(fr, people(("GLITCH", KIT_A, 0.7)), fps=fps, cfg=ON)
                # persistent break: one odd clean crop is only pending; the next clean crop restores continuity
                self.assertEqual(step_breaks(r), [], "no suspension from one failed crop")
                self.assertTrue(r.frames[n1].decision.get("continuity_pending"))
                self.assertFalse(r.frames[n1].identity_trusted, "the pending frame is untrusted")
                self.assertEqual(r.frames[n1].identity_uncertainty_reasons, (T.COMPETING_CANDIDATE,))
                self.assertTrue((r.frames[n1 + 1].decision.get("continuity_step") or {}).get("pending_cleared"))
                self.assertTrue(all(r.frames[k].identity_trusted for k in range(n1 + 1, len(fr))),
                                "trusted again from the next clean frame")
                self.assertEqual(r.tracker._handoff_suspect, set())
                self.assertEqual(wrong_trusted(r), [])
                banked_only_with_proven_step(self, r)

    def test_no_known_other_is_recorded_on_a_pending_frame(self):
        # a body first seen on the pending frame must not be ruled out as the target (identity is in doubt there)
        fps = 30.0
        n1 = int(1.5 * fps)
        fr = [SS.SceneFrame({"TARGET": SS.box_at(400, 400, 40, H)}) for _ in range(n1)]
        fr.append(SS.SceneFrame({"GLITCH": SS.box_at(401, 400, 40, H), "NEW": SS.box_at(900, 400, 40, H)}))
        fr += [SS.SceneFrame({"TARGET": SS.box_at(400, 400, 40, H)}) for _ in range(5)]
        seen = {}

        def obs(k, res, img, tf, tt, ext):
            if k == n1:
                seen["new"] = [o.tracklet_id for o in res.observations
                               if o.tracklet_id and round((o.bbox[0] + o.bbox[2]) / 2) == 900]
                seen["known"] = set(tt._known_others)
        r = SS.run(fr, people(("GLITCH", KIT_A, 0.7), ("NEW", KIT_B, 0.2)), fps=fps, cfg=ON, observer=obs)
        self.assertTrue(r.frames[n1].decision.get("continuity_pending"))
        self.assertTrue(seen["new"], "control: the new body has a tracklet on the pending frame")
        self.assertFalse(set(seen["new"]) & seen["known"])

    def test_review_high1_a_pending_frame_never_contrast_verifies_the_episode(self):
        # review of 059387d HIGH-1: an unverified substitute episode (S reacquired after the target left), a same-kit
        # body C nearby; one frame shows the target's crop in S's box (step fails: pending). The pending frame must not
        # verify the episode (it beat C with the TARGET's crop); S must never become trusted
        for fps in (8.0, 15.0, 30.0, 60.0):
            with self.subTest(fps=fps):
                pp = SS.make_people([("TARGET", KIT_A, 1.0), ("S", KIT_A, 0.8), ("C", KIT_A, 0.8)])
                n = int(fps)
                fr = [SS.SceneFrame({"TARGET": SS.box_at(500, 400, 40, H)}) for _ in range(n)]
                fr += [SS.SceneFrame({}) for _ in range(int(1.5 * fps))]
                fr += [SS.SceneFrame({"S": SS.box_at(500, 400, 40, H)}) for _ in range(int(3 * fps))]
                m = int(fps)
                fr += [SS.SceneFrame({"S": SS.box_at(500, 400, 40, H), "C": SS.box_at(1100 - 470 * k / m, 400, 40, H)})
                       for k in range(m + 1)]
                kx = len(fr)
                fr.append(SS.SceneFrame({"TARGET": SS.box_at(500, 400, 40, H), "C": SS.box_at(630, 400, 40, H)}))
                fr += [SS.SceneFrame({"S": SS.box_at(500, 400, 40, H), "C": SS.box_at(630, 400, 40, H)})
                       for _ in range(int(2 * fps))]
                r = SS.run(fr, pp, fps=fps, cfg=ON)
                self.assertNotEqual(r.frames[kx].rb_state, T.TRACKING, "a pending frame is never TRACKING")
                self.assertEqual(wrong_trusted(r), [])
                self.assertTrue(r.bank_is_pure())
                self.assertEqual(r.tracker._known_others, set())

    def test_review_medium2_an_alternating_substitute_suspends(self):
        for fps in FPS_SET:
            for pat in (("SA", "SB"), ("SA", "TARGET"), ("TARGET", "SA")):
                with self.subTest(fps=fps, pat=pat):
                    pp = SS.make_people([("TARGET", KIT_A, 1.0), ("SA", KIT_A, 0.7), ("SB", KIT_A, 0.9)])
                    n1 = int(1.5 * fps)
                    fr = [SS.SceneFrame({"TARGET": SS.box_at(400, 400, 40, H)}) for _ in range(n1)]
                    fr += [SS.SceneFrame({pat[k % 2]: SS.box_at(402, 400, 40, H)}) for k in range(int(1.5 * fps))]
                    r = SS.run(fr, pp, fps=fps, cfg=ON)
                    self.assertEqual(wrong_trusted(r), [])
                    self.assertTrue(r.bank_is_pure())
                    self.assertTrue(step_breaks(r), "a repeated failure within the window suspends")

    def test_an_unclean_pending_stretch_stays_untrusted_and_ages(self):
        # first failure, then the box stays unclean (touching the image edge): pending frames are AMBIGUOUS, untrusted,
        # confirm bookkeeping frozen, and the target ages out by max_ambiguous_s
        fps = 30.0
        n1 = int(1.5 * fps)
        fr = [SS.SceneFrame({"TARGET": SS.box_at(60, 400, 40, H)}) for _ in range(n1)]
        fr.append(SS.SceneFrame({"LOOK": SS.box_at(62, 400, 40, H)}))
        fr += [SS.SceneFrame({"LOOK": SS.box_at(12, 400, 40, H)}) for _ in range(int(2.0 * fps))]   # at the edge
        r = SS.run(fr, people(("LOOK", KIT_A, 0.85)), fps=fps, cfg=ON)
        self.assertEqual(wrong_trusted(r), [])
        self.assertTrue(r.bank_is_pure())
        self.assertTrue(all(f.rb_state != T.TRACKING or not f.identity_trusted for f in r.frames[n1:]))
        t_last = r.frames[n1 - 1].time_s
        tracking_late = [k for k in range(n1, len(fr)) if r.frames[k].rb_state == T.TRACKING
                         and r.frames[k].time_s - t_last > ON.get("max_ambiguous_s") + 2 / fps]
        self.assertTrue(all(not r.frames[k].identity_trusted for k in tracking_late))
        self.assertTrue(any(f.rb_state in (T.LOST, T.REACQUIRING) for f in r.frames[n1:]), "ages out")

    def test_a_pending_state_never_crosses_a_tracklet_change(self):
        fps = 30.0
        n1 = int(1.5 * fps)
        fr = [SS.SceneFrame({"TARGET": SS.box_at(400, 400, 40, H)}) for _ in range(n1)]
        fr.append(SS.SceneFrame({"GLITCH": SS.box_at(401, 400, 40, H)}))
        r = SS.run(fr, people(("GLITCH", KIT_A, 0.7)), fps=fps, cfg=ON)
        tt = r.tracker
        self.assertIsNotNone(tt._step_pending)
        from types import SimpleNamespace
        other = SimpleNamespace(tracklet_id="T999", detection_index=0)
        tt._step_ref = ("T999", 0.0, tt._step_ref[2], 0)
        res = r.pops[-1]
        o = res.observations[0]
        import dataclasses
        res2 = dataclasses.replace(res, observations=(dataclasses.replace(o, tracklet_id="T999"),) + res.observations[1:])
        tt._handoff_step({}, res2, SS.render(fr[-1], people(("GLITCH", KIT_A, 0.7))), other)
        self.assertTrue(tt._step_pending is None or tt._step_pending["tracklet"] == "T999")

    def test_a_two_frame_appearance_dip_of_the_true_target_suspends_then_resolves(self):
        for fps in FPS_SET:
            with self.subTest(fps=fps):
                n1, n3 = int(1.5 * fps), int(1.5 * fps)
                fr = [SS.SceneFrame({"TARGET": SS.box_at(400, 400, 40, H)}) for _ in range(n1)]
                fr += [SS.SceneFrame({"GLITCH": SS.box_at(401, 400, 40, H)}) for _ in range(2)]   # two odd crops
                fr += [SS.SceneFrame({"TARGET": SS.box_at(400, 400, 40, H)}) for _ in range(n3)]
                r = SS.run(fr, people(("GLITCH", KIT_A, 0.7)), fps=fps, cfg=ON)
                self.assertEqual(step_breaks(r), [n1 + 1], "a 2-frame break persists: suspended")
                resolved = [k for k, f in enumerate(r.frames) if (f.decision.get("continuity_step") or {}).get("resolved")]
                self.assertEqual(len(resolved), 1, "then resolved by the pre-break reference")
                self.assertEqual(resolved[0] - (n1 + 1), ON.get("continuity_step_resolve_steps"))
                self.assertTrue(all(not r.frames[k].identity_trusted for k in range(n1, resolved[0])))
                self.assertTrue(all(r.frames[k].identity_trusted for k in range(resolved[0], len(fr))))
                self.assertEqual(wrong_trusted(r), [])
                self.assertTrue(r.bank_is_pure())
                banked_only_with_proven_step(self, r)

    def test_an_unclean_frame_between_two_failures_keeps_the_break_pending(self):
        # failure, then an unclean (edge-touching / partial) crop, then a second clean failure: still suspended
        for fps in FPS_SET:
            with self.subTest(fps=fps):
                n1 = int(1.5 * fps)
                fr = [SS.SceneFrame({"TARGET": SS.box_at(400, 400, 40, H)}) for _ in range(n1)]
                fr.append(SS.SceneFrame({"LOOK": SS.box_at(402, 400, 40, H)}))
                fr.append(SS.SceneFrame({"LOOK": SS.box_at(402, 400, 40, H), "P": SS.box_at(425, 400, 40, H)}))
                fr += [SS.SceneFrame({"LOOK": SS.box_at(402, 400, 40, H)}) for _ in range(int(1.5 * fps))]
                r = SS.run(fr, people(("LOOK", KIT_A, 0.85), ("P", KIT_B, 0.2)), fps=fps, cfg=ON)
                self.assertEqual(wrong_trusted(r), [])
                self.assertTrue(r.bank_is_pure())
                self.assertFalse(r.frames[n1 + 1].identity_trusted, "the unclean frame stays untrusted")
                self.assertTrue(all(not f.identity_trusted for f in r.frames[n1:]), "never trusted again")

    def test_target_partial_at_the_image_edge_is_not_a_break(self):
        for fps in FPS_SET:
            with self.subTest(fps=fps):
                n = int(4.0 * fps)
                fr = [SS.SceneFrame({"TARGET": SS.box_at(max(15.0, 300 - 300 * k / fps) if k < n // 2 else
                                                         15.0 + 300 * (k - n // 2) / fps, 400, 40, H)})
                      for k in range(n)]
                r = SS.run(fr, people(), fps=fps, cfg=ON, pose_gain=0.6)
                off = SS.run(fr, people(), fps=fps, cfg=OFF, pose_gain=0.6)
                self.assertEqual(step_breaks(r), [])
                self.assertEqual(wrong_trusted(r), [])
                self.assertLessEqual(len(trusted(off)) - len(trusted(r)), 2, "at most a pending frame or two")

    def test_a_multi_view_target_replaced_by_a_same_kit_substitute_is_never_trusted(self):
        # review of the (withdrawn) recovery stage 9acf2c5, HIGH-1: the target shows two banked views, then a same-kit
        # player takes its box in one of them; any recovery that compares with the best banked view lets it back in
        import numpy as np
        for fps in (15.0, 30.0, 60.0):
            for cos in (0.83, 0.85, 0.9):
                with self.subTest(fps=fps, cos=cos):
                    n1 = int(2.0 * fps)
                    fr = [SS.SceneFrame({"TARGET": SS.box_at(400, 400, 30 + 30 * (0.5 - 0.5 * np.cos(
                        2 * np.pi * k / fps)), H)}) for k in range(n1)]
                    fr += [SS.SceneFrame({"LOOK": SS.box_at(402, 400, 60, H)}) for _ in range(int(3.2 * fps))]
                    r = SS.run(fr, people(("LOOK", KIT_A, cos)), fps=fps, cfg=ON, pose_gain=0.9)
                    self.assertEqual(wrong_trusted(r), [])
                    self.assertTrue(r.bank_is_pure())

    def test_a_cut_resets_the_step_reference(self):
        from player_tracker_rb import camera as C
        fps = 30.0
        n1 = int(1.5 * fps)
        fr = [SS.SceneFrame({"TARGET": SS.box_at(400, 400, 40, H)}) for _ in range(n1)]
        fr.append(SS.SceneFrame({"TARGET": SS.box_at(400, 400, 40, H)},
                                camera=SS.identity_motion(C.CUT, continuity_broken=True)))
        r = SS.run(fr, people(), fps=fps, cfg=ON)
        self.assertTrue(r.tracker._step_ref is None or r.tracker._step_ref[3] == len(fr) - 1,
                        "no reference from before the cut")
        self.assertEqual(r.tracker._step_hist, [])
        self.assertIsNone(r.tracker._handoff)

    def test_no_bank_sample_without_a_tested_step(self):
        # after a gap longer than continuity_step_max_gap_s the first clean frame is not step-linked: no sample there
        fps = 30.0
        n1, g = int(1.5 * fps), 4                             # 4 frames: longer than continuity_step_max_gap_s
        fr = [SS.SceneFrame({"TARGET": SS.box_at(400, 400, 40, H)}, drop=("TARGET",) if n1 <= k < n1 + g else ())
              for k in range(n1 + g + int(1.5 * fps))]
        r = SS.run(fr, people(), fps=fps, cfg=ON)
        back = n1 + g
        f = r.frames[back]
        self.assertTrue(f.identity_trusted and (f.decision.get("continuity_step") or {}).get("tested"),
                        "continuity is still tested against the pre-gap reference")
        self.assertFalse(f.decision["continuity_step"]["consecutive"])
        self.assertIn("CONTINUITY_UNPROVEN", (f.decision.get("bank") or {}).get("refusals", []))
        banked_only_with_proven_step(self, r)

    def test_a_handoff_suspect_tracklet_is_never_contrast_verified(self):
        # force every contrast check to succeed: only the suspect guard can keep the substitute untrusted after its
        # suspension ages out and it is reacquired on the same tracklet
        real = T.TargetTracker._confusable_contrast
        T.TargetTracker._confusable_contrast = lambda self_, ev: True
        try:
            for fps in FPS_SET:
                with self.subTest(fps=fps):
                    fr, pp, n1 = substitution(fps, 0.9)
                    fr += [SS.SceneFrame({"LOOK": SS.box_at(402, 400, 40, H)}) for _ in range(int(3.0 * fps))]
                    r = SS.run(fr, pp, fps=fps, cfg=ON)
                    self.assertTrue(r.tracker._handoff_suspect or any(f.decision.get("handoff_unresolved")
                                                                      for f in r.frames))
                    self.assertEqual(wrong_trusted(r), [])
                    self.assertTrue(r.bank_is_pure())
        finally:
            T.TargetTracker._confusable_contrast = real


if __name__ == "__main__":
    unittest.main()

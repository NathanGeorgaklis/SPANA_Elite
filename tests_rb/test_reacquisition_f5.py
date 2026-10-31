# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2 DEV F5 instruction (Nathan, 2026-11-01 01:35 and 01:56): improve correct reacquisition without
#                 weakening identity safety; C020-style re-verification after an anomaly from sustained strong
#                 appearance + physical continuity + no plausible local competitor; no threshold lowering; no
#                 global irrelevant competitor; no bank update while untrusted; test C020-style success and same-kit
#                 replacement counterexamples, same-kit competitor, long gap, pose change, post-anomaly return,
#                 wrong look-alike, unresolved ambiguity.
"""F5 tests (synthetic, neutral TARGET / OTHER; no benchmark knowledge).

DEV failure reproduced (R2_MCFC_MIT_C020, c67c738): the target's own tracklet carried an uncorroborated SIZE_ANOMALY
(pose); when it ended the target came back by reacquisition onto the SAME tracklet as an unverified episode, and with no
confusable competitor anywhere near it the episode could never be verified: 262 frames TRACKING but untrusted.
"""
import unittest

from player_tracker_rb import population as P
from player_tracker_rb import target as T
from player_tracker_rb.camera import CUT, identity_motion
from player_tracker_rb.config import TrackerConfig
from tests_rb import _synth_scene as SS

KIT_A = (200, 60, 40)
KIT_B = (40, 40, 210)
SAME_KIT_COS = 0.75            # realistic same-kit embedding similarity (R1-D: same-kit P90 ~0.78)
FPS_SET = (30.0, 15.0, 10.0)
CFG = TrackerConfig.with_overrides({"target_reverify_enabled": True})   # the rule is OFF by default (pending decision)


def run(frames, ppl, fps):
    return SS.run(frames, ppl, fps=fps, cfg=CFG)


def people(*others):
    return SS.make_people([("TARGET", KIT_A, 1.0)] + list(others))


def pose_then(fps, after, *, cx=300.0, cy=400.0, w0=40.0, w1=92.0, h=100.0):
    """The target alone: still 1 s, a pose widening (an uncorroborated SIZE_ANOMALY) over 0.5 s held 0.2 s, back to
    its size, then `after(k)` gives its box k frames after the anomaly ended."""
    n_still, n_grow, n_hold = int(fps), max(2, int(fps / 2)), max(1, int(fps / 5))
    out = [SS.box_at(cx, cy, w0, h) for _ in range(n_still)]
    out += [SS.box_at(cx, cy, w0 + (w1 - w0) * (k + 1) / n_grow, h) for k in range(n_grow)]
    out += [SS.box_at(cx, cy, w1, h) for _ in range(n_hold)]
    return out, len(out)


def walk(fps, speed_h_per_s, cx=300.0, cy=400.0, h=100.0, w=40.0):
    return lambda k: SS.box_at(cx + speed_h_per_s * h * k / fps, cy, w, h)


def trusted_on_other(r, start=0):
    """Trusted frames on a box that is not the target alone (another person, or a union holding the target)."""
    return [k for k in range(start, len(r.frames)) if r.frames[k].identity_trusted
            and r.dets[k][r.frames[k].detection_index].owners != ("TARGET",)]


def trusted_without_target(r, start=0):
    """Trusted frames on a box that does not contain the target at all: a wrong-person identity."""
    return [k for k in range(start, len(r.frames)) if r.frames[k].identity_trusted
            and "TARGET" not in r.dets[k][r.frames[k].detection_index].owners]


def bank_only_when_trusted(tc, r):
    for f in r.frames:
        if (f.decision.get("bank") or {}).get("status") == "ACCEPTED":
            tc.assertTrue(f.identity_trusted, "the bank is updated only on trusted frames")


def anomaly_frames(r):
    return [k for k, (p, ds) in enumerate(zip(r.pops, r.dets))
            if any(d.owners == ("TARGET",) and P.SIZE_ANOMALY in p.observations[i].flags for i, d in enumerate(ds))]


class TestPostAnomalyReverification(unittest.TestCase):
    def test_c020_style_return_is_reverified_by_sustained_evidence(self):
        # the DEV failure class: the target alone, its own box widens with pose and comes back, then it runs on;
        # far-away people of the other kit are clearly different. It must become trusted again on its own box.
        for fps in FPS_SET:
            with self.subTest(fps=fps):
                boxes, n0 = pose_then(fps, None)
                go = walk(fps, 1.0)
                boxes += [go(k) for k in range(int(4 * fps))]
                far = [SS.box_at(1000, 380, 40, 100)] * len(boxes)
                ppl = people(("OTHER", KIT_B, 0.2))
                r = run([SS.SceneFrame({"TARGET": b, "OTHER": o}) for b, o in zip(boxes, far)], ppl, fps)
                an = anomaly_frames(r)
                self.assertTrue(an)
                for k in an:
                    self.assertFalse(r.frames[k].identity_trusted)
                self.assertTrue(any(f.decision.get("continuity_broken") == "SIZE_ANOMALY_ENDED" for f in r.frames))
                rev = [k for k, f in enumerate(r.frames) if (f.decision.get("reverify") or {}).get("status")
                       == "REVERIFIED"]
                self.assertEqual(len(rev), 1, "re-verified exactly once")
                k = rev[0]
                rv = r.frames[k].decision["reverify"]
                self.assertGreaterEqual(rv["elapsed_s"], CFG.get("target_reverify_window_s") - 1e-6)
                self.assertGreaterEqual(rv["travel_h"], CFG.get("target_reverify_min_travel_h"))
                self.assertEqual(rv["spots"], {"spot": "CLEAR", "box0": "CLEAR"})
                self.assertGreaterEqual(r.frames[k].identity_confidence_rb, 0.5, "trusted confidence on that frame")
                self.assertGreaterEqual(r.frames[k].target_episode, 2)
                self.assertTrue(all(f.identity_trusted for f in r.frames[k:]), "trusted from re-verification on")
                # it took at least the reacquisition window plus the re-verification window after the anomaly
                self.assertGreaterEqual((k - an[-1]) / fps, CFG.get("reacq_window_s")
                                        + CFG.get("target_reverify_window_s") - 1.5 / fps)
                self.assertEqual(trusted_on_other(r), [])
                self.assertEqual(r.swaps(), [])
                self.assertTrue(r.bank_is_pure())
                bank_only_when_trusted(self, r)

    def test_still_target_after_an_anomaly_is_not_reverified(self):
        # physical continuity needs travel: a person hidden behind a still box could stay hidden. Recorded cost.
        for fps in FPS_SET:
            with self.subTest(fps=fps):
                boxes, _ = pose_then(fps, None)
                boxes += [SS.box_at(300, 400, 40, 100) for _ in range(int(4 * fps))]
                r = run([SS.SceneFrame({"TARGET": b}) for b in boxes], people(), fps=fps)
                self.assertFalse(any((f.decision.get("reverify") or {}).get("status") == "REVERIFIED"
                                     for f in r.frames))
                self.assertFalse(r.frames[-1].identity_trusted)
                self.assertEqual(r.frames[-1].rb_state, T.TRACKING)
                bank_only_when_trusted(self, r)

    def test_unknown_body_near_the_box_withholds_reverification(self):
        # a body of unknown appearance (no bank yet: it has never been seen clean) walks beside the target during
        # the whole run: it could be a replacement, so the episode stays unverified while it is near
        for fps in FPS_SET:
            with self.subTest(fps=fps):
                boxes, n0 = pose_then(fps, None)
                go = walk(fps, 1.0)
                n = int(3 * fps)
                boxes += [go(k) for k in range(n)]
                # touching the target's box (5 px overlap): never isolated, so it never gets a bank
                side = [None] * n0 + [SS.box_at(300 + 100 * k / fps + 35, 400, 40, 100) for k in range(n)]
                ppl = people(("OTHER", KIT_B, 0.2))
                fr = [SS.SceneFrame({"TARGET": b, **({"OTHER": s} if s else {})}) for b, s in zip(boxes, side)]
                r = run(fr, ppl, fps)
                for k, f in enumerate(r.frames[n0:], start=n0):
                    rv = f.decision.get("reverify") or {}
                    self.assertNotEqual(rv.get("status"), "REVERIFIED", f"frame {k}")
                self.assertEqual(trusted_on_other(r), [])
                bank_only_when_trusted(self, r)

    def test_target_lookalike_anywhere_in_the_frame_withholds_reverification(self):
        # a same-kit look-alike elsewhere in the frame (far from the box) passes the target's appearance floor:
        # the target may be there, so the box is not re-verified (withholding only; never used to verify)
        for fps in FPS_SET:
            with self.subTest(fps=fps):
                boxes, n0 = pose_then(fps, None)
                go = walk(fps, 1.0)
                n = int(3 * fps)
                boxes += [go(k) for k in range(n)]
                ppl = people(("OTHER", KIT_A, 0.97))
                far = SS.box_at(1050, 390, 40, 100)
                r = run([SS.SceneFrame({"TARGET": b, "OTHER": far}) for b in boxes], ppl, fps)
                self.assertFalse(any((f.decision.get("reverify") or {}).get("status") == "REVERIFIED"
                                     for f in r.frames))
                self.assertTrue(any((f.decision.get("reverify") or {}).get("why") == "TARGET_LOOKALIKE_IN_FRAME"
                                    for f in r.frames))
                self.assertEqual(trusted_on_other(r), [])
                bank_only_when_trusted(self, r)


class TestSameKitReplacement(unittest.TestCase):
    """The counterexamples: after a union-like anomaly the box may hold ANOTHER same-kit person."""

    def _replacement(self, fps, target_detected, target_moves=False):
        same = SS.make_people([("TARGET", KIT_A, 1.0), ("OTHER", KIT_A, 0.95)])
        tb = SS.box_at(300, 400, 40, 100)
        n_hide, n_union, n_front, n_walk = int(1.5 * fps), max(2, int(fps / 2)), int(fps), int(3 * fps)
        fr = [SS.SceneFrame({"TARGET": tb, "OTHER": SS.box_at(300, 395, 38, 95)}, order=("OTHER", "TARGET"),
                            drop=("OTHER",)) for _ in range(n_hide)]          # OTHER wholly behind, never detected
        fr += [SS.SceneFrame({"TARGET": tb, "OTHER": SS.box_at(300 + 40 * (k + 1) / n_union, 395, 38, 95)},
                             order=("OTHER", "TARGET"), merge=(("TARGET", "OTHER"),)) for k in range(n_union)]
        fr += [SS.SceneFrame({"TARGET": tb, "OTHER": SS.box_at(302, 401, 41, 101)}, order=("TARGET", "OTHER"),
                             drop=("TARGET",)) for _ in range(n_front)]       # OTHER now in front, hiding TARGET
        for k in range(n_walk):                                              # OTHER walks off with the box
            ob = SS.box_at(302 + 100 * (k + 1) / fps, 401, 41, 101)
            tbk = SS.box_at(300 - (100 * (k + 1) / fps if target_moves else 0.0), 400, 40, 100)
            fr.append(SS.SceneFrame({"TARGET": tbk, "OTHER": ob}, order=("TARGET", "OTHER"),
                                    drop=() if target_detected else ("TARGET",)))
        return run(fr, same, fps), n_hide

    def test_replacement_with_the_uncovered_target_detected_is_never_trusted(self):
        for fps in FPS_SET:
            for moves in (False, True):
                with self.subTest(fps=fps, target_moves=moves):
                    r, n_hide = self._replacement(fps, target_detected=True, target_moves=moves)
                    self.assertEqual(trusted_without_target(r), [])
                    self.assertTrue(r.bank_is_pure())
                    bank_only_when_trusted(self, r)

    def test_replacement_with_the_uncovered_target_missed_by_the_detector_is_never_trusted(self):
        # adversarial: the target, uncovered where it stood, is never detected; the box walks away on OTHER.
        # The spot the box left still shows a target look-alike, so the box is not re-verified.
        for fps in FPS_SET:
            with self.subTest(fps=fps):
                r, n_hide = self._replacement(fps, target_detected=False)
                self.assertEqual(trusted_without_target(r), [])
                self.assertTrue(r.bank_is_pure())
                self.assertTrue(any((f.decision.get("reverify") or {}).get("why")
                                    == "TARGET_LOOKALIKE_AT_VACATED_SPOT" for f in r.frames))


class TestOtherReturnsAreNotReverified(unittest.TestCase):
    def test_wrong_lookalike_after_the_target_left_is_never_trusted(self):
        # the target leaves the frame; a same-kit look-alike appears. Its reacquisition is unverified (no confusable
        # competitor to beat) and the post-anomaly rule does not apply (no anomaly, another tracklet): never trusted
        for fps in FPS_SET:
            with self.subTest(fps=fps):
                ppl = people(("OTHER", KIT_A, SAME_KIT_COS + 0.15))
                fr = [SS.SceneFrame({"TARGET": SS.box_at(1180 + 60 * k / fps, 400, 40, 100)})
                      for k in range(int(2 * fps))]
                fr += [SS.SceneFrame({"OTHER": SS.box_at(600 + 100 * k / fps, 400, 40, 100)})
                       for k in range(int(4 * fps))]
                r = run(fr, ppl, fps)
                self.assertEqual(trusted_on_other(r), [])
                self.assertFalse(any("reverify" in f.decision for f in r.frames))
                self.assertTrue(r.bank_is_pure())

    def test_long_gap_return_on_a_new_tracklet_uses_the_ordinary_rules(self):
        # after a 3 s absence the target returns on a new tracklet: the post-anomaly rule never applies
        for fps in FPS_SET:
            with self.subTest(fps=fps):
                fr = [SS.SceneFrame({"TARGET": SS.box_at(300, 400, 40, 100)}) for _ in range(int(fps))]
                fr += [SS.SceneFrame({}) for _ in range(int(3 * fps))]
                fr += [SS.SceneFrame({"TARGET": SS.box_at(320 + 100 * k / fps, 400, 40, 100)})
                       for k in range(int(3 * fps))]
                r = run(fr, people(), fps)
                self.assertFalse(any("reverify" in f.decision for f in r.frames))
                self.assertEqual(trusted_on_other(r), [])
                bank_only_when_trusted(self, r)

    def test_same_kit_competitor_beside_the_returning_target_keeps_it_unverified_or_contrast_verified(self):
        # unresolved ambiguity: a same-kit person runs beside the target during the whole post-anomaly run. Either
        # the ordinary contrast rule verifies (the target beats a confusable competitor by the margin) or nothing
        # does; the post-anomaly rule never verifies while a confusable body is near
        for fps in FPS_SET:
            with self.subTest(fps=fps):
                boxes, n0 = pose_then(fps, None)
                go = walk(fps, 1.0)
                n = int(3 * fps)
                boxes += [go(k) for k in range(n)]
                side = [SS.box_at(300 + 100 * max(0, k - n0) / fps + 70, 400, 40, 100) for k in range(len(boxes))]
                ppl = people(("OTHER", KIT_A, SAME_KIT_COS))
                r = run([SS.SceneFrame({"TARGET": b, "OTHER": s}) for b, s in zip(boxes, side)], ppl, fps)
                self.assertFalse(any((f.decision.get("reverify") or {}).get("status") == "REVERIFIED"
                                     for f in r.frames))
                self.assertEqual(trusted_on_other(r), [])
                self.assertTrue(r.bank_is_pure())

    def test_camera_cut_during_reverification_resets_it(self):
        for fps in FPS_SET:
            with self.subTest(fps=fps):
                boxes, n0 = pose_then(fps, None)
                go = walk(fps, 1.0)
                n = int(4 * fps)
                boxes += [go(k) for k in range(n)]
                fr = [SS.SceneFrame({"TARGET": b}) for b in boxes]
                cut_at = n0 + int(1.2 * fps)
                fr[cut_at] = SS.SceneFrame({"TARGET": boxes[cut_at]},
                                           camera=identity_motion(CUT, reasons=("HISTOGRAM_CHANGE",),
                                                                  continuity_broken=True))
                r = run(fr, people(), fps)
                after = r.frames[cut_at:]
                self.assertFalse(any((f.decision.get("reverify") or {}).get("status") == "REVERIFIED"
                                     for f in after), "no physical continuity across a cut")
                self.assertEqual(trusted_on_other(r), [])
                bank_only_when_trusted(self, r)



def replacement_base(fps, tx=300.0):
    """An undetected same-kit OTHER wholly behind the target, a union-like anomaly as it slides out, then OTHER in
    front hiding the target: the target's tracklet now holds OTHER (geometry cannot tell)."""
    tb = SS.box_at(tx, 400, 40, 100)
    n_hide, n_union, n_front = int(1.5 * fps), max(2, int(fps / 2)), int(fps)
    fr = [SS.SceneFrame({"TARGET": tb, "OTHER": SS.box_at(tx, 395, 38, 95)}, order=("OTHER", "TARGET"),
                        drop=("OTHER",)) for _ in range(n_hide)]
    fr += [SS.SceneFrame({"TARGET": tb, "OTHER": SS.box_at(tx + 40 * (k + 1) / n_union, 395, 38, 95)},
                         order=("OTHER", "TARGET"), merge=(("TARGET", "OTHER"),)) for k in range(n_union)]
    fr += [SS.SceneFrame({"TARGET": tb, "OTHER": SS.box_at(tx + 2, 401, 41, 101)}, order=("TARGET", "OTHER"),
                         drop=("TARGET",)) for _ in range(n_front)]
    return fr


SAME_PAIR = SS.make_people([("TARGET", KIT_A, 1.0), ("OTHER", KIT_A, 0.95)])


class TestReviewRound1(unittest.TestCase):
    """F5 independent review findings on d68b0cd (reproduced; all were wrong trusted identities with a
    contaminated bank that 304f27e does not produce)."""

    def check_safe(self, r):
        self.assertEqual(trusted_without_target(r), [])
        self.assertTrue(r.bank_is_pure())
        bank_only_when_trusted(self, r)

    def test_replaced_box_after_the_detected_target_walks_out_of_frame(self):
        # (b) the uncovered target is detected (a target look-alike in the frame: the rule is cancelled for good,
        # not merely held back while it is visible) and walks out of the frame; OTHER walks on with the box
        for fps in (30.0, 15.0):
            with self.subTest(fps=fps):
                fr = replacement_base(fps, tx=150.0)
                for k in range(int(4 * fps)):
                    s = (k + 1) / fps
                    boxes = {"OTHER": SS.box_at(152 + 100 * s, 401, 41, 101)}
                    if 150 - 300 * s > -30:
                        boxes["TARGET"] = SS.box_at(150 - 300 * s, 400, 40, 100)
                    fr.append(SS.SceneFrame(boxes, order=("TARGET", "OTHER")))
                self.check_safe(run(fr, SAME_PAIR, fps))

    def test_stale_origin_after_the_target_was_seen_for_seconds(self):
        # (h) the real target stays visible for 2 s (cancelling the rule), then leaves; the origin must not come
        # back to life (and expires after target_reverify_max_s anyway)
        for fps in (30.0, 15.0):
            with self.subTest(fps=fps):
                fr = replacement_base(fps, tx=400.0)
                for k in range(int(6 * fps)):
                    s = (k + 1) / fps
                    tx = 330 - 20 * s if s < 2.0 else 330 - 40 - 300 * (s - 2.0)
                    boxes = {"OTHER": SS.box_at(402 + 60 * s, 401, 41, 101)}
                    if tx > -30:
                        boxes["TARGET"] = SS.box_at(tx, 400, 40, 100)
                    fr.append(SS.SceneFrame(boxes))
                self.check_safe(run(fr, SAME_PAIR, fps))

    def test_uncompensated_camera_pan_is_not_travel(self):
        # (f2) both people still, target hidden behind OTHER; the camera pans but cannot be measured (UNKNOWN):
        # image motion is not travel over the scene, so the run never completes
        from player_tracker_rb.camera import UNKNOWN
        for fps in FPS_SET:
            with self.subTest(fps=fps):
                fr = replacement_base(fps, tx=600.0)
                dx = 150.0 / fps
                for k in range(int(4 * fps)):
                    off = -dx * (k + 1)
                    fr.append(SS.SceneFrame({"TARGET": SS.box_at(600 + off, 400, 40, 100),
                                             "OTHER": SS.box_at(602 + off, 401, 41, 101)},
                                            order=("TARGET", "OTHER"), drop=("TARGET",),
                                            camera=identity_motion(UNKNOWN, reasons=("TOO_FEW_TRACKED",))))
                self.check_safe(run(fr, SAME_PAIR, fps))

    def test_lookalike_seen_after_the_anomaly_cancels_for_good_and_never_locks_the_target_out(self):
        # (i) the real target walks away (visible), is hidden by a different-kit THIRD for 2.5 s, then is visible
        # again: OTHER is never re-verified, and the real target's tracklet is never recorded as a known other
        ppl = SS.make_people([("TARGET", KIT_A, 1.0), ("OTHER", KIT_A, 0.95), ("THIRD", KIT_B, 0.2)])
        fps = 30.0
        fr = replacement_base(fps)
        for k in range(int(8 * fps)):
            s = (k + 1) / fps
            tx = 300 - 100 * min(s, 1.5)
            boxes = {"TARGET": SS.box_at(tx, 400, 40, 100), "OTHER": SS.box_at(302 + 100 * min(s, 4.0), 401, 41, 101)}
            order = ("TARGET", "OTHER")
            if 1.5 <= s < 4.0:
                boxes["THIRD"] = SS.box_at(tx, 402, 46, 104)
                order = ("TARGET", "OTHER", "THIRD")
            elif s >= 4.0:
                boxes["THIRD"] = SS.box_at(tx - 100 * (s - 4.0) - 60, 402, 46, 104)
                order = ("TARGET", "OTHER", "THIRD")
            fr.append(SS.SceneFrame(boxes, order=order))
        r = run(fr, ppl, fps)
        self.check_safe(r)
        tgt = {r.pops[-1].observations[i].tracklet_id for i, d in enumerate(r.dets[-1]) if d.owners == ("TARGET",)}
        self.assertFalse(tgt & r.tracker._known_others)

    def test_the_rule_is_off_by_default(self):
        self.assertFalse(TrackerConfig.default().get("target_reverify_enabled"))
        boxes, _ = pose_then(30.0, None)
        go = walk(30.0, 1.0)
        boxes += [go(k) for k in range(120)]
        r = SS.run([SS.SceneFrame({"TARGET": b}) for b in boxes], people(), fps=30.0)
        self.assertFalse(any("reverify" in f.decision for f in r.frames))
        self.assertFalse(r.frames[-1].identity_trusted, "off: the 304f27e behaviour (unverified episode)")


class TestResidualRiskWhenEnabled(unittest.TestCase):
    """KNOWN LIMIT (recorded, the reason the rule is OFF by default): with the rule ON, a same-kit replacement whose
    true target leaves no visible evidence - uncovered but missed by the detector while it walks away, or hidden
    behind the new box moving in lockstep - is re-verified on the wrong person and contaminates the bank. Geometry
    and generic appearance cannot tell this from a correct return."""

    def _residual(self, fps, lockstep):
        fr = replacement_base(fps)
        for k in range(int(4 * fps)):
            s = (k + 1) / fps
            ob = SS.box_at(302 + 100 * s, 401, 41, 101)
            tb = SS.box_at(300 + 100 * s, 400, 38, 98) if lockstep else SS.box_at(300 - 100 * s, 400, 40, 100)
            fr.append(SS.SceneFrame({"TARGET": tb, "OTHER": ob}, order=("TARGET", "OTHER"), drop=("TARGET",)))
        return run(fr, SAME_PAIR, fps)

    @unittest.expectedFailure
    def test_undetected_target_walking_away_after_a_replacement(self):
        self.assertEqual(trusted_without_target(self._residual(30.0, lockstep=False)), [])

    @unittest.expectedFailure
    def test_target_hidden_in_lockstep_behind_the_replacement(self):
        self.assertEqual(trusted_without_target(self._residual(30.0, lockstep=True)), [])


class TestAuditRound1(unittest.TestCase):
    """F5 test-audit gaps (mutants that survived the first suite)."""

    def check_safe(self, r):
        self.assertEqual(trusted_without_target(r), [])
        self.assertTrue(r.bank_is_pure())
        bank_only_when_trusted(self, r)

    def test_target_seen_at_the_vacated_spot_then_walking_off_undetected_cancels_for_good(self):
        # N57: the uncovered target is visible (but missed by the detector) at the spot for 0.6 s, then walks away
        # undetected; the spot then reads clear, but the look-alike seen there must have cancelled the rule for good
        for fps in FPS_SET:
            with self.subTest(fps=fps):
                fr = replacement_base(fps)
                for k in range(int(4 * fps)):
                    s = (k + 1) / fps
                    tx = 300 if s < 0.6 else 300 - 150 * (s - 0.6)
                    fr.append(SS.SceneFrame({"TARGET": SS.box_at(tx, 400, 40, 100),
                                             "OTHER": SS.box_at(302 + 100 * s, 401, 41, 101)},
                                            order=("TARGET", "OTHER"), drop=("TARGET",)))
                r = run(fr, SAME_PAIR, fps)
                self.check_safe(r)
                self.assertTrue(any((f.decision.get("reverify") or {}).get("why") == "TARGET_LOOKALIKE_AT_VACATED_SPOT"
                                    and f.decision["reverify"].get("status") == "CANCELLED" for f in r.frames))

    def test_lookalike_seen_only_around_the_anomaly_cancels_for_good(self):
        # N58/N63: a target look-alike is visible far away for 0.3 s at the end of the anomaly, then gone before the
        # return: the watch (open from the anomaly's start) is cancelled and the episode is never re-verified
        for fps in FPS_SET:
            with self.subTest(fps=fps):
                boxes, n0 = pose_then(fps, None)
                go = walk(fps, 1.0)
                boxes += [go(k) for k in range(int(4 * fps))]
                ppl = people(("OTHER", KIT_A, 0.97))
                fr = [SS.SceneFrame({"TARGET": b, **({"OTHER": SS.box_at(1050, 390, 40, 100)}
                                                     if n0 <= k < n0 + max(2, int(0.3 * fps)) else {})})
                      for k, b in enumerate(boxes)]
                r = run(fr, ppl, fps)
                self.assertFalse(any((f.decision.get("reverify") or {}).get("status") == "REVERIFIED"
                                     for f in r.frames))
                self.assertTrue(any((f.decision.get("reverify") or {}).get("status") == "CANCELLED"
                                    and f.decision["reverify"].get("phase") in ("ANOMALY", "WAIT") for f in r.frames))
                self.check_safe(r)

    def test_an_unverified_episode_is_never_laundered_by_an_anomaly(self):
        # N32: the target returns after leaving the frame (an UNVERIFIED episode: no confusable competitor), then
        # its box has a pose anomaly and returns: the watch must never open (it needs a verified identity)
        for fps in FPS_SET:
            with self.subTest(fps=fps):
                fr = [SS.SceneFrame({"TARGET": SS.box_at(300, 400, 40, 100)}) for _ in range(int(fps))]
                fr += [SS.SceneFrame({}) for _ in range(int(2 * fps))]
                fr += [SS.SceneFrame({"TARGET": SS.box_at(600, 400, 40, 100)}) for _ in range(int(2 * fps))]
                boxes, _ = pose_then(fps, None, cx=600.0)
                go = walk(fps, 1.0, cx=600.0)
                fr += [SS.SceneFrame({"TARGET": b}) for b in boxes + [go(k) for k in range(int(4 * fps))]]
                r = run(fr, people(), fps)
                self.assertTrue(any(not f.identity_trusted and f.rb_state == T.TRACKING for f in r.frames),
                                "the return was an unverified episode")
                self.assertFalse(any("reverify" in f.decision for f in r.frames))
                self.check_safe(r)

    def test_a_return_that_waits_past_the_expiry_is_not_reverified(self):
        # N52/N53: back on its tracklet the target stands still for 3.5 s (> target_reverify_max_s after the
        # anomaly), then walks cleanly: the watch has expired, so the episode stays unverified
        for fps in FPS_SET:
            with self.subTest(fps=fps):
                boxes, _ = pose_then(fps, None)
                boxes += [SS.box_at(300, 400, 40, 100) for _ in range(int(3.5 * fps))]
                go = walk(fps, 1.0)
                boxes += [go(k) for k in range(int(3 * fps))]
                r = run([SS.SceneFrame({"TARGET": b}) for b in boxes], people(), fps)
                self.assertFalse(any((f.decision.get("reverify") or {}).get("status") == "REVERIFIED"
                                     for f in r.frames))
                self.assertTrue(any((f.decision.get("reverify") or {}).get("why") == "EXPIRED" for f in r.frames))
                self.assertFalse(r.frames[-1].identity_trusted)

    def test_one_frame_detector_flicker_restarts_the_run(self):
        # N18/N19: one missed detection 0.8 s into the run restarts it: re-verification comes at least one full
        # window after the flicker
        for fps in (30.0, 15.0):
            with self.subTest(fps=fps):
                boxes, n0 = pose_then(fps, None)
                go = walk(fps, 1.0)
                boxes += [go(k) for k in range(int(4 * fps))]
                r0 = run([SS.SceneFrame({"TARGET": b}) for b in boxes], people(), fps)
                rev0 = next(k for k, f in enumerate(r0.frames)
                            if (f.decision.get("reverify") or {}).get("status") == "REVERIFIED")
                run_start = next(k for k, f in enumerate(r0.frames)
                                 if (f.decision.get("reverify") or {}).get("status") == "RUNNING")
                miss = run_start + int(0.8 * fps)
                fr = [SS.SceneFrame({"TARGET": b}, drop=("TARGET",) if k == miss else ())
                      for k, b in enumerate(boxes)]
                r = run(fr, people(), fps)
                rev = [k for k, f in enumerate(r.frames)
                       if (f.decision.get("reverify") or {}).get("status") == "REVERIFIED"]
                self.assertTrue(rev)
                self.assertGreater(rev[0], rev0)
                self.assertGreaterEqual((rev[0] - miss) / fps, CFG.get("target_reverify_window_s") - 1.5 / fps)


if __name__ == "__main__":
    unittest.main()

# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2 DEV target jersey-number evidence instruction (Nathan, 2026-11-04 08:13): temporal consensus per
#                 tracklet; UNKNOWN when unreadable; duplicates possible; support or veto only, never proof; one OCR
#                 reading never decides; only clean, geometrically valid crops; target-centric reading; tracker unchanged.
"""Jersey-number evidence (player_tracker_rb.jersey) with a scripted reader on synthetic scenes."""
import unittest

from player_tracker_rb import camera as C
from player_tracker_rb import jersey as J
from player_tracker_rb import target as T
from player_tracker_rb.config import TrackerConfig
from tests_rb import _synth_scene as SS

KIT_A, KIT_B = (200, 60, 40), (40, 60, 200)
H = 120.0                                 # boxes above jersey_min_box_h_px
CFG = TrackerConfig.default()


def people(*others):
    return SS.make_people([("TARGET", KIT_A, 1.0)] + list(others))


class OwnerReader:
    """A scripted OCR: returns `numbers[owner]` for the box's true owner (looked up by the box centre x), with a
    confidence; owners missing from `numbers` are unreadable. Records every box it is asked to read."""

    def __init__(self, numbers, conf=0.9):
        self.numbers, self.conf, self.calls, self.where = numbers, conf, [], {}

    def __call__(self, image, box):
        self.calls.append(tuple(box))
        owner = self.where.get(round((box[0] + box[2]) / 2))
        n = self.numbers.get(owner)
        return [] if n is None else [(str(n), self.conf)]


def run(frames, pp, fps, reader, expected, cfg=CFG, select=None):
    """Run the unchanged tracker and the jersey layer beside it; returns (result, per-frame layer output)."""
    out = []
    ev = J.JerseyEvidence(cfg, expected)

    def obs(k, res, img, tf, tt, ext):
        reader.where = {round((b[0] + b[2]) / 2): name for name, b in frames[k].boxes.items()}
        tids = select(res, tf) if select else [o.tracklet_id for o in res.observations if o.tracklet_id]
        out.append(ev.update(res, img, reader, tids))
    r = SS.run(frames, pp, fps=fps, cfg=cfg, observer=obs)
    return r, out, ev


class TestParse(unittest.TestCase):
    def test_only_clean_one_or_two_digit_numbers_count(self):
        m = CFG.get("jersey_min_read_conf")
        self.assertEqual(J.parse_reading([("7", 0.9)], m), (7, 0.9))
        self.assertEqual(J.parse_reading([(" 1 0 ", 0.8)], m), (10, 0.8))
        for bad in ([("07", 0.9)], [("123", 0.9)], [("7a", 0.9)], [("", 0.9)], [("7", m / 2)], [("0", 0.9)]):
            with self.subTest(bad=bad):
                self.assertIsNone(J.parse_reading(bad, m))
        self.assertIsNone(J.parse_reading([("7", 0.9), ("1", 0.9)], m), "disagreeing results on one crop")
        self.assertEqual(J.parse_reading([("7", 0.9), ("7", 0.6)], m), (7, 0.9))

    def test_expected_number_is_validated(self):
        for bad in (0, 100, -3, "7", 7.0, True):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    J.JerseyEvidence(CFG, bad)
        J.JerseyEvidence(CFG, 99)
        J.JerseyEvidence(CFG, None)


class TestConsensus(unittest.TestCase):
    def scene(self, n, **extra):
        return [SS.SceneFrame({"TARGET": SS.box_at(400, 400, 48, H), **extra}) for _ in range(n)]

    def target_rows(self, r, out):
        return [o.get(r.frames[k].observation_tracklet_id) for k, o in enumerate(out)]

    def test_one_reading_never_decides_and_confirmation_needs_reads_and_span(self):
        for fps in (30.0, 15.0):
            with self.subTest(fps=fps):
                fr = self.scene(int(2 * fps))
                r, out, ev = run(fr, people(), fps, OwnerReader({"TARGET": 7}), 7)
                rows = self.target_rows(r, out)
                need = CFG.get("identity_number_min_reads")
                span = CFG.get("jersey_confirm_span_s")
                first_conf = next(k for k, x in enumerate(rows) if x and x["status"] == J.CONFIRMED)
                self.assertGreaterEqual(rows[first_conf]["reads"], need)
                self.assertGreaterEqual(rows[first_conf]["span_s"] + 1e-9, span)
                self.assertTrue(all(x["status"] == J.UNKNOWN for x in rows[:need - 1] if x), "one reading never decides")
                self.assertEqual(rows[-1]["evidence"], J.SUPPORT)

    def test_a_different_confirmed_number_vetoes_and_a_conflict_blocks(self):
        fr = self.scene(60)
        r, out, ev = run(fr, people(), 30.0, OwnerReader({"TARGET": 9}), 7)
        self.assertEqual(self.target_rows(r, out)[-1]["evidence"], J.VETO)
        rd = OwnerReader({"TARGET": 7})
        r, out, ev = run(fr, people(), 30.0, rd, 7)
        tid = r.frames[-1].observation_tracklet_id
        ev._reads[tid].append((r.frames[-1].time_s, 1, 0.9))       # one conflicting reading
        self.assertEqual(ev.consensus(tid)["status"], J.CONFLICT)
        self.assertEqual(ev.consensus(tid)["evidence"], J.UNKNOWN)

    def test_unreadable_is_unknown_and_no_expected_number_is_always_unknown(self):
        fr = self.scene(60)
        r, out, ev = run(fr, people(), 30.0, OwnerReader({}), 7)
        self.assertTrue(all(x["status"] == J.UNKNOWN for x in self.target_rows(r, out) if x))
        r, out, ev = run(fr, people(), 30.0, OwnerReader({"TARGET": 7}), None)
        self.assertTrue(all(x["evidence"] == J.UNKNOWN for x in self.target_rows(r, out) if x))
        self.assertEqual(self.target_rows(r, out)[-1]["status"], J.CONFIRMED)

    def test_a_duplicate_number_only_ever_supports(self):
        # a same-kit player wearing the same number confirms the same number: SUPPORT is all it can give (support,
        # never proof); the tracker itself is unchanged
        fr = [SS.SceneFrame({"TARGET": SS.box_at(400, 400, 48, H), "TWIN": SS.box_at(800, 400, 48, H)})
              for _ in range(60)]
        r, out, ev = run(fr, people(("TWIN", KIT_A, 0.8)), 30.0, OwnerReader({"TARGET": 7, "TWIN": 7}), 7)
        twin_tid = next(o.tracklet_id for o in r.pops[-1].observations if round((o.bbox[0] + o.bbox[2]) / 2) == 800)
        self.assertEqual(out[-1][twin_tid]["evidence"], J.SUPPORT)
        plain = SS.run(fr, people(("TWIN", KIT_A, 0.8)), fps=30.0)
        self.assertEqual([f.as_record() for f in r.frames], [f.as_record() for f in plain.frames],
                         "the jersey layer never changes the tracker")

    def test_unclean_crops_are_never_read(self):
        fps = 30.0
        fr = [SS.SceneFrame({"TARGET": SS.box_at(30, 400, 48, H)}) for _ in range(20)]               # at the edge
        fr += [SS.SceneFrame({"TARGET": SS.box_at(400, 400, 48, H), "P": SS.box_at(420, 400, 48, H)})
               for _ in range(20)]                                                                       # overlapped
        fr += [SS.SceneFrame({"TARGET": SS.box_at(600, 400, 48, 60.0)}) for _ in range(20)]           # too small
        rd = OwnerReader({"TARGET": 7, "P": 4})
        r, out, ev = run(fr, people(("P", KIT_B, 0.2)), fps, rd, 7)
        self.assertEqual(len(rd.calls), ev.counts["crops_clean"])
        self.assertLess(ev.counts["crops_clean"], ev.counts["crops_offered"])
        for k, f in enumerate(r.pops):
            for o in f.observations:
                row = out[k].get(o.tracklet_id) if o.tracklet_id else None
                if row is not None and not J.clean_crop(o, CFG):
                    self.assertIsNone(row["read"], (k, o.flags))

    def test_only_the_selected_tracklets_are_read(self):
        fr = [SS.SceneFrame({"TARGET": SS.box_at(400, 400, 48, H), "FAR": SS.box_at(1000, 400, 48, H)})
              for _ in range(30)]
        rd = OwnerReader({"TARGET": 7, "FAR": 5})

        def target_only(res, tf):
            return [tf.observation_tracklet_id] if tf.observation_tracklet_id else []
        r, out, ev = run(fr, people(("FAR", KIT_B, 0.2)), 30.0, rd, 7, select=target_only)
        self.assertTrue(rd.calls)
        self.assertTrue(all(round((b[0] + b[2]) / 2) == 400 for b in rd.calls), "target-centric reading only")

    def test_a_cut_a_break_flag_and_a_tracklet_end_clear_the_readings(self):
        import dataclasses
        from player_tracker_rb import population as P
        fps = 30.0
        fr = [SS.SceneFrame({"TARGET": SS.box_at(400, 400, 48, H)}) for _ in range(30)]
        fr.append(SS.SceneFrame({"TARGET": SS.box_at(400, 400, 48, H)},
                                camera=SS.identity_motion(C.CUT, continuity_broken=True)))
        r, out, ev = run(fr, people(), fps, OwnerReader({"TARGET": 7}), 7)
        self.assertTrue(all(len(v) <= 1 for v in ev._reads.values()), "a cut clears every tracklet's readings")
        r, out, ev = run(fr[:30], people(), fps, OwnerReader({"TARGET": 7}), 7)
        res = r.pops[-1]
        o = res.observations[0]
        tid = o.tracklet_id
        self.assertTrue(ev._reads.get(tid))
        bad = dataclasses.replace(res, observations=(dataclasses.replace(o, flags=o.flags + (P.CONTESTED,)),))
        ev.update(bad, SS.render(fr[29], people()), OwnerReader({"TARGET": 7}), [tid])
        self.assertNotIn(tid, ev._reads, "a break flag clears the readings")
        gone = dataclasses.replace(res, observations=(), tracklets=())
        ev._reads[tid] = [(0.0, 7, 0.9)]
        ev.update(gone, SS.render(fr[29], people()), OwnerReader({}), [])
        self.assertNotIn(tid, ev._reads, "a tracklet that ended is forgotten")

    def test_memory_is_bounded(self):
        fr = self.scene(int(CFG.get("identity_history_max_samples") * 3))
        r, out, ev = run(fr, people(), 30.0, OwnerReader({"TARGET": 7}), 7)
        self.assertLessEqual(ev.state_sizes()["jersey_reads"], CFG.get("identity_history_max_samples"))

    def test_a_same_kit_substitute_with_another_number_never_gains_support(self):
        # the target (7) vanishes and a same-kit player (9) takes its box: the carried tracklet shows readings of two
        # numbers -> CONFLICT (never SUPPORT); the substitute's own evidence can only be UNKNOWN or VETO
        for fps in (15.0, 30.0, 60.0):
            with self.subTest(fps=fps):
                n1, n2 = int(1.5 * fps), int(1.5 * fps)
                fr = [SS.SceneFrame({"TARGET": SS.box_at(400, 400, 48, H)}) for _ in range(n1)]
                fr += [SS.SceneFrame({"LOOK": SS.box_at(402, 400, 48, H)}) for _ in range(n2)]
                r, out, ev = run(fr, people(("LOOK", KIT_A, 0.85)), fps, OwnerReader({"TARGET": 7, "LOOK": 9}), 7)
                after = [row for k, o in enumerate(out) if k >= n1 for row in o.values()]
                self.assertTrue(after)
                self.assertTrue(all(x["evidence"] != J.SUPPORT for x in after), "never SUPPORT after the swap")
                self.assertTrue(after[-1]["status"] == J.CONFLICT or after[-1]["evidence"] == J.VETO,
                                "mixed readings -> CONFLICT; once the old readings roll out of the bounded history the "
                                "substitute's own number is confirmed -> VETO")
                self.assertEqual([k for k in r.swaps() if r.frames[k].identity_trusted], [],
                                 "the handoff rule (unchanged) keeps the substitute untrusted")


if __name__ == "__main__":
    unittest.main()

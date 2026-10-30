# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R1-E instruction (Nathan, 2026-10-24); R1/R1_DESIGN_DECISIONS.md D06, D08, D10, D11, D13, D16
"""R1-E population tests (synthetic, neutral TARGET / PERSON_A / PERSON_B; no benchmark knowledge).

Covered: determinism; births and provisional competitors; continuity; short gaps with and without contact; long
gaps; leaving the frame; crossings (merged box, rear person hidden, partial overlap) with a ground-truth no-swap
check; same-kit people (the population never uses appearance, so kit cannot matter); merged boxes (multi-claim,
area, aspect, persistent); partial detections; duplicates vs genuine overlap; pan; zoom; hard cut; unknown camera;
provisional competitor blocking; observed vs inferred invariants; one-detection-one-tracklet exclusion; appearance
bank contamination; fps and resolution invariance; input validation.
"""
import json
import math
import unittest

import numpy as np

from player_tracker_rb import population as P
from player_tracker_rb.appearance import AppearanceExtractor, EmbeddingBackend, l2_normalise
from player_tracker_rb.camera import CUT, UNKNOWN, identity_motion
from player_tracker_rb.config import TrackerConfig
from player_tracker_rb.detector import PersonCandidate
from player_tracker_rb.motion import DETECTION, HELD, PREDICTED
from tests_rb._synth_people import (H, W, Frame, box_at, cand, combine, crossing, pan, run, still, walker,
                                    zoom)
from tests_rb._synth_people import iou as P_iou

FPS = 30.0


def invariants(tc: unittest.TestCase, r) -> None:
    """Checks that must hold on every frame of every scenario."""
    seen_dets = {}
    for v in r.tracklets:
        if v.observed:
            tc.assertEqual(v.observation_source, DETECTION)
            tc.assertIsNotNone(v.observed_box)
            tc.assertIsNotNone(v.detection_index)
            tc.assertIn(v.status, (P.NEW, P.ACTIVE))
            tc.assertEqual(tuple(v.box), tuple(r.observations[v.detection_index].bbox))
            tc.assertEqual(r.observations[v.detection_index].tracklet_id, v.tracklet_id)
            tc.assertNotIn(v.detection_index, seen_dets, "one detection, two tracklets")
            seen_dets[v.detection_index] = v.tracklet_id
            tc.assertNotIn(P.MERGED_SUSPECTED, r.observations[v.detection_index].flags)
            tc.assertTrue(r.observations[v.detection_index].is_body)
        else:
            tc.assertIn(v.observation_source, (PREDICTED, HELD))
            tc.assertIsNone(v.observed_box)
            tc.assertIsNone(v.detection_index)
            tc.assertEqual(v.status, P.HIDDEN)
            tc.assertFalse(v.bank_eligible)
        tc.assertTrue(v.is_competitor)
        if v.bank_eligible:
            tc.assertTrue(v.observed)
    for o in r.observations:
        if o.tracklet_id is not None:
            tc.assertEqual(seen_dets.get(o.index), o.tracklet_id)


class Base(unittest.TestCase):
    def check_all(self, rn):
        for r in rn.results:
            invariants(self, r)

    def assert_no_swap(self, rn):
        for tid, owners in rn.tracklet_owners().items():
            self.assertLessEqual(len(owners), 1, f"tracklet {tid} was observed on {sorted(owners)}")


class TestBasics(Base):
    def test_births_are_provisional_competitors_from_first_frame(self):
        rn = run(combine(walker(3, FPS, name="TARGET"), walker(3, FPS, x0=800, name="PERSON_A")))
        r0 = rn.results[0]
        self.assertEqual(len(r0.tracklets), 2)
        for v in r0.tracklets:
            self.assertEqual(v.status, P.NEW)
            self.assertTrue(v.provisional)
            self.assertTrue(v.is_competitor)
        self.check_all(rn)

    def test_continuity_and_ids(self):
        rn = run(walker(20, FPS))
        ids = {v.tracklet_id for r in rn.results for v in r.tracklets}
        self.assertEqual(len(ids), 1)
        tid = ids.pop()
        self.assertTrue(tid.startswith("T") and len(tid) == 4)
        self.assertNotEqual(tid, TrackerConfig.default().get("target_track_id"))
        self.assertEqual(rn.results[-1].tracklets[0].status, P.ACTIVE)
        self.assertFalse(rn.results[-1].tracklets[0].provisional)
        self.check_all(rn)

    def test_deterministic(self):
        frames = combine(crossing(40, FPS, speed_h_per_s=3, dy_h=0.1, merge_iou=0.4),
                         walker(40, FPS, x0=900, name="TARGET", drop=(10, 11)))
        a, b = run(frames), run(frames)
        self.assertEqual(a.records_json(), b.records_json())

    def test_candidate_order_does_not_change_the_partition(self):
        """Permuting the detector's output order changes indices only: which detections are grouped into which
        tracklet (over the whole run) must be identical."""
        base = combine(crossing(50, FPS, speed_h_per_s=3, dy_h=0.1, merge_iou=0.4),
                       walker(50, FPS, x0=900, name="TARGET", drop=(20, 21)))
        perm = [Frame(list(reversed(f.dets)), list(reversed(f.owners)), f.camera) for f in base]

        def partition(rn):
            groups = {}
            for r in rn.results:
                for v in r.tracklets:
                    if v.observed:
                        groups.setdefault(v.tracklet_id, []).append((r.rel_frame, tuple(round(x, 6) for x in v.box)))
            return sorted(tuple(g) for g in groups.values())
        self.assertEqual(partition(run(base)), partition(run(perm)))

    def test_input_validation(self):
        cfg = TrackerConfig.default()
        pop = P.Population(cfg, W, H)
        pop.update(0, 0.0, [], still())
        with self.assertRaises(P.PopulationError):
            pop.update(0, 0.1, [], still())                        # frame not increasing
        with self.assertRaises(P.PopulationError):
            pop.update(1, 0.0, [], still())                        # time not increasing
        with self.assertRaises(P.PopulationError):
            pop.update(1, 0.1, [(0, 0, 1, 1)], still())            # not a PersonCandidate
        with self.assertRaises(P.PopulationError):
            pop.update(1, 0.1, [cand((0, 0, float("nan"), 5))], still())
        with self.assertRaises(P.PopulationError):
            pop.update(1, 0.1, [cand((10, 10, 5, 20))], still())   # negative width
        with self.assertRaises(P.PopulationError):
            pop.update(1, 0.1, [], "still")
        with self.assertRaises(P.PopulationError):
            P.Population(cfg, 0, H)


class TestDisappearance(Base):
    def test_short_gap_in_open_space_continues(self):
        rn = run(walker(30, FPS, drop=(10, 11, 12)))               # 0.1 s gap, nobody near
        ids = {v.tracklet_id for r in rn.results for v in r.tracklets}
        self.assertEqual(len(ids), 1)
        for k in (10, 11, 12):
            v = rn.results[k].tracklets[0]
            self.assertEqual(v.status, P.HIDDEN)
            self.assertEqual(v.observation_source, PREDICTED)
            self.assertIsNone(v.observed_box)
        self.assertIn(P.REASSOCIATED_AFTER_GAP, rn.results[13].tracklets[0].flags)
        self.check_all(rn)

    def test_predicted_box_follows_velocity_during_gap(self):
        rn = run(walker(30, FPS, drop=(15, 16)))
        last = rn.results[14].tracklets[0].box
        pred = rn.results[16].tracklets[0].box
        self.assertGreater(pred[0], last[0])                        # moved on, in the walking direction

    def test_short_gap_with_contact_starts_new_tracklet(self):
        # TARGET undetected for 0.1 s while PERSON_A walks over its predicted position, then PERSON_A vanishes
        tgt = walker(45, FPS, x0=300, vx_h_per_s=0.0, drop=(10, 11, 12), name="TARGET")
        other = []
        for k in range(45):
            if 8 <= k <= 12:
                other.append(Frame([cand(box_at(300 + (k - 10) * 6, 360, 32, 80))], [("PERSON_A",)]))
            else:
                other.append(Frame([], []))
        rn = run(combine(tgt, other))
        self.assert_no_swap(rn)
        before = {v.tracklet_id for r in rn.results[:10] for v in r.tracklets}
        hold_frames = math.ceil(TrackerConfig.default().get("population_merged_region_hold_s") * FPS)
        new_ids = set()
        for k in range(13, 45):
            for o in rn.results[k].observations:
                # the reappearance is never a continuation of a pre-contact tracklet: merged-suspected, or new
                self.assertNotIn(o.tracklet_id, before)
                if o.tracklet_id is None:
                    self.assertIn(P.MERGED_SUSPECTED, o.flags)
                elif o.tracklet_id not in new_ids:
                    new_ids.add(o.tracklet_id)
                    self.assertTrue(set(o.possible_predecessors) & before,
                                    "the new tracklet must be linked to the pre-contact tracklets")
        # coverage: once the merged region's hold has elapsed, the TARGET is tracked again
        self.assertIsNotNone(rn.results[min(44, 10 + hold_frames + 2)].observations[0].tracklet_id)
        self.check_all(rn)

    def test_gap_longer_than_max_gap_starts_new_tracklet_with_predecessor(self):
        drop = tuple(range(10, 18))                                  # 0.27 s > 0.2 s, < 0.5 s
        rn = run(walker(30, FPS, drop=drop))
        old = rn.results[9].tracklets[0].tracklet_id
        new = [v for v in rn.results[18].tracklets if v.observed][0]
        self.assertNotEqual(new.tracklet_id, old)
        self.assertIn(old, new.possible_predecessors)
        hidden_old = [v for v in rn.results[18].tracklets if v.tracklet_id == old]
        self.assertEqual(len(hidden_old), 1)                         # not closed: could be someone else hidden
        self.assertIn(new.tracklet_id, hidden_old[0].possible_successors)
        self.check_all(rn)

    def test_long_gap_closes_and_holds(self):
        drop = tuple(range(10, 40))
        rn = run(walker(45, FPS, drop=drop))
        old = rn.results[9].tracklets[0].tracklet_id
        sources = [r.tracklet(old).observation_source for r in rn.results[10:25] if r.tracklet(old)]
        self.assertIn(PREDICTED, sources)
        self.assertIn(HELD, sources)                                 # extrapolation stops after the horizon
        self.assertEqual(sources.index(HELD), sum(1 for s in sources if s == PREDICTED))
        self.assertEqual(rn.closed()[old], P.UNMATCHED_TIMEOUT)
        new = [v for v in rn.results[40].tracklets if v.observed][0]
        self.assertNotEqual(new.tracklet_id, old)
        self.assertEqual(new.possible_predecessors, ())              # too long: no positional link
        self.check_all(rn)

    def test_leaving_frame_closes(self):
        frames = []
        for k in range(40):
            x = 1180 + k * 10
            if x + 16 < W:
                frames.append(Frame([cand(box_at(x, 360, 32, 80), edge=x + 16 >= W - 2)], [("TARGET",)]))
            else:
                frames.append(Frame([], []))
        rn = run(frames)
        self.assertIn(P.LEFT_FRAME, rn.closed().values())


class TestCrossings(Base):
    def test_crossing_matrix_never_swaps(self):
        n = 0
        for speed in (1.0, 3.0, 6.0):
            for dy in (0.0, 0.1, 0.3, 0.6):
                for mode in ("merge", "hide", "both_visible"):
                    with self.subTest(speed=speed, dy=dy, mode=mode):
                        frames = crossing(60, FPS, speed_h_per_s=speed, dy_h=dy,
                                          merge_iou=0.3 if mode == "merge" else None, hide_back=mode == "hide")
                        rn = run(frames)
                        self.assert_no_swap(rn)
                        self.check_all(rn)
                        # coverage: once apart again, both people are tracked (no pass by fragmenting forever)
                        self.assertEqual(sum(v.observed for v in rn.results[-1].tracklets), 2)
                        n += 1
        self.assertEqual(n, 36)

    def test_same_kit_is_irrelevant_population_is_geometry_only(self):
        # The population takes no image: two identical-looking people are handled exactly like any two people.
        frames = crossing(60, FPS, speed_h_per_s=3.0, dy_h=0.05, merge_iou=None)
        rn = run(frames)
        self.assert_no_swap(rn)
        ev = [e for r in rn.results for e in r.contact_events]
        self.assertTrue(ev, "a near-full overlap must produce a contact event")

    def test_crossing_events_link_predecessors(self):
        rn = run(crossing(60, FPS, speed_h_per_s=3.0, dy_h=0.0, merge_iou=0.3))
        kinds = {e.kind for r in rn.results for e in r.contact_events}
        self.assertIn(P.EVENT_MERGED, kinds)
        before = {v.tracklet_id for v in rn.results[0].tracklets}
        after = [v for v in rn.results[-1].tracklets]
        self.assertEqual(len(after), 2)
        for v in after:
            self.assertNotIn(v.tracklet_id, before)

    def test_side_by_side_slight_overlap_keeps_continuity(self):
        # coverage: two people walking together with slightly overlapping boxes are not split every frame
        a = walker(40, FPS, x0=300, name="PERSON_A")
        b = walker(40, FPS, x0=328, name="PERSON_B")                # boxes overlap by 4 px of 32
        rn = run(combine(a, b))
        ids = {v.tracklet_id for r in rn.results for v in r.tracklets}
        self.assertEqual(len(ids), 2)
        self.assertTrue(all(P.OVERLAPS_OTHER in o.flags for o in rn.results[5].observations))
        self.assert_no_swap(rn)


class TestMerged(Base):
    def test_multi_claim_merged_box(self):
        a = box_at(400, 360, 32, 80)
        b = box_at(440, 360, 32, 80)
        frames = [Frame([cand(a), cand(b)], [("PERSON_A",), ("PERSON_B",)]) for _ in range(5)]
        m = (a[0], a[1], b[2], b[3])
        frames += [Frame([cand(m)], [("PERSON_A", "PERSON_B")])]
        rn = run(frames)
        r = rn.results[-1]
        o = r.observations[0]
        self.assertIn(P.MERGED_SUSPECTED, o.flags)
        self.assertIn(P.MULTI_CLAIM, o.merge_reasons)
        self.assertIsNone(o.tracklet_id)
        self.assertEqual(sorted(rn.closed().values()), [P.MERGED_CONTACT, P.MERGED_CONTACT])
        comp = rn.population.competitors(m, 80.0)
        self.assertTrue(any(c.ref == o.ref and c.kind == P.OBSERVATION for c in comp))
        self.check_all(rn)

    def test_area_ratio_single_claimant(self):
        # R2 DEV remediation F3 (NEW_R2): with no possible second body touching the box, size-only evidence is first a
        # SIZE_ANOMALY (abstained on: not an appearance sample; the tracklet is not closed), and becomes a merge with
        # its AREA_RATIO reason once it lasts merged_anomaly_max_s. Previously the merge was immediate.
        a = box_at(400, 360, 32, 80)
        frames = [Frame([cand(a)], [("TARGET",)]) for _ in range(12)]
        big = box_at(400, 360, 60, 90)                               # area x2.1 in one frame
        hold = int(math.ceil(TrackerConfig.default().get("merged_anomaly_max_s") * 30.0)) + 2
        frames += [Frame([cand(big)], [("TARGET", "PERSON_A")]) for _ in range(hold)]
        rn = run(frames)
        first = rn.results[12].observations[0]
        self.assertIn(P.SIZE_ANOMALY, first.flags)
        v = rn.results[12].tracklet(first.tracklet_id)
        self.assertFalse(v.bank_eligible)
        self.assertIn(P.SIZE_ANOMALY, v.bank_refusals)
        o = rn.results[-1].observations[0]
        self.assertIn(P.AREA_RATIO, o.merge_reasons)
        self.assertIsNone(o.tracklet_id)

    def test_aspect_wide_side_by_side(self):
        a = box_at(400, 360, 32, 80)
        frames = [Frame([cand(a)], [("TARGET",)]) for _ in range(12)]
        wide = box_at(405, 360, 56, 80)                              # x1.75 wider, same height
        hold = int(math.ceil(TrackerConfig.default().get("merged_anomaly_max_s") * 30.0)) + 2
        frames += [Frame([cand(wide)], [("TARGET", "PERSON_A")]) for _ in range(hold)]
        rn = run(frames, cfg=TrackerConfig.with_overrides({"merged_area_ratio": 10.0,
                                                           "merged_uncorroborated_area_ratio": 10.0}))
        self.assertIn(P.SIZE_ANOMALY, rn.results[12].observations[0].flags)
        o = rn.results[-1].observations[0]
        self.assertEqual(o.merge_reasons, (P.ASPECT_WIDE,))

    def test_persistent_merge_is_not_tracked(self):
        a, b = box_at(400, 360, 32, 80), box_at(440, 360, 32, 80)
        m = (a[0], a[1], b[2], b[3])
        frames = [Frame([cand(a), cand(b)], [("PERSON_A",), ("PERSON_B",)]) for _ in range(12)]
        frames += [Frame([cand(m)], [("PERSON_A", "PERSON_B")]) for _ in range(8)]
        rn = run(frames)
        for r in rn.results[12:]:
            o = r.observations[0]
            self.assertIn(P.MERGED_SUSPECTED, o.flags)
            self.assertIsNone(o.tracklet_id)
        self.assertIn(P.PERSISTENT_MERGE, rn.results[-1].observations[0].merge_reasons)
        self.check_all(rn)

    def test_split_after_merge_links_both_claimants_and_refuses_bank(self):
        a, b = box_at(400, 360, 32, 80), box_at(440, 360, 32, 80)
        m = (a[0], a[1], b[2], b[3])
        frames = [Frame([cand(a), cand(b)], [("PERSON_A",), ("PERSON_B",)]) for _ in range(5)]
        frames += [Frame([cand(m)], [("PERSON_A", "PERSON_B")]) for _ in range(3)]
        frames += [Frame([cand(a), cand(b)], [("PERSON_A",), ("PERSON_B",)]) for _ in range(3)]
        rn = run(frames)
        first_ids = sorted(v.tracklet_id for v in rn.results[0].tracklets)
        r = rn.results[8]
        births = [v for v in r.tracklets if v.status == P.NEW]
        self.assertEqual(len(births), 2)
        for v in births:
            self.assertEqual(sorted(v.possible_predecessors), first_ids)
            self.assertIn(P.IN_MERGED_REGION, v.bank_refusals)
        self.assert_no_swap(rn)


class TestOcclusionVsMerge(Base):
    """Real footage has people at different depths. A large near person passing in front of a small far person is
    OCCLUSION, not a merge: the near person's box is still one body (regression for the R1-E development-clip finding, where
    size tests against the small hidden body flagged most detections as merged)."""

    def near_far(self, n=60):
        frames = []
        far = box_at(640, 330, 24, 60)                                 # small, far away, standing
        for k in range(n):
            nx = 400 + 9.0 * k                                         # large, near, walking across in front
            near = box_at(nx, 400, 64, 160)
            if far[0] >= near[0] and far[2] <= near[2]:
                frames.append(Frame([cand(near)], [("PERSON_A",)]))    # far person fully hidden
            else:
                frames.append(Frame([cand(near), cand(far)], [("PERSON_A",), ("PERSON_B",)]))
        return frames

    def test_near_person_passing_in_front_is_not_merged(self):
        rn = run(self.near_far())
        self.assert_no_swap(rn)
        self.check_all(rn)
        near_ids = set()
        for r, fr in zip(rn.results, rn.frames):
            for o in r.observations:
                if fr.owners[o.index] == ("PERSON_A",):
                    self.assertNotIn(P.MERGED_SUSPECTED, o.flags, f"rel {r.rel_frame}: occluder flagged merged")
                    near_ids.add(o.tracklet_id)
        self.assertEqual(len(near_ids), 1, "the near person's tracklet must continue through the occlusion")

    def test_occluder_box_is_not_an_appearance_sample_while_covering(self):
        cfg = TrackerConfig.default()
        rn = run(self.near_far())
        refused = 0
        for r, fr in zip(rn.results, rn.frames):
            for v in r.tracklets:
                if v.observed and fr.owners[v.detection_index] == ("PERSON_A",):
                    o = r.observations[v.detection_index]
                    if o.covered_fraction > cfg.get("bank_max_covered_fraction") or \
                            o.max_iou_other > cfg.get("bank_contact_iou"):
                        self.assertFalse(v.bank_eligible)
                        self.assertIn(P.NOT_ISOLATED, v.bank_refusals)
                        refused += 1
        self.assertGreater(refused, 0, "the occlusion must actually be exercised")

    def test_new_large_box_over_small_hidden_body_is_not_area_merged(self):
        small = box_at(640, 330, 24, 60)
        frames = [Frame([cand(small)], [("PERSON_B",)]) for _ in range(5)]
        big = box_at(645, 360, 64, 160)                                # a near person appears, covering it
        frames += [Frame([cand(big)], [("PERSON_A",)]) for _ in range(5)]
        rn = run(frames)
        for r in rn.results[5:]:
            o = r.observations[0]
            self.assertNotIn(P.MERGED_SUSPECTED, o.flags)
            self.assertIsNotNone(o.tracklet_id)
        self.assert_no_swap(rn)


class TestNestedPartBoxes(Base):
    """Real detectors also return part-body boxes nested inside the same person's full box (regression for the
    R1-E development-clip finding: nested boxes became tracklets, the next single box 'claimed' them all and was
    flagged merged, and the merged region kept a part-box as its size reference)."""

    def test_nested_part_boxes_are_not_tracklets_and_cause_no_merge(self):
        frames = []
        for k in range(30):
            cx = 400 + 4.0 * k
            full = box_at(cx, 400, 60, 150)
            dets = [cand(full)]
            owners = [("PERSON_A",)]
            if k % 3 != 2:                                            # part boxes come and go
                dets.append(cand((full[0] + 8, full[1] + 5, full[2] - 10, full[1] + 80), 0.7))
                owners.append(("PERSON_A",))
            if k % 4 == 0:
                dets.append(cand((full[0] + 15, full[1] + 70, full[2] - 12, full[3] - 2), 0.6))
                owners.append(("PERSON_A",))
            frames.append(Frame(dets, owners))
        rn = run(frames)
        self.check_all(rn)
        ids = {v.tracklet_id for r in rn.results for v in r.tracklets}
        self.assertEqual(len(ids), 1, "one person, one tracklet")
        for r in rn.results:
            self.assertFalse(any(P.MERGED_SUSPECTED in o.flags for o in r.observations))
            full_obs = r.observations[0]
            if len(r.observations) > 1:
                self.assertIn(P.CONTAINS_OTHER, full_obs.flags)
                v = r.tracklet(full_obs.tracklet_id)
                self.assertFalse(v.bank_eligible)                     # a box containing another is not isolated
                for o in r.observations[1:]:
                    self.assertIn(P.NESTED, o.flags)
                    self.assertIsNone(o.tracklet_id)
                    self.assertEqual(o.nested_in, 0)

    def test_alternative_box_mostly_inside_is_nested_but_side_by_side_contact_is_not(self):
        person = box_at(400, 400, 50, 140)
        with_shadow = (person[0] - 6, person[1] + 20, person[2] + 30, person[3] + 6)   # most of person inside
        rn = run([Frame([cand(with_shadow), cand(person)], [("PERSON_A",), ("PERSON_A",)]) for _ in range(3)])
        cont = rn.results[0].observations
        self.assertIn(P.NESTED, cont[1].flags)
        self.assertEqual(len({v.tracklet_id for r in rn.results for v in r.tracklets}), 1)
        a, b = box_at(400, 400, 50, 140), box_at(430, 405, 50, 140)                    # two people in contact
        rn = run([Frame([cand(a), cand(b)], [("PERSON_A",), ("PERSON_B",)]) for _ in range(3)])
        self.assertFalse(any(P.NESTED in o.flags for o in rn.results[0].observations))
        self.assertEqual(len(rn.results[0].tracklets), 2)

    def test_over_wide_alternative_box_does_not_hijack_or_merge(self):
        """The detector sometimes adds a wider box (person + shadow) around a tracked person: the tracklet keeps
        the box that matches its prediction, the wide box starts nothing and nothing is flagged merged."""
        frames = []
        for k in range(40):
            p_ = box_at(400 + 3.0 * k, 400, 50, 140)
            dets, owners = [cand(p_)], [("PERSON_A",)]
            if 15 <= k <= 25:
                dets.insert(0, cand((p_[0] - 4, p_[1] + 25, p_[2] + 45, p_[3] + 4), 0.8))
                owners.insert(0, ("PERSON_A",))
            frames.append(Frame(dets, owners))
        rn = run(frames)
        self.check_all(rn)
        self.assertEqual(len({v.tracklet_id for r in rn.results for v in r.tracklets}), 1)
        for r in rn.results:
            self.assertFalse(any(P.MERGED_SUSPECTED in o.flags for o in r.observations))
        r = rn.results[20]
        self.assertIn(P.ALTERNATIVE, r.observations[0].flags)
        self.assertIsNone(r.observations[0].tracklet_id)
        self.assertEqual(r.observations[1].tracklet_id, r.tracklets[0].tracklet_id)

    def test_far_person_nested_in_near_box_keeps_both_tracklets(self):
        """Two real people: a small far person standing inside the box of a large near person (visible between
        the near person's legs). Both are tracked; neither box is an appearance sample while nested."""
        near = box_at(400, 400, 90, 200)
        far = box_at(395, 440, 20, 50)
        frames = [Frame([cand(near), cand(box_at(700, 440, 20, 50))], [("PERSON_A",), ("PERSON_B",)])]
        # the far person walks into the near person's box
        for k in range(1, 30):
            fx = 700 - 10.5 * k if k < 29 else 395
            fb = box_at(max(fx, 395), 440, 20, 50)
            frames.append(Frame([cand(near), cand(fb)], [("PERSON_A",), ("PERSON_B",)]))
        rn = run(frames)
        self.assert_no_swap(rn)
        self.check_all(rn)
        last = rn.results[-1]
        # far too small to be a part of the near body: a separate, occluded body (not NESTED), and the near box
        # contains it (never an appearance sample)
        self.assertNotIn(P.NESTED, last.observations[1].flags)
        self.assertIn(P.CONTAINS_OTHER, last.observations[0].flags)
        self.assertIsNotNone(last.observations[0].tracklet_id)
        self.assertIsNotNone(last.observations[1].tracklet_id)
        for v in last.tracklets:
            self.assertFalse(v.bank_eligible)

    def test_nested_box_is_still_a_competitor(self):
        full = box_at(400, 400, 60, 150)
        inner = (full[0] + 5, full[1] + 5, full[0] + 30, full[1] + 70)
        rn = run([Frame([cand(full), cand(inner)], [("PERSON_A",), ("PERSON_B",)]) for _ in range(3)])
        v = rn.results[-1].tracklets[0]
        comp = rn.population.competitors(v.box, 150.0, exclude_tracklet_ids=[v.tracklet_id])
        self.assertEqual([c.ref for c in comp], ["D1"])
        self.assertIn(P.NESTED, comp[0].flags)

    def test_partial_birth_growing_to_full_is_not_a_merge(self):
        """A person first seen as an upper-body box (rest hidden) whose box then grows to the full body: the
        first size was never an established reference, so the growth is not a merge."""
        frames = [Frame([cand((380.0, 330.0, 420.0, 395.0))], [("PERSON_A",)]) for _ in range(2)]
        frames += [Frame([cand((376.0, 330.0, 424.0, 470.0))], [("PERSON_A",)]) for _ in range(5)]
        rn = run(frames)
        for r in rn.results:
            self.assertFalse(any(P.MERGED_SUSPECTED in o.flags for o in r.observations))
        self.assert_no_swap(rn)


class TestReviewRegressions(Base):
    """Regressions for the R1-E independent code review (general failure classes, neutral names)."""

    def test_box_born_from_an_unresolved_merge_never_feeds_a_bank(self):
        """H1: two people merge into one box and stay merged longer than the region hold. The box that becomes a
        tracklet after the region expires may still hold two bodies: it carries MERGE_LINEAGE and is never an
        appearance sample."""
        a, b = box_at(400, 360, 50, 100), box_at(460, 360, 50, 100)
        m = (a[0], a[1], b[2], b[3])
        frames = [Frame([cand(a), cand(b)], [("PERSON_A",), ("PERSON_B",)]) for _ in range(3)]
        frames += [Frame([cand(m)], [("PERSON_A", "PERSON_B")]) for _ in range(45)]
        rn = run(frames)
        born = None
        for r in rn.results[3:]:
            for v in r.tracklets:
                if v.observed:
                    born = v
                    self.assertFalse(v.bank_eligible, f"rel {r.rel_frame}: merged lineage sampled")
                    self.assertIn(P.MERGE_LINEAGE, v.bank_refusals)
                    self.assertIn(P.MERGE_LINEAGE, v.flags)
        self.assertIsNotNone(born, "coverage: the merged box is tracked again after the hold (as lineage)")

    def test_merge_lineage_inherits_through_gaps(self):
        """H1 variant: the merged box goes undetected longer than the hold, then returns."""
        a, b = box_at(400, 360, 50, 100), box_at(460, 360, 50, 100)
        m = (a[0], a[1], b[2], b[3])
        frames = [Frame([cand(a), cand(b)], [("PERSON_A",), ("PERSON_B",)]) for _ in range(3)]
        frames += [Frame([cand(m)], [("PERSON_A", "PERSON_B")]) for _ in range(3)]
        frames += [Frame([], []) for _ in range(20)]
        frames += [Frame([cand(m)], [("PERSON_A", "PERSON_B")]) for _ in range(30)]
        rn = run(frames)
        for r in rn.results[6:]:
            for v in r.tracklets:
                if v.observed:
                    self.assertFalse(v.bank_eligible, f"rel {r.rel_frame}")

    def test_clean_split_clears_lineage(self):
        """A merged region that resolves (both bodies come out separately) is evidence of separation: the new
        tracklets are not merge lineage, and once isolated they may be sampled."""
        a, b = box_at(400, 360, 50, 100), box_at(460, 360, 50, 100)
        m = (a[0], a[1], b[2], b[3])
        frames = [Frame([cand(a), cand(b)], [("PERSON_A",), ("PERSON_B",)]) for _ in range(3)]
        frames += [Frame([cand(m)], [("PERSON_A", "PERSON_B")]) for _ in range(3)]
        for k in range(1, 30):
            frames.append(Frame([cand(box_at(400 - 8 * k, 360, 50, 100)), cand(box_at(460 + 8 * k, 360, 50, 100))],
                                [("PERSON_A",), ("PERSON_B",)]))
        rn = run(frames)
        last = rn.results[-1]
        self.assertEqual(len(last.tracklets), 2)
        for v in last.tracklets:
            self.assertNotIn(P.MERGE_LINEAGE, v.flags)
            self.assertTrue(v.bank_eligible)
        self.assert_no_swap(rn)

    def test_short_box_is_partial_without_an_established_reference(self):
        """M1: a young tracklet's box loses its lower part (hidden by a non-person). The short box continues the
        tracklet as PARTIAL: no velocity or size update, no bank sample, and the full body continues afterwards."""
        frames = [Frame([cand(box_at(400, 400, 40, 100))], [("PERSON_A",)]) for _ in range(3)]
        frames += [Frame([cand((380.0, 350.0, 420.0, 405.0))], [("PERSON_A",)]) for _ in range(3)]
        frames += [Frame([cand(box_at(400, 400, 40, 100))], [("PERSON_A",)]) for _ in range(5)]
        rn = run(frames)
        self.assertEqual(len({v.tracklet_id for r in rn.results for v in r.tracklets}), 1)
        for r in rn.results[3:6]:
            v = r.tracklets[0]
            self.assertIn(P.PARTIAL, v.flags)
            self.assertIn(P.PARTIAL, v.bank_refusals)
        st = rn.population._live[0].motion
        self.assertAlmostEqual(st.vy, 0.0, places=6)
        self.assertAlmostEqual(st.log_h_rate, 0.0, places=6)

    def test_near_person_first_seen_containing_a_far_person_gets_a_tracklet(self):
        """M2: a small far person is tracked; a large near person is first detected with the far person's box
        inside its own. A box that small cannot be a part of the near body: both are bodies, both tracked."""
        far = [box_at(400, 380, 20, 45) for _ in range(30)]
        frames = [Frame([cand(far[k])], [("PERSON_B",)]) for k in range(5)]
        near = box_at(405, 400, 80, 230)
        frames += [Frame([cand(near), cand(far[k])], [("PERSON_A",), ("PERSON_B",)]) for k in range(5, 30)]
        rn = run(frames)
        last = rn.results[-1]
        self.assertIsNotNone(last.observations[0].tracklet_id)
        self.assertIsNotNone(last.observations[1].tracklet_id)
        self.assertNotIn(P.ALTERNATIVE, last.observations[0].flags)
        self.assertIn(P.CONTAINS_OTHER, last.observations[0].flags)
        for v in last.tracklets:
            self.assertFalse(v.bank_eligible)
        self.assert_no_swap(rn)

    def test_sample_appearance_once_per_frame_and_frame_checked(self):
        """M3: a second call for the same frame adds nothing, and the image must belong to the frame."""
        cfg = TrackerConfig.default()
        ext = AppearanceExtractor(cfg, _ColourEmbedding())
        pop = P.Population(cfg, W, H)
        img = np.full((H, W, 3), (40, 120, 40), np.uint8)
        b = box_at(400, 400, 40, 100)
        x1, y1, x2, y2 = (int(v) for v in b)
        img[y1:y2, x1:x2] = (40, 40, 220)
        pop.update(0, 0.0, [cand(b)], identity_motion("FIRST_FRAME"))
        first = pop.sample_appearance(0, img, ext)
        self.assertTrue(first)
        self.assertEqual(pop.sample_appearance(0, img, ext), {})
        with self.assertRaises(P.PopulationError):
            pop.sample_appearance(1, img, ext)

    def test_closed_tracklets_learn_their_successors(self):
        """L1: a tracklet closed earlier still records the later birth that names it as predecessor."""
        a, b = box_at(400, 360, 50, 100), box_at(460, 360, 50, 100)
        m = (a[0], a[1], b[2], b[3])
        frames = [Frame([cand(a), cand(b)], [("PERSON_A",), ("PERSON_B",)]) for _ in range(3)]
        frames += [Frame([cand(m)], [("PERSON_A", "PERSON_B")]) for _ in range(30)]
        rn = run(frames)
        mem = {m_.tracklet_id: m_ for m_ in rn.population.recently_closed()}
        born = [v for v in rn.results[-1].tracklets if v.observed][0]
        for pred in born.possible_predecessors:
            self.assertIn(born.tracklet_id, mem[pred].possible_successors)
            self.assertIn(mem[pred].last_box_source, (P.LAST_OBSERVED, P.LAST_PREDICTED))

    def test_camera_reset_mid_stream_breaks_continuity(self):
        """L3: a FIRST_FRAME camera status after the first frame (estimator reset) cannot vouch for continuity."""
        frames = walker(10, FPS)
        frames.append(Frame(list(frames[-1].dets), [("TARGET",)], identity_motion("FIRST_FRAME")))
        rn = run(frames)
        self.assertEqual([c.reason for c in rn.results[10].closed], [P.CAMERA_RESET])


class TestPartialAndDuplicates(Base):
    def test_partial_height_continues_without_size_update(self):
        frames = walker(10, FPS, vx_h_per_s=0.0)
        legs_hidden = (284.0, 320.0, 316.0, 368.0)                   # top part only, 0.6 of the height
        frames += [Frame([cand(legs_hidden)], [("TARGET",)]) for _ in range(3)]
        frames += walker(3, FPS, vx_h_per_s=0.0)
        rn = run(frames)
        ids = {v.tracklet_id for r in rn.results for v in r.tracklets}
        self.assertEqual(len(ids), 1)                                # continues through partial and back
        v = rn.results[10].tracklets[0]
        self.assertIn(P.PARTIAL, v.flags)
        self.assertIn(P.PARTIAL, v.bank_refusals)
        self.check_all(rn)

    def test_edge_partial(self):
        frames = [Frame([cand((0.0, 300.0, 20.0, 380.0), edge=True)], [("TARGET",)]) for _ in range(3)]
        rn = run(frames)
        o = rn.results[0].observations[0]
        self.assertIn(P.PARTIAL, o.flags)
        self.assertIn(P.TOUCHES_IMAGE_EDGE, o.flags)
        self.assertFalse(rn.results[1].tracklets[0].bank_eligible)

    def test_duplicates_suppressed_but_kept(self):
        a = box_at(400, 360, 32, 80)
        dup = (a[0] + 1, a[1] + 1, a[2] + 1, a[3])
        frames = [Frame([cand(a, 0.9), cand(dup, 0.6)], [("TARGET",), ("TARGET",)]) for _ in range(5)]
        rn = run(frames)
        r = rn.results[-1]
        self.assertEqual(len(r.tracklets), 1)
        self.assertEqual(r.observations[1].duplicate_of, 0)
        self.assertIn(P.DUPLICATE_SUPPRESSED, r.observations[1].flags)
        self.assertIn(P.DUPLICATE_ABSORBED, r.observations[0].flags)
        self.assertIn(P.DUPLICATE_ABSORBED, r.tracklets[0].bank_refusals)
        comp = rn.population.competitors(a, 80.0)
        self.assertFalse(any(c.ref == "D1" for c in comp))           # a suppressed duplicate is not a body

    def test_genuine_overlapping_people_are_not_deleted(self):
        a = box_at(400, 360, 32, 80)
        b = box_at(416, 370, 32, 80)                                 # IoU ~0.4: two people
        frames = [Frame([cand(a), cand(b)], [("PERSON_A",), ("PERSON_B",)]) for _ in range(5)]
        rn = run(frames)
        r = rn.results[0]
        self.assertEqual(len(r.tracklets), 2)
        self.assertTrue(all(o.is_body for o in r.observations))
        self.assertTrue(r.overlap_pairs)

    def test_duplicate_order_independent_of_input_order(self):
        a = box_at(400, 360, 32, 80)
        dup = (a[0] + 1, a[1] + 1, a[2] + 1, a[3])
        r1 = run([Frame([cand(a, 0.9), cand(dup, 0.6)], [("T",), ("T",)])]).results[0]
        r2 = run([Frame([cand(dup, 0.6), cand(a, 0.9)], [("T",), ("T",)])]).results[0]
        self.assertEqual(r1.tracklets[0].box, r2.tracklets[0].box)   # the higher score is kept either way


class TestCamera(Base):
    def _static_world(self, n, cam_fn, mover):
        """Two people static in the world; the camera moves; their image boxes move with the camera."""
        frames = []
        boxes = [box_at(400, 360, 32, 80), box_at(800, 360, 32, 80)]
        for k in range(n):
            if k > 0:
                cam = cam_fn(k)
                boxes = [cam.warp_box(b) for b in boxes]
            else:
                cam = None
            frames.append(Frame([cand(b) for b in boxes], [("PERSON_A",), ("PERSON_B",)], cam))
        return frames

    def test_fast_pan_compensated(self):
        frames = self._static_world(20, lambda k: pan(-40.0), None)          # 40 px per frame = 1.2 W/s
        rn = run(frames)
        ids = {v.tracklet_id for r in rn.results for v in r.tracklets}
        self.assertEqual(len(ids), 2)
        self.assert_no_swap(rn)

    def test_pan_without_compensation_would_break(self):
        frames = self._static_world(20, lambda k: pan(-40.0), None)
        for f in frames[1:]:
            f.camera = still()                                               # lie: no camera motion measured
        rn = run(frames)
        ids = {v.tracklet_id for r in rn.results for v in r.tracklets}
        self.assertGreater(len(ids), 2)                                       # proves the compensation matters
        self.assert_no_swap(rn)

    def test_zoom_compensated(self):
        frames = self._static_world(20, lambda k: zoom(1.03), None)
        rn = run(frames)
        ids = {v.tracklet_id for r in rn.results for v in r.tracklets}
        self.assertEqual(len(ids), 2)

    def test_hard_cut_breaks_all_continuity(self):
        frames = combine(walker(10, FPS, name="TARGET"), walker(10, FPS, x0=800, name="PERSON_A"))
        cut = identity_motion(CUT, reasons=("HISTOGRAM_CHANGE",), continuity_broken=True)
        # after the cut the same image positions hold different people: nothing may continue
        frames += [Frame(list(frames[-1].dets), [("PERSON_B",), ("PERSON_C",)], cut)]
        frames += [Frame(list(frames[-1].dets), [("PERSON_B",), ("PERSON_C",)])]
        rn = run(frames)
        r = rn.results[10]
        self.assertEqual(sorted(c.reason for c in r.closed), [P.CAMERA_CUT, P.CAMERA_CUT])
        self.assertTrue(all(v.status == P.NEW and not v.possible_predecessors for v in r.tracklets))
        self.assert_no_swap(rn)
        comp = rn.population.competitors(box_at(300, 360, 32, 80), 80.0)
        closed = [c for c in comp if c.kind == P.RECENTLY_CLOSED]
        self.assertEqual(len(closed), 2)
        self.assertTrue(all(not c.position_valid and c.distance_h is None for c in closed))

    def test_unknown_camera_breaks_by_default(self):
        frames = walker(10, FPS)
        frames.append(Frame(list(frames[-1].dets), [("TARGET",)],
                            identity_motion(UNKNOWN, reasons=("LOW_INLIER_FRACTION",), continuity_broken=True)))
        rn = run(frames)
        self.assertEqual([c.reason for c in rn.results[10].closed], [P.CAMERA_UNKNOWN])

    def test_unknown_camera_when_configured_not_to_break(self):
        cfg = TrackerConfig.with_overrides({"camera_unknown_breaks_continuity": False})
        frames = walker(10, FPS, drop=(10,))
        frames.append(Frame([], [], identity_motion(UNKNOWN, continuity_broken=False)))
        frames += walker(2, FPS)
        frames[11] = Frame(list(walker(12, FPS)[11].dets), [("TARGET",)],
                           identity_motion(UNKNOWN, continuity_broken=False))
        rn = run(frames, cfg=cfg)
        v = rn.results[10].tracklets[0]
        self.assertIn(P.CAMERA_UNCOMPENSATED, v.flags)
        self.assertIn(P.CAMERA_UNVERIFIED, rn.results[11].tracklets[0].bank_refusals)


class TestCompetitors(Base):
    def test_provisional_competitor_blocks_from_birth(self):
        tgt = walker(10, FPS, x0=400, vx_h_per_s=0.0, name="TARGET")
        late = [Frame([], []) for _ in range(9)] + [Frame([cand(box_at(420, 360, 32, 80))], [("PERSON_A",)])]
        rn = run(combine(tgt, late))
        r = rn.results[9]
        target_view = [v for v in r.tracklets if rn.frames[9].owners[v.detection_index] == ("TARGET",)][0]
        comp = rn.population.competitors(target_view.box, 80.0, exclude_tracklet_ids=[target_view.tracklet_id])
        self.assertEqual(len(comp), 1)
        self.assertTrue(comp[0].provisional)
        self.assertEqual(comp[0].kind, P.OBSERVATION)

    def test_hidden_and_far_bodies(self):
        tgt = walker(12, FPS, x0=400, vx_h_per_s=0.0, name="TARGET")
        a = walker(12, FPS, x0=450, vx_h_per_s=0.0, name="PERSON_A", drop=(11,))
        far = walker(12, FPS, x0=1100, vx_h_per_s=0.0, name="PERSON_B")
        rn = run(combine(tgt, a, far))
        r = rn.results[11]
        tv = [v for v in r.tracklets if v.observed and rn.frames[11].owners[v.detection_index] == ("TARGET",)][0]
        comp = rn.population.competitors(tv.box, 80.0, exclude_tracklet_ids=[tv.tracklet_id])
        kinds = [c.kind for c in comp]
        self.assertEqual(kinds, [P.HIDDEN_BODY])                    # PERSON_A hidden nearby; PERSON_B too far
        self.assertIn(comp[0].box_source, (PREDICTED, HELD))

    def test_hidden_predecessor_is_identifiable_for_self_exclusion(self):
        """Rule 28 support: a body re-detected as a new tracklet while its old tracklet is still HIDDEN at the same
        place appears twice (old hidden + new observed). The population cannot know they are one body, so it keeps
        both as competitors, but it records the link (possible_predecessors / possible_successors) so the identity
        layer can exclude the target's own predecessor instead of treating it as a different person."""
        rn = run(walker(30, FPS, drop=tuple(range(10, 18))))
        r = rn.results[18]
        new = [v for v in r.tracklets if v.observed][0]
        old = [v for v in r.tracklets if not v.observed][0]
        self.assertIn(old.tracklet_id, new.possible_predecessors)
        self.assertIn(new.tracklet_id, old.possible_successors)
        comp = rn.population.competitors(new.box, 80.0, exclude_tracklet_ids=[new.tracklet_id])
        self.assertEqual([c.tracklet_id for c in comp], [old.tracklet_id])
        comp = rn.population.competitors(new.box, 80.0,
                                         exclude_tracklet_ids=[new.tracklet_id, *new.possible_predecessors])
        self.assertEqual(comp, ())

    def test_exclusions(self):
        rn = run(combine(walker(3, FPS, x0=400), walker(3, FPS, x0=430, name="PERSON_A")))
        q = rn.results[-1].observations[0].bbox
        self.assertEqual(len(rn.population.competitors(q, 80.0)), 2)
        self.assertEqual(len(rn.population.competitors(q, 80.0, exclude_detection_indices=[0])), 1)
        with self.assertRaises(P.PopulationError):
            rn.population.competitors(q, 0.0)


class _ColourEmbedding(EmbeddingBackend):
    """Test embedding: the crop's mean colour (so each synthetic person, drawn in its own colour, has its own
    vector). A stand-in for DINOv2 that makes contamination visible."""

    def embed(self, crop):
        m = crop.reshape(-1, 3).mean(axis=0)
        return l2_normalise(np.concatenate([m, [40.0]]))

    def provenance(self):
        return {"backend": "test_colour"}


class TestBankContamination(Base):
    COLOURS = {"PERSON_A": (40, 40, 220), "PERSON_B": (220, 60, 40)}

    def _render(self, frame, fr):
        img = np.full((H, W, 3), (40, 120, 40), np.uint8)
        # draw the person further from the camera first; the GT owner of a box is its colour
        for name in ("PERSON_B", "PERSON_A"):
            for d, owners in zip(fr.dets, fr.owners):
                if owners == (name,):
                    x1, y1, x2, y2 = (int(round(v)) for v in d.bbox)
                    img[y1:y2, x1:x2] = self.COLOURS[name]
        return img

    def test_crossing_never_mixes_people_in_a_bank(self):
        cfg = TrackerConfig.default()
        ext = AppearanceExtractor(cfg, _ColourEmbedding())
        frames = crossing(80, FPS, speed_h_per_s=2.0, dy_h=0.05, merge_iou=None, w=40, h=100)
        sampled = []

        def hook(k, pop, res):
            out = pop.sample_appearance(k, self._render(k, frames[k]), ext)
            sampled.extend(tid for tid, (st, _) in out.items() if st == "ACCEPTED")

        rn = run(frames, cfg=cfg, hook=hook)
        self.assertTrue(sampled, "isolated people must be sampled at all (coverage)")
        owners = rn.tracklet_owners()
        banks = {}
        for tid in owners:
            bank = rn.population.bank_of(tid)
            if bank is not None and len(bank):
                banks[tid] = bank
        self.assertGreaterEqual(len(banks), 2)
        ref = {n: l2_normalise(np.concatenate([np.asarray(c, float), [40.0]])) for n, c in self.COLOURS.items()}
        for tid, bank in banks.items():
            self.assertEqual(len(owners[tid]), 1)
            who = next(iter(owners[tid]))
            for s in bank.samples:
                self.assertAlmostEqual(float(s.embedding @ ref[who]), 1.0, places=6,
                                       msg=f"{tid} ({who}) bank holds another person's appearance")
        for r in rn.results:
            for v in r.tracklets:
                if P.OVERLAPS_OTHER in v.flags and P.NOT_ISOLATED not in v.bank_refusals:
                    o = r.observations[v.detection_index]
                    self.assertLessEqual(o.max_iou_other, cfg.get("bank_contact_iou"))

    def test_same_kit_crossing_banks_sample_only_isolated_people(self):
        """Two people in IDENTICAL colours cross: colour cannot tell them apart, so the only protection is
        geometry. Every accepted bank sample must come from a frame where the two people's true boxes are
        separated (IoU <= bank_contact_iou), and no tracklet may span both people."""
        cfg = TrackerConfig.default()
        ext = AppearanceExtractor(cfg, _ColourEmbedding())
        frames = crossing(90, FPS, speed_h_per_s=2.0, dy_h=0.05, merge_iou=None, w=40, h=100)
        same = (200, 200, 200)
        accepted = []

        def render(fr):
            img = np.full((H, W, 3), (40, 120, 40), np.uint8)
            for d in fr.dets:
                x1, y1, x2, y2 = (int(round(v)) for v in d.bbox)
                img[y1:y2, x1:x2] = same
            return img

        def hook(k, pop, res):
            out = pop.sample_appearance(k, render(frames[k]), ext)
            accepted.extend(k for tid, (st, _) in out.items() if st == "ACCEPTED")

        rn = run(frames, cfg=cfg, hook=hook)
        self.assert_no_swap(rn)
        self.assertTrue(accepted, "coverage: isolated same-kit people are still sampled")
        for k in accepted:
            fr = frames[k]
            self.assertEqual(len(fr.dets), 2)
            self.assertLessEqual(P_iou(fr.dets[0].bbox, fr.dets[1].bbox), cfg.get("bank_contact_iou"))

    def test_sampling_interval_respected(self):
        cfg = TrackerConfig.default()
        ext = AppearanceExtractor(cfg, _ColourEmbedding())
        frames = walker(30, FPS, w=40, h=100, name="PERSON_A")
        offered = []

        def hook(k, pop, res):
            out = pop.sample_appearance(k, self._render(k, frames[k]), ext)
            offered.extend([k] * len(out))

        run(frames, cfg=cfg, hook=hook)
        gaps = np.diff(offered) / FPS
        self.assertTrue(np.all(gaps >= cfg.get("population_bank_sample_interval_s") - 1e-9))


class TestInvariance(Base):
    def _signature(self, rn):
        return (len({v.tracklet_id for r in rn.results for v in r.tracklets}), sorted(rn.closed().values()),
                [e.kind for r in rn.results for e in r.contact_events])

    def test_fps_invariance(self):
        a = run(crossing(60, 30.0, speed_h_per_s=3.0, dy_h=0.1, merge_iou=0.3), fps=30.0)
        b = run(crossing(120, 60.0, speed_h_per_s=3.0, dy_h=0.1, merge_iou=0.3), fps=60.0)
        sa, sb = self._signature(a), self._signature(b)
        self.assertEqual(sa[0], sb[0])
        self.assertEqual(sa[1], sb[1])
        self.assertEqual(set(sa[2]), set(sb[2]))

    def test_resolution_invariance(self):
        a = run(crossing(60, FPS, speed_h_per_s=3.0, dy_h=0.1, merge_iou=0.3))
        b = run(crossing(60, FPS, speed_h_per_s=3.0, dy_h=0.1, merge_iou=0.3, scale=2.0), width=2 * W,
                height=2 * H)
        self.assertEqual(self._signature(a), self._signature(b))

    def test_gap_rules_in_seconds(self):
        # the same 0.1 s gap at 30 and 60 fps continues; the same 0.27 s gap at both does not
        for fps in (30.0, 60.0):
            with self.subTest(fps=fps):
                short = tuple(range(int(fps * 0.4), int(fps * 0.5)))
                rn = run(walker(int(fps), fps, drop=short), fps=fps)
                self.assertEqual(len({v.tracklet_id for r in rn.results for v in r.tracklets}), 1)
                long_ = tuple(range(int(fps * 0.4), int(fps * 0.67)))
                rn = run(walker(int(fps), fps, drop=long_), fps=fps)
                self.assertEqual(len({v.tracklet_id for r in rn.results for v in r.tracklets}), 2)


class TestEndToEnd(Base):
    """Camera estimator + population on rendered frames: a textured world, a panning camera, two people standing
    still in the world (so all their image motion is camera motion), then a hard cut to another shot."""

    def test_pan_then_cut(self):
        import cv2
        from player_tracker_rb.camera import CameraMotionEstimator
        from tests_rb.test_camera import canvas, view
        cfg = TrackerConfig.default()
        world, other = canvas(1), canvas(7, brightness=0.35)
        people = [(1100.0, 700.0), (1400.0, 690.0)]                     # world centres
        pw, ph = 36.0, 90.0
        est = CameraMotionEstimator(cfg)
        pop = P.Population(cfg, W, H)
        ids_before = set()
        results = []
        n_pan = 30
        for k in range(n_pan + 5):
            cut = k >= n_pan
            src = other if cut else world
            cx = 1200.0 + 8.0 * k if not cut else 1000.0                  # 8 px per frame pan, then a new shot
            img = view(src, cx, 700.0, 1.0).copy()
            boxes = []
            for j, (wx, wy) in enumerate(people):
                ix, iy = wx - cx + W / 2, wy - 700.0 + H / 2
                b = (ix - pw / 2, iy - ph / 2, ix + pw / 2, iy + ph / 2)
                x1, y1, x2, y2 = (int(round(v)) for v in b)
                img[y1:y2, x1:x2] = (200, 60, 60) if j == 0 else (60, 60, 200)
                boxes.append(b)
            dets = [cand(b) for b in boxes]
            cam = est.update(img, boxes, 1.0 / FPS)
            r = pop.update(k, k / FPS, dets, cam)
            invariants(self, r)
            results.append(r)
            if not cut:
                if k > 0:
                    self.assertEqual(cam.status, "ESTIMATED")
                    self.assertAlmostEqual(cam.shift_px[0], -8.0, delta=0.5)
                ids_before |= {v.tracklet_id for v in r.tracklets}
        self.assertEqual(len(ids_before), 2, "a pan must not break tracklets of people standing still")
        rc = results[n_pan]
        self.assertTrue(rc.camera.continuity_broken)
        self.assertEqual(sorted(c.reason for c in rc.closed), [P.CAMERA_CUT, P.CAMERA_CUT])
        after = {v.tracklet_id for v in results[-1].tracklets}
        self.assertFalse(after & ids_before)


class TestLongRun(Base):
    def test_state_stays_bounded_over_a_long_run(self):
        """~3 minutes at 30 fps with people constantly entering, crossing, merging and leaving: live tracklets,
        remembered closed tracklets, merged regions and per-tracklet size histories stay bounded (no growth with
        match length)."""
        cfg = TrackerConfig.default()
        pop = P.Population(cfg, W, H)
        rng = np.random.default_rng(3)
        people = []
        peak = {"live": 0, "memory": 0, "regions": 0, "sizes": 0}
        n = int(180 * FPS)
        for k in range(n):
            if rng.random() < 0.05 and len(people) < 12:
                people.append([float(rng.uniform(0, W)), float(rng.uniform(200, 600)),
                               float(rng.uniform(-6, 6)), float(rng.uniform(60, 140))])
            dets = []
            for p_ in people:
                p_[0] += p_[2]
            people = [p_ for p_ in people if -50 < p_[0] < W + 50]
            for p_ in people:
                if rng.random() < 0.9:
                    dets.append(cand(box_at(p_[0], p_[1], p_[3] * 0.4, p_[3])))
            cam = identity_motion("FIRST_FRAME") if k == 0 else still()
            r = pop.update(k, k / FPS, dets, cam)
            if k % 97 == 0:
                invariants(self, r)
            peak["live"] = max(peak["live"], len(pop._live))
            peak["memory"] = max(peak["memory"], len(pop._memory))
            peak["regions"] = max(peak["regions"], len(pop._regions))
            peak["sizes"] = max([peak["sizes"]] + [len(tr.sizes) for tr in pop._live])
        window_frames = cfg.get("population_size_ref_window_s") * FPS
        self.assertLessEqual(peak["sizes"], window_frames + 2)
        self.assertLess(peak["live"], 60)
        self.assertLess(peak["memory"], 200)
        self.assertLess(peak["regions"], 30)


class TestRecords(unittest.TestCase):
    def test_records_are_json_safe(self):
        rn = run(crossing(40, FPS, speed_h_per_s=3.0, dy_h=0.0, merge_iou=0.3))
        for r in rn.results:
            json.dumps(r.as_record())
        comp = rn.population.competitors(box_at(400, 360, 32, 80), 80.0)
        for c in comp:
            json.dumps(c.as_record())

    def test_no_identity_fields(self):
        rn = run(walker(3, FPS))
        rec = rn.results[-1].tracklets[0].as_record()
        for k in ("team", "number", "colour", "identity", "role", "target"):
            self.assertFalse(any(k in key for key in rec), k)


if __name__ == "__main__":
    unittest.main()

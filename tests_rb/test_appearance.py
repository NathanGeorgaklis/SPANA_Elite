# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R1-D instruction and acceptance criteria (Nathan, 2026-10-24); R1/R1_GENERIC_TRACKER_REQUIREMENT.md
#                 (neutral identities TARGET, PERSON_A, PERSON_B, PERSON_C); R1/R1_DESIGN_DECISIONS.md D04, D12;
#                 R1-D independent code review and test audit findings (R1/R1D/R1D_STAGE_RECORD.md)
"""Tests for the generic appearance layer: crop quality, LAB cue, embedding cue, similarity bounds, identity support
(neither cue alone), fail-closed behaviour on non-finite inputs, and the trusted appearance bank.

DINOv2 backend tests need the pinned model file in models/. Set SPANA_REQUIRE_MODELS=1 to turn a missing model into a
test failure instead of a skip (so the model-pinning criteria cannot pass silently)."""
import hashlib
import math
import os
import shutil
import tempfile
import unittest

import numpy as np

from player_tracker_rb import appearance as A
from player_tracker_rb import embedding_dinov2 as E
from player_tracker_rb.config import TrackerConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMB_MODEL = os.path.join(ROOT, "models", "dinov2_vits14_cls_patchmean_224x112.onnx")
HAVE_EMB = os.path.isfile(EMB_MODEL)
REQUIRE_MODELS = os.environ.get("SPANA_REQUIRE_MODELS") == "1"
TOL = 1e-6


def cfg(**over):
    return TrackerConfig.with_overrides(over)


def person(shirt, shorts, h=96, w=48, bg=(40, 140, 40), seed=0, noise=6):
    """A synthetic BGR person crop: background sides, a shirt band and a shorts band (plus mild noise)."""
    rng = np.random.default_rng(seed)
    img = np.empty((h, w, 3), np.uint8)
    img[:] = bg
    x1, x2 = int(w * 0.2), int(w * 0.8)            # body spans the central 60% (the LAB sampling width)
    img[int(h * 0.12):int(h * 0.5), x1:x2] = shirt
    img[int(h * 0.5):int(h * 0.8), x1:x2] = shorts
    if noise:
        img = np.clip(img.astype(int) + rng.integers(-noise, noise + 1, img.shape), 0, 255).astype(np.uint8)
    return img


RED, WHITE, BLUE, BLACK = (40, 40, 200), (230, 230, 230), (200, 60, 30), (20, 20, 20)
TARGET = person(RED, WHITE, seed=1)
PERSON_A = person(RED, WHITE, seed=2)          # same kit as TARGET, a different physical person
PERSON_B = person(BLUE, BLACK, seed=3)         # different kit
GOOD_Q = A.CropQuality(96.0, 1.0, 0.0, 1.0, True, ())


def feat(crop, vec, quality=None):
    return A.AppearanceFeature(lab=A.lab_descriptor(crop, cfg()), embedding=A.l2_normalise(np.asarray(vec, float)),
                               quality=quality or GOOD_Q)


def sim(lab, emb, c=None):
    c = c or cfg()
    return A.AppearanceSimilarity(lab=lab, embedding=emb,
                                  composite=c.get("composite_weight_lab") * lab + c.get("composite_weight_embedding") * emb)


class TestCropQuality(unittest.TestCase):
    def test_formula(self):
        c = cfg()
        good = c.get("crop_good_height_px")
        q = A.assess_crop((100, 100, 140, 100 + 1.5 * good), 1280, 720, c, others=[])
        self.assertEqual((q.quality, q.valid, q.flags), (1.0, True, ()))
        q = A.assess_crop((100, 100, 120, 100 + good / 2), 1280, 720, c, others=[])
        self.assertAlmostEqual(q.quality, 0.5)

    def test_others_must_be_explicit(self):
        with self.assertRaises(TypeError):
            A.assess_crop((100, 100, 140, 196), 1280, 720, cfg())

    def test_low_resolution_invalid(self):
        q = A.assess_crop((100, 100, 110, 100 + cfg().get("crop_min_height_px") - 1), 1280, 720, cfg(), others=[])
        self.assertFalse(q.valid)
        self.assertIn(A.LOW_RESOLUTION, q.flags)

    def test_mostly_outside_invalid_and_edge_flag(self):
        q = A.assess_crop((1260, 600, 1340, 700), 1280, 720, cfg(), others=[])     # 25% inside
        self.assertFalse(q.valid)
        self.assertIn(A.MOSTLY_OUTSIDE, q.flags)
        self.assertIn(A.TOUCHES_IMAGE_EDGE, q.flags)

    def test_occluded_by_other_person(self):
        q = A.assess_crop((100, 100, 140, 196), 1280, 720, cfg(), others=[(120, 100, 180, 196)])
        self.assertAlmostEqual(q.covered_fraction, 0.5)
        self.assertAlmostEqual(q.quality, 0.5)
        self.assertIn(A.OCCLUDED_BY_OTHER, q.flags)
        self.assertTrue(q.valid)                                           # usable for comparison, flagged

    def test_covered_fraction_is_union(self):
        self.assertAlmostEqual(A.covered_fraction((0, 0, 100, 100), [(0, 0, 50, 100), (25, 0, 75, 100)]), 0.75)

    def test_covered_at_limit_not_flagged(self):
        c = cfg()
        lim = c.get("crop_max_covered_fraction")
        q = A.assess_crop((100, 100, 200, 200), 1280, 720, c, others=[(100, 100, 100 + 100 * lim, 200)])
        self.assertAlmostEqual(q.covered_fraction, lim)
        self.assertNotIn(A.OCCLUDED_BY_OTHER, q.flags)

    def test_non_finite_box_refused(self):
        for box in ((float("nan"), 0, 10, 10), (0, 0, float("inf"), 10)):
            with self.assertRaises(A.AppearanceError):
                A.assess_crop(box, 1280, 720, cfg(), others=[])
            with self.assertRaises(A.AppearanceError):
                A.crop_image(np.zeros((720, 1280, 3), np.uint8), box)


class TestLab(unittest.TestCase):
    def test_shape_and_normalisation(self):
        d = A.lab_descriptor(TARGET, cfg())
        b = cfg().get("lab_bins_per_channel")
        self.assertEqual(d.shape, (2, b ** 3))
        np.testing.assert_allclose(d.sum(axis=1), [1.0, 1.0])

    def test_bounds_and_identity(self):
        c = cfg()
        self.assertAlmostEqual(A.lab_similarity(A.lab_descriptor(TARGET, c), A.lab_descriptor(TARGET, c)), 1.0)
        for x in (PERSON_A, PERSON_B):
            s = A.lab_similarity(A.lab_descriptor(TARGET, c), A.lab_descriptor(x, c))
            self.assertTrue(0.0 <= s <= 1.0)

    def test_same_kit_high_different_kit_low(self):
        c = cfg()
        t = A.lab_descriptor(TARGET, c)
        self.assertGreater(A.lab_similarity(t, A.lab_descriptor(PERSON_A, c)), 0.8)    # colour cannot separate
        self.assertLess(A.lab_similarity(t, A.lab_descriptor(PERSON_B, c)), c.get("lab_floor"))   # colour rejects

    def test_background_sides_ignored(self):
        c = cfg()
        a = person(RED, WHITE, bg=(40, 140, 40))
        b = person(RED, WHITE, bg=(200, 200, 30))                          # only the side background differs
        self.assertGreater(A.lab_similarity(A.lab_descriptor(a, c), A.lab_descriptor(b, c)), 0.9)

    def test_neutral_kits_robust_to_small_tint(self):
        # Neutral colours sit mid-range in a/b; a 2-level white-balance shift must not flip them to "incompatible".
        c = cfg()
        for base in ((230, 230, 230), (128, 128, 128), (20, 20, 20)):
            for tint in ((2, 0, -2), (-2, 0, 2), (0, 2, 0), (2, 2, 2)):
                with self.subTest(base=base, tint=tint):
                    a = person(base, base, noise=0)
                    b = person(tuple(int(np.clip(v + t, 0, 255)) for v, t in zip(base, tint)),
                               tuple(int(np.clip(v + t, 0, 255)) for v, t in zip(base, tint)), noise=0)
                    s = A.lab_similarity(A.lab_descriptor(a, c), A.lab_descriptor(b, c))
                    self.assertGreaterEqual(s, 0.8)

    def test_too_few_pixels_gives_no_descriptor(self):
        c = cfg()
        tiny = person(RED, WHITE, h=24, w=10)
        self.assertIsNone(A.lab_descriptor(tiny, c))

    def test_missing_and_bad_descriptor(self):
        self.assertIsNone(A.lab_similarity(None, A.lab_descriptor(TARGET, cfg())))
        self.assertIsNone(A.lab_descriptor(np.zeros((0, 0, 3), np.uint8), cfg()))
        d = A.lab_descriptor(TARGET, cfg())
        with self.assertRaises(A.AppearanceError):
            A.lab_similarity(d, np.full_like(d, np.nan))


class TestEmbeddingSimilarity(unittest.TestCase):
    def test_bounds(self):
        u = A.l2_normalise([1, 0, 0])
        self.assertEqual(A.embedding_similarity(u, u, TOL), 1.0)
        self.assertEqual(A.embedding_similarity(u, A.l2_normalise([-1, 0, 0]), TOL), 0.0)   # negative cosine -> 0
        self.assertAlmostEqual(A.embedding_similarity(u, A.l2_normalise([1, 1, 0]), TOL), math.sqrt(0.5))
        self.assertIsNone(A.embedding_similarity(None, u, TOL))

    def test_bad_vectors_raise(self):
        u = A.l2_normalise([1, 0])
        for a, b in ((u, A.l2_normalise([1, 0, 0])), (np.array([2.0, 0.0]), np.array([2.0, 0.0])),
                     (np.array([np.nan, 1.0]), u)):
            with self.subTest(a=a, b=b):
                with self.assertRaises(A.AppearanceError):
                    A.embedding_similarity(a, b, TOL)
        for bad in ([0, 0, 0], [np.nan, 1], [np.inf, 0]):
            with self.assertRaises(A.AppearanceError):
                A.l2_normalise(bad)


class TestIdentitySupport(unittest.TestCase):
    def test_composite_formula(self):
        c = cfg()
        s = A.compare(feat(TARGET, [1, 0]), feat(TARGET, [1, 0]), c)
        self.assertAlmostEqual(s.composite, c.get("composite_weight_lab") * s.lab
                               + c.get("composite_weight_embedding") * s.embedding)
        self.assertTrue(0.0 <= s.composite <= 1.0)

    def test_colour_alone_cannot_establish_identity(self):
        # Identical colour and a composite above its floor: only the embedding floor stands in the way.
        c = cfg()
        s = sim(1.0, math.nextafter(c.get("embedding_floor"), 0.0), c)
        self.assertGreaterEqual(s.composite, c.get("appearance_floor"))     # precondition: composite would pass
        ok, why = A.identity_support(s, c)
        self.assertFalse(ok)
        self.assertEqual(why, ("EMBEDDING_BELOW_FLOOR",))

    def test_embedding_alone_cannot_establish_identity(self):
        c = cfg()
        s = sim(math.nextafter(c.get("lab_floor"), 0.0), 1.0, c)
        ok, why = A.identity_support(s, c)
        self.assertFalse(ok)
        self.assertIn("LAB_INCOMPATIBLE", why)

    def test_exact_floors_supported(self):
        c = cfg()
        lab, emb = 1.0, c.get("embedding_floor")
        s = sim(lab, emb, c)
        if s.composite < c.get("appearance_floor"):
            self.skipTest("floors chosen so that the composite floor binds first")
        self.assertTrue(A.identity_support(s, c)[0])
        s2 = A.AppearanceSimilarity(lab=c.get("lab_floor"), embedding=1.0, composite=c.get("appearance_floor"))
        self.assertTrue(A.identity_support(s2, c)[0])
        s3 = A.AppearanceSimilarity(lab=1.0, embedding=1.0, composite=math.nextafter(c.get("appearance_floor"), 0.0))
        self.assertEqual(A.identity_support(s3, c)[1], ("COMPOSITE_BELOW_FLOOR",))

    def test_same_kit_different_person_not_supported(self):
        c = cfg()
        s = A.compare(feat(TARGET, [1, 0, 0]), feat(PERSON_A, [0.2, 1, 0]), c)
        self.assertGreater(s.lab, 0.8)
        ok, why = A.identity_support(s, c)
        self.assertFalse(ok)
        self.assertIn("EMBEDDING_BELOW_FLOOR", why)

    def test_missing_or_non_finite_never_supports(self):
        c = cfg()
        nan = float("nan")
        for s in (A.AppearanceSimilarity(nan, nan, nan), A.AppearanceSimilarity(1.0, nan, 1.0),
                  A.AppearanceSimilarity(nan, 1.0, 1.0), A.AppearanceSimilarity(1.0, 1.0, nan),
                  A.AppearanceSimilarity(1.0, None, None), A.AppearanceSimilarity(None, 1.0, None)):
            with self.subTest(s=s):
                self.assertFalse(A.identity_support(s, c)[0])

    def test_contaminated_crop_never_supports(self):
        c = cfg()
        for flag in A.CONTAMINATION_FLAGS:
            ok, why = A.identity_support(sim(1.0, 1.0, c), c, flags=(flag,))
            self.assertFalse(ok)
            self.assertIn(f"CROP_{flag}", why)

    def test_both_cues_agree_supported(self):
        c = cfg()
        ok, _ = A.identity_support(A.compare(feat(TARGET, [1, 0.05]), feat(PERSON_A, [1, 0.0]), c), c)
        self.assertTrue(ok)        # appearance is CONSISTENT; contrast against competitors is decided later


class TestExtractor(unittest.TestCase):
    def setUp(self):
        self.img = np.zeros((720, 1280, 3), np.uint8)
        self.img[100:196, 100:148] = TARGET

    def test_invalid_crop_gives_no_features(self):
        f = A.AppearanceExtractor(cfg(), A.ScriptedEmbedding([[1, 0]])).extract(self.img, (10, 10, 18, 25), others=[])
        self.assertFalse(f.usable)
        self.assertIsNone(f.lab)
        self.assertIsNone(f.embedding)

    def test_no_embedder_flagged_not_usable(self):
        f = A.AppearanceExtractor(cfg(), None).extract(self.img, (100, 100, 148, 196), others=[])
        self.assertIn(A.EMBEDDING_UNAVAILABLE, f.quality.flags)
        self.assertFalse(f.usable)

    def test_valid_crop(self):
        f = A.AppearanceExtractor(cfg(), A.ScriptedEmbedding([[1, 0]])).extract(self.img, (100, 100, 148, 196),
                                                                                 others=[])
        self.assertTrue(f.usable)

    def test_lab_missing_not_usable(self):
        f = A.AppearanceFeature(lab=None, embedding=A.l2_normalise([1, 0]), quality=GOOD_Q)
        self.assertFalse(f.usable)

    def test_broken_backend_fails_closed(self):
        class Broken(A.EmbeddingBackend):
            def embed(self, crop):
                return np.array([2.0, 0.0])           # not unit length
            def provenance(self):
                return {}
        with self.assertRaises(A.AppearanceError):
            A.AppearanceExtractor(cfg(), Broken()).extract(self.img, (100, 100, 148, 196), others=[])


# Embedding vectors for bank tests: similar enough to pass the bank's self-consistency check, distinct enough to be
# novel. ANCHOR-B and ANCHOR-C cosines are ~0.89; D is close to B (cos ~0.94) but not redundant (< 0.98).
V_ANCHOR, V_B, V_C, V_D = [1, 0, 0, 0], [1, 0.5, 0, 0], [1, 0, 0.5, 0], [1, 0.45, 0.25, 0]


class TestBank(unittest.TestCase):
    CTX = A.BankUpdateContext(identity_trusted=True)

    def test_trusted_observation_accepted(self):
        bank = A.AppearanceBank(cfg())
        self.assertEqual(bank.add(feat(TARGET, V_ANCHOR), self.CTX)[0], A.ACCEPTED)
        self.assertEqual(len(bank), 1)

    def test_context_refusals(self):
        c = cfg()
        cases = [
            (A.BankUpdateContext(identity_trusted=False), "IDENTITY_NOT_TRUSTED"),
            (A.BankUpdateContext(identity_trusted=True, merged_suspected=True), A.MERGED_SUSPECTED),
            (A.BankUpdateContext(identity_trusted=True, max_competitor_iou=math.nextafter(c.get("bank_contact_iou"), 1)),
             "CONTESTED_BY_OTHER_PERSON"),
            (A.BankUpdateContext(identity_trusted=True, max_competitor_iou=float("nan")), "CONTESTED_BY_OTHER_PERSON"),
            (A.BankUpdateContext(identity_trusted=True,
                                 seconds_since_reacquisition=math.nextafter(c.get("post_reacq_quarantine_s"), 0)),
             "POST_REACQUISITION_QUARANTINE"),
            (A.BankUpdateContext(identity_trusted=True, seconds_since_reacquisition=float("nan")),
             "POST_REACQUISITION_QUARANTINE"),
            (A.BankUpdateContext(identity_trusted=True, seconds_since_reacquisition=-1.0),
             "POST_REACQUISITION_QUARANTINE"),
        ]
        for ctx, reason in cases:
            with self.subTest(reason=reason, ctx=ctx):
                bank = A.AppearanceBank(c)
                status, why = bank.add(feat(TARGET, V_ANCHOR), ctx)
                self.assertEqual(status, A.REFUSED)
                self.assertIn(reason, why)
                self.assertEqual(len(bank), 0)

    def test_context_boundaries_accepted(self):
        c = cfg()
        for ctx in (A.BankUpdateContext(identity_trusted=True, max_competitor_iou=c.get("bank_contact_iou")),
                    A.BankUpdateContext(identity_trusted=True,
                                        seconds_since_reacquisition=c.get("post_reacq_quarantine_s"))):
            with self.subTest(ctx=ctx):
                self.assertEqual(A.AppearanceBank(c).add(feat(TARGET, V_ANCHOR), ctx)[0], A.ACCEPTED)

    def test_poor_or_partial_crops_refused(self):
        c = cfg()
        cases = [
            (A.CropQuality(30.0, 1.0, 0.0, 0.47, True, ()), "LOW_QUALITY"),
            (A.CropQuality(96.0, 1.0, 0.4, 0.6, True, (A.OCCLUDED_BY_OTHER,)), A.OCCLUDED_BY_OTHER),
            (A.CropQuality(96.0, 1.0, 0.2, 0.8, True, ()), "PARTLY_COVERED_BY_OTHER"),
            (A.CropQuality(96.0, 1.0, 0.0, 1.0, True, (A.MERGED_SUSPECTED,)), A.MERGED_SUSPECTED),
            (A.CropQuality(96.0, 1.0, 0.0, 1.0, True, (A.TOUCHES_IMAGE_EDGE,)), A.TOUCHES_IMAGE_EDGE),
            (A.CropQuality(10.0, 1.0, 0.0, 0.15, False, (A.LOW_RESOLUTION,)), "FEATURE_NOT_USABLE"),
            (A.CropQuality(96.0, 1.0, 0.0, float("nan"), True, ()), "NON_FINITE_INPUT"),
        ]
        for q, reason in cases:
            with self.subTest(reason):
                bank = A.AppearanceBank(c)
                st, why = bank.add(feat(TARGET, V_ANCHOR, quality=q), self.CTX)
                self.assertEqual(st, A.REFUSED)
                self.assertIn(reason, why)
                self.assertEqual(len(bank), 0)

    def test_partial_body_at_image_edge_refused_end_to_end(self):
        img = np.zeros((720, 1280, 3), np.uint8)
        img[624:720, 100:148] = TARGET
        f = A.AppearanceExtractor(cfg(), A.ScriptedEmbedding([V_ANCHOR])).extract(img, (100, 640, 148, 720), others=[])
        st, why = A.AppearanceBank(cfg()).add(f, self.CTX)
        self.assertEqual(st, A.REFUSED)
        self.assertIn(A.TOUCHES_IMAGE_EDGE, why)

    def test_inconsistent_sample_refused(self):
        # A "trusted" offer that does not look like the bank (other kit) is refused: no silent contamination.
        bank = A.AppearanceBank(cfg())
        bank.add(feat(TARGET, V_ANCHOR), self.CTX)
        st, why = bank.add(feat(PERSON_B, V_B), self.CTX)
        self.assertEqual(st, A.REFUSED)
        self.assertIn("BANK_SELF_INCONSISTENT", why)
        self.assertEqual(len(bank), 1)

    def test_redundant_sample_not_stored(self):
        bank = A.AppearanceBank(cfg())
        bank.add(feat(TARGET, V_ANCHOR), self.CTX)
        self.assertEqual(bank.add(feat(TARGET, [1, 0.001, 0, 0]), self.CTX)[0], A.REDUNDANT)
        self.assertEqual(len(bank), 1)

    def test_capacity_eviction_choice(self):
        bank = A.AppearanceBank(cfg(), capacity=3)
        for v in (V_ANCHOR, V_B, V_C):
            self.assertEqual(bank.add(feat(TARGET, v), self.CTX)[0], A.ACCEPTED)
        st, _ = bank.add(feat(TARGET, V_D), self.CTX)
        self.assertIn(st, (A.ACCEPTED, A.NOT_STORED_CAPACITY))
        stored = [tuple(np.round(s.embedding, 6)) for s in bank.samples]
        norm = lambda v: tuple(np.round(A.l2_normalise(v), 6))          # noqa: E731
        self.assertEqual(len(bank), 3)
        self.assertEqual(stored[0], norm(V_ANCHOR))                      # anchor kept
        self.assertIn(norm(V_C), stored)                                 # the distinct view kept
        self.assertEqual(sum(norm(v) in stored for v in (V_B, V_D)), 1)  # exactly one of the near-duplicates kept

    def test_capacity_one_reports_not_stored(self):
        bank = A.AppearanceBank(cfg(), capacity=1)
        bank.add(feat(TARGET, V_ANCHOR), self.CTX)
        self.assertEqual(bank.add(feat(TARGET, V_B), self.CTX)[0], A.NOT_STORED_CAPACITY)
        self.assertEqual(len(bank), 1)

    def test_invalid_capacity(self):
        for cap in (0, -1, 1.5, True):
            with self.assertRaises(A.AppearanceError):
                A.AppearanceBank(cfg(), capacity=cap)

    def test_best_match_ranks_by_composite_from_one_sample(self):
        c = cfg()
        bank = A.AppearanceBank(c)
        x = feat(TARGET, [0.8, 0.6, 0])                 # same colour, embedding cos 0.8 with the query
        y = feat(PERSON_B, [0.9, math.sqrt(1 - 0.81), 0])   # other colour, embedding cos 0.9
        bank.samples += [y, x]                          # direct placement: tests ranking only
        q = feat(TARGET, [1, 0, 0])
        s, idx = bank.best_match(q)
        self.assertEqual(idx, 1)                        # x wins on composite although y wins on embedding
        self.assertEqual(s, A.compare(q, x, c))         # cue values from that ONE sample
        self.assertEqual(bank.best_match(A.AppearanceFeature(None, None, GOOD_Q)), (None, None))

    def test_competitor_bank_needs_no_trust_but_still_quality(self):
        bank = A.AppearanceBank(cfg(), require_trust=False)
        self.assertEqual(bank.add(feat(TARGET, V_ANCHOR), A.BankUpdateContext(identity_trusted=False))[0], A.ACCEPTED)
        merged = A.CropQuality(96.0, 1.0, 0.0, 1.0, True, (A.MERGED_SUSPECTED,))
        self.assertEqual(bank.add(feat(TARGET, V_B, quality=merged),
                                  A.BankUpdateContext(identity_trusted=False))[0], A.REFUSED)


class TestEmbeddingPreprocessNoModel(unittest.TestCase):
    """Preprocessing and pooling are pure functions: tested without the model file."""

    def test_geometry_and_padding(self):
        c = cfg()
        x = E.preprocess_crop(np.full((96, 96, 3), 128, np.uint8), c)   # square crop into a 224 x 112 input
        hh, ww = c.get("embedding_input_hw")
        self.assertEqual(x.shape, (1, 3, hh, ww))
        self.assertEqual(x.dtype, np.float32)
        pad = (hh - ww) // 2
        self.assertTrue(np.allclose(x[0, :, :pad, :], 0.0, atol=0.01))  # padding = mean colour -> ~0 after norm
        self.assertTrue(np.allclose(x[0, :, -pad:, :], 0.0, atol=0.01))
        self.assertFalse(np.allclose(x[0, :, pad:hh - pad, :], 0.0, atol=0.01))

    def test_channel_order_and_normalisation(self):
        c = cfg()
        blue_bgr = np.zeros((224, 112, 3), np.uint8)
        blue_bgr[..., 0] = 255                                           # pure blue in BGR
        x = E.preprocess_crop(blue_bgr, c)
        mean, std = np.array(c.get("embedding_mean_rgb")), np.array(c.get("embedding_std_rgb"))
        want = (np.array([0.0, 0.0, 1.0]) - mean) / std                  # RGB order: blue is the last channel
        np.testing.assert_allclose(x[0, :, 112, 56], want, atol=1e-5)

    def test_extreme_shapes_and_bad_crops(self):
        c = cfg()
        for shape in ((20, 400, 3), (2000, 5, 3), (1, 1, 3)):
            with self.subTest(shape=shape):
                self.assertTrue(np.all(np.isfinite(E.preprocess_crop(np.full(shape, 90, np.uint8), c))))
        for bad in (np.zeros((0, 10, 3), np.uint8), np.zeros((10, 0, 3), np.uint8), np.zeros((10, 10), np.uint8),
                    np.zeros((10, 10, 3), np.float32), np.zeros((10, 10, 4), np.uint8), None):
            with self.subTest(bad=None if bad is None else bad.shape):
                with self.assertRaises(A.AppearanceError):
                    E.preprocess_crop(bad, c)

    def test_pooling(self):
        rng = np.random.default_rng(0)
        cls, pm = rng.normal(size=384), rng.normal(size=384)
        np.testing.assert_allclose(E.pool(cls, pm, "cls"), A.l2_normalise(cls))
        v = E.pool(cls, pm, "cls_patchmean")
        self.assertEqual(v.shape, (768,))
        np.testing.assert_allclose(v[:384], A.l2_normalise(cls) / math.sqrt(2))
        np.testing.assert_allclose(v[384:], A.l2_normalise(pm) / math.sqrt(2))
        with self.assertRaises(A.AppearanceError):
            E.pool(cls, pm, "mean")


@unittest.skipUnless(HAVE_EMB or REQUIRE_MODELS, "models/dinov2_vits14_cls_patchmean_224x112.onnx not present")
class TestDinov2Backend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not HAVE_EMB:
            raise AssertionError("SPANA_REQUIRE_MODELS=1 but the pinned embedding model is missing")
        cls.tmp = tempfile.mkdtemp(prefix="rb_emb_")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_pinned_hash(self):
        self.assertEqual(hashlib.sha256(open(EMB_MODEL, "rb").read()).hexdigest(),
                         cfg().get("embedding_model_sha256"))

    def test_refusals(self):
        with self.assertRaises(A.EmbeddingModelError):
            E.Dinov2OnnxEmbedding(cfg(embedding_model_sha256="0" * 64))
        with self.assertRaises(A.EmbeddingModelError):
            E.Dinov2OnnxEmbedding(cfg(embedding_model_file=os.path.join(self.tmp, "absent.onnx")))
        bad = os.path.join(self.tmp, "bad.onnx")
        with open(bad, "wb") as f:
            f.write(b"garbage" * 1000)
        h = hashlib.sha256(open(bad, "rb").read()).hexdigest()
        with self.assertRaises(A.EmbeddingModelError):
            E.Dinov2OnnxEmbedding(cfg(embedding_model_file=bad, embedding_model_sha256=h))

    def test_input_size_mismatch_refused(self):
        with self.assertRaises(A.EmbeddingModelError):
            E.Dinov2OnnxEmbedding(cfg(embedding_input_hw=(224, 224)))

    def test_deterministic_and_normalised(self):
        e = E.Dinov2OnnxEmbedding(cfg())
        a, b = e.embed(TARGET), e.embed(TARGET.copy())
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.shape, (e.dim,))
        self.assertAlmostEqual(float(np.linalg.norm(a)), 1.0, places=9)
        self.assertEqual(e.dim, 384)
        np.testing.assert_array_equal(a, E.Dinov2OnnxEmbedding(cfg()).embed(TARGET))   # fresh session, same vector

    def test_outputs_read_by_name_and_pooling(self):
        e = E.Dinov2OnnxEmbedding(cfg(embedding_pooling="cls_patchmean"))
        cls, pm = e.raw_outputs(TARGET)
        self.assertEqual(e.dim, 768)
        np.testing.assert_allclose(e.embed(TARGET), E.pool(cls, pm, "cls_patchmean"))
        self.assertFalse(np.allclose(A.l2_normalise(cls), A.l2_normalise(pm)))   # two genuinely different outputs

    def test_people_differ(self):
        e = E.Dinov2OnnxEmbedding(cfg())
        s_same = A.embedding_similarity(e.embed(TARGET), e.embed(TARGET), TOL)
        s_diff = A.embedding_similarity(e.embed(TARGET), e.embed(PERSON_B), TOL)
        self.assertAlmostEqual(s_same, 1.0, places=12)
        self.assertLess(s_diff, 1.0)

    def test_provenance_says_generic(self):
        p = E.Dinov2OnnxEmbedding(cfg()).provenance()
        self.assertIn("not a person/football re-identification model", p["note"])
        self.assertEqual(p["model_sha256"], cfg().get("embedding_model_sha256"))


if __name__ == "__main__":
    unittest.main()

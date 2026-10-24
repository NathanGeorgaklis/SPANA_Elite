# R1-D stage record: generic appearance backend

**Stage:** R1-D, approved by Nathan on 2026-10-24 at 19:39 Sydney. `R1/R1_GENERIC_TRACKER_REQUIREMENT.md` is binding.

**Built:**
- a tracker-owned LAB colour descriptor;
- a DINOv2 ViT-S/14 embedding behind a generic `EmbeddingBackend` interface;
- crop preprocessing and quality assessment;
- bounded similarity scores and the `identity_support` verdict;
- the trusted `AppearanceBank`.

**Not built:** no motion, population, target state machine, reacquisition, ball or event logic.

**Contracts:** the output contract (`output_schema.py`) is unchanged. The config registry is now `r1d-1`, with 121 parameters. The default config hash is `7aacbd2d18d745de55b149cb55708c57c529ecb566d9f36e5e5faa0423316b07`.

## Model

The full record, with every hash, is in `R1D_MODEL_RECORD.md`.
- **Licence, captured before download:** the repository README and LICENSE, and the Hugging Face model card, all state **Apache-2.0**, and they agree with the gate record.
- **Status:** **LICENCE_UNCLEAR**. Approved for development and evaluation; not PRODUCTION_APPROVED until Nathan signs off and the LVD-142M training-data question is resolved.
- **Source:** HF `facebook/dinov2-small` at commit `ed25f3a3…`. Its weights are **bit-identical** to Meta's official `dinov2_vits14_pretrain.pth`.
- **Export:** `tools_rb/export_models.py` produced ONNX file `0ab62713cf7759dd820b53576b39010cb0fe8dcfbe22af444c945c519bf9efea`. It is byte-reproducible, and differs from PyTorch by at most 1.7e-5.
- **Embedding:** 384 dimensions (CLS token). DINOv2 is a generic visual embedding, not football ReID.

## Methods

- **Crop quality**, range [0, 1]: `quality = inside_fraction × min(1, h / 64 px) × (1 − covered_fraction)`.
  - Flags: LOW_RESOLUTION, MOSTLY_OUTSIDE, TOUCHES_IMAGE_EDGE, OCCLUDED_BY_OTHER, MERGED_SUSPECTED.
  - `others` (the other people's boxes) must be passed explicitly; there is no silent default.
- **LAB cue:**
  - Two bands (upper 0.15–0.50 and lower 0.50–0.80 of the box height), using the central 60% of the width.
  - Each band is an 8×8×8 **soft (trilinear) binned** LAB histogram, L1-normalised, with at least 150 pixels per band.
  - Similarity is the mean Bhattacharyya coefficient over the two bands, in [0, 1].
- **Embedding cue:**
  - The crop is converted BGR → RGB and letterboxed into 224 × 112: INTER_AREA when shrinking, INTER_LINEAR when enlarging, padded with the normalisation-mean colour.
  - ImageNet mean/std normalisation is applied.
  - The CLS output is read by name and L2-normalised.
  - Similarity is max(0, cosine) of unit vectors, in [0, 1]. Non-unit or non-finite vectors raise.
- **Composite:** 0.35 × LAB + 0.65 × embedding, in [0, 1]. It is reported but never used alone.
- **identity_support** requires all of:
  - LAB ≥ 0.30 (colour only vetoes);
  - embedding ≥ **0.70**;
  - composite ≥ **0.80**;
  - finite values;
  - no MERGED or OCCLUDED crop flag.

  So neither cue alone can establish identity. Support is necessary but not sufficient: contrast against competitors is R1-F.
- **AppearanceBank** (target):
  - A sample is accepted only if it is usable, has quality ≥ 0.6, carries no OCCLUDED / MERGED / EDGE / EMBEDDING_UNAVAILABLE flag, and has covered_fraction ≤ 0.05.
  - The context must be trusted, uncontested (IoU ≤ 0.05) and outside the post-reacquisition quarantine.
  - The sample must be appearance-consistent with what the bank already holds.
  - Non-finite inputs are refused.
  - The anchor is never evicted, and eviction removes the most redundant sample, which keeps multiple views.
  - The status is honest: ACCEPTED, REDUNDANT, REFUSED or NOT_STORED_CAPACITY.

## Development-set appearance analysis

GOLD is used as appearance footage only. No tracker, identity-label or event result is used.

**Tool:** `tools_rb/appearance_eval.py` on the Mac (onnxruntime 1.23.2), using the R1-C saved person candidates (2×2 tiles).

**Pairs.** These are weak, physics-based labels, NOT human-verified:
- **same-person:** isolated frame-to-frame continuity chains, sampled at least 0.5 s apart;
- **different-person:** two separate bodies in the same frame;
- **LAB-confusable (same-kit proxy):** different pairs with LAB ≥ 0.6. The contact sheet `appearance_eval/*_lab_confusable_pairs.png` confirms these are mostly different players in the same kit.

Counts: 233 same, 220 different, 109 LAB-confusable. **Caveat:** the same-person pairs come from only 9 long chains, so few individuals.

| Cue | Same median (P10) | Different median (P90) | AUC same vs different | AUC same vs same-kit proxy |
|---|---|---|---|---|
| LAB | 0.970 (0.921) | 0.598 (0.795) | 0.991 | 0.982 |
| Embedding (CLS) | 0.860 (0.759) | 0.636 (0.776) | 0.961 | 0.939 |
| Composite | 0.895 (0.818) | 0.609 (0.768) | 0.989 | 0.977 |

**Pooling**, by the pre-declared rule: CLS (AUC 0.939 on the same-kit proxy) against CLS + patch-mean (0.934); within 0.01, so the simpler CLS.

**Floors.** The pre-declared rule takes the 5th percentile of same-person similarity, floored to 0.05: embedding P5 0.741 → **0.70**, composite P5 0.806 → **0.80**. The LAB floor stays at 0.30 (veto only).

**identity_support at these floors** (verification run, configured equals calibrated):

| Pair type | Supported |
|---|---|
| Same person | **95.7%** |
| Different person | **7.3%** |
| Same-kit proxy different person | **14.7%** |

### Negative result (recorded, project rule 12)

- **Hypothesis:** placeholder floors (embedding 0.5, composite 0.55) are good enough.
- **Observed:** at those floors, appearance supported **76% of same-kit different-person pairs** (Mac pre-calibration run), because DINOv2 similarities between different people sit around 0.6–0.7.
- **Decision:** the floors must be calibrated to the embedding's scale (done). The residual 14.7% is the reason competitor contrast (R1-F) is mandatory. Appearance alone **cannot** be the identity decision.

## Independent review (project subagent policy)

- **Code reviewer:** 14 findings.
- **Test auditor:** 12 findings, including 5 demonstrated mutants that passed the original tests.

All were validated by the main agent. The genuine ones were fixed as general rules, and each now has a test:

| # | Finding | Fix |
|---|---|---|
| 1 | Body cut by the image border entered the bank | Bank refuses TOUCHES_IMAGE_EDGE (end-to-end test) |
| 2 | A large occluder passed the IoU-only contact test at 30% coverage | `bank_max_covered_fraction` 0.05 on covered_fraction; `others` required |
| 3 | No self-consistency check on new bank samples | BANK_SELF_INCONSISTENT refusal |
| 4 | Non-finite values supported identity or entered the bank | Fail closed everywhere (tests) |
| 5 | Colour-alone acceptance not truly tested (mutant passed) | Test with LAB = 1, composite above floor, embedding just below floor: only EMBEDDING_BELOW_FLOOR |
| 6 | Neutral kits flipped to LAB 0 after a 2-level tint (hard bin edges at a = b = 128) | Soft trilinear binning; test over white, grey and black with tints |
| 7 | Sparse histograms on tiny crops | `lab_min_band_pixels` 150 |
| 8 | ONNX outputs read by position | Read by name; fixed-shape, float32 and dimension checks |
| 9 | Self-evicted sample reported ACCEPTED; capacity 0 emptied the bank | NOT_STORED_CAPACITY; capacity ≥ 1 enforced |
| 10 | BGR/RGB order, pooling, eviction choice and best-match ranking untested (mutants passed) | Direct tests; preprocessing is model-free and runs everywhere |
| 11 | Contaminated crops (merged or occluded) could still get appearance support | `identity_support(..., flags)` refuses |
| 12 | Embedding norm unchecked | Unit length enforced (`embedding_norm_tolerance` 1e-6) |
| 13 | Shrinking used INTER_LINEAR (aliasing) | INTER_AREA when shrinking |
| 14 | Model skips could hide pinning criteria | `SPANA_REQUIRE_MODELS=1` turns a skip into a failure |

**Not changed, recorded as risks:**
- Redundancy eviction is kept, to preserve multiple views; outliers are blocked at entry instead.
- `best_match` over a full bank raises the chance of a lucky match (multiple comparisons). It is unmeasured, and is for R1-F/R2.
- The floors are in-sample on one clip (UNVALIDATED).

## Tests

- **Mac:** **188 tests OK, 0 skipped**, with `SPANA_REQUIRE_MODELS=1` and `SPANA_GOLD_VIDEO` set (Python 3.10.12, onnxruntime 1.23.2, cv2 5.0.0). The output is in `R1D_TEST_OUTPUT_mac.txt`.
- **Cloud:** 188 OK.
- **Regressions:** none. All R1-A, R1-B and R1-C tests pass.
  - One R1-C test changed: onnxruntime imports are now allowed in `embedding_dinov2.py` as well as `detector_yolox.py`.
  - The YOLOX tests also honour `SPANA_REQUIRE_MODELS`.
- **Config consequence:** `reacq_appearance_floor` was raised from 0.62 to 0.85, because the config cross-check requires reacquisition to stay stricter than `appearance_floor`. To be revisited in R1-F.

## Determinism

- **Same environment:** identical vectors from repeated calls and from fresh sessions (tested).
- **Across machines** (Mac onnxruntime 1.23.2 against cloud 1.23.2 or 1.29.0): vectors agree to about 1e-6 but are not bit-identical, because of different CPU kernels and OpenCV builds.

## Known limits

- **Weak labels:** few individuals, one clip, not human-verified.
- **Floors calibrated in-sample:** UNVALIDATED until R2 human-labelled data.
- **Same-kit residual:** 14.7% of same-kit different pairs still receive appearance support, so R1-F contrast is required.
- **Front/back views, long gaps and lighting changes** are not specifically measured. The chains cover at most about 2.5 s.
- **DINOv2 on small crops** (40–80 px tall) is upsampled, and its reliability there is unmeasured.
- **Performance:** about 40–130 ms per crop on one CPU thread.

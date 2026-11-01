# R2 DEV: football/player ReID evaluation (offline, diagnostic; 2026-11-01)

**Instruction (Nathan, 2026-11-01 18:57):**
- Determine whether a stronger person-ReID model can safely improve reacquisition and ambiguous identity cases.
- Focus on C006, C020, C033, SYO C012 and the F3b reproducers. DEV only.
- Do not replace DINOv2. Evaluate the candidates offline. Prioritise identity separation over raw similarity.
- Record licence and provenance. No tracker change until a model clearly beats the current appearance evidence.

**EVAL was not run, opened or scored. The tracker is unchanged.** No frozen label, fixture, holdout or Task 3R material
was touched. The ReID models ran in a separate environment (`venv_reid`); the production DEV environment was not
modified.

## Method (`tools_rb/r2_reid_eval.py`, `tools_rb/r2_reid_safe_point.py`)

**Frames.** Every frozen-truth DEV frame where the target is available and boxed: 635 true-target query crops over the
four clips.

**Crops.**
- TRUE: the detector box matching the human box.
- COMPETITORS: every other body within 3 target heights. That gives 5,816 crops, of which 3,220 are SAME-KIT.
- SAME-KIT: LAB colour similarity to the gallery ≥ the 10th percentile of the target's own LAB similarities. This
  was fixed before any ReID result was seen.

**Gallery.** True-target crops from the first 1.0 s (at most 5) stand in for the appearance bank. A query's
similarity is the maximum cosine to the gallery.

**Backends compared:**
- LAB part colour (production);
- DINOv2 ViT-S/14 (production ONNX);
- the production COMPOSITE (0.35 LAB + 0.65 DINOv2);
- three person-ReID candidates;
- the candidate composites 0.35 LAB + 0.65 ReID, where a ReID model would replace DINOv2.

**Safety metrics, all scale-free:**
- TPR at a same-kit false-match rate ≤ 5 % / ≤ 1 %, using one global threshold per backend: how many true-target
  crops a safe threshold still keeps;
- per-clip TPR at that same threshold;
- same-kit rank-1: the true target beats the hardest same-kit competitor in the frame;
- AUC, and same-kit FMR at 90 % TPR.

## Candidate models and provenance

| Backend | Source / revision | SHA-256 | Code licence | Training data (weights) | Production status |
|---|---|---|---|---|---|
| DINOv2 ViT-S/14 (current) | facebookresearch/dinov2; our ONNX export (`models/dinov2_EXPORT_RECORD.json`) | as pinned in config | Apache-2.0 | LVD-142M | as recorded in R1 (production candidate) |
| OSNet-AIN x1.0, MSMT17 | huggingface.co/kaiyangzhou/osnet @ `a5c5cc03` (official author repository); model definition from `torchreid` 0.2.5 | `8a07e8da…919f` | MIT | **MSMT17** | **DEVELOPMENT_ONLY**: MSMT17 is distributed for non-commercial research [general knowledge, not re-verified] |
| Youtu ReID (2021-11) | huggingface.co/opencv/person_reid_youtureid @ `b7bb1eda` (OpenCV Zoo; origin ReID-Team/ReID_extra_testdata) | `05796833…580d` | Apache-2.0 (Zoo directory) | **UNKNOWN** (not stated on the card) | **LICENCE_UNCLEAR** (training data unknown) |
| "football-osnet-reid" OSNet x1.0 | huggingface.co/rudrasinghm/football-osnet-reid @ `f2999390` | `f66346ff…1351` | none stated | **Market-1501, not football**: the classifier has 751 classes (Market-1501's identity count) and the checkpoint records rank-1 0.895 | **NOT_APPROVED** (no licence or card; misleadingly named; diagnostic only) |

**Loading.**
- Checkpoints were loaded with `torch.load(weights_only=True)`. The upload also needed numpy dtype types allowed;
  no arbitrary code was executed.
- PersonViT (Hugging Face) was **not** tried: it needs remote code to be executed.

**Records.**
- Hashes: `MODEL_SHA256SUMS.txt`; model definitions: `TORCHREID_MODEL_DEF_SHA256.txt`.
- Model cards: `model_card_*.txt`. Environment: `ENVIRONMENT_venv_reid_pip_freeze.txt` (torch 2.15.0+cpu).
- The model files themselves are not committed (size, licences).

## Results (all four clips pooled; `REID_EVAL.json`, `SAFE_POINT.json`)

| Backend | TPR @ same-kit FMR 5 % | TPR @ same-kit FMR 1 % | Same-kit rank-1 (593 frames) | AUC | Same-kit FMR @ TPR 90 % |
|---|---|---|---|---|---|
| **COMPOSITE (current)** | **0.400** | **0.312** | 0.504 | 0.817 | 0.846 |
| DINOv2 alone (current embedding) | 0.362 | 0.277 | 0.528 | 0.803 | 0.830 |
| LAB alone | 0.468 | 0.343 | 0.543 | 0.851 | (1.0, by the same-kit definition) |
| OSNet-AIN MSMT17 | 0.353 | 0.313 | 0.470 | 0.788 | 0.877 |
| Youtu ReID | 0.346 | 0.224 | 0.535 | 0.854 | 0.642 |
| OSNet Market-1501 upload | 0.239 | 0.184 | 0.405 | 0.842 | 0.623 |
| LAB + OSNet-AIN | 0.383 | 0.321 | 0.514 | — | — |
| LAB + Youtu | 0.394 | 0.257 | 0.550 | — | — |

**Per clip (TPR at the global same-kit-FMR-5 % threshold):**

| | C006 | C020 | C033 | SYO C012 |
|---|---|---|---|---|
| COMPOSITE | 0.26 | 0.99 | 0.44 | 0.19 |
| DINOv2 | 0.13 | 0.88 | 0.42 | 0.24 |
| OSNet-AIN | 0.08 | 1.00 | 0.43 | 0.18 |
| Youtu | 0.19 | 0.82 | 0.46 | 0.17 |
| LAB + Youtu | 0.27 | 0.97 | 0.48 | 0.16 |

**Ranking quality (AUC / rank-1) per clip:**

| Backend | C006 | C020 | C033 | SYO C012 |
|---|---|---|---|---|
| COMPOSITE | 0.76 / 0.50 | 1.00 / 1.00 | 0.81 / 0.94 | 0.76 / 0.15 |
| Youtu | 0.78 / 0.54 | 1.00 / 1.00 | 0.85 / 0.96 | 0.82 / 0.19 |
| OSNet-AIN | 0.57 / 0.29 | 1.00 / 1.00 | 0.86 / 0.93 | 0.77 / 0.21 |

**Post-gap (recovery after a pose/view change).** 45 true and 16 competitor crops; the ReID models rank-1 1.0
against 0.83. The sample is too small to support any conclusion.

## Findings

1. **No candidate materially improves safe identity separation.** At the operating points that matter for
   fail-closed identity (same-kit FMR ≤ 5 % / 1 %):
   - the current composite keeps more true-target crops (0.400 / 0.312) than any ReID model alone (≤ 0.353 / 0.313);
   - it also beats any LAB + ReID composite at 5 % (≤ 0.394);
   - no backend is better on every clip.
2. **Youtu ReID ranks better overall but is worse in the safe tail.**
   - AUC 0.854 against 0.817, and same-kit FMR at 90 % TPR 0.64 against 0.85.
   - But it keeps fewer true crops once same-kit false matches must be rare (0.346 / 0.224), and it is not better on
     C020 or SYO C012.
   - Better average ranking is not safer identity.
3. **Same-kit crowds defeat every backend.** In SYO C012 the true target beats the hardest same-kit competitor on
   only 11–26 % of frames for every model, including DINOv2. The SYO C012 abstention (F4) is an evidence limit no
   tested model removes.
4. **C006 and C033 are not resolved.** All backends keep ≤ 0.48 of true crops at the safe threshold. C020 is already
   separable with the current evidence.
5. **F3b reproducers cannot be evaluated here.** They are synthetic (`SignatureEmbedding` vectors, no real pixels).
   Their failure is population geometry (part boxes), not appearance discrimination.
6. **Provenance.** The only well-sourced ReID weights tested (OSNet-AIN) are trained on MSMT17 (non-commercial
   research data). Youtu's training data is unknown. The "football" model is a misnamed Market-1501 checkpoint. None
   is production-cleared.

## Limits of this evaluation

- Two broadcasts and four clips; 635 true crops, 82 % of them from SYO C012.
- The gallery is the first second only (the production bank grows).
- Crops are detector boxes, so occluded and contact crops are included.
- SAME-KIT is defined by LAB colour, which makes LAB's own same-kit FMR uninformative.
- The post-gap subset is too small.
- No football-specific ReID weights with known provenance were found on the reachable model sources. A SoccerNet-ReID
  trained model remains untested.

## Decision

**RECOMMEND INTEGRATION: NO.** No model clearly beats the current appearance evidence on safe identity separation.
Per the instruction, the tracker is unchanged.

**Next options (Nathan's decision):**
1. A football-trained ReID model (for example SoccerNet-ReID), with licensable weights and documented training data.
   Re-run this exact harness, with its thresholds fixed in advance.
2. Better evidence rather than a better embedding: a longer and more view-diverse bank, jersey-number OCR, or temporal
   aggregation.
3. Accept abstention in same-kit crowds.

**Raw:** `rows/REID_EVAL_rows.tar.gz` (sha256 in `rows/SHA256SUMS.txt`).

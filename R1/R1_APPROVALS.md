# R1_APPROVALS.md

This file records Nathan's approvals for SPANA engine v2-rebuild R1, so that no decision exists only in chat.

## 2026-10-24 13:44 (Sydney): R1 implementation approval, conditional

**Approved in principle**
- an isolated `player_tracker_rb/` package;
- detector and appearance backend abstractions;
- ONNX Runtime deployment;
- a tracker-owned LAB descriptor;
- one logical target identity, with separate per-frame detection or tracklet identity;
- explicit loss and reacquisition records;
- historical weights used as audit and reference values only;
- motion used as a plausibility and veto signal only;
- no imports from `football_analysis/`;
- no changes to recovered files;
- no event-engine changes in R1.

**Person detector.** YOLOX-s on ONNX Runtime is approved for DEVELOPMENT / EVALUATION. YOLOX-m may be used as a fallback if small-player recall is inadequate. It is **not** to be called PRODUCTION_APPROVED until the licensing record, including the COCO caveat, is resolved.

**Detector input size.** This may be evaluated on GOLD using **player-detection quality only**. Record for each candidate size:
- target availability;
- nearby recall;
- partial and merged detections;
- runtime and memory.

Choose the smallest adequate setting. **Never** choose it on event success.

**Appearance.** A composite of the LAB descriptor and a DINOv2 ViT-S/14 embedding is approved for DEVELOPMENT / EVALUATION. **DINOv2 is a generic embedding, not player-ReID truth.** Its usefulness must be measured on:
- same-kit players;
- front/back views;
- partial crops;
- pose changes;
- long-gap reacquisition.

**DINOv2 licence.** Record the current official licence text and the model-card licence before download. Do not rely on old issues.

**Ultralytics.** DEVELOPMENT_ONLY and not production-approved. It is not an R1 production dependency, and there is no silent fallback to it.

**Model export.** `tools_rb/export_models.py` is approved:
- inputs pinned;
- outputs hashed;
- settings recorded;
- any cloud conversion copied to the Mac and hash-verified before use.

**GOLD human labels are required before coding.** The steps are:
1. Nathan saves `R1/gold_label_sheet/GOLD_HUMAN_LABELS_v1.csv`, using HUMAN_UNCERTAIN where unsure.
2. Validate, hash, mark FROZEN, commit, and include in the next snapshot.
3. Never modify the frozen file. Corrections go in a v2 file.

The labels are the GOLD tracking truth; the historical 3N output is DIAGNOSTIC / REFERENCE only.

**Royal-blue ball boy.** He is an ordinary distractor. No coordinate, colour, frame or person-specific exclusions are allowed; the identity system must reject him on its own merits.

**Second backup.** This remains an open risk. Never report FULL DISASTER RECOVERY = YES until a second independent copy exists. Establish one if practical.

**Before implementation:**
- verify the working tree is clean;
- after the labels are committed, create the `R1_IMPL_PRE_<ts>` snapshot, verified from the archive copy.

**Coding order:**
- A: contracts/config
- B: video/provenance
- C: detector
- D: appearance
- E: population
- F: motion/camera
- G: target state
- H: output
- I: run entry
- J: tests
- K: GOLD run

Start with `config.py` and `output_schema.py` and their tests.

**Rules carried into implementation:**
- Every threshold lives in config, with name, type, units, meaning, default, source (HISTORICAL_REFERENCE / NEW_R1) and validation status.
- The historical contract is preserved, and additions are additive only.
- Identity decisions are contrastive, never nearest-only, IoU-only, colour-only or motion-only.
- The target appearance bank is updated only from trusted, uncontested observations.
- All synthetic identity tests must pass before GOLD.
- GOLD is not a tuning loop: classify each failure, write any fix as a general rule, and record every change.
- No event scoring is used to judge the tracker.
- Commit at every meaningful checkpoint.
- Every milestone archive contains the complete project state.

**Next action.** Wait for the labels. Then validate, hash, commit, snapshot, verify, and report a label summary including any HUMAN_UNCERTAIN frames. **Stop for final implementation approval.**

## 2026-10-24 18:29 (Sydney): Generic player tracker requirement (binding for every R1 stage)

Nathan issued `R1/R1_GENERIC_TRACKER_REQUIREMENT.md` (verbatim copy). In summary:

- `player_tracker_rb/` knows only a TARGET_PLAYER, which is seeded from generic inputs (seed frame and box), and other people, which are generic PERSON_CANDIDATEs.
- **No benchmark-specific logic anywhere.** That rules out shirt numbers, clubs, team colours, clips, frame numbers, camera positions, stadiums, and known distractors.
- Benchmark subjects (for example the T001 → SOL #23 mapping) may live only in labels, fixtures, tests, evaluation reports and notes.
- Any nearby person can compete for the target identity, without first needing a persistent ID or a role or team label.
- Neither shirt-number OCR nor kit colour may ever establish identity on its own.
- A failure exposed by a benchmark must be fixed as a general rule, never a frame-specific one.
- Reacquisition must be generic and stricter than ordinary association, and must answer AMBIGUOUS or LOST when evidence is insufficient.
- Core unit tests use neutral identities: TARGET, PERSON_A, PERSON_B, PERSON_C. GOLD is a real-footage integration benchmark only.
- **The test for every design decision:** "Would this still work if the selected player were someone completely different in another match?"

**Applied the same day (commit "r1-generic"):**
- Benchmark wording removed from `player_tracker_rb/`. The historical exemplars are cited only as HE1 and HE2.
- The R1-A synthetic fixture now uses neutral, made-up seed values.
- `tests_rb/test_package_hygiene.py::TestNoBenchmarkKnowledgeInTracker` fails if benchmark terms appear in the package.
- No logic or contract change. The config hash is unchanged, because evidence text is not hashed.

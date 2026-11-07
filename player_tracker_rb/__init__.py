# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R1/R1_DESIGN_DECISIONS.md D02 (isolated package); R1/R1_APPROVALS.md (stage order)
"""player_tracker_rb: SPANA engine v2-rebuild selected-player tracker (R1).

NEW code. It does not reproduce the lost historical player_tracker / player_tracker_mo internals. It reproduces
the historical Pass-1 JSON contract (see output_schema) and adds clearly labelled v2 blocks.

Allowed imports: Python standard library, numpy, cv2, onnxruntime, and this package. Never football_analysis,
player_tracker, player_tracker_mo, event_engine_v1 or anything under recovered/ (D02, D21).

Stages: R1-A (config, output_schema), R1-B (video_io, provenance), R1-C (detector), R1-D (appearance), R1-E
(population, motion, camera), R1-F (target identity), R1-G (run_rb: the integrated Pass-1 pipeline).
"""
__version__ = "1.3.2-r2rc"   # R2 RESIDUAL SAME-KIT CONTACT STAGE (2026-11-07: credible contained body, depth-apart touch) + R2 STRIDE / SIZE-ANOMALY STAGE (2026-11-07) + R2 PHYSICAL-SEPARATION CONTRAST EXCLUSION (2026-11-07) + WEAK-PART NESTING RULE (2026-11-07) on top of the R2 PRODUCTION CONSOLIDATION (2026-11-06): handoff safety default OFF; otherwise R2 DEV remediation F1 + F3 + F3c (population safety) + F5 (post-anomaly re-verification, off) + F8 (camera scale corroboration) + F9 (LOST search gate) + F4/F6 (time-boundary comparisons) + F2 (deferred seed anchor) + identity layer (diagnostic, not used by target trust) + NOT_TARGET memory (off) + target reference profile (diagnostic) + same-frame handoff safety (on) + jersey-number evidence (diagnostic) + human-assisted reacquisition checkpoints (opt-in)

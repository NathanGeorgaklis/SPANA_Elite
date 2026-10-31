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
__version__ = "0.8.2-r2c"   # R2 DEV remediation F1 (seed containment) + F3 (merge corroboration) + F3c (population safety)

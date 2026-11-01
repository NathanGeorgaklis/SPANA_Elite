# R2 DEV F8: camera zoom drift (2026-11-01)

**Instruction (Nathan, 12:00):**
- Fix false camera-scale estimates that distort target position and size. Focus on C006; keep the logic generic.
- DEV only, no EVAL. Fail closed on an unreliable camera scale.
- Do not change appearance or reacquisition; `target_reverify_enabled` stays False.
- Fast execution mode.

**Root cause (camera layer).** While a broadcast camera pans over a deep scene (pitch near, stands far), the
similarity fit (`cv2.estimateAffinePartial2D`) reports a steady false zoom.
- C006: the camera reports ×0.66 over 4 s, while the median height of tracked people rose (×1.09).
- That scale was applied to the held target size, the gate and the predicted boxes.
- Measured with `f8_scale_probe*.py.txt`, comparing windowed fitted scale against the heights of IoU-matched people:

| Clip | 1 s window | 3 s window |
|---|---|---|
| C006 | 1.26 | 1.52 |
| C033 | 1.50 | 1.62 |
| SYO C012 | 1.18 | 1.33 |
| All other run clips | ≤ 1.13 | ≤ 1.13 |

**Fix (`camera.py`, step 8; commits `8baf561` and `62a4968`).**
- People's scale: previous person boxes (moved by the fit, size kept, nested part boxes counted once) are matched
  one-to-one to current boxes by IoU ≥ 0.5. With ≥ 3 matches, the median log height ratio is taken.
- If the fitted scale and people's scale disagree by more than ×1.15 over 1 s or 3 s of ESTIMATED frames, the scale
  is UNCORROBORATED. It stays rejected until both windows agree within ×1.07.
- While rejected, the frame is a translation-only motion: scale 1, the image centre moved as the fit moves it, reason
  SCALE_UNCORROBORATED, and the full fit kept as `measured_matrix`. This fails closed: no size change is propagated.
- With fewer than 3 matched people the fit is used as before. Cuts and unmeasured frames restart the check.
- Registry r2f8-2.

**Tests.** `tests_rb/test_camera_scale_f8.py` (11 tests, at 30/15/10 fps):
- pan; real zoom kept; false zoom rejected; too few people; cut; crowded scene with each person's own motion and
  depth;
- review regressions: slow drift, no flapping, unbiased fallback, part boxes, resolution change.

Full suite 666 passed, 9 skipped (env), 7 xfailed. Synthetic eval identical.

**Review (one focused pass on `8baf561`).** H1 (slow drift / flapping), H2 (biased fallback translation), M3 (part
boxes) and L5 (resolution reset) were reproduced and fixed in `62a4968`. Open LOW items:
- per-frame height noise adds up more at high fps;
- IoU matching drops fast movers;
- a real zoom while the crowd recedes would be rejected (fails closed).

**DEV (affected clips only; EVAL not run).** The other 5 run clips never exceed the threshold (probe ≤ ×1.13), so
their camera output, and hence their run, is identical to `7df8045`; they were not rerun.

| Clip | Pre `7df8045` | Post `62a4968` | Uncorroborated frames |
|---|---|---|---|
| C006 | P1 3, P2 0, P3 0, P4 0, cov 0.201, bank 1/5 | identical | 188 |
| C033 | P1 1, P2 0, P3 1, P4 1, cov 0.249, bank 0/7 | identical | 323 |
| SYO C012 | P1 0, P2 0, P3 0, P4 0, cov 0.004, bank 0/1 | identical (merged-reason frames 6 → 8) | 98 |

**C006 target size.** The held target height reference at f250 is 65 px, up from 45 px (truth is about 95–100 px).
The remaining error built up before the check could decide (f100–160).

**C006 coverage is unchanged.** The target moved 3–4 true body heights while LOST, outside the 3 h gate. When the
true box is evaluated, it ties a neighbour on appearance (F5). The camera is no longer the failing layer for C006.

**Safety.** No new false confident identity, no swaps, no wrong reacquisition, and the same bank samples. Appearance
and reacquisition code are untouched.

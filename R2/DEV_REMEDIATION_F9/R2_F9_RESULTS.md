# R2 DEV F9: lost-target search region (2026-11-01)

**Instruction (Nathan, 2026-11-01 13:18):**
- Let the tracker find the target after larger real movement while LOST, without weakening identity trust.
- Focus on C006; keep the logic generic. DEV only.
- Expand the candidate search, not identity trust: finding a candidate is not trusting it.
- Keep the F5 rules and `target_reverify_enabled = false`. No bank updates while untrusted.

**EVAL was not run, opened or scored.** No frozen label, fixture, holdout or Task 3R material was touched.

## Root cause (first failing layer: target search gate)

- C006 (62a4968 logs) is LOST from f160.
- The reacquisition gate is capped at `gate_max_h` = 3 h, which it reaches about one second after the last
  confirmation.
- The true target box was 3–6.2 h away (tracker-height units) at absences of 1.1–2.6 s. That is a running player,
  not a tracking error.
- So the box was never evaluated (`TRUE_BOX_NOT_EVALUATED`, 68 boxed available frames) until `lost_long_s` dropped
  the positional prior.
- The identity tests were never reached. This is a search-region problem, not an appearance problem.

## Change (`15a438d`, review fix `3793d7e`; `player_tracker_rb/target.py`, SEARCH GATE docstring)

- While LOST, or REACQUIRING from LOST, the cap no longer applies. The radius is
  `g0 + gate_lost_growth_h_per_s · (t − t0)`:
  - `g0` is the ordinary capped gate on the first LOST search frame `t0`;
  - `gate_lost_growth_h_per_s` = 3.0 h/s, a running speed in height units (new parameter; registry `r2f9-1`).
- Only the search grows. Every candidate still faces the same tests: appearance floors, contrast against
  competitors, contact/occlusion, motion, the reacquisition verification window, and episode verification.
  The F5 rule stays OFF. The bank still updates only on trusted frames.
- **Review M2 fix (`3793d7e`).** The first version (`15a438d`) computed the uncapped radius from the last confirmation.
  - When an AMBIGUOUS target (gate frozen at about 0.82 h) aged into LOST, the gate jumped to about 3.95 h, partly
    reopening the F6 class.
  - Now F9 adds no jump: the first LOST frame uses the pre-F9 gate, and growth starts from there.
- **Known limit, pre-existing and not introduced by F9.** At AMBIGUOUS→LOST the ordinary gate already jumped from
  the frozen value to the capped value (0.82 → 2.88 h in the synthetic probe). Left as is: shrinking it would change
  pre-F9 behaviour.

## Tests (`tests_rb/test_lost_search_f9.py`, 8 tests at 10/15/30 fps where relevant)

| Test | Notes |
|---|---|
| Large movement during loss is searched before the prior drops | |
| Found ≠ trusted | REACQUIRING untrusted through the window; new episode ≥ 2 |
| Camera pan during loss | Compensated |
| Same-kit neighbour at the return | Never trusted, bank pure |
| Wrong distant candidate | Never trusted; never searched outside the gate |
| Target leaving and re-entering the frame | |
| AMBIGUOUS → LOST adds no jump; growth rate pinned at `gate_lost_growth_h_per_s` | Fails on `15a438d` |
| Gate continuous from LOST into REACQUIRING | Outgrows the old cap |

The search tests fail on `3d17dc3`, the pre-F9 code.

`test_pipeline_longrun`: `target_known_others` was added to the bounded-growth exemptions. It is bounded per frame;
a 3600-frame run shows no growth trend, 0 swaps and 0 bank violations.

**Suite (`3793d7e`, cloud):** 674 passed, 9 skipped (environment), 7 xfailed (5 F3b + 2 F5 residual), as before.
**Synthetic tracker evaluation:** identical to F3c.

## Focused review and test audit (on `15a438d`)

- No new path to a wrong trusted identity, a swap or bank contamination.
- **M2 (gate jump at AMBIGUOUS→LOST):** fixed in `3793d7e`, with a regression test.
- **MEDIUM, accepted and recorded.** A look-alike can be locked onto (untrusted) up to about 1.9 s earlier than
  before, because it now falls inside the search sooner.
  - In the reviewer's probe, wrong untrusted TRACKING frames rose from 46 to 63–103.
  - All of those frames were untrusted, and the bank stayed pure.
  - On DEV the count of untrusted off-target frames is unchanged on all 7 clips (`diag/UNTRUSTED_OFF_TARGET.txt`).
- **LOW (test gaps: growth rate not pinned; REACQUIRING continuity):** both added.

## DEV results (EVAL not run)

Affected clips are those with frames LOST at the cap while the prior was still valid on `7df8045`: C006, C016, C020,
C033, SYO C004, SYO C012 and SYO C016. All 7 were rerun on `3793d7e`. The other 4 clips (SYO C021 unaffected; C034,
C038 and SYO C032 refused at the seed) are reused from `7df8045`.

The baseline is `62a4968` (F8), combined in the same way.

| | 62a4968 | 3793d7e (F9) |
|---|---|---|
| Initialised / refused | 8 / 3 | 8 / 3 |
| P1 false confident identity | 5 | 5 |
| P2 swaps | 0 | 0 |
| P3 wrong reacquisition | 1 | 1 |
| P4 trusted on unavailable | 1 | 1 |
| Trusted / available | 818 / 2105 (0.389) | 818 / 2105 (0.389) |
| Bank accepted / contaminated / unverified | 72 / 10 / 35 | 72 / 10 / 35 |
| C006 trusted coverage | 0.201 (57/284) | 0.201 (57/284) |
| Merged-reason frames (diagnostic) | 837 | 882 (SYO C004 28 → 72, SYO C012 8 → 9; untrusted) |

Per-clip official metrics are identical on every clip (`COMPARE_vs_62a4968.json`).

**C006 candidate recovery** (`diag/`, 214 boxed available frames):

| Blocker class | Pre | Post |
|---|---|---|
| TRUE_BOX_NOT_EVALUATED | 68 | **35** |
| FAILED: below reacq floor + crop occluded by other + unknown competitor in contact | 4 | 27 |
| Other FAILED classes (floor / contact) | 8 | 18 |
| TRUSTED | 33 | 33 |

- The search now reaches the true box on 33 more frames.
- On those frames it fails the unchanged identity tests. The main reason is a contact/occlusion with an unknown
  body, which is a correct abstention.
- So candidate recovery improved, and trust is unchanged.
- The other 6 clips show no change in any blocker class.

## Remaining

- **C006:** 35 frames are still not evaluated, and the newly found frames are blocked by contact/occlusion. Not
  diagnosed further in F9.
- **Carried over:**
  - F5: residual risk; the rule stays OFF.
  - F3b: 5 reproducers.
  - F2: seed refusals.
  - F4: SYO C012 coverage.
  - F6: one-frame lag, plus the pre-existing AMBIGUOUS→LOST gate jump.
  - F7: box convention.

**Raw:** `raw/R2_DEV_F9_raw_3793d7e.tar.gz`
(sha256 `58a560c7e7e59a6e9328b5e586fca30137cacafe2a23be94e974b3f4217c53d3`).

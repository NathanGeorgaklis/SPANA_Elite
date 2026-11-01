# R2 DEV F2: seed anchor refusals (2026-11-01)

**Instruction (Nathan, 2026-11-01 17:38):**
- Allow safe initialisation when the correct seed player is detected but the crop is partial, near the image edge,
  or has nested/alternative detections. Focus on C034, C038 and SYO C032. Keep it generic. DEV only.
- The seed must still fail closed if physical identity is ambiguous. Do not weaken downstream identity or
  reacquisition rules. No contaminated seed bank.

**EVAL was not run, opened or scored.** No frozen label, fixture, holdout or Task 3R material was touched.

## Root causes (first failing layer; `tools_rb/r2_f2_seed_diag.py`, DEV only; `F2_SEED_DIAG_*.txt`)

**The rule.** At the seed, the target bank had to be anchored on the seed-frame crop itself. That crop faces every
ordinary bank rule, so any PARTIAL / TOUCHES_IMAGE_EDGE / NESTED / ALTERNATIVE / CONTAINS_OTHER / NOT_ISOLATED flag
refused the whole run, even when the seed picked exactly one body.

| Clip | Seed frame | Same body afterwards | Layer |
|---|---|---|---|
| C034 | One clean match (IoU 0.69, tracklet T002), PARTIAL + TOUCHES_IMAGE_EDGE, nothing touching it | Bank-clean on T002 at f17 (0.28 s); similarity to the seed crop 0.899–0.988 while pending | Seed-anchor policy |
| C038 | The human box matches a NESTED/ALTERNATIVE part box (IoU 0.61, no tracklet) inside host T011 (IoU 0.59), among further part boxes; another body T014 overlaps | T011 bank-clean at f4 | Seed-anchor policy, but the part box cannot be shown to be the host's own body (review HIGH-1 below) |
| SYO C032 | Clean match T012 (IoU 0.76), NOT_ISOLATED: neighbour T005 overlaps (IoU 0.15) | The truth labels contact with another person from f1; T012 is never bank-clean and merges by f32 | Genuine second person: refusing is correct |

## Change: deferred seed anchor (`bd1127c`, review fixes `27dedec`; `target.py` "DEFERRED SEED ANCHOR")

**When the anchor is deferred.** The seed-frame crop must be refused **only** for geometry (`_SEED_DEFERRABLE`), and
**all** of these must hold:
- the matched detection **itself** has a population tracklet;
- no box nested with it has another tracklet;
- the tracklet has no merge lineage;
- no other body could be the boxed person: IoU ≥ `seed_min_iou` with the seed box, or ≥ `seed_contain_competitor_max`
  of it inside the box.

Otherwise the seed is refused at once, as before.

**While pending.** Every frame is TRACKING on that tracklet, **untrusted** (LOW_QUALITY_CROP), and is never a bank
sample. The seed frame is included.

**The anchor.** It is the first crop passing the **ordinary** bank update; no bank rule is relaxed. Its confidence is
`0.5 + 0.5·a`, where `a` is the margin of its similarity to the seed crop above `reacq_appearance_floor`. The normal
tracker runs from the next frame.

**Refused (fail closed) before the anchor if any of these happens:**
- a camera cut;
- a frame where the tracklet is not observed;
- CONTESTED / DUPLICATE_ABSORBED / IN_MERGED_REGION / MERGED_SUSPECTED / SIZE_ANOMALY or MERGE_LINEAGE;
- a usable crop matches the human-identified seed crop below `reacq_appearance_floor`;
- `seed_anchor_max_s` (1.0 s) passes.

`run_rb` turns the refusal into RUN_REFUSED and removes the outputs.

**Parameters and version.** New parameter `seed_anchor_max_s` (registry `r2f2-1`); version `0.13.1-r2f2`. Seeds the
bank accepted before follow the identical code path.

## Review and test audit (on `bd1127c`): two wrong-identity paths, both fixed in `27dedec`

- **HIGH-1 (reproduced).** The human box matches the target's own box nested inside a larger box of another person.
  - The host box carried the tracklet, so the bank anchored on the other person: trusted at 1.0, bank contaminated.
  - **Fix:** a nested / alternative box without its own tracklet is refused; a host never carries the identity.
    Geometry cannot tell a part box from a second body (the F6 finding).
  - **Cost:** C038 stays refused.
- **HIGH-2 (reproduced).** The edge-cut target walks out and a same-kit player enters at the same spot on the next
  frame.
  - The population carried the tracklet over (84 wrong trusted frames).
  - **Fix:** every usable pending crop must match the seed crop at `reacq_appearance_floor`. The entrant scores
    0.838, so the seed is refused.
  - **Residual:** a near-identical look-alike (embedding cosine ≥ about 0.77 in the same kit) is not separable. This
    is the same limit as ordinary tracking.
- **MEDIUM:**
  - the anchor confidence of 1.0 was undocumented; it now uses the formula and is documented in IDENTITY CONFIDENCE;
  - the known-other timer counted pending time; it is now reset at the anchor.
- **LOW:** an empty refusal set never defers.
- **Tests:** strengthened, and all four mutants are now killed: no own-tracklet check, no seed-crop check, no break
  flags, no cut check.
- **Pre-existing, not F2.** A hidden same-kit person emerging during a one-frame detector miss gives one wrong trusted
  frame at `b293a28` too, with a clean seed.

**Not adopted.** "Refuse a box shrinking at the edge" (against a walk-out swap): C034's real target box shrinks by 27%
while pending.

## Tests (`tests_rb/test_seed_anchor_f2.py`, 16 tests, at 10/15/30/60 fps where relevant)

**Anchored cases:**
- image-edge player, anchored once clear; untrusted until then; the anchor is on the target at the documented
  confidence;
- partial body cut by the bottom edge;
- body seed with its own duplicate part box (seeded directly, nothing deferred);
- a clean seed is unchanged;
- a second person overlapping at the seed who then clears: anchored, never wrong-trusted;
- `seed_anchor_max_s` boundary: an anchor at exactly the limit is accepted, one frame later it is refused.

**Refused cases, each with its reason checked:**
- a seed on a nested part box with no track of its own;
- a genuine second person in contact;
- an ambiguous seed box holding two bodies;
- an edge player who never clears;
- a cut while pending;
- a broken tracklet;
- each break flag, injected into the pending tracklet;
- the HIGH-1 host box of another person;
- the HIGH-2 edge swap onto a same-kit entrant;
- a merge while pending.

**Suite (`27dedec`, cloud):** 704 passed, 9 skipped (environment), 8 xfailed (as F4/F6).
**Synthetic evaluation:** identical to F3c.

## DEV results (EVAL not run; the 3 refused clips rerun on `27dedec`; the other 8 reused from `15966c3`)

Seeds the bank accepted take an unchanged code path. The `_frame` change is a new keyword whose default is the
previous behaviour.

| | 15966c3 | 27dedec |
|---|---|---|
| Initialised / refused | 8 / 3 | **9 / 2** |
| P1 false confident identity | 5 | 5 |
| P2 swaps | 0 | 0 |
| P3 wrong reacquisition | 1 | 1 |
| P4 trusted on unavailable | 1 | 1 |
| Trusted / available (run clips) | 818 / 2105 (0.389) | 1138 / 2674 (0.426) |
| Bank accepted / contaminated / unverified | 72 / 10 / 35 | 86 / 10 / 46 |

- **C034:** initialised.
  - Anchored at f17 (0.283 s), similarity to the seed crop 0.903, confidence 0.677.
  - P1 0, P2 0, P3 0, P4 0.
  - Trusted coverage 0.562 (320/569).
  - Bank: 14 accepted, 0 contaminated, 11 unverified (no human box).
- **C038:** refused (a nested part box without its own track: physical identity cannot be shown).
- **SYO C032:** refused (the seed body's crop fell to 0.829 against its seed crop while in genuine contact).
- **Unchanged:** every metric of the other 8 clips (same code path for accepted seeds).

## Remaining

- **C038:** would need evidence that a nested part box belongs to its host, such as appearance consistency between
  the part and the host body or a part-aware detector. Geometry is not enough.
- **SYO C032:** abstention is correct, because the target is in genuine contact.
- **Carried over:**
  - F4 residual (Nathan's decision);
  - SYO C012 / C033 appearance discrimination;
  - F5 (off);
  - F3b;
  - F7.

**Raw:** `raw/R2_DEV_F2_raw_27dedec.tar.gz` (sha256 in `raw/SHA256SUMS.txt`).

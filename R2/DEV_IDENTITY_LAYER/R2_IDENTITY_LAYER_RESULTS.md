# R2 DEV: persistent football identity layer (diagnostic; 2026-11-01)

**Instruction (Nathan, 2026-11-01 21:39):**
- Build persistent `PERSON_ID`s across tracklet breaks from the identity evidence that already exists:
  - colour + DINOv2/ReID evidence;
  - appearance history from trusted observations;
  - trajectory;
  - temporal team evidence;
  - optional temporal jersey numbers.
- DEV only. Do not replace the tracker or weaken `identity_trusted`.
- ReID, team and number are supporting evidence, never proof alone. Aggregate over time; UNKNOWN stays valid.
- No untrusted target frame may update the trusted appearance history.
- Stop before changing target trust logic or running EVAL.

**EVAL was not run, opened or scored. The target tracker is unchanged.**
- The layer is not called by `run_rb`, and no DEV metric changes.
- Tests confirm that the tracker's output is identical with and without the layer.

## Design (`player_tracker_rb/identity.py`; module docstring is normative)

`PersonIdentityLayer` is **read-only** over the population (tracklets, closures, clean per-tracklet banks) and the
tracker's output (the tracklet it is TRACKING on, `identity_trusted`, its bank). It never writes to the tracker.

**Statuses.** Every tracklet is one of:
- **PENDING**: evidence is being gathered;
- **LINKED**: it continues an earlier person;
- **NEW**: there is positive evidence that it is no earlier person;
- **UNKNOWN**: final, never forced into a link later; the earlier persons it might be are recorded.

**Link rule.** A newborn tracklet B is linked to person P only if **all** of these hold:
1. **Timing.** P is not visible now; P was last observed within `identity_link_window_s` (3 s) of B's birth; and no
   other person's tracklet is still live but unseen nearby (otherwise B stays undecided).
2. **Physical.** B is reachable: within `identity_gate_base_h` + `identity_max_speed_h_per_s` (3 h/s) × the time P
   has been unseen. Boxes are compared in current-frame coordinates, with camera compensation. With no valid positions
   (after a cut) there is **no link at all**.
3. **Kit (team).** Kit evidence can only refuse a link (median LAB below `identity_kit_veto_lab`).
   - At the current floors this veto is **redundant**: a LAB similarity below 0.3 already caps the composite below
     `appearance_floor`.
4. **Number.** A confirmed different jersey number refuses a link.
   - A number is confirmed only after `identity_number_min_reads` agreeing readings and no conflicting one.
   - Duplicate, unreadable or conflicting readings confirm nothing, and an equal number never makes a link.
   - No number reader exists in R2; this is a hook.
5. **Appearance, aggregated over time.** The measure is the median, over B's own clean observed frames (at least 0.3 s
   and at least 2 frames), of the best composite similarity (colour + DINOv2) to P's history. It must reach
   `appearance_floor`, or `reacq_appearance_floor` for a link **into the target person**.
6. **Contrastive.** P must beat every other candidate by `ambiguity_margin`. This includes candidates refused only by
   the physical gate, because the gate can be wrong.
7. **Exclusive in both directions.** No other recent newborn may be linked to P, list P as a possible person, or match
   P as well.

**Histories.**
- A person's history is the clean samples from its own tracklets' population banks.
- The **target person's history is the tracker's own trusted bank only**, so an untrusted target frame never enters
  it. The tests check this sample by sample.

**Target / not-target marks.**
- These are set only once the tracker has trusted the same tracklet for `target_known_other_min_s`. That tracklet's
  person becomes the target person, and every separately observed body in that frame is NOT_TARGET.
- A person flagged both ways, or two target persons seen at once, is a **CONFLICT** and narrows nothing (fail closed).

**Memory.** Bounded: persons gone longer than the window are forgotten, unless a pending newborn still lists them.

**New parameters.** Nine `identity_*` parameters, all NEW_R1 and UNVALIDATED (registry `r2id-1`). Version
`0.14.1-r2id`.

## How the ReID evidence is used

- The layer uses the production composite (0.35 LAB + 0.65 DINOv2) and no new model. The R2 ReID evaluation found no
  candidate model that improves safe same-kit separation.
- Appearance is used **only contrastively and in time aggregate**: the median over the newborn's frames, a margin over
  every other candidate, and exclusivity in both directions. It is never used alone: a valid physical gate is always
  required.
- Kit and number can only veto.

## Review and test audit

**Review of `e234b47`.** It reproduced the following in the diagnostic layer (the tracker was never affected); all
are fixed in `8e45920` with regression tests:
- a false target person (a look-alike stranger took the target person through one-shot exclusivity);
- a false exclusion (the true target linked to a departed same-kit NOT_TARGET person while its old tracklet was still
  held);
- wrong links in the open world;
- a tracker error becoming a permanent mark;
- a physical gate in mixed coordinates;
- a crash when a candidate was pruned while still referenced.

**Mutants.** Every safety rule's mutant is killed except the kit veto, which is equivalent at the current floors (see
above).

**Residual, recorded as `expectedFailure`.** A **never-seen** same-kit player whose embedding is near-identical to the
target's (cosine 0.85, composite about 0.90 ≥ 0.85) appears where the target was last seen. That player is linked
into the target person. Appearance cannot reject an unseen look-alike, as the ReID evaluation showed.

**Tests:** `tests_rb/test_identity_layer.py`, 22 tests + 1 xfail, at 15/30 fps. They cover:
- same-kit crossings (0.75–0.97, and twins stay UNKNOWN);
- break/restart;
- re-entry after occlusion;
- a different-team crossing;
- unreadable, duplicate, conflicting and different jersey numbers;
- the tracker being unchanged;
- NOT_TARGET surviving a break;
- bounded memory;
- the review regressions.

**Suite (`8e45920`, cloud):** 726 passed, 9 skipped (environment), 9 xfailed (+1 identity residual).
**Synthetic tracker evaluation:** identical to F3c.

## DEV results (`tools_rb/r2_identity_eval.py`, layer at `8e45920`; 6 clips)

**Truth.** Only the target is labelled. A tracklet counts as TARGET when its observations match the human target box.
Links between non-target players **cannot be scored** on DEV.

| Clip | Decisions | Untrusted target frames (true box detected) | Others excluded: layer / tracker / layer only | False exclusion (layer) | True tracklet = target person | False target person |
|---|---|---|---|---|---|---|
| C016 | NEW 9, UNKNOWN 2 | 2 | 4 / 7 / 0 | 0 | 2 | 0 |
| C006 | NEW 13, UNKNOWN 4 | 46 | 6 / 88 / 0 | 0 | 15 | 0 |
| C020 | NEW 8, UNKNOWN 2, LINKED 2 | 65 | 151 / 19 / **132** | 0 | **65** | 0 |
| C033 | NEW 10, UNKNOWN 16, LINKED 1 | 52 | 45 / 45 / 0 | 0 | 0 | 0 |
| SYO C012 | NEW 28, UNKNOWN 43 | 119 | 0 / 1202 / 0 | 0 | 0 | 0 |
| C034 | NEW 9, UNKNOWN 13 | 17 | 0 / 95 / 0 | 0 | 0 | 0 |
| **Total** | NEW 77, UNKNOWN 80, **LINKED 3** | 301 | 206 / 1456 / 132 | **0** | 82 | **0** |

**Findings:**
1. **Safety held on DEV.** There were 0 false exclusions of the true target and 0 non-target bodies marked as the
   target person, on all clips. None of the 3 LINKED decisions involved a target tracklet (0 wrong target links).
2. **Links across tracklet breaks almost never fire on real footage.**
   - 3 links against 157 NEW/UNKNOWN decisions.
   - The target's own new tracklets were never relinked across a break: 5 NEW at the start, 2 UNKNOWN.
   - The contrastive, exclusive and open-world rules, together with same-kit appearance overlap, make the layer
     abstain, as designed.
3. **Candidate narrowing improves in one clip only.**
   - On C020 the layer excluded 132 other bodies the tracker's tracklet-level known-others no longer excluded, with no
     false exclusion. Its NOT_TARGET marks survive the tracker's episode reset.
   - On every other clip the layer added nothing. The tracker's own known-others exclude far more (1456 against 206).
4. **The C020 "true tracklet = target person" (65 frames) is same-tracklet continuity, not a link.**
   - The layer keeps the target person on tracklet T002 through a SIZE_ANOMALY that the tracker deliberately treats as
     an identity break (an unseen second body may have been in the box).
   - It was correct here, but the layer does not model anomaly breaks or merge lineage on a continuing tracklet. That
     is a **known gap that would need closing before any integration**.

## Decision

**RECOMMEND TARGET-INTEGRATION: NO.** The reasons:
- **Narrowing evidence is thin:** a gain on one of six clips, and almost no cross-break links.
- **Linking into the target cannot be validated:** the target is never relinked on DEV.
- **Open-world residual:** a never-seen near-identical player.
- **Known gap:** continuing-tracklet anomalies.
- **No truth for other players:** links between non-target players cannot be scored on DEV.

The layer stays in the package as a diagnostic, unused by target trust.

**Next options (Nathan's decision):**
1. Close the anomaly/merge-lineage gap: break the person on SIZE_ANOMALY end or MERGE_LINEAGE, as the tracker does.
2. Label non-target identities on a few DEV clips, so that links can be scored.
3. Re-evaluate once a stronger, licensable football ReID or jersey-number reader exists.
4. Only then consider using NOT_TARGET persistence as an extra exclusion in reacquisition. That is the one narrowing
   the data supports, and it would still need a separate stage and approval.

**Raw:** `raw/R2_DEV_IDENTITY_raw_8e45920.tar.gz` (sha256 in `raw/SHA256SUMS.txt`); `IDENTITY_EVAL.json`.

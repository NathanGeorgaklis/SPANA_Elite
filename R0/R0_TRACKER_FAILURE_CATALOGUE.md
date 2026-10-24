# R0_TRACKER_FAILURE_CATALOGUE.md

SPANA engine v2-rebuild, R0. Read-only. Written 2026-10-24.

This file lists every historical tracker or identity failure mode with surviving evidence. It is not limited to the selected-player tracker: it also covers identity failures that reached Pass 2.

Each entry gives:
- **symptom**;
- **first failing layer**;
- **historical evidence**, with its level;
- **downstream consequence**;
- **R1 protection**: whether the rebuilt tracker needs explicit protection.

Labels "VISUAL (Claude)" mark per-frame visual judgements made by Claude in earlier tasks. They are **not** human-signed.

---

## F1. Target swap onto an opponent during a crossing occlusion

| | |
|---|---|
| Symptom | The selected track is confirmed on the occluding opponent (MAR #4) for 7 frames while the target (SOL #23) is hidden behind him, then recovers. |
| Where / when | GOLD_DUEL_01, extract frames 266–273. Stale pre-3L Pass-1 run (Task 3K). |
| First failing layer | **ASSOCIATION.** The trigger was a DETECTION merge, which makes no identity claim. |
| Mechanism | (a) One merged box; #23's kit-pixel share was 0.086. (b) The identity-gate fast path skipped appearance for an "uncontested single candidate". (c) #4's own track was PROVISIONAL, so it was absent from the Hungarian matrix and the contest test. (d) One box was confirmed at appearance 0.289, below the 0.30 floor. (e) After the swap, motion, IoU and size (0.35 combined) plus contaminated HSV colour (0.15) outvoted ReID (0.40), even though ReID preferred #23 in every frame. |
| Evidence | DOCUMENTED: TASK3M_MOTION_ASSOCIATION_AUDIT §1–3; TASK3M_REID_APPEARANCE_AUDIT. VERIFIED_OUTPUT: `replay/association_capture.json` (deterministic: 109/110 records match the scored run). VISUAL (Claude) frame labels. |
| Online health during the swap | ACTIVE, CONFIRMED_DETECTION, occlusion NONE and continuity 1.0 in 6 of 8 frames. `overlap_contests` empty; 0 identity conflicts. |
| Downstream consequence | A false PASS COMPLETE, a false possession loss and regain, and #4's actions attributed to #23. Actor attribution gave 1.0 because its window ended 4 frames before the swap. |
| Current-code status | Absent in the 3N run: #4's track was promoted, so T001 went OCCLUDED and then AMBIGUOUS instead. The structural weaknesses (b)–(e) remain (TASK3N open risk 1). |
| **R1 protection** | **YES, the top priority.** No appearance-free fast path. Every live candidate, PROVISIONAL included, competes or blocks. When the only box overlaps another body or grows abruptly, go OCCLUDED or AMBIGUOUS rather than confirm. Never confirm below the appearance floor. Do not let a post-swap motion model outvote appearance. |

## F2. Merged person detections during contact

| | |
|---|---|
| Symptom | The detector returns one union box for two players in contact. GOLD 266: box area ×3.1 in one frame and centre shifted by −32 px. |
| First failing layer | DETECTION (by design of the detector; not itself an identity error). |
| Evidence | DOCUMENTED: TASK3M_MOTION §1 and §2 table. VERIFIED_OUTPUT: IPR detections 266–268, one box each. |
| Downstream consequence | It enables F1. It also starves the occluder's own track, which was terminated at 267. |
| **R1 protection** | **YES.** Treat an abrupt scale or area jump and a box that overlaps the predicted positions of two tracks as "merged suspected". Do not confirm identity on it, and do not update the appearance bank from it. |

## F3. Same-kit crossings

| | |
|---|---|
| Symptom | No surviving selected-player run contains an examined same-kit crossing. |
| Evidence | DOCUMENTED: GOLD human label ("No same-kit player within 10 m"); IDENTITY_FEASIBILITY_GATE.md. The contested-moment rule caps identity at WEAK. 5 of 7 pilot identity failures were contested or same-kit congestion. |
| First failing layer | UNKNOWN. Colour cues cannot separate the players; only ReID, motion and number can. |
| Related | TASK3L: colour-only team labels also labelled the target's own fragments TEAMMATE (F5) and labelled a gold ball boy TEAMMATE. |
| **R1 protection** | **YES**, and it must be measured. The R2 benchmark needs real same-kit crossing clips. The default when contrastive evidence is insufficient must be AMBIGUOUS. |

## F4. Provisional competitors ignored

| | |
|---|---|
| Symptom | A real nearby player whose track is PROVISIONAL cannot protect his own detection. |
| Evidence | DOCUMENTED: TASK3M Part 11 item 1. TASK3N Part 6: #4 was PROVISIONAL for 86 frames because team was UNKNOWN in 85 of them (pre-3L classifier). |
| First failing layer | ASSOCIATION / LIFECYCLE |
| Downstream consequence | F1. The SEG_D conflict log is also blind (F9). |
| **R1 protection** | **YES.** Competitor status must not depend on promotion or team label. Any detection-backed hypothesis near the target counts. |

## F5. Target body fragments appearing as separate players

| | |
|---|---|
| Symptom | A leg crop, a head or upper body, or a re-emerging partial body of the target is treated downstream as an independent same-kit player. |
| Variants | 3K: a leg fragment. 3L: the full body outside a box held on #4. 3N/3O: the re-emerging body outside a stale held box (person conf 0.19 / 0.12, LOCAL_ROI). |
| First failing layer | Pass-2 other-player context. The root cause is shared: Pass 2 receives only the held bbox, so it does not know the target is OCCLUDED or where it can re-emerge. |
| Evidence | DOCUMENTED: TASK3L_FINAL_REPORT; TASK3O_FINAL_REPORT findings 1–6. |
| Downstream consequence | A false TEAMMATE receiver, then a false PASS COMPLETE that short-circuits arbitration ahead of the DUEL. This was the GOLD false PASS in 3N. |
| **R1 protection** | **YES, as an output contract.** R1 must publish, per frame: whether the box is held or predicted; the latest observed or contested target position (`observed_position_box` already existed); and the target-plausible region. Pass 2 (R3/R4) can then exclude target re-emergence. R1 itself must not split the target into two tracks. |

## F6. Stale held boxes

| | |
|---|---|
| Symptom | A frozen `bbox` for up to 77 frames (2.60 s) while AMBIGUOUS, followed by a relock up to 234.8 px away. Across SEG_A–D there were 54 stale-then-reassociate cases (max 3.74 s, 619.4 px). |
| First failing layer | LIFECYCLE / REPORTING. AMBIGUOUS never ages out, the gate grows without limit, and the held box is reported as the position. |
| Evidence | DOCUMENTED: SEG_D investigation §2–5. Stage B added `observed_position_box`; stale cases fell from 22 to 13 on the re-tracked SEG_D. |
| Downstream consequence | A confident wrong actor (T021) at a good release (SEG_D frame 103), fixed downstream by a freshness gate. In GOLD, Pass 2 searched around the held box (F5). |
| **R1 protection** | **YES.** Make the age of held state explicit and bounded. AMBIGUOUS must decay to LOST after a documented, frame-rate-independent time. Gate growth must be capped and scale-normalised. Keep held and observed positions in separate, explicit fields. |

## F7. Short occlusion (correct behaviour, the reference case)

| | |
|---|---|
| Symptom | Not a failure. On the 3N current code, GOLD 262–265 was ACTIVE on #23 while partly hidden, 266–268 OCCLUDED (held), 269–271 AMBIGUOUS, and 272 confirmed on #23. |
| Evidence | VERIFIED_OUTPUT: GOLD.json. VISUAL (Claude): TASK3N trace CSV. |
| Caveat | While OCCLUDED, the held box sits on pixels the occluder is running through (kit reads #4 at 268–271). That is a stale position, not an identity transfer. |
| **R1 protection** | This is the behaviour to **match or beat**: abstain, then reacquire with evidence. |

## F8. Occlusion and re-emergence identity errors

| | |
|---|---|
| Symptom | (a) Re-emergence onto the wrong player after a crossing (F1, 269–272). (b) The re-emerging target counted as a different player downstream (F5). (c) A selected-id change after loss with no candidate passing the stitch gate: blind_r2_009 T001 → T064, UNRESOLVED; shot_close_range_01 T001 → T076 (wrong-video fixture). |
| First failing layer | ASSOCIATION for (a); interface for (b); REACQUISITION for (c), where the online tracker moved `selected_player` onto a new track. |
| Evidence | DOCUMENTED: TASK3M; TASK3O; LIVE_IDENTITY_TRUST_DOWNGRADE_VALIDATION §2–4 (153 px jump at the boundary); SHOT_CLOSE_RANGE_OFFLINE_RESOLVER_AUDIT. |
| Downstream consequence | (a) A false event outcome. (b) A false receiver. (c) Was handled safely by the downstream downgrade: 29/29 frames OCCLUDED, with no event in the span. |
| **R1 protection** | **YES.** Reacquisition is a separate decision with its own contrastive evidence (§R1 plan). Never move "selected" to a new track without passing it. Otherwise LOST. |

## F9. Identity conflicts invisible to the conflict log

| | |
|---|---|
| Symptom | `identity_conflicts_logged = 0` in both swap cases (GOLD stale run; SEG_D T021). |
| First failing layer | DIAGNOSTICS. The conflict log was populated only in the DORMANT re-entry loop, so AMBIGUOUS relocks and fast-path swaps were never logged. |
| Evidence | DOCUMENTED: SEG_D §3 (point 3); TASK3M_MOTION ("why the online track stays healthy"). |
| Downstream consequence | Actor attribution's conflict halving never fires. `tracking_health.wrong_id_flags` is structurally always 0 (tracking_health.py:64-71). |
| **R1 protection** | **YES.** Every identity decision with a competing hypothesis must be logged per frame, not only re-entry. |

## F10. Target disappearance (leaving frame or long loss)

| | |
|---|---|
| Symptom | The target leaves view; dormant states TEMPORARILY_LOST and OUT_OF_FRAME follow. The old single-target tracker rescanned forever with a fixed 3-hit reacquisition. |
| Evidence | VERIFIED_OUTPUT: OUT_OF_FRAME 12 / 96 track-frames (GOLD/PC, non-selected tracks). DOCUMENTED: CURRENT_ARCHITECTURE.md ("90-minute viability"); ablation §5 (Fix E). |
| First failing layer | REACQUISITION / LIFECYCLE (no long-absence gate). |
| Downstream consequence | The risk of reacquiring the wrong player after long absence is not measured on any surviving clip. |
| **R1 protection** | **YES.** Make the reacquisition evidence bar grow with absence time. Do not re-enter a track on position alone after leaving frame. |

## F11. Camera movement and cuts

| | |
|---|---|
| Symptom | No surviving output contains an `is_cut = True` frame. shot_close_range_01 had an apparent camera-angle change at t ≈ 7–8 s that the cut detector did not flag; whether it was a real cut is **unresolved** (TASK3C). GOLD zooms out at ~5.0–7.0 s, mostly before the seed at 6.0 s. |
| Evidence | VERIFIED_OUTPUT: `is_cut` false everywhere; `camera_shift` small. DOCUMENTED: TASK3C; GOLD label "visibility.camera". The status doc says there is "no camera-motion compensation" in the duel logic (Pass 2). |
| First failing layer | UNKNOWN (insufficient evidence). |
| **R1 protection** | **YES, conservatively.** On a detected cut, identity must be re-established. The IDENTITY_FEASIBILITY_GATE camera-cut rule: identity is never carried across a cut. Gates must be scale-normalised for zoom. Pan compensation must be measured in R2. |

## F12. Target resolver failure

| | |
|---|---|
| Symptom | (a) By design, the resolver never examines within-run drift, so the GOLD swap was AUTO_RESOLVED. (b) shot_close_range_01: the seed came from a different video at the same path; the resolver correctly found no stitch candidate (7 evaluated, all CONFIDENT_REJECT). |
| First failing layer | (a) RESOLVER SCOPE. (b) FIXTURE PROVENANCE, upstream of tracking. |
| Evidence | DOCUMENTED: TASK3M_OFFLINE_RESOLVER_AUDIT; SHOT_CLOSE_RANGE_OFFLINE_RESOLVER_AUDIT; TASK3C headline finding (mtimes, md5, empty grass at the seed, nearest detection 168.7 px). |
| Downstream consequence | (a) A confident wrong outcome. (b) Rejected fixture: Task 3D invalidated it on provenance. |
| **R1 protection** | **YES.** (a) Per-frame identity state must be produced *online* by the tracker, not only at id boundaries. (b) Seed verification: the seed box must coincide with a real detection at the seed frame, and the source hash must equal the fixture hash. Refuse to start otherwise. |

## F13. Incorrect actor downstream because of tracking error

| | |
|---|---|
| Symptom | Events credited to TARGET_PLAYER at high actor confidence when the box was on another body or a stale position. |
| Instances | GOLD stale-run false PASS (actor 1.0, F1). SEG_D T021 primary actor (F6). duel_lost_01: shadow and boot as "ball" (a ball failure, but credited to the target; TASK3F). |
| First failing layer | Tracking (F1, F6) or ball (duel_lost_01). Actor attribution trusts track self-health. |
| Evidence | DOCUMENTED: TASK3M_ACTOR_CONFIDENCE_AUDIT; TASK3F report "Read this first"; SEG_D §1. |
| **R1 protection** | **YES, through outputs.** Per-frame `identity_trusted` must be computed from contrastive physical-identity evidence, not from the track's own continuity. The R1 schema must let actor attribution see the reason a frame is untrusted. |

## F14. Cross-module coupling changes tracking silently

| | |
|---|---|
| Symptom | A Task 3L change to Pass-2 `TeamClassifier` changed Pass-1 behaviour, but all 3L reruns reused a pre-3L Pass-1 JSON. |
| Evidence | DOCUMENTED: TASK3M "Important correction to Task 3L"; TASK3N_CODE_BASELINE (flags asserted). |
| **R1 protection** | **YES.** `player_tracker_rb` imports nothing from Pass 2 / `football_analysis`. Pin and hash every dependency, and record the code hash in each tracking JSON. |

## F15. Checkpoint resume misregistration

| | |
|---|---|
| Symptom | A resumed Pass-1 run on a .mov was offset by +10 frames from frame 171; there were 354 records instead of 364. |
| Evidence | DOCUMENTED + VERIFIED_OUTPUT: STALE_VS_FRESH_REGISTRATION.txt (IoU 1.000 at offset +10). Also TASK3F: random-access seeks on these .mov files drift by 3+ frames in about 150 frames, so frames must be read sequentially. |
| **R1 protection** | **YES.** No resume in R1. Sequential decode only after one seek. Record the frame index and a per-frame content hash (cheap, e.g. a hash of a downscaled frame) in debug output, so registration can be checked. |

## F16. Pixel-unit thresholds

| | |
|---|---|
| Symptom | Gates of 90 / 35 / 350 px, a 45 px fast-path radius and a 25 px/frame resolver radius, all applied across clips from 640×360 to 2056×1144. |
| Evidence | DOCUMENTED (TASK3C; TASK3M; SHOT_CLOSE_RANGE_OFFLINE_RESOLVER_AUDIT). |
| **R1 protection** | **YES.** Express geometry in units of target body height, and time in seconds. |

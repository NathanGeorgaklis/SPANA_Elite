# R0_ACCEPTANCE_TARGETS.md

SPANA engine v2-rebuild, R0 (R0.14). These are historical acceptance references, used for **R5 validation later**.

**Scope.** These references concern the event engine, not the tracker. Tracker acceptance is defined separately in `R0_TRACKER_ACCEPTANCE_TARGETS.md` and must not borrow anything from this file.

**Verification.** Every value below was re-verified in R0 against frozen Task 3Q outputs. Each output's SHA-256 was checked against `task3q/<clip>_postchange/OUTPUT_HASHES_BEFORE_COMPARISON.txt`, which was written before comparison at the time. The Pass-1 inputs to those runs were the frozen 3N tracking JSONs listed in `task3q/PASS1_INPUT_SHA256.txt`.

| Clip | Pass-1 input |
|---|---|
| GOLD_DUEL_01 | 88246efe… |
| pass_completed_01 | 40fa0df1… |

**Rules carried forward:**
- Semantic matches are **MUST MATCH LATER**.
- Numeric values marked **REFERENCE ONLY** must **not** be tuned for. Do not adjust the tracker or the event engine to reproduce them.

---

## GOLD_DUEL_01

**Truth source.** Human-signed truth: `GOLD_DUEL_HUMAN_LABEL.json` v1 (05fe67c6…), sign-off 423ad2ba… by Nathan Georgaklis on 2026-10-19.
- Event: DUEL, outcome WON.
- Target: SOL #23. Opponent: MAR #4.
- Label window: 7.30–9.67 s extract time.

**Historical 3Q output** (`gold_postchange/gold_duel_01_player_analysis.json` 71da1430…, event log 206a9a36…, both hash-verified in R0):

| Event | Window (s) | Outcome | Event conf (band) | Actor (conf) |
|---|---|---|---|---|
| CARRY | 6.53–8.73 | — | 0.94 HIGH | TARGET_PLAYER (1.0) |
| POSSESSION_WON | 6.53 | — | 0.94 | TARGET_PLAYER — *known tracking-start artifact* |
| **DUEL** | **7.43–9.23** | **WON** | **0.858 HIGH** | **TARGET_PLAYER (0.873)** |
| CARRY | 9.87–11.43 | — | 0.909 HIGH | TARGET_PLAYER (0.958) |
| UNKNOWN_RELEASE | 11.43–11.47 | — | 0.88 HIGH | TARGET_PLAYER |
| POSSESSION_LOST | 11.47 | — | 0.88 | TARGET_PLAYER |

**Historical 3Q stats:** duels 1, duels_won 1, duels_lost 0, passes_attempted 0, passes_completed 0, carries 2, possession_won 1, possession_lost 1.

**Conflict noted.** The project status doc lists UNKNOWN_RELEASE at "11.43". The frozen output says 11.43–11.47, with POSSESSION_LOST at 11.47. These are consistent, just less precise in the doc.

### MUST MATCH SEMANTICALLY LATER
- A **DUEL** event exists that overlaps the human label window 7.30–9.67 s.
- Its actor is **TARGET_PLAYER**.
- Its outcome is **WON**.
- Stats: **duels = 1, duels_won = 1, passes = 0** (passes_attempted 0 and passes_completed 0).
- **No PASS COMPLETE** in 8.7–9.7 s. The false PASS was the 3K–3P failure; see the failure catalogue, F1 and F5.

### REFERENCE ONLY (do not tune toward these)
- Historical DUEL window 7.43–9.23 s.
- Historical DUEL confidence 0.858.
- Historical actor confidence 0.873.
- CARRY windows and confidences.
- The 6.53 s tracking-start POSSESSION_WON is a **known artifact**, so reproducing it is **not** required. Its absence would be an improvement.

### Known fragility (DOCUMENTED, status doc "Open risks")
- GOLD margins: closing 0.78 h against a 0.5 h minimum; ball locality 0.88 h against a 1.0 h maximum.
- In a ±30% sweep, 2 of 18 constants flip the verdict.
- Path B has only 1 positive and 2 negatives on real footage, so precision and recall are unmeasured.

A rebuilt engine reaching DUEL WON must not be presented as validated on the strength of GOLD alone.

---

## pass_completed_01

**Truth source.** Event truth (DOCUMENTED): PASS COMPLETE by White #26, Task 3A hand-check "carry from 1.06 s, kick at 1.66 s", label window 1.0–3.0 s (`PASS_COMPLETED_01_BENCHMARK_FIXTURE.json`).
- The receiver was judged CORRECT in Task 3N (TEAMMATE #13 area, 40.81 px, frame 161, score 0.646).
- No separate human-signed sign-off file was found for this clip (**DOCUMENTED, not human-signed in the GOLD sense**).

**Historical 3Q output** (`pc_postchange/pass_completed_01_player_analysis.json` 55b1ee7e…, hash-verified in R0):

| Event | Window (s) | Outcome | Event conf | Actor |
|---|---|---|---|---|
| CARRY | 1.06–1.66 | — | 0.926 HIGH | TARGET_PLAYER (1.0) |
| POSSESSION_WON | 1.06 | — | 0.926 | TARGET_PLAYER |
| **PASS** | **1.66–3.35** | **COMPLETE** | **0.804 HIGH** | **TARGET_PLAYER (1.0)** |
| POSSESSION_LOST | 3.35 | — | 0.804 | TARGET_PLAYER |

**Historical 3Q stats:** passes_attempted 1, passes_completed 1, carries 1, duels 0.

### MUST MATCH SEMANTICALLY LATER
- **PASS** with outcome **COMPLETE**.
- **Correct target** (White #26, the seed player).
- **Correct receiver**: the teammate actually receiving. A self-receiver or an opponent is a FAIL.
- **passes_attempted = 1, passes_completed = 1.**

### REFERENCE ONLY
- Historical PASS confidence 0.804.
- Window 1.66–3.35 s.
- Receiver distance 40.81 px.

### Re-runnability caveat (R0 finding)

**The pass_completed_01 source video (`f5bd8ce0-pass_completed_01.mov`, sha256 a2e45703…) was not found on the Mac.** Documents and Downloads were searched by name, by size (22,656,639 B) and inside zip listings.

Until it is recovered:
- this acceptance reference can be checked **only from Pass 2 onward**, by feeding the frozen 3N Pass-1 JSON (40fa0df1…), and **not end to end**;
- an end-to-end R5 run on it is impossible.

Other gaps:
- The seed bbox for this clip is circular: it was taken from an earlier Pass-1 output.
- The fixture's `tracking_artifact_sha256` points at a stale, misregistered JSON (aba3464b…).

---

## Other historical references, not acceptance targets

| Clip | Status | Use |
|---|---|---|
| duel_lost_01 | REJECTED (seed bystander; contest in progress at frame 0) | Negative evidence only |
| shot_close_range_01 | REJECTED (provenance mismatch) | Negative evidence only |
| src2_clip_003 | REJECTED (identity ambiguity) | Negative evidence only |
| Task 3R R01–R10 | FROZEN multi-clip benchmark. Labels v1/v2 exist; Q1–Q3 still open; scorer lost | Resume only after R6, per charter §48 |

**Acceptance base.** With only two positive references, one of which cannot be re-run end to end, the R5 acceptance base is **thin**. This was already a stated open risk before the code loss.

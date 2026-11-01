# SPANA Elite: build order and roadmap

**Status:** canonical roadmap for all future SPANA Elite development (Nathan, 2026-11-01 15:10).

**Scope:**
- This file sets **what** SPANA Elite must output and **in what order** it is built.
- The project operating instructions still govern **how**: generalisation, first failing layer, identity first,
  physical evidence before semantics, abstention, frozen truth before inference, no tuning on the holdout,
  persistence.
- Any change to this order needs Nathan's approval and a new dated entry in the change log at the end.

**Priority order (unchanged):**
1. correct physical identity;
2. correct physical evidence;
3. event semantics;
4. statistics.

Nothing at a later layer may compensate for a failure at an earlier one.

---

## 1. Core outputs (the target of the core build)

### 1.1 Core events

| Event | Minimum physical evidence before it may be emitted (else UNKNOWN / REVIEW) |
|---|---|
| **TOUCH** | Credible ball contact by the target: ball observed at the target's foot/body, with a ball-trajectory change or a stop |
| **RECEIVE** | A ball arriving from a release elsewhere, then controlled by the target (a touch followed by control, not a deflection) |
| **CARRY** | Sustained control: the ball stays within control distance of the target, who moves with it, for a minimum time (seconds) |
| **PASS** | A real release by the target, then the ball's travel, then a physically independent receiver. Outcome COMPLETE / INCOMPLETE / UNKNOWN |
| **SHOT** | Credible ball interaction plus trajectory evidence toward goal. Outcome includes GOAL where goal evidence exists (see §2.4 dependency) |
| **CROSS** | A release from a wide area into the box, judged from spatial context (§2.1). Until pitch context exists, a CROSS candidate is reported as PASS or UNKNOWN_RELEASE, never guessed |
| **DUEL** | A real opponent contest: closing, reach to the ball, a contest duration and an interruption of control. Outcome WON / LOST / UNKNOWN |
| **UNKNOWN_RELEASE** | A release whose type or receiver cannot be justified. This is a valid output, not a failure |
| **POSSESSION_WON** | The target gains sustained control from an opponent or a loose ball. Never at tracking start (known historical artifact) |
| **POSSESSION_LOST** | The target's possession ends with the opponent in control, or out of play, by the target's action |

### 1.2 Core player stats

Stats are a pure reducer over the event log. Eligibility rules:
- A stat counts an event only when the actor is the target **and** `identity_trusted` held for the decisive frames
  **and** the event passed its evidence and confidence rules.
- Review-required events are counted separately, never silently included (historical open risk: `stat_eligible`
  ignored `review_required`).
- Every percentage reports its numerator and denominator.
- A zero denominator gives **UNKNOWN**, not 0 %.

| Stat | Definition |
|---|---|
| touches | Eligible TOUCH events, plus the touch inside each RECEIVE / CARRY / release, counted once per physical contact (no double counting; rule documented in the stats module) |
| receptions | Eligible RECEIVE events |
| carries | Eligible CARRY events |
| passes attempted | Eligible PASS events (any outcome) |
| passes completed | Eligible PASS with outcome COMPLETE |
| pass completion % | passes completed ÷ passes with a decided outcome. UNKNOWN-outcome passes are reported alongside, not in the denominator (proposed; Nathan to confirm) |
| progressive passes | Eligible completed PASS classified progressive (§2.2) |
| crosses | Eligible CROSS events (completed count reported alongside) |
| shots | Eligible SHOT events (by outcome where known) |
| assists | Eligible assists (§2.4) |
| duels attempted | Eligible DUEL events (any outcome) |
| duels won | Eligible DUEL with outcome WON |
| duel win % | duels won ÷ duels with a decided outcome. UNKNOWN outcomes count as neither won nor lost (historical 3Q rule) and are reported alongside (proposed; Nathan to confirm) |
| possession won | Eligible POSSESSION_WON events |
| possession lost | Eligible POSSESSION_LOST events |

---

## 2. Build order

Stages run in this order. A stage starts only when the previous gate is met **and** Nathan has approved it.

| Stage | Content | Gate to leave the stage |
|---|---|---|
| R1 (done) | Selected-player tracker rebuild (`player_tracker_rb`) | Contract, synthetic and GOLD development checks |
| **R2 (current)** | Tracker DEV remediation (F-series), then the frozen-label EVAL with Nathan's approval | EVAL on the approved clean tracker commit: false confident identity, swaps, wrong reacquisition and coverage reported |
| R3 | The Pass-1 → Pass-2 boundary as a versioned artifact:<br>• ball detection and tracking (OBSERVED / PREDICTED / HELD / LOST states);<br>• physical interaction evidence;<br>• possession;<br>• nearby-player identity, with target exclusion (the target never appears twice) | Ball and possession benchmarks on frozen human truth |
| R4 | **Basic event detection** for the core events in §1.1:<br>• candidate generation, arbitration and actor attribution obeying `identity_trusted`;<br>• abstention as UNKNOWN / REVIEW | Per-event precision, recall and abstention on a frozen multi-clip event benchmark (development and evaluation clips separate) |
| R5 | **Core stats** (§1.2) and event reports; validation against human-labelled events and stats | **"Basic event detection is reliable"**: per-event precision and coverage meet targets recorded and frozen before the evaluation run; zero stats from untrusted identity |

### Build-order additions (only after the R5 gate: basic event detection reliable)

**2.1 Spatial / pitch context**
- Calibrated player and ball positions (image → pitch).
- Direction of play.
- Distance to the opponent goal.
- Requirements:
  - calibration quality is reported, and an uncalibrated frame gives UNKNOWN pitch position;
  - observed and inferred positions stay distinguished;
  - camera cuts and zoom are handled.

**2.2 Progressive pass classification**
- Applies to **completed** passes only.
- The rule is a meaningful reduction in the ball's distance to the opponent goal, from release point to reception
  point, in pitch units.
- **Not** every forward pass counts as progressive.
- The threshold must be generic, pitch-normalised and documented: formula, inputs, units, rationale and calibration
  status.
- Thresholds are **UNKNOWN** until proposed with evidence in this stage, and fixed before any evaluation.
- A pass with an UNKNOWN pitch position is **not** classified as progressive.

**2.3 Sequence / outcome logic**
- Event chains: connect passes and crosses to the next receptions, shots and goals.
- Possession continuity is tracked across the chain.
- Each chain link records its evidence and its break reason, for example:
  - opponent control;
  - DUEL LOST;
  - out of play;
  - a meaningful gap;
  - an unverified receiver.

**2.4 Assist detection**
- Credit a **completed** pass or cross that **directly** leads to a goal: the receiver is the scorer and the shot
  ends in a goal.
- **No assist** if possession is meaningfully broken between the pass/cross and the goal, for example:
  - opponent control;
  - DUEL lost;
  - a loose ball not regained by the scorer;
  - out of play.
- The break rules are documented and generic.
- Actor attribution must still obey `identity_trusted`: the assister is credited only when the target's identity
  was trusted at the release.
- The scorer must be a physically independent teammate who is not the target (physical-entity exclusion). Naming
  the scorer is not required.
- **Dependency:** SHOT outcome GOAL needs goal evidence (ball over the line / in the net, or a restart signal).
  Until it exists, an assist is UNKNOWN, never inferred.

**2.5 Canonical outputs**
- Add progressive passes and assists to the canonical stats and event reports.
- This is a versioned schema change: increment the version, document the migration, keep it additive.

### Standing rules for every stage

- Each new event or stat has the following:
  - a documented definition;
  - a frozen human-truth benchmark created **before** inference;
  - unit, synthetic and real-footage tests;
  - confidence values with a documented formula, kept separate per kind (detection / identity / event / actor /
    outcome).
- Report both error rate and coverage/abstention.
- Use seconds and pitch- or body-height-normalised units; no clip-, player-, team- or venue-specific logic.
- Pass 1 (tracking) and Pass 2 (events) stay decoupled behind a versioned artifact.

---

## Change log

- **2026-11-01 15:10.** Nathan set the core events (§1.1), core player stats (§1.2), and the build-order additions
  after basic event detection is reliable (§2.1–2.5). He asked for this file to be kept as the roadmap for all
  future SPANA Elite development.
- The stage table R1–R5 records the existing plan, as described across the R0 documents (R3/R4 Pass 2 / event
  engine, R5 validation).
- Items marked "proposed; Nathan to confirm" are defaults, not decisions.

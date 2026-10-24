"""
event_candidates.py -- Task 5 ("Event Vocabulary / Arbitration Layer").

BUILDS ONE-CANDIDATE-EACH evidence for PASS, CROSS, SHOT, DUEL and UNKNOWN,
as separate, explainable, independently-scored objects (item #3/#15). This
module NEVER decides the final event type itself -- that is
event_arbitration.py's job. It also never mutates or reruns anything
upstream: it consumes only the SAME already-computed, frozen values
shot_candidate.py itself already reads (ball_tracker.history,
possession_frames, pitch_context, keeper_candidate) plus release_geometry.py/
goalkeeper_context.py's existing public functions, read-only, exactly as
shot_candidate.py already imports and calls them (rule #16: reuse, don't
rewrite, upstream frozen modules).

CORE PRINCIPLE (item #3): SHOT is one candidate among several, not "ball was
released and nothing else could be proven". This module's SHOT-candidate
builder does not re-derive SHOT evidence from scratch -- it takes the
frozen shot_candidate.evaluate_release()'s own Event as an INPUT and turns
it into a comparable score, because rule #16 says that engine's existing
temporal/evidence logic must remain the (only) source of SHOT evidence.
What changes is that its output no longer wins automatically: it is
weighed against CROSS/PASS/DUEL/UNKNOWN evidence built here from the exact
same underlying frame data.

DISCLOSED LIMITATION (parallels release_geometry.py's own CALIBRATION
HIERARCHY): none of these validation clips has ever produced real detected
goal posts (visual_pitch_calibration always fell back to the coarse global
net-drift PitchContext for all three named clips traced for this task --
"spatial_context_source": "global_net_drift_fallback"/"none"). That means
release_geometry.py's own Level A/B origin-width estimate is almost always
unavailable in practice for this dataset (Level C, wide_score=None). Rather
than pretend that's good enough evidence for CROSS, or invent a new,
un-corroborated pixel heuristic to replace it (exactly what release_
geometry.py's own docstring already forbids doing for the wide-origin
signal), CROSS's builder here uses a SEPARATE, always-available multi-signal
fallback (controlled-delivery speed, non-keeper contest identity, moderate
goal-directedness) explicitly capped at a lower confidence than a true
origin-width read would carry, and is honest when even that is inconclusive
(a CROSS score near zero, not a forced guess).
"""
import math
from dataclasses import dataclass, field
from typing import Optional, List

from .geometry import dist, center
from .pitch_context import goal_direction_score, distance_to_goal
from .goalkeeper_context import keeper_evidence_at
from .release_geometry import estimate_release_width, nearby_ball_centre

# ---------------------------------------------------------------- constants
# All thresholds below are expressed in PIXELS-PER-SECOND (px/frame * fps)
# wherever they compare a speed, specifically so they generalise across this
# project's very different clip frame rates (the three named clips for this
# task run at ~59fps; several of the existing regression clips run at
# ~25fps) rather than silently only being valid at one fps -- a general
# lesson from this task's own dataset, not a per-clip fix.

CONTESTED_INCOMING_SPEED_PX_S = 3000.0   # above this, the ball arriving at the selected player reads as a
                                           # loose/contested ball (a deflection, a flick-on, a 50-50), not a
                                           # controlled reception -- duel_lost_01's own incoming speed
                                           # (~20,600 px/s) sits an order of magnitude above this; cross_low_01
                                           # (~570 px/s) and missed_pass_01 (~160 px/s) sit two orders below it
CONTROLLED_DELIVERY_SPEED_PX_S = 1500.0   # below this, the ball's ARRIVAL at the selected player reads as a
                                           # deliberate, controlled touch (consistent with a player about to
                                           # play a considered CROSS or PASS), not a scramble

DUEL_MAX_TOUCH_FRAMES = 2                 # a possession segment this short (a single-frame or two-frame
                                           # touch) before release is itself weak circumstantial evidence of
                                           # a contested first-time challenge rather than a controlled take
DUEL_OPPONENT_RADIUS_PX = 100.0           # wider than shot_candidate.py's own RECEPTION_RADIUS (55px) --
                                           # deliberately generous, because ball tracking is at its LEAST
                                           # precise exactly during a genuine physical duel (crowding/partial
                                           # occlusion), so a strict reception-grade radius systematically
                                           # under-counts real contests (this is the concrete, evidence-based
                                           # failure this task's diagnosis found in duel_lost_01: a real
                                           # opponent sat 68px from the ball at the exact release frame --
                                           # just outside the 55px reception gate -- and was discarded)
DUEL_CONTACT_WINDOW_FRAMES = 3            # how many frames either side of the touch/release frame count as
                                           # "at the moment of contact" for duel opponent-proximity evidence
                                           # (item #8: "both close to ball... converge on ball... physical
                                           # overlap around touch" -- a tight window around contact, not the
                                           # whole multi-second release-candidate window a SHOT/PASS/CROSS
                                           # evaluates)

PASS_INTENT_RADIUS_PX = 150.0             # a generous "plausible target corridor" radius (item #12) -- wider
                                           # than shot_candidate.py's strict 55px reception gate, because this
                                           # candidate's whole point is to give PASS credit WITHOUT requiring
                                           # a confirmed reception (item #11)

CROSS_GSCORE_BAND_LOW = 0.35              # a cross-shaped delivery must show at least some goal/box-ward
CROSS_GSCORE_BAND_HIGH = 0.85             # travel direction, but NOT the near-perfect goal-centre alignment
                                           # that is much more consistent with a direct strike at goal --
                                           # deliberately a BAND, not "the higher the better" (item #7: goal-
                                           # directed geometry ALONE must not be read as SHOT evidence, but an
                                           # extremely tight alignment is still more shot-like than cross-like)
LOW_KEEPER_IDENTITY_CONFIDENCE = 0.15     # at/below this, "a defender contested this ball" carries no honest
                                           # evidence that the contesting person was anything like a keeper
CROSS_MIN_SUPPORTIVE_WIDE_SCORE = 0.45    # release_geometry.py's own CENTRAL/HALF_SPACE boundary (CENTRAL_MAX_M
                                           # / WIDE_SCORE_SATURATION_M) -- a wide_score below this reads as a
                                           # CENTRAL release origin, not a wide one, so it must not itself count
                                           # as cross-supporting evidence just because a real measurement exists
MIN_CROSS_DISPLACEMENT_PX = 150.0         # a contest with no keeper-identity evidence only supports CROSS once
                                           # the ball has actually travelled a real delivery distance from the
                                           # release point -- a contest right at the release point with no
                                           # keeper evidence reads as a BLOCK on a shot, not an intercepted cross

UNKNOWN_BASELINE_SCORE = 0.35             # always-available fallback candidate (item #13) -- deliberately a
UNKNOWN_BASELINE_CONFIDENCE = 0.5         # fixed, moderate constant so genuinely undistinguished releases
                                           # settle here instead of being forced into any specific type


@dataclass
class Candidate:
    """One event-type's independently-scored evidence packet (item #15).
    `score`: 0-1, how strongly the evidence on hand supports this type.
    `confidence`: 0-1, how much of the evidence this score is actually built
        from is itself trustworthy/available (never conflated with score
        itself -- a candidate can have a moderate score built from strong
        evidence, or the same moderate score built from almost no evidence
        at all; `confidence` is what tells those two apart, exactly the
        distinction shot_candidate.py's own composite evidence-fusion
        already draws between value and confidence).
    `effective_score`: score damped by confidence -- what arbitration
        actually compares candidates on (item #15's "not simple highest-
        number-wins... when confidence/coverage differ materially").
    `supporting`/`contradictory`: short human-readable evidence strings.
    `evidence`: the full structured evidence dict, for reporting."""
    event_type: str
    score: float
    confidence: float
    supporting: List[str] = field(default_factory=list)
    contradictory: List[str] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)

    @property
    def effective_score(self):
        return round(max(0.0, min(1.0, self.score)) * (0.5 + 0.5 * max(0.0, min(1.0, self.confidence))), 4)

    def to_dict(self):
        return {
            "event_type": self.event_type, "score": round(self.score, 4),
            "confidence": round(self.confidence, 4), "effective_score": self.effective_score,
            "supporting": self.supporting, "contradictory": self.contradictory, "evidence": self.evidence,
        }


# ----------------------------------------------------------- SHOT candidate
# shot_candidate.py's evaluate_release() is FROZEN (item #16/#17) -- this
# function never recomputes SHOT evidence, it only reads the Event it
# already produced and converts it into a comparable (score, confidence)
# pair, optionally cross-checking it against ONE additional, independent,
# already-existing signal it did not itself consult on this code path
# (goalkeeper identity confidence at the contest point) -- never a new
# detector, never a changed threshold inside shot_candidate.py itself.
def build_shot_candidate(shot_side_event, ball_tracker, pitch_context, keeper_candidate, i_end):
    ev = shot_side_event.evidence
    reason = ev.get("resolution_reason", "")
    supporting, contradictory = [], []

    if shot_side_event.type != "SHOT":
        # the frozen SHOT engine itself did not call this a shot at all --
        # weak residual SHOT evidence only (still reported, never zero,
        # since UNKNOWN_RELEASE sometimes reflects a borderline case that
        # DID show some fast/straight motion, just not enough to confirm)
        score = 0.15
        confidence = 0.3
        contradictory.append(f"shot_candidate.py's own frozen evaluation returned {shot_side_event.type} "
                              f"({reason}), not SHOT")
        return Candidate("SHOT", score, confidence, supporting, contradictory,
                          {"frozen_resolution_reason": reason, "frozen_event_type": shot_side_event.type})

    release_pos = nearby_ball_centre(ball_tracker, i_end)
    contest_pos = nearby_ball_centre(ball_tracker, shot_side_event.end_frame)

    if reason == "SHOT_CONFIRMED: resolved_by_pitch_context":
        composite = ev.get("composite_shot_evidence") or 0.0
        coverage = ev.get("evidence_coverage") or 0.0
        score = min(0.95, 0.5 + 0.4 * composite)
        confidence = max(0.3, coverage)
        supporting.append(f"frozen composite spatial evidence resolved this SHOT with composite={composite}, "
                           f"evidence_coverage={coverage}")

    elif reason == "SHOT_CONFIRMED: contested_by_opponent_during_purposeful_release":
        base = 0.55
        keeper_ev = keeper_evidence_at(contest_pos, keeper_candidate) if contest_pos else None
        contest_displacement = dist(release_pos, contest_pos) if (release_pos and contest_pos) else None
        far_contest = contest_displacement is not None and contest_displacement >= MIN_CROSS_DISPLACEMENT_PX
        if (keeper_ev is None or keeper_ev.identity_confidence < LOW_KEEPER_IDENTITY_CONFIDENCE) and far_contest:
            # lacking keeper-identity evidence only meaningfully undercuts
            # 'contested/blocked/saved SHOT' framing when the contest
            # happened far enough from the release to raise the genuine
            # question 'was this really saved near goal, or just intercepted
            # out wide' -- a contest immediately at the release point (a
            # block in the shooting lane) does not require the blocker to be
            # a keeper by definition, so it is not dampened on that basis
            # (item #24.W's block-vs-ambiguous-far-contest distinction)
            base *= 0.75
            contradictory.append("the contesting person at the point of contact carries little to no evidence "
                                  "of being a goalkeeper, and the contest happened far from the release point -- "
                                  "'contested/blocked/saved shot' framing is weaker without at least some "
                                  "keeper-identity evidence at that distance")
        elif keeper_ev is not None and keeper_ev.identity_confidence >= LOW_KEEPER_IDENTITY_CONFIDENCE:
            base = min(0.9, base + 0.25 * keeper_ev.identity_confidence * keeper_ev.proximity_value)
            supporting.append(f"contest point carries real goalkeeper-identity evidence "
                               f"(identity_confidence={keeper_ev.identity_confidence}, "
                               f"proximity={keeper_ev.proximity_value})")
        else:
            supporting.append("contest happened immediately at the release point -- reads as a block in the "
                               "shooting lane, which does not require the blocker to be a keeper")

        gscore = goal_direction_score(release_pos, [contest_pos], pitch_context) if release_pos else None
        keeper_identity_conf = keeper_ev.identity_confidence if keeper_ev is not None else 0.0
        if gscore is not None:
            base *= (0.6 + 0.4 * gscore)
            (supporting if gscore >= 0.5 else contradictory).append(f"trajectory-to-goal alignment gscore={gscore}")
        else:
            contradictory.append("no goal-direction evidence available at all for this release")
        # confidence (how much of this branch's evidence is actually
        # available/trustworthy, item #15) scales with how much real,
        # corroborating evidence exists -- a strong keeper-identity read
        # AND a real trajectory-to-goal measurement together earn much
        # higher confidence than either alone or neither at all
        coverage = 0.3 + 0.4 * keeper_identity_conf + 0.3 * (1.0 if gscore is not None else 0.0)
        score = base
        confidence = coverage

    else:
        # bottom fallback path: no reception of any kind found anywhere in
        # the window -- SHOT_CONFIRMED purely from accumulated speed/
        # straightness/evidence-mass (this is exactly the "SHOT as a
        # catch-all for any unclaimed release" pattern this task diagnosed
        # as duel_lost_01's own root cause -- see this module's docstring)
        evidence_mass = ev.get("evidence_mass") or 0.0
        straightness = ev.get("straightness") or 0.0
        mass_component = min(1.0, evidence_mass / 3.0)   # shot_candidate.CONFIRM_EVIDENCE_MASS, duplicated as a
                                                            # plain literal here rather than importing a private
                                                            # constant across modules -- see note below
        base = min(0.92, 0.4 + 0.3 * mass_component + 0.2 * straightness)
        supporting.append(f"no receiver of any kind found anywhere in the candidate window; accumulated "
                           f"evidence_mass={evidence_mass}, straightness={straightness}")
        gscore = ev.get("goal_direction_score")
        if gscore is not None:
            base *= (0.6 + 0.4 * gscore)
            (supporting if gscore >= 0.5 else contradictory).append(f"trajectory-to-goal alignment gscore={gscore}")
            coverage = 0.55
        else:
            coverage = 0.35
            contradictory.append("no goal-direction evidence available at all for this release")
        score = base
        confidence = coverage

    return Candidate("SHOT", score, confidence, supporting, contradictory,
                      {"frozen_resolution_reason": reason, "release_pos": release_pos, "contest_pos": contest_pos})


def apply_contact_ambiguity_dampener(shot_candidate_obj, opponent_distance_at_contact):
    """NEW evidence-fusion rule living entirely in this arbitration layer
    (never inside frozen shot_candidate.py -- item #16): when an opponent
    was within DUEL_OPPONENT_RADIUS_PX of the ball at the moment of contact
    but NOT within shot_candidate.py's own stricter 55px reception gate,
    the frozen 'fallback, no-receiver-found-anywhere' SHOT path's framing
    ('nobody contested this release') is undermined by evidence that path
    structurally cannot see. This does not touch shot_candidate.py's own
    threshold or logic; it only tempers how much weight ITS OUTPUT carries
    in arbitration once an independent signal it never consulted
    contradicts its own implicit premise."""
    if shot_candidate_obj.evidence.get("frozen_resolution_reason") != "SHOT_CONFIRMED":
        return shot_candidate_obj
    if opponent_distance_at_contact is None:
        return shot_candidate_obj
    shot_candidate_obj.score *= 0.65
    shot_candidate_obj.contradictory.append(
        f"an opponent was {opponent_distance_at_contact:.1f}px from the ball at the moment of contact -- outside "
        f"shot_candidate.py's own strict reception gate, but close enough that 'no one contested this release' "
        f"is not a safe premise; dampened accordingly")
    return shot_candidate_obj


def apply_pass_intent_dampener(shot_candidate_obj, nearest_teammate_distance, gscore):
    """A second, independent NEW evidence-fusion rule (item #16: living
    here, never inside frozen shot_candidate.py): the fallback 'no receiver
    found anywhere, fast+straight => SHOT' path also structurally ignores a
    teammate who was near the ball but outside the strict 55px reception
    radius (item #11/#12's whole point -- an unconfirmed-but-plausible pass
    target). When such a teammate exists AND there is no strong independent
    goal-directed evidence backing the SHOT reading, the fallback path's
    'nothing else claimed this release' premise is weaker than it assumes."""
    if shot_candidate_obj.evidence.get("frozen_resolution_reason") != "SHOT_CONFIRMED":
        return shot_candidate_obj
    if nearest_teammate_distance is None or nearest_teammate_distance > PASS_INTENT_RADIUS_PX:
        return shot_candidate_obj
    if gscore is not None and gscore >= CROSS_GSCORE_BAND_HIGH:
        # strong, independent goal-directed evidence -- the nearby teammate
        # does not meaningfully undercut a genuinely goal-ward strike
        return shot_candidate_obj
    shot_candidate_obj.score *= 0.7
    shot_candidate_obj.contradictory.append(
        f"a teammate was {nearest_teammate_distance:.1f}px from the ball during this release, and there is no "
        f"strong independent goal-directed evidence -- 'nothing else claimed this release' is not a safe premise "
        f"when a plausible pass target existed nearby; dampened accordingly")
    return shot_candidate_obj


# ---------------------------------------------------------------- CROSS candidate
def build_cross_candidate(release_pos, contest_or_end_pos, pitch_context, keeper_candidate,
                           incoming_speed_px_s, receiver_evidence_present):
    supporting, contradictory = [], []
    if receiver_evidence_present:
        contradictory.append("a teammate reception was already confirmed for this release -- a confirmed, "
                              "completed reception is handled directly as PASS, not evaluated as a cross")
        return Candidate("CROSS", 0.05, 0.5, supporting, contradictory, {})

    wide_est = estimate_release_width(release_pos, pitch_context)
    gscore = goal_direction_score(release_pos, [contest_or_end_pos], pitch_context) if release_pos else None
    keeper_ev = keeper_evidence_at(contest_or_end_pos, keeper_candidate) if contest_or_end_pos else None
    low_keeper_conf = (keeper_ev is None) or (keeper_ev.identity_confidence < LOW_KEEPER_IDENTITY_CONFIDENCE)
    controlled = incoming_speed_px_s is not None and incoming_speed_px_s < CONTROLLED_DELIVERY_SPEED_PX_S

    if wide_est.wide_score is not None:
        # LEVEL A/B origin-width geometry actually available -- the real,
        # disclosed-confidence signal item #6 asks for. FIX (found via the
        # known_match_shot_195_223 real-clip regression check, item #17):
        # the base score here must actually scale with how wide the release
        # origin reads -- a release measured as CENTRAL (low wide_score) is
        # not itself cross-supporting evidence merely because a real
        # measurement exists; only a genuinely wide origin is. The old flat
        # 0.35 floor treated "we have real geometry" as if it were "the
        # geometry says wide", which is not the same claim.
        origin_component = wide_est.wide_score * wide_est.confidence
        score = 0.15 + 0.65 * origin_component
        confidence = max(0.4, wide_est.confidence)
        if wide_est.wide_score >= CROSS_MIN_SUPPORTIVE_WIDE_SCORE:
            supporting.append(f"release origin geometry ({wide_est.method}) reads {wide_est.zone} "
                               f"(wide_score={wide_est.wide_score}, confidence={wide_est.confidence})")
        else:
            contradictory.append(f"release origin geometry ({wide_est.method}) reads {wide_est.zone} -- "
                                  f"not actually a wide release position (wide_score={wide_est.wide_score}), "
                                  f"which is weak support for a cross reading on origin alone")
        if gscore is not None and CROSS_GSCORE_BAND_LOW <= gscore <= CROSS_GSCORE_BAND_HIGH:
            score += 0.20
            supporting.append(f"trajectory heads toward the box without the near-perfect goal-centre alignment "
                               f"more typical of a direct strike (gscore={gscore})")
        elif gscore is not None and gscore > CROSS_GSCORE_BAND_HIGH:
            # same distinction the LEVEL C fallback branch below already
            # makes -- near-perfect goal-centre alignment is more
            # consistent with a direct strike than a delivery across the
            # box, whether or not real origin-width geometry exists
            score *= 0.5
            contradictory.append(f"trajectory shows near-perfect goal-centre alignment (gscore={gscore}) -- "
                                  f"more consistent with a direct strike at goal than a delivery across the "
                                  f"box, even with real origin-width geometry available")
        return Candidate("CROSS", min(1.0, score), confidence, supporting, contradictory,
                          {"wide_release_geometry": wide_est.as_dict(), "goal_direction_score": gscore})

    # LEVEL C fallback (item's own disclosed limitation -- see module
    # docstring): no origin-width geometry at all. Built entirely from
    # independent signals shot_candidate.py's own "purposeful" gate never
    # consults: was the ARRIVAL at the player controlled (not a scramble),
    # does the contest point carry any goalkeeper-identity evidence at all,
    # and does the trajectory show at least moderate (not extreme) goal-ward
    # travel. Requiring gscore to exist AND sit in the disclosed band gates
    # this candidate off entirely when there is no spatial context at all
    # (e.g. missed_pass_01, where pitch_context.goal_centre is None) --
    # never a forced guess (rule: None means nothing to say, not zero).
    if gscore is None:
        contradictory.append("no goal-direction evidence available at all -- cannot honestly support a cross "
                              "reading without at least knowing the ball travelled toward the attacking area")
        return Candidate("CROSS", 0.05, 0.15, supporting, contradictory,
                          {"wide_release_geometry": wide_est.as_dict(), "goal_direction_score": None})

    contest_displacement = dist(release_pos, contest_or_end_pos) if (release_pos and contest_or_end_pos) else None
    far_enough_for_delivery = contest_displacement is not None and contest_displacement >= MIN_CROSS_DISPLACEMENT_PX

    score = 0.25
    if controlled and far_enough_for_delivery:
        # a "controlled arrival" only supports CROSS once the ball has
        # actually travelled a real delivery distance -- a controlled touch
        # immediately blocked at the release point is not evidence of a
        # delivery, whatever its incoming speed was (matches the same
        # far_enough_for_delivery gate the low-keeper-identity bonus below
        # uses, and for the same reason -- item #24.W's block case)
        score += 0.20
        supporting.append(f"ball arrived under control (incoming speed {incoming_speed_px_s:.0f}px/s) and "
                           f"travelled a real delivery distance ({contest_displacement:.0f}px) -- consistent "
                           f"with a deliberate delivery, not a scramble or an immediate block")
    elif not controlled:
        contradictory.append(f"ball arrived at high speed (incoming {incoming_speed_px_s:.0f}px/s) -- more "
                              f"consistent with a contested/loose ball than a deliberate cross")
    elif not far_enough_for_delivery:
        contradictory.append("contest happened too close to the release point to read as an actual delivery")
    if low_keeper_conf and far_enough_for_delivery:
        # a contest with no keeper-identity evidence is only meaningful
        # CROSS-supporting evidence once the ball has actually travelled a
        # real delivery distance (item #7/#9's own distinction) -- a
        # contest right at the release point with no keeper evidence reads
        # as a BLOCK on a shot (item #24.W), not an intercepted cross, and
        # must not earn this bonus
        score += 0.20
        supporting.append(f"the contesting person at the point of contact ({contest_displacement:.0f}px from "
                           f"release) carries little to no goalkeeper-identity evidence -- more consistent with "
                           f"a cross intercepted/cleared by an outfield defender than a shot saved by a keeper")
    elif low_keeper_conf:
        contradictory.append(f"contest happened close to the release point itself "
                              f"({contest_displacement:.0f}px away)" if contest_displacement is not None else
                              "no contest point available" + " -- reads more like a block on a shot than an "
                              "intercepted delivery, despite the lack of keeper-identity evidence")
    if CROSS_GSCORE_BAND_LOW <= gscore <= CROSS_GSCORE_BAND_HIGH:
        score += 0.15
        supporting.append(f"trajectory heads toward the box at a moderate (not extreme) goal-ward alignment "
                           f"(gscore={gscore}) -- extreme alignment reads more like a direct strike")
    elif gscore > CROSS_GSCORE_BAND_HIGH:
        contradictory.append(f"trajectory shows near-perfect goal-centre alignment (gscore={gscore}) -- more "
                              f"consistent with a direct strike at goal than a delivery across the box")

    return Candidate("CROSS", min(1.0, score), 0.5, supporting, contradictory,
                      {"wide_release_geometry": wide_est.as_dict(), "goal_direction_score": gscore,
                       "keeper_evidence": (keeper_ev.__dict__ if keeper_ev else None),
                       "incoming_speed_px_s": incoming_speed_px_s, "method": "level_c_fallback_composite"})


# ----------------------------------------------------------- PASS-intent candidate
# PASS-EVIDENCE-FIX CONSTANTS (this task; none of these touch SHOT/CROSS/DUEL/UNKNOWN_RELEASE
# scoring, thresholds, or the UNKNOWN_RELEASE baseline -- item #22/#23/#24 -- they only shape
# how much credit THIS candidate gives to evidence it did not honestly see before):
UNCLASSIFIED_TEAM_PASS_DAMPENER = 0.55   # item #17: an UNKNOWN-team-labelled player near the ball is
                                           # weak, UNCERTAIN support for a pass reception -- never full
                                           # TEAMMATE-strength credit, and never treated as OPPONENT/
                                           # contradictory evidence just because identity is uncertain
UNCLASSIFIED_TEAM_PASS_CONFIDENCE_CAP = 0.35
RECEIVER_CANDIDATE_BASE_SCORE = 0.45     # item #6/#11: shot_candidate.py's own frozen temporal
RECEIVER_CANDIDATE_SCORE_GAIN = 0.45     # reception scorer already found a genuine (if inconclusive)
                                           # TEAMMATE candidate -- "was this a pass?" is a different,
                                           # and here better-evidenced, question than "did it complete?"
EXTENDED_WINDOW_PROXIMITY_DAMPENER = 0.75   # a proximity found ONLY past shot_candidate.py's own
                                              # SHOT-side window (see event_arbitration.resolve_release's
                                              # fallback logic, which additionally requires a SUSTAINED
                                              # multi-frame approach before this path is even reachable --
                                              # see _sustained_pass_receiver_search) still carries none of
                                              # shot_candidate.py's own temporal-control corroboration, so
                                              # it is damped below a same-window finding even once sustained


def build_pass_intent_candidate(nearest_teammate_distance, nearest_unknown_distance, incoming_speed_px_s, gscore,
                                 receiver_evidence_status, receiver_evidence_detail, used_extended_window=False):
    """item #11/#12: PASS intent WITHOUT requiring a confirmed reception --
    an incomplete pass must remain a PASS. Built from 'was there a plausible
    teammate (or, per item #17, an uncertainly-classified) target nearby',
    independent of whether the ball actually reached them, PLUS (this task's
    fix) shot_candidate.py's own richer temporal-reception evidence when it
    exists but fell short of a full confirmation.

    receiver_evidence_status: None (shot_candidate.py never found a TEAMMATE
        anywhere within its own strict reception radius in the window) |
        "RECEIVER_CANDIDATE" | "RECEIVER_REJECTED". RECEIVER_CONFIRMED never
        reaches this function at all -- event_arbitration.resolve_release's
        own authoritative short-circuit already returns a completed PASS
        directly for that case, before any candidate is built (rule #16:
        that existing, extensively-validated logic is untouched).
    receiver_evidence_detail: shot_candidate.ReceiverEvidence.to_dict() when
        receiver_evidence_status is not None, else None. Read-only: this
        candidate never recomputes any of shot_candidate.py's own reception
        scoring, it only uses the numbers that scoring already produced.
    nearest_teammate_distance/nearest_unknown_distance: independent per-team
        nearest-to-ball minimums over PASS's own (wider, see
        event_arbitration.PASS_RECEIVER_SEARCH_FLOOR_S) search window --
        never the legacy any-team-wins field CROSS/SHOT-dampening still use
        unchanged elsewhere in this module/event_arbitration.py."""
    supporting, contradictory = [], []

    if receiver_evidence_status == "RECEIVER_CANDIDATE" and receiver_evidence_detail is not None:
        # A genuine, real teammate candidate that shot_candidate.py's OWN frozen temporal-
        # control scoring already evaluated and found substantial (if inconclusive)
        # evidence for -- distinct from, and stronger than, mere unconfirmed proximity.
        reception_score = receiver_evidence_detail.get("reception_score") or 0.0
        distance_trend = receiver_evidence_detail.get("distance_trend_score") or 0.0
        trajectory_change = receiver_evidence_detail.get("trajectory_change_score") or 0.0
        min_dist = receiver_evidence_detail.get("minimum_ball_distance")
        score = RECEIVER_CANDIDATE_BASE_SCORE + RECEIVER_CANDIDATE_SCORE_GAIN * reception_score
        supporting.append(f"shot_candidate.py's own frozen temporal reception scoring already found a genuine "
                           f"TEAMMATE candidate {min_dist}px from the ball (reception_score={round(reception_score, 3)}, "
                           f"distance_trend={round(distance_trend, 3)}, trajectory_change={round(trajectory_change, 3)}) "
                           f"-- ambiguous as a CONFIRMED completion, but real, substantial pass-INTENT evidence in "
                           f"its own right (item #6: 'was this a pass?' is a different question from 'did it "
                           f"succeed?')")
        if gscore is not None and gscore >= CROSS_GSCORE_BAND_HIGH:
            score *= 0.75
            contradictory.append(f"trajectory shows strong goal-ward alignment (gscore={gscore}), in some tension "
                                  f"with a pass intended for a nearby teammate")
        confidence = 0.55 + 0.25 * reception_score
        pass_intent_score = round(min(1.0, score), 4)
        components = {"pass_intent_score": pass_intent_score, "receiver_likelihood": round(reception_score, 4),
                       "reception_confidence": round(distance_trend, 4),
                       "pass_completion_evidence": "AMBIGUOUS: RECEIVER_CANDIDATE (not confirmed, not rejected)",
                       "pass_final_candidate_score": pass_intent_score}
        return Candidate("PASS", min(1.0, score), min(1.0, confidence), supporting, contradictory,
                          {"receiver_evidence_status": receiver_evidence_status,
                           "receiver_evidence": receiver_evidence_detail, "goal_direction_score": gscore,
                           "components": components})

    if receiver_evidence_status == "RECEIVER_REJECTED":
        # A candidate reception existed but shot_candidate.py's own scoring actively
        # rejected it (rule #8: e.g. the ball went functionally LOST right after, or the
        # other signals were weak) -- this is real, negative evidence about COMPLETION,
        # but (item #6) says nothing definitive about INTENT, so it falls through to the
        # same proximity-based intent scoring below rather than an automatic near-zero
        # score (which is what conflating the two questions produced before this fix).
        contradictory.append("a candidate teammate reception was evaluated and rejected by shot_candidate.py's "
                              "own temporal-control scoring (not confirmed as a completion) -- this candidate "
                              "still asks the separate question of whether a pass was plausibly INTENDED")

    # Pick whichever candidate actually sat closer to the ball, not "TEAMMATE whenever one
    # exists at all" -- a confidently-classified teammate on the far side of the pitch
    # (present, but essentially irrelevant to this specific release) must not outrank an
    # unclassified-team player standing right at the ball; ties favour TEAMMATE, the
    # stronger identity evidence of the two.
    best_distance, best_kind = None, None
    if nearest_teammate_distance is not None and (nearest_unknown_distance is None or
                                                    nearest_teammate_distance <= nearest_unknown_distance):
        best_distance, best_kind = nearest_teammate_distance, "TEAMMATE"
    elif nearest_unknown_distance is not None:
        best_distance, best_kind = nearest_unknown_distance, "UNKNOWN"

    if best_distance is None:
        contradictory.append("no teammate, and no unclassified-team player either, detected anywhere near the "
                              "ball during this release at all")
        components = {"pass_intent_score": 0.05, "receiver_likelihood": 0.0, "reception_confidence": 0.0,
                       "pass_completion_evidence": "NONE", "pass_final_candidate_score": 0.05}
        return Candidate("PASS", 0.05, 0.3, supporting, contradictory,
                          {"nearest_teammate_distance": None, "nearest_unknown_distance": None,
                           "receiver_evidence_status": receiver_evidence_status, "components": components})

    proximity = max(0.0, 1.0 - best_distance / PASS_INTENT_RADIUS_PX)
    control_factor = 1.0
    if incoming_speed_px_s is not None and incoming_speed_px_s >= CONTESTED_INCOMING_SPEED_PX_S:
        control_factor = 0.3
        contradictory.append(f"the touch preceding this release itself arrived as a contested/loose ball "
                              f"(incoming {incoming_speed_px_s:.0f}px/s) -- weaker evidence of deliberate pass intent")

    score = 0.25 + 0.5 * proximity * control_factor
    confidence = 0.6 if best_distance <= PASS_INTENT_RADIUS_PX else 0.3
    if best_kind == "UNKNOWN":
        # item #17: uncertain team identity is weak, damped support -- never full TEAMMATE
        # credit, and never treated as contradictory/opponent evidence merely for being
        # unclassified.
        score *= UNCLASSIFIED_TEAM_PASS_DAMPENER
        confidence = min(confidence, UNCLASSIFIED_TEAM_PASS_CONFIDENCE_CAP)
        supporting.append(f"an unclassified-team player (team identity uncertain at this frame -- not confidently "
                           f"OPPONENT either) was {best_distance:.1f}px from the ball during this release -- weak, "
                           f"uncertain support for a pass reception, damped accordingly")
    else:
        supporting.append(f"a teammate was {best_distance:.1f}px from the ball during this release -- "
                           f"a plausible pass target, even without a confirmed reception")

    if gscore is not None and gscore >= CROSS_GSCORE_BAND_HIGH:
        score *= 0.7
        contradictory.append(f"trajectory shows strong goal-ward alignment (gscore={gscore}), in some tension "
                              f"with a pass intended for a nearby teammate")
    elif gscore is None:
        score = min(1.0, score + 0.05)   # no competing goal-directed evidence at all -- mild, not decisive, support

    if used_extended_window:
        score *= EXTENDED_WINDOW_PROXIMITY_DAMPENER
        confidence = min(confidence, 0.35)
        contradictory.append("this proximity was found only by searching well past shot_candidate.py's own "
                              "SHOT-side window -- it carries none of that window's own temporal-control "
                              "corroboration (distance-closing trend, trajectory change, control persistence), "
                              "so it is damped well below a same-window finding")

    pass_intent_score = round(min(1.0, score), 4)
    components = {"pass_intent_score": pass_intent_score, "receiver_likelihood": round(proximity, 4),
                   "reception_confidence": 0.0,
                   "pass_completion_evidence": "NONE (unconfirmed proximity only)",
                   "pass_final_candidate_score": pass_intent_score}
    return Candidate("PASS", min(1.0, score), confidence, supporting, contradictory,
                      {"nearest_teammate_distance": nearest_teammate_distance,
                       "nearest_unknown_distance": nearest_unknown_distance, "nearest_kind_used": best_kind,
                       "incoming_speed_px_s": incoming_speed_px_s, "goal_direction_score": gscore,
                       "receiver_evidence_status": receiver_evidence_status, "components": components})


# ----------------------------------------------------------------- DUEL candidate
def build_duel_candidate(incoming_speed_px_s, segment_frames, opponent_distance_at_contact):
    """item #8: ground/physical ball duel ONLY. Evidence: the ball arrived
    as a contested/loose ball (very high incoming speed), the preceding
    touch was very short (consistent with a first-time challenge rather
    than a controlled take), and an OPPONENT was near the ball at the
    moment of contact -- using a WIDER radius than shot_candidate.py's
    strict reception gate (see DUEL_OPPONENT_RADIUS_PX's rationale above),
    since ball tracking is least precise exactly during a genuine physical
    contest."""
    supporting, contradictory = [], []
    score = 0.0
    have_incoming, have_opponent = False, False

    controlled_arrival = False
    if incoming_speed_px_s is not None:
        have_incoming = True
        if incoming_speed_px_s >= CONTESTED_INCOMING_SPEED_PX_S:
            score += 0.45
            supporting.append(f"ball arrived as a contested/loose ball (incoming speed "
                               f"{incoming_speed_px_s:.0f}px/s) rather than a controlled reception")
        else:
            contradictory.append(f"ball arrived under control (incoming speed {incoming_speed_px_s:.0f}px/s), "
                                  f"not as a loose/contested ball")
            if incoming_speed_px_s < CONTROLLED_DELIVERY_SPEED_PX_S:
                controlled_arrival = True

    if segment_frames is not None and segment_frames <= DUEL_MAX_TOUCH_FRAMES:
        score += 0.15
        supporting.append(f"only {segment_frames} frame(s) of possession before release -- consistent with a "
                           f"first-time contested touch, not a controlled take")

    if opponent_distance_at_contact is not None:
        have_opponent = True
        if opponent_distance_at_contact <= DUEL_OPPONENT_RADIUS_PX:
            proximity = max(0.0, 1.0 - opponent_distance_at_contact / DUEL_OPPONENT_RADIUS_PX)
            score += 0.4 * proximity
            supporting.append(f"an opponent was {opponent_distance_at_contact:.1f}px from the ball at the "
                               f"moment of contact")
        else:
            contradictory.append(f"nearest opponent at the moment of contact was {opponent_distance_at_contact:.1f}"
                                  f"px away -- beyond a generous contested-ball radius")
    else:
        contradictory.append("no opponent detected near the ball at the moment of contact at all")

    if not have_opponent:
        # a duel fundamentally requires SOME opponent evidence -- without
        # it, this is not a genuine ground/physical duel candidate no
        # matter how fast/short the touch was (item #9: never force DUEL)
        score *= 0.25

    if controlled_arrival:
        # a clearly CONTROLLED arrival is real, independent evidence against
        # a genuine contested duel, not merely "one component didn't fire" --
        # an opponent standing near the ball during an unhurried, controlled
        # touch is marking/positioning, not a 50/50 challenge, however close
        # they are (item #24.W's block-vs-duel distinction: this is what
        # keeps a shot blocked immediately after a controlled touch from
        # being misread as a duel just because someone was close by)
        score *= 0.35
        contradictory.append("the controlled (non-contested) arrival is itself strong evidence against a "
                              "genuine physical duel, regardless of opponent proximity")

    confidence = 0.3 + 0.3 * have_incoming + 0.4 * have_opponent
    return Candidate("DUEL", min(1.0, score), min(1.0, confidence), supporting, contradictory,
                      {"incoming_speed_px_s": incoming_speed_px_s, "segment_frames": segment_frames,
                       "opponent_distance_at_contact": opponent_distance_at_contact})


def build_unknown_candidate():
    return Candidate("UNKNOWN_RELEASE", UNKNOWN_BASELINE_SCORE, UNKNOWN_BASELINE_CONFIDENCE,
                      ["always-available baseline: nothing else clearly distinguishes this release"], [], {})

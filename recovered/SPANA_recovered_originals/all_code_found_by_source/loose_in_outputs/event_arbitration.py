"""
event_arbitration.py -- Task 5's new entry point for resolving a release.

Architecture (item #4):
    POSSESSION -> TOUCH/INTERACTION -> RELEASE OR CONTEST
        -> EVENT CANDIDATES (PASS, SHOT, CROSS, DUEL, UNKNOWN)
        -> ARBITRATION -> FINAL EVENT

This module is the ARBITRATION step. It calls the frozen shot_candidate.
evaluate_release() exactly once (item #16: reuse its existing temporal/
evidence logic unchanged, never re-implement it) and treats its result as
one candidate's evidence source, builds independent PASS-intent/CROSS/DUEL/
UNKNOWN candidates from the same underlying frame data (event_candidates.py),
then picks a final event only when the evidence genuinely distinguishes the
candidates (item #15) -- never a flat highest-number-wins comparison when
confidence/coverage differ materially.

INTEGRATION POINT (item #16's "strictly necessary integration code"):
event_qualifier.py's qualify_release() now calls resolve_release() here
instead of evaluate_release() directly, and event_detector.py's finalize()
passes two additional, already-computed values through the same call
(seg_start, prior_segment_evidence) so this module can read the PRECEDING
possession segment's own already-computed incoming-speed/segment-length
evidence (event_classifier.py's own EventClassifier.classify_segment()
output) without recomputing or duplicating that logic. Nothing about how
that preceding evidence is computed changes."""
from dataclasses import asdict

from .event_classifier import Event
from .shot_candidate import evaluate_release
from .geometry import dist
from .pitch_context import goal_direction_score
from .release_geometry import nearby_ball_centre
from .event_candidates import (
    build_shot_candidate, apply_contact_ambiguity_dampener, apply_pass_intent_dampener, build_cross_candidate,
    build_pass_intent_candidate, build_duel_candidate, build_unknown_candidate,
    DUEL_CONTACT_WINDOW_FRAMES,
)

ARBITRATION_MARGIN = 0.12        # winner must beat the runner-up's effective_score by at least this much
ARBITRATION_MIN_ACCEPT = 0.35    # winner's own effective_score must clear this floor too -- otherwise even
                                   # an uncontested "winner" is too weak on its own evidence to assert confidently

# PASS-EVIDENCE FIX (this task): how much further PAST shot_candidate.py's own SHOT-
# confirmation candidate_timeout_frames the PASS candidate's OWN receiver search keeps
# looking for a plausible teammate/unclassified-team target (item #9/#13). That timeout is
# sized for a different purpose entirely -- confirming/expiring a SHOT candidate -- not for
# how long a genuine pass (particularly a lobbed one) may take to actually reach a receiver.
# Root-cause note (validation-batch instrumented trace, pass_completed_03): the real
# receiving teammate closed to 10px of the ball (well inside shot_candidate.py's own 55px
# reception radius) but only ~1.0s AFTER the existing candidate_timeout_frames-derived
# window had already closed. This constant ONLY widens what PASS's OWN teammate/unknown-
# team search is allowed to see; it never changes candidate_timeout_frames itself, never
# changes hi_cap's use for CROSS/SHOT-dampening lookups below (both still use the original,
# frozen-adjacent hi_cap, byte-identical to before), and never touches shot_candidate.py.
PASS_RECEIVER_SEARCH_FLOOR_S = 3.0


def _nearest_by_team_in_range(possession_frames, lo, hi, team):
    best_d = None
    for k in range(max(0, lo), min(hi, len(possession_frames) - 1) + 1):
        pf = possession_frames[k]
        if pf.nearest_other_team == team and pf.nearest_other_distance is not None:
            if best_d is None or pf.nearest_other_distance < best_d:
                best_d = pf.nearest_other_distance
    return best_d


PASS_SUSTAINED_MIN_CONSECUTIVE_FRAMES = 3   # how many consecutive close frames the EXTENDED-
                                              # window fallback requires before trusting a
                                              # proximity reading at all (see its call site's
                                              # docstring note) -- a single-frame blip is not
                                              # accepted, a genuine multi-frame approach is
PASS_SUSTAINED_RADIUS_PX = 150.0   # matches event_candidates.PASS_INTENT_RADIUS_PX's own
                                     # plausible-target-corridor radius (duplicated as a plain
                                     # literal rather than imported, same pattern event_
                                     # candidates.py itself already uses for a private constant
                                     # shared across these two arbitration-layer modules)


def _sustained_pass_receiver_search(possession_frames, lo, hi):
    """Only used as the PASS candidate's EXTENDED-window fallback (see resolve_release) --
    the original (SHOT-side) window still uses the plain, single-best-frame
    _pass_receiver_search_in_range below unchanged. Returns (teammate_d, unknown_d): the
    minimum distance seen during the best run of >= PASS_SUSTAINED_MIN_CONSECUTIVE_FRAMES
    consecutive in-range frames for each kind, or None if no such run exists."""
    def best_sustained(get_distance):
        best, streak_min, streak_len = None, None, 0
        for k in range(max(0, lo), min(hi, len(possession_frames) - 1) + 1):
            d = get_distance(possession_frames[k])
            if d is not None and d <= PASS_SUSTAINED_RADIUS_PX:
                streak_len += 1
                streak_min = d if streak_min is None else min(streak_min, d)
                if streak_len >= PASS_SUSTAINED_MIN_CONSECUTIVE_FRAMES:
                    if best is None or streak_min < best:
                        best = streak_min
            else:
                streak_len, streak_min = 0, None
        return best
    return (best_sustained(lambda pf: pf.nearest_teammate_distance),
            best_sustained(lambda pf: pf.nearest_unknown_distance))


def _pass_receiver_search_in_range(possession_frames, lo, hi):
    """PASS-EVIDENCE FIX (this task): independent per-team nearest-to-ball minimums over
    [lo, hi], read from the NEW additive PossessionFrame.nearest_teammate_distance/
    nearest_unknown_distance fields (see possession_detector.py) -- never the legacy
    nearest_other_team/nearest_other_distance fields _nearest_by_team_in_range above still
    uses unchanged for DUEL's opponent-at-contact evidence and SHOT's own dampening input,
    both deliberately left untouched by this task."""
    best_teammate_d, best_unknown_d = None, None
    for k in range(max(0, lo), min(hi, len(possession_frames) - 1) + 1):
        pf = possession_frames[k]
        td = pf.nearest_teammate_distance
        if td is None and pf.nearest_other_team == "TEAMMATE":
            # Backward-compatible fallback for any caller/fixture built before this task's
            # additive fields existed (e.g. test_event_arbitration.py's hand-built
            # PossessionFrame timelines, which only set the legacy nearest_other_team/
            # nearest_other_distance pair). This is NEVER a behaviour change for real
            # production data: whenever nearest_other_team == "TEAMMATE" there, the two
            # values are already identical (both come from the same closest-teammate box
            # in that case -- see event_detector.py's process_frame), so this fallback only
            # ever fires when the new field was genuinely never populated at all.
            td = pf.nearest_other_distance
        if td is not None and (best_teammate_d is None or td < best_teammate_d):
            best_teammate_d = td
        ud = pf.nearest_unknown_distance
        if ud is None and pf.nearest_other_team == "UNKNOWN":
            ud = pf.nearest_other_distance
        if ud is not None and (best_unknown_d is None or ud < best_unknown_d):
            best_unknown_d = ud
    return best_teammate_d, best_unknown_d


def resolve_release(ball_tracker, possession_frames, i_end, fps=25.0, log=None,
                     pitch_context=None, keeper_candidate=None, seg_start=None, prior_segment_evidence=None):
    """Drop-in replacement call shape for EventQualifier.qualify_release's
    old direct evaluate_release() call -- same required args, two new
    optional ones. Returns a single Event (PASS | SHOT | CROSS | DUEL |
    UNKNOWN_RELEASE), with a full `arbitration` evidence block explaining
    every candidate considered (item #15/#27)."""
    shot_side_event = evaluate_release(ball_tracker, possession_frames, i_end, fps=fps, log=log,
                                        pitch_context=pitch_context, keeper_candidate=keeper_candidate)

    # AUTHORITATIVE SHORT-CIRCUIT: a genuinely CONFIRMED teammate reception
    # (temporal control evidence, not mere proximity -- shot_candidate.py's
    # own rules #3-#15) is unambiguous, direct ball-to-player evidence.
    # Item #16 protects this existing, extensively-validated logic; there is
    # no honest CROSS/DUEL reading that should ever override a genuinely
    # confirmed reception, so arbitration is skipped entirely here rather
    # than manufacturing competing evidence against a settled fact.
    if shot_side_event.type == "PASS" and shot_side_event.evidence.get("resolution_reason") == \
            "PASS_CONFIRMED: teammate_receiver":
        shot_side_event.evidence["arbitration"] = {
            "short_circuited": True,
            "reason": "confirmed teammate reception (temporal control evidence) is authoritative",
        }
        return shot_side_event

    receiver_evidence_present = shot_side_event.evidence.get("receiver_evidence") is not None
    n = len(ball_tracker.history)
    timeout_frames = shot_side_event.evidence.get("candidate_timeout_frames")
    hi_cap = min(n - 1, i_end + timeout_frames) if timeout_frames is not None else shot_side_event.end_frame

    release_pos = nearby_ball_centre(ball_tracker, i_end)
    # "contest/end" position: wherever the release-candidate window's own
    # evidence says the ball ended up interacting with someone else, or (if
    # no one was ever found within any radius) simply where the ball was by
    # the end of the window -- used identically by both the SHOT-candidate
    # keeper-identity check and the CROSS candidate, so both compare against
    # the same point.
    contest_or_end_pos = nearby_ball_centre(ball_tracker, shot_side_event.end_frame)

    incoming_speed_px_s = None
    segment_frames = None
    if prior_segment_evidence is not None:
        incoming_px_frame = prior_segment_evidence.get("incoming_speed_px_per_frame")
        if incoming_px_frame is not None:
            incoming_speed_px_s = incoming_px_frame * fps
        segment_frames = prior_segment_evidence.get("segment_frames")

    opponent_distance_at_contact = _nearest_by_team_in_range(
        possession_frames, i_end - DUEL_CONTACT_WINDOW_FRAMES, i_end + DUEL_CONTACT_WINDOW_FRAMES, "OPPONENT")
    # UNCHANGED (still feeds only apply_pass_intent_dampener's existing SHOT-side dampening,
    # frozen-adjacent, byte-identical to before this task -- see that function's docstring).
    nearest_teammate_distance = _nearest_by_team_in_range(possession_frames, i_end, hi_cap, "TEAMMATE")

    gscore = goal_direction_score(release_pos, [contest_or_end_pos], pitch_context) if release_pos else None

    shot_cand = build_shot_candidate(shot_side_event, ball_tracker, pitch_context, keeper_candidate, i_end)
    shot_cand = apply_contact_ambiguity_dampener(shot_cand, opponent_distance_at_contact)
    shot_cand = apply_pass_intent_dampener(shot_cand, nearest_teammate_distance, gscore)
    cross_cand = build_cross_candidate(release_pos, contest_or_end_pos, pitch_context, keeper_candidate,
                                        incoming_speed_px_s, receiver_evidence_present)

    # PASS-EVIDENCE FIX (this task): PASS's OWN receiver search uses a wider window
    # (PASS_RECEIVER_SEARCH_FLOOR_S) and the new per-team-split fields (see
    # _pass_receiver_search_in_range's docstring) -- deliberately NOT the same
    # nearest_teammate_distance/hi_cap the SHOT-side dampener above uses, so this change is
    # fully isolated to the PASS candidate. receiver_evidence_status/detail let the PASS
    # candidate use shot_candidate.py's own (frozen, unmodified) temporal reception scoring
    # as strong-but-inconclusive pass-intent support when it found a genuine RECEIVER_
    # CANDIDATE (item #6's "was this a pass?" vs "did it succeed?" distinction) instead of
    # discarding that evidence entirely just because it fell short of RECEIVER_CONFIRMED.
    # FIX (found via this task's own real-clip regression check, shot_close_range_03):
    # the wider window is a FALLBACK, not an unconditional wider search. A proximity
    # found only PAST the original (SHOT-side, frozen-timeout-derived) hi_cap has none of
    # shot_candidate.py's own corroborating temporal signals to back it (its own
    # _teammate_reception_evidence never even looks that far, so receiver_evidence_status
    # is always None for anything found only out there) -- it is just as consistent with
    # an unrelated player eventually walking near a since-dead ball as with a genuine late
    # reception. So: search the ORIGINAL window first, exactly as SHOT's own dampener
    # input does; only fall back to the wider window when the original one found NOTHING
    # at all, and mark that fallback explicitly so build_pass_intent_candidate can weight
    # it lower than a same-window finding (item #9's temporal search, without manufacturing
    # unearned confidence for a stale-ball coincidence).
    teammate_d_orig, unknown_d_orig = _pass_receiver_search_in_range(possession_frames, i_end, hi_cap)
    used_extended_window = False
    nearest_teammate_distance_for_pass, nearest_unknown_distance_for_pass = teammate_d_orig, unknown_d_orig
    # "Nothing meaningful in the original window" means no per-team candidate within the
    # PASS-intent radius at all -- a real (if far, irrelevant) teammate elsewhere on the
    # pitch must not itself block the sustained-fallback search below from ever running.
    orig_has_meaningful_evidence = ((teammate_d_orig is not None and teammate_d_orig <= PASS_SUSTAINED_RADIUS_PX) or
                                     (unknown_d_orig is not None and unknown_d_orig <= PASS_SUSTAINED_RADIUS_PX))
    if not orig_has_meaningful_evidence and timeout_frames is not None:
        pass_hi_cap = min(n - 1, i_end + max(timeout_frames, int(round(PASS_RECEIVER_SEARCH_FLOOR_S * fps))))
        if pass_hi_cap > hi_cap:
            # Require a SUSTAINED approach (several consecutive close frames), not a single
            # frame's proximity -- a genuine late reception shows a player closing in and
            # staying close (this task's pass_completed_03 regression-check evidence: 10-36px
            # for 8 straight frames), which is exactly what a single-frame coincidental
            # overlap with a since-dead ball (this task's shot_close_range_03 regression
            # finding: one isolated close frame, no run of them) does not show.
            ext_teammate_d, ext_unknown_d = _sustained_pass_receiver_search(
                possession_frames, hi_cap + 1, pass_hi_cap)
            if ext_teammate_d is not None or ext_unknown_d is not None:
                nearest_teammate_distance_for_pass, nearest_unknown_distance_for_pass = ext_teammate_d, ext_unknown_d
                used_extended_window = True
    receiver_evidence_detail = shot_side_event.evidence.get("receiver_evidence")
    receiver_evidence_status = receiver_evidence_detail.get("status") if receiver_evidence_detail else None

    pass_cand = build_pass_intent_candidate(nearest_teammate_distance_for_pass, nearest_unknown_distance_for_pass,
                                             incoming_speed_px_s, gscore,
                                             receiver_evidence_status, receiver_evidence_detail,
                                             used_extended_window=used_extended_window)
    duel_cand = build_duel_candidate(incoming_speed_px_s, segment_frames, opponent_distance_at_contact)
    unknown_cand = build_unknown_candidate()

    candidates = [shot_cand, cross_cand, pass_cand, duel_cand, unknown_cand]
    ranked = sorted(candidates, key=lambda c: c.effective_score, reverse=True)
    winner, runner_up = ranked[0], ranked[1]

    arbitration_evidence = {
        "candidates": {c.event_type: c.to_dict() for c in candidates},
        "ranked": [c.event_type for c in ranked],
        "margin": round(winner.effective_score - runner_up.effective_score, 4),
        "winner_effective_score": winner.effective_score,
        "runner_up_effective_score": runner_up.effective_score,
        "shot_side_frozen_event_type": shot_side_event.type,
        "shot_side_frozen_resolution_reason": shot_side_event.evidence.get("resolution_reason"),
    }

    review_required = (winner.effective_score - runner_up.effective_score < ARBITRATION_MARGIN or
                        winner.effective_score < ARBITRATION_MIN_ACCEPT)

    base_evidence = dict(shot_side_event.evidence)
    base_evidence["arbitration"] = arbitration_evidence

    if review_required:
        arbitration_evidence["decision"] = "REVIEW_REQUIRED: top candidates too close or too weak"
        base_evidence["resolution_reason"] = "REVIEW_REQUIRED: " + \
            f"{winner.event_type}({winner.effective_score}) vs {runner_up.event_type}({runner_up.effective_score})"
        return Event("UNKNOWN_RELEASE", i_end, shot_side_event.end_frame, success=None,
                      evidence=base_evidence, qualifiers=["REVIEW_REQUIRED"])

    arbitration_evidence["decision"] = f"{winner.event_type} wins by margin {arbitration_evidence['margin']}"

    if winner.event_type == "SHOT":
        # unchanged frozen SHOT event -- arbitration agreed with it, so its
        # own start/end/evidence/success are returned exactly as evaluate_
        # release produced them, with the arbitration block merged in for
        # explainability (item #15) -- no change to its own decision fields.
        shot_side_event.evidence["arbitration"] = arbitration_evidence
        return shot_side_event

    if winner.event_type == "CROSS":
        base_evidence["cross_outcome_basis"] = "receiver_evidence" if receiver_evidence_present else "no_confirmed_receiver"
        success = True if receiver_evidence_present else None if shot_side_event.type == "UNKNOWN_RELEASE" else False
        base_evidence["resolution_reason"] = "CROSS_CONFIRMED: arbitration"
        return Event("CROSS", i_end, shot_side_event.end_frame, success=success, evidence=base_evidence)

    if winner.event_type == "PASS":
        base_evidence["resolution_reason"] = "PASS_CONFIRMED: arbitration_pass_intent_without_reception"
        base_evidence["note"] = ("no confirmed reception exists, but pass-intent evidence (a plausible nearby "
                                  "teammate target, absent strong competing SHOT/CROSS evidence) outweighed the "
                                  "alternatives -- reported as an INCOMPLETE pass attempt (item #11: lack of a "
                                  "successful receiver does not erase the pass attempt itself)")
        return Event("PASS", i_end, shot_side_event.end_frame, success=False, evidence=base_evidence)

    if winner.event_type == "DUEL":
        outcome = _resolve_duel_outcome(possession_frames, i_end, hi_cap, opponent_distance_at_contact, fps)
        base_evidence["resolution_reason"] = f"DUEL_{outcome}: arbitration"
        success = True if outcome == "WON" else False if outcome == "LOST" else None
        return Event("DUEL", i_end, shot_side_event.end_frame, success=success,
                      evidence=base_evidence, qualifiers=[f"DUEL_{outcome}"])

    base_evidence["resolution_reason"] = "UNKNOWN_RELEASE: arbitration_no_candidate_won"
    return Event("UNKNOWN_RELEASE", i_end, shot_side_event.end_frame, success=None, evidence=base_evidence)


DUEL_RECOVERY_WINDOW_FRAMES_S = 1.5   # how long after the contest to look for the SAME tracked player
                                        # re-establishing close possession, before concluding LOST/UNKNOWN


def _resolve_duel_outcome(possession_frames, i_end, hi_cap, opponent_distance_at_contact, fps=25.0):
    """item #9: WON when the selected player regains controlled possession
    shortly after the contest, LOST when an opponent was the one who
    contested the ball and the selected player never regains it, UNKNOWN
    otherwise. Never forces a winner/loser (item #9's explicit instruction)."""
    n = len(possession_frames)
    recovery_frames = max(6, int(round(DUEL_RECOVERY_WINDOW_FRAMES_S * fps)))
    lo, hi = hi_cap + 1, min(n - 1, hi_cap + recovery_frames)
    recaptured = False
    for k in range(lo, hi + 1):
        pf = possession_frames[k]
        if pf.tracking_state == "TRACKING" and pf.ball_distance is not None and pf.ball_distance <= 40.0:
            recaptured = True
            break
    if recaptured:
        return "WON"
    if opponent_distance_at_contact is not None:
        return "LOST"
    return "UNKNOWN"

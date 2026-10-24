"""
confidence.py -- bands an Event's evidence into HIGH/MEDIUM/LOW and decides
review_required. Same conservative weighting rationale as the rest of this
project: ball-detection recall is measured at ~10-20% per frame, and only
the tracked player has verified identity, so outcome-bearing tags (PASS
success/failure, SHOT) are held to a higher bar than position-only tags
(TOUCH, RECEIVE, CARRY). Per the user's stated preference -- 95% precision
over 95% recall -- this rounds DOWN when in doubt.
"""

HIGH_THRESH = 0.75
MEDIUM_THRESH = 0.5

POSITION_DRIVEN = {"TOUCH", "RECEIVE", "CARRY"}
# EVENT VOCABULARY / ARBITRATION TASK: CROSS and DUEL are now genuinely
# emitted event types (event_arbitration.py), not just reserved vocabulary
# -- both are outcome-bearing (CROSS: COMPLETE/INCOMPLETE; DUEL: WON/LOST/
# UNKNOWN) exactly like PASS/SHOT, so both belong in OUTCOME_DRIVEN.
# UNKNOWN_RELEASE and GOAL always carry success=None, so the existing
# OUTCOME_DRIVEN rule below (review_required = band != HIGH or success is
# None) makes them ALWAYS review_required -- exactly the "never confidently
# assert" behaviour both need. An undetermined DUEL/CROSS outcome
# (success=None) gets the same treatment, unchanged in spirit.
OUTCOME_DRIVEN = {"PASS", "SHOT", "CROSS", "DUEL", "UNKNOWN_RELEASE", "GOAL"}


def _tracking_quality(evidence):
    states = evidence.get("tracking_states", {}) or {}
    total = sum(states.values()) or 1
    good = states.get("TRACKING", 0)
    occ = states.get("OCCLUDED", 0)
    return (good + 0.5 * occ) / total


def _ball_quality(evidence):
    """Weights the new 5-state ball model per rule #16: CONFIRMED carries
    full weight, REACQUIRING (plausible, stability-gated) nearly full,
    PREDICTED (pure motion-model extrapolation) reduced weight, and
    OCCLUDED/LOST (no position at all) contribute nothing."""
    states = evidence.get("ball_states", {}) or {}
    total = sum(states.values()) or 1
    confirmed = states.get("CONFIRMED", 0)
    reacquiring = states.get("REACQUIRING", 0)
    predicted = states.get("PREDICTED", 0)
    return (confirmed + 0.8 * reacquiring + 0.4 * predicted) / total


def _team_quality(evidence):
    sim = evidence.get("team_sim")
    if sim is None:
        return 0.4
    return max(0.0, min(1.0, sim))


class ConfidenceEngine:
    def score(self, event):
        ev = event.evidence
        tq = _tracking_quality(ev)
        bq = _ball_quality(ev)

        if event.type in POSITION_DRIVEN:
            raw = 0.75 * tq + 0.25 * bq
        else:
            teamq = _team_quality(ev)
            raw = 0.35 * tq + 0.45 * bq + 0.20 * teamq
        raw = max(0.0, min(1.0, raw))

        band = "HIGH" if raw >= HIGH_THRESH else ("MEDIUM" if raw >= MEDIUM_THRESH else "LOW")

        if event.type in OUTCOME_DRIVEN:
            review_required = band != "HIGH" or event.success is None
        else:
            review_required = band == "LOW"

        ev["confidence_score"] = round(raw, 3)
        ev["confidence_band"] = band
        ev["review_required"] = review_required
        return raw, band, review_required

#!/usr/bin/env python3
"""test_ab_event_matcher.py -- STRUCTURAL unit tests for
ab_event_matcher.py's matching logic, using entirely synthetic, clearly-
fake event dicts. This is NOT Annotator B data, fabricated or otherwise --
it exists only to prove match_clip() implements the pre-registered
algorithm correctly (timing tolerances, one-to-one assignment, the
CARRY-never-competes rule) before any real B submission exists. Per the
task: "Do NOT generate B's answers. Do NOT ask an LLM to simulate B" --
these fixtures are deliberately abstract/round-numbered so they cannot be
mistaken for a real clip's content, and none of them corresponds to any of
the 10 real calibration clips.

Run: python3 test_ab_event_matcher.py
"""
from ab_event_matcher import match_clip


def _discrete(event_type, t, outcome=None, qualifiers=None):
    return {"event_type": event_type, "interaction_time": t, "outcome": outcome,
            "qualifiers": qualifiers or [], "start_time": None, "end_time": None}


def _span(event_type, s, e, outcome=None):
    return {"event_type": event_type, "start_time": s, "end_time": e, "outcome": outcome,
            "qualifiers": [], "interaction_time": None}


def test_discrete_match_within_tolerance():
    a = [_discrete("PASS", 100.0, "COMPLETE")]
    b = [_discrete("PASS", 100.4, "COMPLETE")]  # B stored clip-relative; ctx_start=0 for this synthetic test
    r = match_clip(a, b, ctx_start=0.0)
    assert len(r["matches"]) == 1, r
    assert r["matches"][0]["timing_error_s"] == 0.4
    assert not r["unmatched_a"] and not r["unmatched_b"]


def test_discrete_no_match_outside_tolerance():
    a = [_discrete("SHOT", 50.0, "SAVED")]
    b = [_discrete("SHOT", 52.0, "SAVED")]  # 2.0s > 1.0s tolerance
    r = match_clip(a, b, ctx_start=0.0)
    assert len(r["matches"]) == 0
    assert len(r["unmatched_a"]) == 1 and len(r["unmatched_b"]) == 1


def test_carry_never_competes():
    """A: [BALL_RECOVERY, PASS], B: [PASS] only -- the PASS events should
    match; BALL_RECOVERY is an honest miss (recall error), never a reason
    to also fail the PASS match. Mirrors the task's own worked example."""
    a = [_discrete("BALL_RECOVERY", 10.0), _discrete("PASS", 16.1, "COMPLETE")]
    b = [_discrete("PASS", 16.4, "COMPLETE")]
    r = match_clip(a, b, ctx_start=0.0)
    assert len(r["matches"]) == 1
    assert r["matches"][0]["event_type"] == "PASS"
    assert len(r["unmatched_a"]) == 1 and r["unmatched_a"][0]["event_type"] == "BALL_RECOVERY"
    assert len(r["unmatched_b"]) == 0


def test_spanning_overlap_match():
    a = [_span("DUEL", 20.0, 22.0, "WON")]
    b = [_span("DUEL", 21.0, 23.0, "WON")]  # overlaps 21-22
    r = match_clip(a, b, ctx_start=0.0)
    assert len(r["matches"]) == 1
    assert r["matches"][0]["kind"] == "span"


def test_spanning_close_but_no_overlap_still_matches_within_tolerance():
    a = [_span("TAKE_ON", 5.0, 6.0, "SUCCESSFUL")]
    b = [_span("TAKE_ON", 7.5, 8.5, "SUCCESSFUL")]  # gap of 1.5s < 2.0s tolerance on both edges? check: |5-7.5|=2.5 >2.0 -> should NOT match
    r = match_clip(a, b, ctx_start=0.0)
    assert len(r["matches"]) == 0, "edges differ by 2.5s > 2.0s tolerance -- must not match"


def test_one_to_one_assignment_prefers_lowest_error():
    a = [_discrete("PASS", 10.0, "COMPLETE"), _discrete("PASS", 10.9, "COMPLETE")]
    b = [_discrete("PASS", 10.05, "COMPLETE")]
    r = match_clip(a, b, ctx_start=0.0)
    assert len(r["matches"]) == 1
    assert r["matches"][0]["a_index"] == 0  # the closer of the two A events wins the single B event
    assert len(r["unmatched_a"]) == 1 and r["unmatched_a"][0] is a[1]


def test_ctx_start_offset_applied_to_b_only():
    """B's tool exports clip-relative time; A's records are absolute.
    ctx_start bridges them -- verifies the conversion direction is correct."""
    a = [_discrete("SHOT", 1005.0, "GOAL")]   # absolute
    b = [_discrete("SHOT", 5.2, "GOAL")]       # clip-relative; ctx_start=1000.0 -> absolute 1005.2
    r = match_clip(a, b, ctx_start=1000.0)
    assert len(r["matches"]) == 1
    assert abs(r["matches"][0]["timing_error_s"] - 0.2) < 1e-6


def test_outcome_disagreement_still_matches_but_flagged():
    a = [_discrete("TACKLE", 40.0, "WON")]
    b = [_discrete("TACKLE", 40.1, "LOST")]
    r = match_clip(a, b, ctx_start=0.0)
    assert len(r["matches"]) == 1
    assert r["matches"][0]["outcome_agreement"] is False


def run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    passed = 0
    for t in tests:
        t()
        passed += 1
        print(f"PASS: {t.__name__}")
    print(f"\n{passed}/{len(tests)} structural tests passed.")


if __name__ == "__main__":
    run_all()

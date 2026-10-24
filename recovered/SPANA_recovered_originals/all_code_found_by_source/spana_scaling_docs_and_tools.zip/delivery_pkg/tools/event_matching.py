"""
event_matching.py -- shared logic for matching a human ground-truth
annotation window to the FROZEN pipeline's own event output for the same
clip. Used by both feature_exporter.py (to find the release event whose
arbitration evidence becomes `heuristic_scores`) and pipeline_benchmark.py
(to compare the pipeline's prediction against the human label). One
implementation, not two, so "how do we decide which pipeline event
corresponds to this annotation" is answered identically in both places.

This module contains no football-analysis logic of its own -- it only reads
already-produced event dicts (each shaped like ClipTagger.tag()'s
`events[i]`, with clip-relative start_frame/end_frame) and an annotation's
already-computed frame_safety block, and picks the best temporal overlap.
"""
from __future__ import annotations

from typing import Optional


def best_matching_event(events: list, gt_start_frame_clip_relative: int,
                         gt_end_frame_clip_relative: int) -> Optional[dict]:
    """Returns the event dict (from a CaptureResult.result["events"] /
    ClipTagger.tag() output list) whose [start_frame, end_frame] overlaps the
    human ground-truth window the most (by overlapping-frame count), with
    ties broken by whichever event's start_frame is closest to the ground
    truth window's start. Returns None if no event overlaps at all -- that
    is itself meaningful signal (the pipeline produced no event anywhere
    near where the human says one occurred) and callers must not silently
    substitute the nearest non-overlapping event for it.
    """
    best = None
    best_overlap = 0
    best_start_gap = None
    for e in events:
        s, en = e["start_frame"], e["end_frame"]
        overlap = max(0, min(en, gt_end_frame_clip_relative) - max(s, gt_start_frame_clip_relative))
        if overlap <= 0:
            continue
        start_gap = abs(s - gt_start_frame_clip_relative)
        if best is None or overlap > best_overlap or (overlap == best_overlap and start_gap < best_start_gap):
            best, best_overlap, best_start_gap = e, overlap, start_gap
    return best


def nearest_event_by_start(events: list, target_frame_clip_relative: int) -> Optional[dict]:
    """Fallback matcher: nearest event by start_frame, regardless of overlap
    -- used only for diagnostic/benchmark reporting of "closest miss," never
    to manufacture a heuristic_scores block for feature export (an export
    with no true overlap gets heuristic_scores=null fields, not a
    best-guess neighbour's scores)."""
    if not events:
        return None
    return min(events, key=lambda e: abs(e["start_frame"] - target_frame_clip_relative))

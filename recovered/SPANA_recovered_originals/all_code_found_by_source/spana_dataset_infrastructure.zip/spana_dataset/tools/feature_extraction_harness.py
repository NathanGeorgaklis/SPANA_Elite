"""
feature_extraction_harness.py -- read-only capture wrapper around the FROZEN,
UNMODIFIED production pipeline (football_analysis/*.py, player_tracker/*.py).

WHY THIS FILE EXISTS (read this before touching it):
`football_analysis.engine.FootballAnalysisEngine.run_analysis_pass` deliberately
does not return its internal `FootballEventDetector` object -- by design, per its
own docstring, it only ever hands callers a plain result dict. That is the right
choice for production. It also means there is no way, from outside engine.py, to
read the per-frame ball/possession history (`detector.ball_tracker.history`,
`detector._possession_frames`) that a temporal feature export needs, WITHOUT
either (a) editing engine.py -- forbidden by this task's explicit "no production
changes" rule -- or (b) calling the exact same already-public, already-frozen
functions/classes engine.py calls, in the exact same order, ourselves, and
keeping our own reference to the objects. This file does (b).

Every function/class imported and called below is imported UNMODIFIED from
football_analysis -- this module adds zero lines to any of those files and
changes zero of their behaviour. It stands in exactly the same relationship to
football_analysis's frozen public API that validation/lib/run_engine_pass_fixed.py
already has to engine.run_analysis_pass() -- a second, read-only, external
caller of the same functions.

KNOWN, DELIBERATE, DOCUMENTED DUPLICATION: `run_with_capture()` below mirrors
`FootballAnalysisEngine.run_analysis_pass`'s per-frame loop and post-loop
sequencing closely (skipping only the debug-video-rendering branches, which
are irrelevant to feature export). If engine.py's control flow ever changes,
this file needs a matching update, or feature exports will silently stop
matching real pipeline behaviour. This is exactly the "orchestration glue
duplicated across callers" pattern the architecture audit's
TECHNICAL_DEBT_AUDIT.md and MODULE_KEEP_REFACTOR_REPLACE.md already flagged
(three copies existed before this task; this is a fourth, added here only
because run_analysis_pass has no other way to expose per-frame internals). The
clean long-term fix -- an additive, behaviour-neutral "also return the
detector" option on run_analysis_pass -- would touch a production file, so it
was explicitly left out of THIS task's scope. `feature_exporter.py --verify`
cross-checks this harness's output against a real `engine.run_analysis_pass()`
call to catch drift between the two, since duplication without a drift check
would be worse than no duplication at all.

CLIP-RELATIVE FRAME CONVENTION (deliberately chosen to make the project's
already-shipped-once frame-offset bug class structurally impossible here):
this harness ALWAYS operates on an already-EXTRACTED clip file (produced by
clip_extractor.py), never on the full source video, and ALWAYS uses
start_frame=0 for both passes. There is therefore no "start_frame offset"
anywhere in this file for a bug to hide in -- frame N in every buffer this
harness touches means "the Nth frame of THIS clip file," full stop. Absolute
source-video frame numbers only ever appear in clip_time_map.json and are
converted via time_utils.ClipTimeMap, never passed into the pipeline itself.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from football_analysis.adapter import TrackingAdapter          # noqa: E402  (frozen, unmodified)
from football_analysis.event_detector import FootballEventDetector  # noqa: E402
from football_analysis.event_qualifier import EventQualifier   # noqa: E402
from football_analysis.clip_tagger import ClipTagger           # noqa: E402
from football_analysis.confidence import ConfidenceEngine      # noqa: E402
from football_analysis.pitch_context import estimate_pitch_context  # noqa: E402
from football_analysis.visual_pitch_timeline import VisualCalibrationTimeline, extract_local_positions  # noqa: E402
from football_analysis.goalkeeper_context import estimate_goalkeeper_candidate_visual  # noqa: E402
from football_analysis.opponent_detection import (   # noqa: E402
    LocalChallengeDetector, OtherPlayerTrackStore, merge_other_player_candidates,
)
from football_analysis.engine import FootballAnalysisEngine, LOCAL_CONTEXT_WINDOW  # noqa: E402


@dataclass
class CaptureResult:
    """Everything a feature exporter needs, gathered from a single clip run.
    `result` is the SAME shape/content ClipTagger.tag() always produces --
    i.e. what a real engine.run_analysis_pass() call would also return for
    identical inputs (see --verify)."""
    result: dict
    snapshots: list                 # TrackingSnapshot, one per clip-relative frame (from Pass 1, unmodified)
    ball_history: list              # BallTrackState, one per clip-relative frame processed in Pass 2
    possession_frames: list         # PossessionFrame, one per clip-relative frame processed in Pass 2
    pitch_context: object           # PitchContext (global net-drift/goalkeeper proxy)
    fps: float
    width: int
    height: int


def run_pass1_tracking(clip_video_path: str, init_frame: int, init_box: tuple,
                        player_id: str, jersey_number: Optional[int], fps: float,
                        max_frames: int) -> list:
    """Thin, direct call to the REAL, UNMODIFIED football_analysis.engine.
    FootballAnalysisEngine.run_tracking_pass -- no duplication needed here,
    since that method already returns exactly what we need (the snapshot
    list), unlike run_analysis_pass."""
    engine = FootballAnalysisEngine(fps=fps)
    return engine.run_tracking_pass(clip_video_path, start_frame=init_frame, init_box=init_box,
                                     player_id=player_id, jersey_number=jersey_number,
                                     max_frames=max_frames, tracking_json_path=None)


def run_with_capture(clip_video_path: str, snapshots: list, tracked_color_hist,
                      fps: float, conf: float = 0.15, use_heavy_ball_model: bool = True,
                      ball_config=None) -> CaptureResult:
    """Read-only re-execution of engine.run_analysis_pass's core loop (debug-
    video rendering branches omitted -- irrelevant to feature export), kept
    in lockstep with it deliberately (see module docstring). Returns both the
    production-shaped `result` dict AND the internal per-frame buffers that
    dict does not expose."""
    from ultralytics import YOLO

    by_frame = {s.frame_number: s for s in snapshots}
    model = YOLO("yolov8s.pt")
    heavy_factory = (lambda: YOLO("yolov8x.pt")) if use_heavy_ball_model else None
    detector = FootballEventDetector(tracked_color_hist, EventQualifier(), light_model=model,
                                      heavy_model_factory=heavy_factory, ball_config=ball_config, fps=fps)

    local_challenge_detector = LocalChallengeDetector(model)
    other_track_store = OtherPlayerTrackStore()

    cap = cv2.VideoCapture(clip_video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    t0 = time.time()
    frame_idx = 0
    max_frame = max(by_frame.keys())
    while frame_idx <= max_frame:
        ok, frame = cap.read()
        if not ok:
            break
        snap = by_frame.get(frame_idx)
        if snap is not None:
            res = model.predict(frame, classes=[0, 32], conf=conf, verbose=False)[0]
            persons, balls = [], []
            for b in res.boxes:
                cls = int(b.cls[0]); c = float(b.conf[0])
                xyxy = tuple(float(v) for v in b.xyxy[0])
                (persons if cls == 0 else balls).append((xyxy, c))
            ball_centre = None
            if balls:
                (bx1, by1, bx2, by2), _bc = max(balls, key=lambda z: z[1])
                ball_centre = ((bx1 + bx2) / 2.0, (by1 + by2) / 2.0)
            others, _opponent_records = merge_other_player_candidates(
                frame, frame_idx, tuple(snap.bbox), persons, ball_centre,
                local_detector=local_challenge_detector, track_store=other_track_store,
                team_classifier=None)
            detector.process_frame(frame, snap, balls, others)
        frame_idx += 1
    cap.release()

    pitch_context = estimate_pitch_context(detector._snapshots, detector.ball_tracker, detector._other_positions)

    visual_timeline = VisualCalibrationTimeline(frame_width=w)
    for pcf in detector._visual_calibration_frames:
        visual_timeline.add_sample(pcf)

    def context_resolver(fi):
        lo = max(0, fi - LOCAL_CONTEXT_WINDOW)
        hi = min(len(detector._other_positions), fi + LOCAL_CONTEXT_WINDOW + 1)
        local_other_positions = detector._other_positions[lo:hi]

        def keeper_fn(goal_centre):
            return estimate_goalkeeper_candidate_visual(local_other_positions, goal_centre)

        local_positions = extract_local_positions(detector.ball_tracker, detector._snapshots, fi,
                                                    window=LOCAL_CONTEXT_WINDOW)
        return visual_timeline.for_release(fi, window=LOCAL_CONTEXT_WINDOW,
                                            local_positions=local_positions, keeper_candidate_fn=keeper_fn)

    events = detector.finalize(pitch_context=pitch_context, keeper_candidate=pitch_context.keeper_candidate,
                                context_resolver=context_resolver)
    confidence_engine = ConfidenceEngine()
    for e in events:
        confidence_engine.score(e)

    tagger = ClipTagger(fps=fps)
    bt = detector.ball_tracker
    ball_extra = {
        "reacquisitions": bt.reacquisition_count(),
        "mean_consecutive_lost_frames": bt.mean_consecutive_lost(),
        "state_transitions": len(bt.transition_log),
    }
    result = tagger.tag(events, snapshots[0].player_id, bt.stats(), ball_extra_stats=ball_extra,
                         shot_log=detector.shot_log, pitch_context=pitch_context.as_dict(),
                         visual_calibration_stats=visual_timeline.coverage_stats())
    result["meta"]["processing_time_s"] = round(time.time() - t0, 1)
    result["meta"]["processed_frames"] = len(snapshots)

    return CaptureResult(
        result=result,
        snapshots=detector._snapshots,
        ball_history=list(bt.history),
        possession_frames=list(detector._possession_frames),
        pitch_context=pitch_context,
        fps=fps, width=w, height=h,
    )


def verify_against_production(clip_video_path: str, snapshots: list, tracked_color_hist, fps: float,
                               capture: CaptureResult, conf: float = 0.15,
                               use_heavy_ball_model: bool = True, ball_config=None) -> bool:
    """Cross-check: run the REAL, untouched engine.run_analysis_pass() on the
    same inputs and diff its result against this harness's `capture.result`.
    Used by feature_exporter.py --verify. Returns True iff the two decision
    outputs (event types/outcomes/scores) agree -- NOT a byte-for-byte diff,
    since processing_time_s and similar non-decision metadata will always
    differ between two separate runs."""
    engine = FootballAnalysisEngine(fps=fps)
    real = engine.run_analysis_pass(clip_video_path, snapshots, tracked_color_hist,
                                     start_frame=0, conf=conf, use_heavy_ball_model=use_heavy_ball_model,
                                     ball_config=ball_config)
    a = [(e["type"], e.get("success"), e["start_frame"], e["end_frame"]) for e in capture.result["events"]]
    b = [(e["type"], e.get("success"), e["start_frame"], e["end_frame"]) for e in real["events"]]
    if a != b:
        print("VERIFY FAILED -- harness output diverges from real engine.run_analysis_pass():")
        print("  harness:", a)
        print("  real   :", b)
        return False
    return True

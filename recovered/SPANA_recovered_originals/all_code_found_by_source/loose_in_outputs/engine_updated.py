"""
engine.py -- FootballAnalysisEngine: the two-pass pipeline (rule #5).

    PASS 1 (run_tracking_pass): video -> existing PlayerTracker -> tracking_results.json
    PASS 2 (run_analysis_pass): video + tracking_results.json -> football_analysis -> events.json

This is the ONE file in football_analysis that imports anything from
player_tracker, because Pass 1 has to actually run the tracker to produce
its output. It calls exactly two of the tracker's own already-public
methods -- `PlayerTracker.initialize(frame, box)` and `PlayerTracker.step(frame)`
-- in exactly the way player_tracker/run.py already does. Nothing here
subclasses PlayerTracker, reaches into PlayerIdentity, changes a config
value, or calls anything from association.py/identity_manager.py/motion.py
directly. Pass 2 never imports player_tracker at all -- it only reads the
JSON Pass 1 wrote.

If real-time (single-pass) operation is ever needed, run_tracking_pass's
loop and run_analysis_pass's loop can be interleaved by a caller frame-by-
frame -- but the conceptual separation stays: the analysis object never
receives anything but a TrackingSnapshot, never a tracker reference.
"""
import json
import time
from dataclasses import asdict

import cv2

from .adapter import TrackingAdapter
from .event_detector import FootballEventDetector
from .event_qualifier import EventQualifier
from .clip_tagger import ClipTagger
from .confidence import ConfidenceEngine
from .debug_overlay import AnalysisDebugOverlay
from .ball_debug_overlay import BallDebugOverlay
from .geometry import iou
from .pitch_context import estimate_pitch_context
from .visual_pitch_timeline import VisualCalibrationTimeline, extract_local_positions
from .goalkeeper_context import estimate_goalkeeper_candidate_visual
from .visual_calibration_overlay import render_calibration_debug_video
from .release_geometry import nearby_ball_centre
from .opponent_detection import LocalChallengeDetector, OtherPlayerTrackStore, merge_other_player_candidates

LOCAL_CONTEXT_WINDOW = 60   # frames either side of a release (rule #11)


class FootballAnalysisEngine:
    def __init__(self, fps=25.0):
        self.fps = fps

    # ------------------------------------------------------------ PASS 1
    def run_tracking_pass(self, video_path, start_frame, init_box, player_id="P1",
                           jersey_number=None, max_frames=150, tracking_json_path=None):
        """Runs the EXISTING, UNMODIFIED PlayerTracker and writes its output
        as TrackingSnapshots. This is the only place football_analysis
        touches player_tracker, and it only calls the two methods
        player_tracker/run.py itself already calls."""
        from player_tracker.tracker import PlayerTracker, TrackerConfig   # local import: isolates the one dependency

        tracker = PlayerTracker(player_id=player_id, jersey_number=jersey_number, config=TrackerConfig())
        adapter = TrackingAdapter(player_id=player_id, fps=self.fps)

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ok, frame0 = cap.read()
        if not ok:
            raise RuntimeError("could not read start frame")

        tracker.initialize(frame0, init_box)
        snapshots = [adapter.snapshot({"frame": 0, "state": "TRACKING", "confidence": 1.0, "box": list(init_box)})]

        n = 0
        while n < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            n += 1
            record, _scored = tracker.step(frame)   # tracker's own public output -- read, not altered
            snapshots.append(adapter.snapshot(record))
        cap.release()

        if tracking_json_path:
            TrackingAdapter.save(snapshots, tracking_json_path)
        return snapshots

    # ------------------------------------------------------------ PASS 2
    def run_analysis_pass(self, video_path, snapshots_or_path, tracked_color_hist,
                           start_frame=0, conf=0.15, out_json_path=None, out_debug_video_path=None,
                           out_ball_debug_video_path=None, use_heavy_ball_model=True, ball_config=None,
                           out_calibration_debug_video_path=None, manual_context_resolver=None,
                           capture_opponent_debug=False):
        """Never imports player_tracker. Consumes only TrackingSnapshots
        (already-produced Pass 1 output) plus a fresh, independent YOLO pass
        for ball + other-person detections (tier A of the ball hierarchy --
        tiers B-E, the staged ROI re-detection and the optional heavier
        model, live inside BallTracker/StagedBallSearch and only fire when
        tier A comes up empty on a given frame).

        NOTE (rule #9): this pipeline has no camera-motion estimate to
        consume -- player_tracker's own camera compensation (camera.py) is
        internal to the frozen tracker and not exposed as a reusable
        output, and this engine doesn't otherwise estimate pans/cuts. So
        `is_cut` is always False here; a real scene cut would currently be
        handled by the ball gating's implausible-jump rejection alone,
        which is more conservative than intended but never wrong in the
        unsafe direction (it can only cause an extra LOST, never a false
        lock). Flagging this rather than quietly calling it solved.

        OPPONENT-DETECTION TASK (rules #3-#14): the combined person+ball
        pass below still runs exactly as before (same model, same conf,
        same classes, no imgsz/iou override) -- it is the ONLY source of
        other-player boxes and is untouched. What changes is what happens
        to its person boxes AFTER that pass: instead of the old inline
        `others = [p for p in persons if iou(p, tracked_bbox) < 0.3]`,
        they now go through opponent_detection.merge_other_player_
        candidates(), which (a) does the exact same target-duplicate
        exclusion (rule #7, now IoU>=0.5 rather than <0.3 -- a tighter,
        more correct match to "is this really the same person" than the
        old ad-hoc 0.3 cutoff, which the rule-#1 diagnosis showed was
        already wide enough that it was never itself the failure), (b)
        optionally adds a local, higher-resolution challenge-ROI pass
        around the tracked player when a proximity/crowding trigger fires
        (rule #12), and (c) bridges brief single-frame misses via short-
        lived temporal tracks (rule #8). `others` -- the plain xyxy list
        handed to `detector.process_frame` -- keeps byte-identical shape
        to what it always was; event_detector.py/possession/team-
        classification are completely unmodified and still do their own
        (frozen) team classification on it. `local_challenge_detector`
        reuses the SAME already-loaded `model` object (no second model is
        ever loaded), so this adds no new weights and no new global
        compute -- the local pass only fires per rule #12's proximity
        trigger, never every frame. capture_opponent_debug (default
        False, so default calls are byte-identical in output shape/size
        to before this task): when True, stores rule #13's full
        structured-confidence OpponentCandidate records per frame in
        result["meta"]["opponent_debug"] -- report/debug only, never
        read by anything upstream of this return.
        """
        from ultralytics import YOLO   # local import: keeps this pass's one dependency explicit too

        snapshots = (TrackingAdapter.load(snapshots_or_path)
                     if isinstance(snapshots_or_path, str) else snapshots_or_path)
        by_frame = {s.frame_number: s for s in snapshots}

        model = YOLO("yolov8s.pt")
        heavy_factory = (lambda: YOLO("yolov8x.pt")) if use_heavy_ball_model else None
        detector = FootballEventDetector(tracked_color_hist, EventQualifier(), light_model=model,
                                          heavy_model_factory=heavy_factory, ball_config=ball_config, fps=self.fps)
        ball_overlay = BallDebugOverlay() if out_ball_debug_video_path else None

        # OPPONENT-DETECTION TASK: built once, outside the per-frame loop,
        # exactly like `model`/`detector` above -- both are read-only from
        # here on except for their own internal, self-contained state
        # (the local detector's crop/predict call; the track store's short-
        # lived bridge dict), never touching player_tracker or the tracked
        # bbox itself.
        local_challenge_detector = LocalChallengeDetector(model)
        other_track_store = OtherPlayerTrackStore()
        opponent_debug_frames = [] if capture_opponent_debug else None

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ball_out = (cv2.VideoWriter(out_ball_debug_video_path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
                    if ball_overlay else None)

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
                # OPPONENT-DETECTION TASK: read-only reuse of this SAME tier-A
                # ball pass's own boxes, purely as a proximity hint for rule
                # #12's trigger -- the highest-confidence ball box this frame,
                # if any. This never touches BallTracker's own (frozen) state
                # machine/history; it's a plain (x,y) or None.
                ball_centre = None
                if balls:
                    (bx1, by1, bx2, by2), _bc = max(balls, key=lambda z: z[1])
                    ball_centre = ((bx1 + bx2) / 2.0, (by1 + by2) / 2.0)
                others, opponent_records = merge_other_player_candidates(
                    frame, frame_idx, tuple(snap.bbox), persons, ball_centre,
                    local_detector=local_challenge_detector, track_store=other_track_store,
                    team_classifier=None)
                if opponent_debug_frames is not None:
                    opponent_debug_frames.append({
                        "frame": frame_idx,
                        "records": [asdict(r) for r in opponent_records],
                    })
                detector.process_frame(frame, snap, balls, others)
                # the BALL debug overlay is written LIVE, in this same loop: it
                # needs BallTracker's per-frame `_last_search_meta` (scored
                # candidates / which search stage fired), which ball_tracker.py
                # -- frozen for this task -- only ever keeps the latest copy
                # of. A shot-candidate annotation is NOT available yet here
                # (candidates aren't resolved until finalize(), below), which
                # is exactly why the ANALYSIS overlay is rendered in a
                # separate, second pass after finalize() instead of here.
                if ball_out:
                    ball_state = detector.ball_tracker.history[-1] if detector.ball_tracker.history else None
                    meta = detector.ball_tracker._last_search_meta
                    ball_out.write(ball_overlay.draw(frame.copy(), ball_state, scored_candidates=meta.get("scored"),
                                                       detector_used=meta.get("stage")))
            frame_idx += 1
            if frame_idx % 25 == 0:
                print(f"  ...analysis frame {frame_idx}/{max_frame} ({time.time()-t0:.0f}s elapsed)", flush=True)

        cap.release()
        if ball_out:
            ball_out.release()

        # spatial/football context (pitch_context.py + goalkeeper_context.py,
        # via estimate_pitch_context) is computed HERE, after the per-frame
        # loop finishes, from this same detector's already-buffered
        # snapshots / ball_tracker history / other_positions -- read-only,
        # post-hoc, downstream of both frozen trackers. It's then threaded
        # into finalize() so the SHOT state machine can use it as
        # *additional*, non-decisive evidence (see shot_candidate.py).
        pitch_context = estimate_pitch_context(detector._snapshots, detector.ball_tracker, detector._other_positions)

        # visual pitch/goal calibration (visual_pitch_calibration.py +
        # visual_pitch_timeline.py): built from the PitchContextFrame samples
        # collected during the main loop above (every VISUAL_CALIBRATION_
        # INTERVAL-th frame, from the SAME frames already decoded there -- no
        # extra video read). This is tried FIRST for each release, per rule
        # #10/#13; the global net-drift `pitch_context` above is only ever
        # used as the fallback when local visual evidence isn't available
        # for that specific release frame (rule #6/#15: no fabricated
        # context either way).
        visual_timeline = VisualCalibrationTimeline(frame_width=w)
        for pcf in detector._visual_calibration_frames:
            visual_timeline.add_sample(pcf)

        def context_resolver(frame_idx):
            lo = max(0, frame_idx - LOCAL_CONTEXT_WINDOW)
            hi = min(len(detector._other_positions), frame_idx + LOCAL_CONTEXT_WINDOW + 1)
            local_other_positions = detector._other_positions[lo:hi]

            def keeper_fn(goal_centre):
                return estimate_goalkeeper_candidate_visual(local_other_positions, goal_centre)

            local_positions = extract_local_positions(detector.ball_tracker, detector._snapshots, frame_idx,
                                                        window=LOCAL_CONTEXT_WINDOW)
            return visual_timeline.for_release(frame_idx, window=LOCAL_CONTEXT_WINDOW,
                                                local_positions=local_positions, keeper_candidate_fn=keeper_fn)

        # manual_context_resolver (rule #16/#17, validation only): an
        # explicit, human-supplied override callable(frame_idx) -> a
        # ManualPitchContext-derived VisualPitchContext or None, used ONLY
        # when a caller passes one in (the report/validation script, never
        # the default pipeline) -- proves whether missing AUTOMATIC visual
        # calibration is really what's holding a release back, without
        # silently mixing manual assumptions into normal runs.
        active_resolver = manual_context_resolver if manual_context_resolver is not None else context_resolver

        events = detector.finalize(pitch_context=pitch_context, keeper_candidate=pitch_context.keeper_candidate,
                                    context_resolver=active_resolver)
        # score confidence now (not just inside tagger.tag(), below) so the
        # retrospective debug-overlay pass can show each SHOT candidate's
        # final confidence too -- ConfidenceEngine.score() is a pure,
        # deterministic function of an event's own evidence, so calling it
        # again inside tagger.tag() afterward is harmless (same result).
        confidence_engine = ConfidenceEngine()
        for e in events:
            confidence_engine.score(e)

        if out_debug_video_path:
            self._render_analysis_debug_pass(video_path, start_frame, max_frame, detector, events,
                                              out_debug_video_path, w, h, pitch_context=pitch_context)

        if out_calibration_debug_video_path:
            # rule #15: per-release wide-origin geometry, drawn only on the
            # frame windows an actual RELEASE event covers -- recomputed here
            # via the same (deterministic, side-effect-free) active_resolver
            # + nearby_ball_centre already used inside finalize(), purely for
            # visualisation; nothing here alters an event's own evidence.
            wide_annotations = {}
            release_types = ("PASS", "SHOT", "CROSS", "UNKNOWN_RELEASE")
            for e in events:
                if e.type not in release_types:
                    continue
                release_frame = e.evidence.get("release_frame", e.start_frame)
                wide_geom = e.evidence.get("wide_release_geometry")
                if wide_geom is None:
                    continue
                local_ctx = active_resolver(release_frame)
                release_pos = nearby_ball_centre(detector.ball_tracker, release_frame)
                info = {
                    "release_point": release_pos,
                    "goal_centre": getattr(local_ctx, "goal_centre", None) if local_ctx else None,
                    "left_post": getattr(local_ctx, "goal_left_post", None) if local_ctx else None,
                    "right_post": getattr(local_ctx, "goal_right_post", None) if local_ctx else None,
                    "geometry": wide_geom,
                }
                for f in range(e.start_frame, e.end_frame + 1):
                    wide_annotations[f] = info

            render_calibration_debug_video(video_path, start_frame, max_frame, self.fps, w, h,
                                            visual_timeline, active_resolver, out_calibration_debug_video_path,
                                            sample_gap=detector._visual_calibration_interval,
                                            wide_annotations=wide_annotations)

        tagger = ClipTagger(fps=self.fps)
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
        result["meta"]["ball_transition_log"] = bt.transition_log
        if out_debug_video_path:
            result["meta"]["debug_clip"] = out_debug_video_path
        if out_ball_debug_video_path:
            result["meta"]["ball_debug_clip"] = out_ball_debug_video_path
        if opponent_debug_frames is not None:
            result["meta"]["opponent_debug"] = opponent_debug_frames

        if out_json_path:
            with open(out_json_path, "w") as f:
                json.dump(result, f, indent=2)
            result["meta"]["json_path"] = out_json_path
        return result

    # ---------------------------------------------- retrospective overlay pass
    def _render_analysis_debug_pass(self, video_path, start_frame, max_frame, detector, events,
                                     out_debug_video_path, w, h, pitch_context=None):
        """Second, overlay-only pass over the same clip, run AFTER finalize()
        so every SHOT candidate's final resolution is known. Does no
        detection/tracking work at all -- it only re-reads frames and reuses
        the already-computed per-frame history FootballEventDetector built
        during the first pass (self._possession_frames, self._snapshots,
        ball_tracker.history), plus a small frame_idx -> annotation map built
        from `events`. This is what lets the debug overlay show release
        frame / resolution reason / confidence per rule #14, without needing
        BallTracker (frozen) to expose anything it doesn't already."""
        annotations = {}
        release_types = ("PASS", "SHOT", "CROSS", "UNKNOWN_RELEASE", "GOAL")
        for e in events:
            if e.type not in release_types:
                continue
            release_frame = e.evidence.get("release_frame", e.start_frame)
            info_base = {
                "event_type": e.type,
                "release_frame": release_frame,
                "resolution_reason": e.evidence.get("resolution_reason"),
                "confidence": e.evidence.get("confidence_score"),
                "release_speed": e.evidence.get("release_speed_px_per_frame"),
            }
            for f in range(e.start_frame, e.end_frame + 1):
                annotations[f] = {**info_base, "active": f < e.end_frame}
            annotations[e.end_frame] = {**info_base, "active": False}

        overlay = AnalysisDebugOverlay()
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        out = cv2.VideoWriter(out_debug_video_path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))

        frame_idx = 0
        while frame_idx <= max_frame:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx < len(detector._snapshots):
                snap = detector._snapshots[frame_idx]
                pframe = detector._possession_frames[frame_idx]
                seg_frame = (detector._segmentation_frames[frame_idx]
                             if frame_idx < len(detector._segmentation_frames) else None)
                ball_state = detector.ball_tracker.history[frame_idx] if frame_idx < len(detector.ball_tracker.history) else None
                out.write(overlay.draw(frame.copy(), snap, pframe, ball_state,
                                        shot_annotation=annotations.get(frame_idx),
                                        pitch_context=pitch_context, segmentation_frame=seg_frame))
            frame_idx += 1
        cap.release()
        out.release()

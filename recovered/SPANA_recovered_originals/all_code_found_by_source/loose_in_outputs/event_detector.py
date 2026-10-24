"""
event_detector.py -- FootballEventDetector: the per-frame orchestrator that
ties BallTracker + PlayerBallInteractionDetector + PossessionDetector +
TeamClassifier together while frames are being processed, then runs
EventClassifier/EventQualifier once at the end to turn the accumulated
possession history into discrete events.

This is the only place in football_analysis that sees both a TrackingSnapshot
and raw per-frame detections together -- and it only ever READS the
snapshot's fields (bbox, tracking_state, ...). It has no method that could
send anything back to the tracker even if someone tried: there is no
tracker reference anywhere in this object.
"""
from .ball_tracker import BallTracker
from .interaction_detector import PlayerBallInteractionDetector
from .possession_detector import PossessionDetector, TeamClassifier
from .possession_segmentation import segment_possession
from .event_classifier import EventClassifier
from .geometry import foot_point, dist
from .visual_pitch_calibration import calibrate_frame
from .release_geometry import estimate_release_width, nearby_ball_centre

VISUAL_CALIBRATION_INTERVAL = 5   # sample every Nth frame for goal/pitch calibration (rule #4) --
                                    # reuses the frame this class already has in process_frame, no extra video read


def _state_counts(snapshots, i0, i1):
    counts = {}
    for s in snapshots[i0:i1 + 1]:
        counts[s.tracking_state] = counts.get(s.tracking_state, 0) + 1
    return counts


def _ball_state_counts(ball_tracker, i0, i1):
    counts = {}
    for o in ball_tracker.history[i0:i1 + 1]:
        counts[o.state] = counts.get(o.state, 0) + 1
    return counts


class FootballEventDetector:
    def __init__(self, tracked_color_hist, event_qualifier, light_model=None, heavy_model_factory=None,
                 ball_config=None, fps=25.0, enable_visual_calibration=True,
                 visual_calibration_interval=VISUAL_CALIBRATION_INTERVAL):
        self.ball_tracker = BallTracker(light_model=light_model, heavy_model_factory=heavy_model_factory,
                                         fps=fps, config=ball_config)
        self.interaction = PlayerBallInteractionDetector()
        self.possession = PossessionDetector()
        self.team_classifier = TeamClassifier(tracked_color_hist)
        self.classifier = EventClassifier()
        self.qualifier = event_qualifier
        self.fps = fps
        self._possession_frames = []
        self._snapshots = []
        self._other_positions = []   # per-frame list of (foot_point, team) for every non-tracked person
                                       # detection -- read-only reuse of the same per-frame person boxes
                                       # already computed by the caller's one YOLO pass; feeds pitch_context.py
                                       # post-hoc (attack direction / goalkeeper candidate). Never a new
                                       # detection model, never fed back into tracking.
        self.shot_log = []   # event-analysis-side transition log (mirrors ball_tracker's rule #19 log)
        self._segmentation_frames = []   # set once, in finalize() -- possession_segmentation.py's
                                           # hysteresis-corrected per-frame decisions, kept around purely
                                           # for the debug overlay (rule #16)

        # visual pitch/goal calibration (visual_pitch_calibration.py): sampled
        # from the SAME frame this class already receives in process_frame --
        # no extra video decode pass. Read-only, downstream-only, never fed
        # back into any tracker (see that module's docstring). Disabled via
        # enable_visual_calibration=False falls back to identical pre-task
        # behaviour (global net-drift-based pitch_context only).
        self._enable_visual_calibration = enable_visual_calibration
        self._visual_calibration_interval = max(1, visual_calibration_interval)
        self._visual_calibration_frames = []
        self._visual_calibration_errors = 0

    def process_frame(self, frame, snapshot, ball_candidates, other_boxes, is_cut=False):
        """frame: the raw video frame (BGR ndarray), needed for team-colour
        classification AND for the ball tracker's staged ROI re-detection
        when the cheap tier-A pass (ball_candidates) comes up empty.
        snapshot: this frame's TrackingSnapshot (read-only tracker output),
        passed to the ball tracker purely as read-only search context
        (rule #6) -- it never flows back into player identity. ball_candidates:
        tier-A (box, conf) ball-class detections already computed by the
        caller's single combined YOLO pass. other_boxes: person boxes that
        are NOT the tracked player (already excluded by the caller via IoU
        against snapshot.bbox -- this class does not decide who the tracked
        player is, it's only ever told)."""
        # BALL-TRACKING ROBUSTNESS TASK: the only change in this file. Passes
        # `other_boxes` one call earlier than before -- it was already being
        # computed by the caller and already flowing into this method
        # unchanged (see the loop below, which still uses it exactly as
        # before, for team classification and nearest-team distance). It is
        # now ALSO handed to BallTracker.step() itself, as optional, read-
        # only crowd-aware search/scoring context (sections #3-#5, #9, #11,
        # #15 -- see ball_tracker.py/ball_roi_search.py/ball_gating.py).
        # Nothing about how other_boxes is computed, or anything below this
        # line, changes.
        ball_obs = self.ball_tracker.step(snapshot.frame_number, frame, ball_candidates,
                                           player_bbox=snapshot.bbox, is_cut=is_cut, other_boxes=other_boxes)
        interaction = self.interaction.step(snapshot, ball_obs)

        # TEAM-CLASSIFICATION TASK (rules #9/#13/#14): two minimal, purely
        # additive calls into the rewritten TeamClassifier. Neither changes
        # this method's control flow, return value, or anything about the
        # tracked player -- `observe_tracked_player` only ever APPENDS a
        # read-only colour sample of the tracked player's own (frozen,
        # already-known) bbox to TeamClassifier's internal TEAM_A reference
        # pool (rule #9: multiple clean frames beat a single frame-0
        # sample), and also lets TeamClassifier know this frame's tracked
        # bbox for its own rule #14 overlap-contamination check.
        # `frame_number=snapshot.frame_number` on the classify() call below
        # is the only other change (an optional 3rd argument -- backward
        # compatible with any 2-arg caller), used purely for rule #13's
        # temporal-consistency bookkeeping inside TeamClassifier itself.
        self.team_classifier.observe_tracked_player(frame, snapshot.bbox)

        nearest_team, nearest_d = None, None
        # PASS-EVIDENCE FIX (this task -- see possession_detector.py's PossessionFrame/
        # PossessionDetector.step() docstring notes for the full rationale): nearest_team/
        # nearest_d immediately below are computed EXACTLY as before this task, in the
        # exact same loop, from the exact same per-box team classification calls -- their
        # values are byte-identical to pre-task behaviour, and shot_candidate.py's frozen
        # confirmed-reception path continues to read only these two. The two accumulators
        # below are NEW and purely additive: independent per-team nearest-to-ball minimums,
        # so a marking OPPONENT who happens to be marginally nearer the ball on a given
        # frame no longer silently discards a genuinely nearby TEAMMATE's (or unclassified-
        # team player's) distance from the evidence event_candidates.py's PASS-intent
        # candidate can see. Root-cause note (validation-batch instrumented trace,
        # pass_completed_04): the actual receiving player was 13px from the ball -- well
        # inside shot_candidate.py's own 55px reception radius -- but TeamClassifier's
        # same-frame colour read labelled them UNKNOWN, not TEAMMATE, so nearest_teammate_
        # distance below is None on many frames where a real (if uncertainly-classified)
        # receiver candidate exists; nearest_unknown_distance below is what lets
        # build_pass_intent_candidate treat that honestly, as weak/uncertain evidence
        # (item #17), instead of as "nothing at all".
        nearest_teammate_d, nearest_unknown_d = None, None
        frame_positions = []
        for b in other_boxes:
            team, _sim = self.team_classifier.classify(frame, b, frame_number=snapshot.frame_number)
            fp = foot_point(b)
            frame_positions.append((fp, team))
            if ball_obs.center is not None:
                d = dist(fp, ball_obs.center)
                if nearest_d is None or d < nearest_d:
                    nearest_d, nearest_team = d, team
                if team == "TEAMMATE" and (nearest_teammate_d is None or d < nearest_teammate_d):
                    nearest_teammate_d = d
                elif team == "UNKNOWN" and (nearest_unknown_d is None or d < nearest_unknown_d):
                    nearest_unknown_d = d
        self._other_positions.append(frame_positions)

        if self._enable_visual_calibration and snapshot.frame_number % self._visual_calibration_interval == 0:
            try:
                self._visual_calibration_frames.append(calibrate_frame(frame, snapshot.frame_number))
            except Exception:
                # calibration is a best-effort, disclosed-confidence signal
                # (see visual_pitch_calibration.py) -- a detector error here
                # must never break event detection itself, it just means one
                # fewer calibration sample is available for this frame
                self._visual_calibration_errors += 1

        pframe = self.possession.step(interaction, nearest_other_team=nearest_team, nearest_other_distance=nearest_d,
                                       nearest_teammate_distance=nearest_teammate_d,
                                       nearest_unknown_distance=nearest_unknown_d)
        self._possession_frames.append(pframe)
        self._snapshots.append(snapshot)
        return pframe

    def finalize(self, pitch_context=None, keeper_candidate=None, context_resolver=None):
        """pitch_context/keeper_candidate (both optional): read-only GLOBAL
        spatial context computed by engine.py AFTER the per-frame loop
        finishes (pitch_context.py's net-drift proxy / goalkeeper_context.py),
        from this same detector's own already-buffered self._snapshots /
        ball_tracker / self._other_positions -- never a new tracker, never
        fed back into one. Used as the fallback when no better evidence
        exists for a given release.

        context_resolver (optional, minimal compatibility addition for
        visual pitch/goal calibration): a callable(frame_idx) -> a
        pitch-context-like object (duck-compatible with pitch_context.
        PitchContext -- see visual_pitch_timeline.VisualPitchContext) or
        None. When supplied, it is tried FIRST for each release/goal-
        follow-up frame (rule #10/#13: direct visual goal evidence outranks
        long-window net drift); its result's own .keeper_candidate (if any)
        is preferred too. Falling back to the flat pitch_context/
        keeper_candidate arguments when the resolver returns None keeps
        this class working exactly as before for any caller that doesn't
        pass one. shot_candidate.py/event_qualifier.py themselves receive
        only a plain pitch_context/keeper_candidate object either way --
        THIS is the only place that decides which one that is."""
        self.ball_tracker.finalize_interpolation()   # no-op now -- prediction already ran online, see ball_tracker.py

        # EVENT-SEGMENTATION FIX (this task): walk hysteresis-corrected
        # segment boundaries (possession_segmentation.py) instead of the
        # raw, single-frame-threshold PossessionFrame.in_possession. This
        # is a pure post-processing pass over the same buffered frames --
        # it changes where a segment is cut, never how a cut segment is
        # subsequently classified (that stays entirely inside
        # event_classifier.py / shot_candidate.py, both otherwise
        # untouched). See possession_segmentation.py's module docstring.
        self._segmentation_frames = segment_possession(self._possession_frames)

        events = []
        n = len(self._possession_frames)
        i = 0
        while i < n:
            if not self._segmentation_frames[i].in_possession:
                i += 1
                continue
            seg_start = i
            while i < n and self._segmentation_frames[i].in_possession:
                i += 1
            seg_end = i - 1

            ev = self.classifier.classify_segment(self._snapshots, self.ball_tracker, seg_start, seg_end)
            ev.evidence["tracking_states"] = _state_counts(self._snapshots, seg_start, seg_end)
            ev.evidence["ball_states"] = _ball_state_counts(self.ball_tracker, seg_start, seg_end)
            ev.evidence["team_sim"] = None
            events.append(ev)

            if seg_end < n - 1:
                local_ctx = context_resolver(seg_end) if context_resolver is not None else None
                effective_ctx = local_ctx if local_ctx is not None else pitch_context
                effective_keeper = (getattr(local_ctx, "keeper_candidate", None) if local_ctx is not None else None) \
                    or keeper_candidate
                release = self.qualifier.qualify_release(self.ball_tracker, self._possession_frames, seg_end,
                                                           fps=self.fps, log=self.shot_log,
                                                           pitch_context=effective_ctx, keeper_candidate=effective_keeper,
                                                           seg_start=seg_start, prior_segment_evidence=ev.evidence)
                release.evidence.setdefault("tracking_states", _state_counts(self._snapshots, seg_end, release.end_frame))
                release.evidence.setdefault("ball_states", _ball_state_counts(self.ball_tracker, seg_end, release.end_frame))
                release.evidence.setdefault("team_sim", None)
                release.evidence["spatial_context_source"] = (
                    getattr(local_ctx, "source", "visual_auto") if local_ctx is not None else
                    ("global_net_drift_fallback" if (pitch_context is not None and
                                                      getattr(pitch_context, "pitch_confidence", 0.0) > 0.0) else "none")
                )
                # rich, perspective-aware wide-release report (this task): computed
                # from the SAME effective_ctx just used above, purely for reporting/
                # debug -- shot_candidate.py already applied its own (confidence-
                # damped scalar) version of this via qualify_release, so this adds
                # detail to the evidence dict, it does not change the event's type,
                # success, or any threshold decision already made.
                release_pos_report = nearby_ball_centre(self.ball_tracker, seg_end)
                release.evidence["wide_release_geometry"] = estimate_release_width(release_pos_report,
                                                                                     effective_ctx).as_dict()
                events.append(release)

                # GOAL follow-up: gated strictly behind a confirmed SHOT (rule
                # -- never evaluated for PASS/UNKNOWN_RELEASE). May return
                # None (no circumstantial signal worth flagging at all).
                if release.type == "SHOT":
                    followup_ctx = context_resolver(release.end_frame) if context_resolver is not None else None
                    effective_followup_ctx = followup_ctx if followup_ctx is not None else pitch_context
                    followup = self.qualifier.qualify_goal_followup(
                        release, self.ball_tracker, self._possession_frames, fps=self.fps, log=self.shot_log,
                        pitch_context=effective_followup_ctx)
                    if followup is not None:
                        followup.evidence.setdefault("tracking_states",
                                                       _state_counts(self._snapshots, release.end_frame, followup.end_frame))
                        followup.evidence.setdefault("ball_states",
                                                       _ball_state_counts(self.ball_tracker, release.end_frame, followup.end_frame))
                        followup.evidence.setdefault("team_sim", None)
                        events.append(followup)

        return events

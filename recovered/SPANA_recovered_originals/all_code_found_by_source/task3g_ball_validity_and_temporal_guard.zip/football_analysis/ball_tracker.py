"""
ball_tracker.py -- BallTracker: the persistent ball-identity state machine.

Rewritten (this task) to replace the old "pick the single highest-confidence
detection per frame, linearly interpolate short post-hoc gaps" version with
real temporal continuity: a single BallTrackState moved through
CONFIRMED / PREDICTED / OCCLUDED / REACQUIRING / LOST by a constant-
velocity/acceleration motion model (ball_motion.py), gated candidate
scoring (ball_gating.py), and a staged cheap-to-expensive re-detection
search (ball_roi_search.py) that only escalates when the cheap tier-A
detection (already run once per frame by the caller) comes up empty.

Independence from the player tracker, restated: this file, and every file
it imports, lives under football_analysis/ and imports nothing from
player_tracker/. The tracked player's bounding box is accepted as a plain
argument (read-only context, per rule #6/#7) purely to bias the search ROI
and give a mild proximity bonus in scoring -- never to decide the ball's
identity, and there is no code path anywhere here that could write
anything back into a PlayerIdentity or PlayerTracker object, because this
module holds no reference to either.

Rule #14 (never overwrite confirmed history with a weak candidate): this
class keeps `last_confirmed` completely separate from the per-frame
`current` state it returns/appends to `history` -- `last_confirmed` is
only ever reassigned on an actual CONFIRMED or REACQUIRING->CONFIRMED
transition, never on a PREDICTED/OCCLUDED frame.

STATIC-DECOY TASK additions:
  - Owns a BackgroundMotionEstimator (ball_background_motion.py) and a
    StaticDecoyTracker (ball_static_decoy.py), both purely internal to this
    class -- callers (event_detector.py/engine.py, both frozen this task)
    are completely unaware of them and need no changes: `step()`'s public
    signature is unchanged, since `frame`/`frame_idx`/`player_bbox` were
    already being passed in.
  - Rule #11 (track init must be harder than "trust the first high-
    confidence detection"): `_advance()` used to treat a cold start
    (never-yet-confirmed track) exactly like a trivial 1-2 frame gap in an
    already-established track, because both start from
    `frames_since_confirmed == 0`. That conflation is exactly what let one
    single accepted candidate reach CONFIRMED immediately at frame 0. Cold
    start (`self.last_confirmed is None`) now requires its own, longer
    evidence-window streak (`init_confirm_frames`) before the first-ever
    CONFIRMED -- separate from, and never weakening, the existing
    `gap<=2` leniency used for genuine short gaps in a track that has
    already been established at least once.
"""
from dataclasses import replace

from .ball_track_state import BallTrackState
from .ball_motion import BallMotionModel
from .ball_gating import BallCandidateScorer
from .ball_roi_search import StagedBallSearch, CONTACT_CLUSTER_RADIUS_PX
from .ball_confidence import BallConfidenceConfig, score_confirmed, score_predicted, score_reacquiring
from .ball_background_motion import BackgroundMotionEstimator
from .ball_static_decoy import StaticDecoyTracker
from .ball_temporal_bridge import bridge_occlusions
from .geometry import center, size as _box_size, foot_point, dist as _dist


class BallTracker:
    def __init__(self, ball_id="ball_01", fps=25.0, light_model=None, heavy_model_factory=None, config=None):
        cfg = config or {}
        self.ball_id = ball_id
        self.fps = fps
        self.motion = BallMotionModel(history_len=cfg.get("motion_history_len", 6))
        self.scorer = BallCandidateScorer(cfg)
        self.search = StagedBallSearch(light_model, heavy_model_factory, cfg) if light_model is not None else None
        self.conf_cfg = BallConfidenceConfig(
            confirmed_threshold=cfg.get("confirmed_threshold", 0.80),
            predicted_threshold=cfg.get("predicted_threshold", 0.50),
            prediction_decay_per_frame=cfg.get("prediction_decay_per_frame", 0.10),
            max_predicted_frames=cfg.get("max_predicted_frames", 10),
        )
        self.heavy_trigger_frames = cfg.get("heavy_trigger_frames", 8)       # rule #12's configurable X
        self.reacquire_required_frames = cfg.get("reacquire_required_frames", 3)
        self.reacquire_anchor_tol = cfg.get("reacquire_anchor_tol", 45.0)
        self.tier_a_accept_thresh = cfg.get("tier_a_accept_thresh", 0.35)
        self.search_accept_thresh = cfg.get("search_accept_thresh", 0.28)
        # STATIC-DECOY TASK rule #11: how many consistent, low-decoy-score
        # hits are required before the very first-ever CONFIRMED -- distinct
        # from reacquire_required_frames, which governs a *returning* track.
        self.init_confirm_frames = cfg.get("init_confirm_frames", 5)

        self.state = BallTrackState(ball_id=ball_id)
        self.last_confirmed = None          # rule #14: separate, never overwritten by a weak candidate
        self.current_candidate = None       # the most recent scored candidate, confirmed or not
        self.history = []                   # one BallTrackState snapshot per processed frame
        self.transition_log = []            # rule #19
        self._reacquire_streak = 0
        self._reacquire_anchor = None
        self._last_search_meta = {"stage": None, "roi": None, "scored": []}
        # TASK 3G: one record per step() -- the accepted candidate (if any), the
        # stage that found it, and whether a trustworthy prediction gated it.
        # Read only by ball_interaction_credibility.py at finalize time; never
        # used by this class's own state machine.
        self.accept_log = []

        # STATIC-DECOY TASK: purely internal evidence-gathering layer (see
        # class docstring). Neither module is imported by, or talks to,
        # anything outside this file.
        self.background_motion = BackgroundMotionEstimator()
        self.static_decoy = StaticDecoyTracker()
        self._bg_cum = (0.0, 0.0)   # running cumulative background/camera displacement, full-res px

    # ------------------------------------------------------------ helpers
    def _last_scale(self):
        if self.last_confirmed is not None and self.last_confirmed.bbox is not None:
            w, h = _box_size(self.last_confirmed.bbox)
            return max(w, h) / 2.0
        return 20.0

    def _log(self, frame_idx, prev_state, new_state, reason, **extra):
        entry = {"frame": frame_idx, "previous_state": prev_state, "new_state": new_state, "reason": reason}
        entry.update(extra)
        self.transition_log.append(entry)

    def _predicted_bbox(self, cx, cy):
        r = self._last_scale()
        return [cx - r, cy - r, cx + r, cy + r]

    def _overlaps_player(self, cx, cy, player_bbox):
        if player_bbox is None:
            return False
        x1, y1, x2, y2 = player_bbox
        pad = 15
        return (x1 - pad) <= cx <= (x2 + pad) and (y1 - pad) <= cy <= (y2 + pad)

    # --------------------------------------------------------------- step
    def step(self, frame_idx, frame, base_candidates, player_bbox=None, allow_heavy=True, is_cut=False,
              other_boxes=None):
        """other_boxes (this task, optional, read-only): person boxes that
        are NOT the tracked player, for the SAME frame -- exactly the same
        list event_detector.py already computes and passes to this call
        site (previously used only afterwards, for team classification).
        Used purely as crowd-aware BALL search/scoring context (sections
        #3-#5, #9, #11, #15): where to look harder, and a mild scoring
        bonus for a candidate near a contact region. Omitting it (None, the
        default) reproduces the exact pre-task behaviour for any caller
        that doesn't pass it."""
        prev_state = self.state.state
        predicted_pos = self.motion.predict(frame_idx)
        last_box = self.last_confirmed.bbox if self.last_confirmed is not None else None
        fsc = self.state.frames_since_confirmed

        # BALL-TRACKING ROBUSTNESS TASK (sections #3, #9): is this frame
        # plausibly a crowded/contact moment at all? Same radius the ROI
        # search stages use for their own contact-cluster gate (section #5),
        # so "crowd detected" means the same thing everywhere in this class.
        focus = predicted_pos if predicted_pos is not None else (center(player_bbox) if player_bbox is not None else None)
        nearby_other_boxes = []
        if other_boxes and focus is not None:
            for b in other_boxes:
                if _dist(center(b), focus) <= CONTACT_CLUSTER_RADIUS_PX:
                    nearby_other_boxes.append(b)
        nearby_feet_points = [foot_point(b) for b in nearby_other_boxes] or None
        crowd_context = len(nearby_other_boxes) >= 1

        # STATIC-DECOY TASK (rule #4): estimate this frame's camera/background
        # motion independent of any object, and fold it into a running
        # cumulative displacement -- ball_static_decoy.py compares a
        # candidate's own displacement between two frames against the
        # background's displacement over that SAME interval, which is why a
        # cumulative sum (rather than just this frame's delta) is what needs
        # to be threaded through.
        bg_dx, bg_dy = self.background_motion.step(frame)
        self._bg_cum = (self._bg_cum[0] + bg_dx, self._bg_cum[1] + bg_dy)
        bg_cum = self._bg_cum

        def decoy_evaluator(box):
            return self.static_decoy.evaluate(box, frame_idx, frame, bg_cum, player_bbox)

        # A constant-velocity/acceleration extrapolation held for many frames
        # (past the PREDICTED patience window) stops being trustworthy as a
        # POSITION -- held velocity compounds into a runaway guess, especially
        # right after a fast release. Past that point this module stops
        # gating candidates against it (predicted_for_gating=None -- distance
        # is simply not scored) and leans on the other signals instead:
        # velocity-consistency against the last CONFIRMED point (which scales
        # with elapsed frames, so a legitimate long-distance pass/shot isn't
        # penalised) and size/player-proximity. This is what actually lets
        # stage #4's "search the whole frame" mean something -- without this,
        # every stage was still silently hard-gated by the same stale guess.
        prediction_trustworthy = fsc <= self.conf_cfg.max_predicted_frames
        predicted_for_gating = predicted_pos if prediction_trustworthy else None

        best, scored = self.scorer.best(base_candidates, predicted_for_gating, fsc, last_box, player_bbox,
                                         allow_cut=is_cut, min_conf=0.10, accept_thresh=self.tier_a_accept_thresh,
                                         decoy_evaluator=decoy_evaluator, nearby_feet_points=nearby_feet_points)
        stage_used = "tier_a_full_frame" if best is not None else None

        if best is None and self.search is not None:
            # BALL-TRACKING ROBUSTNESS TASK (section #9's escalation ladder,
            # made crowd-aware per section #3): in a detected contact
            # cluster, a miss is far more likely to be genuine occlusion
            # (ball hidden between/behind bodies) than an empty patch of
            # grass being briefly under-detected, so it's worth reaching for
            # the heavier detector sooner -- never on every miss, only when
            # BOTH the ordinary frame-count trigger's halfway point is
            # reached AND a real nearby cluster exists this frame.
            heavy_trigger = self.heavy_trigger_frames
            if crowd_context:
                heavy_trigger = max(3, self.heavy_trigger_frames // 2)
            use_heavy = allow_heavy and (fsc + 1) >= heavy_trigger
            # skip straight past the ROI-around-prediction stages once that
            # prediction is no longer trustworthy -- searching around a
            # runaway guess wastes an inference and won't find the real ball
            search_candidates, stage_used = self.search.search(
                frame, predicted_for_gating, self._last_scale(), player_bbox, use_heavy=use_heavy,
                other_boxes=other_boxes)
            if search_candidates:
                best, scored2 = self.scorer.best(search_candidates, predicted_for_gating, fsc, last_box, player_bbox,
                                                  allow_cut=is_cut, min_conf=0.03, accept_thresh=self.search_accept_thresh,
                                                  decoy_evaluator=decoy_evaluator, nearby_feet_points=nearby_feet_points)
                scored = scored + scored2
            if best is None:
                stage_used = stage_used or "no_candidates_any_stage"

        self._last_search_meta = {"stage": stage_used, "scored": scored}

        # STATIC-DECOY TASK rule #6: fold EVERY candidate proposed this frame
        # (accepted or not, winning or not) into that screen location's
        # history -- a recurring decoy needs to be recognised even on the
        # frames it loses out to something else, and a location's history
        # must accumulate whether or not it happened to win the frame.
        # Called only AFTER all of this frame's evaluate() calls above, per
        # ball_static_decoy.py's own documented ordering requirement.
        # De-duplicated by screen-location bin first: tier-A and the staged
        # ROI search can both independently propose essentially the same
        # physical box in one frame, and observing the same location twice
        # inside a single frame would spuriously read as "no motion between
        # these two observations" and wrongly erode its trust.
        seen_bins = set()
        for sc in scored:
            key = self.static_decoy._bin_key(*center(sc.box), bg_cum)
            if key in seen_bins:
                continue
            seen_bins.add(key)
            self.static_decoy.observe(sc.box, frame_idx, frame, bg_cum)

        new_state = self._advance(frame_idx, best, predicted_pos, player_bbox)
        self.accept_log.append({
            "frame": frame_idx,
            "box": list(best.box) if best is not None else None,
            "conf": best.conf if best is not None else None,
            "stage": stage_used if best is not None else None,
            "prediction_trusted": predicted_for_gating is not None,
            "player_bbox": list(player_bbox) if player_bbox is not None else None,
        })
        if new_state.state != prev_state:
            self._log(frame_idx, prev_state, new_state.state,
                       reason=self._transition_reason(prev_state, new_state.state, best, stage_used),
                       detection_confidence=new_state.detection_confidence,
                       prediction_confidence=new_state.prediction_confidence,
                       candidate_distance=(best.distance_from_prediction if best else None),
                       candidate_score=(best.score if best else None))

        self.history.append(new_state)
        return new_state

    def _transition_reason(self, prev, new, best, stage_used):
        if best is not None and new == "CONFIRMED":
            return f"candidate stable/accepted via {stage_used}" if prev != "CONFIRMED" else "continued tracking"
        if best is not None and new == "REACQUIRING":
            return f"candidate plausible via {stage_used}, confirming ({self._reacquire_streak}/{self.reacquire_required_frames})"
        if best is None and new in ("PREDICTED", "OCCLUDED"):
            return "detector miss -- extrapolating from motion model"
        if new == "LOST":
            return f"no usable evidence for {self.state.frames_since_confirmed} frames -- exceeded prediction patience"
        return "state update"

    def _advance(self, frame_idx, best, predicted_pos, player_bbox):
        ts = round(frame_idx / self.fps, 4)

        if best is not None:
            c = center(best.box)
            was_tracking_cleanly = self.state.state == "CONFIRMED" and self.state.frames_since_confirmed == 0

            if was_tracking_cleanly:
                # ordinary frame-to-frame continuation -- no gap to bridge
                self._reacquire_streak = 0
                self._reacquire_anchor = None
                return self._confirm(frame_idx, ts, best, c)

            # coming back from PREDICTED/OCCLUDED/REACQUIRING/LOST: require a
            # run of consistent hits before trusting it (rule #10). How many
            # frames that run needs to be scales with how long the gap was --
            # rule #10 ties the requirement to AMBIGUITY, not a fixed count,
            # and a candidate landing almost exactly on a 1-frame-old
            # prediction is far less ambiguous than one reappearing after a
            # long gap near the edge of the search radius. A trivial 1-2
            # frame detector blip (still well within the motion model's
            # short-term accuracy) only needs a single consistent hit;
            # anything longer falls back to the full configured requirement.
            #
            # STATIC-DECOY TASK bugfix (rule #11): `self.state.frames_since_
            # confirmed` starts at 0 by dataclass default, which used to make
            # a TRUE cold start (this ball identity has never been confirmed
            # even once) indistinguishable from a trivial 1-2 frame gap in an
            # already-established track -- both hit `gap <= 2` and required
            # only a single accepted candidate. That is exactly how one
            # cold-start decoy detection reached CONFIRMED immediately at
            # frame 0. Track initialisation now requires its own longer
            # evidence-window streak (init_confirm_frames, rule #11: "observe
            # multiple frames -> evaluate motion/background/pitch context ->
            # CONFIRMED", not "immediately trust the first high-confidence
            # detection"). This never weakens the existing gap<=2 leniency,
            # which still applies exactly as before once the track has been
            # confirmed at least once.
            if self.last_confirmed is None:
                required = self.init_confirm_frames
            else:
                gap = self.state.frames_since_confirmed
                required = 1 if gap <= 2 else self.reacquire_required_frames

            if self._reacquire_anchor is not None and \
                    ((c[0] - self._reacquire_anchor[0]) ** 2 + (c[1] - self._reacquire_anchor[1]) ** 2) ** 0.5 < self.reacquire_anchor_tol:
                self._reacquire_streak += 1
            else:
                self._reacquire_streak = 1
            self._reacquire_anchor = c

            if self._reacquire_streak >= required:
                return self._confirm(frame_idx, ts, best, c)

            conf = score_reacquiring(best.score, self._reacquire_streak, required)
            state = BallTrackState(
                ball_id=self.ball_id, frame_number=frame_idx, timestamp=ts,
                bbox=list(best.box), centre_x=c[0], centre_y=c[1],
                velocity_x=self.motion.velocity[0], velocity_y=self.motion.velocity[1],
                speed=self.motion.speed(), acceleration=self.motion.acceleration,
                trajectory=list(self.state.trajectory),
                confidence=conf, detection_confidence=best.conf, prediction_confidence=0.0,
                frames_since_confirmed=self.state.frames_since_confirmed + 1, state="REACQUIRING",
            )
            self.current_candidate = best
            self.state = state
            return state

        # no accepted candidate this frame at all
        self._reacquire_streak = 0
        fsc = self.state.frames_since_confirmed + 1

        if predicted_pos is not None and fsc <= self.conf_cfg.max_predicted_frames:
            cx, cy = predicted_pos
            occluded = self._overlaps_player(cx, cy, player_bbox)
            conf = score_predicted(fsc, self.conf_cfg)
            state = BallTrackState(
                ball_id=self.ball_id, frame_number=frame_idx, timestamp=ts,
                bbox=self._predicted_bbox(cx, cy), centre_x=cx, centre_y=cy,
                velocity_x=self.motion.velocity[0], velocity_y=self.motion.velocity[1],
                speed=self.motion.speed(), acceleration=self.motion.acceleration,
                trajectory=list(self.state.trajectory),
                confidence=conf, detection_confidence=0.0, prediction_confidence=conf,
                frames_since_confirmed=fsc, state=("OCCLUDED" if occluded else "PREDICTED"),
            )
            self.state = state
            return state

        state = BallTrackState(
            ball_id=self.ball_id, frame_number=frame_idx, timestamp=ts,
            bbox=None, centre_x=None, centre_y=None,
            velocity_x=self.motion.velocity[0], velocity_y=self.motion.velocity[1],
            speed=0.0, acceleration=0.0, trajectory=list(self.state.trajectory),
            confidence=0.0, detection_confidence=0.0, prediction_confidence=0.0,
            frames_since_confirmed=fsc, state="LOST",
        )
        self.state = state
        return state

    def _confirm(self, frame_idx, ts, best, c):
        self.motion.update(frame_idx, c[0], c[1])
        trajectory = list(self.state.trajectory) + [(frame_idx, c[0], c[1])]
        trajectory = trajectory[-200:]
        run_length = 1
        for (f, _, _) in reversed(trajectory[:-1]):
            if f == trajectory[-1][0] - run_length:
                run_length += 1
            else:
                break
        conf = score_confirmed(best.conf, best.score, run_length)
        state = BallTrackState(
            ball_id=self.ball_id, frame_number=frame_idx, timestamp=ts,
            bbox=list(best.box), centre_x=c[0], centre_y=c[1],
            velocity_x=self.motion.velocity[0], velocity_y=self.motion.velocity[1],
            speed=self.motion.speed(), acceleration=self.motion.acceleration,
            trajectory=trajectory, confidence=conf, detection_confidence=best.conf,
            prediction_confidence=0.0, frames_since_confirmed=0, state="CONFIRMED",
        )
        self.current_candidate = best
        self.last_confirmed = state       # rule #14 -- only ever set here
        self.state = state
        self._reacquire_streak = 0
        self._reacquire_anchor = None
        return state

    # ---------------------------------------------------- downstream API
    def finalize_interpolation(self):
        """BALL-TRACKING ROBUSTNESS TASK (sections #6-#8): runs the OFFLINE,
        bidirectional occlusion bridge over the now-complete `self.history`,
        once, after the whole clip's frame-by-frame loop has already
        finished. This is intentionally separate from the online step()/
        _advance() path above -- nothing in this method's call graph can
        affect a frame's state at the moment step() produced it; it only
        ever revisits already-recorded LOST frames using frames that came
        AFTER them (impossible for the real-time code to have used), and
        only ever upgrades a subset of them to OCCLUDED (never CONFIRMED --
        see ball_temporal_bridge.py's docstring for why OCCLUDED, and never
        REACQUIRING, which requires a real detector box by construction).

        `self.transition_log` (rule #19) gets one entry per bridged run so
        the bridge's effect is auditable exactly like every other state
        transition, without changing that log's existing shape (same keys:
        frame/previous_state/new_state/reason, plus the bridge's own
        `bridge_info` under `extra`)."""
        bridged = bridge_occlusions(self.history, self.fps)
        for old, new in zip(self.history, bridged):
            if new is old:
                continue
            self._log(new.frame_number, old.state, new.state,
                       reason="offline_temporal_bridge: forward/backward trajectories agreed",
                       bridge_info=new.bridge_info)
        self.history = bridged

    def velocity_at(self, idx, window=3):
        """Smoothed px/frame velocity from nearby frames with a real position
        (CONFIRMED, REACQUIRING, or PREDICTED all count -- LOST/OCCLUDED
        carry no position and are excluded)."""
        pts = []
        lo, hi = max(0, idx - window), min(len(self.history), idx + window + 1)
        for k in range(lo, hi):
            o = self.history[k]
            if o.state not in ("LOST", "OCCLUDED") and o.centre_x is not None:
                pts.append((o.frame_number, (o.centre_x, o.centre_y)))
        if len(pts) < 2:
            return None
        (f0, c0), (f1, c1) = pts[0], pts[-1]
        if f1 == f0:
            return None
        return ((c1[0] - c0[0]) / (f1 - f0), (c1[1] - c0[1]) / (f1 - f0))

    def stats(self):
        n = len(self.history) or 1
        counts = {s: 0 for s in ("CONFIRMED", "PREDICTED", "OCCLUDED", "REACQUIRING", "LOST")}
        for o in self.history:
            counts[o.state] = counts.get(o.state, 0) + 1
        return {k: round(100 * v / n, 1) for k, v in counts.items()}

    def reacquisition_count(self):
        return sum(1 for e in self.transition_log if e["new_state"] == "CONFIRMED" and e["previous_state"] in
                   ("REACQUIRING", "LOST", "OCCLUDED", "PREDICTED"))

    def mean_consecutive_lost(self):
        runs, cur = [], 0
        for o in self.history:
            if o.state == "LOST":
                cur += 1
            elif cur > 0:
                runs.append(cur); cur = 0
        if cur > 0:
            runs.append(cur)
        return round(sum(runs) / len(runs), 2) if runs else 0.0

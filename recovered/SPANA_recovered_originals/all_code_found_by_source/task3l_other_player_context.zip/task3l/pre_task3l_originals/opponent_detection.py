"""
opponent_detection.py -- OPPONENT-DETECTION TASK: OTHER-PLAYER DETECTION,
sitting strictly downstream of the frozen selected-player tracker, per the
task's own architecture:

    FROZEN SELECTED-PLAYER TRACKER
        |
    READ-ONLY target player position
        |
    OTHER-PLAYER DETECTION            <- this module
        |
    opponent/team context             (event_detector.py, UNCHANGED)
        |
    event reasoning                   (possession/event/SHOT logic, UNCHANGED)

Nothing here imports player_tracker/, writes to it, or influences the
selected player's identity/bbox -- the tracked bbox is accepted everywhere
as a plain, read-only tuple. Nothing here changes team-classification
ACCURACY either (rule #9): the existing, frozen TeamClassifier is only
ever called read-only, for reporting -- see the module docstring's rule-#1
finding below on why that matters.

RULE #1 DIAGNOSIS, restated (see diagnostics/inspect_missing_defender.py
and validation/reports/opponent_detection_report.md for the full writeup):
directly re-inspecting the critical challenge window in
known_match_carry_0_20 (frames 14-20) found that the person detector
ALREADY produces a separate box for the navy defender at every single one
of those frames -- real confidence (0.15-0.5+), surviving both the
confidence threshold and NMS, and already passing engine.py's own
IoU<0.3 "other player" filter. It never disappears in this window. What
actually breaks contested-possession logic for this specific clip is
downstream: the existing TeamClassifier reads this navy-kit box as
"TEAMMATE" (sim ~0.55-0.67) at every one of these frames, not a detection
gap. That is a team-colour-classification bug, not a detection-recall bug,
and fixing IT is out of this task's scope (rule #9 says to use the
existing classifier as-is) -- it is reported separately, deliberately left
unfixed here, exactly as Case 1's own opponent-association gap was left
unfixed in the prior ball-tracking task. This module still builds the
general detection-robustness improvements the task asks for (rules #3-14):
they generalise to the cases -- far-field small players, heavier overlap,
genuinely brief misses -- where recall truly is the bottleneck, even
though it did not turn out to be the bottleneck in this one clip.

Four pieces, corresponding to rules #3-#14:
  - LocalChallengeDetector: an optional, higher-sensitivity detection pass
    on an upscaled crop around the tracked player (rules #3/#4/#5), with a
    higher local NMS IoU tolerance so two heavily-overlapping-but-distinct
    people are less likely to be merged into one box by NMS (rule #6) --
    the smallest generalisable change; no pose/segmentation model is added
    (rule #11), since nothing in the rule #1 diagnosis showed a need for
    one strong enough to justify the extra weight.
  - classify_vs_target(): target-duplicate suppression (rule #7).
  - OtherPlayerTrackStore: short-lived temporal continuity across a brief
    detection gap (rule #8) -- decays fast (a handful of frames) and is
    never a source of new information on its own, only a bridge over the
    odd missed frame; a genuinely-gone player becomes UNCERTAIN, never an
    indefinitely-hallucinated box (rule H).
  - should_run_local_pass(): the proximity/crowding trigger (rule #12) so
    the expensive local pass does not run on every frame.

merge_other_player_candidates() is the single public entry point engine.py
calls. It takes the SAME full-frame low-res person detections engine.py
already computes (never re-detects those), optionally runs the local pass,
excludes target duplicates, folds in temporal bridging, and returns BOTH:
  (a) a plain list of xyxy boxes -- byte-for-byte the same shape `others`
      always was, so event_detector.py/possession/event logic need zero
      changes and keep doing their own (unchanged, frozen) team
      classification exactly as before; and
  (b) the full list of OpponentCandidate records with rule #13's
      structured confidence fields, for debugging/reporting only -- never
      consumed by the frozen event-reasoning path.
"""
from dataclasses import dataclass
from typing import Optional, List, Tuple

from .geometry import center, dist, iou, foot_point

TARGET_DUPLICATE_IOU = 0.5          # rule #7: boxes overlapping the tracked player this much or more
                                      # are treated as re-detections of the SAME person, not an opponent
LOCAL_PAD_X_SCALE = 1.5              # rule #3: local ROI horizontal padding, in units of the tracked box's own width
LOCAL_PAD_Y_SCALE = 1.2
LOCAL_IMGSZ = 960                    # rule #4: higher effective resolution than the frozen player detector's own imgsz
LOCAL_CONF = 0.10                    # rule #5: lower threshold, LOCAL ONLY -- never applied full-frame
LOCAL_NMS_IOU = 0.85                 # rule #6: higher IoU tolerance in the local ROI only, so two genuinely
                                      # distinct, heavily-overlapping people are less likely to be merged by NMS
                                      # (full-frame NMS, engine.py's own call, is completely untouched)
MERGE_DEDUPE_IOU = 0.55              # a local-ROI box this close to an already-known full-frame box is the
                                      # same physical detection, not a second person
MAX_BRIDGE_FRAMES = 3                # rule #8: temporal bridging is short-lived, never indefinite (rule H)
BRIDGE_MATCH_DIST = 70.0             # px, foot-point distance to associate a fresh detection with an existing track
CHALLENGE_BALL_RADIUS = 130.0        # rule #12 trigger: ball within this many px of the tracked player
CHALLENGE_CROWD_RADIUS = 160.0       # rule #12 trigger: an already-detected full-frame person this close counts
                                      # as visual crowding worth a closer look


@dataclass
class OpponentCandidate:
    """rule #13: separate, non-collapsed confidence fields."""
    box: Tuple[float, float, float, float]
    person_confidence: float
    temporal_confidence: float
    separate_from_target_confidence: float
    classification: str                    # "PERSON_DETECTED" | "TARGET_PLAYER_DUPLICATE"
    source: str                             # "FULL_FRAME" | "LOCAL_ROI" | "TEMPORAL_BRIDGE"
    team: Optional[str] = None
    team_confidence: Optional[float] = None
    track_id: Optional[str] = None


def classify_vs_target(box, tracked_bbox):
    """rule #7. Returns (classification, separate_from_target_confidence).
    Never mutates or reads anything beyond the plain box tuples -- the
    tracked-player box itself is never touched."""
    i = iou(box, tracked_bbox)
    classification = "TARGET_PLAYER_DUPLICATE" if i >= TARGET_DUPLICATE_IOU else "PERSON_DETECTED"
    return classification, max(0.0, 1.0 - i)


class LocalChallengeDetector:
    """rules #3-#6: a dynamic, higher-sensitivity local detection pass
    around the tracked player, used only when should_run_local_pass()
    says a challenge/crowding context makes it worthwhile (rule #12)."""

    def __init__(self, model, pad_x_scale=LOCAL_PAD_X_SCALE, pad_y_scale=LOCAL_PAD_Y_SCALE,
                 imgsz=LOCAL_IMGSZ, conf=LOCAL_CONF, nms_iou=LOCAL_NMS_IOU):
        self.model = model
        self.pad_x_scale = pad_x_scale
        self.pad_y_scale = pad_y_scale
        self.imgsz = imgsz
        self.conf = conf
        self.nms_iou = nms_iou

    def detect(self, frame, tracked_bbox):
        """Returns [(xyxy_in_full_frame_coords, confidence), ...]. Empty
        list on any degenerate crop -- never raises, never fabricates."""
        x1, y1, x2, y2 = tracked_bbox
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            return []
        pad_x, pad_y = w * self.pad_x_scale, h * self.pad_y_scale
        fh, fw = frame.shape[:2]
        cx1, cy1 = max(0.0, x1 - pad_x), max(0.0, y1 - pad_y)
        cx2, cy2 = min(float(fw), x2 + pad_x), min(float(fh), y2 + pad_y)
        if cx2 <= cx1 or cy2 <= cy1:
            return []
        crop = frame[int(cy1):int(cy2), int(cx1):int(cx2)]
        if crop.size == 0:
            return []
        res = self.model.predict(crop, classes=[0], conf=self.conf, iou=self.nms_iou,
                                  imgsz=self.imgsz, verbose=False)[0]
        out = []
        for b in res.boxes:
            c = float(b.conf[0])
            bx1, by1, bx2, by2 = (float(v) for v in b.xyxy[0])
            out.append(((bx1 + cx1, by1 + cy1, bx2 + cx1, by2 + cy1), c))
        return out


def should_run_local_pass(tracked_bbox, ball_centre, full_frame_other_boxes,
                           ball_radius=CHALLENGE_BALL_RADIUS, crowd_radius=CHALLENGE_CROWD_RADIUS):
    """rule #12. Pure heuristic, performs no detection itself. True when the
    ball is near the tracked player (a possible challenge/possession
    contest), or any already-detected full-frame person sits close enough
    to the tracked player to count as visual crowding -- catching exactly
    the case where a full-frame box for the challenger exists but is
    imprecise/low-confidence enough to be worth a closer look, without any
    per-clip flag."""
    tc = center(tracked_bbox)
    if ball_centre is not None and dist(tc, ball_centre) <= ball_radius:
        return True
    for b in full_frame_other_boxes:
        if dist(center(b), tc) <= crowd_radius:
            return True
    return False


@dataclass
class _OtherTrack:
    track_id: str
    last_frame: int
    last_box: tuple
    last_confidence: float
    misses: int


class OtherPlayerTrackStore:
    """rule #8: short-lived OTHER_PLAYER continuity across a brief
    detection gap. Entirely independent of, and invisible to, selected-
    player identity -- nothing here is fed back into player_tracker/ or
    the tracked player's own bbox."""

    def __init__(self, max_bridge_frames=MAX_BRIDGE_FRAMES, match_dist=BRIDGE_MATCH_DIST):
        self.max_bridge_frames = max_bridge_frames
        self.match_dist = match_dist
        self._tracks: dict = {}
        self._next_id = 0

    def _new_id(self):
        self._next_id += 1
        return f"opponent_candidate_{self._next_id:02d}"

    def step(self, frame_idx, fresh_detections):
        """fresh_detections: [(box, confidence, source), ...] already
        deduped/merged for this frame (source is "FULL_FRAME" or
        "LOCAL_ROI"). Returns a list of (box, confidence, temporal_confidence,
        source, track_id) -- every fresh detection (matched to an existing
        track or newly created), PLUS any track that just went missing this
        frame, bridged for up to max_bridge_frames with decaying
        temporal_confidence and source="TEMPORAL_BRIDGE". A track missing
        for longer than max_bridge_frames is dropped for good (rule H:
        genuinely gone means an absent/uncertain candidate, never an
        indefinitely-hallucinated one)."""
        unmatched_track_ids = set(self._tracks.keys())
        out = []

        for box, conf, source in fresh_detections:
            fp = foot_point(box)
            best_id, best_d = None, None
            for tid in unmatched_track_ids:
                tr = self._tracks[tid]
                d = dist(fp, foot_point(tr.last_box))
                if d <= self.match_dist and (best_d is None or d < best_d):
                    best_id, best_d = tid, d
            if best_id is not None:
                tid = best_id
                unmatched_track_ids.discard(tid)
            else:
                tid = self._new_id()
            self._tracks[tid] = _OtherTrack(track_id=tid, last_frame=frame_idx, last_box=box,
                                             last_confidence=conf, misses=0)
            out.append((box, conf, 1.0, source, tid))

        stale = []
        for tid in unmatched_track_ids:
            tr = self._tracks[tid]
            tr.misses += 1
            if tr.misses > self.max_bridge_frames:
                stale.append(tid)
                continue
            temporal_confidence = max(0.0, 1.0 - tr.misses / (self.max_bridge_frames + 1))
            bridged_conf = tr.last_confidence * temporal_confidence
            out.append((tr.last_box, bridged_conf, temporal_confidence, "TEMPORAL_BRIDGE", tid))

        for tid in stale:
            del self._tracks[tid]

        return out

    def reset(self):
        self._tracks = {}
        self._next_id = 0


def merge_other_player_candidates(frame, frame_idx, tracked_bbox, full_frame_persons, ball_centre,
                                   local_detector=None, track_store=None, team_classifier=None,
                                   force_local_pass=False):
    """The single integration point engine.py calls.

    full_frame_persons: [(xyxy, confidence), ...] -- ALL person boxes from
    the existing, unmodified full-frame combined pass (before the
    IoU-vs-tracked filter, so target-duplicate classification can be done
    uniformly here rather than twice).
    ball_centre: (x, y) or None -- read-only, for the rule #12 trigger only.
    local_detector: a LocalChallengeDetector, or None to disable the local
    pass entirely (e.g. in tests that only want to exercise merging/
    temporal logic).
    track_store: an OtherPlayerTrackStore, or None to disable temporal
    bridging (a fresh store is otherwise expected to persist across frames,
    owned by the caller).
    team_classifier: optional callable(frame, box) -> (team, sim), reused
    READ-ONLY purely to populate the report-only team/team_confidence
    fields (rule #9/#13) -- never required, never mutated, and never what
    the actual downstream pipeline uses (event_detector.py does its own,
    completely unaffected, team classification on the plain boxes this
    function returns).

    Returns (plain_boxes, records):
      plain_boxes: list[tuple] of xyxy boxes -- exactly engine.py's
        pre-existing `others` shape.
      records: list[OpponentCandidate] -- rich, debug/report-only.
    """
    full_frame_other = []
    duplicates = []
    for box, conf in full_frame_persons:
        classification, sep_conf = classify_vs_target(box, tracked_bbox)
        if classification == "TARGET_PLAYER_DUPLICATE":
            duplicates.append(OpponentCandidate(box, conf, 1.0, sep_conf, classification, "FULL_FRAME"))
        else:
            full_frame_other.append((box, conf))

    run_local = force_local_pass or (
        local_detector is not None and
        should_run_local_pass(tracked_bbox, ball_centre, [b for b, _ in full_frame_other])
    )

    local_boxes = local_detector.detect(frame, tracked_bbox) if run_local and local_detector is not None else []

    fresh = list(full_frame_other)
    fresh_sources = {id(b): "FULL_FRAME" for b, _ in full_frame_other}
    for lb, lc in local_boxes:
        classification, sep_conf = classify_vs_target(lb, tracked_bbox)
        if classification == "TARGET_PLAYER_DUPLICATE":
            duplicates.append(OpponentCandidate(lb, lc, 1.0, sep_conf, classification, "LOCAL_ROI"))
            continue
        # dedupe against anything the full-frame pass already found -- the
        # local pass exists to find what the full-frame pass MISSED, not to
        # double-count what it already caught
        if any(iou(lb, eb) >= MERGE_DEDUPE_IOU for eb, _ in full_frame_other):
            continue
        fresh.append((lb, lc))
        fresh_sources[id(lb)] = "LOCAL_ROI"

    fresh_with_source = [(b, c, fresh_sources.get(id(b), "FULL_FRAME")) for b, c in fresh]

    if track_store is not None:
        bridged = track_store.step(frame_idx, fresh_with_source)
    else:
        bridged = [(b, c, 1.0, s, None) for b, c, s in fresh_with_source]

    records = list(duplicates)
    plain_boxes = []
    for box, conf, temporal_confidence, source, track_id in bridged:
        _, sep_conf = classify_vs_target(box, tracked_bbox)
        team, team_conf = (None, None)
        if team_classifier is not None:
            team, team_conf = team_classifier(frame, box)
        records.append(OpponentCandidate(box, conf, temporal_confidence, sep_conf, "PERSON_DETECTED",
                                          source, team=team, team_confidence=team_conf, track_id=track_id))
        plain_boxes.append(box)

    return plain_boxes, records

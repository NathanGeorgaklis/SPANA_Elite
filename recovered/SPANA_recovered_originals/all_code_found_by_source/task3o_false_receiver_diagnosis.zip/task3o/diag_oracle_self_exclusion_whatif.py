"""Task 3O ORACLE WHAT-IF (diagnostic only, never a fix). Same wrappers as
diag_receiver_arbitration_capture.py, plus: at the gold release (i_end=82), re-run the unchanged
resolve_release on a COPY of the possession frames in which the two boxes visually verified to be
#23 himself at engine frame 88 are removed from the other-player list (nearest-other fields
recomputed exactly as event_detector.process_frame does). The scored pipeline output is untouched.

Original docstring: Task 3O diagnostic replay of Pass 2 (NO source file edited). Runtime wrappers only:

  * opponent_detection.merge_other_player_candidates (via football_analysis.engine's imported name)
      -> per-frame other-player records: box, source, person conf, store track id, target relation
  * TeamClassifier.classify -> per-box team label / similarities
  * FootballEventDetector.process_frame -> per-frame snapshot state/bbox, ball observation,
      target-ball distance, and the PossessionFrame the frozen detector produced
  * event_qualifier.resolve_release -> release inputs/outputs; plus, for diagnosis only, the DUEL
      candidate that build_duel_candidate() WOULD score from the same inputs (it is never built when
      arbitration short-circuits). Nothing computed here is fed back into the pipeline.

Outputs must be byte-identical to the scored Task 3N run except processing_time_s.

usage: python3 diag_receiver_arbitration_capture.py CLIP_ID MO_JSON VIDEO OUT_DIR LABEL
"""
import json
import sys
from dataclasses import asdict

import football_analysis.engine as eng
import football_analysis.possession_detector as pd
import football_analysis.event_detector as evd
import football_analysis.event_qualifier as evq
import football_analysis.event_arbitration as arb
from football_analysis.event_candidates import build_duel_candidate, DUEL_CONTACT_WINDOW_FRAMES
from football_analysis.opponent_detection import classify_vs_target

CAP = {"frames": {}, "releases": []}


def r(v, n=1):
    return None if v is None else round(float(v), n)


_orig_merge = eng.merge_other_player_candidates


def merge_wrap(frame, frame_idx, tracked_bbox, full_frame_persons, ball_centre, **kw):
    boxes, records = _orig_merge(frame, frame_idx, tracked_bbox, full_frame_persons, ball_centre, **kw)
    CAP["frames"].setdefault(frame_idx, {})["merge"] = {
        "tracked_bbox": [r(v) for v in tracked_bbox],
        "ball_centre_tierA": [r(v) for v in ball_centre] if ball_centre else None,
        "records": [{"box": [r(v) for v in rec.box], "person_conf": r(rec.person_confidence, 3),
                     "temporal_conf": r(rec.temporal_confidence, 3), "classification": rec.classification,
                     "separate_from_target": r(rec.separate_from_target_confidence, 3),
                     "source": rec.source, "track_id": rec.track_id} for rec in records],
        "plain_boxes": [[r(v) for v in b] for b in boxes],
    }
    return boxes, records


eng.merge_other_player_candidates = merge_wrap

_cur = {"frame": None}
_orig_classify = pd.TeamClassifier.classify


def classify_wrap(self, frame, box, frame_number=None):
    team, conf = _orig_classify(self, frame, box, frame_number=frame_number)
    if frame_number is not None:
        CAP["frames"].setdefault(frame_number, {}).setdefault("team", []).append(
            {"box": [r(v) for v in box], "team": team, "conf": r(conf, 3)})
    return team, conf


pd.TeamClassifier.classify = classify_wrap

_orig_pf = evd.FootballEventDetector.process_frame


def pf_wrap(self, frame, snapshot, ball_candidates, other_boxes, is_cut=False):
    pframe = _orig_pf(self, frame, snapshot, ball_candidates, other_boxes, is_cut=is_cut)
    bo = self.ball_tracker.history[-1] if self.ball_tracker.history else None
    CAP["frames"].setdefault(snapshot.frame_number, {})["detector"] = {
        "snapshot_state": snapshot.tracking_state, "snapshot_bbox": [r(v) for v in snapshot.bbox],
        "ball_state": getattr(bo, "state", None),
        "ball_centre": [r(getattr(bo, "centre_x", None)), r(getattr(bo, "centre_y", None))] if bo is not None else None,
        "ball_conf": r(getattr(bo, "confidence", None), 3),
        "possession": {k: (r(v) if isinstance(v, float) else v) for k, v in asdict(pframe).items()},
    }
    return pframe


evd.FootballEventDetector.process_frame = pf_wrap

_orig_resolve = evq.resolve_release


def resolve_wrap(ball_tracker, possession_frames, i_end, fps=25.0, log=None, pitch_context=None,
                 keeper_candidate=None, seg_start=None, prior_segment_evidence=None):
    ev = _orig_resolve(ball_tracker, possession_frames, i_end, fps=fps, log=log, pitch_context=pitch_context,
                       keeper_candidate=keeper_candidate, seg_start=seg_start,
                       prior_segment_evidence=prior_segment_evidence)
    incoming_px_frame = (prior_segment_evidence or {}).get("incoming_speed_px_per_frame")
    incoming_px_s = incoming_px_frame * fps if incoming_px_frame is not None else None
    segment_frames = (prior_segment_evidence or {}).get("segment_frames")
    opp_d = arb._nearest_by_team_in_range(possession_frames, i_end - DUEL_CONTACT_WINDOW_FRAMES,
                                          i_end + DUEL_CONTACT_WINDOW_FRAMES, "OPPONENT")
    duel = build_duel_candidate(incoming_px_s, segment_frames, opp_d)   # diagnostic only, discarded
    if i_end == 82 and possession_frames and len(possession_frames) > 93:
        import copy
        from football_analysis.geometry import foot_point, dist
        ORACLE_SELF = {88: [(834.1, 293.2, 873.7, 362.4), (848.9, 292.0, 872.8, 332.2)]}
        pfs = copy.deepcopy(possession_frames)
        for f, selfboxes in ORACLE_SELF.items():
            fr = CAP["frames"][f]
            bc = fr["detector"]["ball_centre"]
            kept = [t for t in fr.get("team", []) if not any(abs(t["box"][0] - b[0]) < 0.2 and abs(t["box"][1] - b[1]) < 0.2 for b in selfboxes)]
            nd = nt = ntd = nud = None
            for t in kept:
                d = dist(foot_point(t["box"]), bc)
                if nd is None or d < nd:
                    nd, nt = d, t["team"]
                if t["team"] == "TEAMMATE" and (ntd is None or d < ntd):
                    ntd = d
                elif t["team"] == "UNKNOWN" and (nud is None or d < nud):
                    nud = d
            pf = pfs[f]
            pf.nearest_other_team, pf.nearest_other_distance = nt, nd
            pf.nearest_teammate_distance, pf.nearest_unknown_distance = ntd, nud
        cf = _orig_resolve(ball_tracker, pfs, i_end, fps=fps, log=None, pitch_context=pitch_context,
                           keeper_candidate=keeper_candidate, seg_start=seg_start,
                           prior_segment_evidence=prior_segment_evidence)
        CAP["ORACLE_WHATIF_self_boxes_removed"] = {
            "frame88_after_removal": {"nearest_other_team": pfs[88].nearest_other_team,
                                      "nearest_other_distance": pfs[88].nearest_other_distance},
            "type": cf.type, "end_frame": cf.end_frame, "success": cf.success,
            "resolution_reason": cf.evidence.get("resolution_reason"),
            "arbitration": cf.evidence.get("arbitration"),
            "receiver_evidence": cf.evidence.get("receiver_evidence")}
    CAP["releases"].append({
        "i_end": i_end, "seg_start": seg_start, "returned_type": ev.type, "returned_end": ev.end_frame,
        "resolution_reason": ev.evidence.get("resolution_reason"),
        "short_circuited": (ev.evidence.get("arbitration") or {}).get("short_circuited", False),
        "prior_segment_evidence": {k: v for k, v in (prior_segment_evidence or {}).items()
                                   if isinstance(v, (int, float, str, type(None)))},
        "DIAG_duel_candidate_if_built": duel.to_dict(),
        "DIAG_opponent_distance_at_contact": opp_d,
        "DIAG_incoming_speed_px_s": incoming_px_s, "DIAG_segment_frames": segment_frames,
    })
    return ev


evq.resolve_release = resolve_wrap

from event_engine_v1.analyze_target_player import analyze_target_player  # noqa: E402

clip_id, mo_json, video, out_dir, label = sys.argv[1:6]
analyze_target_player(clip_id, mo_json, video, out_dir, player_id="P1", player_label=label)
json.dump(CAP, open(f"{out_dir}/capture.json", "w"), default=str)
print("captured frames", len(CAP["frames"]), "releases", len(CAP["releases"]))

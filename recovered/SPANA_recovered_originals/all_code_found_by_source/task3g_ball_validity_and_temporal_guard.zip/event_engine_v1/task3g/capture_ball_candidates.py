"""Task 3G: read-only capture of per-frame ball-candidate internals.

Runtime-only instrumentation (no .py file on disk is modified): wraps
BallTracker.step in this process to record, for every frame, the accepted
candidate (box, detector conf, score, decoy evidence, search stage), the
resulting ball state, and the tracked player's box; captures the detector
instance to read possession frames and the post-finalize (post-bridge) ball
history. Then re-runs the exact Task 3A/3F orchestration path
(analyze_target_player's snapshot building + run_engine_pass_with_offset).

usage: python3 capture_ball_candidates.py <clip_id> <mo_json> <video> <player_id> <out_json>
       python3 capture_ball_candidates.py --registry <clip_id> <tracking_json> <video> <n_frames> <out_json>
         (validation-registry clips: same engine call as validation/lib/run_engine_pass.py)
"""
import sys, json
sys.path.insert(0, ".")
import football_analysis.engine as engine_mod
import football_analysis.ball_tracker as bt_mod
from football_analysis.event_detector import FootballEventDetector
from football_analysis.geometry import center, foot_point

REGISTRY = sys.argv[1] == "--registry"
if REGISTRY:
    clip_id, TRK, VIDEO, NF, OUT = sys.argv[2:7]
else:
    clip_id, MO, VIDEO, PID, OUT = sys.argv[1:6]
captured = {}
AUDIT = []


class CapturingDetector(FootballEventDetector):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        captured["det"] = self


engine_mod.FootballEventDetector = CapturingDetector
_orig_step = bt_mod.BallTracker.step


def _sc(sc):
    if sc is None:
        return None
    return {"box": [round(v, 1) for v in sc.box], "conf": round(sc.conf, 4), "score": round(sc.score, 4),
            "final_ball_score": sc.final_ball_score, "accepted": sc.accepted, "reason": sc.reason,
            "pitch_region": sc.pitch_region, "decoy_score": sc.decoy_score, "classification": sc.classification,
            "static_prob": sc.background_static_probability, "residual": sc.residual_object_motion,
            "dist_from_pred": sc.distance_from_prediction}


def step(self, frame_idx, frame, base_candidates, player_bbox=None, allow_heavy=True, is_cut=False, other_boxes=None):
    prev_state, prev_fsc = self.state.state, self.state.frames_since_confirmed
    had_confirmed = self.last_confirmed is not None
    out = _orig_step(self, frame_idx, frame, base_candidates, player_bbox=player_bbox, allow_heavy=allow_heavy,
                     is_cut=is_cut, other_boxes=other_boxes)
    meta = self._last_search_meta
    best = self.current_candidate if out.state in ("CONFIRMED", "REACQUIRING") else None
    AUDIT.append({
        "frame": frame_idx, "state": out.state, "prev_state": prev_state, "prev_fsc": prev_fsc,
        "had_confirmed_before": had_confirmed,
        "center": [out.centre_x, out.centre_y] if out.centre_x is not None else None,
        "bbox": out.bbox, "det_conf": out.detection_confidence, "confidence": out.confidence,
        "stage": meta.get("stage"), "best": _sc(best),
        "n_tier_a": len(base_candidates or []),
        "n_scored": len(meta.get("scored", [])),
        "player_bbox": list(player_bbox) if player_bbox is not None else None,
        "foot_point": list(foot_point(player_bbox)) if player_bbox is not None else None,
        "reacquire_streak": self._reacquire_streak,
        "motion_velocity": list(self.motion.velocity) if self.motion.velocity is not None else None,
        "bg_cum": list(self._bg_cum),
    })
    return out


bt_mod.BallTracker.step = step

if REGISTRY:
    sys.path.insert(0, "validation/lib")
    from run_engine_pass import run_engine_pass
    result = run_engine_pass(VIDEO, TRK, int(NF), OUT.replace(".json", "_events_raw.json"))
    S, fps = 0, 25.0
else:
    from event_engine_v1.analyze_target_player import run_engine_pass_with_offset
    from player_tracker_mo.tracklet_builder import build_tracklets_from_frame_log
    from player_tracker_mo.selected_player_resolver import resolve_selected_player
    from player_tracker_mo.offline_identity_enrichment import enrich_frame_log_with_offline_identity
    from player_tracker_mo.event_adapter import build_integrated_event_snapshots
    from player_tracker.appearance import AppearanceEncoder
    d = json.load(open(MO))
    fl, S, fps = d["frame_log"], d["summary"]["start_frame"], d["summary"]["fps"]
    tracklets = build_tracklets_from_frame_log(fl)
    res = resolve_selected_player(fl, tracklets, video_path=VIDEO, encoder=AppearanceEncoder(use_ocr=False))
    snaps = build_integrated_event_snapshots(enrich_frame_log_with_offline_identity(fl, res), fps, player_id=PID)
    result = run_engine_pass_with_offset(VIDEO, snaps, S, snaps[-1].frame_number + 1, OUT.replace(".json", "_events_raw.json"))
det = captured["det"]
post = det.ball_tracker.history
pfs = det._possession_frames
for i, row in enumerate(AUDIT):
    if i < len(post):
        row["post_bridge_state"] = post[i].state
    if i < len(pfs):
        row["ball_distance"] = pfs[i].ball_distance
        row["in_possession_raw"] = pfs[i].in_possession
seg = getattr(det, "_segmentation_frames", None)
if seg:
    for i, row in enumerate(AUDIT):
        if i < len(seg):
            row["in_possession_segmented"] = seg[i].in_possession
snaps_out = [{"frame": sn.frame_number, "tracking_state": sn.tracking_state} for sn in det._snapshots]
json.dump({"clip_id": clip_id, "start_frame": S, "fps": fps, "frames": AUDIT, "snapshots": snaps_out,
           "events": result.get("events", [])}, open(OUT, "w"), default=str)
print("wrote", OUT, len(AUDIT), "frames,", len(result.get("events", [])), "events")

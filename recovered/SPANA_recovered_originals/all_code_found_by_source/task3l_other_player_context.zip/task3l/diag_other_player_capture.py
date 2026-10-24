"""Task 3L diagnostic replay (NO engine file edited). Wraps, at runtime only:
  - engine.merge_other_player_candidates  -> per-frame candidate records (source, track id, person conf)
  - possession_detector.shirt_color_hist  -> per-box KitDescriptor (LAB, pixels, grass fraction)
  - TeamClassifier._decide                -> teammate_sim / opponent_sim inputs and decision
  - TeamClassifier.classify               -> final label, contamination, team_a/team_b references
Re-runs analyze_target_player into a separate directory. Outputs must be
byte-identical to the scored run except the processing-time field.

usage: python3 -m event_engine_v1.task3l.diag_other_player_capture CLIP_ID MO_JSON VIDEO OUT_DIR CAPTURE_JSON
"""
import json, sys
import numpy as np
import football_analysis.engine as eng
import football_analysis.possession_detector as pd
from football_analysis.geometry import iou

CAP = {"merge": [], "classify": []}
_ctx = {}

_orig_merge = eng.merge_other_player_candidates
def merge_wrap(frame, frame_idx, tracked_bbox, full_frame_persons, ball_centre, **kw):
    boxes, records = _orig_merge(frame, frame_idx, tracked_bbox, full_frame_persons, ball_centre, **kw)
    CAP["merge"].append({"frame": frame_idx, "tracked_bbox": [round(float(v), 1) for v in tracked_bbox],
                         "n_full_frame_persons": len(full_frame_persons),
                         "records": [{"box": [round(float(v), 1) for v in r.box], "person_conf": round(float(r.person_confidence), 3),
                                      "temporal_conf": round(float(r.temporal_confidence), 3), "classification": r.classification,
                                      "source": r.source, "track_id": r.track_id} for r in records]})
    return boxes, records
eng.merge_other_player_candidates = merge_wrap

_orig_hist = pd.shirt_color_hist
def hist_wrap(frame, box, avoid_side_of=None):
    d = _orig_hist(frame, box, avoid_side_of=avoid_side_of)
    _ctx.setdefault("descs", []).append({"avoid": avoid_side_of is not None,
        "desc": None if d is None else {"lab": [round(float(v), 1) for v in d.dominant_lab],
                                         "lab2": None if d.secondary_lab is None else [round(float(v), 1) for v in d.secondary_lab],
                                         "pixels": d.pixel_count, "grass": round(d.grass_fraction, 3), "low": d.low_evidence}})
    return d
pd.shirt_color_hist = hist_wrap

_orig_decide = pd.TeamClassifier._decide
def decide_wrap(teammate_sim, opponent_sim):
    r = _orig_decide(teammate_sim, opponent_sim)
    _ctx["decide"] = {"teammate_sim": round(float(teammate_sim), 4),
                      "opponent_sim": None if opponent_sim is None else round(float(opponent_sim), 4), "decision": r[0]}
    return r
pd.TeamClassifier._decide = staticmethod(decide_wrap)

_orig_classify = pd.TeamClassifier.classify
def classify_wrap(self, frame, box, frame_number=None):
    _ctx.clear()
    nb_before = len(self._team_b_trusted)
    ta, tb = self._team_a_centroid(), self._team_b_centroid()
    r = _orig_classify(self, frame, box, frame_number=frame_number)
    CAP["classify"].append({"frame": frame_number, "box": [round(float(v), 1) for v in box],
        "tracked_bbox": None if self._last_tracked_bbox is None else [round(float(v), 1) for v in self._last_tracked_bbox],
        "contamination_iou": None if self._last_tracked_bbox is None else round(float(iou(box, self._last_tracked_bbox)), 3),
        "descs": _ctx.get("descs", []), "decide": _ctx.get("decide"),
        "team_a_ref": None if ta is None else [round(v, 1) for v in ta],
        "team_b_ref": None if tb is None else [round(v, 1) for v in tb],
        "n_team_b_before": nb_before, "n_team_b_after": len(self._team_b_trusted),
        "label": r[0], "conf": round(float(r[1]), 4)})
    return r
pd.TeamClassifier.classify = classify_wrap

if __name__ == "__main__":
    clip, mo, video, out_dir, cap_path = sys.argv[1:6]
    from event_engine_v1.analyze_target_player import analyze_target_player
    analyze_target_player(clip, mo, video, out_dir, player_id="P1", player_label=sys.argv[6] if len(sys.argv) > 6 else "target player")
    json.dump(CAP, open(cap_path, "w"))
    print("merge frames", len(CAP["merge"]), "classify calls", len(CAP["classify"]))

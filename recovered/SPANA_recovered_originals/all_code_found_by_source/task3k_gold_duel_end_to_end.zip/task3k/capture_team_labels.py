"""Task 3K diagnostic replay. Wraps TeamClassifier.classify at runtime to log
(frame, box, label, sim); no engine file is edited. Re-runs the same
analyze_target_player call into a separate directory; outputs must be
byte-identical to the scored run (checked afterwards)."""
import json, sys
import football_analysis.possession_detector as pd
LOG = []
_orig = pd.TeamClassifier.classify
def wrapped(self, frame, box, frame_number=None):
    r = _orig(self, frame, box, frame_number=frame_number)
    LOG.append({"frame": frame_number, "box": [round(float(v), 1) for v in box], "label": r[0], "sim": None if r[1] is None else round(float(r[1]), 3)})
    return r
pd.TeamClassifier.classify = wrapped
from event_engine_v1.analyze_target_player import analyze_target_player
analyze_target_player('gold_duel_01', 'event_engine_v1/task3k/mo_tracking/GOLD_DUEL_01_syo_mar_3036_3048_mo_tracking_results.json',
                      'event_engine_v1/task3k/GOLD_DUEL_01_syo_mar_3036_3048.mp4', 'event_engine_v1/task3k/diag_replay',
                      player_id='P1', player_label='SOL #23 (seed box)')
json.dump(LOG, open('event_engine_v1/task3k/team_label_capture.json', 'w'))
print('captured', len(LOG))

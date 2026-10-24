"""Task 3C Step 5: broader (not just boundary-local) scan of ALL tracklets
built from shot_close_range_01's frame_log for any that pass near the
SEED location in the early (shot-window) frames, not just near the one
boundary at rel-frame 115/116. Read-only; no frozen file touched."""
import sys, json, math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
from player_tracker_mo.tracklet_builder import build_tracklets_from_frame_log

with open("event_engine_v1/shot_close_range_01/mo_tracking/shot_close_range_01_mo_tracking_results.json") as f:
    data = json.load(f)
fl = data["frame_log"]

tracklets = build_tracklets_from_frame_log(fl)
print(f"Total tracklets built: {len(tracklets)}")

SEED_BOX = (1474.0, 457.2, 1547.1, 634.6)
SEED_CX, SEED_CY = (SEED_BOX[0]+SEED_BOX[2])/2, (SEED_BOX[1]+SEED_BOX[3])/2

candidates = []
for tl in tracklets:
    early_obs = [(fr, bbox) for (fr, bbox) in tl.bbox_history if fr <= 20]
    if not early_obs:
        continue
    min_dist, min_frame, min_box = None, None, None
    for fr, bbox in early_obs:
        cx, cy = (bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2
        d = math.hypot(cx-SEED_CX, cy-SEED_CY)
        if min_dist is None or d < min_dist:
            min_dist, min_frame, min_box = d, fr, bbox
    team = max(tl.team_probability.items(), key=lambda kv: kv[1]) if tl.team_probability else (None, None)
    candidates.append((min_dist, tl.tracklet_id, tl.online_track_id, min_frame, min_box, team, tl.frame_start, tl.frame_end, len(tl.bbox_history)))

candidates.sort(key=lambda x: x[0])
print(f"\n{len(candidates)} tracklets have any observation in relative frames 1-20.")
print("Sorted by min distance to SEED center (all of them, since this determines TARGET_VISIBLE_IN_ANONYMOUS_TRACKLETS):\n")
for d, tid, otid, fr, box, team, fs, fe, n in candidates:
    print(f"  tracklet={tid:4} online_track_id={otid} min_dist={d:7.1f}px @frame={fr:3} box={tuple(round(v,1) for v in box)} team={team} span=[{fs}-{fe}] n_obs={n}")

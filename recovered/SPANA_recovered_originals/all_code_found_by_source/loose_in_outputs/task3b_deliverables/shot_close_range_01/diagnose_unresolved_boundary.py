"""Read-only diagnostic for Task 3B Step 3/8/12: shot_close_range_01's single
UNRESOLVED identity boundary. Calls only frozen, unmodified functions
(tracklet_builder.build_tracklets_from_frame_log, stitcher.evaluate_stitch) to
show every rejected candidate's score and reason, not just the fact that
resolve_selected_player() returned nothing. No frozen file is edited; this
script performs no YOLO/video decode of its own -- cheap, re-run of logic that
already ran inside analyze_target_player.py, for visibility only."""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from player_tracker_mo.tracklet_builder import build_tracklets_from_frame_log, populate_appearance_samples
from player_tracker_mo.selected_player_resolver import (
    _selected_history, _split_into_runs, _tracklet_to_fragment,
    _last_raw_state, MAX_STITCH_GAP_FRAMES, CANDIDATE_SEARCH_WINDOW_FRAMES,
    CANDIDATE_SPATIAL_RADIUS_PX_PER_FRAME, CANDIDATE_TOP_K,
)
from player_tracker_mo.stitcher import evaluate_stitch
from player_tracker.appearance import AppearanceEncoder
from collections import defaultdict

VIDEO_PATH = "/mnt/user-data/uploads/shot_close_range_01.mov"
_encoder = AppearanceEncoder(use_ocr=False)
_appearance_cache = {}

mo_json = REPO_ROOT / "event_engine_v1/shot_close_range_01/mo_tracking/shot_close_range_01_mo_tracking_results.json"
d = json.load(open(mo_json))
frame_log = d["frame_log"]

tracklets = build_tracklets_from_frame_log(frame_log)
tracklet_by_online_id = {t.online_track_id: t for t in tracklets}
frame_log_by_online_id = defaultdict(list)
for rec in frame_log:
    for tr in rec["tracks"]:
        frame_log_by_online_id[tr["online_track_id"]].append(tr)

hist = _selected_history(frame_log)
runs = _split_into_runs(hist)
print(f"n runs (continuous online_track_id segments for 'selected_player'): {len(runs)}")
for i, run in enumerate(runs):
    tid = run[0][1]
    t = tracklet_by_online_id.get(tid)
    print(f"  run {i}: online_track_id={tid} frames {run[0][0]}-{run[-1][0]} "
          f"({len(run)} frames) tracklet_span={t.frame_start}-{t.frame_end if t else '?'}")

def _fragment_for(track_id):
    tracklet = tracklet_by_online_id.get(track_id)
    if tracklet is None:
        return None
    ended_dormant = _last_raw_state(frame_log_by_online_id, track_id)
    if track_id not in _appearance_cache:
        _appearance_cache[track_id] = populate_appearance_samples(tracklet, VIDEO_PATH, _encoder)
    tracklet, mean_crop_quality = _appearance_cache[track_id]
    return _tracklet_to_fragment(
        tracklet, ended_dormant in ("TEMPORARILY_LOST", "OUT_OF_FRAME", "REIDENTIFYING"),
        mean_crop_quality=mean_crop_quality)

for run_before, run_after in zip(runs[:-1], runs[1:]):
    before_id = run_before[0][1]
    online_chosen_id = run_after[0][1]
    gap_start = run_before[-1][0]
    gap_end = run_after[0][0]
    print(f"\n=== BOUNDARY: before={before_id} online_chosen={online_chosen_id} "
          f"gap_frames={gap_start}->{gap_end} ({gap_end - gap_start} frames) ===")

    before_tracklet = tracklet_by_online_id.get(before_id)
    if before_tracklet is None or not before_tracklet.trajectory:
        print("  before_tracklet missing/empty trajectory -- boundary skipped by resolver")
        continue
    if before_tracklet.last_good_center is not None:
        before_end_frame = before_tracklet.last_good_frame
        before_end_center = before_tracklet.last_good_center
    else:
        before_end_frame = before_tracklet.frame_end
        before_end_center = before_tracklet.trajectory[-1]
    print(f"  before_tracklet last-good: frame={before_end_frame} center={before_end_center}")

    scored_candidates = []
    all_considered = []
    for t in tracklets:
        if t.online_track_id == before_id or not t.trajectory:
            continue
        gap = t.frame_start - before_end_frame
        all_considered.append((t.online_track_id, gap, t.frame_start, t.trajectory[0]))
        if gap <= 0 or gap > min(MAX_STITCH_GAP_FRAMES, CANDIDATE_SEARCH_WINDOW_FRAMES):
            continue
        dx = t.trajectory[0][0] - before_end_center[0]
        dy = t.trajectory[0][1] - before_end_center[1]
        dist = (dx * dx + dy * dy) ** 0.5
        if dist > CANDIDATE_SPATIAL_RADIUS_PX_PER_FRAME * gap:
            continue
        scored_candidates.append((dist, t.online_track_id, gap))
    scored_candidates.sort(key=lambda x: x[0])
    candidate_ids = [tid for _dist, tid, _gap in scored_candidates[:CANDIDATE_TOP_K]]
    if online_chosen_id not in candidate_ids and online_chosen_id != before_id:
        candidate_ids.append(online_chosen_id)

    print(f"  tracklets in clip: {len(tracklets)}; geometric-prefilter candidates considered: "
          f"{len(all_considered)}; passed gap+radius prefilter: {len(scored_candidates)}; "
          f"evaluated (incl. online tracker's own choice): {len(candidate_ids)}")

    frag_before = _fragment_for(before_id)
    for cid in candidate_ids:
        frag_c = _fragment_for(cid)
        if frag_c is None:
            print(f"    candidate {cid}: no fragment (tracklet missing)")
            continue
        decision = evaluate_stitch(frag_before, frag_c)
        tag = " <- ONLINE TRACKER'S OWN CHOICE" if cid == online_chosen_id else ""
        print(f"    candidate {cid}: stitched={decision.stitched} "
              f"confidence={decision.stitch_confidence:.3f} reason={decision.stitch_reason_code}{tag}")
        print(f"       frag_c: start_frame={frag_c.start_frame} start_center={frag_c.start_center} "
              f"team={frag_c.team}({frag_c.team_confidence:.2f})")
    print(f"  frag_before: team={frag_before.team}({frag_before.team_confidence:.2f}) "
          f"end_center={frag_before.end_center} end_velocity={frag_before.end_velocity}")

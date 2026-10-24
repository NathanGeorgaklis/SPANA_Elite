"""Task 3D, Step 12 diagnostic: why did duel_lost_01 register zero touches
despite excellent (96.6% TRACKING) player tracking?

Read-only, non-invasive instrumentation: monkeypatches the FootballEventDetector
NAME bound inside football_analysis.engine's own module namespace (at runtime,
in this process only -- nothing is written to any frozen .py file on disk) so
that this script can retain a reference to the real detector instance the real
run_analysis_pass() call constructs, and inspect its internal
_possession_frames / ball_tracker.history after the call returns. This re-runs
Pass 2 (ball detection + possession/event logic) once more against the exact
same cached Pass-1 tracking JSON already used for the real duel_lost_01 run --
same inputs, same frozen code path, zero behavioural difference from the real
run; the only change is that this script keeps a handle to the detector object
afterward so it can be read, which the real orchestration throws away.
"""
import sys, json
sys.path.insert(0, ".")

import football_analysis.engine as engine_mod
from football_analysis.event_detector import FootballEventDetector

captured = {}

class CapturingDetector(FootballEventDetector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        captured["detector"] = self

engine_mod.FootballEventDetector = CapturingDetector

from event_engine_v1.analyze_target_player import run_engine_pass_with_offset
from player_tracker_mo.tracklet_builder import build_tracklets_from_frame_log
from player_tracker_mo.selected_player_resolver import resolve_selected_player
from player_tracker_mo.offline_identity_enrichment import enrich_frame_log_with_offline_identity
from player_tracker_mo.event_adapter import build_integrated_event_snapshots
from player_tracker.appearance import AppearanceEncoder

VIDEO = "/root/.claude/uploads/98d69dca-3149-5a43-9670-3c417f2af6d4/8a710009-dual_lost_01.mov"
MO_JSON = "event_engine_v1/duel_lost_01/mo_tracking/8a710009-dual_lost_01_mo_tracking_results.json"

with open(MO_JSON) as f:
    mo_data = json.load(f)
frame_log = mo_data["frame_log"]
start_frame = mo_data["summary"]["start_frame"]
fps = mo_data["summary"]["fps"]

tracklets = build_tracklets_from_frame_log(frame_log)
encoder = AppearanceEncoder(use_ocr=False)
resolver_result = resolve_selected_player(frame_log, tracklets, video_path=VIDEO, encoder=encoder)
enriched = enrich_frame_log_with_offline_identity(frame_log, resolver_result)
snapshots = build_integrated_event_snapshots(enriched, fps, player_id="P20")

n_frames = len(snapshots)
out_json = "event_engine_v1/duel_lost_01/_diag_rerun_events_raw.json"
print(f"Rerunning Pass 2 for {n_frames} frames (this re-does ball detection; same cost as the real run)...")
result = run_engine_pass_with_offset(VIDEO, snapshots, start_frame, n_frames, out_json)

det = captured["detector"]
pframes = det._possession_frames
ball_hist = det.ball_tracker.history
print(f"\ncaptured {len(pframes)} possession frames, {len(ball_hist)} ball-history entries")

# Find minimum ball_distance over the whole clip, and print the full
# near-player trace for the human-labelled 0.0-1.2s window.
dists = [(i, pf.ball_distance) for i, pf in enumerate(pframes) if pf.ball_distance is not None]
if dists:
    min_i, min_d = min(dists, key=lambda x: x[1])
    print(f"\nMIN ball_distance over whole clip: {min_d:.1f}px at frame {min_i} (t={min_i/fps:.2f}s)")
else:
    print("\nNo frame ever had a non-None ball_distance at all.")

print(f"\nTOUCH_RADIUS = 40.0px (from possession_detector.py)")
print(f"\n--- per-frame trace, frames 0-80 (t=0.0-1.36s, covers the human-verified 0.0-1.2s DUEL window) ---")
print(f"{'frame':>5} {'t(s)':>6} {'tracking_state':>14} {'ball_state':>12} {'ball_dist':>10} {'player_cxy':>16} {'ball_cxy':>16} {'in_poss':>8}")

snaps = det._snapshots
export = []
for i in range(len(pframes)):
    pf = pframes[i]
    bs = ball_hist[i] if i < len(ball_hist) else None
    snap = snaps[i] if i < len(snaps) else None
    pcx, pcy = (None, None)
    if snap is not None and getattr(snap, "bbox", None):
        x1, y1, x2, y2 = snap.bbox
        pcx, pcy = (x1 + x2) / 2, (y1 + y2) / 2
    bcx = getattr(bs, "centre_x", None) if bs is not None else None
    bcy = getattr(bs, "centre_y", None) if bs is not None else None
    export.append({
        "frame": i, "t": round(i / fps, 3),
        "tracking_state": pf.tracking_state, "ball_state": pf.ball_state,
        "ball_distance": pf.ball_distance,
        "player_cx": pcx, "player_cy": pcy,
        "ball_cx": bcx, "ball_cy": bcy,
        "in_possession": pf.in_possession,
    })

with open("event_engine_v1/duel_lost_01/_diag_ball_vs_player_trace.json", "w") as f:
    json.dump(export, f, indent=2)
print("\nwrote event_engine_v1/duel_lost_01/_diag_ball_vs_player_trace.json")

for row in export[:81]:
    pc = f"({row['player_cx']:.0f},{row['player_cy']:.0f})" if row['player_cx'] is not None else "None"
    bc = f"({row['ball_cx']:.0f},{row['ball_cy']:.0f})" if row['ball_cx'] is not None else "None"
    bd = f"{row['ball_distance']:.1f}" if row['ball_distance'] is not None else "None"
    print(f"{row['frame']:5d} {row['t']:6.2f} {row['tracking_state']:>14} {row['ball_state']:>12} {bd:>10} {pc:>16} {bc:>16} {str(row['in_possession']):>8}")

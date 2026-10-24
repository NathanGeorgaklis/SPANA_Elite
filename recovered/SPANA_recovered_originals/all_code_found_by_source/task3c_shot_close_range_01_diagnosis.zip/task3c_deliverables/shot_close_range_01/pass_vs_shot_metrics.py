"""Task 3C Step 8: compare target-tracking metrics between pass_completed_01
(Task 3A's clean PASS baseline) and shot_close_range_01, to characterize what
makes the shot clip structurally harder for the ONLINE tracker (independent
of the separate video-provenance finding). Read-only."""
import json
from collections import Counter

def load_snapshots(path):
    with open(path) as f:
        d = json.load(f)
    return d["snapshots"]

def summarize(snaps, label):
    n = len(snaps)
    states = Counter(s["tracking_state"] for s in snaps)
    confs = [s["tracking_confidence"] for s in snaps]
    src = Counter(s.get("position_source") for s in snaps)
    print(f"\n=== {label} ({n} snapshots) ===")
    for k, v in states.items():
        print(f"  tracking_state {k}: {v} ({100*v/n:.1f}%)")
    print(f"  tracking_confidence: min={min(confs):.3f} max={max(confs):.3f} mean={sum(confs)/n:.3f}")
    for k, v in src.items():
        print(f"  position_source {k}: {v} ({100*v/n:.1f}%)")
    return states, confs, src

pc_snaps = load_snapshots("event_engine_v1/pass_completed_01/pass_completed_01_integrated_snapshots.json")
sc_snaps = load_snapshots("event_engine_v1/shot_close_range_01/shot_close_range_01_integrated_snapshots.json")

summarize(pc_snaps, "pass_completed_01 (CLEAN PASS baseline)")
summarize(sc_snaps, "shot_close_range_01 (0 events)")

# crowding proxy: n_live_tracks from frame_log mo_tracking results, if available
for path, label in [
    ("event_engine_v1/pass_completed_01/mo_tracking/f5bd8ce0-pass_completed_01_mo_tracking_results.json", "pass_completed_01"),
    ("event_engine_v1/shot_close_range_01/mo_tracking/shot_close_range_01_mo_tracking_results.json", "shot_close_range_01"),
]:
    with open(path) as f:
        d = json.load(f)
    fl = d["frame_log"]
    n_live = [rec.get("n_live_tracks", 0) for rec in fl]
    n_det = [rec.get("n_detections", 0) for rec in fl]
    overlap = [len(rec.get("overlap_contests") or []) for rec in fl]
    print(f"\n{label}: n_live_tracks min={min(n_live)} max={max(n_live)} mean={sum(n_live)/len(n_live):.1f}")
    print(f"{label}: n_detections min={min(n_det)} max={max(n_det)} mean={sum(n_det)/len(n_det):.1f}")
    print(f"{label}: overlap_contests total={sum(overlap)} mean={sum(overlap)/len(overlap):.2f}")
    print(f"{label}: summary selected_state_pct={d['summary'].get('selected_state_pct')}")
    print(f"{label}: spawn_guard_stats={d['summary'].get('spawn_guard_stats')}")

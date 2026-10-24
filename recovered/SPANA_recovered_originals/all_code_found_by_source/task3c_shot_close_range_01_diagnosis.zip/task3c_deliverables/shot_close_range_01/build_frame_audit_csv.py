"""Task 3C Step 1: per-frame target-tracking audit CSV for shot_close_range_01.
Pure read/reshape of already-generated frame_log data -- no recompute of
tracking, no frozen file touched, nothing changed."""
import json, csv, math

with open("event_engine_v1/shot_close_range_01/mo_tracking/shot_close_range_01_mo_tracking_results.json") as f:
    data = json.load(f)
fl = data["frame_log"]
fps = data["summary"]["fps"]

rows = []
for rec in fl:
    frame_log_frame = rec["frame"]
    abs_frame = frame_log_frame + 321
    t = abs_frame / fps
    tracks = rec["tracks"]
    selected = [t_ for t_ in tracks if t_.get("selected_player")]
    others = [t_ for t_ in tracks if not t_.get("selected_player")]

    if not selected:
        rows.append({
            "frame_log_frame": frame_log_frame, "abs_frame": abs_frame, "t_s": round(t, 3),
            "selected_online_track_id": None, "tracking_state": "NO_SELECTED_TRACK",
            "bbox": None, "position_source": None, "detection_confidence": None,
            "motion_confidence": None, "appearance_confidence": None, "reid_confidence": None,
            "overall_track_confidence": None, "position_confidence": None,
            "track_continuity_confidence": None, "identity_confidence": None,
            "frames_since_observed_position": None, "frames_since_confirmed_identity": None,
            "n_candidate_tracks": len(tracks), "nearest_competitor_1": None, "nearest_competitor_2": None,
            "camera_shift": rec.get("camera_shift"), "is_cut": rec.get("is_cut"),
            "n_detections": rec.get("n_detections"), "n_live_tracks": rec.get("n_live_tracks"),
            "overlap_contests": len(rec.get("overlap_contests") or []),
        })
        continue

    sel = selected[0]
    sx1, sy1, sx2, sy2 = sel["bbox"]
    scx, scy = (sx1 + sx2) / 2, (sy1 + sy2) / 2

    comp = []
    for o in others:
        ox1, oy1, ox2, oy2 = o["bbox"]
        ocx, ocy = (ox1 + ox2) / 2, (oy1 + oy2) / 2
        d = math.hypot(ocx - scx, ocy - scy)
        comp.append((d, o["online_track_id"], o.get("team")))
    comp.sort(key=lambda x: x[0])

    def fmt_comp(c):
        if c is None:
            return None
        d, tid, team = c
        return f"{tid}({team},{d:.0f}px)"

    rows.append({
        "frame_log_frame": frame_log_frame, "abs_frame": abs_frame, "t_s": round(t, 3),
        "selected_online_track_id": sel["online_track_id"], "tracking_state": sel["state"],
        "bbox": "|".join(f"{v:.1f}" for v in sel["bbox"]),
        "position_source": sel["position_source"],
        "detection_confidence": sel["detection_confidence"],
        "motion_confidence": sel["motion_confidence"],
        "appearance_confidence": sel["appearance_confidence"],
        "reid_confidence": sel["reid_confidence"],
        "overall_track_confidence": sel["overall_track_confidence"],
        "position_confidence": sel["position_confidence"],
        "track_continuity_confidence": sel["track_continuity_confidence"],
        "identity_confidence": sel["identity_confidence"],
        "frames_since_observed_position": sel["frames_since_observed_position"],
        "frames_since_confirmed_identity": sel["frames_since_confirmed_identity"],
        "n_candidate_tracks": len(tracks),
        "nearest_competitor_1": fmt_comp(comp[0]) if len(comp) > 0 else None,
        "nearest_competitor_2": fmt_comp(comp[1]) if len(comp) > 1 else None,
        "camera_shift": rec.get("camera_shift"),
        "is_cut": rec.get("is_cut"),
        "n_detections": rec.get("n_detections"),
        "n_live_tracks": rec.get("n_live_tracks"),
        "overlap_contests": len(rec.get("overlap_contests") or []),
    })

out_path = "event_engine_v1/shot_close_range_01/SHOT_CLOSE_RANGE_TARGET_TRACKING_FRAME_AUDIT.csv"
fieldnames = list(rows[0].keys())
with open(out_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
        w.writerow(r)

print(f"Wrote {len(rows)} rows to {out_path}")
# sanity: print first 12 and the transition rows around frame 9-11 and 115-116
for r in rows[:3]:
    print(r)
print("---transition 9-12---")
for r in rows[8:12]:
    print(r)
print("---boundary 114-117---")
for r in rows[113:117]:
    print(r)

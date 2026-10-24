import json

with open("event_engine_v1/shot_close_range_01/mo_tracking/shot_close_range_01_mo_tracking_results.json") as f:
    data = json.load(f)

fl = data["frame_log"]
fps = data["summary"]["fps"]
cuts = [rec for rec in fl if rec.get("is_cut")]
print(f"Total frame_log records: {len(fl)}")
print(f"is_cut=True count: {len(cuts)}")
for rec in cuts:
    absf = rec["frame"] + 321
    print(f"  CUT at frame_log frame={rec['frame']} absolute_frame={absf} t={absf/fps:.2f}s camera_shift={rec.get('camera_shift')}")

# Also print camera_shift stats overall to see magnitude trend
shifts = [(rec["frame"], rec.get("camera_shift")) for rec in fl]
print("\nFirst 15 camera_shift values:")
for fnum, cs in shifts[:15]:
    print(f"  frame={fnum} shift={cs}")

# frames around t=7-8s (absolute ~416-446, relative frame_log ~95-125)
print("\ncamera_shift for relative frames 95-130 (t~6.7-8.2s):")
for rec in fl:
    if 95 <= rec["frame"] <= 130:
        absf = rec["frame"] + 321
        print(f"  frame_log={rec['frame']} abs={absf} t={absf/fps:.2f}s is_cut={rec.get('is_cut')} shift={rec.get('camera_shift')} n_detections={rec.get('n_detections')} n_live_tracks={rec.get('n_live_tracks')}")

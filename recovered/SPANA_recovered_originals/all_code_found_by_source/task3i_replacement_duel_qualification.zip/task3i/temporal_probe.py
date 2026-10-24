"""Task 3I temporal-coverage probe for src2_clip_003.
NOT a benchmark fixture: target identity is not grounded, so target fields are
explicit placeholders and nothing is saved as a fixture. Only the timing fields
are read by validate_temporal_coverage(); the guard code is unchanged."""
import json, sys
from dataclasses import replace
from event_engine_v1.benchmark_fixture import BenchmarkFixture, validate_temporal_coverage, validate_provenance
CLIP = "/mnt/user-data/uploads/Downloads/_spana_clip_tools/_spana_clips_out/src2_clip_003/clip.mp4"
base = BenchmarkFixture.create(
    clip_id="src2_clip_003", video_path=CLIP,
    target_player_label="UNGROUNDED_PLACEHOLDER", seed_frame=-1, seed_bbox=[0, 0, 0, 0],
    expected_primary_event="DUEL", expected_outcome="WON",
    expected_event_window="5.0-10.0 s clip time (annotation 2824.0-2829.0 s absolute, clip starts 2819.0 s)",
    fixture_provenance_notes="Task 3I temporal probe only; not a fixture.")
out = {"probe_file_sha256": base.video_sha256, "width": base.width, "height": base.height,
       "fps": base.fps, "frame_count": base.frame_count, "duration_s": base.duration_seconds,
       "provenance_self_check": validate_provenance(base, CLIP).status, "cases": {}}
cases = {
  "label_as_annotated_5.0_10.0__tracking_full_clip": dict(label_window_start_s=5.0, label_window_end_s=10.0, tracking_start_s=0.0, tracking_end_s=base.duration_seconds),
  "label_as_annotated__tracking_from_label_start (must fail)": dict(label_window_start_s=5.0, label_window_end_s=10.0, tracking_start_s=5.0, tracking_end_s=base.duration_seconds),
  "visually_observed_tackle_3.2_4.0__tracking_full_clip (NOT a label; reference only)": dict(label_window_start_s=3.2, label_window_end_s=4.0, tracking_start_s=0.0, tracking_end_s=base.duration_seconds),
}
for name, kw in cases.items():
    f = replace(base, label_window_source="annotation 13a88c17/37ef0164 converted to clip time", required_pre_event_context_s=None, **kw)
    r = validate_temporal_coverage(f)
    out["cases"][name] = json.loads(r.to_json())
print(json.dumps(out, indent=2))

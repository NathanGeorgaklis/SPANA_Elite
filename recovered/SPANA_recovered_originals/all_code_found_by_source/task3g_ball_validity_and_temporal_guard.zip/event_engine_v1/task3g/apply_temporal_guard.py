"""Task 3G Part A: build/refresh fixtures with temporal-coverage fields and run
the evaluation gate on every current fixture. Label windows are VIDEO time and
come from validation/reports/new_clips_validation_report.md (supplied window and
human-refined window); the governing window is their union (earliest start,
latest end) so a fixture is never declared covered for part of a label."""
import sys, json, copy
sys.path.insert(0, ".")
from event_engine_v1.benchmark_fixture import (BenchmarkFixture, evaluation_gate, validate_temporal_coverage,
                                               tracking_range_from_mo, sha256_file)

OUT = "event_engine_v1/task3g/fixtures"
import os; os.makedirs(OUT, exist_ok=True)
results = {}

# ---- pass_completed_01 (no fixture existed before; created from the real file)
pmo = json.load(open("event_engine_v1/pass_completed_01/mo_tracking/f5bd8ce0-pass_completed_01_mo_tracking_results.json"))
ts, te = tracking_range_from_mo(pmo["summary"])
seed = None
for rec in pmo["frame_log"]:
    for t in rec["tracks"]:
        if t.get("selected_player"):
            seed = t["bbox"]; break
    if seed: break
pf = BenchmarkFixture.create(
    clip_id="pass_completed_01",
    video_path="/root/.claude/uploads/98d69dca-3149-5a43-9670-3c417f2af6d4/f5bd8ce0-pass_completed_01.mov",
    target_player_label="White #26", seed_frame=pmo["summary"]["start_frame"], seed_bbox=seed,
    expected_primary_event="PASS", expected_outcome="COMPLETE",
    expected_event_window="supplied 1.0-3.0s; refined release 1.8-2.2s; Task 3A hand-check: carry from 1.06s, kick at 1.66s",
    fixture_provenance_notes=("Created in Task 3G from the content-hash-prefixed upload used by every Task 3A-3F run. "
                              "seed_bbox = first selected-player box recorded by the Pass-1 run (start_frame 28)."),
    tracking_artifact_path="event_engine_v1/pass_completed_01/mo_tracking/f5bd8ce0-pass_completed_01_mo_tracking_results.json")
pf.label_window_start_s, pf.label_window_end_s = 1.0, 3.0
pf.label_window_source = "union of supplied 1.0-3.0s and refined 1.8-2.2s (validation report CLIP 1)"
pf.tracking_start_s, pf.tracking_end_s = ts, te
pf.required_pre_event_context_s = None
pf_hash = pf.save(f"{OUT}/PASS_COMPLETED_01_BENCHMARK_FIXTURE.json")

# ---- duel_lost_01 (Task 3D fixture kept untouched; v2 adds temporal fields)
df = BenchmarkFixture.load("event_engine_v1/DUEL_LOST_01_BENCHMARK_FIXTURE.json")
dmo = json.load(open("event_engine_v1/duel_lost_01/mo_tracking/8a710009-dual_lost_01_mo_tracking_results.json"))
df.tracking_start_s, df.tracking_end_s = tracking_range_from_mo(dmo["summary"])
df.label_window_start_s, df.label_window_end_s = 0.0, 2.0
df.label_window_source = "union of supplied 1.0-2.0s and refined 0.0-1.2s (contest peak 0.0-0.9s; validation report CLIP 3)"
df.fixture_provenance_notes += (" | TASK 3G: temporal fields added; status TEMPORAL_COVERAGE_INVALID_FOR_LABELLED_EVENT "
                                 "(Pass-1 seeded at video frame 53 = 0.90s, after the labelled contest).")
df_hash = df.save(f"{OUT}/DUEL_LOST_01_BENCHMARK_FIXTURE_v2.json")

# ---- shot_close_range_01: labels belong to a video that is no longer on disk
sf = copy.deepcopy(df)
sf.clip_id = "shot_close_range_01"
sf.video_path = "/mnt/user-data/uploads/shot_close_range_01.mov"
sf.video_sha256 = "UNKNOWN_ORIGINAL_FILE_REPLACED"   # the labelled file's hash was never recorded (Task 3C/3D)
sf.label_window_start_s, sf.label_window_end_s = 5.0, 6.0
sf.label_window_source = "union of supplied 5.0-6.0s and refined 5.35-5.65s -- for the ORIGINAL file"

for name, fx in (("pass_completed_01", pf), ("duel_lost_01", df), ("shot_close_range_01", sf)):
    g = evaluation_gate(fx)
    tc = validate_temporal_coverage(fx) if g.provenance_valid else None
    results[name] = {"PROVENANCE_VALID": g.provenance_valid,
                     "TEMPORAL_COVERAGE_VALID": g.temporal_coverage_valid,
                     "EVENT_EVALUATION_ALLOWED": g.event_evaluation_allowed,
                     "reasons": g.reasons, "temporal": (json.loads(tc.to_json()) if tc else None)}
    print(name, json.dumps(results[name], indent=1))
results["fixture_file_sha256"] = {"PASS_COMPLETED_01_BENCHMARK_FIXTURE.json": pf_hash,
                                  "DUEL_LOST_01_BENCHMARK_FIXTURE_v2.json": df_hash,
                                  "DUEL_LOST_01_BENCHMARK_FIXTURE.json (Task 3D, unchanged)":
                                      sha256_file("event_engine_v1/DUEL_LOST_01_BENCHMARK_FIXTURE.json")}
json.dump(results, open(f"{OUT}/temporal_guard_results.json", "w"), indent=2)

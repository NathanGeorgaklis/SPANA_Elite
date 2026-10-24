"""
general_football_benchmark_runner.py -- GENERAL_FOOTBALL_BENCHMARK deliverable 7
(benchmark runner). Produces GENERAL_BENCHMARK_BASELINE_V1: runs the frozen,
unmodified canonical pipeline (run_tracking_pipeline(), DEFAULT_CANDIDATE_
CONFIGURATION, pipeline_mode="INTEGRATED") against every benchmark item in
GENERAL_FOOTBALL_BENCHMARK_MANIFEST.json, then (where the manifest marks
run_event_engine=true) feeds the result into the real, frozen
football_analysis.engine.FootballAnalysisEngine -- exactly as every prior
live validation in this project has.

Hard constraints (unchanged from the task spec, verified by inspection,
never edited by this script): A-E (track_manager.py), detector thresholds,
ReID thresholds, event thresholds, possession rules, ball-tracking
thresholds, event semantics. This script only calls existing functions.

Records, per item: pipeline_version, configuration_hash, tracking_model_
version, schema_version, plus this run's own measured stage_runtime_s and
engine_runtime_s -- the "GENERAL_BENCHMARK_BASELINE_V1" frozen-config
fingerprint the task requires.

For items already computed by an earlier task under the IDENTICAL frozen
configuration (confirmed by matching configuration_hash, not assumed), this
script reuses the existing artifact rather than re-running it -- named
explicitly per item in the output, never silently. blind_r2_009 is the one
such case: its 419-frame run is the literal output of the completed A-E
ablation ladder's winning stage, and its first-150-frames event-engine
output is the literal output of this project's own prior live validation
task. Reusing it is not a shortcut around "frozen baseline" -- it IS a
frozen-baseline run, already executed, under the exact same config.
"""
import json
import os
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402
from football_analysis.adapter import TrackingAdapter  # noqa: E402
from football_analysis.colour import shirt_color_hist  # noqa: E402
from football_analysis.engine import FootballAnalysisEngine  # noqa: E402

from player_tracker_mo.pipeline import run_tracking_pipeline, DEFAULT_CANDIDATE_CONFIGURATION, _config_hash  # noqa: E402

MANIFEST_PATH = "GENERAL_FOOTBALL_BENCHMARK_MANIFEST.json"
OUT_ROOT = "rootcause_out/general_football_benchmark_v1"
EXPECTED_HASH = _config_hash(DEFAULT_CANDIDATE_CONFIGURATION)


def _run_one(item):
    clip_id = item["benchmark_item_id"]
    out_dir = f"{OUT_ROOT}/{clip_id}"
    result = {"benchmark_item_id": clip_id, "source_match_id": item["source_match_id"],
              "split": item["split"]}

    # --- reuse path: an existing artifact under a CONFIRMED-identical configuration_hash ---
    reuse = item.get("reuse_existing_artifact")
    if reuse:
        with open(reuse["tracking_summary_path"]) as f:
            prior_summary = json.load(f)
        if prior_summary.get("configuration_hash") != EXPECTED_HASH:
            result["status"] = "FAIL"
            result["failure_reason"] = (f"reuse_existing_artifact requested for {clip_id} but its "
                                         f"configuration_hash {prior_summary.get('configuration_hash')!r} "
                                         f"!= current DEFAULT_CANDIDATE_CONFIGURATION hash {EXPECTED_HASH!r} "
                                         f"-- refusing to silently reuse a mismatched config")
            return result
        result["status"] = "REUSED"
        result["reused_from"] = reuse
        result["pipeline_version"] = prior_summary["pipeline_version"]
        result["configuration_hash"] = prior_summary["configuration_hash"]
        result["schema_version"] = prior_summary["schema_version"]
        result["tracking_model_version"] = prior_summary.get("tracking_model_version", "yolo26n-reid.onnx")
        result["run_summary"] = prior_summary["run_summary"]
        result["max_frames_processed"] = prior_summary["run_summary"]["processed_frames"]
        result["n_boundaries"] = prior_summary.get("n_boundaries")
        result["n_auto_resolved_boundaries"] = prior_summary.get("n_auto_resolved_boundaries")
        result["n_review_required"] = prior_summary.get("n_review_required")
        result["n_unresolved"] = prior_summary.get("n_unresolved")
        result["event_snapshot_trust"] = prior_summary.get("event_snapshot_trust")
        result["artifact_paths"] = reuse
        if reuse.get("event_engine_report_path") and os.path.exists(reuse["event_engine_report_path"]):
            with open(reuse["event_engine_report_path"]) as f:
                evt = json.load(f)
            result["event_engine"] = {
                "status": evt.get("status"), "n_events_total": evt.get("n_events_total", evt.get("n_events")),
                "all_events": evt.get("all_events", evt.get("events")),
                "trust_state_counts_fed_to_engine": evt.get("trust_state_counts_fed_to_engine"),
                "engine_runtime_s": evt.get("engine_runtime_s"),
            }
        return result

    # --- fresh run path ---
    video_path = item["video_path"]
    init_box = item["init_box"]
    max_frames = item["max_frames"]
    try:
        t0 = time.time()
        pipeline_result = run_tracking_pipeline(
            video_path, 0, init_box, max_frames=max_frames, out_dir=out_dir,
            pipeline_mode="INTEGRATED", enable_appearance_resolution=True, resume=True,
            clip_id=clip_id, config=None,  # None -> DEFAULT_CANDIDATE_CONFIGURATION, untouched
        )
        result["total_pipeline_runtime_s"] = round(time.time() - t0, 1)
        summ = pipeline_result["summary"]
        if summ["configuration_hash"] != EXPECTED_HASH:
            raise AssertionError(f"configuration_hash drifted: {summ['configuration_hash']} != {EXPECTED_HASH}")
        result["status"] = "PASS"
        result["pipeline_version"] = summ["pipeline_version"]
        result["configuration_hash"] = summ["configuration_hash"]
        result["schema_version"] = summ["schema_version"]
        result["tracking_model_version"] = summ.get("tracking_model_version", "yolo26n-reid.onnx")
        result["stage_runtime_s"] = pipeline_result["stage_runtime_s"]
        result["run_summary"] = summ["run_summary"]
        result["max_frames_processed"] = summ["run_summary"]["processed_frames"]
        result["n_boundaries"] = summ["n_boundaries"]
        result["n_auto_resolved_boundaries"] = summ["n_auto_resolved_boundaries"]
        result["n_review_required"] = summ["n_review_required"]
        result["n_unresolved"] = summ["n_unresolved"]
        result["event_snapshot_trust"] = summ.get("event_snapshot_trust")
        result["artifact_paths"] = pipeline_result["artifact_paths"]

        if item.get("run_event_engine"):
            snapshots = TrackingAdapter.load(pipeline_result["artifact_paths"]["selected_player_trajectory"])
            fps = summ["run_summary"]["fps"]
            cap = cv2.VideoCapture(video_path)
            ok, frame0 = cap.read()
            cap.release()
            if not ok or not snapshots:
                result["event_engine"] = {"status": "SKIPPED", "reason": "no frame0 or no snapshots"}
            else:
                tracked_color_hist = shirt_color_hist(frame0, tuple(init_box))
                engine = FootballAnalysisEngine(fps=fps)
                t_e0 = time.time()
                real_result = engine.run_analysis_pass(video_path, snapshots, tracked_color_hist, start_frame=0)
                events = real_result.get("events", [])
                result["event_engine"] = {
                    "status": "PASS", "engine_runtime_s": round(time.time() - t_e0, 1),
                    "n_events_total": len(events),
                    "all_events": [{"type": e.get("type"), "success": e.get("success"),
                                     "start_frame": e.get("start_frame"), "end_frame": e.get("end_frame"),
                                     "confidence": e.get("confidence")} for e in events],
                    "trust_state_counts_fed_to_engine": {
                        s: sum(1 for sn in snapshots if sn.tracking_state == s)
                        for s in set(sn.tracking_state for sn in snapshots)
                    },
                }
    except Exception as e:
        result["status"] = "FAIL"
        result["failure_reason"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()
    return result


def main():
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    os.makedirs(OUT_ROOT, exist_ok=True)
    results = []
    for item in manifest["benchmark_items"]:
        print(f"=== running {item['benchmark_item_id']} ===", flush=True)
        t0 = time.time()
        r = _run_one(item)
        r["item_wall_time_s"] = round(time.time() - t0, 1)
        results.append(r)
        print(json.dumps({k: v for k, v in r.items() if k not in ("traceback",)}, indent=2, default=str), flush=True)
        # write incrementally so a container reclaim loses at most the in-flight item
        with open(f"{OUT_ROOT}/baseline_results.json", "w") as f:
            json.dump({"expected_configuration_hash": EXPECTED_HASH, "results": results}, f, indent=2, default=str)
    print(f"\nALL DONE -- wrote {OUT_ROOT}/baseline_results.json", flush=True)


if __name__ == "__main__":
    main()

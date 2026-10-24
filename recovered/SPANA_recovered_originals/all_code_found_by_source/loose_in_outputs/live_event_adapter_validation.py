"""
live_event_adapter_validation.py -- Integration task Section 12/13/14/15/16/17:
the mandatory LIVE run of the integrated (offline-trust-gated) event-engine
adapter against the REAL, FROZEN football_analysis event engine
(FootballAnalysisEngine.run_analysis_pass -- imported and called completely
unmodified, exactly as run_all_pass2_new.py already does for the raw
adapter). This is the first time event_adapter.build_integrated_event_
snapshots()'s output is consumed by a live engine run rather than only unit-
tested against the TrackingSnapshot contract.

Deliberately reuses run_tracking_pipeline() itself for Stages 1-8 (so the
trajectory fed to the event engine is exactly what a real caller would get
from the canonical pipeline, not a hand-constructed stand-in), then hands
its selected_player_trajectory.json straight to the frozen engine's own
Pass 2, via the exact same shirt_color_hist()-based team-histogram seeding
run_all_pass2_new.py already uses (no new team-classification logic).

Frame/timestamp contract check (Section 13): verifies fps and frame-number
convention match between the pipeline's own run_summary and the snapshots
actually handed to the engine, before interpreting any event result -- a
live run is treated as invalid (reported, not silently accepted) if this
check fails.

No silent legacy fallback (Section 15): if INTEGRATED-mode snapshot
generation raises or produces zero snapshots, this script does NOT fall back
to new_tracker_snapshots.py's raw adapter -- it reports FAIL and stops.
"""
import json
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from football_analysis.adapter import TrackingAdapter  # noqa: E402
from football_analysis.colour import shirt_color_hist  # noqa: E402
from football_analysis.engine import FootballAnalysisEngine  # noqa: E402
import cv2  # noqa: E402

from player_tracker_mo.pipeline import run_tracking_pipeline  # noqa: E402

CLIP_ID = "blind_r2_001"
VIDEO_PATH = f"spana_dataset/dataset/blind_package_v2_round2/{CLIP_ID}/clip.mp4"
SEL = json.load(open("spana_dataset/dataset/production_diagnostic_v2_round2/SELECTED_PLAYER_BBOXES.json"))["clips"][CLIP_ID]
OUT_DIR = f"rootcause_out/live_event_adapter_validation/{CLIP_ID}"
MAX_FRAMES = 150   # a genuine live run, scoped to keep wall-clock tractable on this 2-CPU sandbox
                     # (Pass 2's own YOLO+ball pass is "comparably expensive" to Pass 1 per
                     # run_all_pass2_new.py's own docstring -- a full 420-frame clip measured at
                     # ~800s in that script's own prior real run). This is a real execution, not a
                     # synthetic stand-in; it is simply not the full clip, which is reported honestly
                     # below rather than silently presented as a full-clip validation.


def main():
    report = {"clip_id": CLIP_ID, "video_path": VIDEO_PATH, "max_frames": MAX_FRAMES}
    try:
        t0 = time.time()
        pipeline_result = run_tracking_pipeline(
            VIDEO_PATH, 0, SEL["bbox"], max_frames=MAX_FRAMES, out_dir=OUT_DIR,
            pipeline_mode="INTEGRATED", enable_appearance_resolution=True, resume=True,
            clip_id=CLIP_ID,
        )
        report["pipeline_stage_runtime_s"] = pipeline_result["stage_runtime_s"]
        report["pipeline_summary_excerpt"] = {
            k: pipeline_result["summary"][k] for k in
            ("n_boundaries", "n_auto_resolved_boundaries", "n_review_required", "n_unresolved",
             "event_snapshot_trust", "schema_version", "pipeline_version", "configuration_hash")
        }

        snapshots = TrackingAdapter.load(pipeline_result["artifact_paths"]["selected_player_trajectory"])
        if not snapshots:
            report["status"] = "FAIL"
            report["failure_reason"] = "INTEGRATED snapshot generation produced zero snapshots -- " \
                                        "no silent fallback to legacy adapter performed."
            _write(report)
            return

        # --- Section 13: frame/timestamp contract check, BEFORE interpreting any event result ---
        run_summary = pipeline_result["summary"]["run_summary"]
        fps = run_summary["fps"]
        expected_frames = set(range(0, min(MAX_FRAMES, run_summary["processed_frames"]) + 1))
        actual_frames = {s.frame_number for s in snapshots}
        missing = sorted(expected_frames - actual_frames)
        contract_ok = True
        contract_notes = []
        if missing:
            contract_ok = False
            contract_notes.append(f"{len(missing)} frame number(s) never appear in the snapshot "
                                   f"list: {missing[:10]}{'...' if len(missing) > 10 else ''}")
        for s in snapshots:
            expected_ts = round(s.frame_number / fps, 4)
            if abs(s.timestamp - expected_ts) > 1e-3:
                contract_ok = False
                contract_notes.append(f"frame {s.frame_number}: timestamp {s.timestamp} != "
                                       f"frame_number/fps ({expected_ts})")
                break
        report["frame_timestamp_contract"] = {"ok": contract_ok, "notes": contract_notes, "fps": fps}
        if not contract_ok:
            report["status"] = "FAIL"
            report["failure_reason"] = "frame/timestamp contract check failed -- live run not interpreted."
            _write(report)
            return

        # --- Section 16: identity-trust breakdown actually fed to the engine ---
        trust_counts = {}
        for s in snapshots:
            trust_counts[s.tracking_state] = trust_counts.get(s.tracking_state, 0) + 1
        report["trust_state_counts_fed_to_engine"] = trust_counts

        # --- Section 12: the REAL, FROZEN event engine, called directly and unmodified ---
        cap = cv2.VideoCapture(VIDEO_PATH)
        ok, frame0 = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"could not read frame 0 of {VIDEO_PATH}")
        tracked_color_hist = shirt_color_hist(frame0, tuple(SEL["bbox"]))

        engine = FootballAnalysisEngine(fps=fps)
        t_engine0 = time.time()
        real_result = engine.run_analysis_pass(VIDEO_PATH, snapshots, tracked_color_hist, start_frame=0)
        engine_runtime_s = round(time.time() - t_engine0, 1)

        report["status"] = "PASS"
        report["engine_runtime_s"] = engine_runtime_s
        report["total_runtime_s"] = round(time.time() - t0, 1)
        report["n_events"] = len(real_result.get("events", []))
        report["events"] = [
            {"type": e.get("type"), "success": e.get("success"), "start_frame": e.get("start_frame"),
             "end_frame": e.get("end_frame"), "confidence": e.get("confidence")}
            for e in real_result.get("events", [])
        ]
        report["meta"] = real_result.get("meta", {})
        report["silent_legacy_fallback_occurred"] = False
        _write(report)
    except Exception as e:
        report["status"] = "FAIL"
        report["failure_reason"] = f"{type(e).__name__}: {e}"
        report["traceback"] = traceback.format_exc()
        _write(report)


def _write(report):
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    out_path = Path(OUT_DIR) / "live_event_adapter_validation_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(json.dumps({k: v for k, v in report.items() if k not in ("traceback",)}, indent=2, default=str))
    print(f"\nwritten to {out_path}")


if __name__ == "__main__":
    main()

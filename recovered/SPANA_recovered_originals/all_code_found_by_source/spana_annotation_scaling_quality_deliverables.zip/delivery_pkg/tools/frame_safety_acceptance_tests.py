#!/usr/bin/env python3
"""
frame_safety_acceptance_tests.py -- infrastructure-acceptance task items #7
and #8: a concrete regression test for the exact stale-fps bug class this
project has already shipped once, plus a frame/timestamp round-trip test
over every canonical clip that has actually been extracted.

This tool trains nothing and calls no football_analysis/player_tracker
code. It only exercises tools/time_utils.py and reads real, already-
extracted clip.mp4 files plus their clip_time_map.json sidecars.

TEST 1 -- FPS REGRESSION (task item #7)
-----------------------------------------------------------------------
Reproduces, exactly, the historical bug found during this dataset's own
migration: a registry/metadata source claims one fps (25.0, in the real
incident) while the actual source media probes to a different one (30.0,
in the real incident). This test does not need to *simulate* a video file
to prove the point -- it takes a REAL extracted clip, corrupts a COPY of
its clip_time_map.json to claim the wrong (stale) fps, and confirms that
dataset_validator.py's fps-consistency check (see dataset_validator.py
check (g)) refuses to accept it as a HARD error rather than silently
tolerating the drift. It also independently re-probes the real clip file
with time_utils.probe_video() to show the corrupted claim and the real,
measured fps disagree -- i.e., time_utils itself never had a chance to be
fooled, because it never trusts a claimed fps in the first place.

TEST 2 -- FRAME/TIMESTAMP ROUND-TRIP (task item #8)
-----------------------------------------------------------------------
For every extracted clip, samples several absolute source-video timestamps
spanning its analysis_context_window and, for each, computes:
    absolute_s -> absolute_frame -> clip_relative_s -> absolute_s (again)
via ClipTimeMap's own conversion methods, and reports the round-trip error
in both seconds and frames. A round-trip error above one frame's duration
(1/fps) would indicate the conversions are not mutually inverse -- this
test's whole job is to prove they are, on real data, not just by code
inspection.

Usage:
    python frame_safety_acceptance_tests.py
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
DATASET_ROOT = TOOLS_DIR.parent
sys.path.insert(0, str(TOOLS_DIR))

import time_utils                     # noqa: E402
import dataset_validator              # noqa: E402

CLIPS_DIR = DATASET_ROOT / "dataset" / "clips"
REPORTS_DIR = DATASET_ROOT / "dataset" / "reports"
ANNOTATIONS_DIR = DATASET_ROOT / "dataset" / "annotations"

# The exact historical incident this regression test encodes.
REAL_FPS_INCIDENT = {"registry_claimed_fps": 25.0, "actual_probed_fps": 30.0}

N_SAMPLES_PER_CLIP = 7


def run_fps_regression_test() -> dict:
    extracted_clip_ids = [d.name for d in CLIPS_DIR.iterdir()
                           if d.is_dir() and (d / "clip_time_map.json").exists()] if CLIPS_DIR.exists() else []
    if not extracted_clip_ids:
        return {"status": "SKIPPED", "reason": "no extracted clips available to test against"}

    clip_id = sorted(extracted_clip_ids)[0]
    clip_dir = CLIPS_DIR / clip_id
    real_ctm_dict = json.loads((clip_dir / "clip_time_map.json").read_text())
    real_fps = real_ctm_dict["source_fps"]

    # Independent re-probe of the REAL clip file -- time_utils never trusts
    # a claimed fps, so this must agree with what clip_time_map.json already
    # says (both ultimately come from the same ffprobe call chain, but this
    # re-probe is a fresh, independent invocation, not a cached value).
    real_probe = time_utils.probe_video(str(DATASET_ROOT / real_ctm_dict["clip_path"]))

    # Fabricate the historical incident: same clip, wrong claimed fps.
    stale_fps = REAL_FPS_INCIDENT["registry_claimed_fps"] if abs(real_fps - REAL_FPS_INCIDENT["registry_claimed_fps"]) > 0.5 \
        else 24.0  # fall back to a different wrong value if real fps happens to already be 25.0
    corrupted_ctm_dict = dict(real_ctm_dict)
    corrupted_ctm_dict["source_fps"] = stale_fps

    with tempfile.TemporaryDirectory() as tmp:
        fake_clips_dir = Path(tmp) / "clips"
        fake_clip_dir = fake_clips_dir / clip_id
        fake_clip_dir.mkdir(parents=True)
        (fake_clip_dir / "clip_time_map.json").write_text(json.dumps(corrupted_ctm_dict, indent=2))
        # dataset_validator's frame-range check also loads the clip's own
        # video file via probe_video through ClipTimeMap.clip_path, which is
        # an absolute-or-DATASET_ROOT-relative path baked into the sidecar --
        # point it at the real clip file so only source_fps is corrupted,
        # nothing else.

        report = dataset_validator.validate_all(annotations_dir=ANNOTATIONS_DIR, clips_dir=fake_clips_dir)

    hard_errors_for_clip = [e for e in report["hard_errors"] if e.get("check") == "frame_fps_consistency"
                             and clip_id in e.get("message", "")]

    caught = len(hard_errors_for_clip) > 0
    return {
        "status": "PASS" if caught else "FAIL",
        "scenario": f"clip_time_map.json corrupted to claim source_fps={stale_fps} "
                    f"while the real, freshly-probed clip fps is {real_probe.fps} "
                    f"(historical incident: registry claimed {REAL_FPS_INCIDENT['registry_claimed_fps']}, "
                    f"actual was {REAL_FPS_INCIDENT['actual_probed_fps']})",
        "clip_id_used": clip_id,
        "real_probed_fps": real_probe.fps,
        "corrupted_claimed_fps": stale_fps,
        "dataset_validator_raised_hard_error": caught,
        "hard_error_messages": [e["message"] for e in hard_errors_for_clip],
        "conclusion": (
            "PASS: dataset_validator.py refused the stale-fps claim as a HARD error, exactly "
            "as the real 25-vs-30fps incident should have been (and, once migrate_existing_"
            "annotations.py re-probed the real files, WAS) caught." if caught else
            "FAIL: a stale/wrong claimed fps was NOT flagged -- this is the exact bug class "
            "that has already shipped once and must never silently recur."
        ),
    }


def run_frame_roundtrip_test() -> dict:
    if not CLIPS_DIR.exists():
        return {"status": "SKIPPED", "reason": "no dataset/clips directory yet"}
    extracted_clip_ids = sorted(d.name for d in CLIPS_DIR.iterdir()
                                 if d.is_dir() and (d / "clip_time_map.json").exists())
    if not extracted_clip_ids:
        return {"status": "SKIPPED", "reason": "no extracted clips available to test against"}

    per_clip = []
    max_error_s_overall = 0.0
    max_error_frames_overall = 0.0

    for clip_id in extracted_clip_ids:
        ctm = time_utils.load_clip_time_map(str(CLIPS_DIR / clip_id))
        ctx_start, ctx_end = ctm.analysis_context_window_absolute_s
        fps = ctm.source_fps

        samples = [ctx_start + i * (ctx_end - ctx_start) / max(1, N_SAMPLES_PER_CLIP - 1)
                   for i in range(N_SAMPLES_PER_CLIP)]

        max_err_s, max_err_frames = 0.0, 0.0
        sample_rows = []
        for abs_s in samples:
            # absolute_s -> absolute_frame -> clip_relative_s -> absolute_s
            abs_frame = time_utils.seconds_to_frames(abs_s, fps)
            clip_rel_s = ctm.absolute_frame_to_clip_relative_seconds(abs_frame)
            abs_s_roundtrip = ctm.clip_relative_seconds_to_absolute_seconds(clip_rel_s)
            err_s = abs(abs_s_roundtrip - abs_s)
            err_frames = err_s * fps
            max_err_s = max(max_err_s, err_s)
            max_err_frames = max(max_err_frames, err_frames)
            sample_rows.append({
                "absolute_s_in": round(abs_s, 6), "absolute_frame": abs_frame,
                "clip_relative_s": round(clip_rel_s, 6),
                "absolute_s_roundtrip": round(abs_s_roundtrip, 6),
                "error_s": round(err_s, 6), "error_frames": round(err_frames, 4),
            })

        tolerance_s = 1.0 / fps   # one frame's duration -- the defensible tolerance per task item #8
        clip_pass = max_err_s <= tolerance_s
        per_clip.append({
            "clip_id": clip_id, "fps": fps, "n_samples": len(samples),
            "max_error_s": round(max_err_s, 6), "max_error_frames": round(max_err_frames, 4),
            "tolerance_s": round(tolerance_s, 6), "status": "PASS" if clip_pass else "FAIL",
            "samples": sample_rows,
        })
        max_error_s_overall = max(max_error_s_overall, max_err_s)
        max_error_frames_overall = max(max_error_frames_overall, max_err_frames)

    overall_pass = all(c["status"] == "PASS" for c in per_clip)
    return {
        "status": "PASS" if overall_pass else "FAIL",
        "n_clips_tested": len(per_clip),
        "max_error_s_overall": round(max_error_s_overall, 6),
        "max_error_frames_overall": round(max_error_frames_overall, 4),
        "per_clip": per_clip,
    }


def main():
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "fps_regression_test": run_fps_regression_test(),
        "frame_roundtrip_test": run_frame_roundtrip_test(),
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "frame_safety_acceptance_report.json"
    out.write_text(json.dumps(report, indent=2))

    print("=== FPS regression test (task item #7) ===")
    print(json.dumps(report["fps_regression_test"], indent=2))
    print("\n=== frame/timestamp round-trip test (task item #8) ===")
    rt = report["frame_roundtrip_test"]
    print(f"status={rt.get('status')} n_clips_tested={rt.get('n_clips_tested')} "
          f"max_error_s_overall={rt.get('max_error_s_overall')} "
          f"max_error_frames_overall={rt.get('max_error_frames_overall')}")
    for c in rt.get("per_clip", []):
        print(f"  {c['clip_id']:<28} fps={c['fps']:<8} max_error_frames={c['max_error_frames']:<8} {c['status']}")
    print(f"\nfull report written to {out}")


if __name__ == "__main__":
    main()

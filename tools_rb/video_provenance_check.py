# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R1-B instruction (GOLD may be used only as a video/provenance/frame-alignment fixture);
#                 player_tracker_rb/video_io.py, player_tracker_rb/provenance.py
"""Run the R1-B video/provenance layer on a real clip, twice, and check alignment and determinism.

Reads ONLY these fixture fields: video_sha256, seed_frame, seed_bbox, fps, frame_count, width, height.
Event fields (expected event, outcome, label window) are never read.

Checks:
  1. run A and run B (seek_mode decode_from_zero): identical deterministic digest and per-frame fingerprints;
  2. run C (seek_mode container_seek): identical per-frame fingerprints to run A (no seek misregistration);
  3. the fixture's fps / frame count / frame size / sha256 agree with what the reader measured;
  4. frame numbering: frames 1..N map to video frames seed+1..seed+N with no gaps.
Writes the run-A sidecar and a check report into OUT_DIR (refuses to overwrite).

Usage: python3 tools_rb/video_provenance_check.py VIDEO FIXTURE_JSON OUT_DIR
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from player_tracker_rb import output_schema as S  # noqa: E402
from player_tracker_rb import provenance as P  # noqa: E402
from player_tracker_rb import video_io as V  # noqa: E402
from player_tracker_rb.config import TrackerConfig  # noqa: E402

FIXTURE_FIELDS_USED = ("video_sha256", "seed_frame", "seed_bbox", "fps", "frame_count", "width", "height")


def run(video, fx, cfg):
    with V.SequentialVideoReader(video, fx["seed_frame"], cfg, expected_sha256=fx["video_sha256"]) as r:
        frames = [(f.rel_frame, f.video_frame_index, f.thumb_sha1) for f in r]
        contract = P.contract_provenance(r, fx["seed_bbox"], cfg)
    side = P.video_provenance_sidecar(r, contract, cfg, frame_thumbs=[t for _, _, t in frames])
    return r, frames, contract, side


def main(video, fixture_json, out_dir):
    full = json.load(open(fixture_json))
    fx = {k: full[k] for k in FIXTURE_FIELDS_USED}
    os.makedirs(out_dir, exist_ok=True)
    base = TrackerConfig.default()
    rA, fA, cA, sA = run(video, fx, base)
    rB, fB, cB, sB = run(video, fx, base)
    rC, fC, cC, sC = run(video, fx, TrackerConfig.with_overrides({"seek_mode": "container_seek"}))
    seed = fx["seed_frame"]
    n = len(fA)
    checks = {
        "A_vs_B_deterministic_digest_equal": P.deterministic_digest(sA) == P.deterministic_digest(sB),
        "A_vs_B_frame_thumbs_equal": fA == fB,
        "A_vs_B_seed_thumb_equal": rA.seed.thumb_sha1 == rB.seed.thumb_sha1,
        "A_decode_from_zero_vs_C_container_seek_frame_thumbs_equal": [t for *_, t in fA] == [t for *_, t in fC],
        "A_vs_C_seed_thumb_equal": rA.seed.thumb_sha1 == rC.seed.thumb_sha1,
        "frames_are_1_to_N": [f for f, _, _ in fA] == list(range(1, n + 1)),
        "video_index_is_seed_plus_frame": all(v == seed + f for f, v, _ in fA),
        "fixture_fps_matches": abs(rA.info.fps - fx["fps"]) < 1e-9,
        "fixture_frame_count_matches": rA.info.frame_count_reported == fx["frame_count"],
        "fixture_frame_size_matches": (rA.info.frame_width, rA.info.frame_height) == (fx["width"], fx["height"]),
        "fixture_sha256_matches": rA.info.sha256 == fx["video_sha256"],
        "processed_frames_is_frame_count_minus_seed_minus_1": n == fx["frame_count"] - seed - 1,
        "eof_reached": rA.eof_reached,
        "no_timing_drift_warnings": rA.timing_drift_warnings == 0,
        "source_reverified_at_close_all_runs": all(x.source_verified_at_close for x in (rA, rB, rC)),
        "container_seek_calls_A_B_C": [rA.seek_calls, rB.seek_calls, rC.seek_calls] == [0, 0, 1],
    }
    sidecar_file = P.sidecar_path(out_dir, video)
    sidecar_sha = P.write_sidecar(sA, sidecar_file)
    report = {
        "tool": "tools_rb/video_provenance_check.py",
        "fixture": os.path.basename(fixture_json), "fixture_fields_used": list(FIXTURE_FIELDS_USED),
        "event_fields_read": False,
        "deterministic_digest_A": P.deterministic_digest(sA), "deterministic_digest_B": P.deterministic_digest(sB),
        "frame_thumbs_sha256_A": sA["frame_alignment"]["frame_thumbs_sha256"],
        "frame_thumbs_sha256_C": sC["frame_alignment"]["frame_thumbs_sha256"],
        "seed_thumb_sha1": rA.seed.thumb_sha1, "processed_frames": n,
        "timing_drift_max_frames": rA.timing_drift_max_frames,
        "config_hash": cA["config_hash"], "git_commit": cA["git_commit"], "git_dirty": cA["git_dirty"],
        "sidecar_file": os.path.basename(sidecar_file), "sidecar_sha256": sidecar_sha,
        "checks": checks, "all_checks_pass": all(checks.values()),
    }
    report_path = os.path.join(out_dir, "VIDEO_PROVENANCE_CHECK.json")
    if os.path.exists(report_path):
        raise SystemExit(f"refusing to overwrite {report_path}")
    with open(report_path, "x") as f:
        f.write(S.dumps_canonical(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["all_checks_pass"] else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))

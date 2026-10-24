#!/usr/bin/env python3
"""
leakage_synthetic_acceptance_tests.py -- infrastructure-acceptance task item
#15: proves leakage_checker.py's check_leakage() fails LOUDLY on each of the
four leakage scenarios named in the task brief, using entirely synthetic,
in-memory metadata. This tool touches NO real dataset files (no real
annotation, no real dataset/splits/assignment.json, no real
dataset/metadata/source_videos.json is read or written) -- it only exercises
the importable `check_leakage()` function with fabricated dicts, which is
exactly what that function's own docstring says it is for.

This is deliberately NOT a test of whether THIS project currently has real,
independent matches to split (see item #14 / SPLIT_NOT_STATISTICALLY_
MEANINGFUL for that question, answered separately) -- it is a test that the
CHECKER ITSELF would catch these four leakage shapes if and when a real
split with more matches existed.

Scenario 1 -- SAME match_id assigned to two different splits (a hand-edited
             or badly-concatenated assignment.json).
Scenario 2 -- Two overlapping clips (by absolute source frame range) drawn
             from the SAME source_video_id, whose owning match_ids resolve
             to two different splits.
Scenario 3 -- Duplicate source content under two different filenames/
             source_video_ids (same file_sha256), registered under two
             different match_ids.
Scenario 4 -- Near-duplicate extracted clips: modelled here as two clips
             from the same source_video_id with heavily overlapping (not
             merely adjacent) frame ranges assigned to different splits --
             the metadata-only proxy this project's tooling can actually
             compute without re-decoding video (see module docstring for
             why byte-identical framing is the only content-level signal
             available pre-ML; true near-duplicate-by-visual-content
             detection would require a perceptual hash or embedding model,
             which is out of scope for infrastructure acceptance and would
             itself be a form of "additional event/ML tooling").

Usage:
    python leakage_synthetic_acceptance_tests.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
DATASET_ROOT = TOOLS_DIR.parent
REPORTS_DIR = DATASET_ROOT / "dataset" / "reports"
sys.path.insert(0, str(TOOLS_DIR))

import leakage_checker as lc          # noqa: E402

RESULTS = []


def _record(name: str, description: str, expected_hard: bool, report: dict, expect_key: str | None = None):
    hard = not report["clean"]
    ok = hard == expected_hard
    if expect_key is not None:
        ok = ok and len(report.get(expect_key, [])) > 0
    RESULTS.append({
        "scenario": name,
        "description": description,
        "expected_to_fail_loudly": expected_hard,
        "actually_failed_loudly": hard,
        "relevant_finding_key": expect_key,
        "relevant_finding_count": len(report.get(expect_key, [])) if expect_key else None,
        "status": "PASS" if ok else "FAIL",
        "report": report,
    })


def scenario_1_same_match_two_splits():
    # A dict literally cannot hold two values for one key -- that half of
    # finding (a) is structurally guaranteed by Python itself. What we CAN
    # simulate is the exact failure mode load_assignment_file() exists to
    # catch: a hand-edited/concatenated assignment.json whose raw JSON text
    # contains the same match_id key twice with conflicting values, which a
    # plain json.loads() would silently resolve to "last value wins" with no
    # warning at all.
    raw_text = '{"match_alpha": "TRAIN", "match_beta": "TEST", "match_alpha": "TEST"}'
    tmp_assignment_path = Path("/tmp/_leakage_synthetic_scenario1_assignment.json")
    tmp_assignment_path.write_text(raw_text)
    try:
        assignment, duplicates = lc.load_assignment_file(tmp_assignment_path)
        annotations = [
            {"annotation_id": "a1", "match_id": "match_alpha", "source_video_id": "sv1",
             "frame_safety": {"absolute_start_frame": 0, "absolute_end_frame": 30}},
            {"annotation_id": "a2", "match_id": "match_beta", "source_video_id": "sv2",
             "frame_safety": {"absolute_start_frame": 0, "absolute_end_frame": 30}},
        ]
        report = lc.check_leakage(assignment, annotations, {})
        report["duplicate_split_keys"] = duplicates
        report["hard_error_count"] += len(duplicates)
        report["clean"] = report["hard_error_count"] == 0
        _record(
            "1_same_match_train_and_test",
            "assignment.json's raw text assigns match_alpha to TRAIN once and TEST once "
            "(a corrupted/hand-edited file) -- the SAME match must never resolve to two splits.",
            expected_hard=True, report=report, expect_key="duplicate_split_keys",
        )
    finally:
        tmp_assignment_path.unlink(missing_ok=True)


def scenario_2_overlapping_clips_same_source_video_across_splits():
    # Two clips of the SAME source_video_id whose absolute frame ranges
    # overlap (frames 100-400 and frames 300-600 -- a 100-frame overlap),
    # but registered under two different match_ids that a (fabricated)
    # assignment.json puts on opposite sides of a split.
    assignment = {"match_x": "TRAIN", "match_y": "TEST"}
    annotations = [
        {"annotation_id": "ov1", "clip_id": "clip_ov1", "match_id": "match_x", "source_video_id": "sv_shared",
         "frame_safety": {"absolute_start_frame": 100, "absolute_end_frame": 400}},
        {"annotation_id": "ov2", "clip_id": "clip_ov2", "match_id": "match_y", "source_video_id": "sv_shared",
         "frame_safety": {"absolute_start_frame": 300, "absolute_end_frame": 600}},
    ]
    report = lc.check_leakage(assignment, annotations, {})
    _record(
        "2_overlapping_clips_same_source_video_across_splits",
        "Two clips (frames [100,400) and [300,600)) from the SAME source_video_id overlap "
        "by 100 frames, but their match_ids (match_x=TRAIN, match_y=TEST) put them on "
        "opposite sides of the split -- direct evaluation-set contamination.",
        expected_hard=True, report=report, expect_key="overlapping_frame_range_split_conflicts",
    )


def scenario_3_duplicate_source_content_different_filenames():
    # Same file_sha256 registered under two different source_video_ids /
    # filenames / match_ids -- exactly the "two different filenames do NOT
    # imply different recordings" trap from task item #3.
    source_videos = {
        "sv_original": {"path": "/footage/match_2026_04_12.mov", "match_id": "match_original",
                         "file_sha256": "deadbeef" * 8},
        "sv_reupload": {"path": "/footage/renamed_copy_final_v2.mov", "match_id": "match_reupload",
                         "file_sha256": "deadbeef" * 8},
    }
    assignment = {"match_original": "TRAIN", "match_reupload": "TEST"}
    annotations = [
        {"annotation_id": "d1", "match_id": "match_original", "source_video_id": "sv_original",
         "frame_safety": {"absolute_start_frame": 0, "absolute_end_frame": 30}},
        {"annotation_id": "d2", "match_id": "match_reupload", "source_video_id": "sv_reupload",
         "frame_safety": {"absolute_start_frame": 500, "absolute_end_frame": 530}},
    ]
    report = lc.check_leakage(assignment, annotations, source_videos)
    _record(
        "3_duplicate_source_content_different_filenames",
        "sv_original and sv_reupload have completely different filenames/match_ids but "
        "IDENTICAL file_sha256 (the same underlying recording registered twice) -- must be "
        "caught even though nothing about the filenames or match_ids themselves looks linked.",
        expected_hard=True, report=report, expect_key="duplicate_content_hash_findings",
    )


def scenario_4_near_duplicate_extracted_clips():
    # Near-duplicate extracted clips modelled as heavy (90%+) frame-range
    # overlap from the same source video, split across TRAIN/VALIDATION --
    # a stricter case of scenario 2's overlap check, included separately
    # because the task brief lists it as its own bullet.
    assignment = {"match_p": "TRAIN", "match_q": "VALIDATION"}
    annotations = [
        {"annotation_id": "nd1", "clip_id": "clip_nd1", "match_id": "match_p", "source_video_id": "sv_dup",
         "frame_safety": {"absolute_start_frame": 1000, "absolute_end_frame": 1150}},
        {"annotation_id": "nd2", "clip_id": "clip_nd2", "match_id": "match_q", "source_video_id": "sv_dup",
         "frame_safety": {"absolute_start_frame": 1005, "absolute_end_frame": 1145}},
    ]
    report = lc.check_leakage(assignment, annotations, {})
    overlap_entry = report["overlapping_frame_range_split_conflicts"][0] if report["overlapping_frame_range_split_conflicts"] else None
    _record(
        "4_near_duplicate_extracted_clips",
        "clip_nd2's frame range [1005,1145) sits almost entirely inside clip_nd1's "
        "[1000,1150) (near-duplicate re-extraction of the same moment) from the same "
        "source video, split TRAIN vs VALIDATION.",
        expected_hard=True, report=report, expect_key="overlapping_frame_range_split_conflicts",
    )
    return overlap_entry


def scenario_0_control_clean_case():
    # A NEGATIVE control: two genuinely different source videos, non-
    # overlapping frame ranges, different match_ids, consistent single-split
    # assignment each -- must report clean=True so we know the checker isn't
    # just failing everything by default.
    source_videos = {
        "sv_a": {"path": "/footage/match_a.mov", "match_id": "match_a", "file_sha256": "aaaa" * 16},
        "sv_b": {"path": "/footage/match_b.mov", "match_id": "match_b", "file_sha256": "bbbb" * 16},
    }
    assignment = {"match_a": "TRAIN", "match_b": "TEST"}
    annotations = [
        {"annotation_id": "c1", "match_id": "match_a", "source_video_id": "sv_a",
         "frame_safety": {"absolute_start_frame": 0, "absolute_end_frame": 30}},
        {"annotation_id": "c2", "match_id": "match_b", "source_video_id": "sv_b",
         "frame_safety": {"absolute_start_frame": 0, "absolute_end_frame": 30}},
    ]
    report = lc.check_leakage(assignment, annotations, source_videos)
    _record(
        "0_control_genuinely_independent_sources",
        "Negative control: two different source videos (different hashes), non-overlapping "
        "frame ranges, consistently split -- must report clean=True, proving the checker "
        "does not simply fail everything unconditionally.",
        expected_hard=False, report=report, expect_key=None,
    )


def main():
    scenario_0_control_clean_case()
    scenario_1_same_match_two_splits()
    scenario_2_overlapping_clips_same_source_video_across_splits()
    scenario_3_duplicate_source_content_different_filenames()
    scenario_4_near_duplicate_extracted_clips()

    all_pass = all(r["status"] == "PASS" for r in RESULTS)
    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Acceptance task item #15: prove leakage_checker.check_leakage() fails "
                   "loudly on synthetic same-match, overlapping-clip, duplicate-content, and "
                   "near-duplicate-clip leakage scenarios. Uses fabricated in-memory metadata "
                   "only -- no real dataset file is read or written by this tool.",
        "overall_status": "PASS" if all_pass else "FAIL",
        "n_scenarios": len(RESULTS),
        "n_passed": sum(1 for r in RESULTS if r["status"] == "PASS"),
        "results": RESULTS,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "leakage_synthetic_acceptance_report.json"
    out.write_text(json.dumps(summary, indent=2))

    print(f"leakage_synthetic_acceptance_tests: {summary['n_passed']}/{summary['n_scenarios']} scenarios PASS")
    for r in RESULTS:
        print(f"  [{r['status']}] {r['scenario']}: expected_fail={r['expected_to_fail_loudly']} "
              f"actual_fail={r['actually_failed_loudly']}"
              + (f" ({r['relevant_finding_key']}={r['relevant_finding_count']})" if r['relevant_finding_key'] else ""))
    print(f"\noverall: {summary['overall_status']}")
    print(f"full report written to {out}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()

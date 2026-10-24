"""
clip_pipeline_runner.py -- shared "load one annotation's clip and run the
frozen pipeline over it" logic, factored out so feature_exporter.py and
pipeline_benchmark.py (both of which need exactly this) share ONE
implementation rather than a third and fourth copy of it (audit item #45:
"eliminate duplicate evidence interfaces where necessary for export").

Everything here is read-only orchestration of already-frozen code
(feature_extraction_harness.py, itself a read-only wrapper around
football_analysis.engine.FootballAnalysisEngine) plus this project's own
time_utils.py. No football-analysis decision logic lives in this file.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2

TOOLS_DIR = Path(__file__).resolve().parent
DATASET_ROOT = TOOLS_DIR.parent
REPO_ROOT = DATASET_ROOT.parent
for p in (TOOLS_DIR, DATASET_ROOT, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import time_utils                                     # noqa: E402
import event_matching                                  # noqa: E402
import feature_extraction_harness as harness            # noqa: E402
from football_analysis.colour import shirt_color_hist   # noqa: E402  (frozen, unmodified)

CLIPS_DIR = DATASET_ROOT / "dataset" / "clips"


class ClipPipelineError(Exception):
    pass


@dataclass
class LoadedClip:
    annotation: dict
    clip_time_map: "time_utils.ClipTimeMap"
    clip_meta: "time_utils.VideoMeta"
    clip_video_path: Path
    tracking_init: dict


def _resolve_clip_path(raw_path: str) -> Path:
    p = Path(raw_path)
    return p if p.is_absolute() else (DATASET_ROOT / p)


def load_tracking_init(clip_dir: Path) -> dict:
    p = clip_dir / "tracking_init.json"
    if not p.exists():
        raise ClipPipelineError(
            f"no tracking_init.json in {clip_dir} -- see feature_exporter.py's module "
            f"docstring for the expected sidecar contract."
        )
    return json.loads(p.read_text())


def load_annotation_and_clip(annotation_path) -> LoadedClip:
    ann = json.loads(Path(annotation_path).read_text())
    clip_id = ann["clip_id"]
    clip_dir = CLIPS_DIR / clip_id
    ctm = time_utils.load_clip_time_map(str(clip_dir))
    clip_video_path = _resolve_clip_path(ctm.clip_path)
    clip_meta = time_utils.probe_video(str(clip_video_path))
    if abs(clip_meta.fps - ctm.source_fps) > 0.01:
        raise ClipPipelineError(
            f"clip {clip_id}: extracted clip's own fps ({clip_meta.fps}) does not match "
            f"clip_time_map.json's source_fps ({ctm.source_fps}) -- refusing to guess which "
            f"is right."
        )
    tracking_init = load_tracking_init(clip_dir)
    return LoadedClip(annotation=ann, clip_time_map=ctm, clip_meta=clip_meta,
                       clip_video_path=clip_video_path, tracking_init=tracking_init)


def run_pipeline(loaded: LoadedClip, verify: bool = False):
    """Runs Pass 1 + the read-only capture Pass 2 for one loaded clip.
    Returns a feature_extraction_harness.CaptureResult. Raises
    ClipPipelineError if --verify is requested and the harness's output
    diverges from a real engine.run_analysis_pass() call."""
    ti = loaded.tracking_init
    init_box = tuple(ti["init_box"])
    max_frames = ti.get("max_frames") or loaded.clip_meta.frame_count

    snapshots = harness.run_pass1_tracking(
        str(loaded.clip_video_path), ti["init_frame"], init_box,
        ti["player_id"], ti.get("jersey_number"), fps=loaded.clip_meta.fps, max_frames=max_frames,
    )

    cap = cv2.VideoCapture(str(loaded.clip_video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, ti["init_frame"])
    ok, frame0 = cap.read()
    cap.release()
    if not ok:
        raise ClipPipelineError(f"could not read init_frame {ti['init_frame']} from {loaded.clip_video_path}")
    tracked_color_hist = shirt_color_hist(frame0, init_box)

    capture = harness.run_with_capture(str(loaded.clip_video_path), snapshots, tracked_color_hist,
                                        fps=loaded.clip_meta.fps)

    if verify:
        ok = harness.verify_against_production(str(loaded.clip_video_path), snapshots, tracked_color_hist,
                                                loaded.clip_meta.fps, capture)
        if not ok:
            raise ClipPipelineError(
                f"clip {loaded.annotation['clip_id']}: feature_extraction_harness diverges from "
                f"the real engine.run_analysis_pass() -- see stderr above for the diff."
            )
    return capture


def find_matched_event(loaded: LoadedClip, capture) -> Optional[dict]:
    """The pipeline event (clip-relative start/end frames) whose window
    overlaps the annotation's human ground-truth window the most, or None if
    the pipeline produced nothing overlapping it at all."""
    gt = loaded.annotation["human_label"]["ground_truth_event_window"]
    ctm = loaded.clip_time_map
    fps = loaded.clip_meta.fps
    gt_start_frame = time_utils.seconds_to_frames(
        ctm.absolute_seconds_to_clip_relative_seconds(gt["start_time_s"]), fps)
    gt_end_frame = time_utils.seconds_to_frames(
        ctm.absolute_seconds_to_clip_relative_seconds(gt["end_time_s"]), fps)
    return event_matching.best_matching_event(capture.result.get("events", []), gt_start_frame, gt_end_frame)

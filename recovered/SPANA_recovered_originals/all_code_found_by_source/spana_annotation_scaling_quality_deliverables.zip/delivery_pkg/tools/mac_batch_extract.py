#!/usr/bin/env python3
"""Driver that runs mac_clip_extract.extract() for every entry in
clip_specs.json, writing each clip to _spana_clips_out/<clip_id>/ and
collecting a summary. Run from the Mac's _spana_clip_tools directory."""
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mac_clip_extract as mce

HERE = Path(__file__).resolve().parent
DOWNLOADS = HERE.parent  # _spana_clip_tools is directly under Downloads
SPECS_PATH = HERE / "clip_specs.json"
OUT_ROOT = HERE / "_spana_clips_out"


def main():
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    end = int(sys.argv[2]) if len(sys.argv) > 2 else 10**9
    specs = json.loads(SPECS_PATH.read_text())
    results = []
    for spec in specs[start:end]:
        clip_id = spec["clip_id"]
        source_path = str(DOWNLOADS / spec["source_path"])
        out_dir = OUT_ROOT / clip_id
        try:
            result = mce.extract(
                source_path=source_path,
                source_video_id=spec["source_video_id"],
                match_id=spec["match_id"],
                clip_id=clip_id,
                event_start_s=spec["event_start_s"],
                event_end_s=spec["event_end_s"],
                pre_roll_s=spec["pre_roll_s"],
                post_roll_s=spec["post_roll_s"],
                out_dir=str(out_dir),
            )
            results.append({
                "clip_id": clip_id, "ok": True,
                "clip_bytes": result["clip_bytes"],
                "source_fps": result["source_meta"]["fps"],
                "clip_fps_probed": result["clip_meta"]["fps"],
                "analysis_context_window_absolute_s": result["ctm"]["analysis_context_window_absolute_s"],
                "ground_truth_event_window_absolute_s": result["ctm"]["ground_truth_event_window_absolute_s"],
                "absolute_start_frame": result["ctm"]["absolute_start_frame"],
                "absolute_end_frame": result["ctm"]["absolute_end_frame"],
            })
        except Exception as e:
            results.append({"clip_id": clip_id, "ok": False, "error": str(e), "trace": traceback.format_exc()[-1500:]})

    print(json.dumps(results, indent=2))
    (HERE / f"batch_result_{start}_{end}.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

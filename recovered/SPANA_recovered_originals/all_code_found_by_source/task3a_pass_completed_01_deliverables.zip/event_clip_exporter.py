"""
event_engine_v1/event_clip_exporter.py -- Task 3A, Step 8.

Reuses the project's one canonical ffmpeg extraction primitive,
spana_dataset/tools/clip_extractor.py's _run_ffmpeg_extract() (its own
docstring: "the one and only place ffmpeg is invoked, so every clip this
tool ever produces goes through the same frame-accurate path") -- re-encode,
libx264, forced constant-frame-rate at the source's own ffprobe'd fps.
Nothing about that ffmpeg command is duplicated or re-implemented here.

What this module does NOT reuse is clip_extractor.py's extract_clip()
wrapper itself, because that function assumes a full match registered
under a source_video_id in dataset_videos_registry.json, with
event_start_s/event_end_s clamped against THAT registry's probed source
duration. Task 3A's test clips (pass_completed_01.mov, etc.) are already
short, standalone source files handed to this pipeline directly by path --
there is nothing to register and no full-match duration to clamp against.
So this module clamps padding against the clip's OWN probed duration
(time_utils.probe_video, the same ground-truth-fps discipline, just
applied to a different source) and calls _run_ffmpeg_extract directly.

~2 seconds before/after the event, per the task document, clamped to
[0, clip_duration_s] -- never a fabricated pre/post-roll beyond what the
source actually has.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for p in (REPO_ROOT, REPO_ROOT / "spana_dataset" / "tools"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from clip_extractor import _run_ffmpeg_extract  # noqa: E402  (canonical, unduplicated ffmpeg call)
import time_utils  # noqa: E402

PRE_ROLL_S = 2.0
POST_ROLL_S = 2.0

# event_engine_v1 schema `type` -> output folder, per the approved spec's
# folder structure (output/{passes,carries,shots,duels,touches,
# possession_won,possession_lost}/). GOAL/CROSS/UNKNOWN_RELEASE/RECEIVE
# are not in the user's named V1 folder list -- routed to the nearest
# named folder (GOAL -> shots, CROSS -> passes, RECEIVE -> touches) rather
# than silently dropped or given an unrequested new folder; UNKNOWN_RELEASE
# has no reasonable home in the named list and is skipped (not exported),
# which is reported, not hidden, in the run's deliverables.
OUTPUT_FOLDER_BY_TYPE = {
    "PASS": "passes", "CROSS": "passes",
    "CARRY": "carries",
    "SHOT": "shots", "GOAL": "shots",
    "DUEL": "duels",
    "TOUCH": "touches", "RECEIVE": "touches",
    "POSSESSION_WON": "possession_won",
    "POSSESSION_LOST": "possession_lost",
}


def _timestamp_tag(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(round(seconds - m * 60))
    return f"{m:02d}m{s:02d}s"


def export_event_clips(video_path: str, event_log: list, out_dir: str) -> list:
    """event_log: the final normalized event log (event_engine_v1 schema).
    Writes one clip per event whose type has a mapped output folder.
    Returns the list of written clip paths (absolute), in event_log order,
    plus logs (does not export) any event type with no folder mapping."""
    source_meta = time_utils.probe_video(video_path)
    out_root = Path(out_dir)
    written = []
    skipped_types = set()
    counters = {}
    for e in event_log:
        folder = OUTPUT_FOLDER_BY_TYPE.get(e["type"])
        if folder is None:
            skipped_types.add(e["type"])
            continue
        counters[folder] = counters.get(folder, 0) + 1
        idx = counters[folder]

        start_s = max(0.0, e["start_time"] - PRE_ROLL_S)
        end_s = min(source_meta.duration_s, e["end_time"] + POST_ROLL_S)
        duration_s = max(end_s - start_s, 1.0 / source_meta.fps)

        outcome_tag = (e.get("outcome") or "EVENT").upper()
        fname = f"{idx:03d}_{e['type']}_{outcome_tag}_{_timestamp_tag(e['start_time'])}.mp4"
        folder_dir = out_root / folder
        folder_dir.mkdir(parents=True, exist_ok=True)
        out_path = folder_dir / fname

        _run_ffmpeg_extract(video_path, start_s, duration_s, source_meta.fps, out_path)
        written.append(str(out_path))

    if skipped_types:
        print(f"event_clip_exporter: no output folder mapped for types {sorted(skipped_types)} -- not exported")
    return written

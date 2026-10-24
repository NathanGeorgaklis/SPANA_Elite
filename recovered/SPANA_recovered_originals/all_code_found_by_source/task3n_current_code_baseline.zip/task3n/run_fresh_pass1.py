"""Task 3N: fresh Pass-1 runs from source video with the current, unmodified code.
Seeds come from the frozen fixtures. Default config, no jersey hint, default max_frames (450:
stops at end of video). Fresh output directories, so no checkpoint or tracking JSON is reused."""
import json, sys, time
from player_tracker_mo.run_mo import run
import football_analysis.possession_detector as pd
import football_analysis.opponent_detection as od
from event_engine_v1.benchmark_fixture import BenchmarkFixture

assert pd.TASK3L_OTHER_PLAYER_CONTEXT is True and od.TASK3L_FRAGMENT_EXCLUSION is True  # current code state
fx_path, out_dir = sys.argv[1], sys.argv[2]
fx = BenchmarkFixture.load(fx_path)
t0 = time.time()
s = run(fx.video_path, fx.seed_frame, list(fx.seed_bbox), jersey_number=None, out_dir=out_dir, debug=True)
print(json.dumps({k: s[k] for k in ("video", "start_frame", "processed_frames", "processing_time_s",
                                    "selected_state_pct", "identity_conflicts_logged")}), flush=True)

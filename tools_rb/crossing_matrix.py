# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R1-F instruction (zero silent identity swaps in synthetic crossing tests); tests_rb/_synth_scene
"""R1-F synthetic crossing matrix: kit (same kit at the realistic same-kit cosine / opponent) x speed x vertical
offset x detection mode (both visible / merged into one box / far person missed while >= half hidden).

Reports per case: silent swaps (TRACKING on a box that is not exactly the target), frames TRACKING, frames on which
the target was detected alone, coverage = TRACKING / detected-alone, final state and episode count. Synthetic
only: it says nothing about real footage.

    python3 tools_rb/crossing_matrix.py OUT.json
"""
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests_rb.test_target import KIT_A, KIT_B, SAME_KIT_COS, crossing_frames, people
from tests_rb._synth_scene import run


def main(out_path):
    rows = []
    for kit_name, kit, cos in (("same_kit", KIT_A, SAME_KIT_COS), ("opponent", KIT_B, 0.2)):
        for speed in (2, 5, 9):
            for dy in (0, 15, 45):
                for mode in ("visible", "merge", "hide"):
                    fr = crossing_frames(int(600 / speed) + 30, speed, dy, merge=mode == "merge",
                                         hide=mode == "hide")
                    r = run(fr, people(("PERSON_A", kit, cos)))
                    alone = [k for k in range(len(fr)) if any(d.owners == ("TARGET",) for d in r.dets[k])]
                    trk = r.tracking_frames()
                    rows.append(dict(kit=kit_name, cos=cos, speed_px_per_frame=speed, dy_px=dy, mode=mode,
                                     frames=len(fr), swaps=len(r.swaps()), tracking=len(trk),
                                     target_detected_alone=len(alone), coverage=round(len(trk) / len(alone), 3),
                                     final=r.frames[-1].rb_state, episodes=r.frames[-1].target_episode,
                                     bank_pure=r.bank_is_pure()))
    cov = [x["coverage"] for x in rows]
    summary = dict(cases=len(rows), total_swaps=sum(x["swaps"] for x in rows),
                   final_tracking=sum(x["final"] == "TRACKING" for x in rows),
                   banks_pure=sum(x["bank_pure"] for x in rows),
                   coverage_median=statistics.median(cov), coverage_min=min(cov))
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=1)
    print(json.dumps(summary))


if __name__ == "__main__":
    main(sys.argv[1])

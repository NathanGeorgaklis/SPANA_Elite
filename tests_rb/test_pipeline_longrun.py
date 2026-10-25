# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R1-G instruction (Nathan, 2026-10-25: long-run stability/memory test); project rule 32 (design for
#                 long matches: identity drift, repeated exits/re-entry, camera cuts, memory growth, stale state)
"""Long-run stability of the integrated pipeline on a synthetic 'match' (default 1800 frames = 60 s at 30 fps, three cuts; set
SPANA_LONG_RUN_FRAMES for a longer one-off run).

The clip has hard cuts (new shot every 600 frames, everyone repositioned), a slow camera pan, people entering and
leaving on schedules, crossings that arise naturally, and the target leaving the frame and coming back. Checked:
  * the whole document validates (R1 contract and historical contract after strip_v2);
  * identity safety over the whole run: zero silent swaps, the target never written twice, no contaminated bank
    sample, never trusted without the target's own detection;
  * no state kept between frames grows with time: every population / target state size in the last third of the run
    is within what the first two thirds already reached; writer state is bounded by the number of distinct tracks;
  * Python heap growth (tracemalloc) over the second half of the run stays small.
"""
import math
import os
import shutil
import tempfile
import tracemalloc
import unittest

import numpy as np

from player_tracker_rb import output_schema as S
from player_tracker_rb.config import TrackerConfig
from tests_rb._synth_pipeline import H, PH, PW, W, ClipFrame, box, run_clip
from tests_rb._synth_scene import SceneFrame, make_people

N_FRAMES = int(os.environ.get("SPANA_LONG_RUN_FRAMES", "1800"))
SHOT_LEN = 600
KIT_A = (200, 60, 40)
KIT_B = (40, 40, 210)
HEAP_GROWTH_MAX_MB = 10.0


def _match(n):
    rng = np.random.default_rng(7)
    spec = [("TARGET", KIT_A, 1.0)]
    for k in range(3):
        spec.append((f"SAME_{k}", KIT_A, (0.6, 0.7, 0.75)[k]))      # same kit, below the reacquisition floor
    for k in range(4):
        spec.append((f"OPP_{k}", KIT_B, 0.2))
    people = make_people(spec)
    lo_x, hi_x, lo_y, hi_y = PW, W - PW, PH, H - PH / 2
    pos = {}
    frames = []
    shot = -1
    for f in range(n):
        if f % SHOT_LEN == 0:                                         # a new shot: everyone repositioned
            shot += 1
            for name, _, _ in spec:
                pos[name] = [rng.uniform(lo_x, hi_x), rng.uniform(lo_y, hi_y), *rng.uniform(-2.5, 2.5, 2)]
        pan = (f % SHOT_LEN) // 2 if shot % 3 == 1 else 0             # slow pan in every third shot
        boxes = {}
        for i, (name, _, _) in enumerate(spec):
            p = pos[name]
            if f % 90 == i * 7 % 90:                                  # new heading now and then
                p[2], p[3] = rng.uniform(-2.5, 2.5, 2)
            p[0] += p[2]
            p[1] += p[3]
            if not lo_y <= p[1] <= hi_y:
                p[3] = -p[3]
                p[1] = min(max(p[1], lo_y), hi_y)
            in_scene = True
            if name == "TARGET":
                t = f % SHOT_LEN
                if 300 <= t < 360:                                    # walks out of view and back
                    in_scene = False
                if t == 360:
                    p[0], p[2] = W - PW, -2.0
            else:
                period = 240 + 40 * i
                in_scene = (f + 37 * i) % period < period * 2 // 3
            if not lo_x <= p[0] <= hi_x:
                p[2] = -p[2]
                p[0] = min(max(p[0], lo_x), hi_x)
            if in_scene:
                boxes[name] = box(p[0], p[1])     # image coordinates (any image motion is a valid world motion
                #                                   under the pan, which moves only the static background)
        frames.append(ClipFrame(SceneFrame(boxes), shot, pan))
    return frames, people


class LongRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="r1g_long_")
        frames, people = _match(N_FRAMES)
        cls.sizes = []
        cls.heap = {}

        def on_frame(f, sizes):
            cls.sizes.append((f, sizes))
            if f == N_FRAMES // 4:
                tracemalloc.start()
            if f in (N_FRAMES // 2, N_FRAMES - 2):
                cls.heap[f] = tracemalloc.get_traced_memory()[0]
        try:
            cls.e = run_clip(frames, people, cls.tmp, name="long", start_frame=1, on_frame=on_frame)
        finally:
            if tracemalloc.is_tracing():
                tracemalloc.stop()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_document_validates(self):
        self.assertEqual(S.validate_document(self.e.doc, require_v2=True), [])
        self.assertEqual(S.validate_document(S.strip_v2(self.e.doc)), [])
        self.assertEqual(self.e.doc["summary"]["processed_frames"], N_FRAMES - 2)
        cuts = sum(1 for fr in self.e.doc["frame_log"] if fr["is_cut"])
        self.assertEqual(cuts, (N_FRAMES - 1) // SHOT_LEN)

    def test_identity_safety_over_the_whole_run(self):
        e = self.e
        self.assertEqual(e.swaps(), [])
        self.assertEqual(e.target_duplicated(), [])
        self.assertEqual(e.bank_violations(), [])
        for f in range(1, e.n + 1):
            if e.target_det_box(f) is None:
                self.assertFalse(e.target(f)["v2"]["identity_trusted"], f)
        visible = [f for f in range(1, e.n + 1) if e.target_det_box(f) is not None]
        trusted = e.trusted_frames()
        self.assertGreater(len(trusted) / len(visible), 0.3, "the target must not be abandoned for the whole run")
        self.assertGreaterEqual(len(e.summary["v2"]["reacquisition_events"]), 2)

    def test_no_state_grows_with_time(self):
        """Structural bounds that hold on every frame, plus no upward trend in the last third of the run."""
        cfg = TrackerConfig.default()
        fps = self.e.doc["summary"]["fps"]
        size_hist_max = math.ceil(cfg.get("target_height_window_s") * fps) + 1
        verif_max = math.ceil(cfg.get("reacq_window_s") * fps) + 2
        for f, s in self.sizes:
            known_bodies = s["population_live"] + s["population_closed_memory"]
            self.assertLessEqual(s["target_known_others"], known_bodies, f)
            self.assertLessEqual(s["target_tracklets"], known_bodies + 1, f)
            self.assertLessEqual(s["target_size_history"], size_hist_max, f)
            self.assertLessEqual(s["target_verification_history"], verif_max, f)
            self.assertLessEqual(s["writer_live_last_observed"], s["population_live"], f)
            self.assertLessEqual(s["target_bank_samples"], cfg.get("bank_max_samples"), f)
        n = len(self.sizes)
        early, late = self.sizes[: 2 * n // 3], self.sizes[2 * n // 3:]
        growing_ok = {"target_reacquisition_events", "writer_final_records", "writer_lifetimes",
                      "writer_population_ids_seen", "writer_conflict_pairs", "writer_provisional_ids"}
        for key in self.sizes[0][1]:
            if key in growing_ok:
                continue
            with self.subTest(key=key):
                e_mean = sum(s[key] for _, s in early) / len(early)
                l_mean = sum(s[key] for _, s in late) / len(late)
                self.assertLessEqual(l_mean, 1.5 * e_mean + 1, f"{key} trends upwards")
        last = self.sizes[-1][1]
        s = self.e.doc["summary"]
        self.assertEqual(last["writer_final_records"], s["n_tracks_ever_created"])
        self.assertLessEqual(last["writer_final_records"], last["writer_population_ids_seen"] + 1)

    def test_heap_growth_is_small(self):
        a, b = self.heap[N_FRAMES // 2], self.heap[N_FRAMES - 2]
        self.assertLess((b - a) / 1e6, HEAP_GROWTH_MAX_MB, f"heap grew {(b - a) / 1e6:.1f} MB over half the run")


if __name__ == "__main__":
    unittest.main()

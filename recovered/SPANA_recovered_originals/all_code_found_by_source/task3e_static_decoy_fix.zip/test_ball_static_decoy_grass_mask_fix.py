"""
test_ball_static_decoy_grass_mask_fix.py -- TASK 3E: required general
synthetic tests (I-P) for the grass-mask false-positive fix in
PitchRegionClassifier (football_analysis/ball_static_decoy.py).

Root cause being regression-tested (full evidence in
DUEL_LOST_STATIC_DECOY_ROOT_CAUSE.md): visual_pitch_calibration._grass_mask()
can fire on scattered, isolated dark/desaturated pixels that coincidentally
fall inside its HSV range -- e.g. shadow/anti-aliasing pixels inside a
broadcast sponsor-logo or scorebug graphic -- and its own internal 9x9
morphological CLOSE then bridges that scatter into one small, solid, FALSE
"grass" island. Because PitchRegionClassifier.classify() samples a small
local window and OFF_PITCH_WEIGHT["ON_PITCH"] == 0.0, landing that window
inside such a false island makes decoy_score mathematically 0.0 forever,
regardless of how persistent/static the candidate is -- a structural,
general bug (not specific to this one clip's logo), reproduced here with
fresh synthetic geometry unrelated to any real clip.

The fix (ball_static_decoy.py, MIN_GRASS_COMPONENT_WINDOWS /
_filter_small_grass_components): PitchRegionClassifier discards connected
components of its own private copy of the grass mask that are smaller than
a multiple of its own existing GRASS_LOCAL_WINDOW sampling-window area,
before using that mask for classification. visual_pitch_calibration.py
itself is never modified (rule #8 preserved) -- every other caller of
_grass_mask/_pitch_horizon (goal detection, release geometry, pitch
timeline, clip tagging, calibration overlays) is completely unaffected.

These frames deliberately use the SAME full-resolution scale as the real
duel_lost_01 clip (2056x1144) so PitchRegionClassifier's default
downsample_width=240 produces the same scale/window_area/min_area as the
real diagnosis -- this is a property of the classifier's own math (ties the
filter to its own constant), not a clip-specific tuning.
"""
import sys
sys.path.insert(0, "/tmp/claude-0/-home-claude/98d69dca-3149-5a43-9670-3c417f2af6d4/scratchpad")

import numpy as np
import cv2

from football_analysis.ball_static_decoy import PitchRegionClassifier, StaticDecoyTracker
from football_analysis.ball_tracker import BallTracker

W, H = 2056, 1144        # same full-res scale as the real diagnosed clip
HORIZON_Y = 500           # rows [0, HORIZON_Y) = crowd/board, [HORIZON_Y, H) = grass
GRASS_BGR = (40, 160, 40)
BOARD_BGR = (20, 20, 20)  # dark background, like the real logo's dark panel


def make_frame(decoy_blobs=None):
    """decoy_blobs: list of (x, y, w, h) rects, drawn in GRASS_BGR inside the
    BOARD region -- simulating small color-coincidence false positives
    (never touching the real grass band, exactly like the real logo sitting
    well inside the frame away from the true horizon)."""
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    frame[:HORIZON_Y, :] = BOARD_BGR
    frame[HORIZON_Y:, :] = GRASS_BGR
    for (x, y, w, h) in (decoy_blobs or []):
        frame[y:y + h, x:x + w] = GRASS_BGR
    return frame


def new_tracker(**cfg):
    return BallTracker(light_model=None, heavy_model_factory=None, fps=25.0, config=cfg or None)


def last_scored(tracker):
    return tracker._last_search_meta.get("scored", [])


def find_candidate_near(scored, cx, cy, tol=40.0):
    from football_analysis.geometry import center, dist
    best = None
    for sc in scored:
        d = dist(center(sc.box), (cx, cy))
        if d <= tol and (best is None or d < dist(center(best.box), (cx, cy))):
            best = sc
    return best


RESULTS = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    RESULTS.append((name, status))
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))
    if status == "FAIL":
        raise AssertionError(f"{name}: {detail}")


# ---------------------------------------------------------------- Scenario I
# A small (~60x50px full-res, i.e. logo/scorebug-sized -- far smaller than
# any real patch of pitch) isolated grass-coloured blob, well inside the
# BOARD region, nowhere near the true horizon. Direct PitchRegionClassifier
# test: must NOT be ON_PITCH (this is the exact structural bug: before the
# fix, any grass-coloured island -- however small -- that survives
# _grass_mask()'s own internal CLOSE scored ON_PITCH).
def scenario_i():
    blob = (1700, 200, 60, 50)   # cx=1730, cy=225 -- well above HORIZON_Y=500, inside the board band
    frame = make_frame(decoy_blobs=[blob])
    clf = PitchRegionClassifier()
    clf.prepare(frame, 0)
    region = clf.classify(1730, 225)
    check("I: small logo-sized grass-coloured island is NOT classified ON_PITCH",
          region != "ON_PITCH", f"region={region}")


# ---------------------------------------------------------------- Scenario J
# Paired regression check for I, same frame: the genuine, large, continuous
# grass band is completely unaffected by the component-area filter.
def scenario_j():
    blob = (1700, 200, 60, 50)
    frame = make_frame(decoy_blobs=[blob])
    clf = PitchRegionClassifier()
    clf.prepare(frame, 0)
    mid_pitch = clf.classify(1000, 800)
    near_player_area = clf.classify(300, 700)
    check("J: genuine large grass band still classifies ON_PITCH (mid-pitch point)",
          mid_pitch == "ON_PITCH", f"region={mid_pitch}")
    check("J: genuine large grass band still classifies ON_PITCH (far-side point)",
          near_player_area == "ON_PITCH", f"region={near_player_area}")


# ---------------------------------------------------------------- Scenario K
# End-to-end regression test for the EXACT real failure mode: a high-
# confidence, static candidate sitting on a small grass-coloured island in
# the board region must never reach CONFIRMED and must be classified as a
# decoy by the time its static history has matured -- this is scenario
# A/B's structure, but specifically through a small ON_PITCH-mimicking
# island rather than a plain off-pitch board colour, so it exercises the
# exact mechanism this task fixed (pre-fix, this candidate's decoy_score was
# mathematically 0.0 forever and it DID reach CONFIRMED).
def scenario_k():
    blob = (1700, 200, 60, 50)
    decoy_box = (1715, 215, 1745, 240)   # centre (1730, 227.5), inside the blob
    tracker = new_tracker()
    last_state = None
    for f in range(15):
        frame = make_frame(decoy_blobs=[blob])
        state = tracker.step(f, frame, [(decoy_box, 0.7)])
        last_state = state
    scored = last_scored(tracker)
    sc = find_candidate_near(scored, 1730, 227.5)
    check("K: static candidate on a small grass-coloured island never reaches CONFIRMED",
          last_state.state != "CONFIRMED", f"final state={last_state.state}")
    check("K: that candidate is classified STATIC_DECOY or OFF_PITCH_CANDIDATE, not REAL_BALL_CANDIDATE",
          sc is not None and sc.classification in ("STATIC_DECOY", "OFF_PITCH_CANDIDATE"),
          f"classification={sc.classification if sc else None} pitch_region={sc.pitch_region if sc else None} "
          f"decoy_score={sc.decoy_score if sc else None}")


# ---------------------------------------------------------------- Scenario L
# Several small SEPARATE grass-coloured speckles, spaced far enough apart
# that even _grass_mask()'s internal 9x9 CLOSE cannot bridge them into one
# bigger blob -- confirms the fix rejects each individually, not relying on
# them being contiguous, and confirms the filter copes with multiple
# simultaneous small components in one frame (not just a single-blob case).
def scenario_l():
    blobs = [(1300, 200, 60, 50), (1600, 250, 55, 45), (1850, 150, 50, 55)]
    frame = make_frame(decoy_blobs=blobs)
    clf = PitchRegionClassifier()
    clf.prepare(frame, 0)
    regions = [clf.classify(x + w // 2, y + h // 2) for (x, y, w, h) in blobs]
    check("L: all scattered small grass-coloured speckles are NOT classified ON_PITCH",
          all(r != "ON_PITCH" for r in regions), f"regions={regions}")


# ---------------------------------------------------------------- Scenario M
# Honest edge-case check, not a claim of perfection: a small but GENUINE
# patch of grass (below the filter's min-area threshold, e.g. visible
# through a narrow gap) is treated the same as any other sub-threshold
# island -- i.e. not asserted as ON_PITCH, but the classifier must still
# return a valid, non-crashing region (falls through to the existing
# horizon-based NEAR_PITCH/OFF_PITCH logic, same as any location with no
# nearby grass island, rather than raising or returning garbage). This
# documents a known, bounded limitation rather than hiding it.
def scenario_m():
    tiny_patch = (1700, 200, 15, 12)   # genuinely "grass" in this synthetic world, but logo-sized
    frame = make_frame(decoy_blobs=[tiny_patch])
    clf = PitchRegionClassifier()
    clf.prepare(frame, 0)
    region = clf.classify(1707, 206)
    check("M: a sub-threshold genuine-grass patch still returns a valid region without crashing",
          region in ("ON_PITCH", "NEAR_PITCH", "OFF_PITCH", "UNKNOWN"), f"region={region}")


# ---------------------------------------------------------------- Scenario N
# A real ball moving normally across the genuine grass band, in the SAME
# frame as a small rejected decoy island elsewhere -- confirms the fix
# introduces no cross-talk: the real ball's own tracking is unaffected by
# the presence/rejection of an unrelated small island far away.
def scenario_n():
    blob = (1700, 200, 60, 50)
    tracker = new_tracker()
    x, y = 300.0, 700.0
    last_state = None
    for f in range(15):
        frame = make_frame(decoy_blobs=[blob])
        ball_box = (x - 10, y - 10, x + 10, y + 10)
        state = tracker.step(f, frame, [(ball_box, 0.55)])
        last_state = state
        x += 15.0
    scored = last_scored(tracker)
    sc = find_candidate_near(scored, x - 15, y, tol=25.0)
    check("N: real moving ball on genuine grass reaches CONFIRMED despite an unrelated rejected island in-frame",
          last_state.state == "CONFIRMED", f"final state={last_state.state}")
    check("N: that real ball is classified REAL_BALL_CANDIDATE",
          sc is not None and sc.classification == "REAL_BALL_CANDIDATE",
          f"classification={sc.classification if sc else None}")


# ---------------------------------------------------------------- Scenario O
# No-crash / sane-default check: a frame with NO grass pixels at all (pure
# board/crowd, e.g. a tight close-up or a replay-graphic-only frame) must
# not crash the connected-components filter (n<=1 early-return path) and
# must classify as a non-ON_PITCH region everywhere.
def scenario_o():
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    frame[:, :] = BOARD_BGR
    clf = PitchRegionClassifier()
    clf.prepare(frame, 0)
    region = clf.classify(1000, 600)
    check("O: an all-board frame (zero grass pixels) does not crash and is not ON_PITCH",
          region in ("NEAR_PITCH", "OFF_PITCH", "UNKNOWN"), f"region={region}")


# ---------------------------------------------------------------- Scenario P
# Direct tie-back to the real diagnosed clip: re-run PitchRegionClassifier
# against real duel_lost_01.mov frames at the exact Mitre-logo screen
# location that was empirically measured as the root cause. Confirms the
# fix holds on the real footage, not just synthetic geometry. Skips cleanly
# (does not fail the suite) if the real upload is unavailable in whatever
# environment this test is later run from.
def scenario_p():
    import os
    VIDEO = "/root/.claude/uploads/98d69dca-3149-5a43-9670-3c417f2af6d4/8a710009-dual_lost_01.mov"
    if not os.path.exists(VIDEO):
        check("P: (skipped -- real duel_lost_01.mov not present in this environment)", True)
        return
    cap = cv2.VideoCapture(VIDEO)
    clf = PitchRegionClassifier()
    regions = []
    for abs_frame in (53, 150, 300):
        cap.set(cv2.CAP_PROP_POS_FRAMES, abs_frame)
        ok, frame = cap.read()
        if not ok:
            continue
        clf.prepare(frame, abs_frame)
        regions.append(clf.classify(1888.0, 1049.0))
    cap.release()
    check("P: real Mitre-logo screen location is NOT ON_PITCH on real duel_lost_01.mov frames",
          len(regions) > 0 and all(r != "ON_PITCH" for r in regions), f"regions={regions}")


def main():
    scenarios = [
        ("I", scenario_i), ("J", scenario_j), ("K", scenario_k), ("L", scenario_l),
        ("M", scenario_m), ("N", scenario_n), ("O", scenario_o), ("P", scenario_p),
    ]
    failures = []
    for name, fn in scenarios:
        print(f"\n--- scenario {name} ---")
        try:
            fn()
        except AssertionError as e:
            failures.append((name, str(e)))
            print(f"[SCENARIO {name} FAILED] {e}")

    print("\n=== summary ===")
    for name, _ in scenarios:
        status = "FAIL" if any(n == name for n, _ in failures) else "PASS"
        print(f"  {name}: {status}")
    if failures:
        raise SystemExit(f"{len(failures)}/{len(scenarios)} scenario(s) failed")
    print(f"\nALL {len(scenarios)} SCENARIOS PASSED")


if __name__ == "__main__":
    main()

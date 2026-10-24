"""
test_ball_static_decoy.py -- STATIC-DECOY TASK, rule #18: required general
synthetic tests (A-H), driving the real BallTracker (light_model=None, so
no YOLO/search-tier escalation is involved -- every candidate comes from
the caller, exactly the way engine.py's tier-A candidates do) against
fully synthetic frames, so each named scenario from the task spec can be
checked directly and fast, without needing a real video.

None of these scenarios depend on any clip-specific location or threshold
(rule #9/#23.7) -- every synthetic geometry (grass/horizon split, decoy
location, pan speed) is chosen fresh per scenario and is unrelated to the
two real validated clips.
"""
import sys
sys.path.insert(0, "/tmp/claude-0/-home-claude/98d69dca-3149-5a43-9670-3c417f2af6d4/scratchpad")

import numpy as np
import cv2

from football_analysis.ball_tracker import BallTracker

W, H = 300, 200
HORIZON_Y = 70          # rows [0, HORIZON_Y) are "crowd/board" (non-grass), [HORIZON_Y, H) is grass
GRASS_BGR = (40, 160, 40)
BOARD_BGR = (90, 90, 90)


def make_frame(offset=0):
    """A simple, static-camera broadcast-style frame: a non-grass band
    (crowd/advertising boards) above a horizon line, grass below. `offset`
    shifts BOTH bands vertically by a few px to give the frame some
    (irrelevant, non-pan) frame-to-frame texture variation without
    changing the pitch geometry meaningfully -- unused by most scenarios
    (offset=0)."""
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    frame[:HORIZON_Y + offset, :] = BOARD_BGR
    frame[HORIZON_Y + offset:, :] = GRASS_BGR
    return frame


def make_pan_world(world_w=900, world_h=200, horizon_y=70, seed=0):
    """A wide 'world' canvas with the same grass/board vertical split PLUS
    random speckle texture, so BackgroundMotionEstimator's optical flow has
    real corners to lock onto -- required for scenario F, which needs a
    genuine camera pan to be detected, not just simulated by shifting a
    flat, textureless colour field (no corners on a flat field)."""
    rng = np.random.RandomState(seed)
    world = np.zeros((world_h, world_w, 3), dtype=np.uint8)
    world[:horizon_y, :] = BOARD_BGR
    world[horizon_y:, :] = GRASS_BGR
    speckle_xy = rng.randint(0, [world_w, world_h], size=(500, 2))
    speckle_color = rng.randint(0, 255, size=(500, 3))
    for (x, y), col in zip(speckle_xy, speckle_color):
        cv2.circle(world, (int(x), int(y)), 1, tuple(int(c) for c in col), -1)
    return world


def draw_box(frame, box, color=(255, 255, 255)):
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
    return frame


def new_tracker(**cfg):
    return BallTracker(light_model=None, heavy_model_factory=None, fps=25.0, config=cfg or None)


def last_scored(tracker):
    return tracker._last_search_meta.get("scored", [])


def find_candidate_near(scored, cx, cy, tol=25.0):
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


# ---------------------------------------------------------------- Scenario A
# Printed football on an ad board: fixed off-pitch location, every frame,
# deliberately HIGH confidence (0.65 -- higher than most real detections in
# this project ever score) so this genuinely tests rule #2/#16 ("high
# detector confidence alone must never be sufficient"), not just a low-
# confidence candidate losing on merit.
def scenario_a():
    tracker = new_tracker()
    decoy_box = (150, 20, 166, 34)   # centre (158, 27) -- well above HORIZON_Y=70, i.e. OFF_PITCH
    last_state = None
    for f in range(15):
        frame = make_frame()
        state = tracker.step(f, frame, [(decoy_box, 0.65)])
        last_state = state
    scored = last_scored(tracker)
    decoy_sc = find_candidate_near(scored, 158, 27)
    check("A: printed football on ad board never reaches CONFIRMED",
          last_state.state != "CONFIRMED", f"final state={last_state.state}")
    check("A: static decoy is classified STATIC_DECOY or OFF_PITCH_CANDIDATE by the last frame",
          decoy_sc is not None and decoy_sc.classification in ("STATIC_DECOY", "OFF_PITCH_CANDIDATE"),
          f"classification={decoy_sc.classification if decoy_sc else None}")


# ---------------------------------------------------------------- Scenario B
# Static circular league logo -- same structural test, different fixed
# off-pitch location/size, confirming this isn't scenario-A-shape-specific.
def scenario_b():
    tracker = new_tracker()
    decoy_box = (40, 10, 58, 28)   # centre (49, 19), OFF_PITCH
    last_state = None
    for f in range(15):
        frame = make_frame()
        state = tracker.step(f, frame, [(decoy_box, 0.55)])
        last_state = state
    scored = last_scored(tracker)
    decoy_sc = find_candidate_near(scored, 49, 19)
    check("B: static league logo never reaches CONFIRMED",
          last_state.state != "CONFIRMED", f"final state={last_state.state}")
    check("B: static logo is classified STATIC_DECOY or OFF_PITCH_CANDIDATE by the last frame",
          decoy_sc is not None and decoy_sc.classification in ("STATIC_DECOY", "OFF_PITCH_CANDIDATE"),
          f"classification={decoy_sc.classification if decoy_sc else None}")


# ---------------------------------------------------------------- Scenario C
# Real ball rolling across the pitch (ON_PITCH, moving ~6px/frame) -> must
# still track normally exactly as before this task (no regression).
def scenario_c():
    tracker = new_tracker()
    x, y = 40.0, 130.0
    last_state = None
    for f in range(15):
        frame = make_frame()
        box = (x - 6, y - 6, x + 6, y + 6)
        state = tracker.step(f, frame, [(box, 0.5)])
        last_state = state
        x += 6.0
    scored = last_scored(tracker)
    rolling_sc = find_candidate_near(scored, x - 6, y, tol=15.0)
    check("C: real ball rolling on pitch reaches CONFIRMED",
          last_state.state == "CONFIRMED", f"final state={last_state.state}")
    check("C: rolling ball classified REAL_BALL_CANDIDATE, not suppressed",
          rolling_sc is not None and rolling_sc.classification == "REAL_BALL_CANDIDATE",
          f"classification={rolling_sc.classification if rolling_sc else None}")


# ---------------------------------------------------------------- Scenario D
# Ball temporarily hidden behind players during a challenge, WHILE a static
# off-pitch decoy (already with several frames of accumulated static
# history, exactly like a real ad board that has been visible the whole
# clip) sits within plausible search radius of the occlusion point. Must
# stay PREDICTED/OCCLUDED/LOST, never switch identity to the decoy -- this
# is the exact failure mode diagnosed in the real second_match_carry clip.
def scenario_d():
    tracker = new_tracker()
    decoy_box = (120, 40, 134, 52)   # fixed, OFF_PITCH, present every frame from frame 0
    x, y = 40.0, 130.0
    last_states = []
    # frames 0-7: real ball tracked normally on-pitch (also builds up the
    # decoy's static history in the background, as it would in real footage)
    for f in range(8):
        frame = make_frame()
        ball_box = (x - 6, y - 6, x + 6, y + 6)
        state = tracker.step(f, frame, [(ball_box, 0.5), (decoy_box, 0.6)])
        last_states.append(state)
        x += 3.0
    assert last_states[-1].state == "CONFIRMED", "setup precondition failed: real ball should be tracked by frame 7"

    # frames 8-15: real ball detection vanishes (occluded in a challenge);
    # only the (by now well-established) static decoy is still detected.
    for f in range(8, 16):
        frame = make_frame()
        state = tracker.step(f, frame, [(decoy_box, 0.6)])
        last_states.append(state)

    occlusion_states = [s.state for s in last_states[8:]]
    check("D: never confirms onto the decoy during real-ball occlusion",
          all(s != "CONFIRMED" for s in occlusion_states), f"states={occlusion_states}")
    check("D: settles into PREDICTED/OCCLUDED/LOST during occlusion (never REACQUIRING stuck on the decoy forever)",
          occlusion_states[-1] in ("PREDICTED", "OCCLUDED", "LOST"), f"final state={occlusion_states[-1]}")


# ---------------------------------------------------------------- Scenario E
# Real stationary free-kick/penalty ball: perfectly static, but ON_PITCH
# (ordinary mid-pitch placement) -- must remain a possible match ball
# (i.e. still reach CONFIRMED) despite never moving, because rule #10
# forbids rejecting on motion-staticness alone without off-pitch/board
# context.
def scenario_e():
    tracker = new_tracker()
    ball_box = (150, 140, 162, 152)   # centre (156, 146), deep in the ON_PITCH grass band
    last_state = None
    for f in range(15):
        frame = make_frame()
        state = tracker.step(f, frame, [(ball_box, 0.5)])
        last_state = state
    scored = last_scored(tracker)
    sc = find_candidate_near(scored, 156, 146)
    check("E: stationary on-pitch free-kick ball still reaches CONFIRMED",
          last_state.state == "CONFIRMED", f"final state={last_state.state}")
    check("E: stationary on-pitch ball stays classified REAL_BALL_CANDIDATE despite high static motion history",
          sc is not None and sc.classification == "REAL_BALL_CANDIDATE",
          f"classification={sc.classification if sc else None} decoy_score={sc.decoy_score if sc else None}")


# ---------------------------------------------------------------- Scenario F
# Camera pan: a static ad graphic embedded in a wide textured "world" canvas
# moves in image-space pixel coordinates every frame purely because the
# camera (crop window) pans -- must be rejected once background motion
# compensation is applied, i.e. NOT simply because it "doesn't move" (it
# does move in pixels) but because it moves WITH the background.
def scenario_f():
    world = make_pan_world()
    pan_dx = 4.0    # px/frame camera pan speed
    # fixed in WORLD coordinates -- attached to the background. Chosen so the
    # board stays fully visible inside the panning WxH crop window for all
    # 20 frames (crop_x0 ranges 0..76 for a 300px-wide frame): world x=200
    # with a 16px-wide box stays within [crop_x0, crop_x0+300) throughout.
    board_world_x, board_world_y = 200, 30
    tracker = new_tracker()
    last_state = None
    prev_box_centre = None
    moved_in_pixels = False
    for f in range(20):
        crop_x0 = int(f * pan_dx)
        frame = world[:, crop_x0:crop_x0 + W].copy()
        bx = board_world_x - crop_x0
        box = (bx - 8, board_world_y - 7, bx + 8, board_world_y + 7)
        cx = (box[0] + box[2]) / 2.0
        if prev_box_centre is not None and abs(cx - prev_box_centre) > 0.5:
            moved_in_pixels = True
        prev_box_centre = cx
        state = tracker.step(f, frame, [(box, 0.6)])
        last_state = state
    scored = last_scored(tracker)
    sc = find_candidate_near(scored, prev_box_centre, board_world_y, tol=20.0)
    check("F: the ad graphic genuinely moves in pixel coordinates during the pan (sanity check on the test itself)",
          moved_in_pixels)
    check("F: never confirms the panning ad graphic as the ball",
          last_state.state != "CONFIRMED", f"final state={last_state.state}")
    check("F: after background-motion compensation, the panning static graphic is classified as a decoy "
          "(residual motion near zero despite nonzero raw pixel motion)",
          sc is not None and sc.classification in ("STATIC_DECOY", "OFF_PITCH_CANDIDATE"),
          f"classification={sc.classification if sc else None} residual={sc.residual_object_motion if sc else None}")


# ---------------------------------------------------------------- Scenario G
# Real airborne ball: OFF_PITCH (high above the grass, e.g. a cross/lofted
# pass), but moving with real independent motion every frame -- must NOT be
# rejected merely for being above the grass line.
def scenario_g():
    tracker = new_tracker()
    x, y = 40.0, 20.0   # y=20 is well above HORIZON_Y=70 -> OFF_PITCH throughout
    last_state = None
    for f in range(15):
        frame = make_frame()
        box = (x - 6, y - 6, x + 6, y + 6)
        state = tracker.step(f, frame, [(box, 0.5)])
        last_state = state
        x += 12.0   # fast, clearly independent motion -- an airborne ball, not scenery
    scored = last_scored(tracker)
    sc = find_candidate_near(scored, x - 12, y, tol=15.0)
    check("G: real airborne (off-grass) ball still reaches CONFIRMED",
          last_state.state == "CONFIRMED", f"final state={last_state.state}")
    check("G: airborne ball classified REAL_BALL_CANDIDATE despite being OFF_PITCH",
          sc is not None and sc.classification == "REAL_BALL_CANDIDATE",
          f"classification={sc.classification if sc else None} pitch_region={sc.pitch_region if sc else None}")


# ---------------------------------------------------------------- Scenario H
# Ball leaves the frame entirely: real ball tracked normally, then no
# further detections at all, while a static scoreboard-like decoy (built up
# from frame 0, exactly like scenario D) keeps appearing -- must settle
# into LOST (once past the PREDICTED patience window), never switch.
def scenario_h():
    tracker = new_tracker()
    decoy_box = (270, 15, 288, 30)   # fixed scoreboard-corner-like location, OFF_PITCH
    x, y = 40.0, 130.0
    states = []
    for f in range(8):
        frame = make_frame()
        ball_box = (x - 6, y - 6, x + 6, y + 6)
        state = tracker.step(f, frame, [(ball_box, 0.5), (decoy_box, 0.6)])
        states.append(state)
        x += 3.0
    assert states[-1].state == "CONFIRMED", "setup precondition failed: real ball should be tracked by frame 7"

    # ball has left the frame: no ball-shaped detections at all from here on,
    # only the persistent scoreboard/decoy candidate. Run well past
    # max_predicted_frames (default 10) so PREDICTED has a chance to decay to LOST.
    for f in range(8, 24):
        frame = make_frame()
        state = tracker.step(f, frame, [(decoy_box, 0.6)])
        states.append(state)

    check("H: never confirms onto the decoy after the real ball leaves frame",
          all(s.state != "CONFIRMED" for s in states[8:]), f"states={[s.state for s in states[8:]]}")
    check("H: settles into LOST (not stuck switched onto the decoy) once past prediction patience",
          states[-1].state == "LOST", f"final state={states[-1].state}")


def main():
    scenarios = [
        ("A", scenario_a), ("B", scenario_b), ("C", scenario_c), ("D", scenario_d),
        ("E", scenario_e), ("F", scenario_f), ("G", scenario_g), ("H", scenario_h),
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

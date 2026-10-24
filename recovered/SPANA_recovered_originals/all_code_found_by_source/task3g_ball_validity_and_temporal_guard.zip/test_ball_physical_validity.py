"""
test_ball_physical_validity.py -- TASK 3G required tests (Section 14) for
football_analysis/ball_interaction_credibility.py. Synthetic per-frame ball
evidence + the real PossessionDetector replay; geometry is fresh per scenario
(player box 100x160 at an arbitrary spot) and unrelated to any real clip.
Mitre/static-decoy (case 10) and pass_completed_01 (case 11) are covered by
the existing static-decoy suites and the real-clip regression run.
"""
import sys
sys.path.insert(0, "/tmp/claude-0/-home-claude/98d69dca-3149-5a43-9670-3c417f2af6d4/scratchpad")

from football_analysis.ball_interaction_credibility import (
    FrameBallEvidence, assess, gate_possession_frames, CREDIBLE, REVIEW, NOT_CREDIBLE, evidence_from_tracker)
from football_analysis.possession_detector import PossessionFrame
from football_analysis.possession_segmentation import segment_possession
from football_analysis.geometry import foot_point, dist

PB = [500.0, 300.0, 560.0, 460.0]          # player box, h = 160 -> expected ball ~19 px, band 9.6-32 px
FOOT = foot_point(PB)                        # (530, 460)
R = []


def check(name, cond, detail=""):
    R.append((name, cond)); print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


def obs(f, cx, cy, size=18, aspect=1.0, stage="stage3_player_feet_roi", conf=0.06, trusted=False,
        state="REACQUIRING", pb=PB):
    w, h = size, size / aspect
    return FrameBallEvidence(f, state, (cx, cy), [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], stage, conf, trusted, pb)


def pred(f, cx, cy, state="PREDICTED", pb=PB):
    return FrameBallEvidence(f, state, (cx, cy), None, None, None, True, pb)


def lost(f):
    return FrameBallEvidence(f, "LOST", None, None, None, None, False, PB)


def touches(evidence):
    """Real possession pipeline on the gated evidence: distance from the foot
    point, PossessionDetector replay, segment_possession."""
    labels, _ = assess(evidence)
    pfs = []
    for ev in evidence:
        d = dist(FOOT, ev.center) if (ev.center is not None and ev.state not in ("OCCLUDED", "LOST")) else None
        pfs.append(PossessionFrame(frame=ev.frame, ball_distance=d, ball_state=ev.state, in_possession=False,
                                   tracking_state="TRACKING"))
    gated = gate_possession_frames(pfs, labels)
    seg = segment_possession(gated)
    return sum(1 for s in seg if s.in_possession), labels


# 1. boot detected as ball: cold, at the foot, low-conf feet-ROI hits, ball-sized, moves WITH the foot only
def t1():
    ev = [lost(0)] + [obs(f, 528, 452, size=18) for f in range(1, 3)] + [lost(3), lost(4)]
    n, labels = touches(ev)
    check("1 boot false positive (cold, feet ROI, 2 frames) -> no credible touch", n == 0 and CREDIBLE not in labels,
          f"possession frames={n} labels={set(l for l in labels if l)}")


# 2. player shadow: cold, oversized and elongated (0.30 x player height), 3 consistent frames
def t2():
    ev = [lost(0)] + [obs(f, 532, 468, size=48, aspect=1.5, stage="stage4_full_frame_low_conf") for f in range(1, 4)] + \
         [obs(4, 532, 468, size=49, aspect=1.4, stage="tier_a_full_frame", conf=0.37, state="CONFIRMED")] + \
         [pred(f, 532 - 40 * (f - 4), 466) for f in range(5, 9)]
    n, labels = touches(ev)
    check("2 shadow false positive (oversized, even with one tier-A hit) -> no credible touch",
          n == 0 and labels[1] == NOT_CREDIBLE, f"possession frames={n} label={labels[1]}")


# 3. pitch marking: thin elongated box at the feet, consistent, never leaves
def t3():
    ev = [lost(0)] + [obs(f, 526, 462, size=30, aspect=5.0, stage="stage2c_contact_cluster_roi") for f in range(1, 5)]
    n, labels = touches(ev)
    check("3 pitch marking at the feet -> no credible touch (review at most)", n == 0 and labels[1] in (REVIEW, NOT_CREDIBLE),
          f"possession frames={n} label={labels[1]}")


# 4. true ball stationary at the feet (e.g. clip starts with player on the ball): tier-A, physical size
def t4():
    ev = [obs(0, 532, 462, size=20, stage="tier_a_full_frame", conf=0.42, state="REACQUIRING")] + \
         [obs(f, 532, 462, size=20, stage="tier_a_full_frame", conf=0.40, state="CONFIRMED", trusted=True) for f in range(1, 8)]
    n, labels = touches(ev)
    check("4 true stationary ball at feet (tier-A, physical size) -> retained", n >= 5 and labels[0] == CREDIBLE,
          f"possession frames={n} label={labels[0]}")


# 5. true ball rolling into the feet from open grass (continuous, weak detections on arrival)
def t5():
    ev = [obs(0, 700, 470, size=19, stage="tier_a_full_frame", conf=0.3, state="REACQUIRING")]
    x = 700.0
    for f in range(1, 16):
        x = max(534.0, x - 16)          # rolls in, then stops at the feet
        ev.append(obs(f, x, 466, size=19, stage="stage1_small_roi", conf=0.07, state="CONFIRMED", trusted=True))
    n, labels = touches(ev)
    check("5 true ball rolling into feet -> retained", n >= 3 and all(l == CREDIBLE for l in labels),
          f"possession frames={n}")


# 6. true ball released from feet: acquired weakly AT the feet, then leaves on a continuous trajectory
def t6():
    ev = [lost(0)] + [obs(f, 530, 462, size=19, conf=0.06) for f in range(1, 5)]
    x = 530.0
    for f in range(5, 12):
        x += 25
        ev.append(obs(f, x, 460, size=19, stage="stage1_small_roi", conf=0.08, state="CONFIRMED", trusted=True))
    n, labels = touches(ev)
    check("6 true ball released from feet (exit trajectory) -> touch retained", n >= 2 and labels[1] == CREDIBLE,
          f"possession frames={n} label={labels[1]}")


# 7. true ball temporarily occluded by the foot: credible track -> OCCLUDED frames -> reacquired on prediction
def t7():
    ev = [obs(0, 640, 466, size=19, stage="tier_a_full_frame", conf=0.4, state="REACQUIRING")]
    ev += [obs(f, 640 - 25 * f, 466, size=19, stage="stage1_small_roi", conf=0.2, state="CONFIRMED", trusted=True) for f in range(1, 5)]
    ev += [pred(f, 535, 462, state="OCCLUDED") for f in range(5, 10)]
    ev += [obs(f, 532, 462, size=19, stage="stage1_small_roi", conf=0.06, state="CONFIRMED", trusted=True) for f in range(10, 13)]
    n, labels = touches(ev)
    check("7 occluded-by-foot gap bridged by continuity -> retained", n >= 6 and all(l == CREDIBLE for l in labels if l),
          f"possession frames={n}")


# 8. weak single-frame player-adjacent detection -> abstain
def t8():
    ev = [lost(0), obs(1, 529, 461, size=18, conf=0.05), pred(2, 529, 461), pred(3, 529, 461), lost(4)]
    n, labels = touches(ev)
    check("8 weak one-frame player-adjacent detection -> abstain", n == 0 and labels[1] == NOT_CREDIBLE,
          f"possession frames={n} label={labels[1]}")


# 9. strong multi-frame real trajectory near the player -> retained
def t9():
    ev = [obs(f, 600 - 10 * f, 464, size=20, stage="tier_a_full_frame", conf=0.5,
              state="CONFIRMED" if f else "REACQUIRING", trusted=f > 0) for f in range(0, 12)]
    n, labels = touches(ev)
    check("9 strong multi-frame trajectory -> retained", n >= 3 and all(l == CREDIBLE for l in labels), f"possession frames={n}")


# extra: an all-credible clip is reproduced exactly by the gate replay
def t10():
    ev = [obs(f, 600 - 10 * f, 464, size=20, stage="tier_a_full_frame", conf=0.5,
              state="CONFIRMED" if f else "REACQUIRING", trusted=f > 0) for f in range(0, 12)]
    labels, _ = assess(ev)
    from football_analysis.possession_detector import PossessionDetector
    from football_analysis.interaction_detector import InteractionFrame
    det = PossessionDetector(); orig = []
    for e in ev:
        orig.append(det.step(InteractionFrame(e.frame, dist(FOOT, e.center), e.state, "TRACKING")))
    gated = gate_possession_frames(orig, labels)
    check("10 gate replay is the identity when every frame is credible", gated == orig)


# extra: real BallTracker populates accept_log and evidence_from_tracker reads it
def t11():
    import numpy as np
    from football_analysis.ball_tracker import BallTracker
    bt = BallTracker(light_model=None, heavy_model_factory=None, fps=25.0)
    frame = np.zeros((600, 900, 3), dtype=np.uint8); frame[:] = (40, 160, 40)
    for f in range(8):
        bt.step(f, frame, [((522, 452, 542, 472), 0.5)], player_bbox=PB)
    ev = evidence_from_tracker(bt)
    labels, verdicts = assess(ev)
    check("11 BallTracker.accept_log -> evidence -> tier-A ball at feet is CREDIBLE",
          len(bt.accept_log) == 8 and ev[0].stage == "tier_a_full_frame" and labels[-1] == CREDIBLE,
          f"labels={labels}")


for fn in (t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11):
    fn()
fails = [n for n, ok in R if not ok]
print(f"\n{len(R) - len(fails)}/{len(R)} physical-validity checks passed")
if fails:
    raise SystemExit(f"FAILED: {fails}")

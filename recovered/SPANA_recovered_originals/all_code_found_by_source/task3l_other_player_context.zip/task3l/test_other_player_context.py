"""
test_other_player_context.py -- TASK 3L regression tests for the other-player
context: TeamClassifier (possession_detector.py) partial-box gate + multi-wearer
opponent reference, and opponent_detection.py target-fragment exclusion.

All synthetic. No match-specific colour, coordinate or shirt number appears in
the implementation under test; the colour pairs here are arbitrary.

What the event layer consumes is (a) the plain other-player box list returned by
merge_other_player_candidates and (b) the TeamClassifier label of each box. A box
that is not in (a) can never be a teammate, opponent, pass receiver or duel
opponent; tests 5/8/11 assert exclusion at that point, tests 6/7/9/12 assert
retention plus the correct label.
"""
import sys
sys.path.insert(0, "/tmp/claude-0/-home-claude/98d69dca-3149-5a43-9670-3c417f2af6d4/scratchpad")
import numpy as np

from football_analysis.colour import shirt_color_hist
from football_analysis import possession_detector as pdm
from football_analysis.possession_detector import TeamClassifier, dominant_multi_wearer_cluster
from football_analysis.opponent_detection import (merge_other_player_candidates, is_target_fragment,
                                                   OtherPlayerTrackStore)

GRASS = (40, 160, 40)
TEAM_A = (190, 70, 30)      # target kit (BGR)
TEAM_B = (40, 40, 200)      # opposing kit
THIRD = (30, 220, 220)      # a singleton third colour (e.g. a referee)
results = {}


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}{': ' + detail if detail else ''}")
    results[name] = bool(cond)


def frame(w=640, h=360):
    f = np.zeros((h, w, 3), np.uint8)
    f[:, :] = GRASS
    return f


def person(f, box, bgr, noise=6, seed=0):
    x1, y1, x2, y2 = [int(v) for v in box]
    rng = np.random.RandomState(seed)
    p = np.full((y2 - y1, x2 - x1, 3), bgr, np.int32) + rng.randint(-noise, noise + 1, (y2 - y1, x2 - x1, 3))
    f[y1:y2, x1:x2] = np.clip(p, 0, 255).astype(np.uint8)


TARGET = (300, 120, 340, 220)


def warmed_classifier():
    """Classifier that has seen the target and, over a few frames, a two-player
    opposing team (so a genuine two-sided reference exists)."""
    f = frame()
    person(f, TARGET, TEAM_A)
    tc = TeamClassifier(shirt_color_hist(f, TARGET))
    for k in range(3):
        f = frame()
        person(f, TARGET, TEAM_A)
        person(f, (100, 120, 140, 220), TEAM_B, seed=k + 1)
        person(f, (480, 120, 520, 220), TEAM_B, seed=k + 7)
        tc.observe_tracked_player(f, TARGET)
        tc.classify(f, (100, 120, 140, 220), frame_number=k)
        tc.classify(f, (480, 120, 520, 220), frame_number=k)
    return tc


# 1. clear target-team player -> TEAMMATE
tc = warmed_classifier()
f = frame(); person(f, TARGET, TEAM_A); mate = (180, 120, 220, 220); person(f, mate, TEAM_A, seed=11)
tc.observe_tracked_player(f, TARGET)
lab, conf = tc.classify(f, mate, frame_number=10)
check("01_target_kit_player_is_TEAMMATE", lab == "TEAMMATE", f"{lab}/{conf:.2f}")

# 2. clear distinct-kit player -> OPPONENT (two-sided reference in use)
opp = (420, 120, 460, 220); person(f, opp, TEAM_B, seed=12)
lab, conf = tc.classify(f, opp, frame_number=10)
check("02_distinct_kit_player_is_OPPONENT", lab == "OPPONENT" and tc._team_b_centroid() is not None,
      f"{lab}/{conf:.2f}")

# 3. ambiguous crop (half-way colour) -> UNKNOWN
mid = tuple((a + b) // 2 for a, b in zip(TEAM_A, TEAM_B))
amb = (540, 120, 580, 220); person(f, amb, mid, seed=13)
lab, conf = tc.classify(f, amb, frame_number=10)
check("03_ambiguous_kit_is_UNKNOWN", lab == "UNKNOWN", f"{lab}/{conf:.2f}")

# 4. partial person truncated by the frame border -> UNKNOWN, never pooled
part = (600, 280, 640, 360); person(f, part, TEAM_B, seed=14)
n_before = len(tc._team_b_trusted)
lab, conf = tc.classify(f, part, frame_number=10)
check("04_frame_truncated_person_is_UNKNOWN_and_not_pooled",
      lab == "UNKNOWN" and len(tc._team_b_trusted) == n_before, f"{lab}/{conf:.2f}")

# 5. small target-body fragment inside the target box -> excluded from context
leg = (318, 182, 332, 220)          # 35% of target width, 38% of height, fully inside
boxes, recs = merge_other_player_candidates(None, 0, TARGET, [(leg, 0.4)], None)
check("05_target_fragment_excluded", leg not in boxes and any(r.classification == "TARGET_PLAYER_FRAGMENT" for r in recs))

# 6. independent opponent overlapping the target (duel) -> retained
duel_opp = (285, 118, 325, 222)     # full height, half outside the target
boxes, _ = merge_other_player_candidates(None, 0, TARGET, [(duel_opp, 0.7)], None)
f6 = frame(); person(f6, TARGET, TEAM_A); person(f6, (285, 118, 300, 222), TEAM_B, seed=15)
tc6 = warmed_classifier(); tc6.observe_tracked_player(f6, TARGET)
lab6, _ = tc6.classify(f6, duel_opp, frame_number=10)
check("06_overlapping_independent_opponent_retained", duel_opp in boxes, f"label={lab6}")

# 7. same-team independent player overlapping the target -> retained
mate_ov = (325, 116, 365, 224)
boxes, _ = merge_other_player_candidates(None, 0, TARGET, [(mate_ov, 0.7)], None)
check("07_overlapping_independent_teammate_retained", mate_ov in boxes)

# 8. a one-frame fragment cannot become a pass receiver: it never reaches the
#    other-player list, not even through the temporal bridge on later frames
store = OtherPlayerTrackStore()
b0, _ = merge_other_player_candidates(None, 0, TARGET, [(leg, 0.4)], None, track_store=store)
b1, _ = merge_other_player_candidates(None, 1, TARGET, [], None, track_store=store)
check("08_fragment_never_reaches_receiver_context", leg not in b0 and leg not in b1)

# 9. a persistent real teammate near the ball is retained and labelled TEAMMATE
store = OtherPlayerTrackStore()
recv = (380, 122, 420, 222)
ok = True
tc9 = warmed_classifier()
for k in range(5):
    fk = frame(); person(fk, TARGET, TEAM_A); person(fk, recv, TEAM_A, seed=20 + k)
    tc9.observe_tracked_player(fk, TARGET)
    bk, _ = merge_other_player_candidates(fk, k, TARGET, [(recv, 0.8)], None, track_store=store)
    lab9, _ = tc9.classify(fk, recv, frame_number=20 + k)
    ok = ok and recv in bk and lab9 == "TEAMMATE"
check("09_persistent_real_teammate_can_be_receiver", ok)

# 10. UNKNOWN remains possible: with no multi-wearer opponent reference, a
#     single unlike-coloured person does not create a two-sided reference
f10 = frame(); person(f10, TARGET, TEAM_A)
tc10 = TeamClassifier(shirt_color_hist(f10, TARGET))
for k in range(4):
    fk = frame(); person(fk, TARGET, TEAM_A); person(fk, (100, 120, 140, 220), THIRD, seed=30 + k)
    tc10.observe_tracked_player(fk, TARGET)
    tc10.classify(fk, (100, 120, 140, 220), frame_number=k)
lab10, _ = tc10.classify(f, amb, frame_number=9)
check("10_UNKNOWN_still_possible_and_singleton_is_not_a_team",
      tc10._team_b_centroid() is None and lab10 == "UNKNOWN", f"ambiguous={lab10}")

# 11. a target fragment does not enter the DUEL opponent context
torso = (304, 122, 336, 190)        # upper-body crop of the target
boxes, _ = merge_other_player_candidates(None, 0, TARGET, [(torso, 0.5), (duel_opp, 0.7)], None)
check("11_fragment_not_in_duel_context", torso not in boxes)

# 12. the real overlapping opponent is in the DUEL opponent context, as OPPONENT
check("12_real_opponent_in_duel_context", duel_opp in boxes and lab6 == "OPPONENT", f"label={lab6}")

# extra: cluster reference ignores a singleton third colour seen on many frames
samples = [((200.0, 120.0, 120.0), k) for k in range(10)]            # one referee, 10 frames
samples += [((60.0, 140.0, 110.0), 3), ((62.0, 139.0, 112.0), 3)]    # two opponents in one frame
c = dominant_multi_wearer_cluster(samples)
check("13_reference_is_multi_wearer_kit_not_most_frequent_singleton", c is not None and abs(c[0] - 61.0) < 2)

# extra: the flag reproduces the old behaviour (diagnostic replays depend on it)
pdm.TASK3L_OTHER_PLAYER_CONTEXT = False
tcx = warmed_classifier()
labx, _ = tcx.classify(f, part, frame_number=10)
pdm.TASK3L_OTHER_PLAYER_CONTEXT = True
check("14_flag_off_restores_previous_behaviour", labx == "OPPONENT", f"truncated box with flag off -> {labx}")

n = sum(results.values())
print(f"\n{n}/{len(results)} other-player-context checks passed")
sys.exit(0 if n == len(results) else 1)

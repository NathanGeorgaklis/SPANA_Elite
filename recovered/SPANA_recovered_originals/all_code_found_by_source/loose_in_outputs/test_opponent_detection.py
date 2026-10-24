"""
test_opponent_detection.py -- OPPONENT-DETECTION TASK, rule #16: required
general synthetic tests (A-H), driving the real merge_other_player_
candidates() / OtherPlayerTrackStore / classify_vs_target logic against
fully synthetic boxes, so each named scenario from the task spec can be
checked directly and fast, without needing a real video or a real YOLO
pass.

A fake local-ROI detector (_StubLocalDetector) stands in for
LocalChallengeDetector wherever a scenario needs to control exactly what
the local high-res pass "sees" -- the ONLY piece of opponent_detection.py
that needs a real model/frame is LocalChallengeDetector.detect() itself
(a thin crop+predict wrapper, already exercised directly against real
footage in diagnostics/inspect_missing_defender.py and the frame-14-20
real-clip smoke test run for this task -- see
validation/reports/opponent_detection_report.md). Everything tested here
is the actual, unmodified merge/classification/temporal-continuity logic
that engine.py calls every frame.

None of these scenarios depend on any clip-specific location or threshold
(rule #9-equivalent generality requirement, mirrored from the static-decoy
task's own test suite) -- every synthetic box/geometry is chosen fresh per
scenario and is unrelated to the real validated clips.
"""
import sys
sys.path.insert(0, "/tmp/claude-0/-home-claude/98d69dca-3149-5a43-9670-3c417f2af6d4/scratchpad")

from football_analysis.opponent_detection import (
    classify_vs_target, OtherPlayerTrackStore, merge_other_player_candidates,
    should_run_local_pass,
)

FRAME = None   # never actually read by anything in these tests -- classify_vs_target/
                # OtherPlayerTrackStore/merge_other_player_candidates(local_detector=None or
                # a _StubLocalDetector) never touch pixel data themselves.

TRACKED = (500.0, 200.0, 540.0, 300.0)   # a stand-in tracked-player box, reused across scenarios


class _StubLocalDetector:
    """Fake LocalChallengeDetector: returns a fixed, scenario-authored list
    of (box, confidence) pairs regardless of the frame/tracked_bbox it's
    called with, so each scenario can assert exactly what the local pass
    "found" without depending on a real model. Records whether it was
    actually invoked, so scenario D can assert the proximity trigger
    correctly PREVENTED a call that would otherwise have hallucinated a box."""

    def __init__(self, boxes):
        self._boxes = boxes
        self.calls = 0

    def detect(self, frame, tracked_bbox):
        self.calls += 1
        return list(self._boxes)


def _run(scenario, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {scenario}{': ' + detail if detail else ''}")
    return cond


results = {}

# --- A: two players apart -> two detections -------------------------------
# full-frame pass already found the target (duplicate) plus one clearly
# separate other player; no local pass needed/triggered (no ball, no crowding).
full_frame = [(TRACKED, 0.9), ((50.0, 50.0, 80.0, 120.0), 0.7)]
plain, records = merge_other_player_candidates(
    FRAME, 0, TRACKED, full_frame, ball_centre=None,
    local_detector=None, track_store=None, team_classifier=None)
ok = (len(plain) == 1 and plain[0] == (50.0, 50.0, 80.0, 120.0)
      and sum(1 for r in records if r.classification == "PERSON_DETECTED") == 1
      and sum(1 for r in records if r.classification == "TARGET_PLAYER_DUPLICATE") == 1)
results["A"] = _run("A", ok, f"plain_boxes={plain}")

# --- B: two players crossing paths -> both remain detectable --------------
# two DISTINCT other-player boxes (simulating the base pass already telling
# them apart, e.g. mid-crossing before/after full overlap) must both survive
# untouched -- this module must never merge two non-target boxes together.
p1, p2 = (100.0, 100.0, 130.0, 170.0), (135.0, 102.0, 165.0, 172.0)   # adjacent, not identical
full_frame = [(TRACKED, 0.9), (p1, 0.6), (p2, 0.55)]
plain, records = merge_other_player_candidates(
    FRAME, 0, TRACKED, full_frame, ball_centre=None,
    local_detector=None, track_store=None, team_classifier=None)
ok = len(plain) == 2 and p1 in plain and p2 in plain
results["B"] = _run("B", ok, f"plain_boxes={plain}")

# --- C: heavy shoulder-to-shoulder overlap -> local pass preserves two ----
# hypotheses: the full-frame pass only produced ONE box for the overlapping
# pair (simulating full-frame NMS merging them, since neither is the
# target); the local challenge pass recovers a second, distinct-enough box.
merged_pair_box = (300.0, 150.0, 340.0, 230.0)
local_second_person = (320.0, 152.0, 362.0, 232.0)   # overlaps merged_pair_box some, but
                                                       # IoU < MERGE_DEDUPE_IOU -- a genuinely
                                                       # separate body, not the same detection
from football_analysis.geometry import iou as _iou
assert _iou(merged_pair_box, local_second_person) < 0.55, "scenario C fixture must be below MERGE_DEDUPE_IOU"
full_frame = [(TRACKED, 0.9), (merged_pair_box, 0.65)]
stub = _StubLocalDetector([(local_second_person, 0.3)])
plain, records = merge_other_player_candidates(
    FRAME, 0, TRACKED, full_frame, ball_centre=TRACKED[:2],  # ball "at" the tracked player -> triggers local pass
    local_detector=stub, track_store=None, team_classifier=None, force_local_pass=True)
ok = (stub.calls == 1 and len(plain) == 2 and merged_pair_box in plain and local_second_person in plain)
results["C"] = _run("C", ok, f"plain_boxes={plain}")

# --- D: target player alone -> no phantom opponent -------------------------
# no ball nearby, no crowding -> should_run_local_pass is False, so even a
# local detector that WOULD hallucinate a box must never be invoked.
full_frame = [(TRACKED, 0.9)]   # only the tracked player detected at all
stub = _StubLocalDetector([((10.0, 10.0, 40.0, 80.0), 0.4)])   # a would-be phantom, far away
plain, records = merge_other_player_candidates(
    FRAME, 0, TRACKED, full_frame, ball_centre=None,
    local_detector=stub, track_store=None, team_classifier=None)
ok = (stub.calls == 0 and len(plain) == 0
      and all(r.classification == "TARGET_PLAYER_DUPLICATE" for r in records))
results["D"] = _run("D", ok, f"local_calls={stub.calls} plain_boxes={plain}")

# --- E: far-field small defender -> local high-res pass improves recall ---
# full-frame pass misses the small far player entirely; a full-frame OTHER
# box sits close enough to the tracked player to count as crowding (rule
# #12), triggering the local pass, which finds the missed player.
nearby_other = (560.0, 210.0, 585.0, 270.0)   # close enough to tracked centre to trigger crowding
far_field_defender = (610.0, 205.0, 622.0, 235.0)   # small, far -- only the local pass finds this
full_frame = [(TRACKED, 0.9), (nearby_other, 0.5)]
stub = _StubLocalDetector([(far_field_defender, 0.12)])   # low local conf, as rule #5 expects
plain, records = merge_other_player_candidates(
    FRAME, 0, TRACKED, full_frame, ball_centre=None,
    local_detector=stub, track_store=None, team_classifier=None)
ok = stub.calls == 1 and far_field_defender in plain
local_records = [r for r in records if r.source == "LOCAL_ROI"]
results["E"] = _run("E", ok and len(local_records) == 1, f"local_calls={stub.calls} plain_boxes={plain}")

# --- F: teammate close to target -> detected, team classification used ----
def fake_team_classifier(frame, box):
    return ("TEAMMATE", 0.8) if box[0] < 200 else ("OPPONENT", 0.7)

teammate_box = (100.0, 100.0, 130.0, 170.0)
full_frame = [(TRACKED, 0.9), (teammate_box, 0.6)]
plain, records = merge_other_player_candidates(
    FRAME, 0, TRACKED, full_frame, ball_centre=None,
    local_detector=None, track_store=None, team_classifier=fake_team_classifier)
person_records = [r for r in records if r.classification == "PERSON_DETECTED"]
ok = (len(person_records) == 1 and person_records[0].team == "TEAMMATE"
      and person_records[0].team_confidence == 0.8)
results["F"] = _run("F", ok, f"records={[(r.box, r.team, r.team_confidence) for r in person_records]}")

# rule #9 corollary: an UNCERTAIN team result must still keep the detection
# (PERSON_DETECTED, team=UNKNOWN), never discard it.
def uncertain_team_classifier(frame, box):
    return ("UNKNOWN", 0.2)

plain2, records2 = merge_other_player_candidates(
    FRAME, 0, TRACKED, full_frame, ball_centre=None,
    local_detector=None, track_store=None, team_classifier=uncertain_team_classifier)
ok2 = len(plain2) == 1 and any(r.team == "UNKNOWN" for r in records2 if r.classification == "PERSON_DETECTED")
results["F (uncertain team kept)"] = _run("F (uncertain team kept)", ok2, f"plain_boxes={plain2}")

# --- G: local detector sees target twice -> duplicate removed, target -----
# tracker untouched: the local pass re-detects the tracked player itself
# (heavily overlapping TRACKED) as well as one genuine other player.
target_redetect = (502.0, 202.0, 538.0, 298.0)   # near-identical to TRACKED -> IoU >= 0.5
assert _iou(target_redetect, TRACKED) >= 0.5, "scenario G fixture must exceed TARGET_DUPLICATE_IOU"
genuine_other = (700.0, 400.0, 730.0, 470.0)
full_frame = [(TRACKED, 0.9)]
stub = _StubLocalDetector([(target_redetect, 0.5), (genuine_other, 0.3)])
tracked_before = tuple(TRACKED)
plain, records = merge_other_player_candidates(
    FRAME, 0, TRACKED, full_frame, ball_centre=None,
    local_detector=stub, track_store=None, team_classifier=None, force_local_pass=True)
ok = (target_redetect not in plain and genuine_other in plain
      and TRACKED == tracked_before   # the tracked-bbox VALUE passed in is never mutated
      and any(r.classification == "TARGET_PLAYER_DUPLICATE" and r.source == "LOCAL_ROI" for r in records))
results["G"] = _run("G", ok, f"plain_boxes={plain}")

# --- H: occluded defender disappears completely -> uncertain/lost, --------
# never hallucinated indefinitely: a track is bridged with DECAYING
# confidence for a few frames, then dropped entirely -- it must never come
# back as a fresh, full-confidence phantom detection once truly gone.
store = OtherPlayerTrackStore()
defender_box = (250.0, 250.0, 280.0, 320.0)
frame0 = store.step(0, [(defender_box, 0.6, "FULL_FRAME")])
assert len(frame0) == 1 and frame0[0][2] == 1.0   # fresh detection: full temporal confidence
bridge1 = store.step(1, [])   # defender occluded -- no detection this frame
bridge2 = store.step(2, [])
bridge3 = store.step(3, [])
gone = store.step(4, [])      # past MAX_BRIDGE_FRAMES=3 misses -> must be fully dropped
decaying = [b[2] for b in (bridge1[0] if bridge1 else (None, None, 0),
                            bridge2[0] if bridge2 else (None, None, 0),
                            bridge3[0] if bridge3 else (None, None, 0))]
ok = (len(bridge1) == 1 and len(bridge2) == 1 and len(bridge3) == 1
      and decaying[0] > decaying[1] > decaying[2] > 0.0   # strictly decaying, never re-inflated
      and len(gone) == 0)                                  # genuinely gone -> empty, not hallucinated
results["H"] = _run("H", ok, f"temporal_confidences={decaying} final={gone}")

print()
print("=== summary ===")
for k, v in results.items():
    print(f"  {k}: {'PASS' if v else 'FAIL'}")
n_pass = sum(1 for v in results.values() if v)
print()
if n_pass == len(results):
    print(f"ALL {len(results)} SCENARIOS PASSED")
else:
    print(f"{n_pass}/{len(results)} PASSED -- SEE FAILURES ABOVE")
    sys.exit(1)

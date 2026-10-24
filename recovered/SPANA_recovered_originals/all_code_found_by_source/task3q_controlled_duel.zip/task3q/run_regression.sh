# Task 3Q regression run: every existing suite (same set as Task 3P), the Task 3P receiver suite, and the new
# controlled-duel suite. Exit code and the suite's own summary line are recorded for each.
cd /tmp/claude-0/-home-claude/98d69dca-3149-5a43-9670-3c417f2af6d4/scratchpad
for t in test_other_player_context.py test_team_classification.py test_opponent_detection.py test_receiver_reception.py \
         test_pass_evidence_fix.py test_event_arbitration.py test_possession_segmentation.py test_actor_attribution.py \
         test_ball_physical_validity.py test_ball_static_decoy.py test_ball_robustness.py \
         event_engine_v1/test_temporal_coverage_guard.py event_engine_v1/test_benchmark_fixture_guard.py \
         test_shot_scenarios.py test_receiver_eligibility.py test_controlled_duel.py; do
  echo "=== $t"
  PYTHONPATH=. python3 $t > _reg.log 2>&1; rc=$?
  echo "exit=$rc  pass_lines=$(grep -cE '^\s*(\[PASS\]|PASS:)' _reg.log)  fail_lines=$(grep -cE '^\s*(\[FAIL\]|FAIL:)' _reg.log)"
  grep -E "TOTAL|passed|PASSED|checks passed|scenarios run" _reg.log | tail -2
done
rm -f _reg.log

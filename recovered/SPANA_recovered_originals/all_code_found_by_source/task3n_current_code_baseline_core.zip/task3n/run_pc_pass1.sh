cd /tmp/claude-0/-home-claude/98d69dca-3149-5a43-9670-3c417f2af6d4/scratchpad
T=event_engine_v1/task3n
echo "pc attempt2 start $(date -u +%FT%TZ)"
PYTHONPATH=. python3 $T/run_fresh_pass1.py event_engine_v1/task3g/fixtures/PASS_COMPLETED_01_BENCHMARK_FIXTURE.json $T/pass_completed_01/pass1 > $T/pass_completed_01/pass1/pass1.log 2>&1
echo "pc attempt2 done $(date -u +%FT%TZ) rc=$?"

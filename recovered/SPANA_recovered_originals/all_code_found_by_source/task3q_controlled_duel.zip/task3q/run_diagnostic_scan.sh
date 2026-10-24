cd /tmp/claude-0/-home-claude/98d69dca-3149-5a43-9670-3c417f2af6d4/scratchpad
D=event_engine_v1/task3q/diagnostic_scan
mkdir -p $D/pass_completed_01 $D/blind_r2_006 $D/blind_r2_001
PYTHONPATH=. python3 event_engine_v1/task3q/diag_pathb_all_releases.py pass_completed_01 event_engine_v1/task3n/pass_completed_01/pass1/f5bd8ce0-pass_completed_01_mo_tracking_results.json /root/.claude/uploads/98d69dca-3149-5a43-9670-3c417f2af6d4/f5bd8ce0-pass_completed_01.mov $D/pass_completed_01 "target player" > $D/pass_completed_01/run.log 2>&1
echo pc $? $(date -u +%T)
for c in 006 001; do
PYTHONPATH=. python3 event_engine_v1/task3q/diag_pathb_all_releases.py blind_r2_$c spana_dataset/dataset/track_continuity_no_b/blind_r2_$c/clip_mo_tracking_results.json spana_dataset/dataset/blind_package_v2_round2/blind_r2_$c/clip.mp4 $D/blind_r2_$c "target player" > $D/blind_r2_$c/run.log 2>&1
echo $c $? $(date -u +%T)
done

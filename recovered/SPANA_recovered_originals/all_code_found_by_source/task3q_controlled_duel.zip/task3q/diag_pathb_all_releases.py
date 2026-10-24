"""Task 3Q diagnostic (NO source file edited): run the unchanged Task 3N Pass-2 driver, but record what
DUEL path B (controlled_duel.evaluate) says at EVERY release -- including releases the confirmed-reception
short-circuit settles before path B is normally evaluated. Runtime wrapper only; nothing it computes is fed
back into the pipeline, so outputs must equal the production run's.
usage: diag_pathb_all_releases.py CLIP_ID MO_JSON VIDEO OUT_DIR LABEL"""
import json, sys, runpy
import football_analysis.event_arbitration as arb
import football_analysis.event_qualifier as evq
from football_analysis import controlled_duel as cd

REC = []
_orig = arb.resolve_release


def wrapped(ball_tracker, possession_frames, i_end, *a, controlled_duel_context=None, seg_start=None, **kw):
    out = _orig(ball_tracker, possession_frames, i_end, *a, controlled_duel_context=controlled_duel_context,
                seg_start=seg_start, **kw)
    rec = {"i_end": i_end, "seg_start": seg_start, "returned_type": out.type, "returned_success": out.success,
           "short_circuited": bool(out.evidence.get("arbitration", {}).get("short_circuited"))}
    if controlled_duel_context is not None:
        a_ = cd.evaluate(controlled_duel_context, possession_frames, i_end, seg_start)
        rec.update({"pathB_valid": a_.valid, "pathB_failed_gates": a_.failed_gates, "pathB_outcome": a_.outcome,
                    "pathB_window": a_.window(), "pathB_score": round(a_.score, 4),
                    "pathB_measurements": a_.evidence.get("measurements")})
    REC.append(rec)
    return out


evq.resolve_release = wrapped          # the name qualify_release actually calls
out_dir = sys.argv[4]
sys.argv = ["run_fresh_pass2.py"] + sys.argv[1:]
try:
    runpy.run_path("event_engine_v1/task3n/run_fresh_pass2.py", run_name="__main__")
finally:
    json.dump(REC, open(out_dir + "/pathB_all_releases.json", "w"), indent=1, default=str)

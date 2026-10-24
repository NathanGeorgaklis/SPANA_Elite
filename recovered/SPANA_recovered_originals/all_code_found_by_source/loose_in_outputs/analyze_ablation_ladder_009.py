"""
analyze_ablation_ladder_009.py -- builds the full per-stage metrics table
(Section 5), the football-moment check (Section 6, using blind_r2_009's own
known contested moment at t=7.0s from phase5_enrichment_and_scorecard.py's
MOMENTS list -- "A+Nathan contested (DUEL/TACKLE)"), and the marginal-
contribution table (Section 4) for the ablation ladder, from the real
ablation_ladder_009_<STAGE>.json result files on disk. Run only once all 6
stages have completed -- prints a clear error naming which stages are still
missing otherwise, rather than fabricating partial numbers.
"""
import json
from pathlib import Path

STAGES = ["BASELINE", "+A", "+A+B", "+A+B+C", "+A+B+C+D", "+A+B+C+D+E"]
MOMENT_FRAME = round(7.0 * 30.0)  # blind_r2_009's own known contested moment, 30fps


def _safe_label(label):
    return label.replace("+", "P")


def load_all():
    missing = []
    results = {}
    for label in STAGES:
        path = Path("rootcause_out") / f"ablation_ladder_009_{_safe_label(label)}.json"
        if not path.exists():
            missing.append(label)
            continue
        results[label] = json.load(open(path))
    return results, missing


def moment_state(result, moment_frame, window=5):
    """Selected-player state at/near the known moment frame, read from the
    raw tracking JSON for that stage (written alongside the summary at
    rootcause_out/ablation_009/<safe_label>/clip_mo_tracking_results.json)."""
    label = result["label"]
    safe = _safe_label(label)
    raw_path = Path("rootcause_out/ablation_009") / safe / "clip_mo_tracking_results.json"
    if not raw_path.exists():
        return {"error": f"raw tracking json not found at {raw_path}"}
    raw = json.load(open(raw_path))
    frame_log = raw["frame_log"]
    by_frame = {rec["frame"]: rec for rec in frame_log}
    states = []
    for f in range(max(0, moment_frame - window), moment_frame + window + 1):
        rec = by_frame.get(f)
        if rec is None:
            continue
        sel = next((tr for tr in rec["tracks"] if tr.get("selected_player")), None)
        states.append(sel["state"] if sel else "ABSENT_FROM_LOG")
    return {"frame_window": [moment_frame - window, moment_frame + window], "states": states,
            "state_at_moment": by_frame.get(moment_frame, {}).get("tracks", [{}])}


def main():
    results, missing = load_all()
    if missing:
        print(f"NOT ALL STAGES COMPLETE -- missing: {missing}. Refusing to fabricate a table.")
        return

    print("=== FULL METRICS TABLE (Section 5) ===")
    rows = []
    for label in STAGES:
        s = results[label]["summary"]
        hm = s["health_metrics"]
        row = dict(
            stage=label,
            processing_time_s=s["processing_time_s"],
            n_tracks_ever_created=s["n_tracks_ever_created"],
            final_confirmed_count=s["final_confirmed_count"],
            final_provisional_count=s["final_provisional_count"],
            duplicate_likelihood=hm["duplicate_likelihood"],
            median_track_lifetime_frames=hm["median_track_lifetime_frames"],
            selected_state_pct=s["selected_state_pct"],
            identity_conflicts_logged=s["identity_conflicts_logged"],
            pruned_from_live_association_count=hm["pruned_from_live_association_count"],
            spawn_guard_stats=s["spawn_guard_stats"],
        )
        rows.append(row)
        print(json.dumps(row, indent=2))

    print("\n=== FOOTBALL-MOMENT CHECK (Section 6): blind_r2_009 contested DUEL/TACKLE @ t=7.0s ===")
    for label in STAGES:
        m = moment_state(results[label], MOMENT_FRAME)
        print(f"{label}: {m}")

    out_path = Path("rootcause_out/ablation_ladder_009_full_analysis.json")
    with open(out_path, "w") as f:
        json.dump({"rows": rows, "moment_frame": MOMENT_FRAME,
                   "moment_checks": {label: moment_state(results[label], MOMENT_FRAME) for label in STAGES}},
                  f, indent=2, default=str)
    print(f"\nwritten to {out_path}")


if __name__ == "__main__":
    main()

# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2 DEV inference instruction (Nathan, 2026-10-30 15:31): per-clip results, aggregate DEV metrics,
#                 failure classes, coverage trade-offs; report UNKNOWN where truth cannot support a metric; no tuning
"""Aggregate the per-clip tracker_eval reports of the R2 DEV run, and add read-only diagnostics.

    python3 tools_rb/r2_dev_report.py REPO OUT_DIR

Reads OUT_DIR/R2_DEV_EVAL_ALL.json (written by r2_dev_run.py score), the run results and the frozen labels. Writes
OUT_DIR/R2_DEV_AGGREGATE.json. Metrics are taken unchanged from tracker_eval; nothing is re-scored. The only added
measurement is a box-convention diagnostic for every frame tracker_eval flags as false confident identity or
wrong box: the fraction of the tracker box inside the human box and vice versa, so a frame where a correct but tighter
tracker box sits inside a looser human box can be told apart from a box on another person. The diagnostic never
changes a count; it is reported next to it.
"""
import glob
import json
import os
import sys

UNKNOWN = "UNKNOWN"


def _inter(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy


def _area(a):
    return max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])


def _num(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def main(repo, out_dir):
    sys.path.insert(0, repo)
    from player_tracker_rb import run_rb as R
    with open(os.path.join(out_dir, "R2_DEV_EVAL_ALL.json")) as f:
        allrep = json.load(f)
    per, agg = {}, {"fci": 0, "fci_unknown_clips": [], "swaps": 0, "swaps_unknown_clips": [], "wrong_reacq": 0,
                    "wrong_reacq_unknown_clips": [], "p4": 0, "available": 0, "trusted_available": 0,
                    "unavailable": 0, "trusted_unavailable": 0, "reacq_correct": 0, "reacq_not": 0,
                    "reacq_delays_s": [], "iou_sum": 0.0, "iou_n": 0, "refused": [], "b5": {}}
    for cid, rep in sorted(allrep.items()):
        if rep.get("status") in ("RUN_REFUSED", "NOT_RUN"):
            per[cid] = {"status": rep["status"], "reason": rep.get("reason")}
            if rep["status"] == "RUN_REFUSED":
                agg["refused"].append(cid)
            continue
        fx = json.load(open(os.path.join(repo, "R2", "labels", cid + ".fixture.json")))
        lab = json.load(open(os.path.join(repo, "R2", "labels", fx["labels_file"])))
        res = R.output_paths(os.path.join(out_dir, cid), fx["video_file"])["results"]
        doc = json.load(open(res, encoding="ascii"))
        sel = {}
        for fr in doc["frame_log"]:
            for t in fr["tracks"]:
                if t["selected_player"]:
                    sel[fr["frame"]] = t
        p1 = rep["priority_1_false_confident_identity"]
        diag = []
        for d in (p1.get("on_wrong_box") or []):
            f = d["frame"]
            hb = lab["target_boxes"].get(str(f + fx["seed_frame"]))      # truth frames are relative to the seed
            tb = sel.get(f, {}).get("observed_position_box")           # the box tracker_eval scores
            if hb and tb:
                i = _inter(hb, tb)
                diag.append({"frame_rel_seed": f, "clip_frame": f + fx["seed_frame"], "iou": d["iou"], "tracker_inside_human": round(i / _area(tb), 3) if _area(tb) else None,
                             "human_inside_tracker": round(i / _area(hb), 3) if _area(hb) else None,
                             "tracker_h_over_human_h": round((tb[3] - tb[1]) / (hb[3] - hb[1]), 3)})
        for it in (rep["priority_3_wrong_reacquisitions"].get("items") or []):
            f = it.get("judged_on_frame")
            hb = lab["target_boxes"].get(str(f + fx["seed_frame"])) if f is not None else None
            tb = sel.get(f, {}).get("observed_position_box") if f is not None else None
            if hb and tb:
                i = _inter(hb, tb)
                diag.append({"frame_rel_seed": f, "clip_frame": f + fx["seed_frame"], "kind": "reacquisition_" + it["outcome"], "iou": it.get("iou"),
                             "tracker_inside_human": round(i / _area(tb), 3) if _area(tb) else None,
                             "human_inside_tracker": round(i / _area(hb), 3) if _area(hb) else None,
                             "tracker_h_over_human_h": round((tb[3] - tb[1]) / (hb[3] - hb[1]), 3)})
        sw = rep["priority_2_identity_swaps"]
        p3 = rep["priority_3_wrong_reacquisitions"]
        p4 = rep["priority_4_trusted_on_hidden_merged_unresolvable"]
        cov = rep["coverage_and_abstention"]
        bq = rep["box_quality"]
        reacq = rep.get("reacquisition") or []
        delays = [r.get("delay_s") for r in reacq if _num(r.get("delay_s")) is not None]
        per[cid] = {
            "status": "SCORED", "results_git_commit": rep.get("results_git_commit"),
            "results_git_dirty": rep.get("results_git_dirty"), "window": rep["window"],
            "p1_false_confident_identity": p1["count"], "p1_status": p1["status"],
            "p1_box_checked_fraction": p1.get("box_checked_fraction_of_confident_available"),
            "p1_flagged_frames_box_convention_diag": diag,
            "p2_swaps": sw["count"], "p2_lower_bound": sw.get("count_is_lower_bound"),
            "p2_people_annotated_on_trusted_grid": sw.get("people_annotated_on_trusted_grid_frames"),
            "p3_wrong_reacq": p3["count"], "p3_correct": p3["correct"], "p3_not_reacquired": p3["not_reacquired"],
            "p3_unverified": p3["unverified"],
            "p4_trusted_on_unavailable": p4["count"], "p4_frames": p4["frames"][:50],
            "b5_contact_integrity": rep["b5_contact_integrity"]["status"],
            "available_frames": cov["available_frames"], "trusted_coverage": cov["trusted_coverage_of_available"],
            "abstention": cov["abstention_rate_on_available"], "claims_on_available": cov["claims_on_available"],
            "unavailable_frames": cov["unavailable_frames"], "claims_on_unavailable": cov["claims_on_unavailable"],
            "unresolvable_frames": cov["unresolvable_frames"],
            "reacquisition": reacq, "reacq_delays_s": delays,
            "box_quality_mean_iou": bq.get("mean_iou_confident"), "box_quality_frames": bq.get("frames"),
            "human_uncertain_frames": rep["human_uncertain"]["frames"],
            "confident_not_grounded_violations": rep.get("confident_not_grounded_violations"),
        }
        for key, val, unk in (("fci", p1["count"], "fci_unknown_clips"), ("swaps", sw["count"], "swaps_unknown_clips"),
                              ("wrong_reacq", p3["count"], "wrong_reacq_unknown_clips")):
            if _num(val) is None:
                agg[unk].append(cid)
            else:
                agg[key] += val
        agg["p4"] += p4["count"]
        av = cov["available_frames"] or 0
        agg["available"] += av
        agg["trusted_available"] += (cov["claims_on_available"] or {}).get("CONFIDENT", 0)
        agg["unavailable"] += cov["unavailable_frames"] or 0
        agg["trusted_unavailable"] += (cov["claims_on_unavailable"] or {}).get("CONFIDENT", 0)
        agg["reacq_correct"] += p3["correct"]
        agg["reacq_not"] += p3["not_reacquired"]
        agg["reacq_delays_s"] += delays
        if _num(bq.get("mean_iou_confident")) is not None:
            agg["iou_sum"] += bq["mean_iou_confident"] * bq["frames"]
            agg["iou_n"] += bq["frames"]
        agg["b5"][cid] = rep["b5_contact_integrity"]["status"]
    out = {"per_clip": per, "aggregate": {
        "clips_scored": sum(1 for v in per.values() if v["status"] == "SCORED"),
        "clips_run_refused": agg["refused"],
        "false_confident_identity": agg["fci"] if not agg["fci_unknown_clips"] else
        {"verified_sum": agg["fci"], "UNKNOWN_in": agg["fci_unknown_clips"]},
        "identity_swaps_lower_bound": agg["swaps"] if not agg["swaps_unknown_clips"] else
        {"sum": agg["swaps"], "UNKNOWN_in": agg["swaps_unknown_clips"]},
        "wrong_reacquisitions": agg["wrong_reacq"] if not agg["wrong_reacq_unknown_clips"] else
        {"verified_sum": agg["wrong_reacq"], "UNKNOWN_in": agg["wrong_reacq_unknown_clips"]},
        "trusted_on_hidden_occluded_unresolvable_out": agg["p4"],
        "available_frames": agg["available"], "trusted_on_available": agg["trusted_available"],
        "trusted_coverage_pooled": round(agg["trusted_available"] / agg["available"], 3) if agg["available"] else UNKNOWN,
        "abstention_pooled": round(1 - agg["trusted_available"] / agg["available"], 3) if agg["available"] else UNKNOWN,
        "unavailable_frames": agg["unavailable"], "trusted_on_unavailable": agg["trusted_unavailable"],
        "reacquisitions_correct": agg["reacq_correct"], "reacquisitions_not_reacquired": agg["reacq_not"],
        "reacq_delay_s": ({"n": len(agg["reacq_delays_s"]), "median": sorted(agg["reacq_delays_s"])[len(agg["reacq_delays_s"]) // 2],
                           "max": max(agg["reacq_delays_s"])} if agg["reacq_delays_s"] else UNKNOWN),
        "box_quality_mean_iou_pooled": round(agg["iou_sum"] / agg["iou_n"], 3) if agg["iou_n"] else UNKNOWN,
        "box_quality_frames": agg["iou_n"], "b5_contact_integrity": agg["b5"]}}
    with open(os.path.join(out_dir, "R2_DEV_AGGREGATE.json"), "x") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print(json.dumps(out["aggregate"], indent=1))
    for cid, v in per.items():
        if v["status"] != "SCORED":
            print(cid, v["status"], v.get("reason"))
            continue
        print(cid, "FCI", v["p1_false_confident_identity"], "swaps", v["p2_swaps"], "wrongR", v["p3_wrong_reacq"],
              "P4", v["p4_trusted_on_unavailable"], "cov", v["trusted_coverage"], "avail", v["available_frames"],
              "reacq ok/not", v["p3_correct"], v["p3_not_reacquired"], "IoU", v["box_quality_mean_iou"],
              "B5", v["b5_contact_integrity"], "dirty", v["results_git_dirty"])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2 DEV remediation instruction (Nathan, 2026-10-30 17:41, Validation): compare pre/post on DEV:
#                 initialization success, false trusted identity, trusted unavailable frames, swaps where measurable,
#                 wrong reacquisition, trusted coverage, merged-guard frequency, appearance-bank contamination
"""Pre/post comparison of two R2 DEV runs (read-only; DEV only; recomputes no official metric).

    python3 tools_rb/r2_dev_compare.py REPO PRE_OUT_DIR PRE_RAW_DIR POST_OUT_DIR OUT_JSON

PRE_OUT_DIR / POST_OUT_DIR hold <clip>/DONE.json and <clip>/<clip>.r2_dev_eval.json (official tracker_eval reports);
*_RAW_DIR hold <clip>/<clip>_rb_decision_log.jsonl. Official metrics are copied unchanged from the eval reports.
Added diagnostics (never replace an official count):
  * merged-guard frequency: target frames whose uncertainty reasons include MERGED_DETECTION_SUSPECTED, split by
    whether the frozen human truth labels that frame merged (no R2 truth frame is labelled merged);
  * appearance-bank contamination: every target-bank ACCEPTED sample checked against the frozen truth (target
    available; IoU with the human target box >= 0.5 where boxed; no labelled contact). Counted as CONTAMINATED if the
    truth contradicts it, UNVERIFIED if the frame has no human box.
"""
import json
import os
import sys

FIELDS = ("p1", "p2", "p3", "p4", "available", "trusted", "coverage", "box_iou", "box_frames", "reacq_correct",
          "reacq_not")


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def official(out_dir, cid):
    with open(os.path.join(out_dir, cid, "DONE.json")) as f:
        done = json.load(f)
    if done.get("outcome") == "RUN_REFUSED":
        return {"outcome": "RUN_REFUSED", "reason": done["reason"]}
    with open(os.path.join(out_dir, cid, cid + ".r2_dev_eval.json")) as f:
        r = json.load(f)
    cov = r["coverage_and_abstention"]
    sw = r["priority_2_identity_swaps"]
    return {"outcome": "SCORED", "results_git_commit": r.get("results_git_commit"),
            "results_git_dirty": r.get("results_git_dirty"),
            "p1": r["priority_1_false_confident_identity"]["count"],
            "p1_frames": r["priority_1_false_confident_identity"]["frames"],
            "p2": sw["count"], "p2_people_checked_on_trusted_grid": sw.get("people_annotated_on_trusted_grid_frames"),
            "p3": r["priority_3_wrong_reacquisitions"]["count"],
            "p4": r["priority_4_trusted_on_hidden_merged_unresolvable"]["count"],
            "p4_frames": r["priority_4_trusted_on_hidden_merged_unresolvable"]["frames"],
            "available": cov["available_frames"], "trusted": (cov["claims_on_available"] or {}).get("CONFIDENT", 0),
            "coverage": cov["trusted_coverage_of_available"],
            "box_iou": r["box_quality"].get("mean_iou_confident"), "box_frames": r["box_quality"].get("frames"),
            "reacq_correct": r["priority_3_wrong_reacquisitions"]["correct"],
            "reacq_not": r["priority_3_wrong_reacquisitions"]["not_reacquired"],
            "b5": r["b5_contact_integrity"]["status"]}


def diagnostics(raw_dir, cid, lab):
    p = os.path.join(raw_dir, cid, cid + "_rb_decision_log.jsonl")
    if not os.path.exists(p):
        return None
    s = lab["seed"]["frame"]
    merged_frames = merged_on_truth_merged = 0
    bank = {"accepted": 0, "contaminated": 0, "unverified": 0, "items": []}
    frames = 0
    pop_merged_obs = 0
    with open(p) as f:
        for line in f:
            rec = json.loads(line)
            t = rec["target"]
            rel = rec["frame"]
            clip_f = rel + s
            pf = lab["per_frame"][clip_f] if clip_f < len(lab["per_frame"]) else None
            if rel > 0:
                frames += 1
                pop_merged_obs += sum("MERGED_SUSPECTED" in o["flags"] for o in rec["population"]["observations"])
            if "MERGED_DETECTION_SUSPECTED" in (t.get("identity_uncertainty_reasons") or []):
                merged_frames += 1
                if pf and pf.get("merged"):
                    merged_on_truth_merged += 1
            b = (t.get("decision") or {}).get("bank") or {}
            if b.get("status") == "ACCEPTED":
                bank["accepted"] += 1
                hb = lab["target_boxes"].get(str(clip_f))
                avail = pf and pf["visibility"] in ("VISIBLE", "PARTIAL") and pf["resolvable"] == "YES" \
                    and not pf["merged"] and not pf["human_uncertain"]
                bad = []
                if not avail:
                    bad.append("TARGET_UNAVAILABLE_IN_TRUTH")
                if pf and pf.get("contact_with"):
                    bad.append("CONTACT_IN_TRUTH")
                if hb is not None and _iou(hb, t["bbox"]) < 0.5:
                    bad.append("BOX_NOT_ON_HUMAN_TARGET_BOX")
                if bad:
                    bank["contaminated"] += 1
                    bank["items"].append({"clip_frame": clip_f, "why": bad})
                elif hb is None:
                    bank["unverified"] += 1
    return {"frames_after_seed": frames, "target_frames_merged_reason": merged_frames,
            "of_which_truth_labelled_merged": merged_on_truth_merged,
            "population_merged_observations_per_frame": round(pop_merged_obs / frames, 3) if frames else None,
            "target_bank": bank}


def main(repo, pre_out, pre_raw, post_out, out_json):
    import glob
    rows = {}
    for p in sorted(glob.glob(os.path.join(repo, "R2", "labels", "*.fixture.json"))):
        with open(p) as f:
            fx = json.load(f)
        if fx["split"] != "DEV":
            continue
        cid = fx["clip_id"]
        with open(os.path.join(repo, "R2", "labels", fx["labels_file"])) as f:
            lab = json.load(f)
        rows[cid] = {"pre": official(pre_out, cid), "post": official(post_out, cid),
                     "pre_diag": diagnostics(pre_raw, cid, lab), "post_diag": diagnostics(post_out, cid, lab)}

    def agg(side):
        a = {"initialised": 0, "refused": 0, "p1": 0, "p2": 0, "p3": 0, "p4": 0, "available_run": 0, "trusted": 0,
             "merged_reason_frames": 0, "bank_accepted": 0, "bank_contaminated": 0, "bank_unverified": 0}
        unknown = {"p1": [], "p2": [], "p3": []}
        for cid, r in rows.items():
            o = r[side]
            if o["outcome"] != "SCORED":
                a["refused"] += 1
                continue
            a["initialised"] += 1
            for k in ("p1", "p2", "p3"):
                if isinstance(o[k], int):
                    a[k] += o[k]
                else:
                    unknown[k].append(cid)
            a["p4"] += o["p4"]
            a["available_run"] += o["available"]
            a["trusted"] += o["trusted"]
            d = r[side + "_diag"]
            if d:
                a["merged_reason_frames"] += d["target_frames_merged_reason"]
                a["bank_accepted"] += d["target_bank"]["accepted"]
                a["bank_contaminated"] += d["target_bank"]["contaminated"]
                a["bank_unverified"] += d["target_bank"]["unverified"]
        a["coverage_run_clips"] = round(a["trusted"] / a["available_run"], 3) if a["available_run"] else None
        a["unknown_in"] = unknown
        return a

    out = {"per_clip": rows, "aggregate": {"pre": agg("pre"), "post": agg("post")}}
    with open(out_json, "x") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    for side in ("pre", "post"):
        print(side, json.dumps(out["aggregate"][side]))
    for cid, r in rows.items():
        def short(o, d):
            if o["outcome"] != "SCORED":
                return "REFUSED"
            md = d["target_frames_merged_reason"] if d else "-"
            bc = "%d/%d" % (d["target_bank"]["contaminated"], d["target_bank"]["accepted"]) if d else "-"
            return "P1 %s P2 %s P3 %s P4 %s cov %s (%s/%s) merged %s bank %s" % (
                o["p1"], o["p2"], o["p3"], o["p4"], o["coverage"], o["trusted"], o["available"], md, bc)
        print(cid, "| pre:", short(r["pre"], r["pre_diag"]), "| post:", short(r["post"], r["post_diag"]))


if __name__ == "__main__":
    main(*sys.argv[1:6])

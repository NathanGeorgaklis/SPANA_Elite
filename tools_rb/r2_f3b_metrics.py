# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2 DEV remediation instruction (Nathan, 2026-10-31 05:05, F3b): compare duplicate same-body
#                 tracklet births, MULTI_CLAIM, false merge closures before/after
"""Read-only F3b diagnostics on R2 DEV decision logs (DEV only; never opens video; recomputes no official metric).

    python3 tools_rb/r2_f3b_metrics.py REPO OUT_DIR OUT_JSON

Per clip, from <clip>/<clip>_rb_decision_log.jsonl:
  * births: tracklets born this frame; part_born = born with >= PART (0.5) of the smaller of (its box, the box of a
    tracklet observed in the same frame and born earlier) inside the other (the F3b geometry; independent of the
    tracker's own population_part_birth_containment so it can be measured on runs that predate it);
  * multi_claim_frames: frames with a detection whose merge_reasons include MULTI_CLAIM;
  * merged_contact_closures: tracklets closed MERGED_CONTACT; of which with_part_born_claimant: closed together with
    (or as) a part-born tracklet whose birth partner is among the closed set (the F3b signature);
  * target_merged_closures: the target's own tracklet (target.observation_tracklet_id on the previous frame) closed
    MERGED_CONTACT;
  * target_frames_merged_reason: target frames whose reasons include MERGED_DETECTION_SUSPECTED;
  * unproven_pairs_seen: distinct unordered pairs reported in tracklets[].unproven_distinct_from (0 on old runs).
"""
import glob
import json
import os
import sys

PART = 0.5


def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _part_overlap(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    m = min(_area(a), _area(b))
    return ix * iy / m if m > 0 else 0.0


def clip_metrics(log_path):
    births = part_born = 0
    partner = {}
    mc_frames = mcc = mcc_part = tgt_mc = tgt_merged = 0
    pairs = set()
    prev_tid = None
    with open(log_path) as f:
        for line in f:
            r = json.loads(line)
            pop = r["population"]
            obs = pop["observations"]
            trk = {v["tracklet_id"]: v for v in pop["tracklets"]}
            for v in pop["tracklets"]:
                for u in v.get("unproven_distinct_from") or []:
                    pairs.add(frozenset((v["tracklet_id"], u)))
            observed = {tid: obs[v["detection_index"]]["bbox"] for tid, v in trk.items()
                        if v.get("detection_index") is not None and v["observation_source"] == "DETECTION"}
            for tid, v in trk.items():
                if v["first_frame"] != r["frame"] or tid not in observed:
                    continue
                births += 1
                b = observed[tid]
                best = max(((_part_overlap(b, b2), t2) for t2, b2 in observed.items()
                            if t2 != tid and trk[t2]["first_frame"] < r["frame"]), default=(0.0, None))
                if best[0] >= PART:
                    part_born += 1
                    partner[tid] = best[1]
            if any("MULTI_CLAIM" in (o.get("merge_reasons") or []) for o in obs):
                mc_frames += 1
            closed = {c["tracklet_id"] for c in pop["closed"] if c["reason"] == "MERGED_CONTACT"}
            mcc += len(closed)
            if any(partner.get(t) in closed for t in closed):
                mcc_part += len(closed)
            if prev_tid is not None and prev_tid in closed:
                tgt_mc += 1
            t = r["target"]
            if "MERGED_DETECTION_SUSPECTED" in (t.get("identity_uncertainty_reasons") or []) and r["frame"] > 0:
                tgt_merged += 1
            prev_tid = t.get("observation_tracklet_id")
    return {"births": births, "part_born": part_born, "multi_claim_frames": mc_frames,
            "merged_contact_closures": mcc, "merged_contact_closures_with_part_born_claimant": mcc_part,
            "target_merged_closures": tgt_mc, "target_frames_merged_reason": tgt_merged,
            "unproven_pairs_seen": len(pairs)}


def main(repo, out_dir, out_json):
    rows = {}
    for p in sorted(glob.glob(os.path.join(repo, "R2", "labels", "*.fixture.json"))):
        with open(p) as f:
            fx = json.load(f)
        if fx["split"] != "DEV":
            continue
        cid = fx["clip_id"]
        lp = os.path.join(out_dir, cid, cid + "_rb_decision_log.jsonl")
        if os.path.exists(lp):
            rows[cid] = clip_metrics(lp)
    tot = {k: sum(r[k] for r in rows.values()) for k in next(iter(rows.values()))} if rows else {}
    with open(out_json, "x") as f:
        json.dump({"per_clip": rows, "total": tot}, f, indent=1, sort_keys=True)
    print("total", json.dumps(tot))
    for cid, r in rows.items():
        print(cid, json.dumps(r))


if __name__ == "__main__":
    main(*sys.argv[1:4])

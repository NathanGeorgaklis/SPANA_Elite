# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2 stride / size-anomaly stage (Nathan, 2026-11-07 10:34). Evaluation tooling only: per-clip
#                 counts from two user-test output directories' rb decision logs (no tracker logic).
"""Per-clip comparison of two R2 user-test runs (native or 360p) from their decision logs.

    python3 tools_rb/r2_native_compare.py PRE_OUT_DIR POST_OUT_DIR CLIP_ID [CLIP_ID ...]

Prints, for each clip and run: frames, trusted frames and coverage, LOST frames / %, first LOST frame, target
frames flagged merged_suspected, SIZE_ANOMALY / merge flags on the target's observed box, size_anomaly_held frames,
size_anomaly_ended reasons, state counts, the tracklets carrying trusted frames (a second trusted tracklet is a
wrong-player candidate to review visually) and the commonest uncertainty reasons. Writes NATIVE_COMPARE.json into
POST_OUT_DIR. Counts only; nothing here decides anything.
"""
import json, sys, os
from collections import Counter

def stats(out_dir, cid):
    f = os.path.join(out_dir, cid, cid + "_rb_decision_log.jsonl")
    if not os.path.exists(f):
        return None
    n = 0; trusted = 0; lost = 0; mds = 0; sa_held = 0; sa_ended = Counter(); first_lost = None
    states = Counter(); trusted_tracklets = Counter(); anomaly_frames = 0; merged_contact = 0; reasons = Counter()
    for line in open(f):
        d = json.loads(line); t = d.get("target") or {}
        if not t: continue
        n += 1
        st = t.get("rb_state") or t.get("state"); states[st] += 1
        if t.get("identity_trusted"):
            trusted += 1; trusted_tracklets[t.get("observation_tracklet_id")] += 1
        if st == "LOST":
            lost += 1
            if first_lost is None: first_lost = d["frame"]
        if t.get("merged_suspected"): mds += 1
        for r in t.get("identity_uncertainty_reasons") or []: reasons[r] += 1
        dec = t.get("decision") or {}
        if dec.get("size_anomaly_held"): sa_held += 1
        if dec.get("size_anomaly_ended"): sa_ended[dec["size_anomaly_ended"]] += 1
        # population flags on the target's observed box
        ti = t.get("detection_index")
        if ti is not None and t.get("bbox_is_observed"):
            obs = (d.get("population") or {}).get("observations") or []
            if 0 <= ti < len(obs):
                fl = obs[ti].get("flags") or []
                if "SIZE_ANOMALY" in fl: anomaly_frames += 1
                if "MERGED_SUSPECTED" in fl or "MERGED_CONTACT" in fl: merged_contact += 1
    return dict(frames=n, trusted=trusted, cov=round(trusted / n, 3) if n else None, lost=lost, lost_pct=round(100 * lost / n, 1) if n else None,
                first_lost=first_lost, merged_suspected_target=mds, size_anomaly_target_obs=anomaly_frames, merged_flag_target_obs=merged_contact,
                size_anomaly_held=sa_held, size_anomaly_ended=dict(sa_ended), states=dict(states), trusted_tracklets=dict(trusted_tracklets),
                reasons=dict(reasons.most_common(8)))

if __name__ == "__main__":
    pre, post = sys.argv[1], sys.argv[2]; cids = sys.argv[3:]
    out = {}
    for cid in cids:
        out[cid] = {"pre": stats(pre, cid), "post": stats(post, cid)}
        print(cid); print("  pre ", json.dumps(out[cid]["pre"])); print("  post", json.dumps(out[cid]["post"]))
    json.dump(out, open(os.path.join(post, "NATIVE_COMPARE.json"), "w"), indent=1)

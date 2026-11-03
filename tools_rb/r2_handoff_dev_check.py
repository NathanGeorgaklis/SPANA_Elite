# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2 DEV same-frame identity handoff safety (Nathan, 2026-11-02 20:14): per DEV clip, trusted frames
#                 against the baseline run, and every APPEARANCE_STEP suspension checked against the frozen labels.
"""Read-only DEV check of handoff suspensions (DEV fixtures only).

    python3 tools_rb/r2_handoff_dev_check.py REPO BASE_OUT NEW_OUT CLIP_ID [CLIP_ID ...]

For each clip: trusted frames in BASE_OUT and NEW_OUT; every frame whose decision log says continuity_broken =
APPEARANCE_STEP, with the suspended tracklet's observed box (the target's tracklet on the previous frame) against the
human box (linearly interpolated between labelled frames); resolved and unresolved suspensions.
"""
import json
import os
import sys


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def load(out, cid):
    recs = {}
    for line in open(os.path.join(out, cid, cid + "_rb_decision_log.jsonl")):
        x = json.loads(line)
        if "target" in x:
            recs[x["frame"]] = x
    return recs


def main(repo, base, new, cids):
    rep = {}
    for cid in cids:
        with open(os.path.join(repo, "R2", "labels", cid + ".fixture.json")) as f:
            fx = json.load(f)
        if fx["split"] != "DEV":
            raise SystemExit(f"{cid} is not a DEV clip: refused")
        with open(os.path.join(repo, "R2", "labels", fx["labels_file"])) as f:
            lab = json.load(f)
        s = fx["seed_frame"]
        tb = {int(k): v for k, v in lab["target_boxes"].items()}
        a, b = load(base, cid), load(new, cid)
        rows = []
        for k in sorted(b):
            t = b[k]["target"]
            if t["decision"].get("continuity_broken") != "APPEARANCE_STEP":
                continue
            tid = b[k - 1]["target"]["observation_tracklet_id"] if k - 1 in b else None
            ob = [o for o in b[k]["population"]["observations"] if o.get("tracklet_id") == tid]
            cf = k + s
            lo = max([q for q in tb if q <= cf], default=None)
            hi = min([q for q in tb if q >= cf], default=None)
            v = None
            if ob and lo is not None and hi is not None:
                w = 0.0 if hi == lo else (cf - lo) / (hi - lo)
                v = _iou(ob[0]["bbox"], [x + w * (y - x) for x, y in zip(tb[lo], tb[hi])])
            st = t["decision"]["continuity_step"]
            rows.append({"frame": k, "tracklet": tid, "label_gap": None if lo is None or hi is None else hi - lo,
                         "iou_to_human_box": v, "truth": None if v is None else ("TARGET" if v >= 0.5 else
                                                                                 "UNCLEAR" if v >= 0.1 else "OTHER"),
                         "step": st.get("step"), "baseline": st.get("baseline"), "why": st.get("why")})
        rep[cid] = {"trusted_base": sum(x["target"]["identity_trusted"] for x in a.values()),
                    "trusted_new": sum(x["target"]["identity_trusted"] for x in b.values()),
                    "suspensions": rows,
                    "resolved_frames": [k for k, x in b.items()
                                        if (x["target"]["decision"].get("continuity_step") or {}).get("resolved")],
                    "unresolved_frames": [k for k, x in b.items() if x["target"]["decision"].get("handoff_unresolved")]}
    return rep


if __name__ == "__main__":
    r = main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:])
    print(json.dumps(r, indent=1))

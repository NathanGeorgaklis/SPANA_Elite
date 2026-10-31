# SPANA engine v2-rebuild
# REBUILT 2026-11 — NOT ORIGINAL SOURCE
# Evidence basis: R2 DEV F5 instruction (Nathan, 2026-11-01 01:35): improve correct reacquisition after gaps /
#                 contact / anomalies without weakening identity safety; DEV only; diagnose before changing thresholds.
"""Read-only F5 diagnosis of a DEV run: why is the target NOT trusted on frames where the frozen human truth says it is
available? DEV only (refuses EVAL fixtures); never changes an official metric.

    python3 tools_rb/r2_f5_reacq_diag.py REPO OUT_DIR CLIP_ID [CLIP_ID ...] > report.txt

For every truth frame that is available (VISIBLE / PARTIAL, resolvable, not merged, not human-uncertain) and has a
human target box, the target-layer decision record is matched to the human box:
  * TRUE_BOX_DETECTED: some population observation has IoU >= 0.5 with the human box, or (box convention, F7)
    lies >= 0.9 inside it with >= 0.5 of its height (else NOT_DETECTED);
  * the target's evaluation of that box (continuation or reacquisition candidate), if any: role, details (the gates
    it failed), composite / embedding / LAB target similarity, best competitor and its similarity, contrast, margin;
  * the target state, transition, reasons, episode and whether the episode is contrast-verified.
Frames are grouped into a blocker class:
  TRUSTED, NOT_DETECTED, TRUE_BOX_NOT_EVALUATED (detected but no evaluation of it: e.g. excluded as a known other,
  merged / alternative, or the target was occluded/ambiguous on another box), FAILED:<gates> (evaluated and failed),
  PASSING_VERIFYING (passed, reacquisition window not complete), UNVERIFIED_EPISODE (TRACKING on it, untrusted),
  ON_OTHER_BOX (the target is on a different box).
"""
import collections
import json
import os
import sys


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def _inside(a, b):
    """Fraction of box a inside box b."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1])
    return ix * iy / area if area > 0 else 0.0


def load(repo, cid):
    with open(os.path.join(repo, "R2", "labels", cid + ".fixture.json")) as f:
        fx = json.load(f)
    if fx["split"] != "DEV":
        raise SystemExit(f"{cid} is not a DEV clip: refused")
    with open(os.path.join(repo, "R2", "labels", fx["labels_file"])) as f:
        return json.load(f)


def frames(out_dir, cid):
    with open(os.path.join(out_dir, cid, cid + "_rb_decision_log.jsonl")) as f:
        for line in f:
            yield json.loads(line)


def classify(rec, hb):
    t = rec["target"]
    d = t.get("decision") or {}
    obs = rec["population"]["observations"]
    ious = [(_iou(hb, o["bbox"]), o) for o in obs if o.get("duplicate_of") is None]
    best_iou, true_obs = max(ious, key=lambda x: x[0]) if ious else (0.0, None)
    if best_iou < 0.5:
        # box convention (F7): a detector box wholly inside a looser human box is the same body when it is most of
        # its height (the F1 seed CONTAINMENT geometry): >= 0.9 inside, height >= 0.5 of the human box
        cont = [(_inside(o["bbox"], hb), o) for o in obs if o.get("duplicate_of") is None
                and (o["bbox"][3] - o["bbox"][1]) >= 0.5 * (hb[3] - hb[1])]
        cont = [x for x in cont if x[0] >= 0.9]
        if cont:
            best_iou, true_obs = 0.5, max(cont, key=lambda x: (x[1]["bbox"][3] - x[1]["bbox"][1]))[1]
    row = {"state": t["rb_state"], "trusted": t["identity_trusted"], "episode": t.get("target_episode"),
           "verified": d.get("episode_contrast_verified"), "transition": d.get("transition"),
           "reasons": t.get("identity_uncertainty_reasons"), "true_iou": round(best_iou, 3)}
    if t["identity_trusted"]:
        di = t.get("detection_index")
        on = di is not None and di < len(obs) and _iou(hb, obs[di]["bbox"]) >= 0.5
        return ("TRUSTED" if on else "TRUSTED_ON_OTHER_BOX"), row
    if best_iou < 0.5:
        return "NOT_DETECTED", row
    row["true_tracklet"] = true_obs["tracklet_id"]
    row["true_flags"] = true_obs["flags"]
    ev = None
    for cnd in d.get("candidates", []):
        if cnd["detection_index"] == true_obs["index"]:
            ev = cnd
    di = t.get("detection_index")
    if t["rb_state"] == "TRACKING" and di == true_obs["index"]:
        return "UNVERIFIED_EPISODE", row
    if ev is None:
        excl = (d.get("reacquisition") or {}).get("excluded_known_others") or []
        why = "KNOWN_OTHER" if true_obs["tracklet_id"] in excl else "NOT_EVALUATED"
        return "TRUE_BOX_" + why, row
    ts = ev.get("target_similarity") or {}
    row.update({"role": ev["role"], "details": ev["details"],
                "composite": ts.get("composite") and round(ts["composite"], 3),
                "embedding": ts.get("embedding") and round(ts["embedding"], 3),
                "lab": ts.get("lab") and round(ts["lab"], 3),
                "best_competitor": ev.get("best_competitor"),
                "competitor_sim": ev.get("best_competitor_similarity") and round(ev["best_competitor_similarity"], 3),
                "contrast": ev.get("contrast") and round(ev["contrast"], 3), "margin": ev.get("margin")})
    if ev["passed"]:
        return "PASSING_" + t["rb_state"], row
    return "FAILED:" + "+".join(sorted(ev["details"])), row


def main(repo, out_dir, cids):
    for cid in cids:
        lab = load(repo, cid)
        s = lab["seed"]["frame"]
        counts = collections.Counter()
        runs = []
        for rec in frames(out_dir, cid):
            cf = rec["frame"] + s
            if cf >= len(lab["per_frame"]):
                break
            pf = lab["per_frame"][cf]
            hb = lab["target_boxes"].get(str(cf))
            avail = pf["visibility"] in ("VISIBLE", "PARTIAL") and pf["resolvable"] == "YES" and not pf["merged"] \
                and not pf["human_uncertain"]
            if not avail or hb is None:
                continue
            cls, row = classify(rec, hb)
            counts[cls] += 1
            if runs and runs[-1][0] == cls:
                runs[-1][2] = rec["frame"]
                runs[-1][3] += 1
            else:
                runs.append([cls, rec["frame"], rec["frame"], 1, row])
        print("=" * 100)
        print(cid, "boxed available frames:", sum(counts.values()))
        for k, v in counts.most_common():
            print(f"  {v:5d}  {k}")
        print("  runs (class, first..last rel frame, n, first-frame detail):")
        for cls, a, b, n, row in runs:
            print(f"   {cls:55s} {a:4d}..{b:4d} n={n:3d} {json.dumps(row)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])

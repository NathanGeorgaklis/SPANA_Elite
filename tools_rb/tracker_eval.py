# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R0/R0_TRACKER_ACCEPTANCE_TARGETS.md Part B (B1–B5 metric definitions, NEW in R0);
#                 R1-H instruction (Nathan, 2026-10-26: evaluate Pass-1 against frozen human truth; priorities
#                 false confident identity, swaps, wrong reacquisition, trusted on hidden/merged, then coverage)
"""Pass-1 tracker evaluation against frozen human truth (R1-H). Evaluation code only: it never feeds the tracker.

The tracker's claim per frame is read from the written TARGET record. ``v2.identity_trusted`` is the AUTHORITY for a
confident identity claim (R1-H downstream rule); legacy fields are ignored for that purpose.

Truth is mapped into a generic per-frame model (``TruthFrame``). A label file format is supported through an adapter
(``ADAPTERS``); the metrics never see benchmark column names. Where the truth cannot support a metric (for example no
human boxes), the metric is reported as UNKNOWN with the reason, never estimated.

    python3 tools_rb/tracker_eval.py RESULTS_JSON FIXTURE_JSON LABELS_CSV FREEZE_RECORD_JSON FORMAT OUT_JSON

Fails closed when: the labels' sha256 differs from the freeze record; the results were produced from a different
video than the fixture's; a label frame lies outside the run; the contract is violated.
"""
import csv
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from player_tracker_rb import output_schema as S  # noqa: E402

# Evaluation constants (R0 Part B definitions; evaluation only, NOT tracker parameters).
FCI_IOU_MAX = 0.3            # B1: a confident box with IoU < 0.3 to the human target box is a false confident identity
MATCH_IOU_MIN = 0.5          # B1/B3: a confident box with IoU >= 0.5 to a box "is on" that person
MISSED_REACQ_S = 2.0         # B3: target available this long after a gap while never trusted = missed reacquisition

VISIBLE, PARTIAL, HIDDEN, OUT_OF_FRAME = "VISIBLE", "PARTIAL", "HIDDEN", "OUT_OF_FRAME"
UNKNOWN = "UNKNOWN"


class EvalRefused(Exception):
    pass


@dataclass(frozen=True)
class TruthFrame:
    rel_frame: int
    visibility: str                         # VISIBLE / PARTIAL / HIDDEN / OUT_OF_FRAME
    identity_resolvable: bool
    merged: bool                            # target inside a box shared with / hidden by another person
    target_box: Optional[Tuple[float, float, float, float]] = None
    other_boxes: Optional[Dict[str, Tuple[float, float, float, float]]] = None   # None = not annotated
    clean_reappearance: bool = False

    @property
    def available(self) -> bool:
        """The target could be confidently identified by a careful human from the image."""
        return self.visibility in (VISIBLE, PARTIAL) and self.identity_resolvable and not self.merged


# ---------------------------------------------------------------------------------------------------------------------
# Label adapters. Benchmark column names live ONLY here (evaluation code), never in player_tracker_rb.
# ---------------------------------------------------------------------------------------------------------------------
def _gold_v1(rows: Sequence[Dict[str, str]], seed_frame: int) -> List[TruthFrame]:
    """GOLD_HUMAN_LABELS_v1: identity / visibility only, no boxes (frozen 2026-10-24)."""
    vis_map = {"visible": VISIBLE, "partial": PARTIAL, "hidden": HIDDEN}
    out = []
    for r in rows:
        vis = r["t23_visibility"].strip().lower()
        if vis not in vis_map:
            raise EvalRefused(f"unknown visibility {vis!r} at frame {r['frame']}")
        res = r["t23_identity_resolvable"].strip().lower()
        if res not in ("yes", "no"):
            raise EvalRefused(f"unknown resolvable {res!r} at frame {r['frame']}")
        mh = r["merged_or_hidden"].strip().lower()
        if mh not in ("none", "merged", "t23_hidden"):
            raise EvalRefused(f"unknown merged_or_hidden {mh!r} at frame {r['frame']}")
        out.append(TruthFrame(rel_frame=int(r["frame"]) - seed_frame, visibility=vis_map[vis],
                              identity_resolvable=res == "yes", merged=mh != "none",
                              clean_reappearance=r["first_clean_reappearance"].strip().upper() == "Y"))
    return out


ADAPTERS = {"gold_v1": _gold_v1}


def read_truth(labels_csv: str, fmt: str, seed_frame: int) -> List[TruthFrame]:
    if fmt not in ADAPTERS:
        raise EvalRefused(f"unknown label format {fmt!r}")
    with open(labels_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    truth = ADAPTERS[fmt](rows, seed_frame)
    frames = [t.rel_frame for t in truth]
    if len(set(frames)) != len(frames):
        raise EvalRefused("duplicate label frames")
    return sorted(truth, key=lambda t: t.rel_frame)


# ---------------------------------------------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------------------------------------------
def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def claim(rec: Dict[str, Any]) -> str:
    """The tracker's claim for one TARGET record. identity_trusted is the authority."""
    v2 = rec["v2"]
    if v2["identity_trusted"]:
        return "CONFIDENT"
    if v2["rb_state"] == "TRACKING":
        return "OBSERVED_UNVERIFIED"
    return v2["rb_state"]                      # OCCLUDED / AMBIGUOUS / REACQUIRING / LOST


def _runs(frames: Sequence[int]) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    for f in sorted(frames):
        if runs and f == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], f)
        else:
            runs.append((f, f))
    return runs


def evaluate(doc: Dict[str, Any], truth: Sequence[TruthFrame]) -> Dict[str, Any]:
    fps = float(doc["summary"]["fps"])
    sel = {}
    for fr in doc["frame_log"]:
        tgt = [t for t in fr["tracks"] if t["selected_player"]]
        if len(tgt) != 1:
            raise EvalRefused(f"frame {fr['frame']}: {len(tgt)} target records")
        sel[fr["frame"]] = tgt[0]
    outside = [t.rel_frame for t in truth if t.rel_frame not in sel]
    if outside:
        raise EvalRefused(f"label frames outside the run: {outside[:5]}")
    if not truth:
        raise EvalRefused("no truth frames")

    claims = {t.rel_frame: claim(sel[t.rel_frame]) for t in truth}
    box = {t.rel_frame: tuple(sel[t.rel_frame]["observed_position_box"]) for t in truth}
    confident = [t for t in truth if claims[t.rel_frame] == "CONFIDENT"]

    # contract: a confident claim is always an observed detection (B2, automatic)
    grounded_violations = [t.rel_frame for t in confident
                           if not (sel[t.rel_frame]["v2"]["rb_state"] == "TRACKING"
                                   and sel[t.rel_frame]["v2"]["bbox_is_observed"])]

    # priority 4 / B1 part 1: confident while the human says the identity is not available
    on_unavailable = [{"frame": t.rel_frame, "visibility": t.visibility, "resolvable": t.identity_resolvable,
                       "merged": t.merged} for t in confident if not t.available]

    # B1 part 2: confident on the wrong box (needs human target boxes)
    wrong_box, ious, unverifiable = [], [], []
    for t in confident:
        if not t.available:
            continue
        if t.target_box is None:
            unverifiable.append(t.rel_frame)
            continue
        v = iou(box[t.rel_frame], t.target_box)
        ious.append(v)
        if v < FCI_IOU_MAX:
            wrong_box.append({"frame": t.rel_frame, "iou": round(v, 3)})
    fci_frames = sorted({d["frame"] for d in on_unavailable} | {d["frame"] for d in wrong_box})
    fci = {"frames": fci_frames, "count": len(fci_frames),
           "on_unavailable_target": on_unavailable, "on_wrong_box": wrong_box,
           "confident_frames_without_box_truth": unverifiable,
           "status": "COMPLETE" if not unverifiable else "PARTIAL_NO_BOX_TRUTH"}

    # B1 swaps: maximal runs of confident frames on a DIFFERENT annotated person (needs other people's boxes)
    if any(t.other_boxes is None for t in confident):
        swaps: Dict[str, Any] = {"count": UNKNOWN, "reason": "other people's boxes not annotated"}
    else:
        on_other = [t.rel_frame for t in confident
                    if any(iou(box[t.rel_frame], ob) >= MATCH_IOU_MIN for ob in t.other_boxes.values())]
        swaps = {"count": len(_runs(on_other)), "runs": _runs(on_other)}

    # B3 reacquisition: after each span where the target is not available, the first confident frame
    reacq = []
    unavailable = [t.rel_frame for t in truth if not t.available]
    by_frame = {t.rel_frame: t for t in truth}
    last = truth[-1].rel_frame
    for (g0, g1) in _runs(unavailable):
        back = next((t for t in truth if t.rel_frame > g1 and t.available), None)
        if back is None:
            reacq.append({"gap": [g0, g1], "target_available_from": None, "outcome": "NOT_AVAILABLE_IN_WINDOW"})
            continue
        clean = next((t.rel_frame for t in truth if t.rel_frame > g1 and t.clean_reappearance), None)
        first = next((t for t in truth if t.rel_frame > g1 and claims[t.rel_frame] == "CONFIDENT"), None)
        item: Dict[str, Any] = {"gap": [g0, g1], "target_available_from": back.rel_frame,
                                "human_clean_reappearance": clean}
        if first is None:
            censored = (last - back.rel_frame + 1) / fps
            # "missed" needs the target available for MISSED_REACQ_S; a shorter labelled window cannot tell
            missed = True if censored >= MISSED_REACQ_S - 0.5 / fps else UNKNOWN
            item.update({"outcome": "NOT_REACQUIRED_IN_WINDOW", "delay_s_at_least": round(censored, 3),
                         "missed_reacquisition": missed})
        else:
            item["first_confident_frame"] = first.rel_frame
            item["delay_s"] = round((first.rel_frame - back.rel_frame) / fps, 3)
            if not by_frame[first.rel_frame].available:
                item["outcome"] = "WRONG (target not available)"
            elif first.target_box is None:
                item["outcome"] = "UNVERIFIED_NO_BOX_TRUTH"
            else:
                v = iou(box[first.rel_frame], first.target_box)
                item["iou"] = round(v, 3)
                item["outcome"] = "CORRECT" if v >= MATCH_IOU_MIN else "WRONG"
        reacq.append(item)
    wrong_reacq = [r for r in reacq if str(r["outcome"]).startswith("WRONG")]

    # coverage and abstention (B1 correct-when-visible, B2)
    avail = [t for t in truth if t.available]
    dist_avail: Dict[str, int] = {}
    for t in avail:
        dist_avail[claims[t.rel_frame]] = dist_avail.get(claims[t.rel_frame], 0) + 1
    dist_unavail: Dict[str, int] = {}
    for t in truth:
        if not t.available:
            dist_unavail[claims[t.rel_frame]] = dist_unavail.get(claims[t.rel_frame], 0) + 1
    n_av = len(avail)
    n_un = len(truth) - n_av
    cov = {"available_frames": n_av,
           "trusted_coverage_of_available": round(dist_avail.get("CONFIDENT", 0) / n_av, 3) if n_av else None,
           "abstention_rate_on_available": round(1 - dist_avail.get("CONFIDENT", 0) / n_av, 3) if n_av else None,
           "claims_on_available": dict(sorted(dist_avail.items())),
           "rates_on_available": {k: round(v / n_av, 3) for k, v in sorted(dist_avail.items())} if n_av else {},
           "unavailable_frames": n_un,
           "claims_on_unavailable": dict(sorted(dist_unavail.items())),
           "uncertainty_when_unavailable": round(1 - dist_unavail.get("CONFIDENT", 0) / n_un, 3) if n_un else None}
    by_vis = {}
    for t in avail:
        d = by_vis.setdefault(t.visibility, {"frames": 0, "trusted": 0})
        d["frames"] += 1
        d["trusted"] += claims[t.rel_frame] == "CONFIDENT"
    cov["by_visibility_on_available"] = by_vis

    box_quality = ({"mean_iou_confident": round(sum(ious) / len(ious), 3), "frames": len(ious)} if ious else
                   {"mean_iou_confident": UNKNOWN, "reason": "no human target boxes on confident frames"})

    return {
        "window": [truth[0].rel_frame, truth[-1].rel_frame], "fps": fps, "truth_frames": len(truth),
        "constants": {"FCI_IOU_MAX": FCI_IOU_MAX, "MATCH_IOU_MIN": MATCH_IOU_MIN, "MISSED_REACQ_S": MISSED_REACQ_S},
        "priority_1_false_confident_identity": fci,
        "priority_2_identity_swaps": swaps,
        "priority_3_wrong_reacquisitions": {"count": len(wrong_reacq), "items": wrong_reacq},
        "priority_4_trusted_on_hidden_merged_unresolvable": {"count": len(on_unavailable),
                                                              "frames": [d["frame"] for d in on_unavailable]},
        "confident_not_grounded_violations": grounded_violations,
        "reacquisition": reacq,
        "coverage_and_abstention": cov,
        "box_quality": box_quality,
        "claims_per_frame": {str(t.rel_frame): claims[t.rel_frame] for t in truth},
    }


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(results_json, fixture_json, labels_csv, freeze_json, fmt, out_json):
    with open(freeze_json) as f:
        frozen = json.load(f)
    if _sha(labels_csv) != frozen.get("sha256"):
        raise EvalRefused("labels sha256 differs from the freeze record (fail closed)")
    with open(fixture_json) as f:
        fx = json.load(f)
    with open(results_json, encoding="ascii") as f:
        doc = json.load(f)
    errs = S.validate_document(doc, require_v2=True)
    if errs:
        raise EvalRefused(f"results violate the contract: {errs[:3]}")
    prov = doc["summary"]["v2"]["provenance"]
    if prov.get("video_sha256") != fx["video_sha256"]:
        raise EvalRefused("results were produced from a different video than the fixture's (fail closed)")
    if doc["summary"]["start_frame"] != fx["seed_frame"] or prov.get("seed_frame") != fx["seed_frame"]:
        raise EvalRefused("results start_frame differs from the fixture seed frame (fail closed)")
    if [round(v, 3) for v in prov.get("seed_bbox") or []] != [round(v, 3) for v in fx["seed_bbox"]]:
        raise EvalRefused("results seed box differs from the fixture seed box (fail closed)")
    rep = evaluate(doc, read_truth(labels_csv, fmt, fx["seed_frame"]))
    rep.update({"tool": "tools_rb/tracker_eval.py", "events_evaluated": False, "label_format": fmt,
                "labels_sha256": _sha(labels_csv), "results_sha256": _sha(results_json),
                "results_git_commit": prov.get("git_commit"), "results_git_dirty": prov.get("git_dirty"),
                "results_config_hash": doc["summary"]["v2"]["config_hash"]})
    with open(out_json, "x") as f:
        json.dump(rep, f, indent=1, sort_keys=True)
        f.write("\n")
    return rep


if __name__ == "__main__":
    if len(sys.argv) != 7:
        raise SystemExit(__doc__)
    r = main(*sys.argv[1:7])
    print(json.dumps({k: r[k] for k in r if k.startswith("priority") or k in ("coverage_and_abstention",
                                                                              "reacquisition", "box_quality")},
                     indent=1))

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

    python3 tools_rb/tracker_eval.py RESULTS_JSON FIXTURE_JSON LABELS FREEZE_RECORD_JSON FORMAT OUT_JSON [EVAL_APPROVAL]

EVAL clips (fixture / freeze split EVAL) are refused unless EVAL_APPROVAL is a record with status APPROVED, approved_by
and tracker_git_commit equal to the (clean) commit that produced the results.

Fails closed when:
- the freeze record is not FROZEN / valid, names another labels file, or its sha256 differs from the labels;
- the results were produced from another video, seed frame or seed box than the fixture's;
- the results, fixture and freeze record disagree on fps, or a label's video_time_s does not match its frame;
- a label value is unknown, the truth is not one contiguous frame range, or a label frame lies outside the run;
- the results violate the contract; the report file already exists.

Error counts are numbers only when every relevant frame could be checked against the truth; otherwise they are
UNKNOWN and the verified part is reported separately (``verified_count`` / ``verified_wrong``) as a lower bound.
"""
import csv
import hashlib
import io
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from player_tracker_rb import output_schema as S  # noqa: E402
from player_tracker_rb import provenance as P  # noqa: E402

# Evaluation constants (R0 Part B definitions; evaluation only, NOT tracker parameters).
FCI_IOU_MAX = 0.3            # B1: a confident box with IoU < 0.3 to the human target box is a false confident identity
MATCH_IOU_MIN = 0.5          # B1/B3: a confident box with IoU >= 0.5 to a box "is on" that person
MISSED_REACQ_S = 2.0         # B3: target available this long after a gap while never trusted = missed reacquisition
B5_WINDOW_S = 0.5            # B5: contact window half-width around a human-marked contact

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
    contact: Optional[bool] = None          # target overlapping / in contact with another person; None = not annotated
    box_annotated: bool = True              # boxes are part of the truth on this frame (False: between box samples)
    uncertain: bool = False                 # HUMAN_UNCERTAIN: excluded from error counts and coverage, reported
    contact_people: Tuple[str, ...] = ()    # who the target is in contact with (ids into other_boxes)
    people_absent: Tuple[str, ...] = ()     # annotated people the annotator marked not visible on this frame

    @property
    def available(self) -> bool:
        """The target could be confidently identified by a careful human from the image."""
        return self.visibility in (VISIBLE, PARTIAL) and self.identity_resolvable and not self.merged


# ---------------------------------------------------------------------------------------------------------------------
# Label adapters. Benchmark column names live ONLY here (evaluation code), never in player_tracker_rb.
# ---------------------------------------------------------------------------------------------------------------------
def _choice(r: Dict[str, str], col: str, allowed: Sequence[str]) -> str:
    v = r[col].strip().lower()
    if v not in allowed:
        raise EvalRefused(f"unknown {col} value {r[col]!r} at frame {r['frame']} (fail closed)")
    return v


def _gold_v1(rows: Sequence[Dict[str, str]], seed_frame: int, fps: float) -> List[TruthFrame]:
    """GOLD_HUMAN_LABELS_v1: identity / visibility / contact only, no boxes (frozen 2026-10-24). Every column used is
    validated against its allowed values (case-insensitive) and video_time_s against frame / fps."""
    vis_map = {"visible": VISIBLE, "partial": PARTIAL, "hidden": HIDDEN}
    out = []
    for r in rows:
        frame = int(r["frame"])
        if abs(float(r["video_time_s"]) - frame / fps) > 0.5 / fps:
            raise EvalRefused(f"video_time_s {r['video_time_s']} does not match frame {frame} at {fps} fps")
        vis = _choice(r, "t23_visibility", tuple(vis_map))
        res = _choice(r, "t23_identity_resolvable", ("yes", "no"))
        mh = _choice(r, "merged_or_hidden", ("none", "merged", "t23_hidden"))
        ov = _choice(r, "overlap_23_4", ("yes", "no"))
        clean = _choice(r, "first_clean_reappearance", ("", "y"))
        out.append(TruthFrame(rel_frame=frame - seed_frame, visibility=vis_map[vis], identity_resolvable=res == "yes",
                              merged=mh != "none", clean_reappearance=clean == "y", contact=ov == "yes"))
    return out


def _gold_v1_bytes(data: bytes, seed_frame: int, fps: float) -> List[TruthFrame]:
    return _gold_v1(list(csv.DictReader(io.StringIO(data.decode("utf-8"), newline=""))), seed_frame, fps)


def _r2_v1(data: bytes, seed_frame: int, fps: float) -> List[TruthFrame]:
    """R2 labels (tools_rb/r2_labels.py, schema r2_labels_v1). Pre-registered mapping (R2/R2_PREREGISTRATION.md §5):
    * frames after the seed only (rel = clip frame - seed frame; the seed is the run's frame 0);
    * available = VISIBLE / PARTIAL, resolvable YES, not merged;
    * box_annotated = the frame is on the label's required box grid (5 Hz + event padding) or carries a target box;
      between grid frames box checks are not made (reported as off-grid confident frames);
    * other_boxes = every annotated person box on that frame; contact = contact_with is non-empty;
    * HUMAN_UNCERTAIN frames are excluded from error counts and coverage and reported separately."""
    from tools_rb import r2_labels as L        # evaluation-only dependency
    lab = json.loads(data.decode("utf-8"))
    errs = L.validate(lab)
    if errs:
        raise EvalRefused(f"r2 labels invalid: {errs[:3]}")
    if lab.get("clip_rejected"):
        raise EvalRefused(f"clip rejected by the annotator: {lab['clip_rejected']}")
    if abs(float(lab["fps"]) - fps) > 1e-6 or lab["seed"]["frame"] != seed_frame:
        raise EvalRefused("labels fps / seed frame differ from the run (fail closed)")
    grid = set(L.box_frames(lab))
    out = []
    for i in range(seed_frame + 1, lab["frames"]):
        f = lab["per_frame"][i]
        tb = lab["target_boxes"].get(str(i))
        others = {pid: tuple(p["boxes"][str(i)]) for pid, p in lab["people"].items() if str(i) in p["boxes"]}
        out.append(TruthFrame(
            rel_frame=i - seed_frame,
            visibility={"VISIBLE": VISIBLE, "PARTIAL": PARTIAL, "HIDDEN": HIDDEN,
                        "OUT_OF_FRAME": OUT_OF_FRAME}[f["visibility"]],
            identity_resolvable=f["resolvable"] == "YES", merged=bool(f["merged"]),
            target_box=tuple(tb) if tb else None, other_boxes=others,
            clean_reappearance=i in lab["first_clean_reappearance"], contact=bool(f["contact_with"]),
            contact_people=tuple(f["contact_with"]),
            people_absent=tuple(pid for pid, p in lab["people"].items() if i in p.get("absent_frames", [])),
            box_annotated=(i in grid or tb is not None), uncertain=bool(f["human_uncertain"])))
    return out


ADAPTERS = {"gold_v1": _gold_v1_bytes, "r2_v1": _r2_v1}


def read_truth(labels_csv: str, fmt: str, seed_frame: int, fps: float) -> List[TruthFrame]:
    with open(labels_csv, "rb") as f:
        return parse_truth(f.read(), fmt, seed_frame, fps)


def parse_truth(data: bytes, fmt: str, seed_frame: int, fps: float) -> List[TruthFrame]:
    """Parse label bytes (the SAME bytes whose hash was checked, so nothing can change between check and use)."""
    if fmt not in ADAPTERS:
        raise EvalRefused(f"unknown label format {fmt!r}")
    return check_truth(ADAPTERS[fmt](data, seed_frame, fps))


def check_truth(truth: Sequence[TruthFrame]) -> List[TruthFrame]:
    """Truth must cover one contiguous frame range: runs (swaps, gaps) are built over consecutive frames, so sparse
    truth would split one event into several (fail closed until sparse truth is supported explicitly)."""
    truth = sorted(truth, key=lambda t: t.rel_frame)
    frames = [t.rel_frame for t in truth]
    if not frames:
        raise EvalRefused("no truth frames")
    if len(set(frames)) != len(frames):
        raise EvalRefused("duplicate label frames")
    if frames != list(range(frames[0], frames[-1] + 1)):
        raise EvalRefused("truth frames are not contiguous (fail closed)")
    return truth


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
    if not isinstance(v2["identity_trusted"], bool):
        raise EvalRefused(f"identity_trusted must be a bool, got {v2['identity_trusted']!r} (fail closed)")
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


def _count(verified: int, unverified: int) -> Any:
    """An error count is a number only when every relevant frame could be checked; otherwise UNKNOWN (the verified
    part is reported separately as a lower bound). Never report 0 for 'could not check'."""
    return verified if unverified == 0 else UNKNOWN


def evaluate(doc: Dict[str, Any], truth: Sequence[TruthFrame]) -> Dict[str, Any]:
    truth = check_truth(truth)
    fps = float(doc["summary"]["fps"])
    sel = {}
    for fr in doc["frame_log"]:
        tgt = [t for t in fr["tracks"] if t["selected_player"]]
        if len(tgt) != 1:
            raise EvalRefused(f"frame {fr['frame']}: {len(tgt)} target records")
        if fr["frame"] in sel:
            raise EvalRefused(f"frame {fr['frame']} appears twice in the frame log (fail closed)")
        sel[fr["frame"]] = tgt[0]
    outside = [t.rel_frame for t in truth if t.rel_frame not in sel]
    if outside:
        raise EvalRefused(f"label frames outside the run: {outside[:5]}")

    claims = {t.rel_frame: claim(sel[t.rel_frame]) for t in truth}
    box = {t.rel_frame: tuple(sel[t.rel_frame]["observed_position_box"]) for t in truth}
    confident_all = [t for t in truth if claims[t.rel_frame] == "CONFIDENT"]
    # HUMAN_UNCERTAIN frames: no error can be asserted there; excluded from errors and coverage, reported
    confident = [t for t in confident_all if not t.uncertain]
    confident_on_uncertain = [t.rel_frame for t in confident_all if t.uncertain]
    by_frame = {t.rel_frame: t for t in truth}
    first, last = truth[0].rel_frame, truth[-1].rel_frame

    # B2 (automatic): a confident claim is an observed detection of this frame, never held / predicted
    def grounded(v2):
        return (v2["rb_state"] == "TRACKING" and v2["bbox_is_observed"] and v2.get("observation_source") == "DETECTION"
                and not v2.get("bbox_is_held") and not v2.get("bbox_is_predicted"))
    grounded_violations = [t.rel_frame for t in confident_all if not grounded(sel[t.rel_frame]["v2"])]

    # priority 4 / B1 part 1: confident while the human says the identity is not available
    on_unavailable = [{"frame": t.rel_frame, "visibility": t.visibility, "resolvable": t.identity_resolvable,
                       "merged": t.merged} for t in confident if not t.available]

    # B1 part 2: confident on the wrong box (needs human target boxes)
    wrong_box, ious, unverifiable, off_grid = [], [], [], []
    for t in confident:
        if not t.available:
            continue
        if not t.box_annotated:
            off_grid.append(t.rel_frame)                   # between box samples: no box check on this frame
            continue
        if t.target_box is None:
            unverifiable.append(t.rel_frame)
            continue
        v = iou(box[t.rel_frame], t.target_box)
        ious.append(v)
        if v < FCI_IOU_MAX:
            wrong_box.append({"frame": t.rel_frame, "iou": round(v, 3)})
    fci_frames = sorted({d["frame"] for d in on_unavailable} | {d["frame"] for d in wrong_box})
    fci = {"count": _count(len(fci_frames), len(unverifiable)), "verified_count": len(fci_frames),
           "frames": fci_frames, "on_unavailable_target": on_unavailable, "on_wrong_box": wrong_box,
           "confident_frames_without_box_truth": unverifiable,
           "confident_frames_between_box_samples": len(off_grid),
           "box_checked_fraction_of_confident_available": (round(len(ious) / (len(ious) + len(off_grid)
                                                                              + len(unverifiable)), 3)
                                                           if (ious or off_grid or unverifiable) else None),
           # COMPLETE: every trusted available frame box-checked; COMPLETE_ON_GRID: every frame on the human box
           # grid checked, trusted frames between grid samples not box-checked (pre-registered, Addendum A)
           "status": ("PARTIAL_NO_BOX_TRUTH" if unverifiable else "COMPLETE_ON_GRID" if off_grid else "COMPLETE")}

    # B1 swaps: maximal runs of confident frames on a DIFFERENT annotated person (needs other people's boxes)
    boxed_conf = [t for t in confident if t.box_annotated]
    if any(t.other_boxes is None for t in boxed_conf):
        swaps: Dict[str, Any] = {"count": UNKNOWN, "reason": "other people's boxes not annotated"}
    else:
        on_other = [t.rel_frame for t in boxed_conf
                    if any(iou(box[t.rel_frame], ob) >= MATCH_IOU_MIN for ob in t.other_boxes.values())]
        swaps = {"count": len(_runs(on_other)), "runs": _runs(on_other),
                 "scope": "trusted grid frames vs every ANNOTATED other person; a trusted box on an unannotated "
                          "person is caught as a wrong box under priority 1",
                 "people_annotated_on_trusted_grid_frames": len({p for t in boxed_conf for p in t.other_boxes}),
                 "count_is_lower_bound": True}

    # B3 reacquisition: one item per span where the target is not available. A confident frame INSIDE the span is a
    # premature (wrong) reacquisition. After the span, only the available run up to the next span is searched.
    reacq = []
    gaps = _runs([t.rel_frame for t in truth if not t.available])
    for k, (g0, g1) in enumerate(gaps):
        seg_end = gaps[k + 1][0] - 1 if k + 1 < len(gaps) else last
        inside = [f for f in range(g0, g1 + 1) if claims[f] == "CONFIDENT" and not by_frame[f].uncertain]
        # a gap starting at the first truth frame may have begun before the labelled window: flagged, not dropped
        item: Dict[str, Any] = {"gap": [g0, g1], "trusted_inside_gap": inside, "gap_starts_at_window_start": g0 == first}
        if g1 == last:
            item.update({"target_available_from": None, "outcome": "WRONG_PREMATURE" if inside
                         else "NOT_AVAILABLE_IN_WINDOW"})
            reacq.append(item)
            continue
        back = g1 + 1                                      # truth is contiguous: the next frame is available
        clean = next((t.rel_frame for t in truth if back <= t.rel_frame <= seg_end and t.clean_reappearance), None)
        # first trusted frame the truth can judge (a trusted HUMAN_UNCERTAIN frame is reported, not judged)
        conf = next((f for f in range(back, seg_end + 1) if claims[f] == "CONFIDENT" and not by_frame[f].uncertain),
                    None)
        item.update({"target_available_from": back, "target_available_until": seg_end,
                     "human_clean_reappearance": clean})
        if conf is None:
            avail_s = (seg_end - back + 1) / fps
            if avail_s >= MISSED_REACQ_S - 0.5 / fps:
                missed: Any = True
            elif seg_end < last:
                missed = False                             # the target was not available long enough to require it
            else:
                missed = UNKNOWN                           # the labelled window ends too soon to tell
            item.update({"post_gap_outcome": "NOT_REACQUIRED", "delay_s_at_least": round(avail_s, 3),
                         "censored_by": "window_end" if seg_end == last else "next_gap",
                         "missed_reacquisition": missed})
        else:
            item["first_confident_frame"] = conf
            item["delay_s"] = round((conf - back) / fps, 3)
            # judged on the first box-annotated frame of the confident run that starts at `conf`
            judge = conf
            while judge <= seg_end and claims[judge] == "CONFIDENT" and (not by_frame[judge].box_annotated
                                                                          or by_frame[judge].uncertain):
                judge += 1
            if (judge > seg_end or claims[judge] != "CONFIDENT" or not by_frame[judge].box_annotated
                    or by_frame[judge].uncertain):
                judge = None
            item["judged_on_frame"] = judge
            ft = by_frame[judge] if judge is not None else None
            if ft is not None:
                conf = judge
            if ft is None or ft.target_box is None or ft.uncertain:
                item["post_gap_outcome"] = "UNVERIFIED_NO_BOX_TRUTH"
            else:
                v = iou(box[conf], ft.target_box)
                item["iou"] = round(v, 3)
                item["post_gap_outcome"] = "CORRECT" if v >= MATCH_IOU_MIN else "WRONG"
        item["outcome"] = "WRONG_PREMATURE" if inside else item["post_gap_outcome"]
        reacq.append(item)
    wrong_reacq = [r for r in reacq if r["outcome"] in ("WRONG", "WRONG_PREMATURE")]
    unverified_reacq = [r for r in reacq if r["outcome"] == "UNVERIFIED_NO_BOX_TRUTH"]
    p3 = {"count": _count(len(wrong_reacq), len(unverified_reacq)), "verified_wrong": len(wrong_reacq),
          "correct": sum(r["outcome"] == "CORRECT" for r in reacq), "unverified": len(unverified_reacq),
          "not_reacquired": sum(r["outcome"] == "NOT_REACQUIRED" for r in reacq), "items": wrong_reacq}

    # B5: no confident claim on the contact partner within +-0.5 s of a human-marked contact
    if any(t.contact is None for t in truth):
        b5: Dict[str, Any] = {"status": UNKNOWN, "reason": "contact not annotated"}
    else:
        half = int(round(B5_WINDOW_S * fps))
        contact = [t.rel_frame for t in truth if t.contact]
        window = sorted({f for c in contact for f in range(c - half, c + half + 1) if first <= f <= last})
        conf_in = [f for f in window if claims[f] == "CONFIDENT" and not by_frame[f].uncertain
                   and by_frame[f].box_annotated]
        # partners whose contact window covers each frame; a frame is checkable only if each such partner is boxed
        # there or marked absent (review finding H1: never PASS on frames that could not be checked)
        partners: Dict[int, set] = {}
        for t in truth:
            for pid in t.contact_people:
                for g in range(t.rel_frame - half, t.rel_frame + half + 1):
                    partners.setdefault(g, set()).add(pid)
        unchecked = [f for f in conf_in if by_frame[f].other_boxes is None or any(
            pid not in by_frame[f].other_boxes and pid not in by_frame[f].people_absent
            for pid in partners.get(f, ()))]
        if not conf_in:
            status, on_partner = "PASS_NO_CONFIDENT_CLAIM", []
        elif not unchecked:
            on_partner = [f for f in conf_in
                          if any(iou(box[f], ob) >= MATCH_IOU_MIN for ob in by_frame[f].other_boxes.values())]
            status = "FAIL" if on_partner else "PASS"
        else:
            status, on_partner = UNKNOWN, []
        b5 = {"status": status, "contact_frames": contact, "window_frames": [window[0], window[-1]] if window else None,
              "confident_frames_in_window": conf_in, "confident_on_partner": on_partner,
              "confident_frames_without_partner_boxes": unchecked if conf_in else [],
              "window_clipped_to_truth": True}

    # coverage and abstention (B1 correct-when-visible, B2)
    avail = [t for t in truth if t.available and not t.uncertain]
    dist_avail: Dict[str, int] = {}
    for t in avail:
        dist_avail[claims[t.rel_frame]] = dist_avail.get(claims[t.rel_frame], 0) + 1
    dist_unavail: Dict[str, int] = {}
    for t in truth:
        if not t.available and not t.uncertain:
            dist_unavail[claims[t.rel_frame]] = dist_unavail.get(claims[t.rel_frame], 0) + 1
    unres = [t for t in truth if not t.identity_resolvable and not t.uncertain]
    n_av = len(avail)
    n_un = sum(dist_unavail.values())
    cov = {"available_frames": n_av,
           "trusted_coverage_of_available": round(dist_avail.get("CONFIDENT", 0) / n_av, 3) if n_av else None,
           "abstention_rate_on_available": round(1 - dist_avail.get("CONFIDENT", 0) / n_av, 3) if n_av else None,
           "claims_on_available": dict(sorted(dist_avail.items())),
           "rates_on_available": {k: round(v / n_av, 3) for k, v in sorted(dist_avail.items())} if n_av else {},
           "unavailable_frames": n_un,
           "claims_on_unavailable": dict(sorted(dist_unavail.items())),
           "uncertainty_when_unavailable": round(1 - dist_unavail.get("CONFIDENT", 0) / n_un, 3) if n_un else None,
           "unresolvable_frames": len(unres),
           "uncertainty_when_unresolvable_r0_b1": (round(1 - sum(claims[t.rel_frame] == "CONFIDENT" for t in unres)
                                                         / len(unres), 3) if unres else None)}
    by_vis = {}
    for t in avail:
        d = by_vis.setdefault(t.visibility, {"frames": 0, "trusted": 0})
        d["frames"] += 1
        d["trusted"] += claims[t.rel_frame] == "CONFIDENT"
    cov["by_visibility_on_available"] = by_vis

    box_quality = ({"mean_iou_confident": round(sum(ious) / len(ious), 3), "frames": len(ious)} if ious else
                   {"mean_iou_confident": UNKNOWN, "reason": "no human target boxes on confident frames"})

    return {
        "window": [first, last], "fps": fps, "truth_frames": len(truth),
        "constants": {"FCI_IOU_MAX": FCI_IOU_MAX, "MATCH_IOU_MIN": MATCH_IOU_MIN, "MISSED_REACQ_S": MISSED_REACQ_S,
                      "B5_WINDOW_S": B5_WINDOW_S},
        "priority_1_false_confident_identity": fci,
        "priority_2_identity_swaps": swaps,
        "priority_3_wrong_reacquisitions": p3,
        "priority_4_trusted_on_hidden_merged_unresolvable": {"count": len(on_unavailable),
                                                              "frames": [d["frame"] for d in on_unavailable]},
        "b5_contact_integrity": b5,
        "confident_not_grounded_violations": grounded_violations,
        "human_uncertain": {"frames": sum(t.uncertain for t in truth), "confident_on_uncertain": confident_on_uncertain},
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


def main(results_json, fixture_json, labels_csv, freeze_json, fmt, out_json, eval_approval_json=None, *, git=None):
    with open(freeze_json) as f:
        frozen = json.load(f)
    if frozen.get("status") != "FROZEN" or frozen.get("valid") is not True:
        raise EvalRefused("freeze record is not FROZEN / valid (fail closed)")
    if os.path.basename(labels_csv) != frozen.get("labels_file", frozen.get("labels_csv")):
        raise EvalRefused("labels file is not the one named by the freeze record (fail closed)")
    with open(labels_csv, "rb") as f:
        label_bytes = f.read()
    labels_sha = hashlib.sha256(label_bytes).hexdigest()
    if labels_sha != frozen.get("sha256"):
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
    # labels, freeze record, fixture and results must all describe the same clip (review finding H2)
    for what, a, b in (("fixture labels_sha256", fx.get("labels_sha256", labels_sha), labels_sha),
                       ("freeze clip_sha256", frozen.get("clip_sha256", fx["video_sha256"]), fx["video_sha256"]),
                       ("freeze clip_id", frozen.get("clip_id", fx.get("clip_id")), fx.get("clip_id")),
                       ("freeze seed_frame", frozen.get("seed_frame", fx["seed_frame"]), fx["seed_frame"])):
        if a != b:
            raise EvalRefused(f"{what} {a!r} != {b!r}: labels / fixture / results describe different clips")
    if fmt == "r2_v1":
        lab = json.loads(label_bytes.decode("utf-8"))
        if lab.get("clip_sha256") != fx["video_sha256"] or lab.get("clip_id") != fx.get("clip_id"):
            raise EvalRefused("labels belong to another clip than the fixture (fail closed)")
    # EVAL gate (review finding M7): EVAL is scored only for a tracker commit Nathan approved in advance
    if fx.get("split") == "EVAL" or frozen.get("split") == "EVAL":
        if not eval_approval_json:
            raise EvalRefused("EVAL clip: an approval record naming the tracker commit is required (fail closed)")
        with open(eval_approval_json) as f:
            appr = json.load(f)
        if not (appr.get("status") == "APPROVED" and str(appr.get("approved_by", "")).strip()
                and appr.get("tracker_git_commit") == prov.get("git_commit") and prov.get("git_dirty") is False):
            raise EvalRefused("EVAL approval does not cover this clean tracker commit (fail closed)")
    fps = float(doc["summary"]["fps"])
    for name, other in (("fixture", fx.get("fps")), ("freeze record", frozen.get("fps"))):
        if other is None or abs(float(other) - fps) > 1e-6:
            raise EvalRefused(f"fps {fps} of the results differs from the {name}'s {other} (fail closed)")
    rep = evaluate(doc, parse_truth(label_bytes, fmt, fx["seed_frame"], fps))
    head, dirty = (git or P.git_state)()
    rep["eval_approval_sha256"] = _sha(eval_approval_json) if eval_approval_json else None
    rep.update({"tool": "tools_rb/tracker_eval.py", "tool_sha256": _sha(os.path.abspath(__file__)),
                "evaluator_git_commit": head, "evaluator_git_dirty": dirty,
                "command": ["python3", "tools_rb/tracker_eval.py", results_json, fixture_json, labels_csv,
                            freeze_json, fmt, out_json],
                "events_evaluated": False, "label_format": fmt,
                "labels_sha256": labels_sha, "freeze_record_sha256": _sha(freeze_json),
                "fixture_sha256": _sha(fixture_json), "results_sha256": _sha(results_json),
                "results_git_commit": prov.get("git_commit"), "results_git_dirty": prov.get("git_dirty"),
                "results_config_hash": doc["summary"]["v2"]["config_hash"]})
    with open(out_json, "x") as f:
        json.dump(rep, f, indent=1, sort_keys=True)
        f.write("\n")
    return rep


if __name__ == "__main__":
    if len(sys.argv) not in (7, 8):
        raise SystemExit(__doc__)
    r = main(*sys.argv[1:8])
    print(json.dumps({k: r[k] for k in r if k.startswith("priority") or k in ("coverage_and_abstention",
                                                                              "reacquisition", "box_quality",
                                                                              "b5_contact_integrity")},
                     indent=1))

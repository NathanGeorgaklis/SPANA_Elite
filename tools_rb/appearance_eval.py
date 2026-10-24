# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R1-D instruction (GOLD only for appearance analysis, not tracker or event success); R1 approval §4
#                 (DINOv2 usefulness must be measured); R1/R1C/detector_eval (person candidates from R1-C)
"""Appearance-cue analysis on a development clip: do the cues tell the same person from different people?

No tracker, identity label, event or fixture semantics are used. From the fixture only video_sha256 and seed_frame
are read (to open the same frames the R1-C detector evaluation used). Person boxes come from that evaluation's saved
candidates (default setting s_grid2x2), so no detection is re-run here.

How pairs are built (physics, not model output; documented weak labels, NOT human-verified):
  * A candidate is ISOLATED if score >= 0.5, height >= 40 px, and no other candidate intersects its box expanded by
    ISOLATION_PAD x its height on every side.
  * Chains: an isolated candidate in frame f links to an isolated candidate in frame f+1 when their IoU >= 0.6 and the
    match is unique both ways (no other candidate with IoU >= 0.3 on either side). One step is 1/fps s, and with no
    other person nearby a different person cannot take that place. A chain stops at the first frame without such a link.
  * SAME-person pairs: two samples of one chain at least SAME_GAP_S seconds apart (appearance changes with pose/time).
  * DIFFERENT-person pairs: two isolated candidates of different chains in the same frame (two bodies at once).
  * LAB-CONFUSABLE different pairs (a same-kit proxy): different-person pairs whose LAB similarity is >= LAB_CONFUSABLE
    (0.6, declared constant): their colour distributions overlap at least 60%, so colour cannot separate them.
    A contact sheet of these pairs is written next to the report so a human can check they are similar-kit people.
Samples are taken every SAMPLE_STEP frames (deterministic).

Reported per cue (LAB, embedding with each pooling, composite): same/different distributions, AUC (probability that a
same pair scores higher than a different pair), and how often identity_support() holds for same, different and
LAB-confusable pairs. Crop-quality flags and bank refusals are counted over all candidates.

Pooling selection rule (declared before running): the pooling with the higher embedding AUC on same vs LAB-confusable
different pairs; if the two are within 0.01, "cls" (the simpler one).

Floor calibration rule (declared before the authoritative run): embedding_floor and appearance_floor (composite) =
the 5th percentile of same-person similarity for the selected pooling, rounded DOWN to a multiple of 0.05. lab_floor is
not calibrated: colour only vetoes obviously incompatible kits (0.3). Support rates are reported under the configured
floors and under the calibrated floors. The calibration is in-sample on weak labels from one clip: it gives the floors a
meaning on DINOv2's scale, but it is UNVALIDATED until R2.

Usage: python3 tools_rb/appearance_eval.py VIDEO FIXTURE_JSON DETECTOR_EVAL_DIR SETTING OUT_JSON
"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from player_tracker_rb import appearance as A  # noqa: E402
from player_tracker_rb import detector as D  # noqa: E402
from player_tracker_rb import output_schema as S  # noqa: E402
from player_tracker_rb.config import TrackerConfig  # noqa: E402
from player_tracker_rb.embedding_dinov2 import Dinov2OnnxEmbedding, pool  # noqa: E402
from player_tracker_rb.video_io import SequentialVideoReader  # noqa: E402

MIN_SCORE = 0.5
MIN_HEIGHT = 40.0
ISOLATION_PAD = 0.25
LINK_IOU = 0.6
AMBIG_IOU = 0.3
SAME_GAP_S = 0.5
SAMPLE_STEP = 5
POOLINGS = ("cls", "cls_patchmean")
CALIBRATION_PERCENTILE = 5
CALIBRATION_STEP = 0.05
LAB_CONFUSABLE = 0.6
SHEET_PAIRS = 16


def load_candidates(det_dir, setting):
    files = sorted(f for f in os.listdir(det_dir) if f.startswith(f"raw_{setting}__") and f.endswith(".json"))
    frames = [fr for f in files for fr in json.load(open(os.path.join(det_dir, f)))["frames"]]
    if [f["rel"] for f in frames] != list(range(len(frames))):
        raise SystemExit("detector candidates do not cover frames 0..N-1 exactly once")
    return {f["rel"]: np.array(f["cands"], float).reshape(-1, 6) for f in frames}


def isolated_mask(c):
    n = c.shape[0]
    ok = (c[:, 4] >= MIN_SCORE) & ((c[:, 3] - c[:, 1]) >= MIN_HEIGHT)
    for i in range(n):
        if not ok[i]:
            continue
        pad = ISOLATION_PAD * (c[i, 3] - c[i, 1])
        e = c[i, :4] + np.array([-pad, -pad, pad, pad])
        for j in range(n):
            if j != i and not (c[j, 2] <= e[0] or c[j, 0] >= e[2] or c[j, 3] <= e[1] or c[j, 1] >= e[3]):
                ok[i] = False
                break
    return ok


def build_chains(cands):
    rels = sorted(cands)
    iso = {r: isolated_mask(cands[r]) for r in rels}
    chain_of = {}
    chains = []
    for r in rels:
        c = cands[r]
        for i in np.nonzero(iso[r])[0]:
            key = (r, int(i))
            if key in chain_of:
                continue
            chain_of[key] = len(chains)
            chains.append([key])
        if r + 1 not in cands:
            continue
        n = cands[r + 1]
        if n.shape[0] == 0 or c.shape[0] == 0:
            continue
        iou = D.box_iou_matrix(c[:, :4], n[:, :4])
        for i in np.nonzero(iso[r])[0]:
            j = int(np.argmax(iou[i]))
            if iou[i, j] < LINK_IOU or not iso[r + 1][j]:
                continue
            if (iou[i] >= AMBIG_IOU).sum() != 1 or (iou[:, j] >= AMBIG_IOU).sum() != 1:
                continue
            if (r + 1, j) in chain_of:
                continue
            k = chain_of[(r, int(i))]
            chain_of[(r + 1, j)] = k
            chains[k].append((r + 1, j))
    return chains, chain_of, iso


def auc(same, diff):
    same, diff = np.asarray(same, float), np.asarray(diff, float)
    if same.size == 0 or diff.size == 0:
        return None
    gt = (same[:, None] > diff[None, :]).sum() + 0.5 * (same[:, None] == diff[None, :]).sum()
    return float(gt / (same.size * diff.size))


def stats(x):
    x = np.asarray(x, float)
    if x.size == 0:
        return None
    return {"n": int(x.size), "mean": round(float(x.mean()), 4), "p10": round(float(np.percentile(x, 10)), 4),
            "median": round(float(np.median(x)), 4), "p90": round(float(np.percentile(x, 90)), 4)}


def main(video, fx_json, det_dir, setting, out_json):
    fxall = json.load(open(fx_json))
    fx = {k: fxall[k] for k in ("video_sha256", "seed_frame")}
    base = TrackerConfig.default()
    emb = Dinov2OnnxEmbedding(TrackerConfig.with_overrides({"embedding_pooling": "cls"}))
    cands = load_candidates(det_dir, setting)
    chains, chain_of, iso = build_chains(cands)
    sampled = {(r, i) for ch in chains for (r, i) in ch if r % SAMPLE_STEP == 0}
    feats = {}                                          # (rel, idx) -> dict of features per pooling
    crops = {}                                          # (rel, idx) -> small crop for the contact sheet
    quality_flags, bank_refusals, n_cands = {}, {}, 0
    probe_bank = A.AppearanceBank(base, require_trust=True)
    fps = None
    with SequentialVideoReader(video, fx["seed_frame"], base, expected_sha256=fx["video_sha256"]) as reader:
        fps = reader.info.fps
        for fr in [reader.seed] + list(reader):
            r = fr.rel_frame
            c = cands.get(r)
            if c is None or c.shape[0] == 0:
                continue
            h, w = fr.image.shape[:2]
            for i in range(c.shape[0]):
                n_cands += 1
                others = [c[j, :4] for j in range(c.shape[0]) if j != i]
                q = A.assess_crop(c[i, :4], w, h, base, others=others)
                for f in q.flags:
                    quality_flags[f] = quality_flags.get(f, 0) + 1
                if (r, i) not in sampled and r % SAMPLE_STEP != 0:
                    continue
                crop = A.crop_image(fr.image, c[i, :4])
                lab = A.lab_descriptor(crop, base) if (crop is not None and q.valid) else None
                vec = {}
                if crop is not None and q.valid:
                    cls, pm = emb.raw_outputs(crop)
                    vec["cls"] = pool(cls, pm, "cls")
                    vec["cls_patchmean"] = pool(cls, pm, "cls_patchmean")
                # Bank-refusal behaviour on real crops, offered as if trusted (to isolate the quality rules).
                fq = A.AppearanceFeature(lab=lab, embedding=vec.get("cls"), quality=q)
                for x in probe_bank.update_refusals(fq, A.BankUpdateContext(identity_trusted=True)):
                    bank_refusals[x] = bank_refusals.get(x, 0) + 1
                if (r, i) in sampled and lab is not None and vec:
                    feats[(r, i)] = {"lab": lab, "q": q, **vec}
                    crops[(r, i)] = cv2.resize(crop, (48, 96), interpolation=cv2.INTER_AREA)
    gap = int(np.ceil(SAME_GAP_S * fps))
    same_pairs, diff_pairs = [], []
    for k, ch in enumerate(chains):
        pts = [p for p in ch if p in feats]
        for a in range(len(pts)):
            for b in range(a + 1, len(pts)):
                if pts[b][0] - pts[a][0] >= gap:
                    same_pairs.append((pts[a], pts[b]))
    by_frame = {}
    for p in feats:
        by_frame.setdefault(p[0], []).append(p)
    for r, ps in by_frame.items():
        for a in range(len(ps)):
            for b in range(a + 1, len(ps)):
                if chain_of.get(ps[a]) != chain_of.get(ps[b]):
                    diff_pairs.append((ps[a], ps[b]))

    def scores(pairs, pooling, extra=None):
        lab, embs, comp, support = [], [], [], []
        cfg = TrackerConfig.with_overrides(dict({"embedding_pooling": pooling}, **(extra or {})))
        for a, b in pairs:
            fa = A.AppearanceFeature(lab=feats[a]["lab"], embedding=feats[a][pooling], quality=feats[a]["q"])
            fb = A.AppearanceFeature(lab=feats[b]["lab"], embedding=feats[b][pooling], quality=feats[b]["q"])
            s = A.compare(fa, fb, cfg)
            lab.append(s.lab)
            embs.append(s.embedding)
            comp.append(s.composite)
            support.append(A.identity_support(s, cfg)[0])
        return np.array(lab), np.array(embs), np.array(comp), np.array(support, bool)

    res = {}
    same_lab_median = None
    for pooling in POOLINGS:
        sl, se, sc, ss = scores(same_pairs, pooling)
        dl, de, dc, ds = scores(diff_pairs, pooling)
        same_lab_median = float(np.median(sl)) if sl.size else None
        conf = dl >= LAB_CONFUSABLE
        res[pooling] = {
            "lab": {"same": stats(sl), "different": stats(dl), "auc_same_vs_different": auc(sl, dl)},
            "embedding": {"same": stats(se), "different": stats(de), "lab_confusable_different": stats(de[conf]),
                          "auc_same_vs_different": auc(se, de), "auc_same_vs_lab_confusable": auc(se, de[conf])},
            "composite": {"same": stats(sc), "different": stats(dc), "auc_same_vs_different": auc(sc, dc),
                          "auc_same_vs_lab_confusable": auc(sc, dc[conf])},
            "lab_auc_same_vs_lab_confusable": auc(sl, dl[conf]),
            "identity_support_rate": {
                "same": round(float(ss.mean()), 4) if ss.size else None,
                "different": round(float(ds.mean()), 4) if ds.size else None,
                "lab_confusable_different": round(float(ds[conf].mean()), 4) if conf.sum() else None},
            "n_lab_confusable_different": int(conf.sum()),
        }
    a_cls = res["cls"]["embedding"]["auc_same_vs_lab_confusable"] or 0.0
    a_cpm = res["cls_patchmean"]["embedding"]["auc_same_vs_lab_confusable"] or 0.0
    selected = "cls_patchmean" if a_cpm > a_cls + 0.01 else "cls"
    # Floor calibration (selected pooling; rule in the module docstring).
    sl, se, sc, _ = scores(same_pairs, selected)

    def floor_step(x):
        return float(np.floor(np.percentile(x, CALIBRATION_PERCENTILE) / CALIBRATION_STEP) * CALIBRATION_STEP)

    calibrated = {"embedding_floor": round(floor_step(se), 2), "appearance_floor": round(floor_step(sc), 2)}
    cal_over = dict(calibrated, reacq_appearance_floor=max(base.get("reacq_appearance_floor"),
                                                            calibrated["appearance_floor"]))
    _, _, _, ss2 = scores(same_pairs, selected, cal_over)
    dl2, _, _, ds2 = scores(diff_pairs, selected, cal_over)
    conf2 = dl2 >= LAB_CONFUSABLE
    calibration = {"rule": f"P{CALIBRATION_PERCENTILE} of same-person similarity, floored to {CALIBRATION_STEP}",
                   "pooling": selected, "same_p5": {"embedding": round(float(np.percentile(se, 5)), 4),
                                                    "composite": round(float(np.percentile(sc, 5)), 4)},
                   "calibrated_floors": calibrated,
                   "identity_support_rate_with_calibrated_floors": {
                       "same": round(float(ss2.mean()), 4), "different": round(float(ds2.mean()), 4),
                       "lab_confusable_different": round(float(ds2[conf2].mean()), 4) if conf2.sum() else None}}
    lengths = [len(ch) for ch in chains]
    report = {
        "tool": "tools_rb/appearance_eval.py", "fixture_fields_used": list(fx), "detector_setting": setting,
        "tracker_identity_or_event_results_used": False, "fps": fps,
        "pair_construction": {"min_score": MIN_SCORE, "min_height_px": MIN_HEIGHT, "isolation_pad_h": ISOLATION_PAD,
                              "link_iou": LINK_IOU, "ambiguity_iou": AMBIG_IOU, "same_gap_s": SAME_GAP_S,
                              "same_gap_frames": gap, "sample_step_frames": SAMPLE_STEP,
                              "labels": "weak, physics-based (isolated frame-to-frame continuity); NOT human-verified"},
        "chains": {"n": len(chains), "n_len_ge_gap": int(sum(1 for x in lengths if x > gap)),
                   "max_len": int(max(lengths)) if lengths else 0},
        "pairs": {"same": len(same_pairs), "different": len(diff_pairs)},
        "same_lab_median": same_lab_median, "lab_confusable_min": LAB_CONFUSABLE,
        "results": res,
        "pooling_rule": "higher embedding AUC (same vs LAB-confusable different); within 0.01 -> cls",
        "pooling_selected": selected, "floor_calibration": calibration,
        "config_floors": {k: base.get(k) for k in ("lab_floor", "embedding_floor", "appearance_floor",
                                                     "composite_weight_lab", "composite_weight_embedding")},
        "crop_quality_flags_all_candidates": quality_flags, "candidates_seen": n_cands,
        "bank_refusals_if_offered_as_trusted": bank_refusals,
        "embedding_provenance": emb.provenance(),
    }
    # Contact sheet: the LAB-confusable different-person pairs (highest LAB first), for human checking.
    cfg_c = TrackerConfig.with_overrides({"embedding_pooling": "cls"})
    rows = []
    for a, b in diff_pairs:
        fa = A.AppearanceFeature(lab=feats[a]["lab"], embedding=feats[a]["cls"], quality=feats[a]["q"])
        fb = A.AppearanceFeature(lab=feats[b]["lab"], embedding=feats[b]["cls"], quality=feats[b]["q"])
        sm = A.compare(fa, fb, cfg_c)
        if sm.lab >= LAB_CONFUSABLE:
            rows.append((sm.lab, sm.embedding, A.identity_support(sm, cfg_c)[0], a, b))
    rows.sort(key=lambda t: (-t[0], t[3], t[4]))
    tiles = []
    for lab_s, emb_s, sup, a, b in rows[:SHEET_PAIRS]:
        t = np.full((120, 110, 3), 255, np.uint8)
        t[0:96, 4:52] = crops[a]
        t[0:96, 58:106] = crops[b]
        cv2.putText(t, f"L{lab_s:.2f} E{emb_s:.2f}", (2, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 0, 0), 1)
        cv2.putText(t, "SUPPORT" if sup else "no support", (2, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                    (0, 0, 200) if sup else (0, 120, 0), 1)
        tiles.append(t)
    if tiles:
        while len(tiles) % 4:
            tiles.append(np.full_like(tiles[0], 255))
        sheet = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
        cv2.imwrite(os.path.splitext(out_json)[0] + "_lab_confusable_pairs.png", sheet)
    report["lab_confusable_pairs_listed"] = [{"lab": round(l_, 4), "embedding": round(e_, 4), "support": bool(s_),
                                             "a": list(a), "b": list(b)} for l_, e_, s_, a, b in rows]
    with open(out_json, "w") as f:
        f.write(S.dumps_canonical(report))
    print(json.dumps(report, indent=1, sort_keys=True))


if __name__ == "__main__":
    main(*sys.argv[1:6])

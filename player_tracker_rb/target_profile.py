# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2 DEV positive target evidence instruction (Nathan, 2026-11-02 17:46): keep the tracker unchanged;
#                 represent the selected player by multiple trusted, diverse reference crops (varied pose/view anchors)
#                 from colour + DINOv2 evidence; team/kit and jersey number as support only; temporal consistency; only
#                 trusted target observations update it; no single cue proves identity; UNKNOWN / AMBIGUOUS stay valid;
#                 report best match AND consistency across anchors.
"""Target reference profile (NEW_R2, DIAGNOSTIC ONLY).

A read-only layer beside the target tracker. It NEVER changes the tracker: not its state, trust, bank, candidates or
reacquisition. It is not called by run_rb; tools and tests run it beside the tracker. It does not use NOT_TARGET
memory.

ANCHORS (the target profile). The profile starts from the user-selected player and grows only from TRUSTED target
observations: an anchor is offered only on a frame where the tracker's own trusted appearance bank ACCEPTED a sample
(the seed, or a trusted confirm that passed every bank hygiene rule: crop quality, isolation, no merge / contest /
edge / merge lineage, verified camera, post-reacquisition quarantine, self-consistency, novelty), and the offered
feature is exactly that bank sample. An untrusted, ambiguous or held frame can never add an anchor.
  * DIVERSITY: an anchor is kept only if its embedding similarity to every existing anchor is below
    1 - profile_anchor_min_novelty (REDUNDANT otherwise: near-duplicates are not kept). Over
    profile_max_anchors, the most redundant anchor (highest embedding similarity to another anchor) is evicted, never
    the seed anchor: the profile keeps different views (front / side / back / poses) instead of near-duplicates.
    No pose or orientation model exists in R2, so an anchor's view is recorded as UNKNOWN with its box aspect ratio
    (width / height) as a pose proxy for the record only.

MATCH of a crop against the profile (production composite: LAB + DINOv2, per anchor, never mixed across anchors):
  * best    the highest composite over the anchors (what a single best-match bank reports);
  * topk    the mean of the profile_consistency_k highest composites (fewer when fewer anchors): consistency;
  * support the fraction of anchors whose composite reaches appearance_floor;
  * kit     the median LAB similarity over the anchors (team / kit evidence: support only, veto below lab_floor).
A crop matching ONE anchor well (best high) but the rest poorly has a low topk and support: matching a single view
is not consistent target evidence.

TEMPORAL CONSISTENCY. For every tracked body, the matches of its own clean observations (usable crop, no
contamination flag, no partial / edge box, no merge / contest / size-anomaly / duplicate / gap-reassociated box, no
merge lineage) over the last profile_temporal_window_s are kept (pruned every frame); its temporal values are the
medians of topk, support and kit once they span at least profile_temporal_min_s (and >= 2 frames), and exist only on
a frame where the body's own observation is clean (no stale evidence for a box that cannot be judged now). The
history is cleared when its tracklet ends, its box gains a break flag, or on a camera continuity break. Temporal
evidence relies on the population's tracklet continuity: a same-kit body substituted on the target's tracklet with
no break flag is NOT detected by this layer (it inherits that tracker-level residual). The verdict uses the LOWER of the temporal and the current value (topk, support): past
evidence on a tracklet can never outvote what the box shows now (the population may have carried the tracklet onto
another body).

NUMBER (optional, support / veto only). A jersey-number reader (callable(image, box) -> str or None) may be supplied;
none exists in R2 (reader None: number UNKNOWN). A number is CONFIRMED after identity_number_min_reads agreeing
readings and no conflicting one (readings bounded to identity_history_max_samples per body; a body's readings are
cleared with its history). The target's number is read only on frames where the trusted bank accepted a sample. A confirmed different number
vetoes a candidate; an equal number never proves it.

VERDICT (counterfactual: what this evidence would say about the target this frame; never used by the tracker):
  * SUPPORTED for body B only if its topk (lower of temporal and current) >= reacq_appearance_floor, its support
    (lower of temporal and current) >= profile_support_min_fraction, kit (lower of temporal and current) >=
    lab_floor, no number veto, AND it beats every other body's topk (temporal or current, whichever is higher) by
    ambiguity_margin, AND every other body in view has a usable crop (a body that cannot be scored is an unknown
    competitor: UNSCORED_BODY_IN_VIEW caps the verdict at AMBIGUOUS);
  * AMBIGUOUS when the best body's evidence reaches appearance_floor but the above does not hold;
  * UNKNOWN otherwise (including no anchor, no temporal evidence). No single cue can make SUPPORTED.
OPEN WORLD (residual): the verdict is appearance evidence over the bodies in view. When the true target is not in
view, a never-seen same-kit look-alike whose appearance passes the floors has no rival and can be SUPPORTED; this
layer has no positional evidence (the tracker's gate does). It is evidence, never an identity decision.
With fewer than profile_consistency_k anchors, consistency is limited (one anchor: topk = best); the verdict records
n_anchors. The extractor is the tracker's own; it must be stateless (DINOv2 is), or the profile could alter the
tracker's features.
"""
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import population as P
from .appearance import ACCEPTED as _BANK_ACCEPTED
from .appearance import CONTAMINATION_FLAGS, AppearanceExtractor, AppearanceFeature, compare, embedding_similarity
from .config import TrackerConfig, sec_ge, sec_gt

SUPPORTED, AMBIGUOUS, UNKNOWN = "SUPPORTED", "AMBIGUOUS", "UNKNOWN"
ADDED, REDUNDANT, REFUSED, NOT_STORED = "ADDED", "REDUNDANT", "REFUSED", "NOT_STORED"
_BREAK_FLAGS = (P.MERGED_SUSPECTED, P.IN_MERGED_REGION, P.CONTESTED, P.SIZE_ANOMALY, P.DUPLICATE_ABSORBED,
                P.REASSOCIATED_AFTER_GAP)
_NOT_CLEAN_FLAGS = (P.PARTIAL, P.TOUCHES_IMAGE_EDGE)      # no temporal evidence from such a box (history kept)
NumberReader = Callable[[np.ndarray, Sequence[float]], Optional[str]]


@dataclass(frozen=True)
class Anchor:
    feature: AppearanceFeature
    rel_frame: int
    time_s: float
    aspect: float                       # box width / height (pose proxy, record only)
    source: str                         # SEED | TRUSTED_BANK

    def as_record(self) -> Dict[str, Any]:
        return {"rel_frame": self.rel_frame, "time_s": self.time_s, "aspect": self.aspect, "source": self.source,
                "view": UNKNOWN}


@dataclass(frozen=True)
class ProfileMatch:
    best: float
    topk: float
    support: float
    kit: Optional[float]
    n_anchors: int
    best_anchor: int

    def as_record(self) -> Dict[str, Any]:
        return {"best": self.best, "topk": self.topk, "support": self.support, "kit": self.kit,
                "n_anchors": self.n_anchors, "best_anchor": self.best_anchor}


def _confirmed(reads: Sequence[str], n_min: int) -> Optional[str]:
    """A number confirmed by >= n_min agreeing readings and no conflicting one (else None)."""
    vals = set(reads)
    if len(vals) == 1 and len(reads) >= n_min:
        return next(iter(vals))
    return None


class TargetProfile:
    """Diverse trusted reference anchors for the selected player, and their evidence on every tracked body."""

    def __init__(self, cfg: TrackerConfig, ext: AppearanceExtractor, number_reader: Optional[NumberReader] = None):
        self._cfg = cfg
        self._ext = ext
        self._reader = number_reader
        self.anchors: List[Anchor] = []
        self._last_offered: Optional[AppearanceFeature] = None   # the last bank sample offered (kept: no id reuse)
        self._hist: Dict[str, List[Tuple[float, ProfileMatch]]] = {}
        self._target_reads: List[str] = []
        self._reads: Dict[str, List[str]] = {}
        self.decisions: Dict[str, int] = {}
        self.last_features: Dict[int, AppearanceFeature] = {}

    # -- anchors -----------------------------------------------------------------------------------------------
    def offer(self, feat: AppearanceFeature, box: Sequence[float], rel_frame: int, t: float, source: str) -> str:
        """Offer a TRUSTED target sample (the caller guarantees trust: update() only offers bank-accepted samples)."""
        c = self._cfg
        if not feat.usable:
            return self._count(REFUSED)
        tol = c.get("embedding_norm_tolerance")
        if self.anchors:
            near = max(embedding_similarity(feat.embedding, a.feature.embedding, tol) for a in self.anchors)
            if near >= 1.0 - c.get("profile_anchor_min_novelty"):
                return self._count(REDUNDANT)
        w, h = box[2] - box[0], box[3] - box[1]
        self.anchors.append(Anchor(feat, rel_frame, t, w / h if h > 0 else 0.0, source))
        if len(self.anchors) > c.get("profile_max_anchors"):
            embs = np.stack([a.feature.embedding for a in self.anchors], 0)
            sims = embs @ embs.T
            np.fill_diagonal(sims, -np.inf)
            red = sims.max(axis=1)
            red[0] = -np.inf                                 # the seed anchor is never evicted
            i = int(np.argmax(red))
            del self.anchors[i]
            if i == len(self.anchors):
                return self._count(NOT_STORED)
        return self._count(ADDED)

    def _count(self, k: str) -> str:
        self.decisions[k] = self.decisions.get(k, 0) + 1
        return k

    def match(self, feat: AppearanceFeature) -> Optional[ProfileMatch]:
        c = self._cfg
        if not self.anchors or not feat.usable:
            return None
        comps, labs = [], []
        for a in self.anchors:
            s = compare(feat, a.feature, c)
            if s.composite is not None:
                comps.append(s.composite)
            if s.lab is not None:
                labs.append(s.lab)
        if not comps:
            return None
        order = sorted(comps, reverse=True)
        k = min(c.get("profile_consistency_k"), len(order))
        floor = c.get("appearance_floor")
        return ProfileMatch(best=order[0], topk=float(np.mean(order[:k])),
                            support=sum(1 for x in comps if x >= floor) / len(comps),
                            kit=float(np.median(labs)) if labs else None, n_anchors=len(comps),
                            best_anchor=int(np.argmax(comps)))

    # -- per frame ---------------------------------------------------------------------------------------------
    def update(self, res: P.FrameResult, image: np.ndarray, tf: Any, bank: Any) -> Dict[str, Any]:
        """tf: the tracker's TargetFrame for this frame; bank: the tracker's AppearanceBank (both read-only)."""
        c = self._cfg
        t = res.time_s
        if res.camera.continuity_broken:
            self._hist, self._reads = {}, {}
        live = {v.tracklet_id: v for v in res.tracklets}
        win = c.get("profile_temporal_window_s")
        for tid in list(self._hist):
            v = live.get(tid)
            if v is None or P.MERGE_LINEAGE in v.flags:
                self._hist.pop(tid, None)
                self._reads.pop(tid, None)
                continue
            self._hist[tid] = [x for x in self._hist[tid] if not sec_gt(t - x[0], win)]   # pruned every frame
        for tid in list(self._reads):
            if tid not in live:
                self._reads.pop(tid, None)
        bodies = [o for o in res.observations if o.is_body and P.ALTERNATIVE not in o.flags]
        rows = []
        self.last_features = {}                              # this frame's crops (diagnostic tools; overwritten)
        for o in bodies:
            others = [x.bbox for x in bodies if x.index != o.index]
            feat = self._ext.extract(image, o.bbox, others=others, merged_suspected=P.MERGED_SUSPECTED in o.flags)
            m = self.match(feat)
            self.last_features[o.index] = feat
            broken = any(f in o.flags for f in _BREAK_FLAGS)
            clean = m is not None and o.tracklet_id is not None and not broken and \
                not any(f in feat.quality.flags for f in CONTAMINATION_FLAGS) and \
                not any(f in o.flags for f in _NOT_CLEAN_FLAGS)
            if o.tracklet_id is not None and broken:
                self._hist.pop(o.tracklet_id, None)
                self._reads.pop(o.tracklet_id, None)
            if clean:
                self._hist.setdefault(o.tracklet_id, []).append((t, m))
                if self._reader is not None:
                    r = self._reader(image, o.bbox)
                    if r is not None:
                        rd = self._reads.setdefault(o.tracklet_id, [])
                        rd.append(r)
                        del rd[:-c.get("identity_history_max_samples")]
            h = self._hist.get(o.tracklet_id, []) if clean else []
            temporal = None
            if len(h) >= 2 and sec_ge(h[-1][0] - h[0][0], c.get("profile_temporal_min_s")):
                temporal = {"topk": float(np.median([x[1].topk for x in h])),
                            "support": float(np.median([x[1].support for x in h])),
                            "kit": float(np.median([x[1].kit for x in h if x[1].kit is not None]))
                            if any(x[1].kit is not None for x in h) else None,
                            "span_s": h[-1][0] - h[0][0], "frames": len(h)}
            rows.append({"index": o.index, "tracklet": o.tracklet_id, "bbox": list(o.bbox),
                         "match": None if m is None else m.as_record(), "clean": clean, "temporal": temporal,
                         "number": self._number_state(o.tracklet_id)})
        # anchors are offered AFTER this frame's bodies are scored (causal: a crop never matches itself)
        added = None
        b = (tf.decision or {}).get("bank") if isinstance(tf.decision, dict) else None
        if tf.identity_trusted and isinstance(b, dict) and b.get("status") == _BANK_ACCEPTED and bank.samples:
            smp = bank.samples[-1]                          # exactly the sample the trusted bank just accepted
            if smp is not self._last_offered:
                self._last_offered = smp
                added = self.offer(smp, tf.bbox, res.rel_frame, t, "SEED" if not self.anchors else "TRUSTED_BANK")
                if self._reader is not None and added in (ADDED, REDUNDANT, NOT_STORED):
                    r = self._reader(image, tf.bbox)
                    if r is not None:
                        self._target_reads.append(r)
                        del self._target_reads[:-c.get("identity_history_max_samples")]
        return {"rel_frame": res.rel_frame, "anchors": len(self.anchors), "anchor_offer": added,
                "candidates": rows, "verdict": self._verdict(rows)}

    def _number_state(self, tid: Optional[str]) -> str:
        if self._reader is None or tid is None:
            return UNKNOWN
        n_min = self._cfg.get("identity_number_min_reads")
        mine, theirs = _confirmed(self._target_reads, n_min), _confirmed(self._reads.get(tid, []), n_min)
        if mine is None or theirs is None:
            return UNKNOWN
        return "SAME" if mine == theirs else "DIFFERENT"

    def _verdict(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        c = self._cfg

        def level(r):
            vals = [x for x in ((r["temporal"] or {}).get("topk"), (r["match"] or {}).get("topk")) if x is not None]
            return max(vals) if vals else None
        def joint(r):
            # the lower of the temporal and the current evidence: the past never outvotes the present box
            t, m = r["temporal"], r["match"]
            kits = [x for x in (t["kit"], m["kit"]) if x is not None]
            return {"topk": min(t["topk"], m["topk"]), "support": min(t["support"], m["support"]),
                    "kit": min(kits) if len(kits) == 2 else None}
        scored = [r for r in rows if r["temporal"] is not None and r["match"] is not None]
        if not self.anchors or not scored:
            return {"status": UNKNOWN, "tracklet": None, "index": None, "reasons": ["NO_TEMPORAL_EVIDENCE"]}
        best = max(scored, key=lambda r: (joint(r)["topk"], -r["index"]))
        tv = joint(best)
        reasons = []
        if tv["topk"] < c.get("reacq_appearance_floor"):
            reasons.append("TOPK_BELOW_REACQ_FLOOR")
        if tv["support"] < c.get("profile_support_min_fraction"):
            reasons.append("INCONSISTENT_ACROSS_ANCHORS")
        if tv["kit"] is None or tv["kit"] < c.get("lab_floor"):
            reasons.append("KIT_INCOMPATIBLE")
        if best["number"] == "DIFFERENT":
            reasons.append("NUMBER_DIFFERENT")
        rival = [r["index"] for r in rows if r is not best and level(r) is not None
                 and level(r) > tv["topk"] - c.get("ambiguity_margin")]
        if rival:
            reasons.append("RIVAL_WITHIN_MARGIN")
        unscored = [r["index"] for r in rows if r is not best and r["match"] is None]
        if unscored:
            reasons.append("UNSCORED_BODY_IN_VIEW")           # an unknown competitor: cannot be ruled out
        if not reasons:
            status = SUPPORTED
        elif tv["topk"] >= c.get("appearance_floor"):
            status = AMBIGUOUS
        else:
            status = UNKNOWN
        return {"status": status, "tracklet": best["tracklet"], "index": best["index"], "reasons": reasons,
                "rivals": rival, "unscored": unscored, "n_anchors": len(self.anchors)}

    def state_sizes(self) -> Dict[str, int]:
        return {"profile_anchors": len(self.anchors), "profile_histories": len(self._hist),
                "profile_history_samples": sum(len(h) for h in self._hist.values())}

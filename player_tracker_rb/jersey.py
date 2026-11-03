# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2 DEV target jersey-number evidence instruction (Nathan, 2026-11-04 08:13): target-centric
#                 shirt-number evidence from a user-provided expected number; read only target / candidate crops; temporal
#                 aggregation, confidence and consensus per tracklet; UNKNOWN when unreadable; duplicates possible; a
#                 number may support or veto, never prove TARGET; one OCR reading never decides; aggregate only clean,
#                 geometrically valid crops; diagnostic only (stop before changing trust logic).
"""Target jersey-number evidence (NEW_R2, DIAGNOSTIC ONLY).

A read-only layer beside the target tracker. It NEVER changes the tracker (state, trust, bank, candidates,
reacquisition, handoff safety) and is not called by run_rb. The OCR model is NOT part of this package: a reader
callable(image, box) -> list of (text, confidence) is supplied by the caller (tools_rb/r2_jersey_ocr.py wraps a pinned
EasyOCR model in a separate environment); this module only decides which crops may be read and how readings combine.

CROPS. A body is read only when its observation is a CLEAN, GEOMETRICALLY VALID crop of one whole body: not
PARTIAL / TOUCHES_IMAGE_EDGE / OVERLAPS_OTHER / CONTAINS_OTHER / NESTED / CONTESTED / MERGED_SUSPECTED /
IN_MERGED_REGION / SIZE_ANOMALY / DUPLICATE_ABSORBED / ALTERNATIVE, on a tracklet, with a box at least
jersey_min_box_h_px tall (a pixel limit by necessity: the OCR model needs digit strokes of a minimum size). Only the
caller's chosen bodies are read (target-centric: the target's tracklet and candidates near it, never all players).

READINGS. A raw OCR result counts only if it is all digits, 1-2 characters, no leading zero, value 1-99, with OCR
confidence >= jersey_min_read_conf; several results on one crop that disagree make that crop UNREADABLE. Everything
else is UNREADABLE. Readings are kept per tracklet (bounded to identity_history_max_samples), cleared when the
tracklet ends, on a camera continuity break, or when its box shows a break flag (CONTESTED / merged / SIZE_ANOMALY /
DUPLICATE_ABSORBED / REASSOCIATED_AFTER_GAP: it may no longer be one body).

CONSENSUS per tracklet:
  * CONFIRMED n: >= identity_number_min_reads readings of n spanning >= jersey_confirm_span_s, and NO reading of any
    other number (one OCR reading never decides; any disagreement blocks confirmation);
  * CONFLICT: readings of two or more different numbers;
  * UNKNOWN otherwise (unreadable, too few, too short).
  Confidence reported = the number of agreeing readings, their span and the median OCR confidence (no single score).

EVIDENCE against the user-provided expected number (None -> always UNKNOWN):
  * SUPPORT: CONFIRMED and equal (support only: duplicate numbers exist -- a same number never proves the target);
  * VETO: CONFIRMED and different (evidence against; it would only ever WITHHOLD trust, never grant it);
  * UNKNOWN: anything else.
"""
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import population as P
from .config import TrackerConfig, sec_ge

CONFIRMED, CONFLICT, UNKNOWN = "CONFIRMED", "CONFLICT", "UNKNOWN"
SUPPORT, VETO = "SUPPORT", "VETO"
UNREADABLE = "UNREADABLE"
_UNCLEAN = (P.PARTIAL, P.TOUCHES_IMAGE_EDGE, P.OVERLAPS_OTHER, P.CONTAINS_OTHER, P.NESTED, P.CONTESTED,
            P.MERGED_SUSPECTED, P.IN_MERGED_REGION, P.SIZE_ANOMALY, P.DUPLICATE_ABSORBED, P.ALTERNATIVE)
_BREAK = (P.CONTESTED, P.MERGED_SUSPECTED, P.IN_MERGED_REGION, P.SIZE_ANOMALY, P.DUPLICATE_ABSORBED,
          P.REASSOCIATED_AFTER_GAP)
Reader = Callable[[np.ndarray, Sequence[float]], Sequence[Tuple[str, float]]]


def parse_reading(results: Sequence[Tuple[str, float]], min_conf: float) -> Optional[Tuple[int, float]]:
    """(number, confidence) of one crop's OCR results, or None (UNREADABLE). Disagreeing results -> None."""
    found = []
    for text, conf in results:
        t = "".join(ch for ch in str(text) if not ch.isspace())
        if not t or not t.isdigit() or len(t) > 2 or t[0] == "0" or conf is None or conf < min_conf:
            continue
        found.append((int(t), float(conf)))
    if not found or len({n for n, _ in found}) > 1:
        return None
    return found[0][0], max(c for _, c in found)


def clean_crop(o: P.Observation, cfg: TrackerConfig) -> bool:
    return o.is_body and o.tracklet_id is not None and not any(f in o.flags for f in _UNCLEAN) and \
        (o.bbox[3] - o.bbox[1]) >= cfg.get("jersey_min_box_h_px")


class JerseyEvidence:
    """Per-tracklet temporal jersey-number consensus, and its evidence against an expected number."""

    def __init__(self, cfg: TrackerConfig, expected_number: Optional[int] = None):
        if expected_number is not None and not (isinstance(expected_number, int) and not isinstance(expected_number, bool)
                                                and expected_number >= 1 and len(str(expected_number)) <= 2):
            raise ValueError("expected_number must be a shirt number 1-99 (1-2 digits) or None")
        self._cfg = cfg
        self.expected = expected_number
        self._reads: Dict[str, List[Tuple[float, int, float]]] = {}      # tracklet -> (t, number, conf)
        self.counts: Dict[str, int] = {"crops_offered": 0, "crops_clean": 0, "readable": 0}

    def update(self, res: P.FrameResult, image: np.ndarray, reader: Reader,
               tracklets: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        """Read the clean crops of the given tracklets (target-centric selection by the caller) and return the
        consensus of every one of them."""
        c = self._cfg
        if res.camera.continuity_broken:
            self._reads = {}
        live = {v.tracklet_id for v in res.tracklets}
        for tid in list(self._reads):
            if tid not in live:
                del self._reads[tid]
        out = {}
        want = set(tracklets)
        for o in res.observations:
            if o.tracklet_id not in want or not o.is_body:
                continue
            if any(f in o.flags for f in _BREAK):
                self._reads.pop(o.tracklet_id, None)
            self.counts["crops_offered"] += 1
            row: Dict[str, Any] = {"read": None}
            if clean_crop(o, c):
                self.counts["crops_clean"] += 1
                r = parse_reading(reader(image, o.bbox), c.get("jersey_min_read_conf"))
                row["read"] = UNREADABLE if r is None else r[0]
                if r is not None:
                    self.counts["readable"] += 1
                    rd = self._reads.setdefault(o.tracklet_id, [])
                    rd.append((res.time_s, r[0], r[1]))
                    del rd[:-c.get("identity_history_max_samples")]
            row.update(self.consensus(o.tracklet_id))
            out[o.tracklet_id] = row
        return out

    def consensus(self, tid: str) -> Dict[str, Any]:
        c = self._cfg
        rd = self._reads.get(tid, [])
        nums = sorted({n for _, n, _ in rd})
        if len(nums) > 1:
            st, num = CONFLICT, None
        elif nums and len(rd) >= c.get("identity_number_min_reads") and \
                sec_ge(rd[-1][0] - rd[0][0], c.get("jersey_confirm_span_s")):
            st, num = CONFIRMED, nums[0]
        else:
            st, num = UNKNOWN, None
        ev = UNKNOWN
        if st == CONFIRMED and self.expected is not None:
            ev = SUPPORT if num == self.expected else VETO
        return {"status": st, "number": num, "numbers_seen": nums, "reads": len(rd),
                "span_s": (rd[-1][0] - rd[0][0]) if rd else 0.0,
                "median_conf": float(np.median([x[2] for x in rd])) if rd else None, "evidence": ev}

    def state_sizes(self) -> Dict[str, int]:
        return {"jersey_tracklets": len(self._reads), "jersey_reads": sum(len(v) for v in self._reads.values())}

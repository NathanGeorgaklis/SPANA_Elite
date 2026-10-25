# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R0/R0_TRACKER_ACCEPTANCE_TARGETS.md Part C (R2 benchmark design); R2 instruction (Nathan,
#                 2026-10-26 07:53); Nathan's answers 2026-10-26 (Task 3R: read time ranges only; one annotator;
#                 5 Hz boxes + every frame near events)
"""R2 benchmark pre-registration: allowed sources, excluded source-time ranges and the DEV / EVAL split.

Committed BEFORE any R2 footage is viewed for clip selection. Every R2 clip must pass ``check_window`` and takes its
split from ``split_of`` (it is never chosen by hand). Evaluation code only; never imported by player_tracker_rb.

    python3 tools_rb/r2_protocol.py            # prints the block table and exclusions (the pre-registration record)
"""
import hashlib
import json
import math
import sys
from typing import Dict, List, Tuple

PROTOCOL_VERSION = "r2-protocol-1"

# Allowed sources (R0 Part C C1; hashes verified on the Mac 2026-10-26).
SOURCES: Dict[str, Dict] = {
    "SYO_MAR": {"file": "videoplayback.mp4",
                "sha256": "09aecc9b1ca2f64203864f25705618b03f35fa85cafd9610c51bdeecd585bfec",
                "fps": 30.0, "duration_s": 8525.933333, "match": "SYO v MAR", "camera": "multi-camera broadcast"},
    "MCFC_MIT": {"file": "videoplayback (1).mp4",
                 "sha256": "e8bd4d5fd8105a862ad8388f09473732b47481d6bda89ce10480ca18c810c519",
                 "fps": 60.0, "duration_s": 8840.0, "match": "MCFC v MIT", "camera": "single low camera, floodlit"},
}

# Never opened, never a candidate (R0 Part C): videoplayback (2).mp4, anything v2_esfc_bcfc. Not in SOURCES.
HOLDOUT_NAMES = ("videoplayback (2).mp4", "v2_esfc_bcfc")

# Exclusion buffer around every excluded window: the same play, players and camera set-up continue for a while,
# so footage just before / after an excluded window is excluded too.
EXCLUSION_BUFFER_S = 60.0
# GENERAL_BENCHMARK_V2 window lengths are not recorded in the surviving docs: assume up to this length (conservative).
GB_V2_ASSUMED_LENGTH_S = 120.0

# (source, start_s, end_s, reason). Task 3R ranges: from ~/Downloads/_spana_task3r_work/extract_log*.txt, read
# 2026-10-26 with Nathan's approval (time-range lines only; start = ss, end = ss + n / r).
EXCLUDED_WINDOWS: List[Tuple[str, float, float, str]] = [
    ("MCFC_MIT", 3504.0, 3504.0 + 540 / 60, "Task 3R R01"),
    ("MCFC_MIT", 3153.0, 3153.0 + 540 / 60, "Task 3R R02"),
    ("MCFC_MIT", 3328.0, 3328.0 + 600 / 60, "Task 3R R03"),
    ("MCFC_MIT", 3124.0, 3124.0 + 660 / 60, "Task 3R R10"),
    ("SYO_MAR", 5916.933333, 5916.933333 + 330 / 30, "Task 3R R04"),
    ("SYO_MAR", 5665.933333, 5665.933333 + 300 / 30, "Task 3R R05"),
    ("SYO_MAR", 2400.0, 2400.0 + GB_V2_ASSUMED_LENGTH_S, "GENERAL_BENCHMARK_V2 SYO/MAR 2400 s"),
    ("SYO_MAR", 6600.0, 6600.0 + GB_V2_ASSUMED_LENGTH_S, "GENERAL_BENCHMARK_V2 SYO/MAR 6600 s"),
    ("MCFC_MIT", 2400.0, 2400.0 + GB_V2_ASSUMED_LENGTH_S, "GENERAL_BENCHMARK_V2 MCFC/MIT 2400 s"),
    ("MCFC_MIT", 4200.0, 4200.0 + GB_V2_ASSUMED_LENGTH_S, "GENERAL_BENCHMARK_V2 MCFC/MIT 4200 s"),
    ("SYO_MAR", 3036.0, 3048.0, "GOLD_DUEL_01 (development clip; never R2 evaluation truth)"),
]

# DEV / EVAL split by source-time block (R0 Part C: 50/50 by source-time block, fixed before labelling). The split of
# a block is decided by a hash, not by anyone looking at the footage.
BLOCK_S = 300.0
SPLIT_SALT = "SPANA-R2-split-v1"

# Clip rules.
CLIP_MIN_S = 5.0
CLIP_MAX_S = 15.0


def block_of(source: str, t: float) -> int:
    return int(math.floor(t / BLOCK_S))


def split_of_block(source: str, block: int) -> str:
    h = hashlib.sha256(f"{SPLIT_SALT}:{SOURCES[source]['sha256']}:{block}".encode()).digest()
    return "EVAL" if h[0] % 2 else "DEV"


def split_of(source: str, start_s: float, end_s: float) -> str:
    b0, b1 = block_of(source, start_s), block_of(source, end_s - 1e-9)
    if b0 != b1:
        raise ValueError(f"window {start_s}-{end_s} crosses a split block boundary")
    return split_of_block(source, b0)


def excluded_overlaps(source: str, start_s: float, end_s: float) -> List[str]:
    return [why for (s, a, b, why) in EXCLUDED_WINDOWS
            if s == source and start_s < b + EXCLUSION_BUFFER_S and end_s > a - EXCLUSION_BUFFER_S]


def check_window(source: str, start_s: float, end_s: float) -> str:
    """Validate a candidate clip window; returns its split. Raises ValueError (fail closed) on any rule breach."""
    if source not in SOURCES:
        raise ValueError(f"source {source!r} is not an allowed R2 source")
    if not (0.0 <= start_s < end_s <= SOURCES[source]["duration_s"]):
        raise ValueError("window outside the source")
    if not (CLIP_MIN_S - 1e-9 <= end_s - start_s <= CLIP_MAX_S + 1e-9):
        raise ValueError(f"clip length {end_s - start_s:.3f} s outside [{CLIP_MIN_S}, {CLIP_MAX_S}]")
    bad = excluded_overlaps(source, start_s, end_s)
    if bad:
        raise ValueError(f"window overlaps excluded material: {bad}")
    return split_of(source, start_s, end_s)


def block_table() -> List[Dict]:
    rows = []
    for src, meta in SOURCES.items():
        n = int(math.ceil(meta["duration_s"] / BLOCK_S))
        for b in range(n):
            rows.append({"source": src, "block": b, "start_s": b * BLOCK_S,
                         "end_s": min((b + 1) * BLOCK_S, meta["duration_s"]), "split": split_of_block(src, b)})
    return rows


def record() -> Dict:
    return {"protocol_version": PROTOCOL_VERSION, "sources": SOURCES, "holdout_never_opened": list(HOLDOUT_NAMES),
            "exclusion_buffer_s": EXCLUSION_BUFFER_S, "gb_v2_assumed_length_s": GB_V2_ASSUMED_LENGTH_S,
            "excluded_windows": [{"source": s, "start_s": a, "end_s": b, "reason": w}
                                 for (s, a, b, w) in EXCLUDED_WINDOWS],
            "block_s": BLOCK_S, "split_salt": SPLIT_SALT, "clip_length_s": [CLIP_MIN_S, CLIP_MAX_S],
            "blocks": block_table()}


if __name__ == "__main__":
    rec = record()
    json.dump(rec, sys.stdout, indent=1)
    print()
    for src in SOURCES:
        rows = [r for r in rec["blocks"] if r["source"] == src]
        print(src, "blocks", len(rows), "EVAL", sum(r["split"] == "EVAL" for r in rows), file=sys.stderr)

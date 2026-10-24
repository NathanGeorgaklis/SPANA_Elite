#!/usr/bin/env python3
"""
source_registry.py -- the single source of truth for reading, writing, and
classifying entries in dataset/metadata/source_videos.json (task item #4).

This module owns the REGISTRY ENTRY SHAPE and the INDEPENDENCE CLASSIFIER --
both `clip_extractor.py` (the original, lower-level register-source command)
and `ingest_source.py` (the new, richer front door built for this task) call
into this module rather than each maintaining their own copy of the schema
or the duplicate-detection logic, so the registry can never drift into two
incompatible shapes depending on which tool last wrote to it.

ENTRY SHAPE (one dict per source_video_id):
    source_video_id           str, uuid4 hex, stable identity for this file
    match_id                  str, the leakage-relevant grouping unit
    original_filename         str, the file's basename at ingestion time (a
                               filename is NEVER used to establish identity or
                               independence -- see classify_independence())
    path                      str, absolute path on disk (internal use --
                               clip_extractor.py needs this to extract clips)
    content_hash              str, SHA-256 of the raw file bytes
    file_size_bytes           int
    duration_s                float   (ffprobe)
    fps                       float   (ffprobe -- authoritative, see item #7/#8)
    frame_count               int     (ffprobe)
    width, height             int     (ffprobe)
    codec                     str or null (ffprobe, best-effort)
    camera_type               one of schema.taxonomy.SOURCE_TYPES
    camera_type_confidence    one of schema.taxonomy.SOURCE_CAMERA_TYPE_CONFIDENCE
    lighting                  one of schema.taxonomy.SOURCE_LIGHTING
    source_type               one of schema.taxonomy.SOURCE_TYPES (see note
                               in schema/taxonomy.py -- this task's brief lists
                               camera_type and source_type as two separate
                               registry fields; this project treats them as
                               the SAME classification carried in two fields
                               for spec-literal compliance, rather than
                               inventing a second, unrelated taxonomy no part
                               of the brief actually defines)
    competition                str or null
    teams                      list[str] or null (whatever the user supplies --
                               never inferred, never a real name assumed)
    match_date                 str (YYYY-MM-DD) or null
    source_independence_status one of schema.taxonomy.SOURCE_INDEPENDENCE_STATUSES
    source_independence_evidence list[str] -- human-readable reasons for the
                               status above (task item #5: never assert
                               independence without stating the evidence)
    provenance                 dict: {registered_by, tool_version, notes}
    registered_at, updated_at  ISO-8601 timestamps

DUPLICATE / INDEPENDENCE DETECTION -- documented limitations (task item #6):
  - EXACT duplicate (byte-identical file): caught reliably via content_hash.
  - RENAMED duplicate (same bytes, different filename): also caught via
    content_hash -- hashing never looks at the filename at all.
  - RE-ENCODED duplicate (same footage, different container/bitrate/codec so
    the bytes differ): NOT reliably detectable by this module. We flag a
    suspicious case (same claimed match_id with a different hash, or a very
    close duration/resolution/fps match with a different hash) as
    POSSIBLE_DUPLICATE for a human to look at, but we do not claim to
    detect this automatically -- doing so properly needs perceptual/frame
    hashing, which is out of scope for this infrastructure task.
  - PARTIAL overlap (e.g. the same passage of play appears inside two
    otherwise-different recordings): NOT detectable by this module at all.
    No content-level check here inspects video frames or timestamps across
    files. This is stated as a known gap, not silently ignored.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

TOOLS_DIR = Path(__file__).resolve().parent
DATASET_ROOT = TOOLS_DIR.parent
sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(DATASET_ROOT))

from schema import taxonomy   # noqa: E402

REGISTRY_PATH = DATASET_ROOT / "dataset" / "metadata" / "source_videos.json"

# Tolerances for the metadata-similarity half of duplicate detection --
# deliberately loose (a re-encode can shave frames or round duration) but
# tight enough that two genuinely different matches essentially never
# collide by chance on all three at once.
DURATION_TOLERANCE_S = 1.0
FPS_TOLERANCE = 0.5


class SourceRegistryError(Exception):
    pass


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def probe_codec(path: str) -> Optional[str]:
    """Best-effort codec_name via a standalone ffprobe call -- kept separate
    from time_utils.probe_video() (which this task deliberately does not
    modify, since it is core, already-verified infrastructure and codec is
    explicitly 'if useful' per the task brief, not load-bearing for frame/fps
    safety). Returns None on any failure rather than raising -- a missing
    codec name must never block registering an otherwise-valid source."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=30, check=True,
        )
        data = json.loads(out.stdout)
        streams = data.get("streams") or []
        return streams[0].get("codec_name") if streams else None
    except Exception:
        return None


def sha256_of_file(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def load_registry(path: Path = REGISTRY_PATH) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_registry(registry: dict, path: Path = REGISTRY_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, indent=2, sort_keys=True))


def find_by_path(registry: dict, abs_path: str) -> Optional[str]:
    for sv_id, entry in registry.items():
        if entry.get("path") == abs_path:
            return sv_id
    return None


def find_by_hash(registry: dict, content_hash: str) -> list[str]:
    # Checks both the current field name (content_hash) and the legacy name
    # (file_sha256, used by entries clip_extractor.py registered before this
    # task) so hash lookups are correct regardless of which tool wrote an
    # entry -- ingest_source.py writes both fields going forward specifically
    # so this dual check is only needed for pre-existing entries.
    return [sv_id for sv_id, e in registry.items()
            if content_hash in (e.get("content_hash"), e.get("file_sha256"))]


def _metadata_similar(a: dict, b: dict) -> bool:
    if a.get("duration_s") is None or b.get("duration_s") is None:
        return False
    if abs(a["duration_s"] - b["duration_s"]) > DURATION_TOLERANCE_S:
        return False
    if a.get("fps") is not None and b.get("fps") is not None and abs(a["fps"] - b["fps"]) > FPS_TOLERANCE:
        return False
    if a.get("width") != b.get("width") or a.get("height") != b.get("height"):
        return False
    return True


def classify_independence(new_probe: dict, content_hash: str, claimed_match_id: str,
                           registry: dict) -> tuple[str, list[str], list[str]]:
    """Returns (status, evidence, related_source_video_ids).

    new_probe: dict with at least duration_s/fps/width/height (real, ffprobe-
    derived values -- never caller-claimed ones; see item #7/#8).
    claimed_match_id: the match_id the caller wants to register this source
    under (may be a brand-new id or one that already exists in the registry).
    """
    evidence = []
    related = []

    exact_hash_matches = find_by_hash(registry, content_hash)
    if exact_hash_matches:
        evidence.append(
            f"content_hash is byte-identical to already-registered source_video_id(s) "
            f"{exact_hash_matches} -- this is the SAME underlying file (exact or renamed "
            f"duplicate), not a new independent source."
        )
        return "SAME_SOURCE", evidence, exact_hash_matches

    # Same claimed match_id as an existing entry, but different file content:
    # this is the important nuance task item #1 calls out -- two different
    # camera angles (or two different uploads) of the SAME real-world match
    # are still the SAME match for leakage purposes, even though the files
    # themselves are genuinely different bytes.
    same_match_entries = [sv_id for sv_id, e in registry.items() if e.get("match_id") == claimed_match_id]
    if same_match_entries:
        evidence.append(
            f"match_id={claimed_match_id!r} is already used by source_video_id(s) "
            f"{same_match_entries} -- registering under the SAME match_id deliberately "
            f"(e.g. a second camera angle of a match already on file). This file's content "
            f"is NOT byte-identical to those, so this is not a duplicate upload, but it is "
            f"also not a NEW independent match -- it joins the existing one."
        )
        return "SAME_SOURCE", evidence, same_match_entries

    metadata_similar_to = [sv_id for sv_id, e in registry.items() if _metadata_similar(new_probe, e)]
    if metadata_similar_to:
        evidence.append(
            f"duration/resolution/fps closely match already-registered source_video_id(s) "
            f"{metadata_similar_to} (within {DURATION_TOLERANCE_S}s / {FPS_TOLERANCE}fps / "
            f"exact resolution) but content_hash differs -- POSSIBLE re-encoded duplicate or "
            f"POSSIBLE genuinely different match that happens to share technical specs. This "
            f"module cannot distinguish those two cases (see module docstring's documented "
            f"limitation on re-encoded-duplicate detection); a human should look at both "
            f"before treating this as a new independent match."
        )
        return "POSSIBLE_DUPLICATE", evidence, metadata_similar_to

    evidence.append(
        f"content_hash does not match any registered source; no existing entry shares this "
        f"match_id; duration/resolution/fps do not closely match any existing entry. No "
        f"evidence links this file to anything already registered."
    )
    return "CONFIRMED_INDEPENDENT", evidence, []


def fps_bucket(fps: Optional[float]) -> str:
    if fps is None:
        return "UNKNOWN"
    for lo, hi, label in [(23.5, 24.5, "24fps"), (24.5, 25.5, "25fps"), (29.5, 30.5, "30fps"),
                           (49.5, 50.5, "50fps"), (59.5, 60.5, "60fps")]:
        if lo <= fps < hi:
            return label
    return f"{fps:g}fps (other)"


def resolution_bucket(width: Optional[int], height: Optional[int]) -> str:
    if not width or not height:
        return "UNKNOWN"
    long_edge = max(width, height)
    if long_edge >= 3800:
        return "2160p (4K)"
    if long_edge >= 1900:
        return "1080p"
    if long_edge >= 1260:
        return "720p"
    return f"{width}x{height} (other)"


def new_source_video_id() -> str:
    return uuid.uuid4().hex


# --- task item #15: incremental diversity VALUE classification ---------
#
# classify_independence() (above) answers "is this a new match at all?".
# classify_diversity_value() answers a different question: "how much new
# generalisation-relevant diversity would accepting this candidate add?".
# Per the task brief ("DO NOT FORCE PERFECT DIVERSITY -- do not block
# ingestion for resembling an existing source"), this is PURELY INFORMATIONAL:
# it is never consulted by classify_independence() and it never raises or
# refuses anything. A LOW_DIVERSITY_VALUE candidate is still registered if
# the operator proceeds; a genuinely new match still counts toward the raw
# match-count target even at LOW_DIVERSITY_VALUE.
#
# These target lists intentionally mirror source_diversity_report.py's
# TARGET_CAMERA_TYPES / TARGET_LIGHTING / TARGET_KIT_CONTRAST /
# TARGET_RESOLUTIONS (kept as separate constants here, rather than importing
# that module, to avoid a circular import -- source_diversity_report.py
# already imports this module). If the acquisition priorities in one file
# change, update the other to match.
DIVERSITY_TARGET_CAMERA_TYPES = ["BROADCAST", "FIXED_CAMERA", "AUTOMATED_CAMERA", "MOBILE"]
DIVERSITY_TARGET_LIGHTING = ["DAYLIGHT", "FLOODLIT", "MIXED"]
DIVERSITY_TARGET_KIT_CONTRAST = ["LIGHT_VS_DARK", "SIMILAR_COLOURS", "CONTRASTING_COLOURS", "STRIPES_OR_PATTERN"]
DIVERSITY_TARGET_RESOLUTIONS = ["720p", "1080p"]


def _confirmed_match_dimension_sets(registry: dict) -> dict:
    """Per-dimension sets of values already represented among
    CONFIRMED_INDEPENDENT sources. (Using source-video level rather than
    collapsing to match level first is safe for this purpose -- a SAME_SOURCE
    entry always shares its match_id with something already CONFIRMED, so it
    can never introduce a dimension value that match-level collapsing would
    have missed.)"""
    dims = {"camera_type": set(), "lighting": set(), "fps_bucket": set(),
            "resolution_bucket": set(), "kit_contrast": set(), "competition": set()}
    for e in registry.values():
        if e.get("source_independence_status") != "CONFIRMED_INDEPENDENT":
            continue
        dims["camera_type"].add(e.get("camera_type", "UNKNOWN"))
        dims["lighting"].add(e.get("lighting", "UNKNOWN"))
        dims["fps_bucket"].add(fps_bucket(e.get("fps")))
        dims["resolution_bucket"].add(resolution_bucket(e.get("width"), e.get("height")))
        dims["kit_contrast"].add(e.get("kit_contrast", "UNKNOWN"))
        dims["competition"].add(e.get("competition") or "UNKNOWN")
    return dims


def classify_diversity_value(candidate: dict, registry: dict) -> tuple[str, list[str]]:
    """candidate: dict with camera_type, lighting, fps, width, height,
    kit_contrast, competition -- the same fields a registry entry carries
    (for a not-yet-registered file, pass the ffprobe'd + operator-confirmed
    values that ingestion would write).

    Returns (value_level, explanation_lines), value_level one of
    HIGH_DIVERSITY_VALUE / MEDIUM_DIVERSITY_VALUE / LOW_DIVERSITY_VALUE.

    This is deliberately simple (task brief: "do not over-engineer scoring"
    applies here just as much as to annotation_queue.py): count how many
    actively-targeted dimension gaps this candidate would close, plus whether
    it adds any other genuinely new value outside the active target list.
    UNKNOWN/OTHER candidate values are never counted as "adding" diversity --
    an unclassified camera type tells us nothing.
    """
    existing = _confirmed_match_dimension_sets(registry)
    cand_fps_bucket = fps_bucket(candidate.get("fps"))
    cand_res_bucket = resolution_bucket(candidate.get("width"), candidate.get("height"))

    checks = [
        ("camera_type", candidate.get("camera_type", "UNKNOWN"), DIVERSITY_TARGET_CAMERA_TYPES),
        ("lighting", candidate.get("lighting", "UNKNOWN"), DIVERSITY_TARGET_LIGHTING),
        ("kit_contrast", candidate.get("kit_contrast", "UNKNOWN"), DIVERSITY_TARGET_KIT_CONTRAST),
        ("resolution_bucket", cand_res_bucket, DIVERSITY_TARGET_RESOLUTIONS),
        ("fps_bucket", cand_fps_bucket, None),   # no fixed target list -- any genuinely new bucket counts
        ("competition", candidate.get("competition") or "UNKNOWN", None),
    ]

    new_target_dims, new_other_dims, repeated_dims = [], [], []
    for dim, value, target_list in checks:
        if value in ("UNKNOWN", None, "OTHER"):
            continue
        if value in existing.get(dim, set()):
            repeated_dims.append((dim, value))
        elif target_list is None or value in target_list:
            new_target_dims.append((dim, value))
        else:
            new_other_dims.append((dim, value))

    n_confirmed = sum(1 for e in registry.values() if e.get("source_independence_status") == "CONFIRMED_INDEPENDENT")

    explanation = []
    if n_confirmed == 0:
        level = "HIGH_DIVERSITY_VALUE"
        explanation.append("This would be the first CONFIRMED_INDEPENDENT source in the registry -- any real "
                            "diversity metadata it carries is automatically new.")
    elif len(new_target_dims) >= 2:
        level = "HIGH_DIVERSITY_VALUE"
        explanation.append("Closes " + str(len(new_target_dims)) + " currently-open priority gap(s): "
                            + ", ".join(f"{d}={v}" for d, v in new_target_dims) + ".")
    elif len(new_target_dims) == 1:
        level = "MEDIUM_DIVERSITY_VALUE"
        explanation.append(f"Closes 1 currently-open priority gap: {new_target_dims[0][0]}={new_target_dims[0][1]}.")
    elif new_other_dims:
        level = "MEDIUM_DIVERSITY_VALUE"
        explanation.append("Does not close a currently-tracked priority gap, but adds a genuinely new value "
                            "outside the active target list: " + ", ".join(f"{d}={v}" for d, v in new_other_dims) + ".")
    else:
        level = "LOW_DIVERSITY_VALUE"
        overlap = ", ".join(f"{d}={v}" for d, v in repeated_dims) if repeated_dims else "no classifiable dimensions"
        explanation.append(f"Every classifiable dimension on this candidate ({overlap}) already exists among "
                            f"confirmed-independent sources.")

    if repeated_dims and level != "LOW_DIVERSITY_VALUE":
        explanation.append("Also overlaps existing sources on: " + ", ".join(f"{d}={v}" for d, v in repeated_dims) + ".")

    explanation.append("Informational only, per task item #15: this still counts toward the raw match-count "
                        "target regardless of level, and is NEVER a reason to refuse ingestion.")

    return level, explanation

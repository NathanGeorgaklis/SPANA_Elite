# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: player_tracker_rb/output_schema.py PROVENANCE_FIELDS (R1-A contract); R1/R1_DESIGN_DECISIONS.md
#                 D18, D20, D22; R1-B instruction (Nathan, 2026-10-24: provenance fields, fail closed)
"""
player_tracker_rb.provenance: run provenance for the R1 tracker.

Two records are produced:

1. `contract_provenance(...)`: exactly the R1-A contract block `summary.v2.provenance` (PROVENANCE_FIELDS).
   It is validated against output_schema before it is returned.

2. `video_provenance_sidecar(...)`: the fuller R1-B record, written next to the tracking JSON as
   `{video_basename}_rb_video_provenance.json`. It holds what the contract block has no slot for: frame
   dimensions, file size, decoder backend and OpenCV version, seek mode, the frame-index and time rules, the
   frame-fingerprint method and digests, and the end-of-run source re-verification. Keeping these in a sidecar
   leaves the R1-A contract unchanged.

Fail closed: if the git commit cannot be determined, or the record does not validate, a ProvenanceError is raised.
A dirty working tree is recorded (git_dirty = True), not hidden.

Determinism: `deterministic_digest(record)` hashes the record without its wall-clock fields (run_started_utc,
and the absolute path, which depends on where the project is checked out). Two runs on the same inputs with the
same code and config must give the same digest.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import subprocess
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import __version__ as TRACKER_VERSION
from . import output_schema as S
from .config import TrackerConfig
from .video_io import SequentialVideoReader, fingerprint_method

SIDECAR_KIND = "spana_rb_video_provenance"
SIDECAR_SUFFIX = "_rb_video_provenance.json"
NONDETERMINISTIC_KEYS = ("run_started_utc", "abs_path", "sidecar_written_utc")


class ProvenanceError(RuntimeError):
    """Required provenance could not be recorded; the run must not continue."""


PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def git_state(repo_root: str = PACKAGE_ROOT) -> Tuple[str, bool]:
    """(HEAD commit, dirty?) of the source tree that holds this package. Raises ProvenanceError if unknown."""
    try:
        head = subprocess.run(["git", "-C", repo_root, "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
        status = subprocess.run(["git", "-C", repo_root, "status", "--porcelain", "--untracked-files=normal"],
                                capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as e:
        raise ProvenanceError(f"cannot determine the git commit of {repo_root}: {e}") from e
    if not (head and all(c in "0123456789abcdef" for c in head)):
        raise ProvenanceError(f"unexpected git HEAD {head!r}")
    return head, bool(status.strip())


def utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def validate_seed_bbox(seed_bbox: Sequence[float], width: int, height: int) -> List[float]:
    if not (isinstance(seed_bbox, (list, tuple)) and len(seed_bbox) == 4):
        raise ProvenanceError(f"seed_bbox must be [x1, y1, x2, y2], got {seed_bbox!r}")
    try:
        x1, y1, x2, y2 = (float(v) for v in seed_bbox)
    except (TypeError, ValueError) as e:
        raise ProvenanceError(f"seed_bbox must be numeric, got {seed_bbox!r}") from e
    if not (x1 < x2 and y1 < y2):
        raise ProvenanceError(f"seed_bbox must have x1 < x2 and y1 < y2, got {seed_bbox!r}")
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        raise ProvenanceError(f"seed_bbox {seed_bbox!r} lies outside the {width}x{height} frame")
    return [x1, y1, x2, y2]


def contract_provenance(reader: SequentialVideoReader, seed_bbox: Sequence[float], cfg: TrackerConfig,
                        models: Optional[List[Dict[str, Any]]] = None,
                        git: Callable[[], Tuple[str, bool]] = git_state,
                        clock: Callable[[], str] = utc_now_iso) -> Dict[str, Any]:
    """The summary.v2.provenance block, exactly as the R1-A contract defines it. Validated before return."""
    info = reader.info
    if info is None:
        raise ProvenanceError("reader is not open")
    box = validate_seed_bbox(seed_bbox, info.frame_width, info.frame_height)
    commit, dirty = git()
    models = list(models or [])
    for m in models:
        for k in ("role", "filename", "sha256", "status"):
            if not m.get(k):
                raise ProvenanceError(f"model record missing {k!r}: {m!r}")
    rec = {
        "video_path": info.path,
        "video_sha256": info.sha256,
        "fps": info.fps,
        "frame_count": info.frame_count_reported,
        "decode_mode": S.DECODE_MODE,
        "seed_frame": info.start_frame,
        "seed_bbox": [round(v, cfg.get("box_round_dp")) for v in box],
        "models": models,
        "config_hash": cfg.config_hash(),
        "git_commit": commit,
        "git_dirty": dirty,
        "schema_version": S.V2_SCHEMA_VERSION,
        "tracker_version": TRACKER_VERSION,
        "run_started_utc": clock(),
    }
    errors: List[str] = []
    S._check_fields(S.PROVENANCE_FIELDS, rec, "provenance", errors)
    if errors:
        raise ProvenanceError("; ".join(errors))
    return rec


def video_provenance_sidecar(reader: SequentialVideoReader, contract: Dict[str, Any], cfg: TrackerConfig,
                             frame_thumbs: Optional[List[str]] = None) -> Dict[str, Any]:
    """The fuller R1-B provenance record. Call after the frames have been read and the reader closed (verified)."""
    from .video_io import thumbs_digest
    info = reader.info
    if reader.source_verified_at_close is not True:
        raise ProvenanceError("source was not re-verified at close; refusing to write provenance")
    align = reader.alignment_summary()
    if frame_thumbs is not None and len(frame_thumbs) != align["processed_frames"]:
        raise ProvenanceError("frame_thumbs length does not match processed_frames")
    return {
        "kind": SIDECAR_KIND,
        "tracker": S.TRACKER_NAME,
        "tracker_version": TRACKER_VERSION,
        "schema_version": S.V2_SCHEMA_VERSION,
        "config_registry_version": cfg.as_record()["registry_version"],
        "config_hash": cfg.config_hash(),
        "contract_provenance": contract,
        "video": info.as_record(),
        "frame_alignment": dict(align, **{
            "fingerprint_method": fingerprint_method(cfg),
            "frame_thumbs_sha256": thumbs_digest(frame_thumbs) if frame_thumbs is not None else None,
        }),
        "source_reverified_at_close": True,
        "resume_supported": False,
    }


def _strip_nondeterministic(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_nondeterministic(v) for k, v in obj.items() if k not in NONDETERMINISTIC_KEYS}
    if isinstance(obj, list):
        return [_strip_nondeterministic(v) for v in obj]
    return obj


def deterministic_digest(record: Dict[str, Any]) -> str:
    """SHA-256 of the canonical record without wall-clock / checkout-location fields."""
    text = json.dumps(_strip_nondeterministic(record), sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False)
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def sidecar_path(out_dir: str, video_path: str) -> str:
    base = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(out_dir, base + SIDECAR_SUFFIX)


def write_sidecar(record: Dict[str, Any], path: str) -> str:
    """Write canonical JSON (refuses to overwrite an existing file). Returns the file's SHA-256."""
    if os.path.exists(path):
        raise ProvenanceError(f"refusing to overwrite existing provenance file {path}")
    text = S.dumps_canonical(record)
    with open(path, "x", encoding="ascii") as f:
        f.write(text)
    return hashlib.sha256(text.encode("ascii")).hexdigest()

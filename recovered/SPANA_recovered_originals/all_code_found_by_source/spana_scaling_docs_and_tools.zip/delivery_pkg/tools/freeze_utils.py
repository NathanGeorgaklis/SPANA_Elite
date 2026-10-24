"""
freeze_utils.py -- SHA-256 freeze-verification helpers, shared by
feature_exporter.py, pipeline_benchmark.py, and the final delivery check for
this task (audit item #46: "Before and after run freeze verification").

This reuses the exact methodology already established earlier in this
session (see PROTECTED_FULL_PIPELINE_CHECKSUMS.sha256, a standard
`sha256sum`-format manifest of every football_analysis/*.py and
player_tracker/*.py file) rather than inventing a second one. Nothing here
reads a file for any purpose other than hashing its bytes.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
PROTECTED_DIRS = ("football_analysis", "player_tracker")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_production_checksum_manifest(repo_root: Path = REPO_ROOT) -> List[Tuple[str, str]]:
    """Returns a sorted list of (relative_path, sha256_hex) for every .py file
    under football_analysis/ and player_tracker/. Sorted by path so the
    manifest (and anything derived from it) is deterministic regardless of
    filesystem iteration order."""
    entries = []
    for d in PROTECTED_DIRS:
        base = repo_root / d
        if not base.exists():
            continue
        for p in sorted(base.rglob("*.py")):
            rel = str(p.relative_to(repo_root))
            entries.append((rel, _sha256_file(p)))
    entries.sort(key=lambda t: t[0])
    return entries


def manifest_to_sha256sum_text(entries: List[Tuple[str, str]]) -> str:
    """Renders the manifest in standard `sha256sum` output format (hash,
    two spaces, path) -- verifiable with `sha256sum -c` against a saved
    baseline, exactly like PROTECTED_FULL_PIPELINE_CHECKSUMS.sha256."""
    return "".join(f"{h}  {p}\n" for p, h in entries)


def combined_production_sha256(repo_root: Path = REPO_ROOT) -> str:
    """A single hash summarising the whole manifest -- this is the value
    stamped into every exported feature record's
    pipeline_provenance.production_files_sha256 (audit item #44). Two
    exports with the same value are guaranteed to have run against
    byte-identical production code; a differing value is an immediate,
    checkable warning sign that a comparison between them may not be valid."""
    entries = compute_production_checksum_manifest(repo_root)
    combined = hashlib.sha256()
    for p, h in entries:
        combined.update(f"{h}  {p}\n".encode("utf-8"))
    return combined.hexdigest()


def load_baseline_manifest(baseline_path: str) -> Dict[str, str]:
    """Parses a standard sha256sum-format file into {relative_path: hash}."""
    out = {}
    for line in Path(baseline_path).read_text().splitlines():
        line = line.rstrip("\n")
        if not line.strip():
            continue
        h, _, p = line.partition("  ")
        out[p] = h
    return out


def diff_against_baseline(baseline_path: str, repo_root: Path = REPO_ROOT) -> Dict[str, list]:
    """Compares the CURRENT on-disk production files against a previously
    saved baseline manifest. Returns {"changed": [...], "added": [...],
    "removed": [...]} -- all empty means freeze verification passed (audit
    item #46: no production behaviour changes)."""
    baseline = load_baseline_manifest(baseline_path)
    current = dict(compute_production_checksum_manifest(repo_root))
    changed = sorted(p for p in current if p in baseline and current[p] != baseline[p])
    added = sorted(p for p in current if p not in baseline)
    removed = sorted(p for p in baseline if p not in current)
    return {"changed": changed, "added": added, "removed": removed}


if __name__ == "__main__":
    import sys
    entries = compute_production_checksum_manifest()
    sys.stdout.write(manifest_to_sha256sum_text(entries))
    print(f"# combined: {combined_production_sha256()}", file=sys.stderr)

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_artifact_manifest(output_dir: Path, manifest_name: str = "artifact_manifest.json") -> Path:
    """Write a reproducibility manifest for all run artifacts under output_dir.

    The manifest intentionally excludes itself while hashing so repeated calls remain stable
    except for files that actually changed.
    """
    output_dir = Path(output_dir)
    rows: list[dict[str, Any]] = []
    manifest_path = output_dir / manifest_name
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        rows.append(
            {
                "path": str(path.relative_to(output_dir)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    payload = {"artifact_count": len(rows), "artifacts": rows}
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path

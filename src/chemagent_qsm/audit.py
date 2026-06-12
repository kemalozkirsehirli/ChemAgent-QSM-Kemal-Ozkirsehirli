from __future__ import annotations

import importlib.metadata
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np


def _safe_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


class AuditLogger:
    def __init__(self, output_dir: Path):
        self.path = output_dir / "audit.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, payload: dict[str, Any] | None = None) -> None:
        record = {
            "time_unix": time.time(),
            "event": event,
            "payload": payload or {},
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True, default=_json_default) + "\n")

    def environment(self) -> None:
        self.write(
            "environment",
            {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "numpy": np.__version__,
                "pyscf": _safe_version("pyscf"),
                "rdkit": _safe_version("rdkit"),
                "gpu4pyscf": _safe_version("gpu4pyscf"),
            },
        )


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return str(obj)

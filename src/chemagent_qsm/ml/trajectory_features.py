from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


def flatten_statmech_features(statmech_suite: dict[str, Any], prefix: str = "qsm") -> dict[str, float]:
    """Turn trajectory-analysis outputs into a compact ML-ready feature row.

    This is intentionally conservative: it summarizes long curves with interpretable scalar
    statistics rather than silently feeding raw time series into a model.
    """
    row: dict[str, float] = {}
    for name, payload in statmech_suite.items():
        key = f"{prefix}_{name}"
        if isinstance(payload, dict):
            if "value" in payload:
                _curve_features(row, key, payload["value"])
            elif "alpha2" in payload:
                _curve_features(row, key, payload["alpha2"])
            elif "g_r" in payload:
                _curve_features(row, key, payload["g_r"])
            elif "mean_per_frame" in payload:
                _curve_features(row, key, payload["mean_per_frame"])
            elif "tau_ps" in payload and payload["tau_ps"] is not None:
                row[f"{key}_tau_ps"] = float(payload["tau_ps"])
            elif "pearson_r" in payload:
                if payload.get("pearson_r") is not None:
                    row[f"{key}_pearson_r"] = float(payload["pearson_r"])
                if payload.get("spearman_r") is not None:
                    row[f"{key}_spearman_r"] = float(payload["spearman_r"])
            else:
                for subkey, value in payload.items():
                    if isinstance(value, (int, float)):
                        row[f"{key}_{subkey}"] = float(value)
        elif isinstance(payload, list) and payload and isinstance(payload[0], dict):
            for col in payload[0].keys():
                vals = [x[col] for x in payload if isinstance(x.get(col), (int, float))]
                if vals:
                    _curve_features(row, f"{key}_{col}", vals)
    return row


def write_feature_row(path: Path, row: dict[str, float], label: str | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(row)
    if label is not None:
        data["label"] = label
    exists = path.exists()
    fieldnames = sorted(data.keys())
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(data)


def _curve_features(row: dict[str, float], key: str, values) -> None:
    x = np.asarray(values, dtype=float).reshape(-1)
    if x.size == 0:
        return
    row[f"{key}_mean"] = float(np.nanmean(x))
    row[f"{key}_std"] = float(np.nanstd(x))
    row[f"{key}_min"] = float(np.nanmin(x))
    row[f"{key}_max"] = float(np.nanmax(x))
    row[f"{key}_final"] = float(x[-1])
    row[f"{key}_auc"] = float(np.trapezoid(np.nan_to_num(x)))

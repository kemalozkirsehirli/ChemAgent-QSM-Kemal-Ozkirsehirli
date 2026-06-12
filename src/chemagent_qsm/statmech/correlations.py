from __future__ import annotations

import numpy as np


def time_correlation(signal, max_lag: int | None = None, normalize: bool = True) -> np.ndarray:
    x = np.asarray(signal, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    n = x.shape[0]
    max_lag = min(max_lag or n - 1, n - 1)
    x = x - x.mean(axis=0, keepdims=True)
    corr = np.empty(max_lag + 1, dtype=float)
    for lag in range(max_lag + 1):
        corr[lag] = np.mean(x[: n - lag] * x[lag:])
    if normalize and abs(corr[0]) > 1e-15:
        corr = corr / corr[0]
    return corr


def mean_squared_displacement(coords, max_lag: int | None = None) -> np.ndarray:
    r = np.asarray(coords, dtype=float)
    n = r.shape[0]
    max_lag = min(max_lag or n - 1, n - 1)
    out = np.zeros(max_lag + 1, dtype=float)
    for lag in range(max_lag + 1):
        disp = r[lag:] - r[: n - lag]
        out[lag] = np.mean(np.sum(disp * disp, axis=-1))
    return out


def self_intermediate_scattering(coords, q_values, max_lag: int | None = None) -> dict[str, np.ndarray]:
    r = np.asarray(coords, dtype=float)
    n = r.shape[0]
    max_lag = min(max_lag or n - 1, n - 1)
    result = {}
    for q in q_values:
        vals = np.zeros(max_lag + 1, dtype=float)
        for lag in range(max_lag + 1):
            disp = r[lag:] - r[: n - lag]
            vals[lag] = np.mean(np.cos(float(q) * disp[..., 0]))
        result[f"q={float(q):.6g}"] = vals
    return result


def velocity_autocorrelation(coords, timestep_ps: float, max_lag: int | None = None) -> np.ndarray:
    r = np.asarray(coords, dtype=float)
    v = np.diff(r, axis=0) / float(timestep_ps)
    vacf = time_correlation(v.reshape(v.shape[0], -1), max_lag=max_lag, normalize=True)
    return vacf

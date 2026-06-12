from __future__ import annotations

import numpy as np


def relaxation_time(curve, timestep_ps: float, threshold: float = 1 / np.e) -> float | None:
    y = np.asarray(curve, dtype=float)
    below = np.where(y <= threshold)[0]
    if len(below) == 0:
        return None
    idx = int(below[0])
    if idx == 0:
        return 0.0
    y0, y1 = y[idx - 1], y[idx]
    if y0 == y1:
        return idx * timestep_ps
    frac = (threshold - y0) / (y1 - y0)
    return float((idx - 1 + frac) * timestep_ps)


def mobility_field(coords, lags: list[int]) -> dict[str, np.ndarray]:
    r = np.asarray(coords, dtype=float)
    out = {}
    for lag in lags:
        if lag <= 0 or lag >= r.shape[0]:
            continue
        disp = r[lag:] - r[:-lag]
        mobility = np.linalg.norm(disp, axis=-1)
        out[f"lag={lag}"] = mobility.mean(axis=0)
    return out


def non_gaussian_alpha2(coords, max_lag: int | None = None) -> np.ndarray:
    r = np.asarray(coords, dtype=float)
    n = r.shape[0]
    max_lag = min(max_lag or n - 1, n - 1)
    out = np.zeros(max_lag + 1, dtype=float)
    for lag in range(1, max_lag + 1):
        disp = r[lag:] - r[: n - lag]
        dr2 = np.sum(disp * disp, axis=-1).reshape(-1)
        m2 = np.mean(dr2)
        m4 = np.mean(dr2 * dr2)
        out[lag] = 3.0 * m4 / (5.0 * m2 * m2) - 1.0 if m2 > 0 else 0.0
    return out


def overlap_chi4(coords, cutoff: float, max_lag: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    r = np.asarray(coords, dtype=float)
    n_frames, n_atoms, _ = r.shape
    max_lag = min(max_lag or n_frames - 1, n_frames - 1)
    q = np.zeros(max_lag + 1, dtype=float)
    chi4 = np.zeros(max_lag + 1, dtype=float)
    for lag in range(max_lag + 1):
        disp = np.linalg.norm(r[lag:] - r[: n_frames - lag], axis=-1)
        overlap = (disp < cutoff).astype(float)
        q_t = overlap.mean(axis=1)
        q[lag] = q_t.mean()
        chi4[lag] = n_atoms * q_t.var()
    return q, chi4


def dynamical_heterogeneity_metrics(coords, lags: list[int], cutoffs: list[float]) -> dict[str, float | int | str | None]:
    """Compute compact glassy-dynamics heterogeneity metrics.

    The score combines three baseline signatures: four-point susceptibility peak from
    overlap/MSCOPE, non-Gaussian displacement peak, and mobility-field dispersion. It is a
    screening summary for comparing conditions, not a universal phase classifier.
    """
    r = np.asarray(coords, dtype=float)
    max_lag = r.shape[0] - 1
    valid_lags = [lag for lag in lags if 0 < lag <= max_lag] or [min(max_lag, 1)]
    valid_cutoffs = cutoffs or [0.3]

    best_chi4 = -np.inf
    best_lag = None
    best_cutoff = None
    for cutoff in valid_cutoffs:
        _q, chi4 = overlap_chi4(r, cutoff, max_lag=max_lag)
        for lag in valid_lags:
            if chi4[lag] > best_chi4:
                best_chi4 = float(chi4[lag])
                best_lag = int(lag)
                best_cutoff = float(cutoff)

    alpha2 = non_gaussian_alpha2(r, max_lag=max(valid_lags))
    alpha2_peak = float(np.nanmax(alpha2[1:])) if alpha2.size > 1 else 0.0

    mobility = mobility_field(r, valid_lags)
    mobility_cv = None
    if mobility:
        last = mobility[sorted(mobility.keys(), key=lambda k: int(k.split('=')[1]))[-1]]
        mean = float(np.nanmean(last))
        std = float(np.nanstd(last))
        mobility_cv = None if abs(mean) < 1e-12 else std / mean

    raw = max(best_chi4, 0.0) + max(alpha2_peak, 0.0) + (mobility_cv or 0.0)
    if raw < 0.5:
        regime = "weak"
    elif raw < 2.0:
        regime = "moderate"
    else:
        regime = "strong"
    return {
        "heterogeneity_score": float(raw),
        "regime": regime,
        "chi4_peak": None if not np.isfinite(best_chi4) else float(best_chi4),
        "chi4_peak_lag": best_lag,
        "chi4_peak_cutoff": best_cutoff,
        "alpha2_peak": alpha2_peak,
        "mobility_cv": mobility_cv,
    }

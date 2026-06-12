from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.special import sph_harm_y


def radial_distribution_function(coords, r_max: float = 10.0, bins: int = 100) -> tuple[np.ndarray, np.ndarray]:
    r = np.asarray(coords, dtype=float)
    dists = []
    for frame in r:
        diff = frame[:, None, :] - frame[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        iu = np.triu_indices(frame.shape[0], k=1)
        dists.append(dist[iu])
    all_d = np.concatenate(dists) if dists else np.array([])
    hist, edges = np.histogram(all_d, bins=bins, range=(0, r_max))
    centers = 0.5 * (edges[:-1] + edges[1:])
    dr = edges[1] - edges[0]
    n_atoms = r.shape[1]
    span = np.maximum(r.max(axis=(0, 1)) - r.min(axis=(0, 1)), 1e-6)
    volume = float(np.prod(span))
    rho = n_atoms / volume
    shell_vol = 4.0 * np.pi * centers**2 * dr
    ideal_pairs = r.shape[0] * n_atoms * rho * shell_vol / 2.0
    g = hist / np.maximum(ideal_pairs, 1e-12)
    return centers, g


def steinhardt_q_l(coords, l: int = 6, cutoff: float = 3.5) -> np.ndarray:
    r = np.asarray(coords, dtype=float)
    q_frame = []
    for frame in r:
        n = frame.shape[0]
        q_atoms = np.zeros(n, dtype=float)
        for i in range(n):
            vecs = frame - frame[i]
            dist = np.linalg.norm(vecs, axis=1)
            mask = (dist > 1e-12) & (dist < cutoff)
            neigh = vecs[mask]
            if len(neigh) == 0:
                continue
            theta = np.arccos(np.clip(neigh[:, 2] / np.linalg.norm(neigh, axis=1), -1, 1))
            phi = np.arctan2(neigh[:, 1], neigh[:, 0])
            qlm = []
            for m in range(-l, l + 1):
                qlm.append(np.mean(sph_harm_y(l, m, theta, phi)))
            q_atoms[i] = np.sqrt(4 * np.pi / (2 * l + 1) * np.sum(np.abs(qlm) ** 2)).real
        q_frame.append(q_atoms)
    return np.asarray(q_frame)


def structure_dynamics_correlation(local_order, mobility) -> dict[str, float | None]:
    x = np.asarray(local_order, dtype=float).reshape(-1)
    y = np.asarray(mobility, dtype=float).reshape(-1)
    n = min(len(x), len(y))
    if n < 3:
        return {"pearson_r": None, "spearman_r": None}
    x = x[:n]
    y = y[:n]
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return {"pearson_r": None, "spearman_r": None}
    return {
        "pearson_r": float(stats.pearsonr(x, y).statistic),
        "spearman_r": float(stats.spearmanr(x, y).statistic),
    }

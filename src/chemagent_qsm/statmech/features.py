from __future__ import annotations

from typing import Any

import numpy as np

from chemagent_qsm.schema import AnalysisSettings, StatMechTask, task_value
from chemagent_qsm.statmech.correlations import (
    mean_squared_displacement,
    self_intermediate_scattering,
    time_correlation,
    velocity_autocorrelation,
)
from chemagent_qsm.statmech.dynamics import dynamical_heterogeneity_metrics, mobility_field, non_gaussian_alpha2, overlap_chi4, relaxation_time
from chemagent_qsm.statmech.order import radial_distribution_function, steinhardt_q_l, structure_dynamics_correlation


def compute_statmech_suite(coords, timestep_ps: float, tasks, settings: AnalysisSettings) -> dict[str, Any]:
    r = np.asarray(coords, dtype=float)
    max_lag = min(settings.max_lag, r.shape[0] - 1)
    task_names = {task_value(t) for t in tasks}
    out: dict[str, Any] = {}

    if task_value(StatMechTask.MSD) in task_names:
        out["msd"] = {"lag": list(range(max_lag + 1)), "value": mean_squared_displacement(r, max_lag).tolist()}
    if task_value(StatMechTask.SISF) in task_names:
        sisf = self_intermediate_scattering(r, settings.q_values, max_lag)
        out["sisf"] = {k: v.tolist() for k, v in sisf.items()}
    if task_value(StatMechTask.VACF) in task_names:
        out["vacf"] = {"lag": list(range(max_lag + 1)), "value": velocity_autocorrelation(r, timestep_ps, max_lag).tolist()}
    if task_value(StatMechTask.TCF) in task_names:
        scalar = r.reshape(r.shape[0], -1).mean(axis=1)
        out["tcf"] = {"lag": list(range(max_lag + 1)), "value": time_correlation(scalar, max_lag).tolist()}
    if task_value(StatMechTask.RDF) in task_names:
        centers, g = radial_distribution_function(r, settings.rdf_r_max, settings.rdf_bins)
        out["rdf"] = {"r": centers.tolist(), "g_r": g.tolist()}
    local_order = None
    if task_value(StatMechTask.LOCAL_ORDER) in task_names or task_value(StatMechTask.STRUCTURE_DYNAMICS_COUPLING) in task_names:
        local_order = steinhardt_q_l(r, settings.local_order_l, settings.local_order_cutoff)
        out["local_order"] = {
            "l": settings.local_order_l,
            "cutoff": settings.local_order_cutoff,
            "mean_per_frame": local_order.mean(axis=1).tolist(),
            "mean_per_atom": local_order.mean(axis=0).tolist(),
            "global_mean": float(local_order.mean()),
        }
    mobility = None
    if task_value(StatMechTask.MOBILITY) in task_names or task_value(StatMechTask.STRUCTURE_DYNAMICS_COUPLING) in task_names:
        mobility = mobility_field(r, settings.mobility_lags)
        out["mobility"] = {k: v.tolist() for k, v in mobility.items()}
    if task_value(StatMechTask.RELAXATION) in task_names:
        if "sisf" not in out:
            sisf = self_intermediate_scattering(r, settings.q_values, max_lag)
            out["sisf"] = {k: v.tolist() for k, v in sisf.items()}
        first_curve = next(iter(out["sisf"].values()))
        out["relaxation"] = {
            "threshold": settings.relaxation_threshold,
            "tau_ps": relaxation_time(first_curve, timestep_ps, settings.relaxation_threshold),
        }
    if task_value(StatMechTask.NON_GAUSSIAN) in task_names:
        out["non_gaussian"] = {"lag": list(range(max_lag + 1)), "alpha2": non_gaussian_alpha2(r, max_lag).tolist()}
    if task_value(StatMechTask.STRUCTURE_DYNAMICS_COUPLING) in task_names:
        if local_order is None:
            local_order = steinhardt_q_l(r, settings.local_order_l, settings.local_order_cutoff)
        if mobility is None:
            mobility = mobility_field(r, settings.mobility_lags)
        if mobility:
            lag_key = sorted(mobility.keys(), key=lambda s: int(s.split("=")[1]))[-1]
            out["structure_dynamics_coupling"] = structure_dynamics_correlation(local_order.mean(axis=0), mobility[lag_key])
    if task_value(StatMechTask.MSCOPE) in task_names:
        out["mscope"] = mscope_features(r, settings.mscope_lags, settings.mscope_cutoffs)
    if task_value(StatMechTask.DYNAMICAL_HETEROGENEITY) in task_names:
        out["dynamical_heterogeneity"] = dynamical_heterogeneity_metrics(
            r,
            settings.mscope_lags if settings.mscope_lags else settings.mobility_lags,
            settings.mscope_cutoffs,
        )
    return out


def mscope_features(coords, lags: list[int], cutoffs: list[float]) -> list[dict[str, float]]:
    r = np.asarray(coords, dtype=float)
    max_lag = r.shape[0] - 1
    rows = []
    for cutoff in cutoffs:
        q, chi4 = overlap_chi4(r, cutoff, max_lag=max_lag)
        for lag in lags:
            if lag <= max_lag:
                rows.append({"lag": int(lag), "cutoff": float(cutoff), "overlap": float(q[lag]), "chi4": float(chi4[lag])})
    return rows

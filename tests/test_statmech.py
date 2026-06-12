import numpy as np

from chemagent_qsm.schema import AnalysisSettings, StatMechTask
from chemagent_qsm.statmech.correlations import mean_squared_displacement, self_intermediate_scattering
from chemagent_qsm.statmech.features import compute_statmech_suite


def test_msd_monotonic_for_constant_velocity():
    t = np.arange(10, dtype=float)
    coords = np.zeros((10, 2, 3), dtype=float)
    coords[:, :, 0] = t[:, None]
    msd = mean_squared_displacement(coords, max_lag=5)
    assert np.allclose(msd[:6], np.arange(6) ** 2)


def test_statmech_suite_smoke():
    rng = np.random.default_rng(0)
    coords = rng.normal(size=(20, 8, 3)).cumsum(axis=0)
    out = compute_statmech_suite(
        coords,
        timestep_ps=0.01,
        tasks=[StatMechTask.MSD, StatMechTask.SISF, StatMechTask.NON_GAUSSIAN, StatMechTask.MSCOPE],
        settings=AnalysisSettings(max_lag=5, q_values=[1.0], mscope_lags=[1, 2], mscope_cutoffs=[0.3]),
    )
    assert "msd" in out and "sisf" in out and "mscope" in out
    assert len(out["msd"]["value"]) == 6

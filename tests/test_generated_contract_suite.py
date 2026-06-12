from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from chemagent_qsm.evaluation.benchmarks import evaluate_against_baselines
from chemagent_qsm.evaluation.cv_objectives import CV_OBJECTIVES
from chemagent_qsm.planner import HeuristicChemPlanner
from chemagent_qsm.qm.descriptors import frontier_orbital_reactivity, orbital_gap_ev
from chemagent_qsm.qm.mock_runner import MockQMRunner
from chemagent_qsm.qm.spectra import gaussian_ir_spectrum
from chemagent_qsm.schema import AnalysisSettings, Backend, MoleculeSpec, QMSettings, QMTask, StatMechTask, WorkflowPlan, task_value
from chemagent_qsm.statmech.features import compute_statmech_suite
from chemagent_qsm.validators import WorkflowValidator
from chemagent_qsm.workflow.codegen import generate_python_workflow
from chemagent_qsm.workflow.dag import build_workflow_dag, dag_to_jsonable

MATRIX_PATH = Path(__file__).parent / "generated" / "chemagent_qsm_600_testcases.jsonl"
CASES = [json.loads(line) for line in MATRIX_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def _tasks(values, enum_cls):
    return [enum_cls(v) for v in values]


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_my_chemagent_qsm_generated_contract_matrix(case, tmp_path):
    assert case["description"].startswith("I ")
    category = case["category"]
    if category == "planner_qm":
        plan = HeuristicChemPlanner().plan(case["prompt"], backend=Backend.MOCK, output_dir=tmp_path / "run")
        assert plan.molecule is not None
        assert plan.molecule.name == case["expected_molecule"]
        assert str(plan.qm_settings.xc) == case["expected_xc"]
        assert plan.qm_settings.basis.lower() == case["expected_basis"].lower()
        assert [task_value(t) for t in plan.qm_tasks] == case["expected_tasks"]
    elif category == "planner_statmech":
        plan = HeuristicChemPlanner().plan(case["prompt"], backend=Backend.MOCK, output_dir=tmp_path / "run")
        observed = {task_value(t) for t in plan.statmech_tasks}
        assert set(case["expected_stat_tasks"]).issubset(observed)
    elif category == "mock_qm":
        runner = MockQMRunner(QMSettings(basis=case["basis"], xc=case["xc"], ir_points=case["ir_points"]))
        mol = MoleculeSpec(name="my_case_molecule", smiles=case["smiles"])
        sp = runner.single_point(mol)
        desc = runner.descriptors(mol, sp)
        freq = runner.frequencies(mol)
        spec = runner.ir_spectrum(mol, freq)
        assert sp["converged"] is True
        assert sp["total_energy_hartree"] < 0.0
        assert desc["gap_ev"] is not None and desc["gap_ev"] > 0.0
        assert "drug_discovery_bridge" in desc
        assert len(freq["frequencies_cm-1"]) >= 1
        assert len(spec["x_cm-1"]) == case["ir_points"]
        assert max(spec["intensity"]) <= 1.0 + 1e-8
    elif category == "statmech_suite":
        rng = np.random.default_rng(case["seed"])
        increments = rng.normal(0.0, 0.04, size=(case["frames"], case["atoms"], 3))
        coords = np.cumsum(increments, axis=0)
        settings = AnalysisSettings(max_lag=min(6, case["frames"] - 1), rdf_bins=12, mobility_lags=[1, 2, 4], mscope_lags=[1, 2, 4])
        suite = compute_statmech_suite(coords, 0.002, _tasks(case["tasks"], StatMechTask), settings)
        for expected in case["tasks"]:
            if expected == "relaxation":
                assert "relaxation" in suite and "sisf" in suite
            elif expected == "structure_dynamics_coupling":
                assert "structure_dynamics_coupling" in suite
                assert "local_order" in suite and "mobility" in suite
            else:
                assert expected in suite
        assert suite
    elif category == "validator_codegen":
        plan = HeuristicChemPlanner().plan(case["prompt"], backend=Backend.MOCK, output_dir=tmp_path / "run")
        validation = WorkflowValidator().validate(plan)
        assert validation["valid"] is True
        graph = build_workflow_dag(plan)
        graph_json = dag_to_jsonable(graph)
        assert graph_json["nodes"] and graph_json["edges"]
        path = generate_python_workflow(plan, tmp_path / "generated_workflow.py")
        content = path.read_text(encoding="utf-8")
        assert "PLAN_SHA256" in content
        assert "ChemAgent-QSM" in content
    elif category == "descriptor_math":
        homo = case["homo"]
        lumo = case["lumo"]
        gap = orbital_gap_ev([homo, lumo], [2, 0])
        desc = frontier_orbital_reactivity(homo * 27.211386245988, lumo * 27.211386245988)
        assert gap is not None and gap > 0.0
        assert desc["koopmans_ip_ev"] > 0.0
        assert desc["hardness_ev"] > 0.0
    else:
        raise AssertionError(f"Unhandled generated test category: {category}")


def test_my_chemagent_qsm_matrix_has_at_least_500_cases():
    assert len(CASES) >= 500
    assert len({case["id"] for case in CASES}) == len(CASES)
    assert all(case["description"].startswith("I ") for case in CASES)


def test_my_chemagent_qsm_benchmark_objective_is_registered():
    assert "benchmark_against_baselines" in CV_OBJECTIVES

from pathlib import Path

from chemagent_qsm.evaluation.cv_objectives import CV_OBJECTIVES, evaluate_cv_objective_coverage
from chemagent_qsm.orchestrator import ChemAgentQSM
from chemagent_qsm.planner import load_plan_from_yaml
from chemagent_qsm.schema import StatMechTask


def test_full_cv_objective_smoke_run_has_complete_artifact_coverage(tmp_path):
    plan = load_plan_from_yaml(Path("examples/full_cv_objective_smoke.yaml"))
    plan = plan.model_copy(update={"output_dir": tmp_path / "full"})
    result = ChemAgentQSM().run(plan)
    coverage = evaluate_cv_objective_coverage(Path(result["output_dir"]))
    assert coverage["passed"], coverage
    assert coverage["coverage_fraction"] == 1.0
    assert len(CV_OBJECTIVES) >= 10


def test_planner_infers_dynamical_heterogeneity_for_glass():
    from chemagent_qsm.planner import HeuristicChemPlanner

    plan = HeuristicChemPlanner().plan("Analyze glassy dynamical heterogeneity with MSD SISF MSCOPE")
    assert StatMechTask.DYNAMICAL_HETEROGENEITY in plan.statmech_tasks

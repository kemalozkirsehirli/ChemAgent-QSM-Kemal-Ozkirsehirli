from pathlib import Path

from chemagent_qsm.orchestrator import ChemAgentQSM
from chemagent_qsm.planner import load_plan_from_yaml


def test_mock_qm_workflow(tmp_path):
    plan = load_plan_from_yaml(Path("examples/water_qm.yaml"))
    plan = plan.model_copy(update={"output_dir": tmp_path / "water"})
    result = ChemAgentQSM().run(plan)
    out = Path(result["output_dir"])
    assert (out / "generated_workflow.py").exists()
    assert (out / "qm" / "descriptors.json").exists()
    assert (out / "qm" / "ir_spectrum.csv").exists()
    assert result["qm"]["descriptors"]["gap_ev"] is not None


def test_codegen_contains_plan_json(tmp_path):
    plan = load_plan_from_yaml(Path("examples/water_qm.yaml"))
    plan = plan.model_copy(update={"output_dir": tmp_path / "water"})
    ChemAgentQSM().run(plan)
    code = (tmp_path / "water" / "generated_workflow.py").read_text()
    assert "WorkflowPlan.model_validate_json" in code

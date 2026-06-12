from chemagent_qsm.planner import HeuristicChemPlanner
from chemagent_qsm.schema import QMTask


def test_planner_adds_dependencies_for_ir():
    plan = HeuristicChemPlanner().plan("Optimize caffeine with B3LYP/6-31g*, compute descriptors and IR spectrum")
    tasks = [str(t) for t in plan.qm_tasks]
    assert str(QMTask.OPTIMIZE) in tasks
    assert str(QMTask.FREQUENCIES) in tasks
    assert str(QMTask.IR_SPECTRUM) in tasks
    assert plan.molecule.name == "caffeine"

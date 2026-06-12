from __future__ import annotations

from typing import Any

import networkx as nx

from chemagent_qsm.schema import QMTask, StatMechTask, WorkflowPlan, task_value


def build_workflow_dag(plan: WorkflowPlan) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_node("plan", kind="input")
    if plan.molecule:
        graph.add_edge("plan", "molecule")
    if plan.trajectory:
        graph.add_edge("plan", "trajectory")
    previous_qm = "molecule"
    for task in plan.qm_tasks:
        name = f"qm:{task_value(task)}"
        graph.add_edge(previous_qm, name)
        previous_qm = name
    for task in plan.statmech_tasks:
        name = f"statmech:{task_value(task)}"
        graph.add_edge("trajectory", name)
    if any(task_value(t) == task_value(StatMechTask.STRUCTURE_DYNAMICS_COUPLING) for t in plan.statmech_tasks):
        graph.add_edge("statmech:local_order", "statmech:structure_dynamics_coupling")
        graph.add_edge("statmech:mobility", "statmech:structure_dynamics_coupling")
    graph.add_node("report", kind="output")
    for node in list(graph.nodes):
        if node.startswith("qm:") or node.startswith("statmech:"):
            graph.add_edge(node, "report")
    return graph


def dag_to_jsonable(graph: nx.DiGraph) -> dict[str, Any]:
    return {
        "nodes": [{"id": n, **graph.nodes[n]} for n in graph.nodes],
        "edges": [{"source": u, "target": v} for u, v in graph.edges],
    }

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml

from chemagent_qsm.audit import AuditLogger
from chemagent_qsm.artifacts import write_artifact_manifest
from chemagent_qsm.exceptions import ValidationError
from chemagent_qsm.evaluation.cv_objectives import evaluate_cv_objective_coverage
from chemagent_qsm.evaluation.benchmarks import write_benchmark_report
from chemagent_qsm.hpc.slurm import write_slurm_script
from chemagent_qsm.io.trajectory_io import load_trajectory
from chemagent_qsm.ml.trajectory_features import flatten_statmech_features, write_feature_row
from chemagent_qsm.planner import save_plan_yaml
from chemagent_qsm.qm.mock_runner import MockQMRunner
from chemagent_qsm.qm.pyscf_runner import PySCFRunner
from chemagent_qsm.schema import Backend, QMTask, WorkflowPlan, task_value
from chemagent_qsm.statmech.features import compute_statmech_suite
from chemagent_qsm.validators import WorkflowValidator
from chemagent_qsm.workflow.codegen import generate_python_workflow
from chemagent_qsm.workflow.dag import build_workflow_dag, dag_to_jsonable


class ChemAgentQSM:
    def __init__(self, validator: WorkflowValidator | None = None):
        self.validator = validator or WorkflowValidator()

    def run(self, plan: WorkflowPlan) -> dict[str, Any]:
        output_dir = Path(plan.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        audit = AuditLogger(output_dir)
        audit.environment()
        audit.write("plan_received", {"id": plan.id, "backend": str(plan.backend)})

        validation = self.validator.validate(plan)
        _write_json(output_dir / "validation.json", validation)
        if not validation["valid"]:
            audit.write("validation_failed", validation)
            raise ValidationError("Workflow validation failed: " + "; ".join(validation["errors"]))
        audit.write("validation_passed", validation)

        _write_json(output_dir / "plan.json", plan.to_yamlable())
        save_plan_yaml(plan, output_dir / "plan.yaml")
        generate_python_workflow(plan, output_dir / "generated_workflow.py")
        graph = build_workflow_dag(plan)
        _write_json(output_dir / "workflow_dag.json", dag_to_jsonable(graph))
        write_slurm_script(plan, output_dir / "run.slurm")

        results: dict[str, Any] = {"id": plan.id, "output_dir": str(output_dir), "qm": {}, "statmech": {}}
        molecule = plan.molecule
        qm_runner = self._runner(plan)
        sp_cache = None
        freq_cache = None
        if molecule and plan.qm_tasks:
            qm_dir = output_dir / "qm"
            qm_dir.mkdir(exist_ok=True)
            for task in plan.qm_tasks:
                name = task_value(task)
                audit.write("qm_task_start", {"task": name})
                if name == task_value(QMTask.SINGLE_POINT):
                    sp_cache = qm_runner.single_point(molecule)
                    results["qm"]["single_point"] = sp_cache
                    _write_json(qm_dir / "single_point.json", sp_cache)
                elif name == task_value(QMTask.OPTIMIZE):
                    molecule, opt = qm_runner.optimize(molecule)
                    results["qm"]["optimize"] = opt
                    _write_json(qm_dir / "optimize.json", opt)
                elif name == task_value(QMTask.DESCRIPTORS):
                    desc = qm_runner.descriptors(molecule, sp_cache)
                    results["qm"]["descriptors"] = desc
                    _write_json(qm_dir / "descriptors.json", desc)
                elif name == task_value(QMTask.FREQUENCIES):
                    freq_cache = qm_runner.frequencies(molecule)
                    results["qm"]["frequencies"] = freq_cache
                    _write_json(qm_dir / "frequencies.json", freq_cache)
                elif name == task_value(QMTask.IR_SPECTRUM):
                    spec = qm_runner.ir_spectrum(molecule, freq_cache)
                    results["qm"]["ir_spectrum"] = {"num_points": len(spec["x_cm-1"])}
                    _write_json(qm_dir / "ir_spectrum.json", spec)
                    _write_xy_csv(qm_dir / "ir_spectrum.csv", "wavenumber_cm-1", "intensity", spec["x_cm-1"], spec["intensity"])
                audit.write("qm_task_done", {"task": name})

        if plan.trajectory and plan.statmech_tasks:
            stat_dir = output_dir / "statmech"
            stat_dir.mkdir(exist_ok=True)
            audit.write("statmech_load_trajectory", {"path": str(plan.trajectory.coordinates_path)})
            coords = load_trajectory(plan.trajectory.coordinates_path)
            suite = compute_statmech_suite(coords, plan.trajectory.timestep_ps, plan.statmech_tasks, plan.analysis)
            feature_row = flatten_statmech_features(suite)
            results["statmech"] = _summarize_statmech(suite)
            results["statmech"]["ml_feature_count"] = len(feature_row)
            for name, payload in suite.items():
                _write_json(stat_dir / f"{name}.json", payload)
                _maybe_write_stat_csv(stat_dir / f"{name}.csv", payload)
            _write_json(stat_dir / "ml_feature_row.json", feature_row)
            write_feature_row(stat_dir / "ml_features.csv", feature_row, label=plan.metadata.get("label"))
            audit.write("statmech_ml_features_written", {"feature_count": len(feature_row)})
            audit.write("statmech_done", {"tasks": [task_value(t) for t in plan.statmech_tasks]})

        write_artifact_manifest(output_dir)
        benchmark_report = write_benchmark_report(output_dir)
        write_artifact_manifest(output_dir)
        cv_coverage = evaluate_cv_objective_coverage(output_dir)
        _write_json(output_dir / "cv_objective_coverage.json", cv_coverage)
        write_artifact_manifest(output_dir)
        results["cv_objective_coverage"] = {
            "passed": cv_coverage["passed"],
            "coverage_fraction": cv_coverage["coverage_fraction"],
        }
        results["benchmark"] = {
            "passed": benchmark_report["passed"],
            "score": benchmark_report["actual_system"]["score"],
            "margin_over_best_baseline": benchmark_report["margin_over_best_baseline"],
        }
        _write_json(output_dir / "results.json", results)
        _write_report(output_dir / "report.md", plan, validation, results, cv_coverage)
        write_artifact_manifest(output_dir)
        audit.write("workflow_complete", {"output_dir": str(output_dir), "cv_objective_coverage": cv_coverage["coverage_fraction"]})
        return results

    def _runner(self, plan: WorkflowPlan):
        if plan.backend == Backend.PYSCF:
            return PySCFRunner(plan.qm_settings)
        return MockQMRunner(plan.qm_settings)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


def _json_default(obj: Any) -> Any:
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
    except Exception:
        pass
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _write_xy_csv(path: Path, x_name: str, y_name: str, x, y) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([x_name, y_name])
        writer.writerows(zip(x, y))


def _maybe_write_stat_csv(path: Path, payload: Any) -> None:
    if isinstance(payload, dict) and "lag" in payload and "value" in payload:
        _write_xy_csv(path, "lag", "value", payload["lag"], payload["value"])
    elif isinstance(payload, dict) and "r" in payload and "g_r" in payload:
        _write_xy_csv(path, "r", "g_r", payload["r"], payload["g_r"])
    elif isinstance(payload, dict) and "alpha2" in payload:
        _write_xy_csv(path, "lag", "alpha2", payload["lag"], payload["alpha2"])
    elif isinstance(payload, list) and payload and isinstance(payload[0], dict):
        keys = list(payload[0].keys())
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(payload)


def _summarize_statmech(suite: dict[str, Any]) -> dict[str, Any]:
    summary = {}
    for name, payload in suite.items():
        if isinstance(payload, dict):
            summary[name] = {"keys": list(payload.keys())}
        elif isinstance(payload, list):
            summary[name] = {"rows": len(payload)}
        else:
            summary[name] = str(type(payload))
    return summary


def _write_report(path: Path, plan: WorkflowPlan, validation: dict[str, Any], results: dict[str, Any], cv_coverage: dict[str, Any] | None = None) -> None:
    lines = [
        f"# ChemAgent-QSM Report: {plan.id}",
        "",
        "I generated this report as my implementation audit for the ChemAgent-QSM research system.",
        "",
        f"Prompt: {plan.prompt or '(YAML-defined workflow)'}",
        f"Backend: {plan.backend}",
        f"Validation: {'passed' if validation['valid'] else 'failed'}",
        "",
        "## QM tasks",
    ]
    for task in plan.qm_tasks:
        lines.append(f"- {task_value(task)}")
    lines.append("")
    lines.append("## Statistical-mechanics tasks")
    for task in plan.statmech_tasks:
        lines.append(f"- {task_value(task)}")
    lines.append("")
    lines.append("## Output index")
    lines.append("- plan.json / plan.yaml")
    lines.append("- validation.json")
    lines.append("- generated_workflow.py")
    lines.append("- workflow_dag.json")
    lines.append("- audit.jsonl")
    lines.append("- artifact_manifest.json")
    lines.append("- cv_objective_coverage.json")
    lines.append("- results.json")
    if results.get("qm"):
        lines.append("- qm/*.json and qm/*.csv")
    if results.get("statmech"):
        lines.append("- statmech/*.json and statmech/*.csv")
        lines.append("- statmech/ml_feature_row.json and statmech/ml_features.csv")
    if results.get("benchmark"):
        lines.extend(["", "## Baseline benchmark"])
        lines.append(f"- Passed: {results['benchmark']['passed']}")
        lines.append(f"- Score: {results['benchmark']['score']}")
        lines.append(f"- Margin over best baseline: {results['benchmark']['margin_over_best_baseline']}")
    if cv_coverage is not None:
        lines.extend(["", "## CV objective coverage"])
        lines.append(f"Coverage: {cv_coverage['present_artifacts']}/{cv_coverage['required_artifacts']} required artifacts ({cv_coverage['coverage_fraction']:.1%})")
        for objective, payload in cv_coverage["objectives"].items():
            status = "passed" if payload["passed"] else "missing"
            lines.append(f"- {objective}: {status}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

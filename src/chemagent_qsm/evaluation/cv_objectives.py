from __future__ import annotations

from pathlib import Path
from typing import Any

CV_OBJECTIVES: dict[str, list[str]] = {
    "natural_language_to_validated_qm_pipeline": [
        "plan.json",
        "validation.json",
        "generated_workflow.py",
        "workflow_dag.json",
    ],
    "pyscf_electronic_structure_for_drug_like_molecules": [
        "qm/single_point.json",
        "qm/descriptors.json",
    ],
    "optimized_geometries": ["qm/optimize.json"],
    "electronic_structure_descriptors": ["qm/descriptors.json"],
    "vibrational_frequencies_and_ir_spectrum": [
        "qm/frequencies.json",
        "qm/ir_spectrum.json",
        "qm/ir_spectrum.csv",
    ],
    "auditable_python_workflow_generation": [
        "generated_workflow.py",
        "audit.jsonl",
        "artifact_manifest.json",
    ],
    "time_correlation_functions": ["statmech/tcf.json"],
    "local_order_metric": ["statmech/local_order.json"],
    "relaxation_timescale": ["statmech/relaxation.json"],
    "mobility_field": ["statmech/mobility.json"],
    "structure_dynamics_coupling": ["statmech/structure_dynamics_coupling.json"],
    "msd_sisf_mscope_baselines": [
        "statmech/msd.json",
        "statmech/sisf.json",
        "statmech/mscope.json",
        "statmech/dynamical_heterogeneity.json",
    ],
    "llm_trajectory_ml_features": [
        "statmech/ml_feature_row.json",
        "statmech/ml_features.csv",
    ],
    "benchmark_against_baselines": ["evaluation/benchmark_report.json"],
}


def evaluate_cv_objective_coverage(output_dir: Path) -> dict[str, Any]:
    """Evaluate whether a completed run emitted every CV-objective artifact.

    This is a deterministic QA tool, not a scientific proof of correctness. It verifies that
    the implementation path produces the auditable files claimed in the CV objective lines.
    """
    output_dir = Path(output_dir)
    objectives: dict[str, Any] = {}
    total_required = 0
    total_present = 0
    for objective, relpaths in CV_OBJECTIVES.items():
        present = []
        missing = []
        for relpath in relpaths:
            total_required += 1
            if (output_dir / relpath).exists():
                present.append(relpath)
                total_present += 1
            else:
                missing.append(relpath)
        objectives[objective] = {
            "passed": not missing,
            "present": present,
            "missing": missing,
        }
    return {
        "output_dir": str(output_dir),
        "passed": total_present == total_required,
        "coverage_fraction": total_present / max(total_required, 1),
        "present_artifacts": total_present,
        "required_artifacts": total_required,
        "objectives": objectives,
    }

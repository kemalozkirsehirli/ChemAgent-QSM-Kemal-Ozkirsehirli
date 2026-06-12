from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from chemagent_qsm.schema import Backend, QMTask, WorkflowPlan, task_value


class WorkflowValidator:
    def validate(self, plan: WorkflowPlan) -> dict[str, Any]:
        errors: list[str] = []
        warnings: list[str] = []
        if plan.qm_tasks and plan.molecule is None:
            errors.append("QM tasks require a molecule specification.")
        if plan.statmech_tasks and plan.trajectory is None:
            errors.append("Statistical-mechanics tasks require a trajectory specification.")
        if plan.trajectory is not None and not Path(plan.trajectory.coordinates_path).exists():
            errors.append(f"Trajectory file does not exist: {plan.trajectory.coordinates_path}")
        if plan.molecule is not None and plan.molecule.xyz_path is not None and not Path(plan.molecule.xyz_path).exists():
            errors.append(f"XYZ file does not exist: {plan.molecule.xyz_path}")
        if plan.backend == Backend.PYSCF and importlib.util.find_spec("pyscf") is None:
            errors.append("backend='pyscf' requested but PySCF is not installed.")
        if plan.molecule is not None and plan.molecule.smiles and not (plan.molecule.atom_block or plan.molecule.xyz_path):
            if importlib.util.find_spec("rdkit") is None and plan.backend == Backend.PYSCF:
                errors.append("SMILES-only PySCF runs require RDKit to generate 3D coordinates.")
            elif importlib.util.find_spec("rdkit") is None:
                warnings.append("RDKit unavailable; mock backend can proceed, but real PySCF cannot use SMILES-only inputs.")
        task_names = {task_value(t) for t in plan.qm_tasks}
        if task_value(QMTask.IR_SPECTRUM) in task_names and task_value(QMTask.FREQUENCIES) not in task_names:
            warnings.append("IR spectrum requested without frequencies; planner usually adds this dependency.")
        if task_value(QMTask.FREQUENCIES) in task_names and task_value(QMTask.OPTIMIZE) not in task_names:
            warnings.append("Frequencies are most meaningful on optimized geometries.")
        if plan.qm_settings.use_gpu and importlib.util.find_spec("gpu4pyscf") is None:
            warnings.append("use_gpu=True but GPU4PySCF is not installed; PySCF runner will fall back or fail depending on environment.")
        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

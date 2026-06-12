from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import yaml

from chemagent_qsm.schema import (
    AnalysisSettings,
    Backend,
    MoleculeSpec,
    QMMethod,
    QMSettings,
    QMTask,
    StatMechTask,
    WorkflowPlan,
)

COMMON_MOLECULES: dict[str, dict[str, str]] = {
    "water": {
        "smiles": "O",
        "atom_block": "O 0.000000 0.000000 0.000000\nH 0.758602 0.000000 0.504284\nH -0.758602 0.000000 0.504284",
    },
    "ethanol": {"smiles": "CCO"},
    "benzene": {"smiles": "c1ccccc1"},
    "caffeine": {"smiles": "Cn1cnc2n(C)c(=O)n(C)c(=O)c12"},
    "aspirin": {"smiles": "CC(=O)Oc1ccccc1C(=O)O"},
    "acetaminophen": {"smiles": "CC(=O)Nc1ccc(O)cc1"},
}

FUNCTIONALS = ["b3lyp", "pbe0", "pbe", "m06-2x", "wb97x", "wb97x-d", "blyp", "lda"]
BASIS_PATTERN = re.compile(r"(sto-3g|3-21g|6-31g\*\*|6-31g\*|6-31g\(d\)|6-31g|def2-[a-z0-9]+|cc-pv[tqdz]+)", re.I)


class HeuristicChemPlanner:
    """Deterministic natural-language planner used as the safe default agent brain."""

    def plan(self, prompt: str, backend: Backend | str = Backend.MOCK, output_dir: str | Path = "runs/planned") -> WorkflowPlan:
        text = prompt.lower()
        backend_enum = Backend(str(backend))
        molecule = self._infer_molecule(text)
        qm_tasks = self._infer_qm_tasks(text)
        stat_tasks = self._infer_statmech_tasks(text)
        qm_settings = self._infer_qm_settings(text)
        if qm_tasks:
            qm_tasks = self._add_qm_dependencies(qm_tasks)
        if not qm_tasks and not stat_tasks:
            qm_tasks = [QMTask.SINGLE_POINT, QMTask.DESCRIPTORS]
        tags = sorted(set(["chemagent-qsm", "agentic-qm"] + (["statmech"] if stat_tasks else [])))
        return WorkflowPlan(
            prompt=prompt,
            backend=backend_enum,
            molecule=molecule,
            qm_settings=qm_settings,
            qm_tasks=qm_tasks,
            statmech_tasks=stat_tasks,
            analysis=AnalysisSettings(),
            output_dir=Path(output_dir),
            tags=tags,
            metadata={"planner": "heuristic-v0.1", "requires_human_review": True},
        )

    def _infer_molecule(self, text: str) -> MoleculeSpec | None:
        for name, payload in COMMON_MOLECULES.items():
            if name in text:
                return MoleculeSpec(name=name, **payload)
        smiles_match = re.search(r"smiles\s*[:=]\s*([A-Za-z0-9@+\-\[\]\(\)=#$\\/%.]+)", text, re.I)
        if smiles_match:
            return MoleculeSpec(name="prompt_smiles", smiles=smiles_match.group(1))
        if any(k in text for k in ["dft", "hf", "molecule", "geometry", "descriptor", "spectrum", "pyscf"]):
            return MoleculeSpec(name="water", **COMMON_MOLECULES["water"])
        return None

    def _infer_qm_tasks(self, text: str) -> list[QMTask]:
        tasks: list[QMTask] = []
        add = tasks.append
        if any(k in text for k in ["single point", "energy", "scf"]):
            add(QMTask.SINGLE_POINT)
        if any(k in text for k in ["optimize", "optimise", "geometry", "minimum"]):
            add(QMTask.OPTIMIZE)
        if any(k in text for k in ["descriptor", "homo", "lumo", "dipole", "charge", "drug-like"]):
            add(QMTask.DESCRIPTORS)
        if any(k in text for k in ["frequency", "frequencies", "normal mode", "vibrational"]):
            add(QMTask.FREQUENCIES)
        if any(k in text for k in ["ir", "infrared", "spectrum", "spectra"]):
            add(QMTask.IR_SPECTRUM)
        return _dedupe(tasks)

    def _infer_statmech_tasks(self, text: str) -> list[StatMechTask]:
        tasks: list[StatMechTask] = []
        mapping = {
            StatMechTask.TCF: ["tcf", "time correlation", "correlation function"],
            StatMechTask.MSD: ["msd", "mean squared displacement"],
            StatMechTask.SISF: ["sisf", "self intermediate scattering"],
            StatMechTask.VACF: ["vacf", "velocity autocorrelation"],
            StatMechTask.RDF: ["rdf", "radial distribution"],
            StatMechTask.LOCAL_ORDER: ["local order", "q6", "steinhardt"],
            StatMechTask.RELAXATION: ["relaxation", "tau alpha", "alpha relaxation"],
            StatMechTask.MOBILITY: ["mobility", "dynamic heterogeneity"],
            StatMechTask.STRUCTURE_DYNAMICS_COUPLING: ["structure dynamics", "structure/dynamics", "coupling"],
            StatMechTask.NON_GAUSSIAN: ["non-gaussian", "alpha2"],
            StatMechTask.MSCOPE: ["mscope", "multi-scale"],
            StatMechTask.DYNAMICAL_HETEROGENEITY: ["dynamical heterogeneity", "heterogeneity", "glassy dynamics"],
        }
        for task, keys in mapping.items():
            if any(k in text for k in keys):
                tasks.append(task)
        if "glass" in text or "disordered" in text:
            tasks.extend([StatMechTask.MSD, StatMechTask.SISF, StatMechTask.LOCAL_ORDER, StatMechTask.MOBILITY, StatMechTask.MSCOPE, StatMechTask.DYNAMICAL_HETEROGENEITY])
        return _dedupe(tasks)

    def _infer_qm_settings(self, text: str) -> QMSettings:
        method = QMMethod.HF if "hartree" in text or re.search(r"\bhf\b", text) else QMMethod.DFT
        xc = "b3lyp"
        for functional in FUNCTIONALS:
            if functional in text:
                xc = functional
                break
        basis_match = BASIS_PATTERN.search(text)
        basis = basis_match.group(1) if basis_match else "6-31g*"
        return QMSettings(method=method, basis=basis, xc=xc, use_gpu=("gpu" in text))

    def _add_qm_dependencies(self, tasks: Iterable[QMTask]) -> list[QMTask]:
        out: list[QMTask] = []
        requested = set(tasks)
        if QMTask.IR_SPECTRUM in requested:
            requested.add(QMTask.FREQUENCIES)
        if QMTask.FREQUENCIES in requested:
            requested.add(QMTask.OPTIMIZE)
        if QMTask.DESCRIPTORS in requested:
            requested.add(QMTask.SINGLE_POINT)
        order = [QMTask.SINGLE_POINT, QMTask.OPTIMIZE, QMTask.DESCRIPTORS, QMTask.FREQUENCIES, QMTask.IR_SPECTRUM]
        for task in order:
            if task in requested:
                out.append(task)
        return out


def _dedupe(items):
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def load_plan_from_yaml(path: str | Path) -> WorkflowPlan:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return WorkflowPlan.model_validate(data)


def save_plan_yaml(plan: WorkflowPlan, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(plan.to_yamlable(), f, sort_keys=False)

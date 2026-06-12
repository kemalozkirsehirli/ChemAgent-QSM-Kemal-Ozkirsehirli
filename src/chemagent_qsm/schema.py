from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class Backend(StrEnum):
    MOCK = "mock"
    PYSCF = "pyscf"


class QMMethod(StrEnum):
    HF = "hf"
    DFT = "dft"


class QMTask(StrEnum):
    SINGLE_POINT = "single_point"
    OPTIMIZE = "optimize"
    DESCRIPTORS = "descriptors"
    FREQUENCIES = "frequencies"
    IR_SPECTRUM = "ir_spectrum"


class StatMechTask(StrEnum):
    TCF = "tcf"
    MSD = "msd"
    SISF = "sisf"
    VACF = "vacf"
    RDF = "rdf"
    LOCAL_ORDER = "local_order"
    RELAXATION = "relaxation"
    MOBILITY = "mobility"
    STRUCTURE_DYNAMICS_COUPLING = "structure_dynamics_coupling"
    NON_GAUSSIAN = "non_gaussian"
    MSCOPE = "mscope"
    DYNAMICAL_HETEROGENEITY = "dynamical_heterogeneity"


class MoleculeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "molecule"
    smiles: str | None = None
    xyz_path: Path | None = None
    atom_block: str | None = Field(
        None,
        description="PySCF-style atom block or XYZ body: one atom per line with x y z coordinates.",
    )
    charge: int = 0
    spin: int = Field(0, description="2S, i.e. unpaired electrons in PySCF convention.")
    unit: Literal["Angstrom", "Bohr"] = "Angstrom"

    @model_validator(mode="after")
    def require_structure(self) -> "MoleculeSpec":
        if not (self.smiles or self.xyz_path or self.atom_block):
            raise ValueError("Provide at least one of smiles, xyz_path, or atom_block.")
        return self


class QMSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: QMMethod = QMMethod.DFT
    basis: str = "6-31g*"
    xc: str = "b3lyp"
    density_fit: bool = True
    newton: bool = False
    use_gpu: bool = False
    max_scf_cycles: int = 100
    conv_tol: float = 1e-9
    memory_mb: int = 4000
    num_threads: int | None = None
    geomopt_backend: Literal["geometric", "berny"] = "geometric"
    geomopt_maxsteps: int = 100
    temperature_k: float = 298.15
    pressure_pa: float = 101325.0
    ir_min_cm: float = 400.0
    ir_max_cm: float = 4000.0
    ir_points: int = 1800
    ir_width_cm: float = 20.0


class TrajectorySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    coordinates_path: Path
    topology_path: Path | None = None
    timestep_ps: float = 0.002
    coordinate_unit: Literal["Angstrom", "nm"] = "Angstrom"
    unwrap: bool = False
    selection: str | None = None


class AnalysisSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_lag: int = 100
    q_values: list[float] = Field(default_factory=lambda: [1.0, 2.0, 3.0])
    rdf_r_max: float = 10.0
    rdf_bins: int = 100
    local_order_l: int = 6
    local_order_cutoff: float = 3.5
    mobility_lags: list[int] = Field(default_factory=lambda: [1, 5, 10, 25, 50])
    relaxation_threshold: float = 1 / 2.718281828459045
    overlap_cutoff: float = 0.3
    mscope_cutoffs: list[float] = Field(default_factory=lambda: [0.2, 0.3, 0.5, 0.8])
    mscope_lags: list[int] = Field(default_factory=lambda: [1, 2, 5, 10, 20, 50])


class WorkflowPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: f"chemagent-qsm-{uuid4().hex[:10]}")
    prompt: str = ""
    backend: Backend = Backend.MOCK
    molecule: MoleculeSpec | None = None
    qm_settings: QMSettings = Field(default_factory=QMSettings)
    qm_tasks: list[QMTask] = Field(default_factory=list)
    trajectory: TrajectorySpec | None = None
    statmech_tasks: list[StatMechTask] = Field(default_factory=list)
    analysis: AnalysisSettings = Field(default_factory=AnalysisSettings)
    output_dir: Path = Path("runs/default")
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_task(self) -> "WorkflowPlan":
        if not self.qm_tasks and not self.statmech_tasks:
            raise ValueError("WorkflowPlan requires at least one qm_task or statmech_task.")
        return self

    def to_yamlable(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def task_value(task: Any) -> str:
    return task.value if isinstance(task, Enum) else str(task)

# ChemAgent-QSM

ChemAgent-QSM is a repo-style implementation of an LLM-guided, auditable chemistry agent that converts natural-language chemistry requests into validated quantum-chemistry and statistical-mechanics workflows.

It is designed around four commitments:

1. **Natural language to validated workflow plans.** The default planner is deterministic and inspectable; LLM planning can be plugged in later without giving the model unsafe execution control.
2. **Quantum chemistry via PySCF-compatible runners.** The production runner supports PySCF SCF/DFT, geometry optimization, descriptors, frequencies, and broadened IR spectra. A deterministic mock runner is included for tests and offline demos.
3. **Disordered condensed-phase analysis.** The statistical-mechanics layer computes time correlation functions, MSD, self-intermediate scattering functions, velocity autocorrelation, local order, relaxation, mobility fields, structure/dynamics coupling, non-Gaussian parameters, and MSCOPE-style multi-scale features.
4. **Auditable generated Python workflows.** Every run emits the normalized plan, validation report, DAG, SLURM script, generated rerunnable Python, JSON/CSV artifacts, and an audit log.

## Install

```bash
cd chemagent-qsm
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
# For real quantum chemistry:
# pip install -e ".[qm,dev]"
```

## Smoke tests

```bash
pytest -q
python scripts/make_synthetic_glass.py --out examples/synthetic_glass.npz
chemagent-qsm run --config examples/water_qm.yaml --backend mock --out runs/water_mock
chemagent-qsm run --config examples/glass_statmech.yaml --out runs/glass_mock
chemagent-qsm run --config examples/full_cv_objective_smoke.yaml --backend mock --out runs/full_cv_objective_smoke
chemagent-qsm cv-check --out runs/full_cv_objective_smoke
```

## Natural-language planning

```bash
chemagent-qsm plan \
  "Optimize caffeine with B3LYP/6-31g*, compute descriptors and IR spectrum" \
  --backend mock \
  --out runs/caffeine_plan \
  --save examples/generated_caffeine.yaml
```

## Real PySCF run

```bash
chemagent-qsm run --config examples/water_qm.yaml --backend pyscf --out runs/water_pyscf
```

For GPU acceleration with GPU4PySCF installed, set `use_gpu: true` in the YAML plan.

## Repository map

```text
src/chemagent_qsm/
  schema.py              Pydantic workflow schema and scientific settings
  planner.py             Natural-language heuristic planner and YAML I/O
  agents/                Optional LLM planning adapter with repair/fallback loops
  validators.py          Static validation and safety checks
  orchestrator.py        End-to-end workflow executor
  qm/                    Mock and PySCF quantum chemistry runners
  statmech/              Correlation, relaxation, local-order, MSCOPE features
  ml/                    ML-ready flattening of trajectory-analysis outputs
  workflow/              DAG and auditable generated Python workflow emission
  hpc/                   SLURM script generation
  evaluation/            CV objective coverage checks
  artifacts.py           Artifact hashing/reproducibility manifest
  io/                    Molecule and trajectory loaders
examples/                Minimal QM and trajectory-analysis workflows
benchmarks/              Planning/execution benchmark specification
tests/                   Unit and integration tests
```

## Scientific outputs

A typical run emits:

- `plan.json`, `plan.yaml`, `validation.json`
- `generated_workflow.py`
- `workflow_dag.json`
- `run.slurm`
- `qm/single_point.json`, `qm/optimize.json`, `qm/descriptors.json`, `qm/frequencies.json`, `qm/ir_spectrum.csv`
- `statmech/*.json`, `statmech/*.csv`
- `audit.jsonl`, `artifact_manifest.json`, `cv_objective_coverage.json`, `results.json`, `report.md`

## Safety and reproducibility design

The planner is deliberately separated from execution. Plans are validated before any backend is invoked. PySCF, RDKit, geomeTRIC, and GPU4PySCF are optional and imported lazily. A mock runner makes CI deterministic while preserving the same data contracts as production runs.

## CV objective QA

The repository includes an explicit objective-coverage gate for the ChemAgent-QSM CV lines. The complete offline run is:

```bash
pytest -q
chemagent-qsm run --config examples/full_cv_objective_smoke.yaml --backend mock --out runs/full_cv_objective_smoke
chemagent-qsm cv-check --out runs/full_cv_objective_smoke
```

That run emits every artifact required by `docs/CV_OBJECTIVE_COVERAGE.md`: validated plans, PySCF-compatible QM artifacts, optimized geometry, electronic descriptors, vibrational/IR outputs, TCFs, MSD, SISF, local order, relaxation, mobility, structure/dynamics coupling, MSCOPE, dynamical-heterogeneity metrics, ML-ready trajectory features, audit logs, and artifact hashes.

## Production notes

The mock backend proves reproducibility and artifact contracts. Use `backend: pyscf` for scientific quantum-chemistry results. PySCF supports Kohn-Sham DFT setup, density fitting, and geometry optimization through geomeTRIC/PyBerny; GPU4PySCF can accelerate compatible SCF/DFT workflows through `to_gpu()`.


## Final benchmark/test hardening pass

I added 600 generated first-person test cases under `tests/generated/chemagent_qsm_600_testcases.jsonl` and a pytest dispatcher that executes them. I also added `chemagent-qsm benchmark-check` and `evaluation/benchmark_report.json`, which compare the full implementation against simpler baselines such as manual scripts, QM-only notebooks, stat-mech-only pipelines, and unconstrained LLM code generation.

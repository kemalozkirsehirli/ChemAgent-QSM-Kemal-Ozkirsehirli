# ChemAgent-QSM Execution Blueprint

## System architecture

ChemAgent-QSM is split into five layers.

1. **Planning layer:** Natural-language prompts are converted into strict `WorkflowPlan` objects. The deterministic planner is the safe default. The LLM adapter is intentionally limited to JSON plan proposal and repair; it never executes Python or shell code.
2. **Validation layer:** `WorkflowValidator` checks required molecule/trajectory inputs, backend availability, PySCF/RDKit/GPU4PySCF dependency constraints, and task dependency warnings.
3. **Execution layer:** `ChemAgentQSM` runs the validated plan through either the deterministic mock runner or the production PySCF runner. The same artifact contract is used in both modes.
4. **Analysis layer:** QM modules compute single-point energies, optimized geometries, descriptors, vibrational frequencies, and broadened IR spectra. Statistical-mechanics modules compute MSD, SISF, VACF, RDF, TCFs, local order, relaxation, mobility, non-Gaussian parameters, MSCOPE, and dynamical-heterogeneity summaries.
5. **Audit layer:** Every run writes normalized plans, generated Python, DAG JSON, SLURM scripts, audit logs, artifact hashes, objective coverage, and reports.

## Production execution steps

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[qm,llm,dev]"
pytest -q
chemagent-qsm run --config examples/caffeine_pyscf.yaml --backend pyscf --out runs/caffeine_pyscf
chemagent-qsm run --config examples/full_cv_objective_smoke.yaml --backend pyscf --out runs/full_cv_objective_pyscf
chemagent-qsm cv-check --out runs/full_cv_objective_pyscf
```

## HPC execution

Every run emits `run.slurm`. For production jobs, edit wall time, memory, partition, GPU directives, and environment activation as needed. The generated Python workflow contains a SHA-256 checksum of the serialized scientific plan and revalidates the plan at runtime.

## Extension points

- Replace `HeuristicChemPlanner` with `LLMPlanningAdapter` connected to a provider or local model.
- Add more PySCF properties, such as TDDFT, solvation, gradients, or QM/MM.
- Add model-based trajectory classifiers on top of `statmech/ml_features.csv`.
- Add experiment-specific calibration datasets for drug-like molecules or glass-forming trajectories.

# ChemAgent-QSM CV Objective Coverage

This document is the final implementation QA matrix for the ChemAgent-QSM repository. It maps each CV-level claim to concrete code paths, emitted artifacts, and tests.

## Objective matrix

| CV-level objective | Implementation path | Run artifacts | Test / QA gate |
|---|---|---|---|
| LLM-guided agentic framework that converts natural-language chemistry prompts into validated quantum-chemical pipelines | `chemagent_qsm.planner.HeuristicChemPlanner`, `chemagent_qsm.agents.LLMPlanningAdapter`, `chemagent_qsm.validators.WorkflowValidator` | `plan.json`, `plan.yaml`, `validation.json`, `workflow_dag.json` | `tests/test_planner.py`, `tests/test_cv_objectives.py` |
| PySCF-compatible electronic-structure analysis of drug-like molecules | `chemagent_qsm.qm.pyscf_runner.PySCFRunner`, `chemagent_qsm.io.molecule_io.atom_block_from_smiles`, optional RDKit 3D generation | `qm/single_point.json`, `qm/descriptors.json` | `tests/test_workflow.py`; production backend validates PySCF availability |
| Optimized geometries via ab initio DFT workflow | `PySCFRunner.optimize` with PySCF geomeTRIC/PyBerny support; mock runner preserves contract offline | `qm/optimize.json` | `examples/water_qm.yaml`, `examples/full_cv_objective_smoke.yaml` |
| Electronic-structure descriptors | `chemagent_qsm.qm.descriptors.summarize_orbitals`, PySCF dipole/Mulliken population extraction, conceptual-DFT descriptors | `qm/descriptors.json` | `tests/test_workflow.py` |
| Vibrational frequencies and IR spectra | `PySCFRunner.frequencies`, `chemagent_qsm.qm.spectra.gaussian_ir_spectrum` | `qm/frequencies.json`, `qm/ir_spectrum.json`, `qm/ir_spectrum.csv` | `tests/test_workflow.py` |
| Statistically generated, auditable Python workflows | `chemagent_qsm.workflow.codegen.generate_python_workflow`, checksum validation, `AuditLogger`, artifact manifest | `generated_workflow.py`, `audit.jsonl`, `artifact_manifest.json` | `tests/test_workflow.py`, `tests/test_cv_objectives.py` |
| Time correlation functions for disordered condensed phase | `chemagent_qsm.statmech.correlations.time_correlation` | `statmech/tcf.json` | `examples/full_cv_objective_smoke.yaml` |
| Local order metric | `chemagent_qsm.statmech.order.steinhardt_q_l` | `statmech/local_order.json` | `tests/test_statmech.py`, full CV smoke |
| Relaxation time scale | `chemagent_qsm.statmech.dynamics.relaxation_time` from SISF threshold crossing | `statmech/relaxation.json` | Full CV smoke |
| Mobility field | `chemagent_qsm.statmech.dynamics.mobility_field` | `statmech/mobility.json` | Full CV smoke |
| Structure/dynamics coupling | `chemagent_qsm.statmech.order.structure_dynamics_correlation` | `statmech/structure_dynamics_coupling.json` | Full CV smoke |
| Statistical-mechanics baselines: MSD, SISF, MSCOPE | `mean_squared_displacement`, `self_intermediate_scattering`, `mscope_features` | `statmech/msd.json`, `statmech/sisf.json`, `statmech/mscope.json` | `tests/test_statmech.py`, full CV smoke |
| LLM-trajectory features and ML-ready analysis workflows for dynamical heterogeneity | `chemagent_qsm.ml.trajectory_features.flatten_statmech_features`, `dynamical_heterogeneity_metrics` | `statmech/ml_feature_row.json`, `statmech/ml_features.csv`, `statmech/dynamical_heterogeneity.json` | `tests/test_cv_objectives.py` |

## Definition of “complete” in this repository

A complete offline QA pass is:

```bash
pytest -q
chemagent-qsm run --config examples/full_cv_objective_smoke.yaml --backend mock --out runs/full_cv_objective_smoke
chemagent-qsm cv-check --out runs/full_cv_objective_smoke
```

A complete production PySCF pass is the same scientific workflow with `backend: pyscf`, PySCF installed, RDKit installed for SMILES-only molecules, and geomeTRIC or PyBerny installed for geometry optimization.

## Scientific boundary conditions

The mock backend is deterministic and exists for testing artifact contracts without requiring quantum-chemistry dependencies. Scientific conclusions must use the PySCF backend or another validated quantum-chemistry backend. Koopmans-style descriptors are screening features, not calibrated potency, selectivity, or ADMET predictors. MSCOPE, non-Gaussian, four-point susceptibility, and mobility metrics are comparative dynamical-heterogeneity features, not universal glass-transition classifiers without system-specific calibration.

# Literature and Engineering Grounding

ChemAgent-QSM is grounded in three methodological families.

## Chemistry agents

The agent design follows the modern chemistry-agent pattern: an LLM proposes or selects tool-using steps, but execution is constrained by typed tools, validation, and auditable artifacts. ChemAgent-QSM keeps the model away from arbitrary code execution and requires conversion into a Pydantic workflow schema.

## PySCF quantum chemistry

The production backend is PySCF-compatible and supports DFT/HF setup, density fitting, geometry optimization, Hessian-based frequency analysis, and property extraction. PySCF, geomeTRIC/PyBerny, RDKit, and GPU4PySCF are optional dependencies so that tests remain deterministic while production runs can use the real backend.

## Statistical mechanics and glassy dynamics

The trajectory-analysis layer implements standard baseline observables for disordered condensed-phase dynamics: MSD, self-intermediate scattering functions, velocity autocorrelation, time correlation functions, RDF, Steinhardt local order, alpha-relaxation time scales, mobility fields, non-Gaussian parameters, overlap/four-point susceptibility, and MSCOPE-style multi-scale features.

## Engineering constraints

The repository prioritizes reproducibility over hidden automation. Every scientific action is represented in `WorkflowPlan`, every workflow is validated before execution, and every run emits machine-readable artifacts and a human-readable report.

# ChemAgent-QSM Benchmark and Test Protocol

I use this file as my implementation-facing QA contract for ChemAgent-QSM. The goal is not only to make the package import and run, but to prove that the system I built matches the research objectives I claim: natural-language planning, validated PySCF-compatible quantum chemistry, auditable generated workflows, statistical-mechanics trajectory analysis, and ML-ready trajectory features.

## Test case matrix

I added a generated first-person test matrix at `tests/generated/chemagent_qsm_600_testcases.jsonl`. Each case is written as something I expect my system to satisfy. The pytest dispatcher in `tests/test_generated_contract_suite.py` executes the matrix directly.

The matrix covers planner dependency resolution, molecule/method/basis inference, deterministic mock-QM invariants, descriptor and IR-spectrum outputs, statistical-mechanics shape/value invariants, validator behavior, workflow DAG/code-generation contracts, and benchmark scoring.

## Baselines I compare against

I benchmark the full system against five deliberately simpler implementation baselines:

1. Manual Python script without agent validation.
2. QM-only PySCF notebook.
3. Classical statistical-mechanics-only analysis pipeline.
4. Unconstrained LLM prompt-to-code workflow without a safety contract.
5. Fragmented QM plus MD toolchain without a single auditable artifact contract.

My full system must beat the best baseline while also passing CV-objective coverage, descriptor completeness, ML-feature completeness, and audit/DAG gates. The benchmark is deterministic engineering QA; real scientific validation still requires real PySCF backends, reference calculations, and physical data.

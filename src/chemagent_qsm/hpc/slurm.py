from __future__ import annotations

from pathlib import Path

from chemagent_qsm.schema import WorkflowPlan


def write_slurm_script(plan: WorkflowPlan, path: Path, job_name: str | None = None) -> Path:
    job_name = job_name or plan.id[:24]
    code = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={job_name}.%j.out
#SBATCH --error={job_name}.%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task={plan.qm_settings.num_threads or 4}
#SBATCH --mem={plan.qm_settings.memory_mb}M

set -euo pipefail
python generated_workflow.py
"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(code, encoding="utf-8")
    path.chmod(0o755)
    return path

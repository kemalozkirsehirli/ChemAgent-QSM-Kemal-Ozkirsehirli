#!/bin/bash
#SBATCH --job-name=full-cv-objective-smoke
#SBATCH --output=full-cv-objective-smoke.%j.out
#SBATCH --error=full-cv-objective-smoke.%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=4000M

set -euo pipefail
python generated_workflow.py

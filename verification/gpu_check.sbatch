#!/bin/bash
# GPU-vs-CPU equality check (verification/gpu_check.py) as a batch job.
# Needs the CPU runs of verification/run_all.sh first. Set the partition and
# limits for your cluster, e.g.
#   sbatch -p <gpu-partition> --gres=gpu:1 verification/gpu_check.sbatch
# Or run it directly on a GPU node:  python verification/gpu_check.py
# Environment: TORCHVIC_DATA_ROOT (required), TORCHVIC_PYTHON (default python).
#SBATCH --job-name=gpu_check
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=gpu_check_%j.out
set -euo pipefail
: "${TORCHVIC_DATA_ROOT:?set TORCHVIC_DATA_ROOT}"
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."
exec "${TORCHVIC_PYTHON:-python}" verification/gpu_check.py "$@"

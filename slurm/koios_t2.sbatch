#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=slurm/%x-%j.out
# T2 convergence run: env SCAN=t2, N, H, SMOOTH_FORCE, TRAVEL_FRACTION, TAG, MQ_LIST are passed with sbatch --export
set -eo pipefail
cd ~/projects/research_task
source slurm/koios_env.sh
python -u tests/sink_friction_scan.py 2>&1 | tr "\r" "\n" | grep -av "Simulation Progress\|Warning\|_interp\|^0.94\|^12.5\|sol massss\|^$"
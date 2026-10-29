#!/bin/bash
#SBATCH --job-name=energy_A2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/%x-%j.out
set -eo pipefail
cd ~/projects/research_task
source slurm/koios_env.sh
python -u tests/test_sink_pair_smoothed.py 2>&1 | tr "\r" "\n" | grep -a "analytic\|PASS\|Error\|assert"
python -u tests/sink_energy_conservation.py 2>&1 | tr "\r" "\n" | grep -aE "PART|^ +[0-9]|Error|Trace|eps|soft"
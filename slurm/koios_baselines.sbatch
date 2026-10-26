#!/bin/bash
#SBATCH --job-name=baselines_p100
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/%x-%j.out
set -eo pipefail
cd ~/projects/research_task
module purge
source /cvmfs/software.eessi.io/versions/2025.06/init/lmod/bash >/dev/null 2>&1
module load Python/3.12.3-GCCcore-13.3.0 SciPy-bundle/2024.05-gfbf-2024a matplotlib/3.9.2-gfbf-2024a
source ~/venvs/uldm/bin/activate
source ~/venvs/uldm/cuda_libs.sh
B=resources/diagnostics/baselines_koios_p100
echo "== $(hostname) $(date) commit $(git rev-parse --short HEAD)"; nvidia-smi -L
echo "== baseline";             python -u chats_playground.py baseline --baseline-dir $B/chat_regression_baselines
echo "== diagnostics-baseline"; python -u chats_playground.py diagnostics-baseline --baseline-dir $B/chat_regression_baselines
echo "== optimization before";  python -u resources/diagnostics/check_optimization_rtx3070.py before $B/optimization_checks
echo "== reproducibility: compare / diagnostics-check / optimization after"
python -u chats_playground.py compare --baseline-dir $B/chat_regression_baselines
python -u chats_playground.py diagnostics-check --baseline-dir $B/chat_regression_baselines
python -u resources/diagnostics/check_optimization_rtx3070.py after $B/optimization_checks
echo "== done $(date)"

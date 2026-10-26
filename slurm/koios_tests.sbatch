#!/bin/bash
#SBATCH --job-name=tests_p100
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/%x-%j.out
set -eo pipefail
cd ~/projects/research_task
module purge
source /cvmfs/software.eessi.io/versions/2025.06/init/lmod/bash >/dev/null 2>&1
module load Python/3.12.3-GCCcore-13.3.0 SciPy-bundle/2024.05-gfbf-2024a matplotlib/3.9.2-gfbf-2024a
source ~/venvs/uldm/bin/activate
source ~/venvs/uldm/cuda_libs.sh
D=resources/diagnostics
echo "== $(hostname) commit $(git rev-parse --short HEAD) (+ working tree)"
echo "== tests";   python -u tests/run_all.py
echo "== compare against Koios P100 set"
python -u chats_playground.py compare --baseline-dir $D/baselines_koios_p100/chat_regression_baselines
python -u chats_playground.py diagnostics-check --baseline-dir $D/baselines_koios_p100/chat_regression_baselines
echo "== Quadro P4000 set vs Koios P100 set"
python resources/diagnostics/compare_baseline_sets.py $D/baselines_quadro_p4000 $D/baselines_koios_p100
echo "== done"

#!/bin/bash
#SBATCH --job-name=t5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/%x-%j.out
# T5: set-up of arXiv:2403.09038, see tests/t5_bh_soliton.py. Parameters through environment variables, e.g.
#   sbatch --export=ALL,T_MYR=60,SMOOTH=1 slurm/koios_t5.sbatch      (a value must not contain a comma)
set -eo pipefail
cd ~/projects/research_task
module purge
source /cvmfs/software.eessi.io/versions/2025.06/init/lmod/bash >/dev/null 2>&1
module load Python/3.12.3-GCCcore-13.3.0 SciPy-bundle/2024.05-gfbf-2024a matplotlib/3.9.2-gfbf-2024a
source ~/venvs/uldm/bin/activate
source ~/venvs/uldm/cuda_libs.sh
echo "== $(hostname) commit $(git rev-parse --short HEAD)"
echo n | python -u tests/t5_bh_soliton.py

#!/bin/bash
#SBATCH --job-name=cfl_repro
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/%x-%j.out
cd ~/projects/research_task
module purge
source /cvmfs/software.eessi.io/versions/2025.06/init/lmod/bash >/dev/null 2>&1
module load Python/3.12.3-GCCcore-13.3.0 SciPy-bundle/2024.05-gfbf-2024a matplotlib/3.9.2-gfbf-2024a
source ~/venvs/uldm/bin/activate
source ~/venvs/uldm/cuda_libs.sh
for v in "$@"; do echo "######## $v"; python -u resources/diagnostics/cfl_disk_repro.py $v 2>&1 | tr "\r" "\n" | grep -av "Progress\|^$"; done

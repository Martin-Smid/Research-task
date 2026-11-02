#!/bin/bash
#SBATCH --partition=rocky10_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=pyul_t1
# CPU-only job (PyUltraLight is numexpr/pyfftw); no --gres, the GPU is not requested.
# usage: sbatch [opts] job.sbatch <run_friction.py args>
module purge
source /cvmfs/software.eessi.io/versions/2025.06/init/lmod/bash >/dev/null 2>&1
module load Python/3.12.3-GCCcore-13.3.0 SciPy-bundle/2024.05-gfbf-2024a matplotlib/3.9.2-gfbf-2024a
source ~/projects/pyultralight/venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd ~/projects/pyultralight
python -u run_friction.py "$@"
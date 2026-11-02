#!/bin/bash
#SBATCH --job-name=tests_all_spin_P100
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=slurm/%x-%j.out
# Everything that guards the code: tests/run_all.py, the spin tests, the bitwise regression comparison against the Koios P100 set and the
# diagnostics check. A failing step does not stop the others; the summary at the end lists what failed.
set -o pipefail
cd ~/projects/research_task
module purge
source /cvmfs/software.eessi.io/versions/2025.06/init/lmod/bash >/dev/null 2>&1
module load Python/3.12.3-GCCcore-13.3.0 SciPy-bundle/2024.05-gfbf-2024a matplotlib/3.9.2-gfbf-2024a
source ~/venvs/uldm/bin/activate
source ~/venvs/uldm/cuda_libs.sh
D=resources/diagnostics
FAILED=""
step() { local n="$1"; echo; echo "== $n"; shift; "$@" || FAILED="$FAILED | $n"; }
echo "== $(hostname) commit $(git rev-parse --short HEAD) (+ working tree)"
step run_all python -u tests/run_all.py
for t in tests/spin/test_spin_*.py; do step "$t" python -u "$t"; done
step compare python -u chats_playground.py compare --baseline-dir $D/baselines_koios_p100/chat_regression_baselines
step diagnostics-check python -u chats_playground.py diagnostics-check --baseline-dir $D/baselines_koios_p100/chat_regression_baselines
echo; echo "== SUMMARY: failed steps: ${FAILED:-none}"

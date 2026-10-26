#!/bin/bash
#SBATCH --job-name=check_p100
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/%x-%j.out
# Re-verify the current code against the P100 reference set (all must say bitwise identical / exact).
set -eo pipefail
cd ~/projects/research_task
source slurm/koios_env.sh
B=resources/diagnostics/baselines_koios_p100
python -u tests/run_all.py 2>&1 | tr "\r" "\n" | grep -a "ALL TESTS\|FAIL\|Error"
python -u chats_playground.py compare --baseline-dir $B/chat_regression_baselines 2>&1 | grep -a "^PASS\|FAIL"
python -u chats_playground.py diagnostics-check --baseline-dir $B/chat_regression_baselines 2>&1 | grep -a "^PASS\|FAIL"
python -u resources/diagnostics/check_optimization_rtx3070.py after $B/optimization_checks 2>&1 | grep -a "^CHECK\|Error"
echo "== done"

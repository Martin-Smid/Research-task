#!/bin/bash
#SBATCH --job-name=regen_p100
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=slurm/%x-%j.out
set -eo pipefail
cd ~/projects/research_task
source slurm/koios_env.sh
NEW=$HOME/p100_new
echo "== tests"; python -u tests/run_all.py 2>&1 | tr "\r" "\n" | grep -a "^PASS\|FAIL\|Error\|ALL TESTS" | tail -30
echo "== new baselines (current code) into $NEW"
python -u chats_playground.py baseline --baseline-dir $NEW/chat_regression_baselines | tail -1
python -u chats_playground.py diagnostics-baseline --baseline-dir $NEW/chat_regression_baselines 2>&1 | grep -a "^PASS"
python -u resources/diagnostics/check_optimization_rtx3070.py before $NEW/optimization_checks 2>&1 | grep -a "^CHECK"
echo "== old P100 set vs new, relative L2 difference"
python resources/diagnostics/compare_baseline_sets.py resources/diagnostics/baselines_koios_p100 $NEW
echo "== done"

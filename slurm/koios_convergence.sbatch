#!/bin/bash
#SBATCH --job-name=convergence
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=20:00:00
#SBATCH --output=slurm/%x-%j.out
# Convergence test of main.py (relaxation + galaxy) at N = 64 and N = 256 (N = 128 is the normal main.py run).
# Scripts ~/conv/main_N<N>.py are main.py with N, H and the save intervals changed (h shrinks with dx^2 for stability).
cd ~/projects/research_task
source slurm/koios_env.sh
for n in 64 256; do
  echo "######## N=$n  start $(date)"
  PYTHONPATH=$PWD python -u ~/conv/main_N$n.py > ~/conv/log_N$n.txt 2>&1
  echo "######## N=$n  exit $?  end $(date)"
  tr "\r" "\n" < ~/conv/log_N$n.txt | grep -a "FÁZE\|Relaxace uložena\|^resources/data\|černá díra:\|Error\|Traceback"
done

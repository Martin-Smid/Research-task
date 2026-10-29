#!/bin/bash
#SBATCH --job-name=dressed_test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/%x-%j.out
cd ~/projects/research_task
source slurm/koios_env.sh
python -u tests/test_dressed_soliton.py 2>&1 | tr "\r" "\n" | grep -av "Warning\|_interp\|^0.94\|^12.5\|sol massss\|^$\|hbartilde\|Time step\|pracuji"
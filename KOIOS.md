# KOIOS.md — running this project on the Koios cluster (FZU CEICO)

Read this first when a session is about Koios. Cluster docs: https://hpc.ceico.cz

## Layout on Koios (user `smidmar9`)
- Code: `~/projects/research_task` (branch `working_with_chat`, cloned from https://github.com/Martin-Smid/Research-task, public, no login needed). Update with `git pull`.
- Python venv: `~/venvs/uldm`. Rebuild (on a GPU node): `srun -p gpu --gres=gpu:1 -c 4 --mem=8G -t 00:25:00 bash -c "bash ~/fix.sh && ..."`; full recipe in `~/venv.sh` (creates it), `~/fix.sh` (CUDA wheels + `cuda_libs.sh`), then `pip install -c ~/venvs/uldm/constraints.txt tqdm`. Run `venv.sh` first, its final self-test is expected to fail until `fix.sh` has installed the CUDA wheels.
- Home has no backup. Simulation output (`resources/data/`) is huge: delete it when done.

## Hardware and software facts
- GPU partition `gpu`: nodes gpu3, gpu4, 2x Tesla P100 16 GB each (compute capability 6.0), Rocky 10. No `nvcc`, no EESSI CUDA/CuPy modules; CuPy 13.3.0 (`cupy-cuda12x[ctk]`) brings its own CUDA libs from pip `nvidia-*-cu12` wheels.
- Environment: EESSI 2025.06 modules `Python/3.12.3-GCCcore-13.3.0 SciPy-bundle/2024.05-gfbf-2024a matplotlib/3.9.2-gfbf-2024a`, venv with `--system-site-packages`. `numpy` must stay `<2` (bundled scipy breaks otherwise).
- Do not pass `--account`; always set `--time` and `--mem` (no default limit exists).
- Job script header/body to copy: `slurm/koios_baselines.sbatch`, `slurm/koios_tests.sbatch`. Core of it:
  ```bash
  module purge
  source /cvmfs/software.eessi.io/versions/2025.06/init/lmod/bash >/dev/null 2>&1
  module load Python/3.12.3-GCCcore-13.3.0 SciPy-bundle/2024.05-gfbf-2024a matplotlib/3.9.2-gfbf-2024a
  source ~/venvs/uldm/bin/activate
  source ~/venvs/uldm/cuda_libs.sh
  ```
- In batch jobs `evolve()` with `save_max_vals=True` ends in `input()`: use `echo n | python -u script.py`.

## Submit / monitor
```bash
sbatch slurm/koios_tests.sbatch                         # from ~/projects/research_task
squeue -u $USER -o "%i %T %N %M"
sacct -j <ID> -X --format=JobID,State,Elapsed,ExitCode
tr "\r" "\n" < slurm/<name>-<ID>.out | tail -20          # tqdm uses \r
```
Interactive test on a GPU node: `srun -p gpu --gres=gpu:1 -c 4 --mem=16G -t 00:30:00 --pty bash`.

## Reference sets
`resources/diagnostics/baselines_koios_p100/` was created on gpu4 (P100) at commit d19ad64 and reproduces bitwise run to run. Other sets (Quadro P4000, RTX 3070, GTX 1050 Ti) are other hardware; compare sets with `resources/diagnostics/compare_baseline_sets.py <dirA> <dirB>`.

## Connecting from the Windows PC (Claude Code / terminal)
- `~/.ssh/config` has `Host koios` (HostName koios1.fzu.cz, User `smidmar9`, key `~/.ssh/id_ed25519_fzu`, passphrase-protected).
- Claude cannot type the passphrase. The Windows `ssh-agent` service needs admin to start, so instead start a Git Bash agent and let the user add the key themselves:
  1. Claude (Git Bash): `rm -f /tmp/koios-agent.sock; ssh-agent -a /tmp/koios-agent.sock`
  2. User, in a terminal tab (types the passphrase): `& 'C:\Program Files\Git\bin\bash.exe' -c "SSH_AUTH_SOCK=/tmp/koios-agent.sock ssh-add ~/.ssh/id_ed25519_fzu"`
  3. Claude, every Bash call: `export SSH_AUTH_SOCK=/tmp/koios-agent.sock; ssh koios '...'` (use Git Bash `ssh`, not Windows `ssh.exe`; the agent is gone after a reboot or when the app restarts, repeat 1-2).
- The user also has VS Code Remote-SSH open on `koios` -> `/home/smidmar9/projects/research_task` to browse plots and outputs.
- Never ask for or store the passphrase, never read `~/.ssh` keys.

## Git on Koios
Same rules as `CLAUDE.md` / `CONTINUATION.md`: work on `working_with_chat`, commits only in the user's name (no Co-Authored-By), propose a message and wait for OK, push only when told. Git identity on Koios is not configured yet: set `git config user.name`/`user.email` for this repo only after asking the user.

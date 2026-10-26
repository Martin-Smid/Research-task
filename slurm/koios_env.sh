# source this: loads the EESSI Python modules (they provide numpy/scipy/matplotlib) and activates the venv
# "module purge" must run BEFORE EESSI's Lmod is initialised (purging after it would also unload EESSI itself).
# Its errors about modules inherited from the login shell (Magic cookie '#%Module' missing) are harmless, hidden here.
module purge >/dev/null 2>&1
source /cvmfs/software.eessi.io/versions/2025.06/init/lmod/bash >/dev/null 2>&1
module load Python/3.12.3-GCCcore-13.3.0 SciPy-bundle/2024.05-gfbf-2024a matplotlib/3.9.2-gfbf-2024a
source ~/venvs/uldm/bin/activate
source ~/venvs/uldm/cuda_libs.sh

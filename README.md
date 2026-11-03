# ULDM spectral solver with higher integer spin, gas and N-body + sink particles

GPU (CuPy) simulation framework for ultralight (fuzzy) dark matter in 1D, 2D and 3D on a periodic grid.
The dark-matter field is evolved with a pseudo-spectral (FFT) split-step solution of the Schrödinger–Poisson
system, with time-stepping orders 2, 4 and 6. The field can carry an arbitrary non-negative integer spin
(a spin-s field has 2s+1 components), and gravity is solved with the FFT Poisson solver on the same grid.

Baryonic components can be added and coupled to the dark matter only through gravity: collisionless
N-body particles (stars), a finite-volume gas (`NBodyGas`: MUSCL reconstruction with a Rusanov flux),
and sink particles for black holes (analytic softened potential, optional gas accretion, sink-sink gravity
and merging). Runs write snapshots, energy / mass diagnostics and checkpoints, and can be resumed or
continued in new segments. Units are kpc, Msun and Gyr (particle mass `m_s` in eV/c²).

The regression tests include analytic checks: Sod shock tube, Jeans instability, free-fall collapse,
Kepler orbits, and dynamical friction against linear theory (see `tests/README.md`).

## Installation

### Prerequisites

An **NVIDIA GPU is required**, because the solvers run on CuPy. Install, in this order:

1. **NVIDIA driver** (the normal gaming driver is enough). Check with `nvidia-smi`.
2. **CUDA Toolkit 12.x** from <https://developer.nvidia.com/cuda-toolkit-archive> (pick a 12.x release)
   (tested with 12.6). Check with `nvcc --version`. CUDA 13 is not supported (not tested).
3. **Python 3.11** (tested with 3.11.4).

### Install

```bash
git clone https://github.com/Martin-Smid/Research-task.git -b working_with_chat
cd Research-task
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -e ".[cuda12]"
python tests/run_all.py
```

`pip install -e ".[cuda12]"` installs the Python dependencies together with the `cupy-cuda12x` wheel.
All tests should finish with `PASS` (about one minute on a GPU).

To check that CuPy finds CUDA before running anything else:

```bash
python -c "import cupy; print(cupy.__version__, cupy.cuda.runtime.runtimeGetVersion())"
```

### Running

See [`examples/`](examples/README.md); for instance `python -m examples.initialize_simulation` runs a small
soliton simulation. Output goes to `resources/data/`.

"""Minimal runtime setup shared by the examples."""

import os
from pathlib import Path


def configure_cuda_on_windows():
    """Expose an installed CUDA toolkit before importing CuPy."""
    if os.name != "nt":
        return

    candidates = []
    if os.environ.get("CUDA_PATH"):
        candidates.append(Path(os.environ["CUDA_PATH"]))

    cuda_root = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
    cuda_root /= "NVIDIA GPU Computing Toolkit/CUDA"
    if cuda_root.is_dir():
        candidates.extend(sorted(cuda_root.glob("v*"), reverse=True))

    for candidate in candidates:
        bin_directory = candidate / "bin"
        if bin_directory.is_dir() and any(bin_directory.glob("nvrtc64_*.dll")):
            os.environ["CUDA_PATH"] = str(candidate)
            os.environ["PATH"] = (
                str(bin_directory) + os.pathsep + os.environ.get("PATH", "")
            )
            return

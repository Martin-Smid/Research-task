"""Sphinx configuration. Build locally: sphinx-build -b html docs docs/_build/html"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

project = "ULDM spectral solver"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
]
myst_enable_extensions = ["dollarmath", "amsmath"]

# The API reference must build without a GPU: CuPy is replaced by a mock.
autodoc_mock_imports = ["cupy", "cupyx"]
autodoc_member_order = "bysource"

html_theme = "furo"
exclude_patterns = ["_build"]

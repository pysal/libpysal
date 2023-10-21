"""
libpysal: Python Spatial Analysis Library (core)
================================================


Documentation
-------------
PySAL documentation is available in two forms: python docstrings and an html \
        webpage at http://pysal.org/

Available sub-packages
----------------------

cg
    Basic data structures and tools for Computational Geometry
examples
    Example data sets for testing and documentation
io
    Basic functions used by several sub-packages
weights
    Tools for creating and manipulating weights
"""

import contextlib
from importlib.metadata import PackageNotFoundError, version

from . import cg, examples, io, weights

with contextlib.suppress(PackageNotFoundError):
    __version__ = version("libpysal")

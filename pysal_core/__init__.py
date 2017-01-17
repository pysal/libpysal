"""
Python Spatial Analysis Library
===============================


Documentation
-------------
PySAL documentation is available in two forms: python docstrings and an html \
        webpage at http://pysal.org/

Available sub-packages
----------------------

cg
    Basic data structures and tools for Computational Geometry
io
    Basic functions used by several sub-packages
weights
    Tools for creating and manipulating weights
"""
import cg
import weights
import io
import common

try:
    import pandas
    from io import pdio
    common.pandas = pandas
except ImportError:
    common.pandas = None
    
# Load the IOHandlers
from io import IOHandlers
# Assign pysal.open to dispatcher
open = io.FileIO.FileIO

from version import version

MISSINGVALUE = None  # used by fileIO to flag missing values.

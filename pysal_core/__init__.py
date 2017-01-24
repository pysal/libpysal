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
import io
import weights
import common

try:
    import pandas
    from io import pdio
    common.pandas = pandas
except ImportError:
    common.pandas = None

from io import IOHandlers    
# Assign pysal.open to dispatcher
open = io.FileIO.FileIO

from version import version
MISSINGVALUE = common.MISSINGVALUE

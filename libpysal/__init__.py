__version__ = "3.0.8"

# __version__ has to be define in the first line

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
examples
    Example data sets for testing and documentation
io
    Basic functions used by several sub-packages
weights
    Tools for creating and manipulating weights
"""
from . import cg
from . import io
from . import weights
from . import examples

# Assign pysal.open to dispatcher

#from .version import version as __version__


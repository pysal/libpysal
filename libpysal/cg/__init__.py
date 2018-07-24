"""
A module for computational geometry.
"""
from . import shapes
from .standalone import *
from .locators import *
from .kdtree import *
from .sphere import *
from .voronoi import *
from .alpha_shapes import alpha_shape, alpha_shape_auto

del rtree
del kdtree
del locators
del sphere
del voronoi
del standalone
del alpha_shapes

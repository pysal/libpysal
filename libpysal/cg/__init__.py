"""
A module for computational geometry.
"""
from .shapes import *
from .standalone import *
from .locators import *
from .kdtree import *
from .rtree import *
from .sphere import *
from .voronoi import *
from .alpha_shapes import alpha_shape, alpha_shape_auto

del rtree
del kdtree
del locators
del voronoi
del standalone
del alpha_shapes
del shapes

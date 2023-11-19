import os

import pytest

from ... import examples
from .. import user
from ..contiguity import Rook


class Testuser:
    def test_min_threshold_dist_from_shapefile(self):
        f = examples.get_path("columbus.shp")
        min_d = user.min_threshold_dist_from_shapefile(f)
        assert min_d == pytest.approx(0.61886415807685413)

    def test_build_lattice_shapefile(self):
        of = "lattice.shp"
        user.build_lattice_shapefile(20, 20, of)
        w = Rook.from_shapefile(of)
        assert w.n == 400
        os.remove("lattice.dbf")
        os.remove("lattice.shp")
        os.remove("lattice.shx")

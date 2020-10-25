import os
import unittest
import numpy as np
from ..contiguity import Rook, Voronoi
from .. import user
from ... import examples


class Testuser(unittest.TestCase):
    def test_min_threshold_dist_from_shapefile(self):
        f = examples.get_path("columbus.shp")
        min_d = user.min_threshold_dist_from_shapefile(f)
        self.assertAlmostEqual(min_d, 0.61886415807685413)

    def test_build_lattice_shapefile(self):
        of = "lattice.shp"
        user.build_lattice_shapefile(20, 20, of)
        w = Rook.from_shapefile(of)
        self.assertEqual(w.n, 400)
        os.remove("lattice.shp")
        os.remove("lattice.shx")


suite = unittest.TestLoader().loadTestsFromTestCase(Testuser)

if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite)

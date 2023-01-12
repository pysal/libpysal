import os
import tempfile

import unittest
import pytest
import numpy as np
import geopandas as gpd
from ..weights import W, WSP
from ..user import build_lattice_shapefile
from .. import util
from ..contiguity import Rook, _return_length_weighted_w
from ... import examples

NPTA3E = np.testing.assert_array_almost_equal


class TestPerimeter(unittest.TestCase):
    def setUp(self):
        shp = build_lattice_shapefile(3, 3, "tmp.shp")
        gdf = gpd.read_file("tmp.shp")
        dv = [0] * 3
        dv.extend(list(range(1, 7)))
        gdf["dv"] = dv
        gdf = gdf.dissolve(by="dv")
        self.w0 = Rook.from_dataframe(gdf, perimeter=True)
        self.gdf = gdf

        # us case
        usgdf = gpd.read_file(examples.get_path("us48.shp"))
        usgdf.set_crs("epsg:4326", inplace=True)
        usgdf.to_crs(usgdf.estimate_utm_crs(), inplace=True)
        self.usgdf = usgdf
        self.wus = Rook.from_dataframe(usgdf, perimeter=True)

    def test_perimeter(self):
        NPTA3E(self.w0.pct_nonzero, 40.81632653)

    def test_return_length_weighted(self):
        w1 = _return_length_weighted_w(self.w0, self.gdf)
        NPTA3E(w1.pct_nonzero, 40.81632653)
        self.assertEqual(w1.weights[0], [1, 1, 1])
        self.assertEqual(w1.weights[2], [1, 1, 1, 1])

    def test_return_length_weighted_us(self):
        w1 = _return_length_weighted_w(self.wus, self.usgdf)
        self.assertAlmostEqual(w1[0][7], 354625.0684908081)
        self.assertAlmostEqual(w1[0][10],  605834.5010118643)
        NPTA3E(w1[0][7], w1[7][0])
        w1.transform = "r"
        self.assertAlmostEqual(w1[0][7],  0.3692243585791264)
        self.assertAlmostEqual(w1[7][0],  0.12891667056448083)
        self.assertNotAlmostEquals(w1[0][7], w1[7][0])


if __name__ == "__main__":
    unittest.main()

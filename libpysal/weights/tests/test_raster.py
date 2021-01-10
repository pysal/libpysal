"""Unit test for raster.py"""
from .. import raster
import unittest
import numpy as np
import pandas as pd
from xarray import DataArray


class Testraster(unittest.TestCase):
    def setUp(self):
        self.da1 = raster.testDataArray()
        self.da2 = raster.testDataArray((1, 4, 4), missing_vals=False)
        self.da3 = self.da2.rename(
            {"band": "layer", "x": "longitude", "y": "latitude"})
        self.data1 = pd.Series(np.ones(5))
        self.da4 = raster.testDataArray((1, 1), missing_vals=False)
        self.da4.data = np.array([["test"]])

    def test_da2W(self):
        w1 = raster.da2W(self.da1, "queen", k=2, n_jobs=-1)
        self.assertEqual(w1[(1, -30.0, -180.0)],
                         {(1, -90.0, 60.0): 1, (1, -90.0, -60.0): 1})
        self.assertEqual(w1[(1, -30.0, 180.0)],
                         {(1, -90.0, -60.0): 1, (1, -90.0, 60.0): 1})
        self.assertEqual(w1.n, 5)
        self.assertEqual(w1.index.names, self.da1.to_series().index.names)
        self.assertEqual(w1.index.tolist()[0], (1, 90.0, 180.0))
        self.assertEqual(w1.index.tolist()[1], (1, -30.0, -180.0))
        self.assertEqual(w1.index.tolist()[2], (1, -30.0, 180.0))
        self.assertEqual(w1.index.tolist()[3], (1, -90.0, -60.0))
        w2 = raster.da2W(self.da2, "rook")
        self.assertEqual(
            sorted(w2.neighbors[(1, -90.0, 180.0)]), [(1, -90.0, 60.0), (1, -30.0, 180.0)])
        self.assertEqual(sorted(w2.neighbors[(
            1, -90.0, 60.0)]), [(1, -90.0, -60.0), (1, -90.0, 180.0), (1, -30.0, 60.0)])
        self.assertEqual(w2.n, 16)
        self.assertEqual(w2.index.names, self.da2.to_series().index.names)
        self.assertEqual(w2.index.tolist(),
                         self.da2.to_series().index.tolist())
        coords_labels = {
            "z_label": "layer",
            "y_label": "latitude",
            "x_label": "longitude",
        }
        w3 = raster.da2W(self.da3, z_value=1, coords_labels=coords_labels)
        self.assertEqual(sorted(w3.neighbors[(
            1, -90.0, 180.0)]), [(1, -90.0, 60.0), (1, -30.0, 60.0), (1, -30.0, 180.0)])
        self.assertEqual(w3.n, 16)
        self.assertEqual(w3.index.names, self.da3.to_series().index.names)
        self.assertEqual(w3.index.tolist(),
                         self.da3.to_series().index.tolist())

    def test_da2WSP(self):
        w1 = raster.da2WSP(self.da1, "rook", n_jobs=-1)
        rows, cols = w1.sparse.shape
        n = rows * cols
        pct_nonzero = w1.sparse.nnz / float(n)
        self.assertEqual(pct_nonzero, 0.08)
        data = w1.sparse.todense().tolist()
        self.assertEqual(data[3], [0, 0, 0, 0, 1])
        self.assertEqual(data[4], [0, 0, 0, 1, 0])
        self.assertEqual(w1.index.names, self.da1.to_series().index.names)
        self.assertEqual(w1.index.tolist()[0], (1, 90.0, 180.0))
        self.assertEqual(w1.index.tolist()[1], (1, -30.0, -180.0))
        self.assertEqual(w1.index.tolist()[2], (1, -30.0, 180.0))
        self.assertEqual(w1.index.tolist()[3], (1, -90.0, -60.0))
        w2 = raster.da2WSP(self.da2, "queen", k=2, include_nodata=True)
        w3 = raster.da2WSP(self.da2, "queen", k=2, n_jobs=-1)
        self.assertEqual(w2.sparse.nnz, w3.sparse.nnz)
        self.assertEqual(w2.sparse.todense().tolist(),
                         w3.sparse.todense().tolist())
        self.assertEqual(w2.n, 16)
        self.assertEqual(w2.index.names, self.da2.to_series().index.names)
        self.assertEqual(w2.index.tolist(),
                         self.da2.to_series().index.tolist())

    def test_w2da(self):
        w2 = raster.da2W(self.da2, "rook", n_jobs=-1)
        da2 = raster.w2da(self.da2.data.flatten(), w2,
                          self.da2.attrs, self.da2.coords)
        da_compare = DataArray.equals(da2, self.da2)
        self.assertEqual(da_compare, True)

    def test_wsp2da(self):
        wsp1 = raster.da2WSP(self.da1, "queen")
        da1 = raster.wsp2da(self.data1, wsp1)
        self.assertEqual(da1["y"].values.tolist(),
                         self.da1["y"].values.tolist())
        self.assertEqual(da1["x"].values.tolist(),
                         self.da1["x"].values.tolist())
        self.assertEqual(da1.shape, (1, 4, 4))

    def test_da_checker(self):
        self.assertRaises(ValueError, raster.da2W, self.da4)


suite = unittest.TestLoader().loadTestsFromTestCase(Testraster)

if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite)

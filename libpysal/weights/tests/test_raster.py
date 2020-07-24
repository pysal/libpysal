"""Unit test for raster.py"""
from .. import raster
from ..util import lat2SW
import unittest
import numpy as np
import pandas as pd
from xarray import DataArray


class Testraster(unittest.TestCase):
    def setUp(self):
        data1 = np.array([[[-32768, -32768, -32768, -32768],
                           [-32768, -32768, -32768, -32768],
                           [-32768, -32768, -32768,     82],
                           [-32768,     84,     83,     83]]])
        data2 = np.array([[[89, 88, 79, 78],
                           [84, 81, 88, 78],
                           [84, 81, 88, 82],
                           [84, 84, 83, 83]]])
        dims = ('band', 'y', 'x')
        coords = {'band': [1],
                  'y': [33.452222, 33.451944, 33.451667, 33.451389],
                  'x': [-117.556667, -117.556389, -117.556111, -117.555833]}
        attrs = {'nodatavals': (-32768.0,)}
        self.da1 = DataArray(data1, coords, dims, attrs=attrs)
        self.da2 = DataArray(data2, coords, dims, attrs=attrs)
        self.data1 = pd.Series(np.ones(4))

    def test_da2W(self):
        w1 = raster.da2W(self.da1, "queen")
        self.assertEqual(w1[0], {2: 1, 3: 1})
        self.assertEqual(w1[2], {3: 1, 0: 1, 1: 1})
        self.assertEqual(w1.n, 4)
        self.assertEqual(w1.index.names, self.da1.to_series().index.names)
        self.assertEqual(w1.index.tolist()[0], (1, 33.451667, -117.555833))
        self.assertEqual(w1.index.tolist()[1], (1, 33.451389, -117.556389))
        self.assertEqual(w1.index.tolist()[2], (1, 33.451389, -117.556111))
        self.assertEqual(w1.index.tolist()[3], (1, 33.451389, -117.555833))
        w2 = raster.da2W(self.da2, "rook")
        self.assertEqual(w2[6], {10: 1, 7: 1, 2: 1, 5: 1})
        self.assertEqual(w2.neighbors[2], [6, 3, 1])
        self.assertEqual(w2.n, 16)
        self.assertEqual(w2.index.names, self.da2.to_series().index.names)
        self.assertEqual(w2.index.tolist(), self.da2.to_series().index.tolist())

    def test_da2WSP(self):
        w1 = raster.da2WSP(self.da1, "rook")
        rows, cols = w1.sparse.shape
        n = rows * cols
        pct_nonzero = w1.sparse.nnz / float(n)
        self.assertEqual(pct_nonzero, 0.375)
        data = w1.sparse.todense().tolist()
        self.assertEqual(data[0], [0, 0, 0, 1])
        self.assertEqual(data[1], [0, 0, 1, 0])
        self.assertEqual(data[2], [0, 1, 0, 1])
        self.assertEqual(data[3], [1, 0, 1, 0])
        self.assertEqual(w1.index.names, self.da1.to_series().index.names)
        self.assertEqual(w1.index.tolist()[0], (1, 33.451667, -117.555833))
        self.assertEqual(w1.index.tolist()[1], (1, 33.451389, -117.556389))
        self.assertEqual(w1.index.tolist()[2], (1, 33.451389, -117.556111))
        self.assertEqual(w1.index.tolist()[3], (1, 33.451389, -117.555833))
        w2 = raster.da2W(self.da2, "queen")
        sw = lat2SW(4, 4, "queen")
        self.assertEqual(w2.sparse.nnz, sw.nnz)
        self.assertEqual(w2.sparse.todense().tolist(), sw.todense().tolist())
        self.assertEqual(w2.n, 16)
        self.assertEqual(w2.index.names, self.da2.to_series().index.names)
        self.assertEqual(w2.index.tolist(), self.da2.to_series().index.tolist())

    def test_w2da(self):
        w1 = raster.da2WSP(self.da1, "queen")
        da1 = raster.w2da(self.data1, w1).sortby("y", False)
        self.assertEqual(da1["y"].values.tolist(), self.da1["y"].values.tolist())
        self.assertEqual(da1["x"].values.tolist(), self.da1["x"].values.tolist())
        self.assertEqual(da1.shape, (1, 4, 4))
        w2 = raster.da2W(self.da2, "rook")
        da2 = raster.w2da(self.da2.data.flatten(), w2, self.da2.attrs).sortby("y", False)
        da_compare = DataArray.equals(da2, self.da2)
        self.assertEqual(da_compare, True)


suite = unittest.TestLoader().loadTestsFromTestCase(Testraster)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)

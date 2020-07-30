"""Unit test for raster.py"""
from .. import raster
from ..util import lat2SW
import unittest
import numpy as np
import pandas as pd
from xarray import DataArray


class Testraster(unittest.TestCase):
    def setUp(self):
        self.da1 = raster.testDataArray()
        self.da2 = raster.testDataArray((1, 4, 4), missing_vals=False)
        self.data1 = pd.Series(np.ones(10))

    def test_da2W(self):
        w1 = raster.da2W(self.da1, "queen")
        self.assertEqual(w1[0], {2: 1, 3: 1})
        self.assertEqual(w1[2], {5: 1, 3: 1, 0: 1})
        self.assertEqual(w1.n, 10)
        self.assertEqual(w1.index.names, self.da1.to_series().index.names)
        self.assertEqual(w1.index.tolist()[0], (1, 66.4, 92.1))
        self.assertEqual(w1.index.tolist()[1], (1, 66.4, 92.3))
        self.assertEqual(w1.index.tolist()[2], (1, 66.3, 92.1))
        self.assertEqual(w1.index.tolist()[3], (1, 66.3, 92.2))
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
        self.assertEqual(pct_nonzero, 0.14)
        data = w1.sparse.todense().tolist()
        self.assertEqual(data[0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(data[6], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        self.assertEqual(w1.index.names, self.da1.to_series().index.names)
        self.assertEqual(w1.index.tolist()[0], (1, 66.4, 92.1))
        self.assertEqual(w1.index.tolist()[1], (1, 66.4, 92.3))
        self.assertEqual(w1.index.tolist()[2], (1, 66.3, 92.1))
        self.assertEqual(w1.index.tolist()[3], (1, 66.3, 92.2))
        w2 = raster.da2W(self.da2, "queen")
        sw = lat2SW(4, 4, "queen")
        self.assertEqual(w2.sparse.nnz, sw.nnz)
        self.assertEqual(w2.sparse.todense().tolist(), sw.todense().tolist())
        self.assertEqual(w2.n, 16)
        self.assertEqual(w2.index.names, self.da2.to_series().index.names)
        self.assertEqual(w2.index.tolist(), self.da2.to_series().index.tolist())

    def test_w2da(self):
        w2 = raster.da2W(self.da2, "rook")
        da2 = raster.w2da(self.da2.data.flatten(), w2, self.da2.attrs).sortby("y", False)
        da_compare = DataArray.equals(da2, self.da2)
        self.assertEqual(da_compare, True)

    def test_wsp2da(self):
        wsp1 = raster.da2WSP(self.da1, "queen")
        da1 = raster.wsp2da(self.data1, wsp1).sortby("y", False)
        self.assertEqual(da1["y"].values.tolist(), self.da1["y"].values.tolist())
        self.assertEqual(da1["x"].values.tolist(), self.da1["x"].values.tolist())
        self.assertEqual(da1.shape, (1, 4, 4))


suite = unittest.TestLoader().loadTestsFromTestCase(Testraster)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)

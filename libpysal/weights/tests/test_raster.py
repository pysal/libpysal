"""Unit test for raster.py"""

import numpy as np
import pandas as pd
import pytest

from .. import raster


class Testraster:
    def setup_method(self):
        pytest.importorskip("xarray")
        self.da1 = raster.testDataArray()
        self.da2 = raster.testDataArray((1, 4, 4), missing_vals=False)
        self.da3 = self.da2.rename({"band": "layer", "x": "longitude", "y": "latitude"})
        self.data1 = pd.Series(np.ones(5))
        self.da4 = raster.testDataArray((1, 1), missing_vals=False)
        self.da4.data = np.array([["test"]])

    def test_da2_w(self):
        w1 = raster.da2W(self.da1, "queen", k=2, n_jobs=-1)
        assert w1[(1, -30.0, -180.0)] == {(1, -90.0, 60.0): 1, (1, -90.0, -60.0): 1}
        assert w1[(1, -30.0, 180.0)] == {(1, -90.0, -60.0): 1, (1, -90.0, 60.0): 1}
        assert w1.n == 5
        assert w1.index.names == self.da1.to_series().index.names
        assert w1.index.tolist()[0] == (1, 90.0, 180.0)
        assert w1.index.tolist()[1] == (1, -30.0, -180.0)
        assert w1.index.tolist()[2] == (1, -30.0, 180.0)
        assert w1.index.tolist()[3] == (1, -90.0, -60.0)
        w2 = raster.da2W(self.da2, "rook")
        assert sorted(w2.neighbors[(1, -90.0, 180.0)]) == [
            (1, -90.0, 60.0),
            (1, -30.0, 180.0),
        ]
        assert sorted(w2.neighbors[(1, -90.0, 60.0)]) == [
            (1, -90.0, -60.0),
            (1, -90.0, 180.0),
            (1, -30.0, 60.0),
        ]
        assert w2.n == 16
        assert w2.index.names == self.da2.to_series().index.names
        assert w2.index.tolist() == self.da2.to_series().index.tolist()
        coords_labels = {
            "z_label": "layer",
            "y_label": "latitude",
            "x_label": "longitude",
        }
        w3 = raster.da2W(self.da3, z_value=1, coords_labels=coords_labels)
        assert sorted(w3.neighbors[(1, -90.0, 180.0)]) == [
            (1, -90.0, 60.0),
            (1, -30.0, 60.0),
            (1, -30.0, 180.0),
        ]
        assert w3.n == 16
        assert w3.index.names == self.da3.to_series().index.names
        assert w3.index.tolist() == self.da3.to_series().index.tolist()

    def test_da2_wsp(self):
        w1 = raster.da2WSP(self.da1, "rook", n_jobs=-1)
        rows, cols = w1.sparse.shape
        n = rows * cols
        pct_nonzero = w1.sparse.nnz / float(n)
        assert pct_nonzero == 0.08
        data = w1.sparse.todense().tolist()
        assert data[3] == [0, 0, 0, 0, 1]
        assert data[4] == [0, 0, 0, 1, 0]
        assert w1.index.names == self.da1.to_series().index.names
        assert w1.index.tolist()[0] == (1, 90.0, 180.0)
        assert w1.index.tolist()[1] == (1, -30.0, -180.0)
        assert w1.index.tolist()[2] == (1, -30.0, 180.0)
        assert w1.index.tolist()[3] == (1, -90.0, -60.0)
        w2 = raster.da2WSP(self.da2, "queen", k=2, include_nodata=True)
        w3 = raster.da2WSP(self.da2, "queen", k=2, n_jobs=-1)
        assert w2.sparse.nnz == w3.sparse.nnz
        assert w2.sparse.todense().tolist() == w3.sparse.todense().tolist()
        assert w2.n == 16
        assert w2.index.names == self.da2.to_series().index.names
        assert w2.index.tolist() == self.da2.to_series().index.tolist()

    def test_w2da(self):
        xarray = pytest.importorskip("xarray")
        w2 = raster.da2W(self.da2, "rook", n_jobs=-1)
        da2 = raster.w2da(self.da2.data.flatten(), w2, self.da2.attrs, self.da2.coords)
        da_compare = xarray.DataArray.equals(da2, self.da2)
        assert da_compare is True

    def test_wsp2da(self):
        wsp1 = raster.da2WSP(self.da1, "queen")
        da1 = raster.wsp2da(self.data1, wsp1)
        assert da1["y"].values.tolist() == self.da1["y"].values.tolist()
        assert da1["x"].values.tolist() == self.da1["x"].values.tolist()
        assert da1.shape == (1, 4, 4)

    def test_da_checker(self):
        pytest.raises(ValueError, raster.da2W, self.da4)

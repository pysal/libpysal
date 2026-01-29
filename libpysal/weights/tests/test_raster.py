"""Unit test for raster.py"""

import numpy as np
import pandas as pd
import pytest

from libpysal.weights import Queen

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

    @pytest.mark.network
    def test_dataarray(self):
        rioxarray = pytest.importorskip("rioxarray")

        da = rioxarray.open_rasterio(
            "https://geographicdata.science/book/_downloads/5263090bd0bdbd7d1635505ff7d36d04/ghsl_sao_paulo.tif"
        )
        w = Queen.from_xarray(da)
        assert w.n == 97232


class TestNodata:
    def setup_method(self):
        self.xr = pytest.importorskip("xarray")

    def test_nan_nodata_float_raster_not_masked(self):
        """
        Float rasters with NaN nodata should exclude NaN cells from contiguity,
        but current logic does not mask them since `ser != np.nan` is always True.
        """
        data = np.array([[1.0, 1.0, 1.0], [1.0, np.nan, 1.0], [1.0, 1.0, 1.0]])
        da = self.xr.DataArray(data, dims=("y", "x"))

        # Expected: center NaN pixel excluded -> 8 valid cells
        # Actual (current bug): NaN is not masked ->
        wsp = raster.da2WSP(da)
        assert wsp.n == 8

    def test_rio_nodata(self):
        """
        Raster with da.rio.nodata should be correctly masked.
        This guards existing behavior for rioxarray.
        """
        pytest.importorskip("rioxarray")
        import rioxarray  # noqa: F401

        data = np.array([[1, 1, 1], [1, -9999, 1], [1, 1, 1]])
        # xarray with rioxarray needs coords to be valid usually
        # for some ops, but write_nodata might be fine
        da = self.xr.DataArray(data, dims=("y", "x"))
        da = da.rio.write_nodata(-9999)

        wsp = raster.da2WSP(da)
        assert wsp.n == 8

    def test_attrs_nodata(self):
        """
        Raster with da.attrs['nodatavals'] should be correctly masked.
        This guards backward compatibility.
        """
        data = np.array([[1, 1, 1], [1, -1, 1], [1, 1, 1]], dtype=int)
        da = self.xr.DataArray(data, dims=("y", "x"))
        da.attrs["nodatavals"] = (-1,)

        wsp = raster.da2WSP(da)
        assert wsp.n == 8

    def test_no_nodata(self):
        """
        Raster without nodata should include all pixels.
        """
        data = np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 1.0]])
        da = self.xr.DataArray(data, dims=("y", "x"))

        wsp = raster.da2WSP(da)
        assert wsp.n == 9

    def test_nan_nodata_numpy_scalar(self):
        """
        Verify that using a numpy scalar nan (e.g. np.float32(np.nan))
        also triggers the correct masking.
        """
        data = np.array([[1.0, 1.0, 1.0], [1.0, np.nan, 1.0], [1.0, 1.0, 1.0]])
        da = self.xr.DataArray(data, dims=("y", "x"))
        # Set nodata to a numpy scalar nan
        da.attrs["nodatavals"] = (np.float32(np.nan),)

        wsp = raster.da2WSP(da)
        assert wsp.n == 8

    def test_all_nan_raster(self):
        """
        Verify that an all-NaN raster results in an empty weights object
        With n=0.
        """
        data = np.full((3, 3), np.nan)
        da = self.xr.DataArray(data, dims=("y", "x"))
        da.attrs["nodatavals"] = (np.nan,)

        wsp = raster.da2WSP(da)
        assert wsp.n == 0
        assert wsp.sparse.nnz == 0

    def test_infinite_values(self):
        """
        Verify that infinite values are treated as valid data (not masked out)
        as long as nodata is NaN.
        """
        data = np.array([[1.0, np.inf, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        da = self.xr.DataArray(data, dims=("y", "x"))
        da.attrs["nodatavals"] = (np.nan,)

        wsp = raster.da2WSP(da)
        # All 9 cells are valid (infinity is valid data)
        assert wsp.n == 9

import numpy as np
import pandas as pd
import pytest
from packaging.version import Version
from scipy import __version__ as scipy_version

from libpysal import graph
from libpysal.weights.raster import testDataArray as dummy_array  # noqa: N813


class TestRaster:
    def setup_method(self):
        pytest.importorskip("xarray")
        self.da1 = dummy_array()
        self.da2 = dummy_array((1, 4, 4), missing_vals=False)
        self.da3 = self.da2.rename({"band": "layer", "x": "longitude", "y": "latitude"})
        self.data1 = pd.Series(np.ones(5))
        self.da4 = dummy_array((1, 1), missing_vals=False)
        self.da4.data = np.array([["test"]])

    @pytest.mark.skipif(
        Version(scipy_version) < Version("1.12.0"),
        reason="sparse matrix power requires scipy>=1.12.0",
    )
    def test_queen(self):
        g1 = graph.Graph.build_raster_contiguity(self.da1, rook=False, k=2, n_jobs=-1)
        assert g1[(1, -30.0, -180.0)].to_dict() == {
            (1, -90.0, 60.0): 1,
            (1, -90.0, -60.0): 1,
        }
        assert g1[(1, -30.0, 180.0)].to_dict() == {
            (1, -90.0, -60.0): 1,
            (1, -90.0, 60.0): 1,
        }
        assert g1.n == 5
        assert g1._xarray_index_names == self.da1.to_series().index.names
        assert (1, 90.0, 180.0) in g1.isolates

    def test_rook(self):
        g2 = graph.Graph.build_raster_contiguity(self.da2, rook=True)
        assert g2.neighbors[(1, -90.0, 180.0)] == (
            (1, -30.0, 180.0),
            (1, -90.0, 60.0),
        )
        assert g2.neighbors[(1, -90.0, 60.0)] == (
            (1, -30.0, 60.0),
            (1, -90.0, -60.0),
            (1, -90.0, 180.0),
        )
        assert g2.n == 16
        assert g2._xarray_index_names == self.da2.to_series().index.names

    def test_labels(self):
        coords_labels = {
            "z_label": "layer",
            "y_label": "latitude",
            "x_label": "longitude",
        }
        g3 = graph.Graph.build_raster_contiguity(
            self.da3, z_value=1, coords_labels=coords_labels
        )
        assert g3.neighbors[(1, -90.0, 180.0)] == (
            (1, -30.0, 60.0),
            (1, -30.0, 180.0),
            (1, -90.0, 60.0),
        )
        assert g3.n == 16
        assert g3._xarray_index_names == self.da3.to_series().index.names

    def test_generate_da(self):
        xarray = pytest.importorskip("xarray")
        g2 = graph.Graph.build_raster_contiguity(self.da2, rook=True, n_jobs=-1)
        da2 = g2.generate_da(self.da2.data.flatten())
        # the order may be different but shall be align-able
        a, b = xarray.align(da2, self.da2)
        xarray.testing.assert_equal(a, b)

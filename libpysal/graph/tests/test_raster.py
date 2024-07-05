import numpy as np
import pandas as pd
import pytest

from libpysal import graph
from libpysal.weights.raster import testDataArray


class Testraster:
    def setup_method(self):
        pytest.importorskip("xarray")
        self.da1 = testDataArray()
        self.da2 = testDataArray((1, 4, 4), missing_vals=False)
        self.da3 = self.da2.rename({"band": "layer", "x": "longitude", "y": "latitude"})
        self.data1 = pd.Series(np.ones(5))
        self.da4 = testDataArray((1, 1), missing_vals=False)
        self.da4.data = np.array([["test"]])

    def test_queen(self):
        g1 = graph.Graph.build_raster_contiguity(self.da1, "queen", k=2, n_jobs=-1)
        assert w1[(1, -30.0, -180.0)] == {(1, -90.0, 60.0): 1, (1, -90.0, -60.0): 1}
        assert w1[(1, -30.0, 180.0)] == {(1, -90.0, -60.0): 1, (1, -90.0, 60.0): 1}
        assert w1.n == 5
        assert w1.index.names == self.da1.to_series().index.names
        assert w1.index.tolist()[0] == (1, 90.0, 180.0)
        assert w1.index.tolist()[1] == (1, -30.0, -180.0)
        assert w1.index.tolist()[2] == (1, -30.0, 180.0)
        assert w1.index.tolist()[3] == (1, -90.0, -60.0)
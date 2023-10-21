import numpy as np

from libpysal import graph
from libpysal.weights import lat2W
from libpysal.graph._spatial_lag import _lag_spatial


class TestLag:
    def setup_method(self):
        self.neighbors = {
            "a": ["b"],
            "b": ["c", "a"],
            "c": ["b"],
            "d": [],
        }
        self.weights = {"a": [1.0], "b": [1.0, 1.0], "c": [1.0], "d": []}
        self.g = graph.Graph.from_dicts(self.neighbors, self.weights)
        self.y = np.array([0, 1, 2, 3])

    def test_lag_spatial(self):
        yl = _lag_spatial(self.g, self.y)
        np.testing.assert_array_almost_equal(yl, [1.0, 2.0, 1.0, 0])
        g = graph.Graph.from_W(lat2W(3, 3))
        y = np.arange(9)
        yl = _lag_spatial(g, y)
        ylc = np.array([4.0, 6.0, 6.0, 10.0, 16.0, 14.0, 10.0, 18.0, 12.0])
        np.testing.assert_array_almost_equal(yl, ylc)
        g_row = g.transform("r")
        yl = _lag_spatial(g_row, y)
        ylc = np.array([2.0, 2.0, 3.0, 3.33333333, 4.0, 4.66666667, 5.0, 6.0, 6.0])
        np.testing.assert_array_almost_equal(yl, ylc)

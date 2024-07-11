import numpy as np
import pytest

from libpysal import graph
from libpysal.graph._spatial_lag import _lag_spatial
from libpysal.weights import lat2W


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
        self.yc = np.array([*"ababcbcbc"])
        w = lat2W(3, 3)
        w.transform = "r"
        self.gc = graph.Graph.from_W(w)

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

    def test_lag_spatial_categorical(self):
        yl = _lag_spatial(self.gc, self.yc)
        ylc = np.array(["b", "a", "b", "c", "b", "c", "b", "c", "b"], dtype=object)
        np.testing.assert_array_equal(yl, ylc)
        self.yc[3] = "a"  # create ties
        np.random.seed(12345)
        yl = _lag_spatial(self.gc, self.yc, categorical=True, ties="random")
        ylc = np.array(["a", "a", "b", "c", "b", "c", "b", "c", "b"], dtype=object)
        yl1 = _lag_spatial(self.gc, self.yc, categorical=True, ties="random")
        yls = _lag_spatial(self.gc, self.yc, categorical=True, ties="tryself")
        np.testing.assert_array_equal(yl, ylc)
        yl1c = np.array(["b", "a", "b", "c", "b", "c", "b", "c", "b"], dtype=object)
        np.testing.assert_array_equal(yl1, yl1c)
        ylsc = np.array(["a", "a", "b", "c", "b", "c", "a", "c", "b"], dtype=object)
        np.testing.assert_array_equal(yls, ylsc)
        # self-weight
        neighbors = self.gc.neighbors
        neighbors[0] = (0, 3, 1)  # add self neighbor for observation 0
        gc = graph.Graph.from_dicts(neighbors)
        self.yc[3] = "b"
        yls = _lag_spatial(gc, self.yc, categorical=True, ties="tryself")
        assert yls[0] in ["b", "a"]
        self.yc[3] = "a"
        yls = _lag_spatial(gc, self.yc, categorical=True, ties="tryself")
        assert yls[0] == "a"

    def test_ties_raise(self):
        with pytest.raises(ValueError, match="There are 2 ties that must be broken"):
            self.yc[3] = "a"  # create ties
            _lag_spatial(self.gc, self.yc, categorical=True)

    def test_categorical_custom_index(self):
        expected = np.array(["bar", "foo", "bar", "foo"])
        np.testing.assert_array_equal(
            expected, self.g.lag(["foo", "bar", "foo", "foo"])
        )

import os
import unittest
from ..weights import W
from ..util import lat2W
from ..spatial_lag import lag_spatial, lag_categorical
import numpy as np


class Test_spatial_lag(unittest.TestCase):
    def setUp(self):
        self.neighbors = {"c": ["b"], "b": ["c", "a"], "a": ["b"]}
        self.weights = {"c": [1.0], "b": [1.0, 1.0], "a": [1.0]}
        self.id_order = ["a", "b", "c"]
        self.weights = {"c": [1.0], "b": [1.0, 1.0], "a": [1.0]}
        self.w = W(self.neighbors, self.weights, self.id_order)
        self.y = np.array([0, 1, 2])
        self.wlat = lat2W(3, 3)
        self.ycat = ["a", "b", "a", "b", "c", "b", "c", "b", "c"]
        self.ycat2 = ["a", "c", "c", "d", "b", "a", "d", "d", "c"]
        self.ym = np.vstack((self.ycat, self.ycat2)).T
        self.random_seed = 503

    def test_lag_spatial(self):
        yl = lag_spatial(self.w, self.y)
        np.testing.assert_array_almost_equal(yl, [1.0, 2.0, 1.0])
        self.w.id_order = ["b", "c", "a"]
        y = np.array([1, 2, 0])
        yl = lag_spatial(self.w, y)
        np.testing.assert_array_almost_equal(yl, [2.0, 1.0, 1.0])
        w = lat2W(3, 3)
        y = np.arange(9)
        yl = lag_spatial(w, y)
        ylc = np.array([4.0, 6.0, 6.0, 10.0, 16.0, 14.0, 10.0, 18.0, 12.0])
        np.testing.assert_array_almost_equal(yl, ylc)
        w.transform = "r"
        yl = lag_spatial(w, y)
        ylc = np.array([2.0, 2.0, 3.0, 3.33333333, 4.0, 4.66666667, 5.0, 6.0, 6.0])
        np.testing.assert_array_almost_equal(yl, ylc)

    def test_lag_categorical(self):
        yl = lag_categorical(self.wlat, self.ycat)
        np.random.seed(self.random_seed)
        known = np.array(["b", "a", "b", "c", "b", "c", "b", "c", "b"])
        np.testing.assert_array_equal(yl, known)
        ym_lag = lag_categorical(self.wlat, self.ym)
        known = np.array(
            [
                ["b", "c"],
                ["a", "c"],
                ["b", "c"],
                ["c", "d"],
                ["b", "d"],
                ["c", "c"],
                ["b", "d"],
                ["c", "d"],
                ["b", "d"],
            ]
        )
        np.testing.assert_array_equal(ym_lag, np.asarray(known))


suite = unittest.TestLoader().loadTestsFromTestCase(Test_spatial_lag)

if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite)

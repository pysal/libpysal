from collections import Counter

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from libpysal import examples, graph
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

    def test_2d_array(self):
        ys = np.arange(27).reshape(9, 3)
        lag = self.gc.lag(ys)

        expected = np.array(
            [
                [6.0, 7.0, 8.0],
                [6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0],
                [10.0, 11.0, 12.0],
                [12.0, 13.0, 14.0],
                [14.0, 15.0, 16.0],
                [15.0, 16.0, 17.0],
                [18.0, 19.0, 20.0],
                [18.0, 19.0, 20.0],
            ]
        )

        np.testing.assert_array_almost_equal(lag, expected)

        # test equality to 1d
        for i in range(2):
            np.testing.assert_array_equal(self.gc.lag(ys[:, i]), lag[:, i])


@pytest.mark.parametrize("ids", [["z", "a", "m"], [20, 3, 10]])
@pytest.mark.parametrize("as_series", [False, True])
def test_categorical_lag_node_order(ids, as_series):
    g = graph.Graph.from_dicts({ids[0]: [ids[1]], ids[1]: [ids[2]], ids[2]: [ids[0]]})
    values = np.array(["red", "blue", "green"])
    if as_series:
        values = pd.Series(values, index=ids)
    np.testing.assert_array_equal(g.lag(values), ["blue", "green", "red"])


def test_categorical_lag_tryself_node_order():
    g = graph.Graph.from_dicts(
        {"z": ["a", "m"], "a": ["z", "b"], "m": ["b", "z"], "b": ["m", "a"]}
    )
    values = pd.Series(["red", "red", "blue", "blue"], index=g.unique_ids)
    # Every node has a tie, and its own label always resolves it.
    np.testing.assert_array_equal(g.lag(values, ties="tryself"), values)


@pytest.fixture(
    scope="module",
    params=[
        ("us48.shp", "STATE_NAME", "SUB_REGION"),
        ("columbus.shp", "NEIGNO", "NSA"),
    ],
    ids=["us48", "columbus"],
)
def geographic_categories(request):
    filename, id_column, category = request.param
    frame = gpd.read_file(examples.get_path(filename)).set_index(id_column)
    return frame, category


@pytest.mark.parametrize("order", ["original", "sorted", "reversed"])
@pytest.mark.parametrize("kind", ["queen", "knn1", "knn5"])
@pytest.mark.parametrize("ties", ["raise", "random", "tryself"])
def test_categorical_lag_geographic(geographic_categories, order, kind, ties):
    frame, category = geographic_categories
    if order == "sorted":
        frame = frame.sort_index()
    elif order == "reversed":
        frame = frame.iloc[::-1]
    if kind == "queen":
        g = graph.Graph.build_contiguity(frame, rook=False)
    else:
        g = graph.Graph.build_knn(frame.geometry.centroid, k=int(kind[-1]))
    np.testing.assert_array_equal(g.unique_ids, frame.index)
    values = frame[category].astype(str)

    # Count labels per node independently of the grouped categorical reduction.
    modes = {}
    for node in g.unique_ids:
        neighbors = g.neighbors[node] or (node,)
        counts = Counter(values.loc[list(neighbors)])
        modes[node] = {
            label for label, count in counts.items() if count == max(counts.values())
        }
    n_ties = sum(len(labels) > 1 for labels in modes.values())
    if ties == "raise" and n_ties:
        with pytest.raises(ValueError, match=f"There are {n_ties} ties"):
            g.lag(values, ties=ties)
        return

    result = g.lag(values, ties=ties)
    assert result.shape == (len(values),)
    for node, label in zip(g.unique_ids, result, strict=True):
        assert label in modes[node]
        if (
            ties == "tryself"
            and node not in g.neighbors[node]
            and values.loc[node] in modes[node]
        ):
            assert label == values.loc[node]

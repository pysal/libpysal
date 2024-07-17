import os
import string
import tempfile

import geodatasets
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from packaging.version import Version
from scipy import __version__ as scipy_version
from scipy import sparse

from libpysal import graph, weights


@pytest.mark.network
class TestBase:
    def setup_method(self):
        self.neighbor_dict_int = {0: 1, 1: 2, 2: 5, 3: 4, 4: 5, 5: 8, 6: 7, 7: 8, 8: 7}
        self.weight_dict_int_binary = {
            0: 1,
            1: 1,
            2: 1,
            3: 1,
            4: 1,
            5: 1,
            6: 1,
            7: 1,
            8: 1,
        }
        self.index_int = pd.Index(
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
            dtype="int64",
            name="focal",
        )
        self.neighbor_dict_str = {
            string.ascii_letters[k]: string.ascii_letters[v]
            for k, v in self.neighbor_dict_int.items()
        }
        self.weight_dict_str_binary = {
            string.ascii_letters[k]: v for k, v in self.weight_dict_int_binary.items()
        }
        self.index_str = pd.Index(
            [string.ascii_letters[k] for k in self.index_int],
            dtype="object",
            name="focal",
        )
        self.adjacency_int_binary = pd.Series(
            self.weight_dict_int_binary.values(),
            name="weight",
            index=pd.MultiIndex.from_arrays(
                [self.index_int, self.neighbor_dict_int.values()],
                names=["focal", "neighbor"],
            ),
        )
        self.adjacency_str_binary = pd.Series(
            self.weight_dict_str_binary.values(),
            name="weight",
            index=pd.MultiIndex.from_arrays(
                [self.index_str, self.neighbor_dict_str.values()],
                names=["focal", "neighbor"],
            ),
        )

        # one isolate, one self-link
        self.W_dict_int = {
            0: {0: 1, 3: 0.5, 1: 0.5},
            1: {0: 0.3, 4: 0.3, 2: 0.3},
            2: {1: 0.5, 5: 0.5},
            3: {0: 0.3, 6: 0.3, 4: 0.3},
            4: {1: 0.25, 3: 0.25, 7: 0.25, 5: 0.25},
            5: {2: 0.3, 4: 0.3, 8: 0.3},
            6: {3: 0.5, 7: 0.5},
            7: {4: 0.3, 6: 0.3, 8: 0.3},
            8: {5: 0.5, 7: 0.5},
            9: {},
        }
        self.W_dict_str = {
            string.ascii_letters[k]: {
                string.ascii_letters[k_]: v_ for k_, v_ in v.items()
            }
            for k, v in self.W_dict_int.items()
        }
        self.g_int = graph.Graph.from_weights_dict(self.W_dict_int)
        self.g_str = graph.Graph.from_weights_dict(self.W_dict_str)
        rng = np.random.default_rng(seed=0)
        self.letters = np.asarray(list(string.ascii_letters[:26]))
        rng.shuffle(self.letters)
        self.W_dict_str_unordered = {
            self.letters[k]: {self.letters[k_]: v_ for k_, v_ in v.items()}
            for k, v in self.W_dict_int.items()
        }
        self.g_str_unodered = graph.Graph.from_weights_dict(self.W_dict_str_unordered)

        self.nybb = gpd.read_file(geodatasets.get_path("nybb")).set_index("BoroName")
        self.guerry = gpd.read_file(geodatasets.get_path("geoda guerry"))

    def test_init(self):
        g = graph.Graph(self.adjacency_int_binary)
        assert isinstance(g, graph.Graph)
        assert hasattr(g, "_adjacency")
        assert g._adjacency.shape == (9,)
        pd.testing.assert_series_equal(g._adjacency, self.adjacency_int_binary)
        assert hasattr(g, "transformation")
        assert g.transformation == "O"

        g = graph.Graph(self.adjacency_str_binary)
        assert isinstance(g, graph.Graph)
        assert hasattr(g, "_adjacency")
        assert g._adjacency.shape == (9,)
        pd.testing.assert_series_equal(g._adjacency, self.adjacency_str_binary)
        assert hasattr(g, "transformation")
        assert g.transformation == "O"

        with pytest.raises(TypeError, match="The adjacency table needs to be"):
            graph.Graph(self.adjacency_int_binary.values)

        with pytest.raises(ValueError, match="The index of the adjacency table"):
            adj = self.adjacency_int_binary.copy()
            adj.index.names = ["foo", "bar"]
            graph.Graph(adj)

        with pytest.raises(
            ValueError, match="The adjacency needs to be named 'weight'"
        ):
            graph.Graph(self.adjacency_int_binary.rename("foo"))

        with pytest.raises(ValueError, match="The 'weight' needs"):
            graph.Graph(self.adjacency_int_binary.astype(str))

        with pytest.raises(
            ValueError, match="The adjacency table cannot contain missing"
        ):
            adj = self.adjacency_int_binary.copy()
            adj.iloc[0] = np.nan
            graph.Graph(adj)

        with pytest.raises(ValueError, match="'transformation' needs to be"):
            graph.Graph(self.adjacency_int_binary, transformation="foo")

    def test___repr__(self):
        expected = (
            "<Graph of 10 nodes and 25 nonzero edges indexed by\n"
            " [0, 1, 2, 3, 4, ...]>"
        )
        assert repr(self.g_int) == expected

        expected = (
            "<Graph of 10 nodes and 25 nonzero edges indexed by\n"
            " ['a', 'b', 'c', 'd', 'e', ...]>"
        )
        assert repr(self.g_str) == expected

        nybb = graph.Graph.build_contiguity(self.nybb)
        expected = (
            "<Graph of 5 nodes and 10 nonzero edges indexed by\n"
            " ['Staten Island', 'Queens', 'Brooklyn', 'Manhattan', 'Bronx']>"
        )
        assert repr(nybb) == expected

        h3 = {
            "821f87fffffffff": ("821fb7fffffffff", "821f97fffffffff"),
            "821fb7fffffffff": (
                "821f87fffffffff",
                "821f97fffffffff",
                "82186ffffffffff",
                "821867fffffffff",
            ),
            "821f97fffffffff": (
                "821f87fffffffff",
                "821fb7fffffffff",
                "823967fffffffff",
                "82396ffffffffff",
                "82186ffffffffff",
            ),
            "823967fffffffff": (
                "821f97fffffffff",
                "82396ffffffffff",
                "82186ffffffffff",
            ),
            "82396ffffffffff": ("821f97fffffffff", "823967fffffffff"),
            "82186ffffffffff": (
                "821fb7fffffffff",
                "821f97fffffffff",
                "823967fffffffff",
                "821867fffffffff",
            ),
            "821867fffffffff": ("821fb7fffffffff", "82186ffffffffff"),
        }
        h3_g = graph.Graph.from_dicts(h3)
        expected = (
            "<Graph of 7 nodes and 22 nonzero edges indexed by\n"
            " ['821f87fffffffff', '821fb7fffffffff', '821f97fffffffff',"
            " '823967fffffff...]>"
        )
        assert repr(h3_g) == expected

    def test_copy(self):
        g_copy = self.g_str.copy()
        assert g_copy == self.g_str
        g_copy._adjacency.iloc[0] = 2
        assert g_copy != self.g_str

    def test_adjacency(self):
        g = graph.Graph(self.adjacency_int_binary)
        adjacency = g.adjacency
        pd.testing.assert_series_equal(adjacency, self.adjacency_int_binary)

        # ensure copy
        adjacency.iloc[0] = 100
        pd.testing.assert_series_equal(g._adjacency, self.adjacency_int_binary)

    def test_w_roundtrip(self):
        w = self.g_int.to_W()
        pd.testing.assert_series_equal(
            self.g_int._adjacency.sort_index(),
            w.to_adjlist(drop_islands=False)
            .set_index(["focal", "neighbor"])["weight"]
            .sort_index(),
            check_index_type=False,
            check_dtype=False,
        )
        g_roundtripped = graph.Graph.from_W(w)
        assert self.g_int == g_roundtripped
        assert isinstance(w.id_order, list)

        w = self.g_str.to_W()
        pd.testing.assert_series_equal(
            self.g_str._adjacency.sort_index(),
            w.to_adjlist(drop_islands=False)
            .set_index(["focal", "neighbor"])["weight"]
            .sort_index(),
            check_index_type=False,
            check_dtype=False,
        )
        g_roundtripped = graph.Graph.from_W(w)
        assert self.g_str == g_roundtripped

        w = weights.lat2W(3, 3)
        g = graph.Graph.from_W(w)
        pd.testing.assert_series_equal(
            g._adjacency.sort_index(),
            w.to_adjlist().set_index(["focal", "neighbor"])["weight"].sort_index(),
            check_index_type=False,
            check_dtype=False,
        )
        w_exp = g.to_W()
        # assert w.neighbors == w_exp.neighbors
        assert w.weights == w_exp.weights

        w.transform = "r"
        g_rowwise = graph.Graph.from_W(w)
        pd.testing.assert_series_equal(
            g_rowwise._adjacency.sort_index(),
            w.to_adjlist().set_index(["focal", "neighbor"])["weight"].sort_index(),
            check_index_type=False,
            check_dtype=False,
        )
        w_trans = g_rowwise.to_W()
        # assert w.neighbors == w_trans.neighbors
        assert w.weights == w_trans.weights

        diag = weights.fill_diagonal(w)
        g_diag = graph.Graph.from_W(diag)
        pd.testing.assert_series_equal(
            g_diag._adjacency.sort_index(),
            diag.to_adjlist().set_index(["focal", "neighbor"])["weight"].sort_index(),
            check_index_type=False,
            check_dtype=False,
        )
        w_diag = g_diag.to_W()
        # assert diag.neighbors == W_diag.neighbors
        assert diag.weights == w_diag.weights

        w = weights.W(
            neighbors={"a": ["b"], "b": ["a", "c"], "c": ["b"], "d": []},
            silence_warnings=True,
        )
        g_isolate = graph.Graph.from_W(w)
        pd.testing.assert_series_equal(
            g_isolate._adjacency.sort_index(),
            w.to_adjlist().set_index(["focal", "neighbor"])["weight"].sort_index(),
            check_index_type=False,
            check_dtype=False,
        )
        w_isolate = g_isolate.to_W()
        # assert w.neighbors == w_isolate.neighbors
        assert w.weights == w_isolate.weights

        w = self.g_str_unodered.to_W()
        assert w.id_order_set
        np.testing.assert_array_equal(w.id_order, self.letters[:10])

    def test_from_sparse(self):
        row = np.array([0, 0, 1, 2, 3, 3])
        col = np.array([1, 3, 3, 2, 1, 3])
        data = np.array([0.1, 0.5, 0.9, 0, 0.3, 0.1])
        sp = sparse.coo_array((data, (row, col)), shape=(4, 4))
        g = graph.Graph.from_sparse(sp)
        expected = graph.Graph.from_arrays(row, col, data)
        assert g == expected, "sparse constructor does not match arrays constructor"

        g = graph.Graph.from_sparse(sp.tocsr())
        assert g == expected, "csc input does not match coo input"

        g = graph.Graph.from_sparse(sp.tocsc())
        assert g == expected, "csr input does not match coo input"

        ids = ["zero", "one", "two", "three"]

        g_named = graph.Graph.from_sparse(
            sp,
            ids=ids,
        )

        expected = graph.Graph.from_arrays(
            ["zero", "zero", "one", "two", "three", "three"],
            ["one", "three", "three", "two", "one", "three"],
            data,
        )
        assert g_named == expected

        g = graph.Graph.from_sparse(
            sp.tocsr(),
            ids=ids,
        )
        assert g == expected, "sparse csr with ids does not match arrays constructor"

        g = graph.Graph.from_sparse(
            sp.tocsc(),
            ids=ids,
        )
        assert g == (expected), "sparse csr with ids does not match arrays constructor"

        dense = np.array(
            [
                [0, 0, 1, 1, 0],
                [0, 0, 1, 1, 0],
                [0, 1, 0, 1, 0],
                [0, 1, 0, 0, 1],
                [0, 1, 0, 1, 0],
            ]
        )
        sp = sparse.csr_array(dense)
        g = graph.Graph.from_sparse(
            sp, ids=["staten_island", "queens", "brooklyn", "manhattan", "bronx"]
        )
        expected = graph.Graph.from_arrays(
            [
                "staten_island",
                "staten_island",
                "queens",
                "queens",
                "brooklyn",
                "brooklyn",
                "manhattan",
                "manhattan",
                "bronx",
                "bronx",
            ],
            [
                "brooklyn",
                "manhattan",
                "brooklyn",
                "manhattan",
                "queens",
                "manhattan",
                "queens",
                "bronx",
                "queens",
                "manhattan",
            ],
            np.ones(10),
        )
        assert (
            g == expected
        ), "sparse csr nybb with ids does not match arrays constructor"
        np.testing.assert_array_equal(g.sparse.todense(), sp.todense())

        with pytest.raises(ValueError, match="The length of ids "):
            graph.Graph.from_sparse(sp, ids=["staten_island", "queens"])

    def test_from_arrays(self):
        focal_ids = np.arange(9)
        neighbor_ids = np.array([1, 2, 5, 4, 5, 8, 7, 8, 7])
        weight = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])

        g = graph.Graph.from_arrays(focal_ids, neighbor_ids, weight)
        pd.testing.assert_series_equal(
            g._adjacency,
            self.adjacency_int_binary,
            check_index_type=False,
            check_dtype=False,
        )

        focal_ids = np.asarray(list(self.neighbor_dict_str.keys()))
        neighbor_ids = np.asarray(list(self.neighbor_dict_str.values()))

        g = graph.Graph.from_arrays(focal_ids, neighbor_ids, weight)
        pd.testing.assert_series_equal(
            g._adjacency,
            self.adjacency_str_binary,
            check_index_type=False,
            check_dtype=False,
        )

    def test_from_weights_dict(self):
        weights_dict = {
            0: {2: 0.5, 1: 0.5},
            1: {0: 0.5, 3: 0.5},
            2: {
                0: 0.3,
                4: 0.3,
                3: 0.3,
            },
            3: {
                1: 0.3,
                2: 0.3,
                5: 0.3,
            },
            4: {2: 0.5, 5: 0.5},
            5: {3: 0.5, 4: 0.5},
        }
        exp_focal = pd.Index(
            [0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5], dtype="int64", name="focal"
        )
        exp_neighbor = [2, 1, 0, 3, 0, 4, 3, 1, 2, 5, 2, 5, 3, 4]
        exp_weight = [
            0.5,
            0.5,
            0.5,
            0.5,
            0.3,
            0.3,
            0.3,
            0.3,
            0.3,
            0.3,
            0.5,
            0.5,
            0.5,
            0.5,
        ]
        expected = graph.Graph.from_arrays(exp_focal, exp_neighbor, exp_weight)

        g = graph.Graph.from_weights_dict(weights_dict)
        assert g == expected

    def test_from_dicts(self):
        g = graph.Graph.from_dicts(self.neighbor_dict_int)
        pd.testing.assert_series_equal(
            g._adjacency,
            self.adjacency_int_binary,
            check_dtype=False,
            check_index_type=False,
        )

        g = graph.Graph.from_dicts(self.neighbor_dict_str)
        pd.testing.assert_series_equal(
            g._adjacency,
            self.adjacency_str_binary,
            check_dtype=False,
        )

    @pytest.mark.parametrize("y", [3, 5])
    @pytest.mark.parametrize("id_type", ["int", "str"])
    @pytest.mark.parametrize("rook", [True, False])
    def test_from_dicts_via_w(self, y, id_type, rook):
        w = weights.lat2W(3, y, id_type=id_type, rook=rook)
        g = graph.Graph.from_dicts(w.neighbors, w.weights)
        pd.testing.assert_series_equal(
            g._adjacency.sort_index(),
            w.to_adjlist().set_index(["focal", "neighbor"])["weight"].sort_index(),
            check_index_type=False,
            check_dtype=False,
        )

        w.transform = "r"
        g_rowwise = graph.Graph.from_dicts(w.neighbors, w.weights)
        pd.testing.assert_series_equal(
            g_rowwise._adjacency.sort_index(),
            w.to_adjlist().set_index(["focal", "neighbor"])["weight"].sort_index(),
            check_index_type=False,
            check_dtype=False,
        )

        diag = weights.fill_diagonal(w)
        g_diag = graph.Graph.from_dicts(diag.neighbors, diag.weights)
        pd.testing.assert_series_equal(
            g_diag._adjacency.sort_index(),
            diag.to_adjlist().set_index(["focal", "neighbor"])["weight"].sort_index(),
            check_index_type=False,
            check_dtype=False,
        )

        w = weights.W(
            neighbors={"a": ["b"], "b": ["a", "c"], "c": ["b"], "d": []},
            silence_warnings=True,
        )
        g_isolate = graph.Graph.from_dicts(w.neighbors, w.weights)
        pd.testing.assert_series_equal(
            g_isolate._adjacency.sort_index(),
            w.to_adjlist().set_index(["focal", "neighbor"])["weight"].sort_index(),
            check_index_type=False,
            check_dtype=False,
        )

    def test_neighbors(self):
        expected = {
            0: (0, 1, 3),
            1: (0, 2, 4),
            2: (1, 5),
            3: (0, 4, 6),
            4: (1, 3, 5, 7),
            5: (2, 4, 8),
            6: (3, 7),
            7: (4, 6, 8),
            8: (5, 7),
            9: (),
        }
        assert self.g_int.neighbors == expected

        expected = {
            "a": ("a", "b", "d"),
            "b": ("a", "c", "e"),
            "c": ("b", "f"),
            "d": ("a", "e", "g"),
            "e": ("b", "d", "f", "h"),
            "f": ("c", "e", "i"),
            "g": ("d", "h"),
            "h": ("e", "g", "i"),
            "i": ("f", "h"),
            "j": (),
        }
        assert self.g_str.neighbors == expected

    def test_weights(self):
        expected = {
            0: (1.0, 0.5, 0.5),
            1: (0.3, 0.3, 0.3),
            2: (0.5, 0.5),
            3: (0.3, 0.3, 0.3),
            4: (0.25, 0.25, 0.25, 0.25),
            5: (0.3, 0.3, 0.3),
            6: (0.5, 0.5),
            7: (0.3, 0.3, 0.3),
            8: (0.5, 0.5),
            9: (),
        }
        assert self.g_int.weights == expected

        expected = {
            "a": (1.0, 0.5, 0.5),
            "b": (0.3, 0.3, 0.3),
            "c": (0.5, 0.5),
            "d": (0.3, 0.3, 0.3),
            "e": (0.25, 0.25, 0.25, 0.25),
            "f": (0.3, 0.3, 0.3),
            "g": (0.5, 0.5),
            "h": (0.3, 0.3, 0.3),
            "i": (0.5, 0.5),
            "j": (),
        }
        assert self.g_str.weights == expected

    def test_sparse(self):
        sp = self.g_int.sparse
        expected = np.array(
            [
                [1.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.3, 0.0, 0.3, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
                [0.3, 0.0, 0.0, 0.0, 0.3, 0.0, 0.3, 0.0, 0.0, 0.0],
                [0.0, 0.25, 0.0, 0.25, 0.0, 0.25, 0.0, 0.25, 0.0, 0.0],
                [0.0, 0.0, 0.3, 0.0, 0.3, 0.0, 0.0, 0.0, 0.3, 0.0],
                [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.3, 0.0, 0.3, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        np.testing.assert_array_equal(sp.todense(), expected)

        sp = self.g_str.sparse
        np.testing.assert_array_equal(sp.todense(), expected)

        sp_old = self.g_int.to_W().sparse.todense()
        np.testing.assert_array_equal(sp.todense(), sp_old)

        sp_old = self.g_str.to_W().sparse.todense()
        np.testing.assert_array_equal(sp.todense(), sp_old)

        # check proper sorting
        nybb = graph.Graph.build_contiguity(self.nybb)
        nybb_expected = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1],
                [0, 1, 0, 1, 0],
                [0, 1, 1, 0, 1],
                [0, 1, 0, 1, 0],
            ]
        )
        np.testing.assert_array_equal(nybb.sparse.todense(), nybb_expected)

    def test_sparse_roundtrip(self):
        g = graph.Graph(self.adjacency_int_binary)
        sp = g.sparse
        g_sp = graph.Graph.from_sparse(sp, self.index_int)
        assert g == g_sp

        g = graph.Graph(self.adjacency_str_binary)
        sp = g.sparse
        g_sp = graph.Graph.from_sparse(sp, self.index_str)
        assert g == g_sp

    def test_cardinalities(self):
        expected = pd.Series(
            [3, 3, 2, 3, 4, 3, 2, 3, 2, 0],
            index=pd.Index(
                ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
                dtype="object",
                name="focal",
            ),
            name="cardinalities",
        )
        pd.testing.assert_series_equal(self.g_str.cardinalities, expected)

    def test_isolates(self):
        expected = pd.Index(["j"], name="focal")
        pd.testing.assert_index_equal(self.g_str.isolates, expected)

        self.g_str._adjacency.iloc[1] = 0  # zero weight, no isolate
        pd.testing.assert_index_equal(self.g_str.isolates, expected)

        with_additional_zeros = self.g_str.assign_self_weight(0)
        pd.testing.assert_index_equal(with_additional_zeros.isolates, expected)

    def test_n(self):
        assert self.g_int.n == 10
        assert self.g_str.n == 10
        assert graph.Graph(self.adjacency_int_binary).n == 9

    def test_pct_nonzero(self):
        assert self.g_int.pct_nonzero == 26.0
        assert graph.Graph(self.adjacency_int_binary).pct_nonzero == pytest.approx(
            11.1111111111
        )

    def test_nonzero(self):
        assert self.g_int.nonzero == 25
        assert graph.Graph(self.adjacency_int_binary).nonzero == 9

    def test_index_pairs(self):
        focal, neighbor = self.g_str.index_pairs
        exp_focal = pd.Index(
            [
                "a",
                "a",
                "a",
                "b",
                "b",
                "b",
                "c",
                "c",
                "d",
                "d",
                "d",
                "e",
                "e",
                "e",
                "e",
                "f",
                "f",
                "f",
                "g",
                "g",
                "h",
                "h",
                "h",
                "i",
                "i",
                "j",
            ],
            name="focal",
        )
        exp_neighbor = pd.Index(
            [
                "a",
                "b",
                "d",
                "a",
                "c",
                "e",
                "b",
                "f",
                "a",
                "e",
                "g",
                "b",
                "d",
                "f",
                "h",
                "c",
                "e",
                "i",
                "d",
                "h",
                "e",
                "g",
                "i",
                "f",
                "h",
                "j",
            ],
            name="neighbor",
        )
        pd.testing.assert_index_equal(exp_focal, focal)
        pd.testing.assert_index_equal(exp_neighbor, neighbor)

    def test_transform_r(self):
        expected_w = [
            0.5,
            0.25,
            0.25,
            0.33333333,
            0.33333333,
            0.33333333,
            0.5,
            0.5,
            0.33333333,
            0.33333333,
            0.33333333,
            0.25,
            0.25,
            0.25,
            0.25,
            0.33333333,
            0.33333333,
            0.33333333,
            0.5,
            0.5,
            0.33333333,
            0.33333333,
            0.33333333,
            0.5,
            0.5,
            0.0,
        ]
        exp = self.g_int.adjacency
        exp.iloc[:] = expected_w
        expected = graph.Graph(exp)
        assert self.g_int.transform("r") == expected
        assert self.g_int.transform("r").transformation == "R"
        assert self.g_int.transform("R") == expected

        w = self.g_int.to_W()
        w.transform = "r"
        g_from_w = graph.Graph.from_W(w)
        assert g_from_w == self.g_int.transform("r")

    def test_transform_b(self):
        expected_w = [
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
        ]
        exp = self.g_int.adjacency
        exp.iloc[:] = expected_w
        expected = graph.Graph(exp)
        assert self.g_int.transform("b") == expected
        assert self.g_int.transform("b").transformation == "B"
        assert self.g_int.transform("B") == expected

        w = self.g_int.to_W()
        w.transform = "b"
        g_from_w = graph.Graph.from_W(w)
        assert g_from_w == self.g_int.transform("b")

    def test_transform_d(self):
        expected_w = [
            0.10416667,
            0.05208333,
            0.05208333,
            0.03125,
            0.03125,
            0.03125,
            0.05208333,
            0.05208333,
            0.03125,
            0.03125,
            0.03125,
            0.02604167,
            0.02604167,
            0.02604167,
            0.02604167,
            0.03125,
            0.03125,
            0.03125,
            0.05208333,
            0.05208333,
            0.03125,
            0.03125,
            0.03125,
            0.05208333,
            0.05208333,
            0.0,
        ]
        exp = self.g_int.adjacency
        exp.iloc[:] = expected_w
        expected = graph.Graph(exp)
        assert self.g_int.transform("d") == expected
        assert self.g_int.transform("d").transformation == "D"
        assert self.g_int.transform("D") == expected
        assert self.g_int.transform("D")._adjacency.sum() == pytest.approx(1)

        w = self.g_int.to_W()
        w.transform = "d"
        g_from_w = graph.Graph.from_W(w)
        assert g_from_w == self.g_int.transform("d")

    def test_transform_v(self):
        expected_w = [
            0.55154388,
            0.27577194,
            0.27577194,
            0.39000042,
            0.39000042,
            0.39000042,
            0.47765102,
            0.47765102,
            0.39000042,
            0.39000042,
            0.39000042,
            0.33775027,
            0.33775027,
            0.33775027,
            0.33775027,
            0.39000042,
            0.39000042,
            0.39000042,
            0.47765102,
            0.47765102,
            0.39000042,
            0.39000042,
            0.39000042,
            0.47765102,
            0.47765102,
            0.0,
        ]
        exp = self.g_int.adjacency
        exp.iloc[:] = expected_w
        expected = graph.Graph(exp)
        assert self.g_int.transform("v") == expected
        assert self.g_int.transform("v").transformation == "V"
        assert self.g_int.transform("V") == expected

        w = self.g_int.to_W()
        w.transform = "v"
        g_from_w = graph.Graph.from_W(w)
        assert g_from_w == self.g_int.transform("v")

    def test_transform(self):
        # do not transform if transformation == current transformation
        binary = self.g_int.transform("b")
        fast_tracked = binary.transform("b")
        assert binary == fast_tracked

        with pytest.raises(ValueError, match="Transformation 'X' is not"):
            self.g_int.transform("x")

    def test_transform_callable(self):
        contig = graph.Graph.build_contiguity(self.nybb)
        trans = contig.transform(lambda x: x * 10)
        assert trans.transformation == "C"
        assert trans.adjacency.sum() == 100

    def test_asymmetry(self):
        neighbors = {
            "a": ["b", "c", "d"],
            "b": ["b", "c", "d"],
            "c": ["a", "b"],
            "d": ["a", "b"],
        }
        weights_d = {"a": [1, 0.5, 1], "b": [1, 1, 1], "c": [1, 1], "d": [1, 1]}
        g = graph.Graph.from_dicts(neighbors, weights_d)
        intrinsic = pd.Series(
            ["b", "c", "a", "a"],
            index=pd.Index(["a", "a", "b", "c"], name="focal"),
            name="neighbor",
        )
        pd.testing.assert_series_equal(intrinsic, g.asymmetry())

        boolean = pd.Series(
            ["b", "a"],
            index=pd.Index(["a", "b"], name="focal"),
            name="neighbor",
        )
        pd.testing.assert_series_equal(boolean, g.asymmetry(intrinsic=False))

        empty = pd.Series(
            index=pd.Index([], name="focal"),
            name="neighbor",
            dtype="int64",
        )

        pd.testing.assert_series_equal(self.g_int.asymmetry(False), empty)

    def test_parquet(self):
        pytest.importorskip("pyarrow")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "g_int.parquet")
            self.g_int.to_parquet(path)
            g_int = graph.read_parquet(path)
            assert self.g_int == g_int

            path = os.path.join(tmpdir, "g_str.parquet")
            self.g_str.to_parquet(path)
            g_str = graph.read_parquet(path)
            assert self.g_str == g_str

            row_wise = self.g_str.transform("r")
            path = os.path.join(tmpdir, "row.parquet")
            row_wise.to_parquet(path)
            row_read = graph.read_parquet(path)
            assert row_wise == row_read
            assert row_read.transformation == "R"

            path = os.path.join(tmpdir, "pandas.parquet")
            self.g_str._adjacency.to_frame().to_parquet(path)
            g_pandas = graph.read_parquet(path)
            assert self.g_str == g_pandas

    def test_gal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "g_int.gal")
            g_int = self.g_int.transform("b")
            g_int.to_gal(path)
            g_int_ = graph.read_gal(path)
            assert g_int == g_int_

            path = os.path.join(tmpdir, "g_str.gal")
            g_str = self.g_str.transform("b")
            g_str.to_gal(path)
            g_str_ = graph.read_gal(path)
            assert g_str == g_str_

    def test_gwt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "g_int.gwt")
            self.g_int.to_gwt(path)
            g_int = graph.read_gwt(path)
            assert self.g_int == g_int

            path = os.path.join(tmpdir, "g_str.gwt")
            self.g_str.to_gwt(path)
            g_str = graph.read_gwt(path)
            assert self.g_str == g_str

    def test_getitem(self):
        expected = pd.Series(
            [1, 0.5, 0.5],
            index=pd.Index(["a", "b", "d"], name="neighbor"),
            name="weight",
        )
        pd.testing.assert_series_equal(expected, self.g_str["a"])

        expected = pd.Series(
            [1, 0.5, 0.5],
            index=pd.Index([0, 1, 3], name="neighbor"),
            name="weight",
        )
        pd.testing.assert_series_equal(expected, self.g_int[0])

        # isolate
        expected = pd.Series(
            [],
            index=pd.Index([], name="neighbor"),
            name="weight",
        )
        pd.testing.assert_series_equal(expected, self.g_str["j"])

    def test_lag(self):
        expected = np.array([4.0, 2.7, 4.0, 3.9, 5.0, 5.1, 6.0, 6.3, 7.0, 0.0])
        lag = self.g_str.lag(list(range(1, 11)))

        np.testing.assert_allclose(expected, lag)

        with pytest.raises(ValueError, match="The length of `y`"):
            self.g_str.lag(list(range(1, 15)))

    @pytest.mark.skipif(
        Version(scipy_version) < Version("1.12.0"),
        reason="sparse matrix power requires scipy>=1.12.0",
    )
    def test_higher_order(self):
        cont = graph.Graph.build_contiguity(self.nybb)
        k2 = cont.higher_order(2)
        expected = graph.Graph.from_arrays(
            self.nybb.index,
            ["Staten Island", "Queens", "Bronx", "Manhattan", "Brooklyn"],
            [0, 0, 1, 0, 1],
        )
        assert k2 == expected

        diagonal = cont.higher_order(2, diagonal=True)
        expected = graph.Graph.from_arrays(
            [
                "Staten Island",
                "Queens",
                "Brooklyn",
                "Brooklyn",
                "Manhattan",
                "Bronx",
                "Bronx",
            ],
            [
                "Staten Island",
                "Queens",
                "Brooklyn",
                "Bronx",
                "Manhattan",
                "Brooklyn",
                "Bronx",
            ],
            [0, 1, 1, 1, 1, 1, 1],
        )
        assert diagonal == expected

        shortest_false = cont.higher_order(2, shortest_path=False)
        expected = graph.Graph.from_arrays(
            [
                "Staten Island",
                "Queens",
                "Queens",
                "Queens",
                "Brooklyn",
                "Brooklyn",
                "Brooklyn",
                "Manhattan",
                "Manhattan",
                "Manhattan",
                "Bronx",
                "Bronx",
                "Bronx",
            ],
            [
                "Staten Island",
                "Brooklyn",
                "Manhattan",
                "Bronx",
                "Queens",
                "Manhattan",
                "Bronx",
                "Queens",
                "Brooklyn",
                "Bronx",
                "Queens",
                "Brooklyn",
                "Manhattan",
            ],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        )
        assert shortest_false == expected

        lower = cont.higher_order(2, lower_order=True)
        assert lower == expected

    @pytest.mark.skipif(
        Version(scipy_version) < Version("1.12.0"),
        reason="sparse matrix power requires scipy>=1.12.0",
    )
    def test_higher_order_inclusive(self):  # GH738
        contig = graph.Graph.from_arrays(
            [0, 1, 2, 3, 3, 4, 4], [0, 3, 4, 1, 4, 2, 3], [0, 1, 1, 1, 1, 1, 1]
        )
        assert len(contig) == 6
        higher = contig.higher_order(2, lower_order=True)
        assert len(higher) == 10
        assert contig < higher

    def test_n_components(self):
        nybb = graph.Graph.build_contiguity(self.nybb)
        assert nybb.n_components == 2

        nybb = graph.Graph.build_knn(self.nybb.set_geometry(self.nybb.centroid), k=2)
        assert nybb.n_components == 1

    def test_component_labels(self):
        nybb = graph.Graph.build_contiguity(self.nybb)
        expected = pd.Series(
            [0, 1, 1, 1, 1],
            index=pd.Index(self.nybb.index.values, name="focal"),
            dtype=int,
            name="component labels",
        )
        pd.testing.assert_series_equal(
            expected, nybb.component_labels, check_dtype=False
        )

    def test_eliminate_zeros(self):
        nybb = graph.Graph.build_contiguity(self.nybb)
        adj = nybb._adjacency.copy()
        adj["Bronx", "Queens"] = 0
        adj["Queens", "Manhattan"] = 0
        adj["Queens", "Queens"] = 0
        with_zero = graph.Graph(adj)
        expected = adj.drop(
            [("Bronx", "Queens"), ("Queens", "Manhattan"), ("Queens", "Queens")]
        )
        pd.testing.assert_series_equal(with_zero.eliminate_zeros()._adjacency, expected)

    def test_subgraph(self):
        knn = graph.Graph.build_knn(self.nybb.set_geometry(self.nybb.centroid), k=2)
        sub = knn.subgraph(["Staten Island", "Bronx", "Brooklyn"])
        assert sub < knn
        expected = pd.Series(
            [1, 0, 0],
            name="weight",
            index=pd.MultiIndex.from_arrays(
                [
                    ["Staten Island", "Bronx", "Brooklyn"],
                    ["Brooklyn", "Bronx", "Brooklyn"],
                ],
                names=["focal", "neighbor"],
            ),
        )
        pd.testing.assert_series_equal(expected, sub._adjacency, check_dtype=False)

    def test_assign_self_weight(self):
        contig = graph.Graph.build_contiguity(self.nybb)
        diag = contig.assign_self_weight()
        assert len(diag._adjacency) == 15
        assert diag._adjacency.sum() == 15

        diag_array = contig.assign_self_weight([2, 3, 4, 5, 6])
        assert len(diag_array._adjacency) == 15
        assert diag_array._adjacency.sum() == 30

        for i, val in enumerate(range(2, 7)):
            assert (
                diag_array._adjacency[(contig.unique_ids[i], contig.unique_ids[i])]
                == val
            )

    def test_apply(self):
        contig = graph.Graph.build_contiguity(self.nybb)

        # pandas built-in
        expected = pd.Series(
            [1.62382200e09, 3.76087588e09, 3.68168493e09, 6.16961834e09, 3.68168493e09],
            index=pd.Index(
                ["Staten Island", "Queens", "Brooklyn", "Manhattan", "Bronx"],
                name="focal",
            ),
        )
        pd.testing.assert_series_equal(contig.apply(self.nybb.area, "sum"), expected)

        # numpy
        expected = pd.Series(
            [1.62382200e09, 1.18692629e09, 1.84084247e09, 1.93747835e09, 1.84084247e09],
            index=pd.Index(
                ["Staten Island", "Queens", "Brooklyn", "Manhattan", "Bronx"],
                name="focal",
            ),
        )
        pd.testing.assert_series_equal(
            contig.apply(self.nybb.area, np.median), expected
        )

        # lambda over geometry
        expected = pd.Series(
            [2.06271959e09, 6.68788190e09, 7.57087991e09, 8.78957337e09, 7.57087991e09],
            index=pd.Index(
                ["Staten Island", "Queens", "Brooklyn", "Manhattan", "Bronx"],
                name="focal",
            ),
        )
        pd.testing.assert_series_equal(
            contig.apply(self.nybb.geometry, lambda x: x.unary_union.convex_hull.area),
            expected,
        )

        # reduction over a dataframe
        expected = pd.DataFrame(
            [
                [3.30470010e05, 1.62381982e09],
                [1.56477261e06, 3.76087473e09],
                [1.25564314e06, 3.68168433e09],
                [2.10181756e06, 6.16961599e09],
                [1.25564314e06, 3.68168433e09],
            ],
            columns=["Shape_Leng", "Shape_Area"],
            index=pd.Index(
                ["Staten Island", "Queens", "Brooklyn", "Manhattan", "Bronx"],
                name="focal",
            ),
        )
        pd.testing.assert_frame_equal(
            contig.apply(
                self.nybb, lambda x: x[["Shape_Leng", "Shape_Area"]].sum(axis=0)
            ),
            expected,
        )

        # 1D array input
        expected = pd.Series(
            [1.62382200e09, 3.76087588e09, 3.68168493e09, 6.16961834e09, 3.68168493e09],
            index=pd.Index(
                ["Staten Island", "Queens", "Brooklyn", "Manhattan", "Bronx"],
                name="focal",
            ),
        )
        pd.testing.assert_series_equal(
            contig.apply(self.nybb.area.values, "sum"), expected
        )

        # 2D array input
        expected = pd.DataFrame(
            [
                [3.30470010e05, 1.62381982e09],
                [1.56477261e06, 3.76087473e09],
                [1.25564314e06, 3.68168433e09],
                [2.10181756e06, 6.16961599e09],
                [1.25564314e06, 3.68168433e09],
            ],
            index=pd.Index(
                ["Staten Island", "Queens", "Brooklyn", "Manhattan", "Bronx"],
                name="focal",
            ),
        )
        pd.testing.assert_frame_equal(
            contig.apply(
                self.nybb[["Shape_Leng", "Shape_Area"]].values,
                lambda x: x.sum(axis=0),
            ),
            expected,
        )

    def test_aggregate(self):
        contig = graph.Graph.build_contiguity(self.nybb)
        expected = pd.Series(
            [1.0, 20.08553692, 7.3890561, 20.08553692, 7.3890561],
            index=pd.Index(
                ["Staten Island", "Queens", "Brooklyn", "Manhattan", "Bronx"],
                name="focal",
            ),
            name="weight",
        )
        pd.testing.assert_series_equal(
            contig.aggregate(lambda x: np.exp(np.sum(x))),
            expected,
        )

    def test_describe(self):
        contig = graph.Graph.build_knn(self.guerry.geometry.centroid, k=5)
        y = self.guerry.geometry.area
        stats = contig.describe(y)
        pd.testing.assert_series_equal(
            stats["count"],
            contig.cardinalities,
            check_names=False,
            check_dtype=False,
        )
        pd.testing.assert_series_equal(
            stats["sum"],
            pd.Series(contig.lag(y), index=contig.unique_ids),
            check_names=False,
        )
        r_contig = contig.transform("R")
        pd.testing.assert_series_equal(
            stats["mean"],
            pd.Series(r_contig.lag(y), index=contig.unique_ids),
            check_names=False,
        )
        ## compute only some statistics
        specific_stats = contig.describe(y, statistics=["count", "sum", "mean"])
        ## assert only the specified values are computed
        assert list(specific_stats.columns) == ["count", "sum", "mean"]

        pd.testing.assert_frame_equal(
            specific_stats[["count", "sum", "mean"]], stats[["count", "sum", "mean"]]
        )

        percentile_stats = contig.describe(y, q=(25, 75))

        for i in contig.unique_ids:
            neigh_vals = y[contig[i].index.values]
            low, high = neigh_vals.describe()[["25%", "75%"]]
            neigh_vals = neigh_vals[(low <= neigh_vals) & (neigh_vals <= high)]
            expected = neigh_vals.describe()[["count", "mean", "std", "min", "max"]]
            res = percentile_stats.loc[i][["count", "mean", "std", "min", "max"]]
            pd.testing.assert_series_equal(res, expected, check_names=False)

        ## test NA equivalence between filtration and pandas
        nan_areas = y.copy()
        nan_areas.iloc[range(0, len(y), 3),] = np.nan
        res1 = contig.describe(y, statistics=["count"])["count"]
        res2 = contig.describe(y, statistics=["count"], q=(0, 100))["count"]
        pd.testing.assert_series_equal(res1, res2)

        # test with isolates and string index
        nybb_contig = graph.Graph.build_contiguity(self.nybb, rook=False)
        stats = nybb_contig.describe(
            self.nybb.geometry.area, statistics=["count", "sum"]
        )
        ## all isolate values should be nan
        assert stats.loc["Staten Island"].isna().all()

        # for easier comparison and na has already been checked.
        stats = stats.fillna(0)

        pd.testing.assert_series_equal(
            stats["sum"],
            pd.Series(nybb_contig.lag(self.nybb.geometry.area), index=self.nybb.index),
            check_names=False,
        )

        pd.testing.assert_series_equal(
            stats["count"].sort_index(),
            nybb_contig.cardinalities.sort_index(),
            check_dtype=False,
            check_names=False,
        )

        ## test passing ndarray
        stats1 = nybb_contig.describe(self.nybb.geometry.area, statistics=["sum"])[
            "sum"
        ]
        stats2 = nybb_contig.describe(
            self.nybb.geometry.area.values, statistics=["sum"]
        )["sum"]
        pd.testing.assert_series_equal(
            stats1,
            stats2,
            check_dtype=False,
            check_names=False,
        )

        ## test index alignment
        with pytest.raises(
            ValueError, match="The values index is not aligned with the graph index."
        ):
            nybb_contig.describe(self.nybb.geometry.area.reset_index(drop=True))

    def test_summary(self):
        assert isinstance(self.g_int.summary(), graph.GraphSummary)
        assert isinstance(self.g_str.summary(), graph.GraphSummary)
        assert isinstance(self.g_str_unodered.summary(), graph.GraphSummary)

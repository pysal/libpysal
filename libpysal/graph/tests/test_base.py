import string
import os
import tempfile

import pandas as pd
import geopandas as gpd
import geodatasets
import numpy as np
import pytest
from scipy import sparse

from libpysal import graph
from libpysal import weights


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
        self.G_int = graph.Graph.from_weights_dict(self.W_dict_int)
        self.G_str = graph.Graph.from_weights_dict(self.W_dict_str)
        rng = np.random.default_rng(seed=0)
        self.letters = np.asarray(list(string.ascii_letters[:26]))
        rng.shuffle(self.letters)
        self.W_dict_str_unordered = {
            self.letters[k]: {self.letters[k_]: v_ for k_, v_ in v.items()}
            for k, v in self.W_dict_int.items()
        }
        self.G_str_unodered = graph.Graph.from_weights_dict(self.W_dict_str_unordered)

        self.nybb = gpd.read_file(geodatasets.get_path("nybb")).set_index("BoroName")

    def test_init(self):
        G = graph.Graph(self.adjacency_int_binary)
        assert isinstance(G, graph.Graph)
        assert hasattr(G, "_adjacency")
        assert G._adjacency.shape == (9,)
        pd.testing.assert_series_equal(G._adjacency, self.adjacency_int_binary)
        assert hasattr(G, "transformation")
        assert G.transformation == "O"

        G = graph.Graph(self.adjacency_str_binary)
        assert isinstance(G, graph.Graph)
        assert hasattr(G, "_adjacency")
        assert G._adjacency.shape == (9,)
        pd.testing.assert_series_equal(G._adjacency, self.adjacency_str_binary)
        assert hasattr(G, "transformation")
        assert G.transformation == "O"

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

    def test_copy(self):
        G_copy = self.G_str.copy()
        assert G_copy == self.G_str
        G_copy._adjacency.iloc[0] = 2
        assert G_copy != self.G_str

    def test_adjacency(self):
        G = graph.Graph(self.adjacency_int_binary)
        adjacency = G.adjacency
        pd.testing.assert_series_equal(adjacency, self.adjacency_int_binary)

        # ensure copy
        adjacency.iloc[0] = 100
        pd.testing.assert_series_equal(G._adjacency, self.adjacency_int_binary)

    def test_W_roundtrip(self):
        W = self.G_int.to_W()
        pd.testing.assert_series_equal(
            self.G_int._adjacency.sort_index(),
            W.to_adjlist(drop_islands=False)
            .set_index(["focal", "neighbor"])["weight"]
            .sort_index(),
            check_index_type=False,
            check_dtype=False,
        )
        G_roundtripped = graph.Graph.from_W(W)
        assert self.G_int == G_roundtripped

        W = self.G_str.to_W()
        pd.testing.assert_series_equal(
            self.G_str._adjacency.sort_index(),
            W.to_adjlist(drop_islands=False)
            .set_index(["focal", "neighbor"])["weight"]
            .sort_index(),
            check_index_type=False,
            check_dtype=False,
        )
        G_roundtripped = graph.Graph.from_W(W)
        assert self.G_str == G_roundtripped

        W = weights.lat2W(3, 3)
        G = graph.Graph.from_W(W)
        pd.testing.assert_series_equal(
            G._adjacency.sort_index(),
            W.to_adjlist().set_index(["focal", "neighbor"])["weight"].sort_index(),
            check_index_type=False,
            check_dtype=False,
        )
        W_exp = G.to_W()
        # assert W.neighbors == W_exp.neighbors
        assert W.weights == W_exp.weights

        W.transform = "r"
        G_rowwise = graph.Graph.from_W(W)
        pd.testing.assert_series_equal(
            G_rowwise._adjacency.sort_index(),
            W.to_adjlist().set_index(["focal", "neighbor"])["weight"].sort_index(),
            check_index_type=False,
            check_dtype=False,
        )
        W_trans = G_rowwise.to_W()
        # assert W.neighbors == W_trans.neighbors
        assert W.weights == W_trans.weights

        diag = weights.fill_diagonal(W)
        G_diag = graph.Graph.from_W(diag)
        pd.testing.assert_series_equal(
            G_diag._adjacency.sort_index(),
            diag.to_adjlist().set_index(["focal", "neighbor"])["weight"].sort_index(),
            check_index_type=False,
            check_dtype=False,
        )
        W_diag = G_diag.to_W()
        # assert diag.neighbors == W_diag.neighbors
        assert diag.weights == W_diag.weights

        W = weights.W(
            neighbors={"a": ["b"], "b": ["a", "c"], "c": ["b"], "d": []},
            silence_warnings=True,
        )
        G_isolate = graph.Graph.from_W(W)
        pd.testing.assert_series_equal(
            G_isolate._adjacency.sort_index(),
            W.to_adjlist().set_index(["focal", "neighbor"])["weight"].sort_index(),
            check_index_type=False,
            check_dtype=False,
        )
        W_isolate = G_isolate.to_W()
        # assert W.neighbors == W_isolate.neighbors
        assert W.weights == W_isolate.weights

        W = self.G_str_unodered.to_W()
        assert W.id_order_set
        np.testing.assert_array_equal(W.id_order, self.letters[:10])

    def test_from_sparse(self):
        row = np.array([0, 0, 1, 2, 3, 3])
        col = np.array([1, 3, 3, 2, 1, 3])
        data = np.array([0.1, 0.5, 0.9, 0, 0.3, 0.1])
        sp = sparse.coo_array((data, (row, col)), shape=(4, 4))
        G = graph.Graph.from_sparse(sp)
        expected = graph.Graph.from_arrays(row, col, data)
        assert G == expected, "sparse constructor does not match arrays constructor"

        G = graph.Graph.from_sparse(sp.tocsr())
        assert G == expected, "csc input does not match coo input"

        G = graph.Graph.from_sparse(sp.tocsc())
        assert G == expected, "csr input does not match coo input"

        ids = ["zero", "one", "two", "three"]

        G_named = graph.Graph.from_sparse(
            sp,
            ids=ids,
        )

        expected = graph.Graph.from_arrays(
            ["zero", "zero", "one", "two", "three", "three"],
            ["one", "three", "three", "two", "one", "three"],
            data,
        )
        G_named == expected

        G = graph.Graph.from_sparse(
            sp.tocsr(),
            ids=ids,
        )
        assert G == expected, "sparse csr with ids does not match arrays constructor"

        G = graph.Graph.from_sparse(
            sp.tocsc(),
            ids=ids,
        )
        assert G == (expected), "sparse csr with ids does not match arrays constructor"

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
        G = graph.Graph.from_sparse(
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
            G == expected
        ), "sparse csr nybb with ids does not match arrays constructor"
        np.testing.assert_array_equal(G.sparse.todense(), sp.todense())

        with pytest.raises(ValueError, match="The length of ids "):
            graph.Graph.from_sparse(sp, ids=["staten_island", "queens"])

    def test_from_arrays(self):
        focal_ids = np.arange(9)
        neighbor_ids = np.array([1, 2, 5, 4, 5, 8, 7, 8, 7])
        weight = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])

        G = graph.Graph.from_arrays(focal_ids, neighbor_ids, weight)
        pd.testing.assert_series_equal(
            G._adjacency,
            self.adjacency_int_binary,
            check_index_type=False,
            check_dtype=False,
        )

        focal_ids = np.asarray(list(self.neighbor_dict_str.keys()))
        neighbor_ids = np.asarray(list(self.neighbor_dict_str.values()))

        G = graph.Graph.from_arrays(focal_ids, neighbor_ids, weight)
        pd.testing.assert_series_equal(
            G._adjacency,
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

        G = graph.Graph.from_weights_dict(weights_dict)
        assert G == expected

    def test_from_dicts(self):
        G = graph.Graph.from_dicts(self.neighbor_dict_int)
        pd.testing.assert_series_equal(
            G._adjacency,
            self.adjacency_int_binary,
            check_dtype=False,
            check_index_type=False,
        )

        G = graph.Graph.from_dicts(self.neighbor_dict_str)
        pd.testing.assert_series_equal(
            G._adjacency,
            self.adjacency_str_binary,
            check_dtype=False,
        )

    @pytest.mark.parametrize("y", [3, 5])
    @pytest.mark.parametrize("id_type", ["int", "str"])
    @pytest.mark.parametrize("rook", [True, False])
    def test_from_dicts_via_W(self, y, id_type, rook):
        W = weights.lat2W(3, y, id_type=id_type, rook=rook)
        G = graph.Graph.from_dicts(W.neighbors, W.weights)
        pd.testing.assert_series_equal(
            G._adjacency.sort_index(),
            W.to_adjlist().set_index(["focal", "neighbor"])["weight"].sort_index(),
            check_index_type=False,
            check_dtype=False,
        )

        W.transform = "r"
        G_rowwise = graph.Graph.from_dicts(W.neighbors, W.weights)
        pd.testing.assert_series_equal(
            G_rowwise._adjacency.sort_index(),
            W.to_adjlist().set_index(["focal", "neighbor"])["weight"].sort_index(),
            check_index_type=False,
            check_dtype=False,
        )

        diag = weights.fill_diagonal(W)
        G_diag = graph.Graph.from_dicts(diag.neighbors, diag.weights)
        pd.testing.assert_series_equal(
            G_diag._adjacency.sort_index(),
            diag.to_adjlist().set_index(["focal", "neighbor"])["weight"].sort_index(),
            check_index_type=False,
            check_dtype=False,
        )

        W = weights.W(
            neighbors={"a": ["b"], "b": ["a", "c"], "c": ["b"], "d": []},
            silence_warnings=True,
        )
        G_isolate = graph.Graph.from_dicts(W.neighbors, W.weights)
        pd.testing.assert_series_equal(
            G_isolate._adjacency.sort_index(),
            W.to_adjlist().set_index(["focal", "neighbor"])["weight"].sort_index(),
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
        assert self.G_int.neighbors == expected

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
        assert self.G_str.neighbors == expected

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
        assert self.G_int.weights == expected

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
        assert self.G_str.weights == expected

    def test_sparse(self):
        sp = self.G_int.sparse
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

        sp = self.G_str.sparse
        np.testing.assert_array_equal(sp.todense(), expected)

        sp_old = self.G_int.to_W().sparse.todense()
        np.testing.assert_array_equal(sp.todense(), sp_old)

        sp_old = self.G_str.to_W().sparse.todense()
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
        G = graph.Graph(self.adjacency_int_binary)
        sp = G.sparse
        G_sp = graph.Graph.from_sparse(sp, self.index_int)
        assert G == G_sp

        G = graph.Graph(self.adjacency_str_binary)
        sp = G.sparse
        G_sp = graph.Graph.from_sparse(sp, self.index_str)
        assert G == G_sp

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
        pd.testing.assert_series_equal(self.G_str.cardinalities, expected)

    def test_isolates(self):
        expected = pd.Index(["j"], name="focal")
        pd.testing.assert_index_equal(self.G_str.isolates, expected)

        self.G_str._adjacency.iloc[1] = 0  # zero weight, no isolate
        pd.testing.assert_index_equal(self.G_str.isolates, expected)

    def test_n(self):
        assert self.G_int.n == 10
        assert self.G_str.n == 10
        assert graph.Graph(self.adjacency_int_binary).n == 9

    def test_pct_nonzero(self):
        assert self.G_int.pct_nonzero == 26.0
        assert graph.Graph(self.adjacency_int_binary).pct_nonzero == pytest.approx(
            11.1111111111
        )

    def test_nonzero(self):
        assert self.G_int.nonzero == 25
        assert graph.Graph(self.adjacency_int_binary).nonzero == 9

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
        exp = self.G_int.adjacency
        exp.iloc[:] = expected_w
        expected = graph.Graph(exp)
        assert self.G_int.transform("r") == expected
        assert self.G_int.transform("r").transformation == "R"
        assert self.G_int.transform("R") == expected

        w = self.G_int.to_W()
        w.transform = "r"
        G_from_W = graph.Graph.from_W(w)
        assert G_from_W == self.G_int.transform("r")

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
        exp = self.G_int.adjacency
        exp.iloc[:] = expected_w
        expected = graph.Graph(exp)
        assert self.G_int.transform("b") == expected
        assert self.G_int.transform("b").transformation == "B"
        assert self.G_int.transform("B") == expected

        w = self.G_int.to_W()
        w.transform = "b"
        G_from_W = graph.Graph.from_W(w)
        assert G_from_W == self.G_int.transform("b")

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
        exp = self.G_int.adjacency
        exp.iloc[:] = expected_w
        expected = graph.Graph(exp)
        assert self.G_int.transform("d") == expected
        assert self.G_int.transform("d").transformation == "D"
        assert self.G_int.transform("D") == expected
        assert self.G_int.transform("D")._adjacency.sum() == pytest.approx(1)

        w = self.G_int.to_W()
        w.transform = "d"
        G_from_W = graph.Graph.from_W(w)
        assert G_from_W == self.G_int.transform("d")

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
        exp = self.G_int.adjacency
        exp.iloc[:] = expected_w
        expected = graph.Graph(exp)
        assert self.G_int.transform("v") == expected
        assert self.G_int.transform("v").transformation == "V"
        assert self.G_int.transform("V") == expected

        w = self.G_int.to_W()
        w.transform = "v"
        G_from_W = graph.Graph.from_W(w)
        assert G_from_W == self.G_int.transform("v")

    def test_transform(self):
        # do not transform if transformation == current transformation
        binary = self.G_int.transform("b")
        fast_tracked = binary.transform("b")
        assert binary == fast_tracked

        with pytest.raises(ValueError, match="Transformation 'X' is not"):
            self.G_int.transform("x")

    def test_asymmetry(self):
        neighbors = {
            "a": ["b", "c", "d"],
            "b": ["b", "c", "d"],
            "c": ["a", "b"],
            "d": ["a", "b"],
        }
        weights_d = {"a": [1, 0.5, 1], "b": [1, 1, 1], "c": [1, 1], "d": [1, 1]}
        G = graph.Graph.from_dicts(neighbors, weights_d)
        intrinsic = pd.Series(
            ["b", "c", "a", "a"],
            index=pd.Index(["a", "a", "b", "c"], name="focal"),
            name="neighbor",
        )
        pd.testing.assert_series_equal(intrinsic, G.asymmetry())

        boolean = pd.Series(
            ["b", "a"],
            index=pd.Index(["a", "b"], name="focal"),
            name="neighbor",
        )
        pd.testing.assert_series_equal(boolean, G.asymmetry(intrinsic=False))

        empty = pd.Series(
            index=pd.Index([], name="focal"),
            name="neighbor",
            dtype="int64",
        )

        pd.testing.assert_series_equal(self.G_int.asymmetry(False), empty)

    def test_parquet(self):
        pytest.importorskip("pyarrow")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "g_int.parquet")
            self.G_int.to_parquet(path)
            G_int = graph.read_parquet(path)
            assert self.G_int == G_int

            path = os.path.join(tmpdir, "g_str.parquet")
            self.G_str.to_parquet(path)
            G_str = graph.read_parquet(path)
            assert self.G_str == G_str

            row_wise = self.G_str.transform("r")
            path = os.path.join(tmpdir, "row.parquet")
            row_wise.to_parquet(path)
            row_read = graph.read_parquet(path)
            assert row_wise == row_read
            assert row_read.transformation == "R"

            path = os.path.join(tmpdir, "pandas.parquet")
            self.G_str._adjacency.to_frame().to_parquet(path)
            G_pandas = graph.read_parquet(path)
            assert self.G_str == G_pandas

    def test_getitem(self):
        expected = pd.Series(
            [1, 0.5, 0.5],
            index=pd.Index(["a", "b", "d"], name="neighbor"),
            name="weight",
        )
        pd.testing.assert_series_equal(expected, self.G_str["a"])

        expected = pd.Series(
            [1, 0.5, 0.5],
            index=pd.Index([0, 1, 3], name="neighbor"),
            name="weight",
        )
        pd.testing.assert_series_equal(expected, self.G_int[0])

        # isolate
        expected = pd.Series(
            [],
            index=pd.Index([], name="neighbor"),
            name="weight",
        )
        pd.testing.assert_series_equal(expected, self.G_str["j"])

    def test_lag(self):
        expected = np.array([4.0, 2.7, 4.0, 3.9, 5.0, 5.1, 6.0, 6.3, 7.0, 0.0])
        lag = self.G_str.lag(list(range(1, 11)))

        np.testing.assert_allclose(expected, lag)

        with pytest.raises(ValueError, match="The length of `y`"):
            self.G_str.lag(list(range(1, 15)))

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

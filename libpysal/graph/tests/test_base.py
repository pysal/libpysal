import string

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
        self.adjacency_int_binary = pd.DataFrame(
            {
                "neighbor": self.neighbor_dict_int,
                "weight": self.weight_dict_int_binary,
            },
            index=self.index_int,
        )
        self.adjacency_str_binary = pd.DataFrame(
            {
                "neighbor": self.neighbor_dict_str,
                "weight": self.weight_dict_str_binary,
            },
            index=self.index_str,
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

    def test_init(self):
        G = graph.Graph(self.adjacency_int_binary)
        assert isinstance(G, graph.Graph)
        assert hasattr(G, "_adjacency")
        assert G._adjacency.shape == (9, 2)
        pd.testing.assert_frame_equal(G._adjacency, self.adjacency_int_binary)
        assert hasattr(G, "transformation")
        assert G.transformation == "O"

        G = graph.Graph(self.adjacency_str_binary)
        assert isinstance(G, graph.Graph)
        assert hasattr(G, "_adjacency")
        assert G._adjacency.shape == (9, 2)
        pd.testing.assert_frame_equal(G._adjacency, self.adjacency_str_binary)
        assert hasattr(G, "transformation")
        assert G.transformation == "O"

        with pytest.raises(TypeError, match="The adjacency table needs to be"):
            graph.Graph(self.adjacency_int_binary.values)

        with pytest.raises(ValueError, match="The shape of the adjacency table"):
            graph.Graph(self.adjacency_int_binary.assign(col=0))

        with pytest.raises(ValueError, match="The index of the adjacency table"):
            adj = self.adjacency_int_binary.copy()
            adj.index.name = "foo"
            graph.Graph(adj)

        with pytest.raises(
            ValueError, match="The adjacency table needs to contain columns"
        ):
            graph.Graph(self.adjacency_int_binary.rename(columns={"weight": "foo"}))

        with pytest.raises(ValueError, match="The 'weight' column"):
            graph.Graph(self.adjacency_int_binary.astype(str))

        with pytest.raises(
            ValueError, match="The adjacency table cannot contain missing"
        ):
            adj = self.adjacency_int_binary.copy()
            adj.loc[0, "weight"] = np.nan
            graph.Graph(adj)

        with pytest.raises(ValueError, match="'transformation' needs to be"):
            graph.Graph(self.adjacency_int_binary, transformation="foo")

    def test_copy(self):
        G_copy = self.G_str.copy()
        assert G_copy == self.G_str
        G_copy._adjacency.iloc[0, 1] = 2
        assert G_copy != self.G_str

    def test_adjacency(self):
        G = graph.Graph(self.adjacency_int_binary)
        adjacency = G.adjacency
        pd.testing.assert_frame_equal(adjacency, self.adjacency_int_binary)

        # ensure copy
        adjacency.iloc[0, 0] = 100
        pd.testing.assert_frame_equal(G._adjacency, self.adjacency_int_binary)

    def test_W_roundtrip(self):
        W = self.G_int.to_W()
        pd.testing.assert_frame_equal(
            self.G_int._adjacency.sort_values(["focal", "neighbor"]),
            W.to_adjlist(drop_islands=False)
            .set_index("focal")
            .sort_values(["focal", "neighbor"]),
            check_index_type=False,
            check_dtype=False,
        )
        G_roundtripped = graph.Graph.from_W(W)
        assert self.G_int == G_roundtripped

        W = self.G_str.to_W()
        pd.testing.assert_frame_equal(
            self.G_str._adjacency.sort_values(["focal", "neighbor"]),
            W.to_adjlist(drop_islands=False)
            .set_index("focal")
            .sort_values(["focal", "neighbor"]),
            check_index_type=False,
            check_dtype=False,
        )
        G_roundtripped = graph.Graph.from_W(W)
        assert self.G_str == G_roundtripped

        W = weights.lat2W(3, 3)
        G = graph.Graph.from_W(W)
        pd.testing.assert_frame_equal(
            G._adjacency.sort_values(["focal", "neighbor"]),
            W.to_adjlist().set_index("focal").sort_values(["focal", "neighbor"]),
            check_index_type=False,
            check_dtype=False,
        )
        W_exp = G.to_W()
        assert W.neighbors == W_exp.neighbors
        assert W.weights == W_exp.weights

        W.transform = "r"
        G_rowwise = graph.Graph.from_W(W)
        pd.testing.assert_frame_equal(
            G_rowwise._adjacency.sort_values(["focal", "neighbor"]),
            W.to_adjlist().set_index("focal").sort_values(["focal", "neighbor"]),
            check_index_type=False,
            check_dtype=False,
        )
        W_trans = G_rowwise.to_W()
        assert W.neighbors == W_trans.neighbors
        assert W.weights == W_trans.weights

        diag = weights.fill_diagonal(W)
        G_diag = graph.Graph.from_W(diag)
        pd.testing.assert_frame_equal(
            G_diag._adjacency.sort_values(["focal", "neighbor"]),
            diag.to_adjlist().set_index("focal").sort_values(["focal", "neighbor"]),
            check_index_type=False,
            check_dtype=False,
        )
        W_diag = G_diag.to_W()
        assert diag.neighbors == W_diag.neighbors
        assert diag.weights == W_diag.weights

        W = weights.W(
            neighbors={"a": ["b"], "b": ["a", "c"], "c": ["b"], "d": []},
            silence_warnings=True,
        )
        G_isolate = graph.Graph.from_W(W)
        pd.testing.assert_frame_equal(
            G_isolate._adjacency.sort_values(["focal", "neighbor"]),
            W.to_adjlist().set_index("focal").sort_values(["focal", "neighbor"]),
            check_index_type=False,
            check_dtype=False,
        )
        W_isolate = G_isolate.to_W()
        assert W.neighbors == W_isolate.neighbors
        assert W.weights == W_isolate.weights

    def test_from_sparse(self):
        row = np.array([0, 3, 1, 0])
        col = np.array([1, 0, 1, 2])
        data = np.array([0.1, 0.5, 1, 0.9])
        sp = sparse.coo_array((data, (row, col)), shape=(4, 4))
        G = graph.Graph.from_sparse(sp)
        expected = graph.Graph.from_arrays(row, col, data)
        assert G == expected

        G = graph.Graph.from_sparse(sp.tocsr())
        assert G == expected

        G = graph.Graph.from_sparse(sp.tocsc())
        assert G == expected

        ids = ["zero", "one", "two", "three"]

        G_named = graph.Graph.from_sparse(
            sp,
            ids=ids,
        )

        expected = graph.Graph.from_arrays(
            ["zero", "three", "one", "zero"], ["one", "zero", "one", "two"], data
        )
        G_named == expected

        G = graph.Graph.from_sparse(
            sp.tocsr(),
            ids=ids,
        )
        assert G == expected

        G = graph.Graph.from_sparse(
            sp.tocsc(),
            ids=ids,
        )
        assert G == expected

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
        assert G == expected
        np.testing.assert_array_equal(G.sparse.todense(), sp.todense())

    def test_from_arrays(self):
        focal_ids = np.arange(9)
        neighbor_ids = np.array([1, 2, 5, 4, 5, 8, 7, 8, 7])
        weight = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])

        G = graph.Graph.from_arrays(focal_ids, neighbor_ids, weight)
        pd.testing.assert_frame_equal(
            G._adjacency,
            self.adjacency_int_binary,
            check_index_type=False,
            check_dtype=False,
        )

        focal_ids = np.asarray(list(self.neighbor_dict_str.keys()))
        neighbor_ids = np.asarray(list(self.neighbor_dict_str.values()))

        G = graph.Graph.from_arrays(focal_ids, neighbor_ids, weight)
        pd.testing.assert_frame_equal(
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
        pd.testing.assert_frame_equal(
            G._adjacency,
            self.adjacency_int_binary,
            check_dtype=False,
            check_index_type=False,
        )

        G = graph.Graph.from_dicts(self.neighbor_dict_str)
        pd.testing.assert_frame_equal(
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
        pd.testing.assert_frame_equal(
            G._adjacency.sort_values(["focal", "neighbor"]),
            W.to_adjlist().set_index("focal").sort_values(["focal", "neighbor"]),
            check_index_type=False,
            check_dtype=False,
        )

        W.transform = "r"
        G_rowwise = graph.Graph.from_dicts(W.neighbors, W.weights)
        pd.testing.assert_frame_equal(
            G_rowwise._adjacency.sort_values(["focal", "neighbor"]),
            W.to_adjlist().set_index("focal").sort_values(["focal", "neighbor"]),
            check_index_type=False,
            check_dtype=False,
        )

        diag = weights.fill_diagonal(W)
        G_diag = graph.Graph.from_dicts(diag.neighbors, diag.weights)
        pd.testing.assert_frame_equal(
            G_diag._adjacency.sort_values(["focal", "neighbor"]),
            diag.to_adjlist().set_index("focal").sort_values(["focal", "neighbor"]),
            check_index_type=False,
            check_dtype=False,
        )

        W = weights.W(
            neighbors={"a": ["b"], "b": ["a", "c"], "c": ["b"], "d": []},
            silence_warnings=True,
        )
        G_isolate = graph.Graph.from_dicts(W.neighbors, W.weights)
        pd.testing.assert_frame_equal(
            G_isolate._adjacency.sort_values(["focal", "neighbor"]),
            W.to_adjlist().set_index("focal").sort_values(["focal", "neighbor"]),
            check_index_type=False,
            check_dtype=False,
        )

    def test_neighbors(self):
        expected = {
            0: (0, 3, 1),
            1: (0, 4, 2),
            2: (1, 5),
            3: (0, 6, 4),
            4: (1, 3, 7, 5),
            5: (2, 4, 8),
            6: (3, 7),
            7: (4, 6, 8),
            8: (5, 7),
            9: (),
        }
        assert self.G_int.neighbors == expected

        expected = {
            "a": ("a", "d", "b"),
            "b": ("a", "e", "c"),
            "c": ("b", "f"),
            "d": ("a", "g", "e"),
            "e": ("b", "d", "h", "f"),
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

    def test_get_neighbors(self):
        for i in range(10):
            np.testing.assert_array_equal(
                self.G_int.get_neighbors(i), np.asarray(list(self.W_dict_int[i]))
            )
        for i in range(10):
            i = string.ascii_letters[i]
            np.testing.assert_array_equal(
                self.G_str.get_neighbors(i), np.asarray(list(self.W_dict_str[i]))
            )

    def test_get_weights(self):
        for i in range(10):
            np.testing.assert_array_equal(
                self.G_int.get_weights(i), np.asarray(list(self.W_dict_int[i].values()))
            )
        for i in range(10):
            i = string.ascii_letters[i]
            np.testing.assert_array_equal(
                self.G_str.get_weights(i), np.asarray(list(self.W_dict_str[i].values()))
            )

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
        nybb = graph.Graph.build_contiguity(
            gpd.read_file(geodatasets.get_path("nybb")).set_index("BoroName")
        )
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
        G_sp = graph.Graph.from_sparse(sp, np.asarray(list(G._id2i.keys())))
        assert G == G_sp

        G = graph.Graph(self.adjacency_str_binary)
        sp = G.sparse
        G_sp = graph.Graph.from_sparse(sp, np.asarray(list(G._id2i.keys())))
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

        self.G_str._adjacency.iloc[1, 1] = 0  # zero weight, no isolate
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
        assert self.G_int.nonzero == 26
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
        expected = graph.Graph(self.G_int.adjacency.assign(weight=expected_w))
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
        expected = graph.Graph(self.G_int.adjacency.assign(weight=expected_w))
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
        expected = graph.Graph(self.G_int.adjacency.assign(weight=expected_w))
        assert self.G_int.transform("d") == expected
        assert self.G_int.transform("d").transformation == "D"
        assert self.G_int.transform("D") == expected
        assert self.G_int.transform("D")._adjacency.weight.sum() == pytest.approx(1)

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

        expected = graph.Graph(self.G_int.adjacency.assign(weight=expected_w))
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

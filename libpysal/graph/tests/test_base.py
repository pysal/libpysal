import pandas as pd
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
            "a": "b",
            "b": "c",
            "c": "f",
            "d": "e",
            "e": "f",
            "f": "i",
            "g": "h",
            "h": "i",
            "i": "h",
        }
        self.weight_dict_str_binary = {
            "a": 1,
            "b": 1,
            "c": 1,
            "d": 1,
            "e": 1,
            "f": 1,
            "g": 1,
            "h": 1,
            "i": 1,
        }
        self.index_str = pd.Index(
            [
                "a",
                "b",
                "c",
                "d",
                "e",
                "f",
                "g",
                "h",
                "i",
            ],
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
            adj.loc[0, "weight"] = pd.NA
            graph.Graph(adj)

        with pytest.raises(ValueError, match="'transformation' needs to be"):
            graph.Graph(self.adjacency_int_binary, transformation="foo")

    def test_adjacency(self):
        G = graph.Graph(self.adjacency_int_binary)
        adjacency = G.adjacency
        pd.testing.assert_frame_equal(adjacency, self.adjacency_int_binary)

        # ensure copy
        adjacency.iloc[0, 0] = 100
        pd.testing.assert_frame_equal(G._adjacency, self.adjacency_int_binary)

    @pytest.mark.parametrize("y", [3, 5])
    @pytest.mark.parametrize("rook", [True, False])
    @pytest.mark.parametrize("id_type", ["int", "str"])
    def test_W_roundtrip(self, y, id_type, rook):
        W = weights.lat2W(3, y, id_type=id_type, rook=rook)
        G = graph.Graph.from_W(W)
        pd.testing.assert_frame_equal(
            G._adjacency.sort_values(["focal", "neighbor"]),
            W.to_adjlist().set_index("focal").sort_values(["focal", "neighbor"]),
        )
        W_exp = G.to_W()
        assert W.neighbors == W_exp.neighbors
        assert W.weights == W_exp.weights

        W.transform = "r"
        G_rowwise = graph.Graph.from_W(W)
        pd.testing.assert_frame_equal(
            G_rowwise._adjacency.sort_values(["focal", "neighbor"]),
            W.to_adjlist().set_index("focal").sort_values(["focal", "neighbor"]),
        )
        W_trans = G_rowwise.to_W()
        assert W.neighbors == W_trans.neighbors
        assert W.weights == W_trans.weights

        diag = weights.fill_diagonal(W)
        G_diag = graph.Graph.from_W(diag)
        pd.testing.assert_frame_equal(
            G_diag._adjacency.sort_values(["focal", "neighbor"]),
            diag.to_adjlist().set_index("focal").sort_values(["focal", "neighbor"]),
        )
        W_diag = G_diag.to_W()
        assert diag.neighbors == W_diag.neighbors
        # assert diag.weights == W_diag.weights  # buggy due to #538

        W = weights.W(
            neighbors={"a": ["b"], "b": ["a", "c"], "c": ["b"], "d": []},
            silence_warnings=True,
        )
        G_isolate = graph.Graph.from_W(W)
        pd.testing.assert_frame_equal(
            G_isolate._adjacency.sort_values(["focal", "neighbor"]),
            W.to_adjlist().set_index("focal").sort_values(["focal", "neighbor"]),
        )
        W_isolate = G_isolate.to_W()
        assert W.neighbors == W_isolate.neighbors
        assert W.weights == W_isolate.weights

    def test_sparse_roundtrip(self):
        G = graph.Graph(self.adjacency_int_binary)
        sp = G.sparse
        G_sp = graph.Graph.from_sparse(sp, G.focal_label, G.neighbor_label)
        pd.testing.assert_frame_equal(G._adjacency, G_sp._adjacency)

        G = graph.Graph(self.adjacency_str_binary)
        sp = G.sparse
        G_sp = graph.Graph.from_sparse(sp, G.focal_label, G.neighbor_label)
        pd.testing.assert_frame_equal(G._adjacency, G_sp._adjacency)

    def test_from_sparse(self):
        row = np.array([0, 3, 1, 0])
        col = np.array([1, 0, 1, 2])
        data = np.array([0.1, 0.5, 1, 0.9])
        sp = sparse.coo_array((data, (row, col)), shape=(4, 4))
        G = graph.Graph.from_sparse(sp)
        expected = pd.DataFrame(
            {"focal": row, "neighbor": col, "weight": data}
        ).set_index("focal")
        pd.testing.assert_frame_equal(G._adjacency, expected)

        G_named = graph.Graph.from_sparse(
            sp,
            focal_ids=["zero", "one", "two", "three"],
            neighbor_ids=["zero", "one", "two", "three"],
        )
        expected = pd.DataFrame(
            {
                "focal": ["zero", "three", "one", "zero"],
                "neighbor": ["one", "zero", "one", "two"],
                "weight": data,
            }
        ).set_index("focal")
        pd.testing.assert_frame_equal(G_named._adjacency, expected)

        with pytest.raises(ValueError, match="Either both"):
            graph.Graph.from_sparse(
                sp,
                focal_ids=["zero", "one", "two", "three"],
                neighbor_ids=None,
            )

    def test_from_arrays(self):
        focal_ids = np.arange(9)
        neighbor_ids = np.array([1, 2, 5, 4, 5, 8, 7, 8, 7])
        weight = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])

        G = graph.Graph.from_arrays(focal_ids, neighbor_ids, weight)
        pd.testing.assert_frame_equal(G._adjacency, self.adjacency_int_binary)

        focal_ids = np.asarray(self.neighbor_dict_str.keys())
        neighbor_ids = np.asarray(self.neighbor_dict_str.values())

        G = graph.Graph.from_arrays(focal_ids, neighbor_ids, weight)
        pd.testing.assert_frame_equal(G._adjacency, self.adjacency_str_binary)

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
        expected = pd.DataFrame(
            {"neighbor": exp_neighbor, "weight": exp_weight},
            index=exp_focal,
        )
        G = graph.Graph.from_weights_dict(weights_dict)
        pd.testing.assert_frame_equal(G._adjacency, expected)

    def test_from_dicts(self):
        G = graph.Graph.from_dicts(self.neighbor_dict_int)
        pd.testing.assert_frame_equal(G._adjacency, self.adjacency_int_binary)

        G = graph.Graph.from_dicts(self.neighbor_dict_str)
        pd.testing.assert_frame_equal(G._adjacency, self.adjacency_str_binary)


    @pytest.mark.parametrize("y", [3, 5])
    @pytest.mark.parametrize("rook", [True, False])
    @pytest.mark.parametrize("id_type", ["int", "str"])
    def test_from_dicts_via_W(self, y, id_type, rook):
        W = weights.lat2W(3, y, id_type=id_type, rook=rook)
        G = graph.Graph.from_dicts(W.neighbors, W.weights)
        pd.testing.assert_frame_equal(
            G._adjacency.sort_values(["focal", "neighbor"]),
            W.to_adjlist().set_index("focal").sort_values(["focal", "neighbor"]),
        )

        W.transform = "r"
        G_rowwise = graph.Graph.from_dicts(W.neighbors, W.weights)
        pd.testing.assert_frame_equal(
            G_rowwise._adjacency.sort_values(["focal", "neighbor"]),
            W.to_adjlist().set_index("focal").sort_values(["focal", "neighbor"]),
        )

        diag = weights.fill_diagonal(W)
        G_diag = graph.Graph.from_dicts(diag.neighbors, diag.weights)
        pd.testing.assert_frame_equal(
            G_diag._adjacency.sort_values(["focal", "neighbor"]),
            diag.to_adjlist().set_index("focal").sort_values(["focal", "neighbor"]),
        )

        W = weights.W(
            neighbors={"a": ["b"], "b": ["a", "c"], "c": ["b"], "d": []},
            silence_warnings=True,
        )
        G_isolate = graph.Graph.from_dicts(W.neighbors, W.weights)
        pd.testing.assert_frame_equal(
            G_isolate._adjacency.sort_values(["focal", "neighbor"]),
            W.to_adjlist().set_index("focal").sort_values(["focal", "neighbor"]),
        )

# TODO: test additional attributes
# TODO: test additional methods

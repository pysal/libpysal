import pandas as pd
import pytest

from libpysal import graph
from libpysal import weights


class TestBase:
    def setup_method(self):
        self.neighbor_dict_int = {0: 1, 1: 2, 2: 5, 3: 4, 4: 5, 5: 8, 6: 7, 7: 8, 8: 7}
        self.weight_dict_binary = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1}
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
                "weight": self.weight_dict_binary,
            },
            index=self.index_int,
        )
        self.adjacency_str_binary = pd.DataFrame(
            {
                "neighbor": self.neighbor_dict_int,
                "weight": self.weight_dict_binary,
            },
            index=self.index_int,
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

    def test_adjacency(self):
        G = graph.Graph(self.adjacency_int_binary)
        adjacency = G.adjacency
        pd.testing.assert_frame_equal(adjacency, self.adjacency_int_binary)

        # ensure copy
        adjacency.iloc[0, 0] = 100
        pd.testing.assert_frame_equal(G._adjacency, self.adjacency_int_binary)

    @pytest.mark.parametrize("x", [3, 4, 5, 6])
    @pytest.mark.parametrize("y", [3, 4, 5, 6])
    @pytest.mark.parametrize("rook", [True, False])
    @pytest.mark.parametrize("id_type", ["int", "str"])
    def test_W_roundtrip(self, x, y, id_type, rook):
        W = weights.lat2W(x, y, id_type=id_type, rook=rook)
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
        W_trans = G_rowwise.to_W()
        assert W.neighbors == W_trans.neighbors
        assert W.weights == W_trans.weights

        diag = weights.fill_diagonal(W)
        # TODO: test all corner cases (diagonals, islands)


# TODO: test additional attributes
# TODO: test additional methods
# TODO: it may be useful to get lat2Graph working for that

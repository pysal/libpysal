import pytest
import pandas as pd

from libpysal import graph


@pytest.fixture()
def neighbor_dict_int():
    return {0: 1, 1: 2, 2: 5, 3: 4, 4: 5, 5: 8, 6: 7, 7: 8, 8: 7}


@pytest.fixture()
def weight_dict_binary():
    return {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1}


@pytest.fixture()
def index_int():
    return pd.Index(
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
        dtype="int64",
        name="focal",
    )


@pytest.fixture()
def neighbor_dict_str():
    return {
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


@pytest.fixture()
def index_str():
    return pd.Index(
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


@pytest.fixture()
def adjacency_int_binary(neighbor_dict_int, weight_dict_binary, index_int):
    return pd.DataFrame(
        {
            "neighbor": neighbor_dict_int,
            "weight": weight_dict_binary,
        },
        index=index_int,
    )


@pytest.fixture()
def adjacency_str_binary(neighbor_dict_int, weight_dict_binary, index_int):
    return pd.DataFrame(
        {
            "neighbor": neighbor_dict_int,
            "weight": weight_dict_binary,
        },
        index=index_int,
    )


@pytest.fixture()
def graph_int_binary(adjacency_int_binary):
    return graph.Graph(adjacency_int_binary)


@pytest.fixture()
def graph_str_binary(adjacency_str_binary):
    return graph.Graph(adjacency_str_binary)


def test_init_int(graph_int_binary, adjacency_int_binary):
    G = graph_int_binary
    assert isinstance(G, graph.Graph)
    assert hasattr(G, "_adjacency")
    assert G._adjacency.shape == (9, 2)
    pd.testing.assert_frame_equal(G._adjacency, adjacency_int_binary)
    assert hasattr(G, "transformation")
    assert G.transformation == "O"


def test_init_str(graph_str_binary, adjacency_str_binary):
    G = graph_str_binary
    assert isinstance(G, graph.Graph)
    assert hasattr(G, "_adjacency")
    assert G._adjacency.shape == (9, 2)
    pd.testing.assert_frame_equal(G._adjacency, adjacency_str_binary)
    assert hasattr(G, "transformation")
    assert G.transformation == "O"


def test_adjacency(graph_int_binary, adjacency_int_binary):
    adjacency = graph_int_binary.adjacency
    pd.testing.assert_frame_equal(adjacency, adjacency_int_binary)

    # ensure copy
    adjacency.iloc[0, 0] = 100
    pd.testing.assert_frame_equal(graph_int_binary._adjacency, adjacency_int_binary)


# TODO: test additional attributes
# TODO: test additional methods
# TODO: it may be useful to get lat2Graph working for that

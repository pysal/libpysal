"""
For completeness, we need to test a shuffled dataframe
(i.e. always send unsorted data) with:
- numeric ids
- string ids
- point dataframe
- coordinates
- check two kernel functions
- check two tree types
- scikit/no scikit
"""

import geodatasets
import geopandas
import numpy as np
import pytest
import pandas as pd

from libpysal.graph._kernel import (
    _kernel,
    _kernel_functions,
    _distance_band,
    HAS_SKLEARN,
)

grocs = geopandas.read_file(geodatasets.get_path("geoda groceries"))[
    ["OBJECTID", "geometry"]
].explode(ignore_index=True)
grocs["strID"] = grocs.OBJECTID.astype(str)
grocs["intID"] = grocs.OBJECTID.values

kernel_functions = list(_kernel_functions.keys())


def my_kernel(distances, bandwidth):
    output = np.cos(distances / distances.max())
    output[distances < bandwidth] = 0
    return output


kernel_functions.append(my_kernel)

metrics = ("euclidean", "haversine")

np.random.seed(6301)
# create a 2-d laplace distribution as a "degenerate"
# over-concentrated distribution
# and rescale to match the lenght-scale in groceries
lap_coords = np.random.laplace(size=(200, 2)) / 50

# create a 2-d cauchy as a "degenerate"
# spatial outlier-y distribution
cau_coords = np.random.standard_cauchy(size=(200, 2))

data = (grocs, lap_coords)

parametrize_ids = pytest.mark.parametrize("ids", [None, "strID", "intID"])
parametrize_data = pytest.mark.parametrize("data", [grocs, lap_coords, cau_coords])
parametrize_kernelfunctions = pytest.mark.parametrize("kernel", kernel_functions)
parametrize_metrics = pytest.mark.parametrize("metric", metrics, metrics)

# how do we parameterize conditional on sklearn in env?


@parametrize_ids
def test_neighbors(ids):
    if ids:
        data = grocs.set_index(ids)
    else:
        data = grocs
    head, tail, weight = _kernel(data, bandwidth=5000, kernel="boxcar")
    assert head.shape[0] == 437
    assert tail.shape == head.shape
    assert weight.shape == head.shape
    np.testing.assert_array_equal(pd.unique(head), data.index)
    known = np.linspace(9, 436, 10, dtype=int)
    head_exp = [2, 16, 28, 41, 55, 72, 92, 111, 135, 147]
    if ids:
        head_exp = data.index.values[head_exp]
    np.testing.assert_array_equal(head.values[known], head_exp)
    tail_exp = [13, 92, 66, 31, 33, 73, 16, 103, 9, 147]
    if ids:
        tail_exp = data.index.values[tail_exp]
    np.testing.assert_array_equal(tail.values[known], tail_exp)
    np.testing.assert_array_equal(
        weight[known],
        np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0], dtype="int64"),
    )


@parametrize_data
def test_no_taper(data):
    head, tail, weight = _kernel(data, taper=False)
    assert head.shape[0] == len(data) * (len(data) - 1)
    assert tail.shape == head.shape
    assert weight.shape == head.shape
    if hasattr(data, "index"):
        np.testing.assert_array_equal(np.unique(head), data.index)
    else:
        np.testing.assert_array_equal(np.unique(head), np.arange(len(data)))


@parametrize_ids
def test_ids(ids):
    if ids:
        data = grocs.set_index(ids)
    else:
        data = grocs
    head, tail, _ = _kernel(data)
    np.testing.assert_array_equal(pd.unique(head), data.index)
    assert np.in1d(tail, data.index).all()


def test_distance():
    _, _, weight = _kernel(grocs, kernel="identity")
    known = np.linspace(9, weight.shape[0], 10, dtype=int, endpoint=False)
    np.testing.assert_array_almost_equal(
        weight[known],
        np.array(
            [
                39028.10991144,
                51086.85388002,
                21270.55278224,
                8999.11607504,
                91203.25966722,
                36548.75743352,
                58917.81440314,
                63359.35143896,
                24952.48387721,
                65860.55093353,
            ]
        ),
    )

    _, _, weight = _kernel(lap_coords, kernel="identity")
    known = np.linspace(9, weight.shape[0], 10, dtype=int, endpoint=False)
    np.testing.assert_array_almost_equal(
        weight[known],
        np.array(
            [
                0.0112305,
                0.03158595,
                0.06027445,
                0.01274032,
                0.07559474,
                0.02240698,
                0.07024776,
                0.05554498,
                0.06012029,
                0.03836196,
            ]
        ),
    )

    _, _, weight = _kernel(cau_coords, kernel="identity")
    known = np.linspace(9, weight.shape[0], 10, dtype=int, endpoint=False)
    np.testing.assert_array_almost_equal(
        weight[known],
        np.array(
            [
                5.16946327,
                2.29669606,
                1.60345402,
                1.23831204,
                19.2459334,
                2.99545291,
                2.33316209,
                2.09711302,
                2.12594958,
                6.19347592,
            ]
        ),
    )


@parametrize_data
def test_k(data):
    head, tail, weight = _kernel(data, k=3, kernel="identity")
    assert head.shape[0] == data.shape[0] * 3
    assert tail.shape == head.shape
    assert weight.shape == head.shape

    # TODO: what shall be the behaviour when k is set but bandwidth
    # TODO: is too small so weight is zero, eliminating the link with taper?
    # TODO: now the affected focal has less neighbors than K without warning
    # TODO: test it with the code above and default kernel


@parametrize_kernelfunctions
def test_kernels(kernel):
    _, _, weight = _kernel(grocs, kernel=kernel, taper=False)
    if kernel == "triangular":
        assert weight.mean() == pytest.approx(0.09416822598434019)
        assert weight.max() == pytest.approx(0.9874476868358023)
    elif kernel == "parabolic":
        assert weight.mean() == pytest.approx(0.10312196315841769)
        assert weight.max() == pytest.approx(0.749881829575671)
    elif kernel == "gaussian":
        assert weight.mean() == pytest.approx(0.1124559308071747)
        assert weight.max() == pytest.approx(0.22507021331712948)
    elif kernel == "bisquare":
        assert weight.mean() == pytest.approx(0.09084085210598618)
        assert weight.max() == pytest.approx(0.9372045972129259)
    elif kernel == "cosine":
        assert weight.mean() == pytest.approx(0.1008306468068958)
        assert weight.max() == pytest.approx(0.7852455006403666)
    elif kernel == "boxcar":
        assert weight.mean() == pytest.approx(0.2499540356683214)
        assert weight.max() == 1
    elif kernel == "discrete":
        assert weight.mean() == pytest.approx(0.2499540356683214)
        assert weight.max() == 1
    elif kernel == "identity":
        assert weight.mean() == pytest.approx(39758.007361814016)
        assert weight.max() == pytest.approx(127937.75271993055)
    elif kernel is None:
        assert weight.mean() == pytest.approx(39758.007361814016)
        assert weight.max() == pytest.approx(127937.75271993055)
    else:  # function
        assert weight.mean() == pytest.approx(0.6880384553732511)
        assert weight.max() == pytest.approx(0.9855481738848647)


@parametrize_data
@pytest.mark.parametrize("bandwidth", [None, 0.05, 0.4])
def test_bandwidth(data, bandwidth):
    head, tail, weight = _kernel(data, bandwidth=bandwidth)
    assert tail.shape == head.shape
    assert weight.shape == head.shape
    if hasattr(data, "index"):
        np.testing.assert_array_equal(np.unique(head), data.index)
    else:
        np.testing.assert_array_equal(np.unique(head), np.arange(len(data)))


@pytest.mark.parametrize(
    "metric",
    [
        "euclidean",
        "minkowski",
        "cityblock",
        "chebyshev",
        "haversine",
    ],
)
def test_metric(metric):
    if metric == "haversine":
        data = grocs.to_crs(4326)
    else:
        data = grocs
    if not HAS_SKLEARN and metric in ["chebyshev", "haversine"]:
        pytest.skip("metric not supported by scipy")
    head, tail, weight = _kernel(data, metric=metric, kernel="identity", p=1.5)
    assert head.shape[0] == len(data) * (len(data) - 1)
    assert tail.shape == head.shape
    assert weight.shape == head.shape
    np.testing.assert_array_equal(pd.unique(head), data.index)

    if metric == "euclidean":
        assert weight.mean() == pytest.approx(39758.007362)
        assert weight.max() == pytest.approx(127937.75272)
    elif metric == "minkowski":
        assert weight.mean() == pytest.approx(42288.642129)
        assert weight.max() == pytest.approx(140674.095752)
    elif metric == "cityblock":
        assert weight.mean() == pytest.approx(49424.576155)
        assert weight.max() == pytest.approx(173379.431622)
    elif metric == "chebyshev":
        assert weight.mean() == pytest.approx(36590.352895)
        assert weight.max() == pytest.approx(123955.14249)
    else:
        assert weight.mean() == pytest.approx(0.115835)
        assert weight.max() == pytest.approx(0.371465)


@pytest.mark.parametrize(
    "metric",
    [
        "euclidean",
        "minkowski",
        "cityblock",
        "chebyshev",
        "haversine",
    ],
)
def test_metric_k(metric):
    if metric == "haversine":
        data = grocs.to_crs(4326)
    else:
        data = grocs
    if not HAS_SKLEARN and metric in ["chebyshev", "haversine"]:
        pytest.skip("metric not supported by scipy")
    head, tail, weight = _kernel(data, k=3, metric=metric, kernel="identity", p=1.5)
    assert head.shape[0] == len(data) * 3
    assert tail.shape == head.shape
    assert weight.shape == head.shape
    np.testing.assert_array_equal(pd.unique(head), data.index)

    if metric == "euclidean":
        assert weight.mean() == pytest.approx(4577.237441)
        assert weight.max() == pytest.approx(18791.085051)
    elif metric == "minkowski":
        assert weight.mean() == pytest.approx(4884.254721)
        assert weight.max() == pytest.approx(20681.125211)
    elif metric == "cityblock":
        assert weight.mean() == pytest.approx(5665.288523)
        assert weight.max() == pytest.approx(23980.893147)
    elif metric == "chebyshev":
        assert weight.mean() == pytest.approx(4032.283559)
        assert weight.max() == pytest.approx(16374.141739)
    else:
        assert weight.mean() == pytest.approx(0.00021882448)
        assert weight.max() == pytest.approx(0.000897441)

# def test_precomputed(data, ids):
#     raise NotImplementedError()


def test_coincident():
    grocs_duplicated = pd.concat(
        [grocs, grocs.iloc[:10], grocs.iloc[:3]], ignore_index=True
    )
    # plain kernel
    head, tail, weight = _kernel(grocs_duplicated)
    assert head.shape[0] == len(grocs_duplicated) * (len(grocs_duplicated) - 1)
    assert tail.shape == head.shape
    assert weight.shape == head.shape
    np.testing.assert_array_equal(pd.unique(head), grocs_duplicated.index)

    # k, raise
    with pytest.raises(ValueError, match="There are"):
        _kernel(grocs_duplicated, k=2)

    # k, jitter
    head, tail, weight = _kernel(
        grocs_duplicated, taper=False, k=2, coincident="jitter"
    )
    assert head.shape[0] == len(grocs_duplicated) * 2
    assert tail.shape == head.shape
    assert weight.shape == head.shape
    np.testing.assert_array_equal(pd.unique(head), grocs_duplicated.index)

    # k, clique
    with pytest.raises(NotImplementedError):
        _kernel(grocs_duplicated, k=2, coincident="clique")


def test_shape_preservation():
    coordinates = np.vstack(
        [np.repeat(np.arange(10), 10), np.tile(np.arange(10), 10)]
    ).T
    head, tail, weight = _kernel(
        coordinates,
        k=3,
        metric="euclidean",
        p=2,
        kernel="boxcar",
        bandwidth=0.5,
        taper=False,
    )
    np.testing.assert_array_equal(head, np.repeat(np.arange(100), 3))
    assert tail.shape == head.shape, "shapes of head and tail do not match"
    np.testing.assert_array_equal(weight, np.zeros((300,), dtype=int))

    head, tail, weight = _kernel(
        coordinates,
        k=3,
        metric="euclidean",
        p=2,
        kernel="boxcar",
        bandwidth=0.5,
        taper=True,
    )
    np.testing.assert_array_equal(head, np.arange(100))
    assert tail.shape == head.shape, "shapes of head and tail do not match"
    np.testing.assert_array_equal(weight, np.zeros((100,), dtype=int))


def test_haversine_check():
    with pytest.raises(ValueError, match="'haversine'"):
        _kernel(grocs, k=2, metric="haversine")


def test_distance_band_colocated():
    coordinates = np.array([[0, 0], [1, 0], [1, 0], [2, 0], [3, 0]])
    dist = _distance_band(coordinates, 1)
    assert dist.shape == (5, 5)
    np.testing.assert_array_equal(
        dist.data,
        np.array(
            [
                0.0,
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
                1.0,
                1.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                0.0,
                1.0,
                1.0,
                0.0,
            ]
        ),
    )

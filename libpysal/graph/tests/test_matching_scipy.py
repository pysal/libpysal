"""Regression tests for matching without the optional PuLP dependency."""

import builtins

import geopandas
import numpy as np
import pytest
from scipy.spatial.distance import cdist

from libpysal.graph import Graph
from libpysal.graph._matching import _spatial_matching


@pytest.mark.parametrize("between", [False, True])
@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("k", [1, 2])
def test_scipy_matches_pulp_objective(between, partial, k):
    pulp = pytest.importorskip("pulp")
    rng = np.random.default_rng(916)
    x = rng.random((6, 2))
    y = rng.random((8, 2)) if between else None
    expected = _spatial_matching(
        x,
        y,
        n_matches=k,
        allow_partial_match=partial,
        solver=pulp.PULP_CBC_CMD(msg=False),
    )
    actual = _spatial_matching(x, y, n_matches=k, allow_partial_match=partial)
    distances = cdist(x, y if between else x)

    def objective(edges):
        heads, tails, weights = edges
        return (distances[heads, tails] * weights).sum()

    assert objective(actual) == pytest.approx(objective(expected), abs=1e-7)
    heads, tails, weights = actual
    degrees = np.bincount(heads, weights=weights, minlength=len(x))
    if partial and not between:
        np.testing.assert_allclose(degrees, k)
    else:
        assert (degrees >= k - 1e-7).all()
    if between:
        assert (np.bincount(tails, weights=weights, minlength=len(y)) <= k + 1e-7).all()
    else:
        adjacency = np.zeros((len(x), len(x)))
        adjacency[heads, tails] = weights
        np.testing.assert_allclose(adjacency, adjacency.T)
        np.testing.assert_array_equal(adjacency.diagonal(), 0)
    assert (weights > 0).all()
    assert (weights <= 1 + 1e-7).all()
    if not partial:
        np.testing.assert_array_equal(weights, 1)


@pytest.mark.parametrize("partial", [False, True])
def test_matching_without_pulp(monkeypatch, partial):
    original_import = builtins.__import__

    def no_pulp(name, *args, **kwargs):
        if name == "pulp":
            raise ImportError("PuLP deliberately unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_pulp)
    points = geopandas.GeoSeries(
        geopandas.points_from_xy([0, 1, 3, 4], [0, 0, 0, 0]),
        index=["west", "west-near", "east", "east-near"],
    )
    graph = Graph.build_spatial_matches(points, k=1, allow_partial_match=partial)
    assert set(graph.unique_ids) == set(points.index)
    assert graph.isolates.empty
    assert graph.asymmetry().empty
    assert (graph.cardinalities == 1).all()
    heads, tails, weights = _spatial_matching(
        points, points.iloc[::-1], n_matches=1, allow_partial_match=partial
    )
    assert set(heads) == set(points.index)
    assert set(tails) == set(points.index)
    np.testing.assert_array_equal(weights, 1)
    with pytest.raises(ImportError, match="return_mip=True.*require"):
        _spatial_matching(points, n_matches=1, return_mip=True)


def test_coincident_bipartite_matches():
    points = np.array([[0, 0], [1, 0], [2, 0]])
    heads, tails, weights = _spatial_matching(points, points[::-1], n_matches=1)
    np.testing.assert_array_equal(heads, [0, 1, 2])
    np.testing.assert_array_equal(tails, [2, 1, 0])
    np.testing.assert_array_equal(weights, 1)


def test_fractional_odd_cycle():
    points = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]])
    heads, tails, weights = _spatial_matching(
        points, n_matches=1, allow_partial_match=True
    )
    assert len(heads) == len(tails) == 6
    np.testing.assert_allclose(weights, 0.5)


def test_precomputed_matches():
    distances = np.array([[0, 4, 5], [4, 0, 1], [5, 1, 0]])
    heads, tails, weights = _spatial_matching(
        distances, y=True, n_matches=1, metric="precomputed"
    )
    np.testing.assert_array_equal(heads, [0, 1, 2])
    np.testing.assert_array_equal(tails, heads)
    np.testing.assert_array_equal(weights, 1)


def test_infeasible_matching():
    pulp = pytest.importorskip("pulp")
    points = np.array([[0, 0], [1, 0]])
    with pytest.warns(UserWarning, match="Problem is Infeasible"):
        expected = _spatial_matching(
            points, n_matches=2, solver=pulp.PULP_CBC_CMD(msg=False)
        )
    with pytest.warns(UserWarning, match="Problem is Infeasible"):
        actual = _spatial_matching(points, n_matches=2)
    for result, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(result, reference)

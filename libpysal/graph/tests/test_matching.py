"""
For completeness, we need to test a shuffled dataframe
(i.e. always send unsorted data) with:
- numeric ids
- string ids
- point dataframe
- coordinates
- check two kernel functions
- numba/nonumba
"""

import geodatasets
import geopandas
import numpy as np
import pytest

from libpysal.graph._matching import _spatial_matching
from libpysal.graph.base import Graph


@pytest.fixture(scope="session")
def stores():
    stores = geopandas.read_file(geodatasets.get_path("geoda liquor_stores")).explode(
        index_parts=False
    )
    return stores


np.random.seed(85711)
simple = np.random.random(size=(5, 2))


pulp = pytest.importorskip("pulp")
pytest.importorskip("sklearn")
default_solver = pulp.listSolvers(onlyAvailable=True)
if len(default_solver) == 0:
    raise Exception("configuration of pulp has failed, no available solvers")
default_solver = getattr(pulp, default_solver[0])()


def test_correctness_k1():
    # manual solution for simple k=1 by hungarian method
    known = np.row_stack([(0, 3), (1, 4), (2, 3), (3, 0), (3, 2), (4, 1)])
    computed = _spatial_matching(simple, n_matches=1, solver=default_solver)
    np.testing.assert_array_equal(known, np.column_stack((computed[0], computed[1])))
    computed_partial = _spatial_matching(
        simple, n_matches=1, allow_partial_match=True, solver=default_solver
    )
    # manual solution by relaxing the above
    known = np.row_stack(
        [
            (0, 2, 0.5),
            (0, 3, 0.5),
            (1, 4, 1),
            (2, 0, 0.5),
            (2, 3, 0.5),
            (3, 0, 0.5),
            (3, 2, 0.5),
            (4, 1, 1),
        ]
    )
    np.testing.assert_array_equal(known, np.column_stack(computed_partial))


@pytest.mark.network
def test_stores(stores):
    computed_heads, computed_tails, computed_weights = _spatial_matching(
        stores.head(101), n_matches=3, solver=default_solver
    )
    computed_heads_p, computed_tails_p, computed_weights_p = _spatial_matching(
        stores.head(101), allow_partial_match=True, n_matches=3, solver=default_solver
    )
    assert (computed_weights == 1).all()
    assert (computed_weights_p < 1).any()
    assert (computed_weights_p >= 0.5).all()

    for heads, tails in (
        (computed_heads, computed_tails),
        (computed_heads_p, computed_tails_p),
    ):
        _, n_by_heads = np.unique(heads, return_counts=True)
        _, n_by_tails = np.unique(tails, return_counts=True)
        assert n_by_heads.max() == 4
        assert n_by_heads.min() == 3
        assert n_by_tails.max() == 4
        assert n_by_tails.min() == 3

    gf = Graph.from_arrays(computed_heads, computed_tails, computed_weights)
    gp = Graph.from_arrays(computed_heads_p, computed_tails_p, computed_weights_p)
    for g in (gf, gp):
        assert g.asymmetry().empty
        assert g.isolates.empty


def test_returns_mip():
    *computed, mip = _spatial_matching(
        simple, n_matches=4, return_mip=True, solver=default_solver
    )
    assert mip.sol_status == 1
    assert mip.objective.value() > 0
    with pytest.warns(UserWarning, match="Problem is Infeasible"):
        *computed, mip = _spatial_matching(
            simple, n_matches=6, return_mip=True, solver=default_solver
        )
    assert mip.sol_status == -1

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
import pandas as pd
from scipy import sparse
import pytest
import shapely
from libpysal.graph._matching import _spatial_matching
from libpysal.graph.base import Graph

stores = geopandas.read_file(geodatasets.get_path("geoda liquor_stores")).explode(
    index_parts=False
)
stores_unique = stores.drop_duplicates(subset="geometry")

np.random.seed(85711)
simple = np.random.random(size=(5, 2))


def test_correctness_k1():
    # manual solution for simple k=1 by hungarian method
    known = numpy.row_stack([(0, 3), (1, 4), (2, 3), (3, 3), (4, 1)])
    computed = _spatial_matching(simple, k=1)
    numpy.testing.assert_array_equal(
        known, numpy.column_stack(computed[0], computed[1])
    )
    computed_partial = _spatial_matching(simple, k=1)
    # manual solution by relaxing the above
    known = numpy.row_stack(
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
    numpy.testing.assert_array_equal(known, numpy.column_stack(computed_partial))


def test_stores():
    computed_heads, computed_tails, computed_weights = _spatial_matching(
        stores.head(101), n_matches=3
    )
    computed_heads_p, computed_tails_p, computed_weights_p = _spatial_matching(
        stores.head(101), allow_partial_match=True, n_matches=3
    )
    assert (computed_weights == 1).all()
    assert (computed_weights_p < 1).any()
    assert (computed_weights_p >= 0.5).all()

    for heads, tails in (
        (computed_heads, computed_tails),
        (computed_heads_p, computed_tails_p),
    ):
        _, n_by_heads = numpy.unique(heads, return_counts=True)
        _, n_by_tails = numpy.unique(tails, return_counts=True)
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
    *computed, mip = _spatial_matching(simple, n_matches=4, return_mip=True)
    assert mip.sol_status == 1
    assert mip.objective.value() > 0
    with pytest.warns(UserWarning, match="Problem is Infeasible"):
        *computed, mip = _spatial_matching(simple, n_matches=6, return_mip=True)
    assert mip.sol_status == -1

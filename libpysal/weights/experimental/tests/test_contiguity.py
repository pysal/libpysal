"""

For completeness, we need to test a shuffled dataframe
(i.e. always send unsorted data) with:
- numeric ids
- string ids
- mixed polygon/multipolygon dataset
- mixed line/multiline dataset
- dataset with islands
"""

import geopandas, geodatasets, pytest, shapely, numpy
from libpysal.weights.experimental._contiguity import (
    _vertex_set_intersection,
    _rook,
    _queen,
)

numpy.random.seed(111211)
rivers = geopandas.read_file(geodatasets.get_path("eea large_rivers")).sample(frac=1, replace=False)
rivers['strID'] = rivers.NAME
rivers['intID'] = rivers.index.values + 2

nybb = geopandas.read_file(geodatasets.get_path("ny bb"))
nybb['strID'] = nybb.BoroName
nybb['intID'] = nybb.BoroCode

parametrize_ids = pytest.mark.parametrize("ids", [None, "strID", "intID"])
parametrize_geoms = pytest.mark.parametrize("geoms", [rivers, nybb], ['rivers', 'nybb'])
parametrize_perim = pytest.mark.parametrize("by_perimeter", [True, False], ids=['binary', 'perimeter'])


@parametrize_ids
def test_vertex_set_intersection_rivers(ids, data=rivers):
    """
    if ids is not None:
        data = data.set_index(ids)
        ids = data.index.values
    else:
        ids = data.index
    # implement known_heads, known_tails
    known_heads = data.index[known_heads]
    known_tails = data.index[known_tails]
    heads, tails, weights = _vertex_set_intersection(data, ids=ids)
    numpy.testing.assert_array_equal(heads, known_heads)
    numpy.testing.assert_array_equal(tails, known_tails)
    """
    
    ...

@parametrize_ids
def test_rook_rivers(ids, data=rivers):
    ...

@parametrize_ids
def test_queen_rivers(ids, data=rivers):
    ...

@pytest.mark.parametrize("rook", [True, False], ids=['Rook', "Queen"])
@parametrize_perim
@parametrize_ids
def test_vertex_set_intersection_nybb(ids, rook, by_perimeter, data=nybb):
    """
    if ids is not None:
        data = data.set_index(ids)
        ids = data.index.values
    else:
        ids = data.index

    # implement known_heads, known_tails
    known_heads = data.index[known_heads]
    known_tails = data.index[known_tails]
    heads, tails, weights = _vertex_set_intersection(data, ids=ids, rook=rook, by_perimeter=by_perimeter)
    """
    ...

@parametrize_perim
@parametrize_ids
def test_rook_nybb(ids, by_perimeter, data=nybb):
    """

    if ids is not None:
        data = data.set_index(ids)
        ids = data.index.values
    else:
        ids = data.index
    # implement known_heads, known_tails
    known_heads = data.index[known_heads]
    known_tails = data.index[known_tails]
    """
    ...

@parametrize_perim
@parametrize_ids
def test_queen_nybb(ids, by_perimeter, data=nybb):
    if ids is not None:
        data = data.set_index(ids)
        ids = data.index.values
    else:
        ids = data.index
    # implement known_heads, known_tails
    known_heads = numpy.array([0, 0, 0, 1, 1, 2, 2, 2, 3, 4, 4])
    known_tails = numpy.array([1, 2, 4, 0, 2, 0, 1, 4, 3, 0, 2])

    known_heads = data.index.values[known_heads]
    known_tails = data.index.values[known_tails]

    heads, tails, weights = _vertex_set_intersection(data, by_perimeter=by_perimeter, ids=ids)
    
    numpy.testing.assert_array_equal(heads, known_heads)
    numpy.testing.assert_array_equal(tails, known_tails)

    if by_perimeter:
        head_geom = data.geometry.loc[known_heads]
        tail_geom = data.geometry.loc[known_tails]
        perims = head_geom.intersection(tail_geom).length
        numpy.testing.assert_allclose(perims, weights)
    else:
        numpy.testing.assert_array_equal(weights, numpy.ones_like(weights))


def test_vertex_set_contiguity_distinct():
    """
    Check to ensure that vertex set ignores rook/queen neighbors that share 
    an edge whose nodes are *not* in the vertex set. The test case is a set
    of offset squares 
    """
    data = geopandas.GeoSeries((shapely.box(0, 0, 1, 1), shapely.box(0.5, 1, 1.5, 2)))

    vs_rook = numpy.column_stack(_vertex_set_intersection(data, rook=True))
    rook = numpy.column_stack(_rook(data))

    with pytest.raises(AssertionError):
        numpy.testing.assert_array_equal(vs_rook, rook)

    vs_queen = numpy.column_stack(_vertex_set_intersection(data, rook=False))
    queen = numpy.column_stack(_queen(data))

    with pytest.raises(AssertionError):
        numpy.testing.assert_array_equal(vs_queen, queen) 
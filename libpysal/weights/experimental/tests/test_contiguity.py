"""

For completeness, we need to test a shuffled dataframe
(i.e. always send unsorted data) with:
- numeric ids
- string ids
- mixed polygon/multipolygon dataset
- mixed line/multiline dataset
- dataset with islands
"""

import pandas, geopandas, geodatasets, pytest, shapely, numpy
from libpysal.weights.experimental._contiguity import (
    _vertex_set_intersection,
    _rook,
    _queen,
)

numpy.random.seed(111211)
rivers = geopandas.read_file(geodatasets.get_path("eea large_rivers")).sample(
    frac=1, replace=False
)
rivers["strID"] = rivers.NAME
rivers["intID"] = rivers.index.values + 2

nybb = geopandas.read_file(geodatasets.get_path("ny bb"))
nybb["strID"] = nybb.BoroName
nybb["intID"] = nybb.BoroCode

parametrize_ids = pytest.mark.parametrize("ids", [None, "strID", "intID"])
parametrize_geoms = pytest.mark.parametrize("geoms", [rivers, nybb], ["rivers", "nybb"])
parametrize_perim = pytest.mark.parametrize(
    "by_perimeter", [False, True], ids=["binary", "perimeter"]
)
parametrize_rook = pytest.mark.parametrize("rook", [True, False], ids=["rook", "queen"])


@parametrize_ids
def test_user_vertex_set_intersection_rivers(ids, data=rivers):
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
def test_user_rook_rivers(ids, data=rivers):
    ...


@parametrize_ids
def test_user_queen_rivers(ids, data=rivers):
    ...


@parametrize_rook
@parametrize_perim
@parametrize_ids
def test_user_vertex_set_intersection_nybb(ids, rook, by_perimeter, data=nybb):
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

@parametrize_rook
@parametrize_perim
@parametrize_ids
def test_user_pointset_nybb(ids, by_perimeter, rook, data=nybb):
    if ids is not None:
        data = data.set_index(ids)
        ids = data.index.values
    else:
        ids = data.index
    # implement known_heads, known_tails
    known_heads = numpy.array([1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 0])
    known_tails = numpy.array([2, 3, 4, 1, 3, 2, 1, 4, 1, 3, 0])

    known_heads = data.index.values[known_heads]
    known_tails = data.index.values[known_tails]

    if by_perimeter:
        head_geom = data.geometry.loc[known_heads].values
        tail_geom = data.geometry.loc[known_tails].values
        known_weights = head_geom.intersection(tail_geom).length
    else:
        known_weights = numpy.ones_like(known_heads)
    known_weights[known_heads == known_tails] = 0

    known_adj = pandas.DataFrame.from_dict(
        dict(focal=known_heads, neighbor=known_tails, weight=known_weights)
    )

    f = _rook if rook else _queen
    heads, tails, weights = f(data, by_perimeter=by_perimeter, ids=ids)

    adj = pandas.DataFrame.from_dict(dict(focal=heads, neighbor=tails, weight=weights))
    # pandas.testing.assert_frame_equal(adj, known_adj, check_like=True)
    pandas.testing.assert_frame_equal(
        adj.sort_values(["focal", "neighbor"]),
        known_adj.sort_values(["focal", "neighbor"]),
        check_like=True,
        check_dtype=False,
    )

@parametrize_rook
def test_correctness_rook_queen_distinct(rook):
    data = geopandas.GeoSeries((shapely.box(0, 0, 1, 1), shapely.box(1,1,2,2)))



def test_correctness_vertex_set_contiguity_distinct():
    """
    Check to ensure that vertex set ignores rook/queen neighbors that share
    an edge whose nodes are *not* in the vertex set. The test case is a set
    of offset squares
    """
    data = geopandas.GeoSeries((shapely.box(0, 0, 1, 1), shapely.box(0.5, 1, 1.5, 2)))

    vs_rook = numpy.column_stack(_vertex_set_intersection(data, rook=True))
    vs_rook = pandas.DataFrame(vs_rook, columns=['focal', 'neighbor', 'weight'])
    
    rook = numpy.column_stack(_rook(data))
    rook = pandas.DataFrame(rook, columns=['focal', 'neighbor', 'weight'])

    with pytest.raises(AssertionError):
        pandas.testing.assert_frame_equal(
                vs_rook,
                rook,
                check_like=True,
                check_dtype=False,
            )

    vs_queen = numpy.column_stack(_vertex_set_intersection(data, rook=False))
    vs_queen = pandas.DataFrame(vs_queen, columns=['focal', 'neighbor', 'weight'])


    queen = numpy.column_stack(_queen(data))
    queen = pandas.DataFrame(queen, columns=['focal', 'neighbor', 'weight'])



    with pytest.raises(AssertionError):
        pandas.testing.assert_frame_equal(
                vs_queen,
                queen,
                check_like=True,
                check_dtype=False,
            )
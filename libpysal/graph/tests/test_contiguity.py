"""

For completeness, we need to test a shuffled dataframe
(i.e. always send unsorted data) with:
- numeric ids
- string ids
- mixed polygon/multipolygon dataset
- mixed line/multiline dataset
- dataset with islands
"""

import geodatasets
import geopandas
import numpy
import pandas
import pytest
import shapely

from libpysal.graph._contiguity import (
    _block_contiguity,
    _queen,
    _rook,
    _vertex_set_intersection,
    _fuzzy_contiguity,
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
parametrize_pointset = pytest.mark.parametrize(
    "pointset", [True, False], ids=["pointset", "vertex intersection"]
)


@parametrize_pointset
@parametrize_rook
@parametrize_ids
def test_user_rivers(ids, rook, pointset, data=rivers):
    """
    Check whether contiguity is constructed correctly for rivers in Europe.
    """
    data = data.reset_index(drop=False).rename(columns={"index": "original_index"})
    ids = "original_index" if ids is None else ids
    data.index = data[ids].values
    ids = data.index.values
    # implement known_heads, known_tails

    if rook:
        known_heads = known_tails = ids[numpy.arange(len(data))]
        known_weights = numpy.zeros_like(known_heads)
    else:
        known_heads = numpy.array(["Sava", "Danube", "Tisa", "Danube"])
        known_tails = numpy.array(["Danube", "Sava", "Danube", "Tisa"])
        isolates = data[~data.strID.isin(known_heads)].index.values

        tmp_ = (
            data.reset_index(drop=False)
            .rename(columns={"index": "tmp_index"})
            .set_index("strID")
        )

        known_heads = tmp_.loc[known_heads, "tmp_index"].values
        known_tails = tmp_.loc[known_tails, "tmp_index"].values

        known_heads = numpy.hstack((known_heads, isolates))
        known_tails = numpy.hstack((known_tails, isolates))

        known_weights = numpy.ones_like(known_heads)
        known_weights[known_heads == known_tails] = 0
    if pointset:
        f = _rook if rook else _queen
        derived = f(data, ids=ids)
        derived_by_index = f(data, ids=None)
    else:
        derived = _vertex_set_intersection(data, ids=ids, rook=rook)
        derived_by_index = _vertex_set_intersection(data, rook=rook, ids=None)

    assert set(zip(*derived)) == set(zip(known_heads, known_tails, known_weights))
    assert set(zip(*derived_by_index)) == set(
        zip(known_heads, known_tails, known_weights)
    )


@parametrize_rook
@parametrize_perim
@parametrize_ids
def test_user_vertex_set_intersection_nybb(ids, rook, by_perimeter):
    """
    check whether vertexset contiguity is constructed correctly
    for nybb
    """
    data = nybb.copy()
    if ids is not None:
        data.index = data[ids].values
    ids = data.index.values

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

    f = _rook if rook else _queen
    derived = f(data, by_perimeter=by_perimeter, ids=ids)
    derived_by_index = f(data, by_perimeter=by_perimeter, ids=None)

    assert set(zip(*derived)) == set(zip(known_heads, known_tails, known_weights))
    assert set(zip(*derived_by_index)) == set(
        zip(known_heads, known_tails, known_weights)
    )


@parametrize_rook
@parametrize_perim
@parametrize_ids
def test_user_pointset_nybb(ids, by_perimeter, rook):
    """
    check whether pointset weights are constructed correctly
    for nybb
    """
    data = nybb.copy()
    if ids is not None:
        data.index = data[ids].values
    ids = data.index.values

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

    f = _rook if rook else _queen
    derived = f(data, by_perimeter=by_perimeter, ids=ids)
    derived_by_index = f(data, by_perimeter=by_perimeter, ids=None)

    assert set(zip(*derived)) == set(zip(known_heads, known_tails, known_weights))
    assert set(zip(*derived_by_index)) == set(
        zip(known_heads, known_tails, known_weights)
    )


@parametrize_pointset
def test_correctness_rook_queen_distinct(pointset):
    """
    Check that queen and rook generate different contiguities in the case of a
    shared point but no edge.
    """
    data = geopandas.GeoSeries((shapely.box(0, 0, 1, 1), shapely.box(1, 1, 2, 2)))
    if pointset:
        rook_ = _rook(data.geometry)
        queen_ = _queen(data.geometry)

    else:
        rook_ = _vertex_set_intersection(data.geometry, rook=True)
        queen_ = _vertex_set_intersection(data.geometry, rook=False)

    assert set(zip(*rook_)) != set(zip(*queen_))


def test_geom_type_raise():
    """
    Check the error for point geoms
    """
    data = geopandas.GeoSeries((shapely.Point(0, 0), shapely.Point(2, 2)))
    with pytest.raises(ValueError, match="This Graph type is only well-defined"):
        _vertex_set_intersection(data)


def test_overlap_raise():
    data = nybb.set_index("BoroName").geometry.copy()
    data.iloc[1] = shapely.union(
        data.iloc[1], shapely.Point(1021176.479, 181374.797).buffer(10000)
    )

    with pytest.raises(ValueError, match="Some geometries overlap."):
        _vertex_set_intersection(data, by_perimeter=True)


def test_correctness_vertex_set_contiguity_distinct():
    """
    Check to ensure that vertex set ignores rook/queen neighbors that share
    an edge whose nodes are *not* in the vertex set. The test case is two
    offset squares
    """
    data = geopandas.GeoSeries((shapely.box(0, 0, 1, 1), shapely.box(0.5, 1, 1.5, 2)))

    vs_rook = _vertex_set_intersection(data, rook=True)

    rook = _rook(data)

    assert set(zip(*vs_rook)) != set(zip(*rook))

    vs_queen = _vertex_set_intersection(data, rook=False)

    queen = _queen(data)

    assert set(zip(*vs_queen)) != set(zip(*queen))


@pytest.mark.parametrize(
    "regimes",
    [
        ["n", "n", "s", "s", "e", "e", "w", "w", "e", "j"],
        [0, 0, 2, 2, 3, 3, 4, 4, 3, 1],
    ],
)
def test_block_contiguity(regimes):
    neighbors = _block_contiguity(regimes)
    wn = {
        0: [1],
        1: [0],
        2: [3],
        3: [2],
        4: [5, 8],
        5: [4, 8],
        6: [7],
        7: [6],
        8: [4, 5],
        9: [],
    }
    assert {f: n.tolist() for f, n, in neighbors.items()} == wn

    ids = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    neighbors = _block_contiguity(regimes, ids=ids)
    wn_str = {ids[f]: [ids[o] for o in n] for f, n in wn.items()}
    assert {f: n.tolist() for f, n, in neighbors.items()} == wn_str

    regimes = pandas.Series(regimes, index=ids)
    neighbors = _block_contiguity(regimes)
    assert {f: n.tolist() for f, n, in neighbors.items()} == wn_str


def test_fuzzy_contiguity():
    # integer
    head, tail, weight = _fuzzy_contiguity(nybb.set_index("intID"), nybb["intID"])
    numpy.testing.assert_array_equal(
        head,
        [5, 4, 4, 4, 3, 3, 1, 1, 1, 2, 2],
    )
    numpy.testing.assert_array_equal(
        tail,
        [5, 3, 1, 2, 4, 1, 4, 3, 2, 4, 1],
    )
    numpy.testing.assert_array_equal(weight, [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    # string
    head, tail, weight = _fuzzy_contiguity(nybb.set_index("strID"), nybb["strID"])
    numpy.testing.assert_array_equal(
        head,
        [
            "Staten Island",
            "Queens",
            "Queens",
            "Queens",
            "Brooklyn",
            "Brooklyn",
            "Manhattan",
            "Manhattan",
            "Manhattan",
            "Bronx",
            "Bronx",
        ],
    )
    numpy.testing.assert_array_equal(
        tail,
        [
            "Staten Island",
            "Brooklyn",
            "Manhattan",
            "Bronx",
            "Queens",
            "Manhattan",
            "Queens",
            "Brooklyn",
            "Bronx",
            "Queens",
            "Manhattan",
        ],
    )
    numpy.testing.assert_array_equal(weight, [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    # tolerance
    head, tail, weight = _fuzzy_contiguity(
        nybb.set_index("intID"), nybb["intID"], tolerance=0.05
    )
    numpy.testing.assert_array_equal(
        head,
        [5, 4, 4, 4, 3, 3, 3, 1, 1, 1, 2, 2],
    )
    numpy.testing.assert_array_equal(
        tail,
        [3, 3, 1, 2, 5, 4, 1, 4, 3, 2, 4, 1],
    )
    numpy.testing.assert_array_equal(weight, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    # buffer
    head, tail, weight = _fuzzy_contiguity(
        nybb.set_index("intID"), nybb["intID"], buffer=5000
    )
    numpy.testing.assert_array_equal(
        head,
        [5, 4, 4, 4, 3, 3, 3, 1, 1, 1, 2, 2],
    )
    numpy.testing.assert_array_equal(
        tail,
        [3, 3, 1, 2, 5, 4, 1, 4, 3, 2, 4, 1],
    )
    numpy.testing.assert_array_equal(weight, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    # predicate
    head, tail, weight = _fuzzy_contiguity(
        nybb.set_index("intID"), nybb["intID"], predicate="within"
    )
    numpy.testing.assert_array_equal(
        head,
        [5, 4, 3, 1, 2],
    )
    numpy.testing.assert_array_equal(
        tail,
        [5, 4, 3, 1, 2],
    )
    numpy.testing.assert_array_equal(weight, [0, 0, 0, 0, 0])

    with pytest.raises(ValueError, match="Only one"):
        _fuzzy_contiguity(
            nybb.set_index("intID"), nybb["intID"], tolerance=0.05, buffer=5000
        )

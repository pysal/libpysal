from collections import defaultdict

import numpy
import pandas
import shapely

from ._utils import _neighbor_dict_to_edges, _validate_geometry_input

_VALID_GEOMETRY_TYPES = ("Polygon", "MultiPolygon", "LineString", "MultiLineString")


def _vertex_set_intersection(geoms, rook=True, ids=None, by_perimeter=False):
    """
    Use a hash map inversion to construct a graph

    Arguments
    ---------
    ...

    """
    _, ids, geoms = _validate_geometry_input(
        geoms, ids=ids, valid_geometry_types=_VALID_GEOMETRY_TYPES
    )

    # initialise the target map
    graph = dict()
    for idx in ids:
        graph[idx] = set([idx])

    # get all of the vertices for the input
    assert (
        ~geoms.geom_type.str.endswith("Point")
    ).any(), "this graph type is only well-defined for line and polygon geometries."

    ## TODO: this induces a "fake" edge between the closing and opening point
    ##       of two multipolygon parts. This should never enter into calculations,
    ##       *unless* two multipolygons share opening and closing points in the
    ##       same order and part order. Still, this should be fixed by ensuring that
    ##       only adjacent points of the same part of the same polygon are used.
    ##       this bug also exists in the existing contiguity builder.

    # geoms = geoms.explode()
    # multipolygon_ixs = geoms.get_level_values(0)
    # ids = ids[multipolygon_ixs]
    # geoms = geoms.geometry
    vertices, offsets = shapely.get_coordinates(geoms.geometry, return_index=True)
    # initialise the hashmap we want to invert
    vert_to_geom = defaultdict(set)

    # populate the hashmap we intend to invert
    if rook:
        for i, vertex in enumerate(vertices[:-1]):
            if offsets[i] != offsets[i + 1]:
                continue
            edge = tuple(sorted([tuple(vertex), tuple(vertices[i + 1])]))
            # edge to {polygons, with, this, edge}
            vert_to_geom[edge].add(offsets[i])
    else:
        for i, vertex in enumerate(vertices):
            # vertex to {polygons, with, this, vertex}
            vert_to_geom[tuple(vertex)].add(offsets[i])

    # invert vert_to_geom
    for nexus in vert_to_geom.values():
        if len(nexus) < 2:
            continue
        nexus_names = {ids[ix] for ix in nexus}
        for geom_ix in nexus:
            gid = ids[geom_ix]
            graph[gid] |= nexus_names
            graph[gid].remove(gid)

    heads, tails, weights = _neighbor_dict_to_edges(graph)

    if by_perimeter:
        weights = _perimeter_weights(geoms, heads, tails)
        weights[heads == tails] = 0
    
    return heads, tails, weights


def _queen(geoms, ids=None, by_perimeter=False):
    """
    Construct queen contiguity using point-set relations. 

    Queen contiguity occurs when two polygons touch at exactly a point. 
    Overlapping polygons will not be considered as neighboring
    under this rule, since contiguity is strictly planar. 

    Parameters
    ----------
    geoms : geopandas.GeoDataFrame, geopandas.GeoSeries, numpy.array
        The container for the geometries to compute contiguity. Regardless of
        the containing type, the geometries within the container must be Polygons
        or MultiPolygons. 
    ids : numpy.ndarray (default: None)
        names to use for indexing the graph constructed from geoms. If None (default), 
        an index is extracted from `geoms`. If `geoms` has no index, a pandas.RangeIndex
        is constructed. 
    by_perimeter : bool (default: False)
        whether to compute perimeter-weighted contiguity. By default, this returns
        the raw length of perimeter overlap betwen contiguous polygons or lines. 
        In the case of LineString/MultiLineString input geoms, this is likely to 
        result in empty weights, where all observations are isolates. 

    Returns
    -------
    (heads, tails, weights) : three vectors describing the links in the
        queen contiguity graph, with islands represented as a self-loop with
        zero weight. 
    """
    _, ids, geoms = _validate_geometry_input(
        geoms, ids=ids, valid_geometry_types=_VALID_GEOMETRY_TYPES
    )
    heads_ix, tails_ix = shapely.STRtree(geoms).query(geoms, predicate="touches")
    if by_perimeter:
        weights = _perimeter_weights(geoms, heads_ix, tails_ix)
    else:
        weights = numpy.ones_like(heads_ix)
    heads, tails = ids[heads_ix], ids[tails_ix]
    return _resolve_islands(heads, tails, ids, weights=weights)


def _rook(geoms, ids=None, by_perimeter=False):
    _, ids, geoms = _validate_geometry_input(
        geoms, ids=ids, valid_geometry_types=_VALID_GEOMETRY_TYPES
    )
    heads_ix, tails_ix = shapely.STRtree(geoms).query(geoms)
    mask = shapely.relate_pattern(
        geoms.values[heads_ix], geoms.values[tails_ix], "F***1****"
    )
    if by_perimeter:
        weights = _perimeter_weights(geoms, heads_ix[mask], tails_ix[mask])
    heads, tails = ids[heads_ix][mask], ids[tails_ix][mask]
    if not by_perimeter:
        weights = numpy.ones_like(heads)

    return _resolve_islands(heads, tails, ids, weights)

_rook.__doc__ = _queen.__doc__.replace("queen", "rook").replace("Queen", "Rook")

def _perimeter_weights(geoms, heads, tails):
    """
    Compute the perimeter of neighbor pairs for edges describing a contiguity graph. 

    Note that this result will be incorrect if the head and tail polygon overlap. 
    If they do overlap, it is an "invalid" contiguity, so the length of the
    perimeter of the intersection may not express the correct value for relatedness
    in the contiguity graph. 
    
    The check to ensure that the intersection(head,tail) is 1 dimensional is
    expensive, so is omitted. This is a private method, so strict conditions
    on input data are expected.
    """
    lengths = shapely.length(
        shapely.intersection(geoms.values[heads], geoms.values[tails])
    )
    return lengths


def _resolve_islands(heads, tails, ids, weights):
    """
    Induce self-loops for a collection of ids and links describing a 
    contiguity graph. Induced self-loops will have zero weight. 
    """
    islands = numpy.setdiff1d(ids, heads)
    if islands.shape != (0,):
        heads = numpy.hstack((heads, islands))
        tails = numpy.hstack((tails, islands))
        weights = numpy.hstack((weights, numpy.zeros_like(islands)))
    return heads, tails, weights

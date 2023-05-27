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
    _, ids, geoms = _validate_geometry_input(
        geoms, ids=ids, valid_geometry_types=_VALID_GEOMETRY_TYPES
    )
    heads_ix, tails_ix = shapely.STRtree(geoms).query(geoms, predicate="touches")
    heads, tails, weights = ids[heads_ix], ids[tails_ix], numpy.ones_like(heads_ix)
    if by_perimeter:
        weights = _perimeter_weights(geoms, heads, tails)
    return _resolve_islands(heads, tails, ids, weights=weights)


def _rook(geoms, ids=None, by_perimeter=False):
    _, ids, geoms = _validate_geometry_input(
        geoms, ids=ids, valid_geometry_types=_VALID_GEOMETRY_TYPES
    )
    heads_ix, tails_ix = shapely.STRtree(geoms).query(geoms)
    geoms = numpy.asarray(geoms)
    mask = shapely.relate_pattern(geoms[heads_ix], geoms[tails_ix], "F***1****")
    heads, tails, weights = ids[heads_ix[mask]], ids[tails_ix[mask]], numpy.ones_like(heads_ix[mask])
    if by_perimeter:
        weights = _perimeter_weights(geoms, heads, tails)
    return _resolve_islands(heads, tails, ids, weights)


def _perimeter_weights(geoms, heads, tails):
    lengths = shapely.intersection(geoms[heads].values, geoms[tails]).length
    return lengths.values

def _resolve_islands(heads, tails, ids, weights):
    islands = numpy.setdiff1d(ids, heads)
    if islands.shape != (0,):
        heads = numpy.hstack(heads, islands)
        tails = numpy.hstack(tails, islands)
        weights = numpy.hstack(weights, numpy.zeros_like(islands))
    return heads, tails, weights


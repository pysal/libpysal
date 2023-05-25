import shapely, numpy
from collections import defaultdict
from .base import W


def vertex_set_intersection(geoms, by_edge=False, ids=None):
    """
    Use a hash map inversion to construct a graph

    Arguments
    ---------
    ...

    """
    if ids is None:
        ids = getattr(geoms, "index", numpy.arange(len(geoms)))

    # initialise the target map
    graph = defaultdict(set)

    # get all of the vertices for the input
    vertices, offsets = shapely.get_coordinates(geoms, return_index=True)
    # initialise the hashmap we want to invert
    vert_to_geom = defaultdict(set)

    # populate the hashmap we intend to invert
    if by_edge:
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
    for nexus in vert_to_geom:
        if len(nexus) < 2:
            continue
        nexus_names = {ids[ix] for ix in nexus}
        for geom_ix in nexus:
            graph[ids[geom_ix]] |= nexus_names
    return graph
    return W.from_dict(graph)
    # pandas.MultiIndex.from_arrays(pandas.Series(newr).explode().reset_index().values)


def queen(geoms, ids=None):
    if ids is None:
        try:
            ids = geoms.index
        except:
            ids = numpy.arange(len(geoms))
    head, tail = shapely.STRtree(geoms).query(geoms, predicate="touches")
    return head, tail, numpy.ones_like(head)
    return W.from_index(pandas.MultiIndex.from_arrays((head, tail)))


def rook(geoms, ids=None):
    if ids is None:
        try:
            ids = geoms.index
        except:
            ids = numpy.arange(len(geoms))
    head, tail = shapely.STRtree(geoms).query(geoms)
    geoms = numpy.asarray(geoms)
    mask = shapely.relate_pattern(geoms[head], geoms[tail], "F***1****")
    return head[mask], tail[mask], numpy.ones_like(head)
    return W.from_index(pandas.MultiIndexgeoms.from_arrays((head[mask], tail[mask])))

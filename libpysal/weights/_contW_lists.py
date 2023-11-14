# ruff: noqa: N999

import collections
import contextlib
import itertools as it

from ..cg.shapes import Chain, Polygon

QUEEN = 1
ROOK = 2

__author__ = "Jay Laura jlaura@asu.edu"


def _get_verts(shape):
    if isinstance(shape, Polygon | Chain):
        return shape.vertices
    else:
        return _get_boundary_points(shape)


def _get_boundary_points(shape):
    """
    Recursively handle polygons vs. multipolygons to
    extract the boundary point set from each.
    """
    if shape.geom_type.lower() == "polygon":
        shape = shape.boundary
        return _get_boundary_points(shape)
    elif shape.geom_type.lower() == "linestring":
        return list(map(tuple, list(zip(*shape.coords.xy, strict=True))))
    elif shape.geom_type.lower() == "multilinestring":
        return list(
            it.chain(
                *(list(zip(*shape.coords.xy, strict=True)) for shape in shape.geoms)
            )
        )
    elif shape.geom_type.lower() == "multipolygon":
        return list(
            it.chain(*(_get_boundary_points(part.boundary) for part in shape.geoms))
        )
    else:
        raise TypeError(
            "Input shape must be a Polygon, Multipolygon, LineString, "
            f" or MultiLinestring and was instead: {shape.type}"
        )


class ContiguityWeightsLists:
    """
    Contiguity for a collection of polygons using high performance
    list, set, and dict containers
    """

    def __init__(self, collection, wttype=1):
        """
        Parameters
        ----------

        collection: PySAL PolygonCollection

        wttype: int
                1: Queen
                2: Rook
        """
        self.collection = list(collection)
        self.wttype = wttype
        self.jcontiguity()

    def jcontiguity(self):
        num_poly = len(self.collection)

        w = {}
        for i in range(num_poly):
            w[i] = set()

        geoms = []
        offsets = []
        c = 0  # PolyID Counter

        if self.wttype == QUEEN:
            for n in range(num_poly):
                verts = _get_verts(self.collection[n])
                offsets += [c] * len(verts)
                geoms += verts
                c += 1

            items = collections.defaultdict(set)
            for i, vertex in enumerate(geoms):
                items[vertex].add(offsets[i])

            shared_vertices = []
            for _, location in list(items.items()):
                if len(location) > 1:
                    shared_vertices.append(location)

            for vert_set in shared_vertices:
                for v in vert_set:
                    w[v] = w[v] | vert_set
                    with contextlib.suppress(Exception):
                        w[v].remove(v)

        elif self.wttype == ROOK:
            for n in range(num_poly):
                verts = _get_verts(self.collection[n])
                for v in range(len(verts) - 1):
                    geoms.append(tuple(sorted([verts[v], verts[v + 1]])))
                offsets += [c] * (len(verts) - 1)
                c += 1

            items = collections.defaultdict(set)
            for i, item in enumerate(geoms):
                items[item].add(offsets[i])

            shared_vertices = []
            for _, location in list(items.items()):
                if len(location) > 1:
                    shared_vertices.append(location)

            for vert_set in shared_vertices:
                for v in vert_set:
                    w[v] = w[v] | vert_set
                    with contextlib.suppress(Exception):
                        w[v].remove(v)

        else:
            raise Exception(f"Weight type {self.wttype} Not Understood!")
        self.w = w

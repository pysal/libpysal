import warnings
from functools import wraps

import numpy
import pandas
from scipy import sparse, spatial

from libpysal.cg import voronoi_frames

from ._contiguity import _vertex_set_intersection
from ._kernel import _kernel, _kernel_functions, _optimize_bandwidth
from ._utils import (
    CoplanarError,
    _induce_cliques,
    _jitter_geoms,
    _reorder_adjtable_by_ids,
    _validate_geometry_input,
    _vec_euclidean_distances,
)

try:
    from numba import njit  # noqa: E401

    HAS_NUMBA = True
except ModuleNotFoundError:
    from libpysal.common import jit as njit

    HAS_NUMBA = False


_VALID_GEOMETRY_TYPES = ["Point", "Polygon", "MultiPolygon"]

__author__ = """
Levi John Wolf (levi.john.wolf@gmail.com)
Martin Fleischmann (martin@martinfleischmann.net)
Serge Rey (sjsrey@gmail.com)
"""


def _validate_coplanar(triangulator):
    """This is a decorator that validates input for coplanar points"""

    @wraps(triangulator)
    def tri_with_validation(
        coordinates,
        ids=None,
        coplanar="raise",
        kernel=None,
        bandwidth=None,
        seed=None,
        decay=False,
        taper=False,
        **kwargs,
    ):
        if coplanar not in ["raise", "jitter", "clique"]:
            raise ValueError(
                f"Recieved option coplanar='{coplanar}', but only options "
                "'raise','clique','jitter' are suppported."
            )

        # validate geometry input
        coordinates, ids, _ = _validate_geometry_input(
            coordinates,
            ids=ids,
            valid_geometry_types=_VALID_GEOMETRY_TYPES,
        )
        if coplanar == "jitter":
            coordinates = _jitter_geoms(coordinates, seed=seed)

        # generate triangulation (triangulator is the wrapped function)
        # Pass seed to triangulator only if it expects it; otherwise keep in kwargs
        heads_ix, tails_ix, coplanar_loopkup = triangulator(
            coordinates, coplanar=coplanar, seed=seed, **kwargs
        )

        # process weights
        if kernel is None:
            weights = numpy.ones(heads_ix.shape, dtype=numpy.int8)
        else:
            distances = _vec_euclidean_distances(
                coordinates[heads_ix], coordinates[tails_ix]
            ).squeeze()
            sparse_d = sparse.csc_array((distances, (heads_ix, tails_ix)))
            if bandwidth == "auto":
                bandwidth = _optimize_bandwidth(sparse_d, kernel)
            heads_ix, tails_ix, weights = _kernel(
                sparse_d,
                metric="precomputed",
                kernel=kernel,
                bandwidth=bandwidth,
                taper=taper,
                resolve_isolates=False,  # no isolates in triangulation
                decay=decay,
            )
        # create adjacency
        adjtable = pandas.DataFrame.from_dict(
            {"focal": heads_ix, "neighbor": tails_ix, "weight": weights}
        )

        # reinsert points resolved via clique
        if (coplanar_loopkup.shape[0] > 0) & (coplanar == "clique"):
            if kernel is None:
                fill_value = 1
            else:
                fill_value = _kernel_functions[kernel](
                    numpy.array([0]), bandwidth
                ).item()
            coplanar_val, _, nearest = coplanar_loopkup.T
            adjtable = _induce_cliques(adjtable, coplanar_val, nearest, fill_value)
            adjtable["focal"] = ids[adjtable.focal]
            adjtable["neighbor"] = ids[adjtable.neighbor]

            adjtable = _reorder_adjtable_by_ids(adjtable, ids)
        else:
            adjtable["focal"] = ids[adjtable.focal]
            adjtable["neighbor"] = ids[adjtable.neighbor]

        return (
            adjtable.focal.values,
            adjtable.neighbor.values,
            adjtable.weight.values,
        )

    return tri_with_validation


@_validate_coplanar
def _delaunay(coordinates, coplanar="raise", **kwargs):
    if not HAS_NUMBA:
        warnings.warn(
            "The numba package is used extensively in this module...",
            stacklevel=3,
        )
    edges, _, coplanar_out = _voronoi_edges(coordinates, coplanar)
    heads_ix, tails_ix = edges.T

    return heads_ix, tails_ix, coplanar_out


@_validate_coplanar
def _gabriel(coordinates, coplanar="raise", **kwargs):
    if not HAS_NUMBA:
        warnings.warn(
            "The numba package is used extensively in this module...",
            stacklevel=3,
        )

    edges, points, coplanar_out = _voronoi_edges(coordinates, coplanar)
    droplist = _filter_gabriel(edges, points)
    edges = numpy.vstack(list(set(map(tuple, edges)).difference(set(droplist))))
    heads_ix, tails_ix = edges.T
    order = numpy.lexsort((tails_ix, heads_ix))
    return heads_ix[order], tails_ix[order], coplanar_out


@_validate_coplanar
def _relative_neighborhood(coordinates, coplanar="raise", **kwargs):
    if not HAS_NUMBA:
        warnings.warn(
            "The numba package is used extensively in this module...",
            stacklevel=3,
        )

    edges, points, coplanar_out = _voronoi_edges(coordinates, coplanar)
    output, _ = _filter_relativehood(edges, points, return_dkmax=False)

    heads_ix, tails_ix, _ = zip(*output, strict=True)
    return numpy.asarray(heads_ix), numpy.asarray(tails_ix), coplanar_out


def _voronoi(
    coordinates, coplanar="raise", ids=None, clip="bounding_box", rook=True, **kwargs
):
    # 1. Handle IDs
    if ids is None:
        ids = coordinates.index.values if hasattr(coordinates, "index") else numpy.arange(len(coordinates))

    # 2. Logic for Point-only checks (Fixing the Axis Error)
    if coplanar == "raise" and hasattr(coordinates, "geom_type"):
        if (coordinates.geom_type == "Point").all():
            coords_pts = numpy.array(coordinates.tolist()) if coordinates.dtype == object else coordinates
            unique = numpy.unique(coords_pts, axis=0)
            if unique.shape != coords_pts.shape:
                raise CoplanarError("Duplicate points detected.")

    # 3. Handle 'seed' (Fixing the Unexpected Keyword Error)
    kwargs.pop("seed", None)

    # 4. Calculation
    cells = voronoi_frames(coordinates, clip=clip, return_input=False, as_gdf=False, **kwargs)
    h, t, _ = _vertex_set_intersection(cells, rook=rook)

    # 5. Filter Ghost Indices
    h_arr, t_arr = numpy.array(h), numpy.array(t)
    n = len(coordinates)
    mask = (h_arr < n) & (t_arr < n)

    return ids[h_arr[mask]], ids[t_arr[mask]], numpy.ones(mask.sum())

@njit
def _edges_from_simplices(simplices):
    """
    Construct the sets of links that correspond to the edges of each simplex.
    """
    edges = []
    for simplex in simplices:
        edges.append((simplex[0], simplex[1]))
        edges.append((simplex[1], simplex[0]))
        edges.append((simplex[1], simplex[2]))
        edges.append((simplex[2], simplex[1]))
        edges.append((simplex[2], simplex[0]))
        edges.append((simplex[0], simplex[2]))

    return numpy.asarray(edges)


@njit
def _filter_gabriel(edges, coordinates):
    """
    For an input set of edges and coordinates, filter the input edges
    depending on the Gabriel rule.
    """
    edge_pointer = 0
    n_edges = len(edges)
    to_drop = []
    while edge_pointer < n_edges:
        edge = edges[edge_pointer]
        cardinality = 0
        # look ahead to find all neighbors of edge[0]
        for joff in range(edge_pointer, n_edges):
            next_edge = edges[joff]
            if next_edge[0] != edge[0]:
                break
            cardinality += 1
        for ix in range(edge_pointer, edge_pointer + cardinality):
            i, j = edges[ix]  # lookahead ensures that i is always edge[0]
            dij2 = ((coordinates[i] - coordinates[j]) ** 2).sum()
            for ix2 in range(edge_pointer, edge_pointer + cardinality):
                _, k = edges[ix2]
                if j == k:
                    continue
                dik2 = ((coordinates[i] - coordinates[k]) ** 2).sum()
                djk2 = ((coordinates[j] - coordinates[k]) ** 2).sum()

                if dij2 > (dik2 + djk2):
                    to_drop.append((i, j))
                    to_drop.append((j, i))
        edge_pointer += cardinality
    return set(to_drop)


@njit
def _filter_relativehood(edges, coordinates, return_dkmax=False):
    """
    This is a direct reimplementation of the algorithm from Toussaint (1980), RNG-2
    """
    n_coordinates = coordinates.shape[0]
    out = []
    r = []
    for edge in edges:
        i, j = edge
        pi = coordinates[i]
        pj = coordinates[j]
        dij = ((pi - pj) ** 2).sum() ** 0.5
        prune = False
        for k in range(n_coordinates):
            if (i == k) or (j == k):
                continue
            pk = coordinates[k]
            if (pi == pk).all() or (pj == pk).all():  # coplanar
                continue
            dik = ((pi - pk) ** 2).sum() ** 0.5
            djk = ((pj - pk) ** 2).sum() ** 0.5
            dkmax = numpy.array([dik, djk]).max()
            prune |= dkmax <= dij
        if prune:
            pass
        else:
            out.append((i, j, dij))
        if return_dkmax:
            r.append(dkmax)
    return out, r


def _voronoi_edges(coordinates, coplanar):
    dt = spatial.Delaunay(coordinates)

    if dt.coplanar.shape[0] > 0 and coplanar == "raise":
        raise CoplanarError(
            f"There are {len(coordinates) - len(dt.coplanar)} unique locations in "
            f"the dataset, but {len(coordinates)} observations. This means there "
            "are multiple points in the same location, which is undefined "
            "for this graph type. To address this issue, consider setting "
            "`coplanar='clique'` or consult the documentation about "
            "coplanar points."
        )

    edges = _edges_from_simplices(dt.simplices)
    edges = (
        pandas.DataFrame(numpy.asarray(list(edges)))
        .sort_values([0, 1])
        .drop_duplicates()
        .values
    )
    return edges, dt.points, dt.coplanar
import warnings
from functools import wraps

import numpy
import pandas
from scipy import sparse, spatial
from packaging.version import Version

from libpysal.cg import voronoi_frames

from ._contiguity import _vertex_set_intersection
from ._kernel import _kernel, _kernel_functions, _optimize_bandwidth
from ._utils import (
    _build_coincidence_lookup,
    _induce_cliques,
    _jitter_geoms,
    _validate_geometry_input,
    _vec_euclidean_distances,
)

try:
    from numba import njit  # noqa E401

    HAS_NUMBA = True
except ModuleNotFoundError:
    from libpysal.common import jit as njit

    HAS_NUMBA = False

PANDAS_GE_21 = Version(pandas.__version__) >= Version("2.1.0")

_VALID_GEOMETRY_TYPES = ["Point"]

__author__ = """"
Levi John Wolf (levi.john.wolf@gmail.com)
Martin Fleischmann (martin@martinfleischmann.net)
Serge Rey (sjsrey@gmail.com)
"""


# This is in the module, rather than in `utils`, to ensure that it
# can access `_VALID_GEOMETRY_TYPES` without defining a nested decorator.
def _validate_coincident(triangulator):
    """This is a decorator that validates input for coincident points"""

    @wraps(triangulator)
    def tri_with_validation(
        coordinates,
        ids=None,
        coincident="raise",
        kernel=None,
        bandwidth=None,
        seed=None,
        **kwargs,
    ):
        # validate geometry input
        coordinates, ids, geoms = _validate_geometry_input(
            coordinates, ids=ids, valid_geometry_types=_VALID_GEOMETRY_TYPES
        )

        # check for coincident points
        n_coincident, coincident_lut = _build_coincidence_lookup(geoms)

        # resolve coincident points prior triangulation
        if n_coincident > 0:
            if coincident == "raise":
                raise ValueError(
                    f"There are {len(coincident_lut)} "
                    f"unique locations in the dataset, but {len(geoms)} observations. "
                    "This means there are multiple points in the same location, which "
                    "is undefined for this graph type. To address this issue, consider "
                    "setting `coincident='clique' or consult the documentation about "
                    "coincident points."
                )
            elif coincident == "jitter":
                coordinates, geoms = _jitter_geoms(coordinates, geoms, seed=seed)
            elif coincident == "clique":
                raise NotImplementedError(
                    "clique-based resolver of coincident points is not yet implemented."
                )
            else:
                raise ValueError(
                    f"Recieved option coincident='{coincident}', but only options "
                    "'raise','clique','jitter' are suppported."
                )

        # generate triangulation (triangulator is the wrapped function)
        heads_ix, tails_ix = triangulator(coordinates, **kwargs)

        # map ids
        heads, tails = ids[heads_ix], ids[tails_ix]

        # process weights
        if kernel is None:
            weights = numpy.ones(heads_ix.shape, dtype=numpy.int8)
        else:
            distances = _vec_euclidean_distances(
                coordinates[heads_ix], coordinates[tails_ix]
            ).squeeze()
            sparse_D = sparse.csc_array((distances, (heads_ix, tails_ix)))
            if bandwidth == "auto":
                bandwidth = _optimize_bandwidth(sparse_D, kernel)
            _, _, weights = _kernel(
                sparse_D,
                metric="precomputed",
                kernel=kernel,
                bandwidth=bandwidth,
                taper=False,
            )
        # create adjacency
        adjtable = pandas.DataFrame.from_dict(
            dict(focal=heads, neighbor=tails, weight=weights)
        )

        # TODO: fix this
        # reinsert points resolved via clique
        if (n_coincident > 0) & (coincident == "clique"):
            # note that the kernel is only used to compute a fill value for the clique.
            # in the case of the voronoi weights. Using boxcar with an infinite bandwidth
            # also gives us the correct fill value for the voronoi weight: 1.
            fill_value = _kernel_functions[kernel](numpy.array([0]), bandwidth).item()
            adjtable = _induce_cliques(adjtable, coincident_lut, fill_value=fill_value)

        if PANDAS_GE_21:
            # ensure proper sorting
            sorted_index = (
                adjtable[["focal", "neighbor"]]
                .map(list(ids).index)
                .sort_values(["focal", "neighbor"])
                .index
            )
        else:
            # ensure proper sorting
            sorted_index = (
                adjtable[["focal", "neighbor"]]
                .applymap(list(ids).index)
                .sort_values(["focal", "neighbor"])
                .index
            )

        # return data for Graph.from_arrays
        return heads[sorted_index], tails[sorted_index], weights[sorted_index]

    return tri_with_validation


@_validate_coincident
def _delaunay(coordinates):
    """
    Constructor of the Delaunay graph of a set of input points.
    Relies on scipy.spatial.Delaunay and numba to quickly construct
    a graph from the input set of points. Will be slower without numba,
    and will warn if this is missing.

    Parameters
    ----------
    coordinates :  numpy.ndarray, geopandas.GeoSeries, geopandas.GeoDataFrame
        geometries containing locations to compute the delaunay triangulation.  If
        a geopandas object with Point geoemtry is provided, the .geometry attribute
        is used. If a numpy.ndarray with shapely geoemtry is used, then the
        coordinates are extracted and used.  If a numpy.ndarray of a shape (2,n) is
        used, it is assumed to contain x, y coordinates.
    ids : numpy.narray (default: None)
        ids to use for each sample in coordinates. Generally, construction functions
        that are accessed via Graph.build_kernel() will set this automatically from
        the index of the input. Do not use this argument directly unless you intend
        to set the indices separately from your input data. Otherwise, use
        data.set_index(ids) to ensure ordering is respected. If None, then the index
        from the input coordinates will be used.
    bandwidth : float (default: None)
        distance to use in the kernel computation. Should be on the same scale as
        the input coordinates.
    kernel : string or callable
        kernel function to use in order to weight the output graph. See the kernel()
        function for more details.

    Notes
    -----
    The Delaunay triangulation can result in quite a few non-local links among
    spatial coordinates. For a more useful graph, consider the weights.Voronoi
    constructor or the Gabriel graph.

    The weights.Voronoi class builds a voronoi diagram among the points, clips the
    Voronoi cells, and then constructs an adjacency graph among the clipped cells.
    This graph among the clipped Voronoi cells generally represents the structure
    of local adjacencies better than the "raw" Delaunay graph.

    The weights.gabriel.Gabriel graph constructs a Delaunay graph, but only
    includes the "short" links in the Delaunay graph.

    However, if the unresricted Delaunay triangulation is needed, this class
    will compute it much more quickly than Voronoi(coordinates, clip=None).
    """

    if not HAS_NUMBA:
        warnings.warn(
            "The numba package is used extensively in this module"
            " to accelerate the computation of graphs. Without numba,"
            " these computations may become unduly slow on large data.",
            stacklevel=3,
        )

    edges, _ = _voronoi_edges(coordinates)
    heads_ix, tails_ix = edges.T

    return heads_ix, tails_ix


@_validate_coincident
def _gabriel(coordinates):
    """
    Constructs the Gabriel graph of a set of points. This graph is a subset of
    the Delaunay triangulation where only "short" links are retained. This
    function is also accelerated using numba, and implemented on top of the
    scipy.spatial.Delaunay class.

    For a link (i,j) connecting node i to j in the Delaunay triangulation
    to be retained in the Gabriel graph, it must pass a point set exclusion test:

    1. Construct the circle C_ij containing link (i,j) as its diameter
    2. If any other node k is contained within C_ij, then remove link (i,j)
       from the graph.
    3. Once all links are evaluated, the remaining graph is the Gabriel graph.

    Parameters
    ----------
    coordinates :  numpy.ndarray, geopandas.GeoSeries, geopandas.GeoDataFrame
        geometries containing locations to compute the delaunay triangulation.  If
        a geopandas object with Point geoemtry is provided, the .geometry attribute
        is used. If a numpy.ndarray with shapely geoemtry is used, then the
        coordinates are extracted and used.  If a numpy.ndarray of a shape (2,n) is
        used, it is assumed to contain x, y coordinates.
    ids : numpy.narray (default: None)
        ids to use for each sample in coordinates. Generally, construction functions
        that are accessed via Graph.build_kernel() will set this automatically from
        the index of the input. Do not use this argument directly unless you intend
        to set the indices separately from your input data. Otherwise, use
        data.set_index(ids) to ensure ordering is respected. If None, then the index
        from the input coordinates will be used.
    bandwidth : float (default: None)
        distance to use in the kernel computation. Should be on the same scale as
        the input coordinates.
    kernel : string or callable
        kernel function to use in order to weight the output graph. See the kernel()
        function for more details.
    """
    if not HAS_NUMBA:
        warnings.warn(
            "The numba package is used extensively in this module"
            " to accelerate the computation of graphs. Without numba,"
            " these computations may become unduly slow on large data.",
            stacklevel=3,
        )

    edges, dt = _voronoi_edges(coordinates)
    droplist = _filter_gabriel(
        edges,
        dt.points,
    )
    heads_ix, tails_ix = numpy.row_stack(
        list(set(map(tuple, edges)).difference(set(droplist)))
    ).T

    return heads_ix, tails_ix


@_validate_coincident
def _relative_neighborhood(coordinates):
    """
    Constructs the Relative Neighborhood graph from a set of points.
    This graph is a subset of the Delaunay triangulation, where only
    "relative neighbors" are retained. Further, it is a superset of
    the Minimum Spanning Tree, with additional "relative neighbors"
    introduced.

    A relative neighbor pair of points i,j must be closer than the
    maximum distance between i (or j) and each other point k.
    This means that the points are at least as close to one another
    as they are to any other point.

    Parameters
    ----------
    coordinates :  numpy.ndarray, geopandas.GeoSeries, geopandas.GeoDataFrame
        geometries containing locations to compute the delaunay triangulation.  If
        a geopandas object with Point geoemtry is provided, the .geometry attribute
        is used. If a numpy.ndarray with shapely geoemtry is used, then the
        coordinates are extracted and used.  If a numpy.ndarray of a shape (2,n) is
        used, it is assumed to contain x, y coordinates.
    ids : numpy.narray (default: None)
        ids to use for each sample in coordinates. Generally, construction functions
        that are accessed via Graph.build_kernel() will set this automatically from
        the index of the input. Do not use this argument directly unless you intend
        to set the indices separately from your input data. Otherwise, use
        data.set_index(ids) to ensure ordering is respected. If None, then the index
        from the input coordinates will be used.
    bandwidth : float (default: None)
        distance to use in the kernel computation. Should be on the same scale as
        the input coordinates.
    kernel : string or callable
        kernel function to use in order to weight the output graph. See the kernel()
        function for more details.
    """
    if not HAS_NUMBA:
        warnings.warn(
            "The numba package is used extensively in this module"
            " to accelerate the computation of graphs. Without numba,"
            " these computations may become unduly slow on large data.",
            stacklevel=3,
        )

    edges, dt = _voronoi_edges(coordinates)
    output, _ = _filter_relativehood(edges, dt.points, return_dkmax=False)

    heads_ix, tails_ix, distance = zip(*output)
    heads_ix, tails_ix = numpy.asarray(heads_ix), numpy.asarray(tails_ix)

    return heads_ix, tails_ix


@_validate_coincident
def _voronoi(coordinates, clip="extent", rook=True):
    """
    Compute contiguity weights according to a clipped
    Voronoi diagram.

    Parameters
    ---------
    coordinates :  numpy.ndarray, geopandas.GeoSeries, geopandas.GeoDataFrame
        geometries containing locations to compute the delaunay triangulation.  If
        a geopandas object with Point geoemtry is provided, the .geometry attribute
        is used. If a numpy.ndarray with shapely geoemtry is used, then the
        coordinates are extracted and used.  If a numpy.ndarray of a shape (2,n) is
        used, it is assumed to contain x, y coordinates.
    ids : numpy.narray (default: None)
        ids to use for each sample in coordinates. Generally, construction functions
        that are accessed via Graph.build_kernel() will set this automatically from
        the index of the input. Do not use this argument directly unless you intend
        to set the indices separately from your input data. Otherwise, use
        data.set_index(ids) to ensure ordering is respected. If None, then the index
    clip : str (default: 'bbox')
        An overloaded option about how to clip the voronoi cells passed to
        ``libpysal.cg.voronoi_frames()``.
        Default is ``'extent'``. Options are as follows.

        * ``'none'``/``None`` -- No clip is applied. Voronoi cells may be arbitrarily
            larger that the source map. Note that this may lead to cells that are many
            orders of magnitude larger in extent than the original map. Not recommended.
        * ``'bbox'``/``'extent'``/``'bounding box'`` -- Clip the voronoi cells to the
            bounding box of the input points.
        * ``'chull``/``'convex hull'`` -- Clip the voronoi cells to the convex hull of
            the input points.
        * ``'ashape'``/``'ahull'`` -- Clip the voronoi cells to the tightest hull that
            contains all points (e.g. the smallest alphashape, using
            ``libpysal.cg.alpha_shape_auto``).
        * Polygon -- Clip to an arbitrary Polygon.
    rook : bool, optional
        Contiguity method. If True, two geometries are considered neighbours if they
        share at least one edge. If False, two geometries are considered neighbours
        if they share at least one vertex. By default True.

    Notes
    -----
    In theory, the rook contiguity graph for a Voronoi diagram
    is the delaunay triangulation of the generators of the
    voronoi diagram. Yet, this is *not* the case when voronoi
    cells are clipped to an arbitrary shape, including the
    original bounding box of the input points or anything tighter.
    This can arbitrarily delete links present in the delaunay.
    However, clipped voronoi weights make sense over pure
    delaunay triangulations in many applied contexts and
    generally will remove "long" links in the delaunay graph.
    """
    cells, _ = voronoi_frames(coordinates, clip=clip)
    heads_ix, tails_ix, weights = _vertex_set_intersection(cells, rook=rook)

    return heads_ix, tails_ix


#### utilities


@njit
def _edges_from_simplices(simplices):
    """
    Construct the sets of links that correspond to the edges of each
    simplex. Each simplex has three "sides," and thus six undirected
    edges. Thus, the input should be a list of three-length tuples,
    that are then converted into the six non-directed edges for
    each simplex.
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
    depending on the Gabriel rule:

    For each simplex, let i,j be the diameter of the circle defined by
    edge (i,j), and let k be the third point defining the simplex. The
    limiting case for the Gabriel rule is when k is also on the circle
    with diameter (i,j). In this limiting case, then simplex ijk must
    be a right triangle, and dij**2 = djk**2 + dki**2 (by thales theorem).

    This means that when dij**2 > djk**2 + dki**2, then k is inside the circle.
    In contrast, when dij**2 < djk**2 + dji*2, k is outside of the circle.

    Therefore, it's sufficient to take each observation i, iterate over its
    Delaunay neighbors j,k, and remove links whre dij**2 > djk**2 + dki**2
    in order to construct the Gabriel graph.
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
    This is a direct implementation of the algorithm from Toussaint (1980), RNG-2

    1. Compute the delaunay
    2. for each edge of the delaunay (i,j), compute
       dkmax = max(d(k,i), d(k,j)) for k in 1..n, k != i, j
    3. for each edge of the delaunay (i,j), prune
       if any dkmax is greater than d(i,j)
    """
    n = edges.max()
    out = []
    r = []
    for edge in edges:
        i, j = edge
        pi = coordinates[i]
        pj = coordinates[j]
        dkmax = 0
        dij = ((pi - pj) ** 2).sum() ** 0.5
        prune = False
        for k in range(n):
            pk = coordinates[k]
            dik = ((pi - pk) ** 2).sum() ** 0.5
            djk = ((pj - pk) ** 2).sum() ** 0.5
            distances = numpy.array([dik, djk, dkmax])
            dkmax = distances.max()
            prune = dkmax < dij
            if (not return_dkmax) & prune:
                break
        if prune:
            continue
        out.append((i, j, dij))
        if return_dkmax:
            r.append(dkmax)

    return out, r


def _voronoi_edges(coordinates):
    dt = spatial.Delaunay(coordinates)
    edges = _edges_from_simplices(dt.simplices)
    edges = (
        pandas.DataFrame(numpy.asarray(list(edges)))
        .sort_values([0, 1])
        .drop_duplicates()
        .values
    )
    return edges, dt

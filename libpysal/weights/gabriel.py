from scipy.spatial import Delaunay as _Delaunay
from scipy import sparse
from libpysal.weights import W, WSP
import pandas, numpy, warnings

try:
    from numba import njit
except ModuleNotFoundError:
    from libpysal.common import jit as njit

__author__ = """"
Levi John Wolf (levi.john.wolf@gmail.com)
Martin Fleischmann (martin@martinfleischmann.net)
"""

#### Classes


class Delaunay(W):
    """
    Constructor of the Delaunay graph of a set of input points.
    Relies on scipy.spatial.Delaunay and numba to quickly construct
    a graph from the input set of points. Will be slower without numba,
    and will warn if this is missing.

    Parameters
    ----------
    coordinates :   array of points, (N,2)
        numpy array of coordinates containing locations to compute the
        delaunay triangulation
    **kwargs    :   keyword argument list
        keyword arguments passed directly to weights.W

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

    def __init__(self, coordinates, **kwargs):
        try:
            from numba import njit
        except ModuleNotFoundError:
            warnings.warn(
                "The numba package is used extensively in this module"
                " to accelerate the computation of graphs. Without numba,"
                " these computations may become unduly slow on large data."
            )
        edges, _ = self._voronoi_edges(coordinates)
        ids = kwargs.get("ids")
        if ids is not None:
            ids = numpy.asarray(ids)
            edges = numpy.column_stack((ids[edges[:, 0]], ids[edges[:, 1]]))
            del kwargs["ids"]
        else:
            ids = numpy.arange(coordinates.shape[0])

        voronoi_neighbors = pandas.DataFrame(edges).groupby(0)[1].apply(list).to_dict()
        W.__init__(self, voronoi_neighbors, id_order=list(ids), **kwargs)

    def _voronoi_edges(self, coordinates):
        dt = _Delaunay(coordinates)
        edges = _edges_from_simplices(dt.simplices)
        edges = (
            pandas.DataFrame(numpy.asarray(list(edges)))
            .sort_values([0, 1])
            .drop_duplicates()
            .values
        )
        return edges, dt

    @classmethod
    def from_dataframe(cls, df, geom_col=None, ids=None, use_index=None, **kwargs):
        """
        Construct a Delaunay triangulation from a geopandas GeoDataFrame.
        Not that the input geometries in the dataframe must be Points.
        Polygons or lines must be converted to points (e.g. using
        df.geometry.centroid).

        Parameters
        ----------
        df  :   geopandas.GeoDataFrame
            GeoDataFrame containing points to construct the Delaunay
            Triangulation.
        geom_col :  string
            the name of the column in `df` that contains the
            geometries. Defaults to active geometry column.
        ids   :   list-like, string
            a list-like of ids to use to index the spatial weights object or
            the name of the column to use as IDs. If nothing is
            provided, the dataframe index is used if `use_index=True` or
            a positional index is used if `use_index=False`.
            Order of the resulting W is not respected from this list.
        use_index  : bool
            use index of `df` as `ids` to index the spatial weights object.
        **kwargs :  keyword arguments
            Keyword arguments that are passed downwards to the weights.W
            constructor.
        """
        if isinstance(df, pandas.Series):
            df = df.to_frame("geometry")
        if geom_col is None:
            geom_col = df.geometry.name
        geomtypes = df[geom_col].geom_type.unique()

        if ids is None:
            if use_index is None:
                warnings.warn(
                    "`use_index` defaults to False but will default to True in future. "
                    "Set True/False directly to control this behavior and silence this "
                    "warning",
                    FutureWarning,
                    stacklevel=2,
                )
                use_index = False
            if use_index:
                ids = df.index.tolist()

        elif isinstance(ids, str):
            ids = df[ids].tolist()

        try:
            assert len(geomtypes) == 1
            assert geomtypes[0] == "Point"
            point_array = numpy.column_stack(
                (df[geom_col].x.values, df[geom_col].y.values)
            )
            return cls(point_array, ids=ids, **kwargs)
        except AssertionError:
            raise TypeError(
                f"The input dataframe has geometry types {geomtypes}"
                f" but this delaunay triangulation is only well-defined for points."
                f" Choose a method to convert your dataframe into points (like using"
                f" the df.centroid) and use that to estimate this graph."
            )


class Gabriel(Delaunay):
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
    coordinates :   array of points, (N,2)
        numpy array of coordinates containing locations to compute the
        delaunay triangulation
    **kwargs    :   keyword argument list
        keyword arguments passed directly to weights.W
    """

    def __init__(self, coordinates, **kwargs):
        try:
            from numba import njit
        except ModuleNotFoundError:
            warnings.warn(
                "The numba package is used extensively in this module"
                " to accelerate the computation of graphs. Without numba,"
                " these computations may become unduly slow on large data."
            )
        edges, dt = self._voronoi_edges(coordinates)
        droplist = _filter_gabriel(
            edges,
            dt.points,
        )
        output = numpy.row_stack(list(set(map(tuple, edges)).difference(set(droplist))))
        ids = kwargs.get("ids")
        if ids is not None:
            ids = numpy.asarray(ids)
            output = numpy.column_stack((ids[output[:, 0]], ids[output[:, 1]]))
            del kwargs["ids"]
        else:
            ids = numpy.arange(coordinates.shape[0])

        gabriel_neighbors = pandas.DataFrame(output).groupby(0)[1].apply(list).to_dict()
        W.__init__(self, gabriel_neighbors, id_order=list(ids), **kwargs)


class Relative_Neighborhood(Delaunay):
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
    coordinates :   array of points, (N,2)
        numpy array of coordinates containing locations to compute the
        delaunay triangulation
    **kwargs    :   keyword argument list
        keyword arguments passed directly to weights.W
    """

    def __init__(self, coordinates, binary=True, **kwargs):
        try:
            from numba import njit
        except ModuleNotFoundError:
            warnings.warn(
                "The numba package is used extensively in this module"
                " to accelerate the computation of graphs. Without numba,"
                " these computations may become unduly slow on large data."
            )
        edges, dt = self._voronoi_edges(coordinates)
        output, dkmax = _filter_relativehood(edges, dt.points, return_dkmax=False)
        row, col, data = zip(*output)
        if binary:
            data = numpy.ones_like(col, dtype=float)
        sp = sparse.csc_matrix((data, (row, col)))  # TODO: faster way than this?
        ids = kwargs.get("ids")
        if ids is None:
            ids = numpy.arange(sp.shape[0])
        else:
            del kwargs["ids"]
        ids = list(ids)
        tmp = WSP(sp, id_order=ids).to_W()
        W.__init__(self, tmp.neighbors, tmp.weights, id_order=ids, **kwargs)


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
    n = edges.max()
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
    edge_pointer = 0
    n = edges.max()
    n_edges = len(edges)
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

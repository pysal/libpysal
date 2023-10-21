import math
from functools import cached_property

import numpy as np
import pandas as pd
from scipy import sparse

from libpysal.weights import W

from ._contiguity import (
    _block_contiguity,
    _queen,
    _rook,
    _vertex_set_intersection,
    _fuzzy_contiguity,
)
from ._kernel import _distance_band, _kernel
from ._parquet import _read_parquet, _to_parquet
from ._set_ops import _Set_Mixin
from ._spatial_lag import _lag_spatial
from ._triangulation import _delaunay, _gabriel, _relative_neighborhood, _voronoi
from ._utils import _evaluate_index, _neighbor_dict_to_edges, _sparse_to_arrays

ALLOWED_TRANSFORMATIONS = ("O", "B", "R", "D", "V")

# listed alphabetically
__author__ = """"
Martin Fleischmann (martin@martinfleischmann.net)
Eli Knaap (ek@knaaptime.com)
Serge Rey (sjsrey@gmail.com)
Levi John Wolf (levi.john.wolf@gmail.com)
"""


class Graph(_Set_Mixin):
    """Graph class encoding spatial weights matrices

    The :class:`Graph` is currently experimental and its API is incomplete and unstable.
    """

    def __init__(self, adjacency, transformation="O"):
        """Weights base class based on adjacency list

        It is recommenced to use one of the ``from_*`` or ``build_*`` constructors
        rather than invoking ``__init__`` directly.

        Each observation needs to be present in the focal, at least as a self-loop with
        a weight 0.

        Parameters
        ----------
        adjacency : pandas.Series
            A MultiIndexed pandas.Series with ``"focal"`` and ``"neigbor"`` levels
            encoding adjacency, and values encoding weights. By convention,
            isolates are encoded as self-loops with a weight 0.
        transformation : str, default "O"
            weights transformation used to produce the table.

            - **O** -- Original
            - **B** -- Binary
            - **R** -- Row-standardization (global sum :math:`=n`)
            - **D** -- Double-standardization (global sum :math:`=1`)
            - **V** -- Variance stabilizing


        """
        if not isinstance(adjacency, pd.Series):
            raise TypeError(
                f"The adjacency table needs to be a pandas.Series. {type(adjacency)}"
            )
        if not adjacency.index.names == ["focal", "neighbor"]:
            raise ValueError(
                "The index of the adjacency table needs to be a MultiIndex named "
                "['focal', 'neighbor']."
            )
        if not adjacency.name == "weight":
            raise ValueError(
                "The adjacency needs to be named 'weight'. "
                f"'{adjacency.name}' was given instead."
            )
        if not pd.api.types.is_numeric_dtype(adjacency):
            raise ValueError(
                "The 'weight' needs to be of a numeric dtype. "
                f"'{adjacency.dtype}' dtype was given instead."
            )
        if adjacency.isna().any():
            raise ValueError("The adjacency table cannot contain missing values.")
        if transformation.upper() not in ALLOWED_TRANSFORMATIONS:
            raise ValueError(
                f"'transformation' needs to be one of {ALLOWED_TRANSFORMATIONS}. "
                f"'{transformation}' was given instead."
            )
        # adjacency always ordered i-->j on both levels
        ids = adjacency.index.get_level_values(0).unique().values
        adjacency = adjacency.reindex(ids, level=0).reindex(ids, level=1)
        self._adjacency = adjacency
        self.transformation = transformation

    def __getitem__(self, item):
        """Easy lookup based on focal index

        Parameters
        ----------
        item : hashable
            hashable represting an index value

        Returns
        -------
        pandas.Series
            subset of the adjacency table for `item`
        """
        if item in self.isolates:
            return pd.Series(
                [],
                index=pd.Index([], name="neighbor"),
                name="weight",
            )
        return self._adjacency.loc[item]

    def copy(self, deep=True):
        """Make a copy of this Graph's adjacency table and transformation

        Parameters
        ----------
        deep : bool, optional
            Make a deep copy of the adjacency table, by default True

        Returns
        -------
        Graph
            libpysal.graph.Graph as a copy of the original
        """
        return Graph(
            self._adjacency.copy(deep=deep), transformation=self.transformation
        )

    @cached_property
    def adjacency(self):
        """Return a copy of the adjacency list

        Returns
        -------
        pandas.Series
            Underlying adjacency list
        """
        return self._adjacency.copy()

    @classmethod
    def from_W(cls, w):
        """Create an experimental Graph from libpysal.weights.W object

        Parameters
        ----------
        w : libpysal.weights.W

        Returns
        -------
        Graph
            libpysal.graph.Graph from W
        """
        return cls.from_weights_dict(dict(w))

    def to_W(self):
        """Convert Graph to a libpysal.weights.W object

        Returns
        -------
        libpysal.weights.W
            representation of graph as a weights.W object
        """
        ids, labels = pd.factorize(
            self._adjacency.index.get_level_values("focal"), sort=False
        )
        neighbors = (
            self._adjacency.reset_index(level=1)
            .groupby(ids)
            .apply(
                lambda group: list(
                    group[
                        ~((group.index == group.neighbor) & (group.weight == 0))
                    ].neighbor
                )
            )
        )
        neighbors.index = labels[neighbors.index]
        weights = (
            self._adjacency.reset_index(level=1)
            .groupby(ids)
            .apply(
                lambda group: list(
                    group[
                        ~((group.index == group.neighbor) & (group.weight == 0))
                    ].weight
                )
            )
        )
        weights.index = labels[weights.index]
        return W(neighbors.to_dict(), weights.to_dict(), id_order=labels.tolist())

    @classmethod
    def from_adjacency(
        cls, adjacency, focal_col="focal", neighbor_col="neighbor", weight_col="weight"
    ):
        """Create a Graph from a pandas DataFrame formatted as an adjacency list

        Parameters
        ----------
        adjacency : pandas.DataFrame
            a dataframe formatted as an ajacency list. Should have columns
            "focal", "neighbor", and "weight", or columns that can be mapped
            to these (e.g. origin, destination, cost)
        focal : str, optional
            name of column holding focal/origin index, by default 'focal'
        neighbor : str, optional
            name of column holding neighbor/destination index, by default 'neighbor'
        weight : str, optional
            name of column holding weight values, by default 'weight'

        Returns
        -------
        Graph
            libpysal.graph.Graph
        """
        cols = dict(
            zip(
                [focal_col, neighbor_col, weight_col],
                ["focal_col", "neighbor_col", "weight_col"],
            )
        )
        for col in cols.keys():
            assert col in adjacency.columns.tolist(), (
                f'"{col}" was given for `{cols[col]}`, but the '
                f"columns available in `adjacency` are:  {adjacency.columns.tolist()}."
            )
        return cls.from_arrays(
            adjacency[focal_col].values,
            adjacency[neighbor_col].values,
            adjacency[weight_col].values,
        )

    @classmethod
    def from_sparse(cls, sparse, ids=None):
        """Convert a ``scipy.sparse`` array to a PySAL ``Graph`` object.

        Parameters
        ----------
        sparse : scipy.sparse array
            sparse representation of a graph
        ids : list-like, default None
            list-like of ids for geometries that is mappable to
            positions from sparse. If None, the positions are used as labels.

        Returns
        -------
        Graph
            libpysal.graph.Graph based on sparse
        """

        return cls.from_arrays(*_sparse_to_arrays(sparse, ids))

    @classmethod
    def from_arrays(cls, focal_ids, neighbor_ids, weight):
        """Generate Graph from arrays of indices and weights of the same length

        The arrays needs to be sorted in a way ensuring that focal_ids.unique() is
        equal to the index of original observations from which the Graph is being built

        Parameters
        ----------
        focal_index : array-like
            focal indices
        neighbor_index : array-like
            neighbor indices
        weight : array-like
            weights

        Returns
        -------
        Graph
            libpysal.graph.Graph based on arrays
        """

        w = cls(
            pd.Series(
                weight,
                name="weight",
                index=pd.MultiIndex.from_arrays(
                    [focal_ids, neighbor_ids], names=["focal", "neighbor"]
                ),
            )
        )

        return w

    @classmethod
    def from_weights_dict(cls, weights_dict):
        """Generate Graph from a dict of dicts

        Parameters
        ----------
        weights_dict : dictionary of dictionaries
            weights dictionary with the ``{focal: {neighbor: weight}}`` structure.

        Returns
        -------
        Graph
            libpysal.graph.Graph based on weights dictionary of dictionaries
        """
        idx = {f: [k for k in neighbors] for f, neighbors in weights_dict.items()}
        data = {
            f: [k for k in neighbors.values()] for f, neighbors in weights_dict.items()
        }
        return cls.from_dicts(idx, data)

    @classmethod
    def from_dicts(cls, neighbors, weights=None):
        """Generate Graph from dictionaries of neighbors and weights

        Parameters
        ----------
        neighbors : dict
            dictionary of neighbors with the ``{focal: [neighbor1, neighbor2]}``
            structure
        weights : dict, optional
            dictionary of neighbors with the ``{focal: [weight1, weight2]}``
            structure. If None, assumes binary weights.

        Returns
        -------
        Graph
            libpysal.graph.Graph based on dictionaries
        """
        head, tail, weight = _neighbor_dict_to_edges(neighbors, weights=weights)
        return cls.from_arrays(head, tail, weight)

    @classmethod
    def build_contiguity(cls, geometry, rook=True, by_perimeter=False, strict=False):
        """Generate Graph from geometry based on contiguity

        Contiguity builder assumes that all geometries are forming a coverage, i.e.
        a non-overlapping mesh and neighbouring geometries share only points or segments
        of their exterior boundaries. In practice, ``build_contiguity`` is capable of
        creating a Graph of partially overlapping geometries when
        ``strict=False, by_perimeter=False``, but that would not strictly follow the
        definition of queen or rook contiguity.

        Parameters
        ----------
        geometry : array-like of shapely.Geometry objects
            Could be geopandas.GeoSeries or geopandas.GeoDataFrame, in which case the
            resulting Graph is indexed by the original index. If an array of
            shapely.Geometry objects is passed, Graph will assume a RangeIndex.
        rook : bool, optional
            Contiguity method. If True, two geometries are considered neighbours if they
            share at least one edge. If False, two geometries are considered neighbours
            if they share at least one vertex. By default True
        by_perimeter : bool, optional
            If True, ``weight`` represents the length of the shared boundary between
            adjacent units, by default False. For row-standardized version of perimeter
            weights, use
            ``Graph.build_contiguity(gdf, by_perimeter=True).transform("r")``.
        strict : bool, optional
            Use the strict topological method. If False, the contiguity is determined
            based on shared coordinates or coordinate sequences representing edges. This
            assumes geometry coverage that is topologically correct. This method is
            faster but can miss some relations. If True, the contiguity is determined
            based on geometric relations that do not require precise topology. This
            method is slower but will result in correct contiguity even if the topology
            of geometries is not optimal. By default False

        Returns
        -------
        Graph
            libpysal.graph.Graph encoding contiguity weights
        """
        ids = _evaluate_index(geometry)

        if hasattr(geometry, "geometry"):
            # potentially cast GeoDataFrame to GeoSeries
            geometry = geometry.geometry

        if strict:
            # use shapely-based constructors
            if rook:
                return cls.from_arrays(
                    *_rook(geometry, ids=ids, by_perimeter=by_perimeter)
                )
            return cls.from_arrays(
                *_queen(geometry, ids=ids, by_perimeter=by_perimeter)
            )

        # use vertex-based constructor
        return cls.from_arrays(
            *_vertex_set_intersection(
                geometry, rook=rook, ids=ids, by_perimeter=by_perimeter
            )
        )

    @classmethod
    def build_kernel(
        cls,
        data,
        kernel="gaussian",
        k=None,
        bandwidth=None,
        metric="euclidean",
        p=2,
        coincident="raise",
    ):
        """Generate Graph from geometry data based on a kernel function

        Parameters
        ----------
        data : numpy.ndarray, geopandas.GeoSeries, geopandas.GeoDataFrame
            geometries over which to compute a kernel. If a geopandas object with Point
            geoemtry is provided, the .geometry attribute is used. If a numpy.ndarray
            with shapely geoemtry is used, then the coordinates are extracted and used.
            If a numpy.ndarray of a shape (2,n) is used, it is assumed to contain x, y
            coordinates. If metric="precomputed", data is assumed to contain a
            precomputed distance metric.
        kernel : string or callable (default: 'gaussian')
            kernel function to apply over the distance matrix computed by `metric`.
            The following kernels are supported:

            - ``"triangular"``:
            - ``"parabolic"``:
            - ``"gaussian"``:
            - ``"bisquare"``:
            - ``"cosine"``:
            - ``'boxcar'``/discrete: all distances less than `bandwidth` are 1, and all
              other distances are 0
            - ``"identity"``/None : do nothing, weight similarity based on raw distance
            - ``callable`` : a user-defined function that takes the distance vector and
              the bandwidth and returns the kernel: kernel(distances, bandwidth)

        k : int (default: None)
            number of nearest neighbors used to truncate the kernel. This is assumed
            to be constant across samples. If None, no truncation is conduted.
        bandwidth : float (default: None)
            distance to use in the kernel computation. Should be on the same scale as
            the input coordinates.
        metric : string or callable (default: 'euclidean')
            distance function to apply over the input coordinates. Supported options
            depend on whether or not scikit-learn is installed. If so, then any
            distance function supported by scikit-learn is supported here. Otherwise,
            only euclidean, minkowski, and manhattan/cityblock distances are admitted.
        p : int (default: 2)
            parameter for minkowski metric, ignored if metric != "minkowski".
        coincident: str, optional (default "raise")
            Method for handling coincident points when ``k`` is not None. Options include
            ``'raise'`` (raising an exception when coincident points are present),
            ``'jitter'`` (randomly displace coincident points to produce uniqueness), and
            ``'clique'`` (induce fully-connected sub cliques for coincident points).


        Returns
        -------
        Graph
            libpysal.graph.Graph encoding kernel weights
        """
        ids = _evaluate_index(data)

        head, tail, weight = _kernel(
            data,
            bandwidth=bandwidth,
            metric=metric,
            kernel=kernel,
            k=k,
            p=p,
            ids=ids,
            coincident=coincident,
        )

        return cls.from_arrays(head, tail, weight)

    @classmethod
    def build_knn(cls, data, k, metric="euclidean", p=2, coincident="raise"):
        """Generate Graph from geometry data based on k-nearest neighbors search

        Parameters
        ----------
        data : numpy.ndarray, geopandas.GeoSeries, geopandas.GeoDataFrame
            geometries over which to compute a kernel. If a geopandas object with Point
            geoemtry is provided, the .geometry attribute is used. If a numpy.ndarray
            with shapely geoemtry is used, then the coordinates are extracted and used.
            If a numpy.ndarray of a shape (2,n) is used, it is assumed to contain x, y
            coordinates.
        k : int
            number of nearest neighbors.
        metric : string or callable (default: 'euclidean')
            distance function to apply over the input coordinates. Supported options
            depend on whether or not scikit-learn is installed. If so, then any
            distance function supported by scikit-learn is supported here. Otherwise,
            only euclidean, minkowski, and manhattan/cityblock distances are admitted.
        p : int (default: 2)
            parameter for minkowski metric, ignored if metric != "minkowski".
        coincident: str, optional (default "raise")
            Method for handling coincident points. Options include
            ``'raise'`` (raising an exception when coincident points are present),
            ``'jitter'`` (randomly displace coincident points to produce uniqueness), and
            ``'clique'`` (induce fully-connected sub cliques for coincident points).


        Returns
        -------
        Graph
            libpysal.graph.Graph encoding KNN weights
        """
        ids = _evaluate_index(data)

        head, tail, weight = _kernel(
            data,
            bandwidth=np.inf,
            metric=metric,
            kernel="boxcar",
            k=k,
            p=p,
            ids=ids,
            coincident=coincident,
        )

        return cls.from_arrays(head, tail, weight)

    @classmethod
    def build_triangulation(
        cls,
        data,
        method="delaunay",
        bandwidth=np.inf,
        kernel="boxcar",
        clip="extent",
        rook=True,
        coincident="raise",
    ):
        """Generate Graph from geometry based on triangulation

        Parameters
        ----------
        data : numpy.ndarray, geopandas.GeoSeries, geopandas.GeoDataFrame
            geometries containing locations to compute the
            delaunay triangulation. If a geopandas object with Point
            geoemtry is provided, the .geometry attribute is used. If a numpy.ndarray
            with shapely geoemtry is used, then the coordinates are extracted and used.
            If a numpy.ndarray of a shape (2,n) is used, it is assumed to contain x, y
            coordinates.
        method : str, (default "delaunay")
            method of extracting the weights from triangulation. Supports:

            - ``"delaunay"``
            - ``"gabriel"``
            - ``"relative_neighborhood"``
            - ``"voronoi"``

        bandwidth : float, optional
            distance to use in the kernel computation. Should be on the same scale as
            the input coordinates, by default numpy.inf
        kernel : str, optional
            kernel function to use in order to weight the output graph. See
            :meth:`Graph.build_kernel` for details. By default "boxcar"
        clip : str (default: 'bbox')
            Clipping method when ``method="voronoi"``. Ignored otherwise.
            Default is ``'extent'``. Options are as follows.

            - ``'none'``/``None``: No clip is applied. Voronoi cells may be
              arbitrarily larger that the source map. Note that this may lead to
              cells that are many orders of magnitude larger in extent than the
              original map. Not recommended.
            - ``'bbox'``/``'extent'``/``'bounding box'``: Clip the voronoi cells to
              the bounding box of the input points.
            - ``'chull``/``'convex hull'``: Clip the voronoi cells to the convex hull
              of the input points.
            - ``'ashape'``/``'ahull'``: Clip the voronoi cells to the tightest hull
              that contains all points (e.g. the smallest alphashape, using
              :func:`libpysal.cg.alpha_shape_auto`).
            - ``shapely.Polygon``: Clip to an arbitrary Polygon.

        rook : bool, optional
            Contiguity method when ``method="voronoi"``. Ignored otherwise.
            If True, two geometries are considered neighbours if they
            share at least one edge. If False, two geometries are considered neighbours
            if they share at least one vertex. By default True
        coincident: str, optional (default "raise")
            Method for handling coincident points. Options include
            ``'raise'`` (raising an exception when coincident points are present),
            ``'jitter'`` (randomly displace coincident points to produce uniqueness), and
            ``'clique'`` (induce fully-connected sub cliques for coincident points).

        Returns
        -------
        Graph
            libpysal.graph.Graph encoding triangulation weights
        """
        ids = _evaluate_index(data)

        if method == "delaunay":
            head, tail, weights = _delaunay(
                data, ids=ids, bandwidth=bandwidth, kernel=kernel, coincident=coincident
            )
        elif method == "gabriel":
            head, tail, weights = _gabriel(
                data, ids=ids, bandwidth=bandwidth, kernel=kernel, coincident=coincident
            )
        elif method == "relative_neighborhood":
            head, tail, weights = _relative_neighborhood(
                data, ids=ids, bandwidth=bandwidth, kernel=kernel, coincident=coincident
            )
        elif method == "voronoi":
            head, tail, weights = _voronoi(
                data, ids=ids, clip=clip, rook=rook, coincident=coincident
            )
        else:
            raise ValueError(
                f"Method '{method}' is not supported. Use one of ['delaunay', "
                "'gabriel', 'relative_neighborhood', 'voronoi']."
            )

        return cls.from_arrays(head, tail, weights)

    @classmethod
    def build_distance_band(
        cls, data, threshold, binary=True, alpha=-1.0, kernel=None, bandwidth=None
    ):
        """Generate Graph from geometry based on a distance band

        Parameters
        ----------
        data : numpy.ndarray, geopandas.GeoSeries, geopandas.GeoDataFrame
            geometries containing locations to compute the
            delaunay triangulation. If a geopandas object with Point
            geometry is provided, the .geometry attribute is used. If a numpy.ndarray
            with shapely geometry is used, then the coordinates are extracted and used.
            If a numpy.ndarray of a shape (2,n) is used, it is assumed to contain x, y
            coordinates.
        threshold : float
            distance band
        binary : bool, optional
            If True :math:`w_{ij}=1` if :math:`d_{i,j}<=threshold`, otherwise
            :math:`w_{i,j}=0`.
            If False :math:`wij=dij^{alpha}`, by default True.
        alpha : float, optional
            distance decay parameter for weight (default -1.0)
            if alpha is positive the weights will not decline with
            distance. Ignored if ``binary=True`` or ``kernel`` is not None.
        kernel : str, optional
            kernel function to use in order to weight the output graph. See
            :meth:`Graph.build_kernel` for details. Ignored if ``binary=True``.
        bandwidth : float (default: None)
            distance to use in the kernel computation. Should be on the same scale as
            the input coordinates. Ignored if ``binary=True`` or ``kernel=None``.

        Returns
        -------
        Graph
            libpysal.graph.Graph encoding distance band weights
        """
        ids = _evaluate_index(data)

        dist = _distance_band(data, threshold)

        if binary:
            head, tail, weight = _kernel(
                dist,
                kernel="boxcar",
                metric="precomputed",
                ids=ids,
                bandwidth=np.inf,
            )
        elif kernel is not None:
            head, tail, weight = _kernel(
                dist,
                kernel=kernel,
                metric="precomputed",
                ids=ids,
                bandwidth=bandwidth,
            )
        else:
            head, tail, weight = _kernel(
                dist,
                kernel=lambda distances, alpha: np.power(distances, alpha),
                metric="precomputed",
                ids=ids,
                bandwidth=alpha,
            )

        adjacency = pd.DataFrame.from_dict(
            {"focal": head, "neighbor": tail, "weight": weight}
        ).set_index("focal")

        # drop diagonal
        counts = adjacency.index.value_counts()
        no_isolates = counts[counts > 1]
        adjacency = adjacency[
            ~(
                adjacency.index.isin(no_isolates.index)
                & (adjacency.index == adjacency.neighbor)
            )
        ]
        # set isolates to 0 - distance band should never contain self-weight
        adjacency.loc[~adjacency.index.isin(no_isolates.index), "weight"] = 0

        return cls.from_arrays(
            adjacency.index.values, adjacency.neighbor.values, adjacency.weight.values
        )

    @classmethod
    def build_block_contiguity(cls, regimes):
        """Generate Graph from block contiguity (regime neighbors)

        Block contiguity structures are relevant when defining neighbor relations
        based on membership in a regime. For example, all counties belonging to
        the same state could be defined as neighbors, in an analysis of all
        counties in the US.

        Parameters
        ----------
        regimes : list-like
            list-like of regimes. If pandas.Series, its index is used to encode Graph.
            Otherwise a default RangeIndex is used.

        Returns
        -------
        Graph
            libpysal.graph.Graph encoding block contiguity
        """
        ids = _evaluate_index(regimes)

        return cls.from_dicts(_block_contiguity(regimes, ids=ids))

    @classmethod
    def build_fuzzy_contiguity(
        cls,
        geometry,
        tolerance=None,
        buffer=None,
        predicate="intersects",
    ):
        """Generate Graph from fuzzy contiguity

        Fuzzy contiguity relaxes the notion of contiguity neighbors for the case of
        geometry collections that violate the condition of planar enforcement. It
        handles three types of conditions present in such collections that would result
        in missing links when using the regular contiguity methods.

        The first are edges for nearby polygons that should be shared, but are digitized
        separately for the individual polygons and the resulting edges do not
        coincide, but instead the edges intersect. This case can also be covered by
        ``build_contiguty`` with the ``strict=False`` parameter.

        The second case is similar to the first, only the resultant edges do not
        intersect but are "close". The optional buffering of geometry then closes the
        gaps between the polygons and a resulting intersection is encoded as a link.

        The final case arises when one polygon is "inside" a second polygon but is not
        encoded to represent a hole in the containing polygon.

        It is also possible to create a contiguity based on a custom spatial predicate.

        Parameters
        ----------
        geoms :  array-like of shapely.Geometry objects
            Could be geopandas.GeoSeries or geopandas.GeoDataFrame, in which case the
            resulting Graph is indexed by the original index. If an array of
            shapely.Geometry objects is passed, Graph will assume a RangeIndex.
        tolerance : float, optional
            The percentage of the length of the minimum side of the bounding rectangle
            for the ``geoms`` to use in determining the buffering distance. Either
            ``tolerance`` or ``buffer`` may be specified but not both.
            By default None.
        buffer : float, optional
            Exact buffering distance in the units of ``geoms.crs``. Either
            ``tolerance`` or ``buffer`` may be specified but not both.
            By default None.
        predicate : str, optional
            The predicate to use for determination of neighbors. Default is 'intersects'.
            If None is passed, neighbours are determined based on the intersection of
            bounding boxes. See the documentation of ``geopandas.GeoSeries.sindex.query``
            for allowed predicates.

        Returns
        -------
        Graph
            libpysal.graph.Graph encoding fuzzy contiguity
        """
        ids = _evaluate_index(geometry)

        heads, tails, weights = _fuzzy_contiguity(
            geometry, ids, tolerance=tolerance, buffer=buffer, predicate=predicate
        )

        return cls.from_arrays(heads, tails, weights)

    @cached_property
    def neighbors(self):
        """Get neighbors dictionary

        Notes
        -----
        It is recommended to work directly with :meth:`Graph.adjacency` rather than
        using the :meth:`Graph.neighbors`.

        Returns
        -------
        dict
            dict of tuples representing neighbors
        """
        return (
            self._adjacency.reset_index(level=1)
            .groupby(level=0)
            .apply(
                lambda group: tuple(
                    group[
                        ~((group.index == group.neighbor) & (group.weight == 0))
                    ].neighbor
                )
            )
            .to_dict()
        )

    @cached_property
    def weights(self):
        """Get weights dictionary

        Notes
        -----
        It is recommended to work directly with :meth:`Graph.adjacency` rather than
        using the :meth:`Graph.weights`.

        Returns
        -------
        dict
            dict of tuples representing weights
        """
        return (
            self._adjacency.reset_index(level=1)
            .groupby(level=0)
            .apply(
                lambda group: tuple(
                    group[
                        ~((group.index == group.neighbor) & (group.weight == 0))
                    ].weight
                )
            )
            .to_dict()
        )

    @cached_property
    def sparse(self):
        """Return a scipy.sparse array (COO)

        Returns
        -------
        scipy.sparse.COO
            sparse representation of the adjacency
        """
        # pivot to COO sparse matrix and cast to array
        return sparse.coo_array(
            self._adjacency.astype("Sparse[float]").sparse.to_coo(sort_labels=True)[0]
        )

    def transform(self, transformation):
        """Transformation of weights

        Parameters
        ----------
        transformation : str
            Transformation method. The following are
            valid transformations.

            - **B** -- Binary
            - **R** -- Row-standardization (global sum :math:`=n`)
            - **D** -- Double-standardization (global sum :math:`=1`)
            - **V** -- Variance stabilizing

        Returns
        -------
        Graph
            transformed weights

        Raises
        ------
        ValueError
            Value error for unsupported transformation
        """
        transformation = transformation.upper()

        if self.transformation == transformation:
            return self.copy()

        if transformation == "R":
            standardized = (
                (self._adjacency / self._adjacency.groupby(level=0).transform("sum"))
                .fillna(0)
                .values
            )  # isolate comes as NaN -> 0

        elif transformation == "D":
            standardized = (self._adjacency / self._adjacency.sum()).values

        elif transformation == "B":
            standardized = self._adjacency.astype(bool).astype(int)

        elif transformation == "V":
            s = self._adjacency.groupby(level=0).transform(
                lambda group: group / math.sqrt((group**2).sum())
            )
            nQ = self.n / s.sum()
            standardized = (s * nQ).fillna(0).values  # isolate comes as NaN -> 0

        else:
            raise ValueError(
                f"Transformation '{transformation}' is not supported. "
                f"Use one of {ALLOWED_TRANSFORMATIONS[1:]}"
            )

        standardized_adjacency = pd.Series(
            standardized, name="weight", index=self._adjacency.index
        )
        return Graph(standardized_adjacency, transformation)

    @cached_property
    def _components(self):
        """helper for n_components and component_labels"""
        # TODO: remove casting to matrix once scipy supports arrays here
        return sparse.csgraph.connected_components(sparse.coo_matrix(self.sparse))

    @cached_property
    def n_components(self):
        """Get a number of connected components

        Returns
        -------
        int
            number of components
        """
        return self._components[0]

    @cached_property
    def component_labels(self):
        """Get component labels per observation

        Returns
        -------
        numpy.array
            Array of component labels
        """
        return pd.Series(
            self._components[1], index=self.unique_ids, name="component labels"
        )

    @cached_property
    def cardinalities(self):
        """Number of neighbors for each observation

        Returns
        -------
        pandas.Series
            Series with a number of neighbors per each observation
        """
        cardinalities = self._adjacency.astype(bool).groupby(level=0).sum()
        cardinalities.name = "cardinalities"
        return cardinalities

    @cached_property
    def isolates(self):
        """Index of observations with no neighbors

        Isolates are encoded as a self-loop with the weight == 0 in the adjacency table.

        Returns
        -------
        pandas.Index
            Index with a subset of observations that do not have any neighbor
        """
        nulls = self._adjacency[self._adjacency == 0].reset_index(level=1)
        # since not all zeros are necessarily isolates, do the focal == neighbor check
        return nulls[nulls.index == nulls.neighbor].index.unique()

    @cached_property
    def unique_ids(self):
        """Unique IDs used in the Graph"""
        return self._adjacency.index.get_level_values("focal").unique()

    @cached_property
    def n(self):
        """Number of observations."""
        return self.unique_ids.shape[0]

    @cached_property
    def n_nodes(self):
        """Number of observations."""
        return self.unique_ids.shape[0]

    @cached_property
    def n_edges(self):
        """Number of observations."""
        return self._adjacency.shape[0] - self.isolates.shape[0]

    @cached_property
    def pct_nonzero(self):
        """Percentage of nonzero weights."""
        p = 100.0 * self.sparse.nnz / (1.0 * self.n**2)
        return p

    @cached_property
    def nonzero(self):
        """Number of nonzero weights."""
        return (self._adjacency.drop(self.isolates) > 0).sum()

    def asymmetry(self, intrinsic=True):
        """Asymmetry check.

        Parameters
        ----------

        intrinsic : bool, optional
            Default is ``True``. Intrinsic symmetry is defined as:

            .. math::

                w_{i,j} == w_{j,i}

            If ``intrinsic`` is ``False`` symmetry is defined as:

            .. math::

                i \in N_j \ \& \ j \in N_i

            where :math:`N_j` is the set of neighbors for :math:`j`,
            e.g., ``True`` requires equality of the weight to consider
            two links equal, ``False`` requires only a presence of a link
            with a non-zero weight.

        Returns
        -------

        pandas.Series
            A ``Series`` of ``(i,j)`` pairs of asymmetries sorted
            ascending by the focal observation (index value),
            where ``i`` is the focal and ``j`` is the neighbor.
            An empty ``Series`` is returned if no asymmetries are found.

        """
        if intrinsic:
            wd = self.sparse.transpose() - self.sparse
        else:
            transformed = self.transform("b")
            wd = transformed.sparse.transpose() - transformed.sparse

        ids = np.nonzero(wd)
        if len(ids[0]) == 0:
            return pd.Series(
                index=pd.Index([], name="focal"),
                name="neighbor",
                dtype=self._adjacency.index.dtypes["focal"],
            )
        else:
            i2id = dict(zip(np.arange(self.unique_ids.shape[0]), self.unique_ids))
            focal, neighbor = np.nonzero(wd)
            focal = focal.astype(self._adjacency.index.dtypes["focal"])
            neighbor = neighbor.astype(self._adjacency.index.dtypes["focal"])
            for i in i2id:
                focal[focal == i] = i2id[i]
                neighbor[neighbor == i] = i2id[i]
            ijs = pd.Series(
                neighbor, index=pd.Index(focal, name="focal"), name="neighbor"
            ).sort_index()
            return ijs

    def higher_order(self, k=2, shortest_path=True, diagonal=False, lower_order=False):
        """Contiguity weights object of order :math:`k`.

        Proper higher order neighbors are returned such that :math:`i` and :math:`j` are
        :math:`k`-order neighbors if the shortest path from :math:`i-j` is of length
        :math:`k`.

        Parameters
        ----------
        k : int, optional
            Order of contiguity. By default 2.
        shortest_path : bool, optional
            If True, :math:`i,j` and :math:`k`-order neighbors if the shortest
            path for :math:`i,j` is :math:`k`. If False, :math:`i,j` are
            `k`-order neighbors if there is a path from :math:`i,j` of length
            :math:`k`. By default True.
        diagonal : bool, optional
            If True, keep :math:`k`-order (:math:`i,j`) joins when :math:`i==j`.
            If False, remove :math:`k`-order (:math:`i,j`) joins when
            :math:`i==j`. By default False.
        lower_order : bool, optional
            If True, include lower order contiguities. If False return only weights of
            order :math:`k`. By default False.

        Returns
        -------
        Graph
            higher order weights
        """
        # TODO: remove casting to matrix once scipy implements matrix_power on array.
        # [https://github.com/scipy/scipy/pull/18544]
        binary = self.transform("B")
        sp = sparse.csr_matrix(binary.sparse)

        if lower_order:
            wk = sum(map(lambda x: sp**x, range(2, k + 1)))
            shortest_path = False
        else:
            wk = sp**k

        rk, ck = wk.nonzero()
        sk = set(zip(rk, ck))

        if shortest_path:
            for j in range(1, k):
                wj = sp**j
                rj, cj = wj.nonzero()
                sj = set(zip(rj, cj))
                sk.difference_update(sj)
        if not diagonal:
            sk = set([(i, j) for i, j in sk if i != j])

        return Graph.from_sparse(
            sparse.coo_array(
                (
                    np.ones(len(sk), dtype=np.int8),
                    ([s[0] for s in sk], [s[1] for s in sk]),
                ),
                shape=sp.shape,
            ),
            ids=self.unique_ids,
        )

    def lag(self, y):
        """Spatial lag operator

        If weights are row standardized, returns the mean of each observation's neighbors;
        if not, returns the weighted sum of each observation's neighbors.


        Parameters
        ----------
        y : array-like
            array-like (N,) shape where N is equal to number of observations in self.

        Returns
        -------
        numpy.ndarray
            array of numeric values for the spatial lag
        """
        return _lag_spatial(self, y)

    def to_parquet(self, path, **kwargs):
        """Save Graph to a Apache Parquet

        Graph is serialized to the Apache Parquet using the underlying adjacency
        object stored as a Parquet table and custom metadata containing transformation.

        Requires pyarrow package.

        Parameters
        ----------
        path : str | pyarrow.NativeFile
            path or any stream supported by pyarrow
        **kwargs
            additional keyword arguments passed to pyarrow.parquet.write_table

        See also
        --------
        read_parquet
        """
        _to_parquet(self, path, **kwargs)

    def to_networkx(self):
        """Convert Graph to a ``networkx`` graph.

        If Graph is symmetric, returns ``nx.Graph``, otherwise returns a ``nx.DiGraph``.

        Returns
        -------
        networkx.Graph | networkx.DiGraph
            Representation of libpysal Graph as networkx graph
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("NetworkX is required.")

        if self.asymmetry().empty:
            graph_type = nx.Graph
        else:
            graph_type = nx.DiGraph

        return nx.from_pandas_edgelist(
            self._adjacency.reset_index(),
            source="focal",
            target="neighbor",
            edge_attr="weight",
            create_using=graph_type,
        )


def _arrange_arrays(heads, tails, weights, ids=None):
    """
    Rearrange input arrays so that observation indices
    are well-ordered with respect to the input ids. That is,
    an "early" identifier should always preceed a "later" identifier
    in the heads, but the tails should be sorted with respect
    to heads *first*, then sorted within the tails.
    """
    if ids is None:
        ids = np.unique(np.hstack((heads, tails)))
    lookup = list(ids).index
    input_df = pd.DataFrame.from_dict(dict(focal=heads, neighbor=tails, weight=weights))
    return (
        input_df.set_index(["focal", "neighbor"])
        .assign(
            focal_loc=input_df.focal.apply(lookup).values,
            neighbor_loc=input_df.neighbor.apply(lookup).values,
        )
        .sort_values(["focal_loc", "neighbor_loc"])
        .reset_index()
        .drop(["focal_loc", "neighbor_loc"], axis=1)
        .values.T
    )


def read_parquet(path, **kwargs):
    """Read Graph from a Apache Parquet

    Read Graph serialized using `Graph.to_parquet()` back into the `Graph` object. The
    Parquet file needs to contain adjacency table with a structure required by the `Graph`
    constructor and optional metadata with the type of transformation.

    Parameters
    ----------
    path : str | pyarrow.NativeFile | file-like object
        path or any stream supported by pyarrow
    **kwargs
        additional keyword arguments passed to pyarrow.parquet.read_table

    Returns
    -------
    Graph
        deserialized Graph
    """
    adjacency, transformation = _read_parquet(path, **kwargs)
    return Graph(adjacency, transformation)

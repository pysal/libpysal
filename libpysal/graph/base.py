from functools import cached_property
import math

import numpy as np
import pandas as pd
from scipy import sparse

from libpysal.weights import W, block_weights
from ._contiguity import _queen, _rook, _vertex_set_intersection, _block_contiguity
from ._kernel import _kernel, _distance_band
from ._triangulation import _delaunay, _gabriel, _relative_neighborhood, _voronoi
from ._set_ops import _Set_Mixin
from ._utils import _neighbor_dict_to_edges, _evaluate_index
from ._parquet import _read_parquet, _to_parquet
from ._spatial_lag import _lag_spatial

ALLOWED_TRANSFORMATIONS = ("O", "B", "R", "D", "V")


class Graph(_Set_Mixin):
    def __init__(self, adjacency, transformation="O"):
        """Weights base class based on adjacency list

        It is recommenced to use one of the ``from_*`` or ``build_*`` constructors
        rather than invoking ``__init__`` directly.

        Parameters
        ----------
        adjacency : pandas.DataFrame
            pandas.DataFrame with a an index ``"focal"`` and columns
            ``["neighbor", "weight]`` encoding the adjacency. By convention,
            isolates are encoded as self-loops with a weight 0.
        transformation : str, default "O"
            weights transformation used to produce the table.

                - **O** -- Original
                - **B** -- Binary
                - **R** -- Row-standardization (global sum :math:`=n`)
                - **D** -- Double-standardization (global sum :math:`=1`)
                - **V** -- Variance stabilizing
        """
        if not isinstance(adjacency, pd.DataFrame):
            raise TypeError("The adjacency table needs to be a pandas.DataFrame.")
        if not adjacency.shape[1] == 2:
            raise ValueError(
                "The shape of the adjacency table needs to be (x, 2). "
                f"{adjacency.shape} was given instead."
            )
        if not adjacency.index.name == "focal":
            raise ValueError(
                "The index of the adjacency table needs to be named "
                f"'focal'. {adjacency.index.name} was given instead."
            )
        if not adjacency.columns.equals(
            pd.Index(["neighbor", "weight"], dtype="object")
        ):
            raise ValueError(
                "The adjacency table needs to contain columns "
                f"['neighbor', 'weight']. {adjacency.columns.tolist()} were given "
                "instead."
            )
        if not pd.api.types.is_numeric_dtype(adjacency.weight):
            raise ValueError(
                "The 'weight' columns needs to be of a numeric dtype. "
                f"'{adjacency.weight.dtype}' dtype was given instead."
            )
        if adjacency.isna().any().any():
            raise ValueError("The adjacency table cannot contain missing values.")
        if transformation.upper() not in ALLOWED_TRANSFORMATIONS:
            raise ValueError(
                f"'transformation' needs to be one of {ALLOWED_TRANSFORMATIONS}. "
                f"'{transformation}' was given instead."
            )

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
        return self._adjacency.loc[[item]].set_index("neighbor").weight

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
        neighbors = (
            self._adjacency.groupby(level=0)
            .apply(
                lambda group: list(
                    group[
                        ~((group.index == group.neighbor) & (group.weight == 0))
                    ].neighbor
                )
            )
            .to_dict()
        )
        weights = (
            self._adjacency.groupby(level=0)
            .apply(
                lambda group: list(
                    group[
                        ~((group.index == group.neighbor) & (group.weight == 0))
                    ].weight
                )
            )
            .to_dict()
        )
        return W(neighbors, weights)

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
        sparse = sparse.tocoo(copy=False)
        if ids is not None:
            ids = np.asarray(ids)
            sorter = sparse.row.argsort()
            head = ids[sparse.row][sorter]
            tail = ids[sparse.col][sorter]
            data = sparse.data[sorter]
        else:
            sorter = sparse.row.argsort()
            head = sparse.row[sorter]
            tail = sparse.col[sorter]
            data = sparse.data[sorter]

        return cls.from_arrays(head, tail, weight=data)

    @classmethod
    def from_arrays(cls, focal_ids, neighbor_ids, weight):
        """Generate Graph from arrays of indices and weights of the same length

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
            pd.DataFrame(
                index=pd.Index(focal_ids, name="focal"),
                data={"neighbor": neighbor_ids, "weight": weight},
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

        TODO: specify the planarity constraint of the defitnion of queen and rook (e.g
        that there could not be an overlap).

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
            TODO, by default False
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
                - triangular:
                - parabolic:
                - gaussian:
                - bisquare:
                - cosine:
                - boxcar/discrete: all distances less than `bandwidth` are 1, and all
                    other distances are 0
                - identity/None : do nothing, weight similarity based on raw distance
                - callable : a user-defined function that takes the distance vector and
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

        Returns
        -------
        Graph
            libpysal.graph.Graph encoding kernel weights
        """
        ids = _evaluate_index(data)

        sp, ids = _kernel(
            data,
            bandwidth=bandwidth,
            metric=metric,
            kernel=kernel,
            k=k,
            p=p,
            ids=ids,
        )

        return cls.from_sparse(sp, ids)

    @classmethod
    def build_knn(cls, data, k, metric="euclidean", p=2):
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

        Returns
        -------
        Graph
            libpysal.graph.Graph encoding KNN weights
        """
        ids = _evaluate_index(data)

        sp, ids = _kernel(
            data,
            bandwidth=np.inf,
            metric=metric,
            kernel="boxcar",
            k=k,
            p=p,
            ids=ids,
        )

        return cls.from_sparse(sp, ids)

    @classmethod
    def build_triangulation(
        cls,
        data,
        method="delaunay",
        bandwidth=np.inf,
        kernel="boxcar",
        clip="extent",
        rook=True,
        coincident='raise'
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
                - "delaunay"
                - "gabriel"
                - "relative_neighborhood"
                - "voronoi"
        bandwidth : _type_, optional
            distance to use in the kernel computation. Should be on the same scale as
            the input coordinates, by default numpy.inf
        kernel : str, optional
            kernel function to use in order to weight the output graph. See
            :meth:`Graph.build_kernel` for details. By default "boxcar"
        clip : str (default: 'bbox')
            Clipping method when ``method="voronoi"``. Ignored otherwise.
            Default is ``'extent'``. Options are as follows.

            * ``'none'``/``None`` -- No clip is applied. Voronoi cells may be
                arbitrarily larger that the source map. Note that this may lead to
                cells that are many orders of magnitude larger in extent than the
                original map. Not recommended.
            * ``'bbox'``/``'extent'``/``'bounding box'`` -- Clip the voronoi cells to
                the bounding box of the input points.
            * ``'chull``/``'convex hull'`` -- Clip the voronoi cells to the convex hull
                of the input points.
            * ``'ashape'``/``'ahull'`` -- Clip the voronoi cells to the tightest hull
                that contains all points (e.g. the smallest alphashape, using
                ``libpysal.cg.alpha_shape_auto``).
            * Polygon -- Clip to an arbitrary Polygon.
        rook : bool, optional
            Contiguity method when ``method="voronoi"``. Ignored otherwise.
            If True, two geometries are considered neighbours if they
            share at least one edge. If False, two geometries are considered neighbours
            if they share at least one vertex. By default True

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
            head, tail, weights = _voronoi(data, ids=ids, clip=clip, rook=rook, coincident=coincident)
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
            If True w_{ij}=1 if d_{i,j}<=threshold, otherwise w_{i,j}=0
            If False wij=dij^{alpha}, by default True.
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
            sp, ids = _kernel(
                dist,
                kernel="boxcar",
                metric="precomputed",
                ids=ids,
                bandwidth=np.inf,
            )
        elif kernel is not None:
            sp, ids = _kernel(
                dist,
                kernel=kernel,
                metric="precomputed",
                ids=ids,
                bandwidth=bandwidth,
            )
        else:
            sp, ids = _kernel(
                dist,
                kernel=lambda distances, alpha: np.power(distances, alpha),
                metric="precomputed",
                ids=ids,
                bandwidth=alpha,
            )
        sp.setdiag(0)

        adjacency = cls.from_sparse(sp, ids)._adjacency

        # drop diagonal
        counts = adjacency.index.value_counts()
        no_isolates = counts[counts > 1]
        adjacency = adjacency[
            ~(
                adjacency.index.isin(no_isolates.index)
                & (adjacency.index == adjacency.neighbor)
            )
        ]
        return cls(adjacency)

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
            self._adjacency.groupby(level=0)
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
            self._adjacency.groupby(level=0)
            .apply(
                lambda group: tuple(
                    group[
                        ~((group.index == group.neighbor) & (group.weight == 0))
                    ].weight
                )
            )
            .to_dict()
        )

    def get_neighbors(self, ix):
        """Get neighbors for a set focal object

        Parameters
        ----------
        ix : hashable
            index of focal object

        Returns
        -------
        array
            array of indices of neighbor objects
        """
        if ix in self.isolates:
            return np.array([], dtype=self._adjacency.neighbor.dtype)

        return self._adjacency.neighbor[self._adjacency.index == ix].values

    def get_weights(self, ix):
        """Get weights for a set focal object

        Parameters
        ----------
        ix : hashable
            index of focal object

        Returns
        -------
        array
            array of weights of neighbor object
        """
        if ix in self.isolates:
            return np.array([], dtype=self._adjacency.weight.dtype)

        return self._adjacency.weight.loc[ix].values

    @cached_property
    def sparse(self):
        """Return a scipy.sparse array (COO)

        Returns
        -------
        scipy.sparse.COO
            sparse representation of the adjacency
        """
        return sparse.coo_array(
            (
                self._adjacency.weight.values,
                (
                    self._adjacency.index.map(self._id2i),
                    self._adjacency.neighbor.map(self._id2i),
                ),
            ),
            shape=(self.n, self.n),
        )

    @cached_property
    def _id2i(self):
        """Mapping of index to integer position in sparse"""
        ix = np.arange(self.unique_ids.shape[0])
        return dict(zip(self.unique_ids, ix))

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
                (
                    self._adjacency.weight
                    / self._adjacency.weight.groupby(level=0).transform("sum")
                )
                .fillna(0)
                .values
            )  # isolate comes as NaN -> 0

        elif transformation == "D":
            standardized = (
                self._adjacency.weight / self._adjacency.weight.sum()
            ).values

        elif transformation == "B":
            standardized = self._adjacency.weight.astype(bool).astype(int)

        elif transformation == "V":
            s = self._adjacency.weight.groupby(level=0).transform(
                lambda group: group / math.sqrt((group**2).sum())
            )
            nQ = self.n / s.sum()
            standardized = (s * nQ).fillna(0).values  # isolate comes as NaN -> 0

        else:
            raise ValueError(
                f"Transformation '{transformation}' is not supported. "
                f"Use one of {ALLOWED_TRANSFORMATIONS[1:]}"
            )

        standardized_adjacency = self._adjacency.assign(weight=standardized)
        return Graph(standardized_adjacency, transformation)

    @cached_property
    def _components(self):
        # TODO: scipy.sparse.csgraph does not yet support sparse arrays, only matrices
        # TODO: we need this to be implemented in scipy first. Then the code below shall
        # TODO: work as expected (here and in n_components and component_labels).
        return NotImplementedError
        # return sparse.csgraph.connected_components(self.sparse)

    @cached_property
    def n_components(self):
        """Get a number of connected components

        Returns
        -------
        int
            number of components
        """
        return NotImplementedError
        # return self._components[0]

    @cached_property
    def component_labels(self):
        """Get component labels per observation

        Returns
        -------
        numpy.array
            Array of component labels
        """
        return NotImplementedError
        # return self._components[1]

    @cached_property
    def cardinalities(self):
        """Number of neighbors for each observation

        Returns
        -------
        pandas.Series
            Series with a number of neighbors per each observation
        """
        cardinalities = self._adjacency.weight.astype(bool).groupby(level=0).sum()
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
        nulls = self._adjacency[self._adjacency.weight == 0]
        # since not all zeros are necessarily isolates, do the focal == neighbor check
        return nulls[nulls.index == nulls.neighbor].index.unique()

    @cached_property
    def unique_ids(self):
        return pd.concat(
            [self._adjacency.index.to_series(), self._adjacency.neighbor]
        ).unique()

    @cached_property
    def n(self):
        """Number of observations."""
        return self.unique_ids.shape[0]

    @cached_property
    def pct_nonzero(self):
        """Percentage of nonzero weights."""
        p = 100.0 * self.sparse.nnz / (1.0 * self.n**2)
        return p

    @cached_property
    def nonzero(self):
        """Number of nonzero weights."""
        nnz = self.sparse.nnz
        return nnz

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
                dtype=self._adjacency.neighbor.dtype,
            )
        else:
            i2id = {v: k for k, v in self._id2i.items()}
            focal, neighbor = np.nonzero(wd)
            focal = focal.astype(self._adjacency.index.dtype)
            neighbor = neighbor.astype(self._adjacency.neighbor.dtype)
            for i in i2id:
                focal[focal == i] = i2id[i]
                neighbor[neighbor == i] = i2id[i]
            ijs = pd.Series(
                neighbor, index=pd.Index(focal, name="focal"), name="neighbor"
            ).sort_index()
            return ijs

    def higher_order(self, k=2, shortest_path=True, diagonal=False, lower_order=False):
        """Contiguity weights object of order K.

        TODO: This currently does not work as scipy.sparse array does not
        yet implement matrix_power. We need to reimplement it temporarily and
        switch once that is released. [https://github.com/scipy/scipy/pull/18544]

        Parameters
        ----------
        k : int, optional
            order of contiguity, by default 2
        shortest_path : bool, optional
            True: i,j and k-order neighbors if the
            shortest path for i,j is k.
            False: i,j are k-order neighbors if there
            is a path from i,j of length k.
            By default True
        diagonal : bool, optional
            True:  keep k-order (i,j) joins when i==j
            False: remove k-order (i,j) joins when i==j
            By default False
        lower_order : bool, optional
            True: include lower order contiguities
            False: return only weights of order k
            By default False

        Returns
        -------
        Graph
            higher order weights
        """
        return NotImplementedError
        # binary = self.transform("B")
        # sparse = binary.sparse

        # if lower_order:
        #     wk = sum(map(lambda x: sparse**x, range(2, k + 1)))
        #     shortest_path = False
        # else:
        #     wk = sparse**k

        # rk, ck = wk.nonzero()
        # sk = set(zip(rk, ck))

        # if shortest_path:
        #     for j in range(1, k):
        #         wj = sparse**j
        #         rj, cj = wj.nonzero()
        #         sj = set(zip(rj, cj))
        #         sk.difference_update(sj)
        # if not diagonal:
        #     sk = set([(i, j) for i, j in sk if i != j])

        # ix = pd.MultiIndex.from_tuples(sk, names=["focal", "neighbor"])
        # new_index = pd.MultiIndex.from_arrays(
        #     (
        #         binary.focal_label.take(ix.get_level_values("focal")),
        #         binary.neighbor_label.take(ix.get_level_values("neighbor")),
        #     ),
        #     names=["focal", "neighbor"],
        # )
        # return Graph(
        #     pd.Series(
        #         index=new_index,
        #         data=np.ones(len(ix), dtype=int),
        #     )
        # )

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

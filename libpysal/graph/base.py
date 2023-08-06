from functools import cached_property
import math

import numpy as np
import pandas as pd
from scipy import sparse

from libpysal.weights import W
from ._contiguity import _queen, _rook, _vertex_set_intersection
from ._triangulation import _delaunay, _gabriel, _relative_neighborhood, _voronoi
from ._set_ops import _Set_Mixin
from ._utils import _neighbor_dict_to_edges

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

    def copy(self, deep=True):
        """Make a copy of this Graph's adjacency table and transformation

        Parameters
        ----------
        deep : bool, optional
            Make a deep copy of the adjacency table, by default True

        Returns
        -------
        Graph
            libpysal.graph.Graph
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
            libpysal.graph.Graph
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
    def from_sparse(cls, sparse, focal_ids=None, neighbor_ids=None):
        """Convert a ``scipy.sparse`` array to a PySAL ``Graph`` object.

        Parameters
        ----------
        sparse : scipy.sparse array
            sparse representation of a graph
        focal_ids : list-like, default None
            list-like of ids for focal geometries that is mappable to
            positions from sparse. If None, the positions are used as labels.
        neighbor_ids : list-like, default None
            list-like of ids for neighbor geometries that is mappable to
            positions from sparse. If None, the positions are used as labels.

        Returns
        -------
        Graph
            libpysal.graph.Graph
        """
        sparse = sparse.tocoo(copy=False)
        if focal_ids is not None and neighbor_ids is not None:
            focal_ids = np.asarray(focal_ids)
            neighbor_ids = np.asarray(neighbor_ids)
            f = sparse.row
            n = sparse.col
            focal_ids = focal_ids[f]
            neighbor_ids = neighbor_ids[n]
        elif (focal_ids is None) and (neighbor_ids is None):
            focal_ids = sparse.row
            neighbor_ids = sparse.col
        else:
            raise ValueError(
                "Either both focal_ids and neighbor_ids are provided,"
                " or neither may be provided."
            )

        return cls.from_arrays(focal_ids, neighbor_ids, weight=sparse.data)

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
            libpysal.graph.Graph
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
            libpysal.graph.Graph
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
            libpysal.graph.Graph
        """
        head, tail, weight = _neighbor_dict_to_edges(neighbors, weights=weights)
        return cls.from_arrays(head, tail, weight)

    @classmethod
    def build_contiguity(cls, geometry, rook=True, by_perimeter=False, strict=False):
        """Generate Graph from geometry based on the contiguity

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
            libpysal.graph.Graph
        """
        if hasattr(geometry, "index"):
            ids = geometry.index
        else:
            ids = pd.RangeIndex(0, len(geometry))

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
        bandwidth=None,
        metric="euclidean",
        kernel="gaussian",
        k=None,
        p=2,
    ):
        """_summary_

        Parameters
        ----------
        data : numpy.ndarray, geopandas.GeoSeries, geopandas.GeoDataFrame
            geometries over which to compute a kernel. If a geopandas object with Point
            geoemtry is provided, the .geometry attribute is used. If a numpy.ndarray
            with shapely geoemtry is used, then the coordinates are extracted and used.
            If a numpy.ndarray of a shape (2,n) is used, it is assumed to contain x, y
            coordinates. If metric="precomputed", data is assumed to contain a
            precomputed distance metric.
        bandwidth : float (default: None)
            distance to use in the kernel computation. Should be on the same scale as
            the input coordinates.
        metric : string or callable (default: 'euclidean')
            distance function to apply over the input coordinates. Supported options
            depend on whether or not scikit-learn is installed. If so, then any
            distance function supported by scikit-learn is supported here. Otherwise,
            only euclidean, minkowski, and manhattan/cityblock distances are admitted.
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
        ids : numpy.narray (default: None)
            ids to use for each sample in coordinates. Generally, construction functions
            that are accessed via Graph.build_kernel() will set this automatically from
            the index of the input. Do not use this argument directly unless you intend
            to set the indices separately from your input data. Otherwise, use
            data.set_index(ids) to ensure ordering is respected. If None, then the index
            from the input coordinates will be used.
        p : int (default: 2)
            parameter for minkowski metric, ignored if metric != "minkowski".

        Returns
        -------
        Graph
            libpysal.graph.Graph
        """
        return NotImplementedError

    @classmethod
    def build_triangulation(
        cls,
        data,
        method="delaunay",
        bandwidth=np.inf,
        kernel="boxcar",
        clip="extent",
        rook=True,
    ):
        """_summary_

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
            libpysal.graph.Graph
        """
        if hasattr(data, "index"):
            ids = data.index
        else:
            ids = pd.RangeIndex(0, len(data))

        if method == "delaunay":
            head, tail, weights = _delaunay(
                data, ids=ids, bandwidth=bandwidth, kernel=kernel
            )
        elif method == "gabriel":
            head, tail, weights = _gabriel(
                data, ids=ids, bandwidth=bandwidth, kernel=kernel
            )
        elif method == "relative_neighborhood":
            head, tail, weights = _relative_neighborhood(
                data, ids=ids, bandwidth=bandwidth, kernel=kernel
            )
        elif method == "voronoi":
            head, tail, weights = _relative_neighborhood(
                data, ids=ids, clip=clip, rook=rook
            )
        else:
            raise ValueError(
                f"Method '{method}' is not supported. Use one of ['delaunay', "
                "'gabriel', 'relative_neighborhood', 'voronoi']."
            )

        return Graph.from_arrays(head, tail, weights)

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

        Also saves self.focal_label and self.neighbor_label capturing
        the original index labels related to their integer representation.

        Returns
        -------
        scipy.sparse.COO
            sparse representation of the adjacency
        """
        # factorize(sort=True) ensures the order is unchanged (weirdly enough)
        focal_int, self.focal_label = self._adjacency.index.factorize(sort=True)
        neighbor_int, self.neighbor_label = self._adjacency.neighbor.factorize(
            sort=True
        )
        return sparse.coo_array(
            (self._adjacency.weight.values, (focal_int, neighbor_int))
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
    def n(self):
        """Number of observations."""
        return self._adjacency.index.nunique()

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
            Default is ``True``. Intrinsic symmetry is defined as

            .. math::

                w_{i,j} == w_{j,i}

            If ``intrinsic`` is ``False`` symmetry is defined as

            .. math::

                i \in N_j \ \& \ j \in N_i

            where :math:`N_j` is the set of neighbors for :math:`j`,
            e.g., ``True`` requires equality of the weight to consider
            two links equal, ``False`` requires only a presence of a link
            with a non-zero weight.

        Returns
        -------
        pandas.Series
            ``Series`` of ``(i,j)`` pairs of asymmetries
        """
        if intrinsic:
            wd = self.sparse.transpose() - self.sparse
            focal_labels = self.focal_label
            neighbor_labels = self.neighbor_label
        else:
            transformed = self.transform("b")
            wd = transformed.sparse.transpose() - transformed.sparse
            focal_labels = transformed.focal_label
            neighbor_labels = transformed.neighbor_label

        ids = np.nonzero(wd)
        if len(ids[0]) == 0:
            return pd.Series(
                index=pd.Index([], name="focal"),
                name="neighbor",
                dtype=self._adjacency.neighbor.dtype,
            )
        else:
            focal = focal_labels[ids[0]]
            neighbor = neighbor_labels[ids[1]]
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

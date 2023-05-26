from scipy import sparse
import numpy as np
import pandas as pd

from functools import cached_property
from ._contiguity import _queen, _rook, _vertex_set_intersection
from ._utils import _neighbor_dict_to_edges


class W:
    def __init__(self, adjacency, transformation="O"):
        """Weights base class based on adjacency list

        Parameters
        ----------
        adjacency : pandas.Series
            pandas.Series with a MultiIndex with two levels ("focal", "neighbor")
        """
        self._adjacency = adjacency
        self.transformation = transformation

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
    def from_old_w(cls, w):
        """Create an experimental W from libpysal.weights.W object

        Parameters
        ----------
        w : libpysal.weights.W

        Returns
        -------
        W
            libpysal.weights.experimental.W
        """
        return cls.from_weights_dict(dict(w))

    @classmethod
    def from_sparse(cls, sparse, focal_ids=None, neighbor_ids=None):
        """Convert a ``scipy.sparse`` array to a PySAL ``W`` object.

        Parameters
        ----------
        sparse : scipy.sparse array

        Returns
        -------
        W
            libpysal.weights.experimental.W
        """
        if focal_ids is not None and neighbor_ids is not None:
            f, n = sparse.nonzero()
            focal_ids = focal_ids[f]
            neighbor_ids = neighbor_ids[n]
        elif (focal_ids is None) and (neighbor_ids is None):
            focal_ids, neighbor_ids = sparse.nonzero()
        else:
            raise ValueError(
                "Either both focal_ids and neighbor_ids are provided,"
                " or neither may be provided."
            )

        return cls.from_arrays(focal_ids, neighbor_ids, weight=sparse.data)

    @classmethod
    def from_arrays(cls, focal_ids, neighbor_ids, weight):
        """Generate W from arrays of indices and weights of the same length

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
        W
            libpysal.weights.experimental.W
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
        """Generate W from a dict of dicts

        Parameters
        ----------
        weights_dict : dictionary of dictionaries
            weights dictionary with the ``{focal: {neighbor: weight}}`` structure.

        Returns
        -------
        W
            libpysal.weights.experimental.W
        """
        idx = {f: [k for k in neighbors] for f, neighbors in weights_dict.items()}
        data = {
            f: [k for k in neighbors.values()] for f, neighbors in weights_dict.items()
        }
        return cls.from_dicts(idx, data)

    @classmethod
    def from_dicts(cls, neighbors, weights=None):
        """Generate W from dictionaries of neighbors and weights

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
        W
            libpysal.weights.experimental.W
        """
        head, tail, weight = _neighbor_dict_to_edges(neighbors, weights=weights)
        return cls.from_arrays(head, tail, weight)

    @classmethod
    def from_contiguity(cls, geometry, rook=True, by_perimeter=False, strict=False):
        """Generate W from geometry based on the contiguity

        Parameters
        ----------
        geometry : array-like of shapely.Geometry objects
            Could be geopandas.GeoSeries or geopandas.GeoDataFrame, in which case the
            resulting W is indexed by the original index. If an array of
            shapely.Geometry objects is passed, W will assume a RangeIndex.
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
        W
            libpysal.weights.experimental.W
        """
        # TODO: deal with islands, those are now dropped
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

    @cached_property
    def neighbors(self):
        """Get neighbors dictionary

        Returns
        -------
        dict
            dict of tuples representing neighbors
        """
        # TODO: ensure the dict remains unsorted
        return self._adjacency.neighbor.groupby(level=0).agg(tuple).to_dict()

    @cached_property
    def weights(self):
        """Get weights dictionary

        Returns
        -------
        dict
            dict of tuples representing weights
        """
        # TODO: ensure the dict remains unsorted
        return self._adjacency.weight.groupby(level=0).agg(tuple).to_dict()

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
        return self._adjacency[ix].index.values

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
        return self._adjacency[ix].values

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
        W
            transformed weights

        Raises
        ------
        ValueError
            Value error for unsupported transformation
        """
        transformation = transformation.upper()

        if self.transformation == transformation:
            return self

        if transformation == "R":
            standardized = (
                self._adjacency.weight
                / self._adjacency.weight.groupby(level=0).transform("sum")
            ).values

        elif transformation == "D":
            standardized = (
                self._adjacency.weight / self._adjacency.weight.sum()
            ).values

        # TODO: deal with islands once we have W.islands specified
        elif transformation == "B":
            standardized = np.ones(self._adjacency.shape[0], dtype=int)

        elif transformation == "V":
            standardized = (
                self._adjacency.weight
                / np.sqrt(self._adjacency.weight.groupby(level=0).transform("sum"))
            ).values

        else:
            raise ValueError(f"Transformation '{transformation}' is not supported.")

        standardized_adjacency = self._adjacency.copy()
        standardized_adjacency["weight"] = standardized
        return W(standardized_adjacency, transformation)

    @cached_property
    def _components(self):
        return sparse.csgraph.connected_components(self.sparse)

    @cached_property
    def n_components(self):
        """Get a number of connected components

        Returns
        -------
        int
            number of components
        """
        return self._components()[0]

    @cached_property
    def component_labels(self):
        """Get component labels per observation

        Returns
        -------
        numpy.array
            Array of component labels
        """
        return self._components()[1]

    @cached_property
    def cardinalities(self):
        """Number of neighbors for each observation

        Returns
        -------
        pandas.Series
            Series with a number of neighbors per each observation
        """
        return self.adjacency.neighbor.groupby(level=0).count()

    @property
    def n(self):
        """Number of units."""
        n = np.unique(
            np.concatenate([self._adjacency.index, self._adjacency.neighbor])
        ).shape[0]
        return n

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

            where :math:`N_j` is the set of neighbors for :math:`j`.

        Returns
        -------
        pandas.Series
            ``Series`` of ``(i,j)`` pairs of asymmetries
        """
        if intrinsic:
            wd = self.sparse.transpose() - self.sparse
        else:
            transformed = self.transform("b")
            wd = transformed.sparse.transpose() - transformed.sparse

        ids = np.nonzero(wd)
        if len(ids[0]) == 0:
            return []
        else:
            focal = self.focal_label[ids[0]]
            neighbor = self.neighbor_label[ids[1]]
            ijs = pd.Series(
                neighbor, index=pd.Index(focal, name="focal"), name="neighbor"
            ).sort_index()
            return ijs

    def higher_order(self, k=2, shortest_path=True, diagonal=False, lower_order=False):
        """Contiguity weights object of order K.

        TODO: This currently does not work as scipy.sparse array does not
        yet implement matrix_power. We need to reimplement it temporarily and
        switch once that is released.

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
        W
            higher order weights
        """
        binary = self.transform("B")
        sparse = binary.sparse

        if lower_order:
            wk = sum(map(lambda x: sparse**x, range(2, k + 1)))
            shortest_path = False
        else:
            wk = sparse**k

        rk, ck = wk.nonzero()
        sk = set(zip(rk, ck))

        if shortest_path:
            for j in range(1, k):
                wj = sparse**j
                rj, cj = wj.nonzero()
                sj = set(zip(rj, cj))
                sk.difference_update(sj)
        if not diagonal:
            sk = set([(i, j) for i, j in sk if i != j])

        ix = pd.MultiIndex.from_tuples(sk, names=["focal", "neighbor"])
        new_index = pd.MultiIndex.from_arrays(
            (
                binary.focal_label.take(ix.get_level_values("focal")),
                binary.neighbor_label.take(ix.get_level_values("neighbor")),
            ),
            names=["focal", "neighbor"],
        )
        return W(
            pd.Series(
                index=new_index,
                data=np.ones(len(ix), dtype=int),
            )
        )

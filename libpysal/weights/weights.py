"""Spatial Weights."""

__author__ = "Sergio J. Rey <srey@asu.edu>, eli knaap <ek@knaaptime.com>"

import copy
import warnings
from functools import cached_property
from os.path import basename as BASENAME

import numpy as np
import pandas as pd
import scipy.sparse
from scipy.sparse.csgraph import connected_components

from ..io.fileio import FileIO as popen

# from .util import full, WSP2W resolve import cycle by
# forcing these into methods
from . import adjtools

__all__ = ["W", "WSP"]


def _dict_to_df(neighbors, weights):

    combined = dict()
    for key in neighbors.keys():
        combined[key] = dict(zip(neighbors[key], weights[key]))

    combineddf = pd.DataFrame.from_dict(combined).stack()
    combineddf = combineddf.to_frame(name="weight")
    combineddf.index.set_names(["focal", "neighbor"], inplace=True)
    return combineddf


class _LabelEncoder(object):
    """Encode labels with values between 0 and n_classes-1.

    Attributes
    ----------
    classes_: array of shape [n_classes]
        Class labels for each index.

    Examples
    --------
    >>> le = _LabelEncoder()
    >>> le.fit(["NY", "CA", "NY", "CA", "TX", "TX"])
    >>> le.classes_
    array(['CA', 'NY', 'TX'])
    >>> le.transform(["NY", "CA", "NY", "CA", "TX", "TX"])
    array([1, 0, 1, 0, 2, 2])
    """

    def fit(self, y):
        """Fit label encoder.

        Parameters
        ----------
        y : list
            list of labels

        Returns
        -------
        self : instance of self.
          Fitted label encoder.
        """
        self.classes_ = np.unique(y)
        return self

    def transform(self, y):
        """Transform labels to normalized encoding.

        Parameters
        ----------
        y : list
            list of labels

        Returns
        -------
        y : array
            array of normalized labels.
        """
        return np.searchsorted(self.classes_, y)


class W(object):
    """
    Spatial weights class. Class attributes are described by their
    docstrings. to view, use the ``help`` function.

    Parameters
    ----------

    neighbors : dict
        Key is region ID, value is a list of neighbor IDS.
        For example, ``{'a':['b'],'b':['a','c'],'c':['b']}``.
    weights : dict
       Key is region ID, value is a list of edge weights.
       If not supplied all edge weights are assumed to have a weight of 1.
       For example, ``{'a':[0.5],'b':[0.5,1.5],'c':[1.5]}``.
    id_order : list
       An ordered list of ids, defines the order of observations when
       iterating over ``W`` if not set, lexicographical ordering is used
       to iterate and the ``id_order_set`` property will return ``False``.
       This can be set after creation by setting the ``id_order`` property.
    silence_warnings : bool
       By default ``libpysal`` will print a warning if the dataset contains
       any disconnected components or islands. To silence this warning set this
       parameter to ``True``.
    ids : list
        Values to use for keys of the neighbors and weights ``dict`` objects.

    Attributes
    ----------

    asymmetries
    cardinalities
    component_labels
    diagW2
    diagWtW
    diagWtW_WW
    histogram
    ids
    id2i
    id_order
    id_order_set
    islands
    max_neighbors
    mean_neighbors
    min_neighbors
    n
    n_components
    neighbor_offsets
    nonzero
    pct_nonzero
    s0
    s1
    s2
    s2array
    sd
    sparse
    trcW2
    trcWtW
    trcWtW_WW
    transform

    Examples
    --------

    >>> from libpysal.weights import W
    >>> neighbors = {0: [3, 1], 1: [0, 4, 2], 2: [1, 5], 3: [0, 6, 4], 4: [1, 3, 7, 5], 5: [2, 4, 8], 6: [3, 7], 7: [4, 6, 8], 8: [5, 7]}
    >>> weights = {0: [1, 1], 1: [1, 1, 1], 2: [1, 1], 3: [1, 1, 1], 4: [1, 1, 1, 1], 5: [1, 1, 1], 6: [1, 1], 7: [1, 1, 1], 8: [1, 1]}
    >>> w = W(neighbors, weights)
    >>> "%.3f"%w.pct_nonzero
    '29.630'

    Read from external `.gal file <https://geodacenter.github.io/workbook/4a_contig_weights/lab4a.html#gal-weights-file>`_.

    >>> import libpysal
    >>> w = libpysal.io.open(libpysal.examples.get_path("stl.gal")).read()
    >>> w.n
    78
    >>> "%.3f"%w.pct_nonzero
    '6.542'

    Set weights implicitly.

    >>> neighbors = {0: [3, 1], 1: [0, 4, 2], 2: [1, 5], 3: [0, 6, 4], 4: [1, 3, 7, 5], 5: [2, 4, 8], 6: [3, 7], 7: [4, 6, 8], 8: [5, 7]}
    >>> w = W(neighbors)
    >>> round(w.pct_nonzero,3)
    29.63
    >>> from libpysal.weights import lat2W
    >>> w = lat2W(100, 100)
    >>> w.trcW2
    39600.0
    >>> w.trcWtW
    39600.0
    >>> w.transform='r'
    >>> round(w.trcW2, 3)
    2530.722
    >>> round(w.trcWtW, 3)
    2533.667

    Cardinality Histogram:

    >>> w.histogram
    [(2, 4), (3, 392), (4, 9604)]

    Disconnected observations (islands):

    >>> from libpysal.weights import W
    >>> w = W({1:[0],0:[1],2:[], 3:[]})

    UserWarning: The weights matrix is not fully connected:
    There are 3 disconnected components.
    There are 2 islands with ids: 2, 3.

    """

    def __init__(
        self, neighbors, weights=None, id_order=None, silence_warnings=False, ids=None
    ):
        self.silence_warnings = silence_warnings
        islands = list()
        if not weights:
            weights = {}
            for key in neighbors:
                if len(neighbors[key]) == 0:
                    islands.append(key)
                weights[key] = [1.0] * len(neighbors[key])

        self.df = _dict_to_df(neighbors, weights)
        self.islands = islands
        islands_list = list()
        # islands become self-loops with zero weight
        if len(self.islands) > 0:
            for island in self.islands:
                islands_list.append(
                    pd.DataFrame(
                        {"weight": 0},
                        index=pd.MultiIndex.from_arrays(
                            [[island], [island]], names=["focal", "neighbor"]
                        ),
                    )
                )
            islands_list = pd.concat(islands_list)
            self.df = pd.concat([self.df, islands_list])
        # weights transformations are columns in the weights dataframe
        self.df["weight_o"] = self.df["weight"]  # original weights

        # stashed them here in init to see how they work. In practice we can check
        # whether the column exists and create if not when the transformer method is called
        self.df["weight_r"] = self.df["weight"] / self.df.groupby("focal")[
            "weight"
        ].transform("sum")
        self.df["weight_b"] = 1
        self.df["weight_d"] = None  # not yet implemented
        self.df["weight_v"] = np.sqrt(
            self.df.groupby("focal")["weight"].transform("sum")
        )

        self.transform = "O"

        if id_order:
            warnings.warn(
                "`id_order` is deprecated and will be removed in future.",
                FutureWarning,
                stacklevel=2,
            )
            ids = id_order
        if ids is not None and len(ids) > 0:
            # re-align islands
            self.df = self.df.reset_index()
            self.df[["focal", "neighbor"]] = self.df[["focal", "neighbor"]].replace(
                dict(zip(list(weights.keys()), ids))
            )
            self.df = self.df.set_index(["focal", "neighbor"])
            self.df = self.df.reindex(ids, level=0).reindex(ids, level=1)

        if (not self.silence_warnings) and (self.n_components > 1):
            message = (
                "The weights matrix is not fully connected: "
                "\n There are %d disconnected components." % self.n_components
            )
            ni = len(self.islands)
            if ni == 1:
                message = message + "\n There is 1 island with id: %s." % (
                    str(self.islands[0])
                )
            elif ni > 1:
                message = message + "\n There are %d islands with ids: %s." % (
                    ni,
                    ", ".join(str(island) for island in self.islands),
                )
            warnings.warn(message)

    @property
    def weights(self):
        weights = self.df.groupby("focal").agg(list)["weight"].to_dict()
        for island in self.islands:
            weights[island] = []
        return weights

    @property
    def neighbors(self):
        neighbors = (
            self.df.reset_index().groupby("focal").agg(list)["neighbor"].to_dict()
        )
        for island in self.islands:
            neighbors[island] = []
        return neighbors

    @property
    def id_order(self):
        warnings.warn(
            "`id_order` is deprecated and will be removed in future.",
            FutureWarning,
            stacklevel=2,
        )
        return self.ids

    @property
    def transform(self):
        return get_transform

    @transform.setter
    def my_attr(self, value):
        self._transform = value

    def to_file(self, path="", format=None):
        """
        Write a weights to a file. The format is guessed automatically
        from the path, but can be overridden with the format argument.

        See libpysal.io.FileIO for more information.

        Parameters
        ----------
        path    :   string
                    location to save the file
        format  :   string
                    string denoting the format to write the weights to.


        Returns
        -------
        None
        """
        f = popen(dataPath=path, mode="w", dataFormat=format)
        f.write(self)
        f.close()

    @classmethod
    def from_file(cls, path="", format=None):
        """
        Read a weights file into a W object.

        Parameters
        ----------
        path    :   string
                    location to save the file
        format  :   string
                    string denoting the format to write the weights to.

        Returns
        -------
        W object
        """
        f = popen(dataPath=path, mode="r", dataFormat=format)
        w = f.read()
        f.close()
        return w

    @classmethod
    def from_shapefile(cls, *args, **kwargs):
        # we could also just "do the right thing," but I think it'd make sense to
        # try and get people to use `Rook.from_shapefile(shapefile)` rather than
        # W.from_shapefile(shapefile, type=`rook`), otherwise we'd need to build
        # a type dispatch table. Generic W should be for stuff we don't know
        # anything about.
        raise NotImplementedError(
            "Use type-specific constructors, like Rook, Queen, DistanceBand, or Kernel"
        )

    @classmethod
    def from_WSP(cls, WSP, silence_warnings=True):
        """Create a pysal W from a pysal WSP object (thin weights matrix).

        Parameters
        ----------
        wsp                     : WSP
                                PySAL sparse weights object

        silence_warnings        : bool
           By default ``libpysal`` will print a warning if the dataset contains
           any disconnected components or islands. To silence this warning set this
           parameter to ``True``.


        Returns
        -------
        w       : W
                PySAL weights object

        Examples
        --------
        >>> from libpysal.weights import lat2W, WSP, W

        Build a 10x10 scipy.sparse matrix for a rectangular 2x5 region of cells
        (rook contiguity), then construct a PySAL sparse weights object (wsp).

        >>> sp = lat2SW(2, 5)
        >>> wsp = WSP(sp)
        >>> wsp.n
        10
        >>> wsp.sparse[0].todense()
        matrix([[0, 1, 0, 0, 0, 1, 0, 0, 0, 0]], dtype=int8)

        Create a standard PySAL W from this sparse weights object.

        >>> w = W.from_WSP(wsp)
        >>> w.n
        10
        >>> print(w.full()[0][0])
        [0 1 0 0 0 1 0 0 0 0]
        """
        data = WSP.sparse.data
        indptr = WSP.sparse.indptr
        ids = WSP.ids
        if WSP.ids:
            # replace indices with user IDs
            id_order = ids
        if ids:
            # replace indices with user IDs
            indices = [ids[i] for i in WSP.sparse.indices]
        else:
            id_order = list(range(WSP.n))
        neighbors, weights = {}, {}
        start = indptr[0]
        for i in range(WSP.n):
            oid = id_order[i]
            end = indptr[i + 1]
            neighbors[oid] = indices[start:end]
            weights[oid] = data[start:end]
            start = end
        w = W(neighbors, weights, silence_warnings=silence_warnings)
        return w

    @classmethod
    def from_adjlist(
        cls, adjlist, focal_col="focal", neighbor_col="neighbor", weight_col=None
    ):
        """
        Return an adjacency list representation of a weights object.

        Parameters
        ----------

        adjlist : pandas.DataFrame
            Adjacency list with a minimum of two columns.
        focal_col : str
            Name of the column with the "source" node ids.
        neighbor_col : str
            Name of the column with the "destination" node ids.
        weight_col : str
            Name of the column with the weight information. If not provided and
            the dataframe has no column named "weight" then all weights
            are assumed to be 1.
        """
        if weight_col is None:
            weight_col = "weight"
        try_weightcol = getattr(adjlist, weight_col)
        if try_weightcol is None:
            adjlist = adjlist.copy(deep=True)
            adjlist["weight"] = 1
        grouper = adjlist.groupby(focal_col)
        neighbors = dict()
        weights = dict()
        for ix, chunk in grouper:
            neighbors_to_ix = chunk[neighbor_col].values
            weights_to_ix = chunk[weight_col].values
            mask = neighbors_to_ix != ix
            neighbors[ix] = neighbors_to_ix[mask].tolist()
            weights[ix] = weights_to_ix[mask].tolist()
        return cls(neighbors=neighbors, weights=weights)

    def to_adjlist(
        self,
        remove_symmetric=False,
        drop_islands=None,
        focal_col="focal",
        neighbor_col="neighbor",
        weight_col="weight",
        sort_joins=False,
    ):
        """
        Compute an adjacency list representation of a weights object.
        Parameters
        ----------
        remove_symmetric : bool
            Whether or not to remove symmetric entries. If the ``W``
            is symmetric, a standard directed adjacency list will contain
            both the forward and backward links by default because adjacency
            lists are a directed graph representation. If this is ``True``,
            a ``W`` created from this adjacency list **MAY NOT BE THE SAME**
            as the original ``W``. If you would like to consider (1,2) and
            (2,1) as distinct links, leave this as ``False``.
        drop_islands : bool
            Whether or not to preserve islands as entries in the adjacency
            list. By default, observations with no neighbors do not appear
            in the adjacency list. If islands are kept, they are coded as
            self-neighbors with zero weight.
        focal_col : str
            Name of the column in which to store "source" node ids.
        neighbor_col : str
            Name of the column in which to store "destination" node ids.
        weight_col : str
            Name of the column in which to store weight information.
        sort_joins : bool
            Whether or not to lexicographically sort the adjacency
            list by (focal_col, neighbor_col). Default is False.
        """
        try:
            import pandas
        except (ImportError, ModuleNotFoundError):
            raise ImportError(
                "pandas must be installed & importable to use this method"
            )
        if (drop_islands is None) and not (self.silence_warnings):
            warnings.warn(
                "In the next version of libpysal, observations with no neighbors will be included in adjacency lists as loops (row with the same focal and neighbor) with zero weight. In the current version, observations with no neighbors are dropped. If you would like to keep the current behavior, use drop_islands=True in this function",
                DeprecationWarning,
            )
            drop_islands = True

        adjlist = self.df.reset_index()[[focal_col, neighbor_col, weight_col]]
        return adjlist

    def to_networkx(self):
        """Convert a weights object to a ``networkx`` graph.

        Returns
        -------
        A ``networkx`` graph representation of the ``W`` object.
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("NetworkX 2.7+ is required to use this function.")
        G = nx.DiGraph() if len(self.asymmetries) > 0 else nx.Graph()
        return nx.from_scipy_sparse_array(self.sparse, create_using=G)

    @classmethod
    def from_networkx(cls, graph, weight_col="weight"):
        """Convert a ``networkx`` graph to a PySAL ``W`` object.

        Parameters
        ----------
        graph : networkx.Graph
            The graph to convert to a ``W``.
        weight_col : string
            If the graph is labeled, this should be the name of the field
            to use as the weight for the ``W``.

        Returns
        -------
        w : libpysal.weights.W
            A ``W`` object containing the same graph as the ``networkx`` graph.
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("NetworkX 2.7+ is required to use this function.")
        sparse_array = nx.to_scipy_sparse_array(graph)
        w = WSP(sparse_array).to_W()
        return w

    @property
    def sparse(self):
        """Sparse matrix object. For any matrix manipulations required for w,
        ``w.sparse`` should be used. This is based on ``scipy.sparse``.
        """
        return self._build_sparse()

    @classmethod
    def from_sparse(cls, sparse):
        """Convert a ``scipy.sparse`` array to a PySAL ``W`` object.

        Parameters
        ----------
        sparse : scipy.sparse array

        Returns
        -------
        w : libpysal.weights.W
            A ``W`` object containing the same graph as the ``scipy.sparse`` graph.


        Notes
        -----
        When the sparse array has a zero in its data attribute, and
        the corresponding row and column values are equal, the value
        for the pysal weight will be 0 for the "loop".
        """

        adj = pd.Series.sparse.from_coo(sparse.tocoo()).to_frame(name="weight")
        adj.index.set_names(["focal", "neighbor"], inplace=True)
        adj = adj.sparse.to_dense()

        weights = adj.groupby("focal").agg(list)["weight"].to_dict()
        neighbors = (
            adj.reset_index()
            .groupby("focal")
            .agg(lambda x: list(x.unique()))["neighbor"]
            .to_dict()
        )
        if not len(neighbors.keys()) == sparse.shape[0]:
            # the sparse matrix will encode islands as all-null rows. If there are missing indices from the dense table, those are islands
            missing = [
                i for i in list(range(0, sparse.shape[0])) if i not in neighbors.keys()
            ]
            for island in missing:
                neighbors[island] = [island]
                weights[island] = [0]

        return W(neighbors=neighbors, weights=weights)

    def to_sparse(self, fmt="coo"):
        """Generate a ``scipy.sparse`` array object from a pysal W.

        Parameters
        ----------
        fmt : {'bsr', 'coo', 'csc', 'csr'}
          scipy.sparse format

        Returns
        -------
        scipy.sparse array
          A scipy.sparse array with a format of fmt.

        Notes
        -----
        The keys of the w.neighbors are encoded
        to determine row,col in the sparse array.

        """
        kinds = ["bsr", "coo", "csc", "csr"]
        fmt_l = fmt.lower()
        if fmt_l not in kinds:
            raise ValueError(f"unsupported sparse format: {fmt}")
        elif fmt_l == "bsr":
            return self.sparse.tobsr()
        elif fmt_l == "csc":
            return self.sparse.tocsc()
        elif fmt_l == "coo":
            return self.sparse.tocoo()
        else:
            return self.sparse

    @cached_property
    def n_components(self):
        """Store whether the adjacency matrix is fully connected."""
        n_components, component_labels = connected_components(self.sparse)
        return n_components

    @cached_property
    def component_labels(self):
        """Store the graph component in which each observation falls."""
        n_components, component_labels = connected_components(self.sparse)
        return component_labels

    def _build_sparse(self):
        """Construct the sparse attribute."""
        return (
            self.df["weight"]
            # sort the "focal" column by its order in `ids``
            .reindex(self.ids, level=0)
            # sort the "neighbor" column by its order
            # .sort_index(level=1)
            .reindex(self.ids, level=1)
            .astype("Sparse[float]")
            .sparse.to_coo(
                row_levels=["focal"], column_levels=["neighbor"], sort_labels=True
            )[0]
            .tocsr()
        )

    @property
    def ids(self):
        ids = list(self.neighbors.keys())
        return ids

    @property
    def id2i(self):
        """Mapping of the W ids to their index order in the matrix representation
        index in ``W.id_order``.
        """
        id2i = dict(zip(self.ids, list(range(len(self.ids)))))
        return id2i

    @property
    def n(self):
        """Number of units."""
        n = len(self.ids)
        return n

    @cached_property
    def s0(self):
        r"""``s0`` is defined as

        .. math::

               s0=\sum_i \sum_j w_{i,j}

        """
        s0 = self.sparse.sum()
        return s0

    @cached_property
    def s1(self):
        r"""``s1`` is defined as

        .. math::

               s1=1/2 \sum_i \sum_j \Big(w_{i,j} + w_{j,i}\Big)^2

        """
        t = self.sparse.transpose()
        t = t + self.sparse
        t2 = t.multiply(t)  # element-wise square
        s1 = t2.sum() / 2.0
        return s1

    @cached_property
    def s2array(self):
        """Individual elements comprising ``s2``.

        See Also
        --------
        s2

        """
        s = self.sparse
        s2array = np.array(s.sum(1) + s.sum(0).transpose()) ** 2
        return s2array

    @cached_property
    def s2(self):
        r"""``s2`` is defined as

        .. math::

                s2=\sum_j \Big(\sum_i w_{i,j} + \sum_i w_{j,i}\Big)^2

        """
        s2 = self.s2array.sum()
        return s2

    @cached_property
    def trcW2(self):
        """Trace of :math:`WW`.

        See Also
        --------
        diagW2

        """
        t = self.diagW2.sum()
        return t

    @cached_property
    def diagW2(self):
        """Diagonal of :math:`WW`.

        See Also
        --------
        trcW2

        """
        d = (self.sparse * self.sparse).diagonal()
        return d

    @cached_property
    def diagWtW(self):
        """Diagonal of :math:`W^{'}W`.

        See Also
        --------
        trcWtW

        """
        d = (self.sparse.transpose() * self.sparse).diagonal()
        return d

    @cached_property
    def trcWtW(self):
        """Trace of :math:`W^{'}W`.

        See Also
        --------
        diagWtW

        """
        return self.diagWtW.sum()

    @cached_property
    def diagWtW_WW(self):
        """Diagonal of :math:`W^{'}W + WW`."""
        wt = self.sparse.transpose()
        w = self.sparse
        diagWtW_WW = (wt * w + w * w).diagonal()
        return diagWtW_WW

    @cached_property
    def trcWtW_WW(self):
        """Trace of :math:`W^{'}W + WW`."""
        t = self.diagWtW_WW.sum()
        return t

    @cached_property
    def pct_nonzero(self):
        """Percentage of nonzero weights."""
        p = 100.0 * self.sparse.nnz / (1.0 * self.n**2)
        return p

    @cached_property
    def cardinalities(self):
        """Number of neighbors for each observation."""
        cards = (
            self.df.groupby("focal")
            .count()
            .rename(columns={"weight": "n_neighbors"})
            .unstack()["n_neighbors"]
        )
        return cards

    @cached_property
    def max_neighbors(self):
        """Largest number of neighbors."""
        m = self.cardinalities.max()
        return m

    @cached_property
    def mean_neighbors(self):
        """Average number of neighbors."""
        m = self.cardinalities.mean()
        return m

    @cached_property
    def min_neighbors(self):
        """Minimum number of neighbors."""
        m = self.cardinalities.min()
        return m

    @cached_property
    def nonzero(self):
        """Number of nonzero weights."""
        nnz = self.sparse.nnz
        return nnz

    @cached_property
    def sd(self):
        """Standard deviation of number of neighbors."""
        sd = self.cardinalities.std(ddof=0)
        return sd

    @cached_property
    def asymmetries(self):
        """List of id pairs with asymmetric weights."""
        a = self.asymmetry()
        return a

    @cached_property
    def islands(self):
        """List of ids without any neighbors."""
        i = [i for i, c in self.cardinalities.to_dict().items() if c == 0]
        return i

    @property
    def neighbor_offsets(self):
        warnings.warn(
            "`neighbor_offsets` is deprecated and will be removed in the future"
        )
        return self.neighbors

    @property
    def histogram(self):
        """Cardinality histogram as a dictionary where key is the id and
        value is the number of neighbors for that unit.
        """
        return list(self.cardinalities.value_counts().items())

    @property
    def id_order_set(self):
        warnings.warn("`id_order` is deprecated and will be removed in the future")
        return True

    def __getitem__(self, key):
        """Allow a dictionary like interaction with the weights class.

        Examples
        --------
        >>> from libpysal.weights import lat2W
        >>> w = lat2W()

        >>> w[0] == dict({1: 1.0, 5: 1.0})
        True
        """
        loc = self.df.loc[key].to_dict()["weight"]
        return loc

    def __iter__(self):
        """
        Support iteration over weights.

        Examples
        --------
        >>> from libpysal.weights import lat2W
        >>> w=lat2W(3,3)
        >>> for i,wi in enumerate(w):
        ...     print(i,wi[0])
        ...
        0 0
        1 1
        2 2
        3 3
        4 4
        5 5
        6 6
        7 7
        8 8
        >>>
        """
        for i in self.ids:
            yield i, self.df.loc[i].to_dict()["weight"]

    def remap_ids(self, new_ids):
        """
        In place modification throughout ``W`` of id values from
        ``w.id_order`` to ``new_ids`` in all.

        Parameters
        ----------

        new_ids : list, numpy.ndarray
            Aligned list of new ids to be inserted. Note that first
            element of ``new_ids`` will replace first element of
            ``w.id_order``, second element of ``new_ids`` replaces second
            element of ``w.id_order`` and so on.

        Examples
        --------

        >>> from libpysal.weights import lat2W
        >>> w = lat2W(3, 3)
        >>> w.id_order
        [0, 1, 2, 3, 4, 5, 6, 7, 8]
        >>> w.neighbors[0]
        [3, 1]
        >>> new_ids = ['id%i'%id for id in w.id_order]
        >>> _ = w.remap_ids(new_ids)
        >>> w.id_order
        ['id0', 'id1', 'id2', 'id3', 'id4', 'id5', 'id6', 'id7', 'id8']
        >>> w.neighbors['id0']
        ['id3', 'id1']
        """

        old_ids = self.ids
        if new_ids is None:
            new_ids = list(range(len(old_ids)))
        self.df = self.df.reset_index()
        self.df[["focal", "neighbor"]] = self.df[["focal", "neighbor"]].replace(
            dict(zip(old_ids, new_ids))
        )
        self.df = self.df.set_index(["focal", "neighbor"])

    def __set_id_order(self, ordered_ids):
        """Set the iteration order in w. ``W`` can be iterated over. On
        construction the iteration order is set to the lexicographic order of
        the keys in the ``w.weights`` dictionary. If a specific order
        is required it can be set with this method.

        Parameters
        ----------

        ordered_ids : sequence
            Identifiers for observations in specified order.

        Notes
        -----

        The ``ordered_ids`` parameter is checked against the ids implied
        by the keys in ``w.weights``. If they are not equivalent sets an
        exception is raised and the iteration order is not changed.

        Examples
        --------

        >>> from libpysal.weights import lat2W
        >>> w=lat2W(3,3)
        >>> for i,wi in enumerate(w):
        ...     print(i, wi[0])
        ...
        0 0
        1 1
        2 2
        3 3
        4 4
        5 5
        6 6
        7 7
        8 8
        >>> w.ids
        [0, 1, 2, 3, 4, 5, 6, 7, 8]
        >>> w.ids=range(8,-1,-1)
        >>> list(w.ids)
        [8, 7, 6, 5, 4, 3, 2, 1, 0]
        >>> for i,w_i in enumerate(w):
        ...     print(i,w_i[0])
        ...
        0 8
        1 7
        2 6
        3 5
        4 4
        5 3
        6 2
        7 1
        8 0

        """

        if set(self._ids) == set(ordered_ids):
            self._ids = ordered_ids
            self._id_order_set = True
            self._reset()
        else:
            raise Exception("ordered_ids do not align with W ids")

    def __get_id_order(self):
        """Returns the ids for the observations in the order in which they
        would be encountered if iterating over the weights.
        """
        return self.ids

    def get_transform(self):
        """Getter for transform property.

        Returns
        -------
        transformation : str, None
            Valid transformation value. See the ``transform``
            parameters in ``set_transform()`` for a detailed description.

        Examples
        --------
        >>> from libpysal.weights import lat2W
        >>> w=lat2W()
        >>> w.weights[0]
        [1.0, 1.0]
        >>> w.transform
        'O'
        >>> w.transform='r'
        >>> w.weights[0]
        [0.5, 0.5]
        >>> w.transform='b'
        >>> w.weights[0]
        [1.0, 1.0]

        See also
        --------
        set_transform

        """

        return self._transform

    def set_transform(self, value="B"):
        """Transformations of weights.

        Parameters
        ----------
        transform : str
            This parameter is not case sensitive. The following are
            valid transformations.

            * **B** -- Binary
            * **R** -- Row-standardization (global sum :math:`=n`)
            * **D** -- Double-standardization (global sum :math:`=1`)
            * **V** -- Variance stabilizing
            * **O** -- Restore original transformation (from instantiation)

        Notes
        -----

        Transformations are applied only to the value of the weights at
        instantiation. Chaining of transformations cannot be done on a ``W``
        instance.


        Examples
        --------
        >>> from libpysal.weights import lat2W
        >>> w=lat2W()
        >>> w.weights[0]
        [1.0, 1.0]
        >>> w.transform
        'O'
        >>> w.transform='r'
        >>> w.weights[0]
        [0.5, 0.5]
        >>> w.transform='b'
        >>> w.weights[0]
        [1.0, 1.0]
        """
        value = value.upper()
        self._transform = value

        if value == "R":
            self.df["weight"] = self.df["weight_r"]

        elif value == "D":
            self.df["weight"] = self.df["weight_d"]

        elif value == "B":
            self.df["weight"] = self.df["weight_b"]

        elif value == "V":
            self.df["weight"] = self.df["weight_v"]

        elif value == "O":
            self.df["weight"] = self.df["weight_o"]

        else:
            raise Exception("unsupported weights transformation")

    transform = property(get_transform, set_transform)

    def asymmetry(self, intrinsic=True):
        r"""
        Asymmetry check.

        Parameters
        ----------
        intrinsic : bool
            Default is ``True``. Intrinsic symmetry is defined as

            .. math::

                w_{i,j} == w_{j,i}

            If ``intrinsic`` is ``False`` symmetry is defined as

            .. math::

                i \in N_j \ \& \ j \in N_i

            where :math:`N_j` is the set of neighbors for :math:`j`.

        Returns
        -------
        asymmetries : list
            Empty if no asymmetries are found if asymmetries, then a
            ``list`` of ``(i,j)`` tuples is returned.

        Examples
        --------

        >>> from libpysal.weights import lat2W
        >>> w=lat2W(3,3)
        >>> w.asymmetry()
        []
        >>> w.transform='r'
        >>> w.asymmetry()
        [(0, 1), (0, 3), (1, 0), (1, 2), (1, 4), (2, 1), (2, 5), (3, 0), (3, 4), (3, 6), (4, 1), (4, 3), (4, 5), (4, 7), (5, 2), (5, 4), (5, 8), (6, 3), (6, 7), (7, 4), (7, 6), (7, 8), (8, 5), (8, 7)]
        >>> result = w.asymmetry(intrinsic=False)
        >>> result
        []
        >>> neighbors={0:[1,2,3], 1:[1,2,3], 2:[0,1], 3:[0,1]}
        >>> weights={0:[1,1,1], 1:[1,1,1], 2:[1,1], 3:[1,1]}
        >>> w=W(neighbors,weights)
        >>> w.asymmetry()
        [(0, 1), (1, 0)]
        """

        if intrinsic:
            sp = self.sparse
            wd = sp.transpose() - sp
        else:
            transform = self.transform
            self.transform = "b"
            wd = sp.transpose() - sp
            self.transform = transform

        ids = np.nonzero(wd)
        if len(ids[0]) == 0:
            return []
        else:
            ijs = list(zip(ids[0], ids[1]))
            ijs.sort()
            return ijs

    def symmetrize(self, inplace=False):
        """Construct a symmetric KNN weight. This ensures that the neighbors
        of each focal observation consider the focal observation itself as
        a neighbor. This returns a generic ``W`` object, since the object is no
        longer guaranteed to have ``k`` neighbors for each observation.
        """
        neighbors = copy.deepcopy(self.neighbors)
        weights = copy.deepcopy(self.weights)
        for focal, fneighbs in list(self.neighbors.items()):
            for j, neighbor in enumerate(fneighbs):
                neighb_neighbors = self.neighbors[neighbor]
                if focal not in neighb_neighbors:
                    self.neighbors[neighbor].append(focal)
                    self.weights[neighbor].append(self.weights[focal][j])
        out_W = W(neighbors, weights)
        return out_W

    def full(self):
        """Generate a full ``numpy.ndarray``.

        Parameters
        ----------
        self : libpysal.weights.W
            spatial weights object

        Returns
        -------
        (fullw, keys) : tuple
            The first element being the full ``numpy.ndarray`` and second
            element keys being the ids associated with each row in the array.

        Examples
        --------
        >>> from libpysal.weights import W, full
        >>> neighbors = {'first':['second'],'second':['first','third'],'third':['second']}
        >>> weights = {'first':[1],'second':[1,1],'third':[1]}
        >>> w = W(neighbors, weights)
        >>> wf, ids = full(w)
        >>> wf
        array([[0., 1., 0.],
               [1., 0., 1.],
               [0., 1., 0.]])
        >>> ids
        ['first', 'second', 'third']
        """
        wfull = np.array(self.sparse.todense())
        keys = self.ids

        return (wfull, keys)

    def to_WSP(self):
        """Generate a ``WSP`` object.

        Returns
        -------

        implicit : libpysal.weights.WSP
            Thin ``W`` class

        Examples
        --------
        >>> from libpysal.weights import W, WSP
        >>> neighbors={'first':['second'],'second':['first','third'],'third':['second']}
        >>> weights={'first':[1],'second':[1,1],'third':[1]}
        >>> w=W(neighbors,weights)
        >>> wsp=w.to_WSP()
        >>> isinstance(wsp, WSP)
        True
        >>> wsp.n
        3
        >>> wsp.s0
        4

        See also
        --------
        WSP

        """
        return WSP(self.sparse, self.ids)

    def set_shapefile(self, shapefile, idVariable=None, full=False):
        """
        Adding metadata for writing headers of ``.gal`` and ``.gwt`` files.

        Parameters
        ----------
        shapefile : str
            The shapefile name used to construct weights.
        idVariable : str
            The name of the attribute in the shapefile to associate
            with ids in the weights.
        full : bool
            Write out the entire path for a shapefile (``True``) or
            only the base of the shapefile without extension (``False``).
            Default is ``True``.
        """

        if full:
            self._shpName = shapefile
        else:
            self._shpName = BASENAME(shapefile).split(".")[0]

        self._varName = idVariable

    def plot(
        self, gdf, indexed_on=None, ax=None, color="k", node_kws=None, edge_kws=None
    ):
        """Plot spatial weights objects. **Requires** ``matplotlib``, and
        implicitly requires a ``geopandas.GeoDataFrame`` as input.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            The original shapes whose topological relations are modelled in ``W``.
        indexed_on : str
            Column of ``geopandas.GeoDataFrame`` that the weights object uses
            as an index. Default is ``None``, so the index of the
            ``geopandas.GeoDataFrame`` is used.
        ax : matplotlib.axes.Axes
            Axis on which to plot the weights. Default is ``None``, so
            plots on the current figure.
        color : str
            ``matplotlib`` color string, will color both nodes and edges
            the same by default.
        node_kws : dict
            Keyword arguments dictionary to send to ``pyplot.scatter``,
            which provides fine-grained control over the aesthetics
            of the nodes in the plot.
        edge_kws : dict
            Keyword arguments dictionary to send to ``pyplot.plot``,
            which provides fine-grained control over the aesthetics
            of the edges in the plot.

        Returns
        -------
        f : matplotlib.figure.Figure
            Figure on which the plot is made.
        ax : matplotlib.axes.Axes
            Axis on which the plot is made.

        Notes
        -----
        If you'd like to overlay the actual shapes from the
        ``geopandas.GeoDataFrame``, call ``gdf.plot(ax=ax)`` after this.
        To plot underneath, adjust the z-order of the plot as follows:
        ``gdf.plot(ax=ax,zorder=0)``.

        Examples
        --------

        >>> from libpysal.weights import Queen
        >>> import libpysal as lp
        >>> import geopandas
        >>> gdf = geopandas.read_file(lp.examples.get_path("columbus.shp"))
        >>> weights = Queen.from_dataframe(gdf)
        >>> tmp = weights.plot(gdf, color='firebrickred', node_kws=dict(marker='*', color='k'))
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "W.plot depends on matplotlib.pyplot, and this was"
                "not able to be imported. \nInstall matplotlib to"
                "plot spatial weights."
            )
        if ax is None:
            f = plt.figure()
            ax = plt.gca()
        else:
            f = plt.gcf()
        if node_kws is not None:
            if "color" not in node_kws:
                node_kws["color"] = color
        else:
            node_kws = dict(color=color)
        if edge_kws is not None:
            if "color" not in edge_kws:
                edge_kws["color"] = color
        else:
            edge_kws = dict(color=color)

        for idx, neighbors in self:
            if idx in self.islands:
                continue
            if indexed_on is not None:
                neighbors = gdf[gdf[indexed_on].isin(neighbors)].index.tolist()
                idx = gdf[gdf[indexed_on] == idx].index.tolist()[0]
            centroids = gdf.loc[neighbors].centroid.apply(lambda p: (p.x, p.y))
            centroids = np.vstack(centroids.values)
            focal = np.hstack(gdf.loc[idx].geometry.centroid.xy)
            seen = set()
            for nidx, neighbor in zip(neighbors, centroids):
                if (idx, nidx) in seen:
                    continue
                ax.plot(*list(zip(focal, neighbor)), marker=None, **edge_kws)
                seen.update((idx, nidx))
                seen.update((nidx, idx))
        ax.scatter(
            gdf.centroid.apply(lambda p: p.x),
            gdf.centroid.apply(lambda p: p.y),
            **node_kws,
        )
        return f, ax


class WSP(object):
    """Thin ``W`` class for ``spreg``.

    Parameters
    ----------

    sparse : scipy.sparse.{matrix-type}
        NxN object from ``scipy.sparse``

    Attributes
    ----------

    n           : int
                  description
    s0          : float
                  description
    trcWtW_WW   : float
                  description

    Examples
    --------

    From GAL information

    >>> import scipy.sparse
    >>> from libpysal.weights import WSP
    >>> rows = [0, 1, 1, 2, 2, 3]
    >>> cols = [1, 0, 2, 1, 3, 3]
    >>> weights =  [1, 0.75, 0.25, 0.9, 0.1, 1]
    >>> sparse = scipy.sparse.csr_matrix((weights, (rows, cols)), shape=(4,4))
    >>> w = WSP(sparse)
    >>> w.s0
    4.0
    >>> w.trcWtW_WW
    6.395
    >>> w.n
    4

    """

    def __init__(self, sparse, ids=None, id_order=None, index=None):
        if not scipy.sparse.issparse(sparse):
            raise ValueError("must pass a scipy sparse object")
        rows, cols = sparse.shape
        if rows != cols:
            raise ValueError("Weights object must be square")
        self.sparse = sparse.tocsr()
        self.n = sparse.shape[0]
        if id_order:
            self._ids = id_order
        if ids:
            if len(ids) != self.n:
                raise ValueError(
                    "Number of values in id_order must match shape of sparse"
                )
            self._ids = ids
        else:
            self._ids = list(range(self.n))
        # temp addition of index attribute

        if index is not None:
            if not isinstance(index, (pd.Index, pd.MultiIndex, pd.RangeIndex)):
                raise TypeError("index must be an instance of pandas.Index dtype")
            if len(index) != self.n:
                raise ValueError("Number of values in index must match shape of sparse")
        else:
            index = pd.RangeIndex(self.n)
        self.index = index

    @cached_property
    def ids(self):
        """An ordered list of ids, assumed to match the ordering in ``sparse``."""
        # Temporary solution until the refactoring is finished
        return self._ids

    @cached_property
    def s0(self):
        r"""``s0`` is defined as:

        .. math::

               s0=\sum_i \sum_j w_{i,j}

        """
        return self.sparse.sum()

    @cached_property
    def trcWtW_WW(self):
        """Trace of :math:`W^{'}W + WW`."""
        return self.diagWtW_WW.sum()

    @cached_property
    def diagWtW_WW(self):
        """Diagonal of :math:`W^{'}W + WW`."""
        wt = self.sparse.transpose()
        w = self.sparse
        diagWtW_WW = (wt * w + w * w).diagonal()
        return diagWtW_WW

    @classmethod
    def from_W(cls, W):
        """Constructs a ``WSP`` object from the ``W``'s sparse matrix.

        Parameters
        ----------
        W : libpysal.weights.W
            A PySAL weights object with a sparse form and ids.

        Returns
        -------
        A ``WSP`` instance.
        """
        return cls(W.sparse, ids=W.ids)

    def to_W2(self, silence_warnings=False):
        """
        Convert a pysal WSP object (thin weights matrix) to a pysal W object.

        Parameters
        ----------
        self : WSP
            PySAL sparse weights object.
        silence_warnings : bool
            Switch to ``True`` to turn off print statements for every
            observation with islands. Default is ``False``, which does
            not silence warnings.

        Returns
        -------
        w : W
            PySAL weights object.

        Examples
        --------
        >>> from libpysal.weights import lat2SW, WSP, WSP2W

        Build a 10x10 ``scipy.sparse`` matrix for a rectangular 2x5
        region of cells (rook contiguity), then construct a ``libpysal``
        sparse weights object (``self``).

        >>> sp = lat2SW(2, 5)
        >>> self = WSP(sp)
        >>> self.n
        10
        >>> print(self.sparse[0].todense())
        [[0 1 0 0 0 1 0 0 0 0]]

        Convert this sparse weights object to a standard PySAL weights object.

        >>> w = WSP2W(self)
        >>> w.n
        10
        >>> print(w.full()[0][0])
        [0. 1. 0. 0. 0. 1. 0. 0. 0. 0.]

        """
        indices = list(self.sparse.indices)
        data = list(self.sparse.data)
        indptr = list(self.sparse.indptr)
        id_order = self.ids
        if id_order:
            # replace indices with user IDs
            indices = [id_order[i] for i in indices]
        else:
            id_order = list(range(self.n))
        neighbors, weights = {}, {}
        start = indptr[0]
        for i in range(self.n):
            oid = id_order[i]
            end = indptr[i + 1]
            neighbors[oid] = indices[start:end]
            weights[oid] = data[start:end]
            start = end
        w = W(neighbors, weights, ids=id_order, silence_warnings=silence_warnings)
        return w

    def to_W(self, silence_warnings=False):
        """
        Convert a pysal WSP object (thin weights matrix) to a pysal W object.

        Parameters
        ----------
        self : WSP
            PySAL sparse weights object.
        silence_warnings : bool
            Switch to ``True`` to turn off print statements for every
            observation with islands. Default is ``False``, which does
            not silence warnings.

        Returns
        -------
        w : W
            PySAL weights object.

        Examples
        --------
        >>> from libpysal.weights import lat2SW, WSP, WSP2W

        Build a 10x10 ``scipy.sparse`` matrix for a rectangular 2x5
        region of cells (rook contiguity), then construct a ``libpysal``
        sparse weights object (``self``).

        >>> sp = lat2SW(2, 5)
        >>> self = WSP(sp)
        >>> self.n
        10
        >>> print(self.sparse[0].todense())
        [[0 1 0 0 0 1 0 0 0 0]]

        Convert this sparse weights object to a standard PySAL weights object.

        >>> w = WSP2W(self)
        >>> w.n
        10
        >>> print(w.full()[0][0])
        [0. 1. 0. 0. 0. 1. 0. 0. 0. 0.]

        """

        indices = list(self.sparse.indices)
        data = list(self.sparse.data)
        indptr = list(self.sparse.indptr)
        id_order = self.ids
        if id_order:
            # replace indices with user IDs
            indices = [id_order[i] for i in indices]
        else:
            id_order = list(range(self.n))
        neighbors, weights = {}, {}
        start = indptr[0]
        for i in range(self.n):
            oid = id_order[i]
            end = indptr[i + 1]
            neighbors[oid] = indices[start:end]
            weights[oid] = data[start:end]
            start = end
        w = W(neighbors, weights, ids=id_order, silence_warnings=silence_warnings)
        return w

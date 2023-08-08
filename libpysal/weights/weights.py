"""
Weights.
"""
__author__ = "Sergio J. Rey <srey@asu.edu>"

import copy
from os.path import basename as BASENAME
import math
import warnings
import numpy as np
import scipy.sparse
from scipy.sparse.csgraph import connected_components
from collections import defaultdict


# from .util import full, WSP2W resolve import cycle by
# forcing these into methods
from . import adjtools
from ..io.fileio import FileIO as popen

__all__ = ["W", "WSP"]


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
        self.transformations = {}
        self.neighbors = neighbors
        if not weights:
            weights = {}
            for key in neighbors:
                weights[key] = [1.0] * len(neighbors[key])
        self.weights = weights
        self.transformations["O"] = self.weights.copy()  # original weights
        self.transform = "O"
        if id_order is None:
            self._id_order = list(self.neighbors.keys())
            self._id_order.sort()
            self._id_order_set = False
        else:
            self._id_order = id_order
            self._id_order_set = True
        self._reset()
        self._n = len(self.weights)
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

    def _reset(self):
        """Reset properties."""
        self._cache = {}

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
        id_order = WSP.id_order
        if id_order:
            # replace indices with user IDs
            indices = [id_order[i] for i in WSP.sparse.indices]
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
        ids = copy.copy(WSP.id_order)
        w = W(neighbors, weights, ids, silence_warnings=silence_warnings)
        w._sparse = copy.deepcopy(WSP.sparse)
        w._cache["sparse"] = w._sparse
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

        links = []
        focal_ix, neighbor_ix = self.sparse.nonzero()
        idxs = np.array(self.id_order)
        focal_ix = idxs[focal_ix]
        neighbor_ix = idxs[neighbor_ix]
        weights = self.sparse.data
        adjlist = pandas.DataFrame(
            {focal_col: focal_ix, neighbor_col: neighbor_ix, weight_col: weights}
        )
        if remove_symmetric:
            adjlist = adjtools.filter_adjlist(adjlist)
        if not drop_islands:
            island_adjlist = pandas.DataFrame(
                {focal_col: self.islands, neighbor_col: self.islands, weight_col: 0}
            )
            adjlist = pandas.concat((adjlist, island_adjlist)).reset_index(drop=True)
        if sort_joins:
            return adjlist.sort_values([focal_col, neighbor_col])
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
        if "sparse" not in self._cache:
            self._sparse = self._build_sparse()
            self._cache["sparse"] = self._sparse
        return self._sparse

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
        coo = sparse.tocoo()
        neighbors = defaultdict(list)
        weights = defaultdict(list)
        for k, v, w in zip(coo.row, coo.col, coo.data):
            neighbors[k].append(v)
            weights[k].append(w)
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
        disp = {}
        disp["bsr"] = scipy.sparse.bsr_array
        disp["coo"] = scipy.sparse.coo_array
        disp["csc"] = scipy.sparse.csc_array
        disp["csr"] = scipy.sparse.csr_array
        fmt_l = fmt.lower()
        if fmt_l in disp:
            adj_list = self.to_adjlist(drop_islands=False)
            data = adj_list.weight
            row = adj_list.focal
            col = adj_list.neighbor
            le = _LabelEncoder()
            le.fit(row)
            row = le.transform(row)
            col = le.transform(col)
            n = self.n
            return disp[fmt_l]((data, (row, col)), shape=(n, n))
        else:
            raise ValueError(f"unsupported sparse format: {fmt}")

    @property
    def n_components(self):
        """Store whether the adjacency matrix is fully connected."""
        if "n_components" not in self._cache:
            self._n_components, self._component_labels = connected_components(
                self.sparse
            )
            self._cache["n_components"] = self._n_components
            self._cache["component_labels"] = self._component_labels
        return self._n_components

    @property
    def component_labels(self):
        """Store the graph component in which each observation falls."""
        if "component_labels" not in self._cache:
            self._n_components, self._component_labels = connected_components(
                self.sparse
            )
            self._cache["n_components"] = self._n_components
            self._cache["component_labels"] = self._component_labels
        return self._component_labels

    def _build_sparse(self):
        """Construct the sparse attribute."""

        row = []
        col = []
        data = []
        id2i = self.id2i
        for i, neigh_list in list(self.neighbor_offsets.items()):
            card = self.cardinalities[i]
            row.extend([id2i[i]] * card)
            col.extend(neigh_list)
            data.extend(self.weights[i])
        row = np.array(row)
        col = np.array(col)
        data = np.array(data)
        s = scipy.sparse.csr_matrix((data, (row, col)), shape=(self.n, self.n))
        return s

    @property
    def id2i(self):
        """Dictionary where the key is an ID and the value is that ID's
        index in ``W.id_order``.
        """
        if "id2i" not in self._cache:
            self._id2i = {}
            for i, id_i in enumerate(self._id_order):
                self._id2i[id_i] = i
            self._id2i = self._id2i
            self._cache["id2i"] = self._id2i
        return self._id2i

    @property
    def n(self):
        """Number of units."""
        if "n" not in self._cache:
            self._n = len(self.neighbors)
            self._cache["n"] = self._n
        return self._n

    @property
    def s0(self):
        r"""``s0`` is defined as

        .. math::

               s0=\sum_i \sum_j w_{i,j}

        """
        if "s0" not in self._cache:
            self._s0 = self.sparse.sum()
            self._cache["s0"] = self._s0
        return self._s0

    @property
    def s1(self):
        r"""``s1`` is defined as

        .. math::

               s1=1/2 \sum_i \sum_j \Big(w_{i,j} + w_{j,i}\Big)^2

        """
        if "s1" not in self._cache:
            t = self.sparse.transpose()
            t = t + self.sparse
            t2 = t.multiply(t)  # element-wise square
            self._s1 = t2.sum() / 2.0
            self._cache["s1"] = self._s1
        return self._s1

    @property
    def s2array(self):
        """Individual elements comprising ``s2``.

        See Also
        --------
        s2

        """
        if "s2array" not in self._cache:
            s = self.sparse
            self._s2array = np.array(s.sum(1) + s.sum(0).transpose()) ** 2
            self._cache["s2array"] = self._s2array
        return self._s2array

    @property
    def s2(self):
        r"""``s2`` is defined as

        .. math::

                s2=\sum_j \Big(\sum_i w_{i,j} + \sum_i w_{j,i}\Big)^2

        """
        if "s2" not in self._cache:
            self._s2 = self.s2array.sum()
            self._cache["s2"] = self._s2
        return self._s2

    @property
    def trcW2(self):
        """Trace of :math:`WW`.

        See Also
        --------
        diagW2

        """
        if "trcW2" not in self._cache:
            self._trcW2 = self.diagW2.sum()
            self._cache["trcw2"] = self._trcW2
        return self._trcW2

    @property
    def diagW2(self):
        """Diagonal of :math:`WW`.

        See Also
        --------
        trcW2

        """
        if "diagw2" not in self._cache:
            self._diagW2 = (self.sparse * self.sparse).diagonal()
            self._cache["diagW2"] = self._diagW2
        return self._diagW2

    @property
    def diagWtW(self):
        """Diagonal of :math:`W^{'}W`.

        See Also
        --------
        trcWtW

        """
        if "diagWtW" not in self._cache:
            self._diagWtW = (self.sparse.transpose() * self.sparse).diagonal()
            self._cache["diagWtW"] = self._diagWtW
        return self._diagWtW

    @property
    def trcWtW(self):
        """Trace of :math:`W^{'}W`.

        See Also
        --------
        diagWtW

        """
        if "trcWtW" not in self._cache:
            self._trcWtW = self.diagWtW.sum()
            self._cache["trcWtW"] = self._trcWtW
        return self._trcWtW

    @property
    def diagWtW_WW(self):
        """Diagonal of :math:`W^{'}W + WW`."""
        if "diagWtW_WW" not in self._cache:
            wt = self.sparse.transpose()
            w = self.sparse
            self._diagWtW_WW = (wt * w + w * w).diagonal()
            self._cache["diagWtW_WW"] = self._diagWtW_WW
        return self._diagWtW_WW

    @property
    def trcWtW_WW(self):
        """Trace of :math:`W^{'}W + WW`."""
        if "trcWtW_WW" not in self._cache:
            self._trcWtW_WW = self.diagWtW_WW.sum()
            self._cache["trcWtW_WW"] = self._trcWtW_WW
        return self._trcWtW_WW

    @property
    def pct_nonzero(self):
        """Percentage of nonzero weights."""
        if "pct_nonzero" not in self._cache:
            self._pct_nonzero = 100.0 * self.sparse.nnz / (1.0 * self._n**2)
            self._cache["pct_nonzero"] = self._pct_nonzero
        return self._pct_nonzero

    @property
    def cardinalities(self):
        """Number of neighbors for each observation."""
        if "cardinalities" not in self._cache:
            c = {}
            for i in self._id_order:
                c[i] = len(self.neighbors[i])
            self._cardinalities = c
            self._cache["cardinalities"] = self._cardinalities
        return self._cardinalities

    @property
    def max_neighbors(self):
        """Largest number of neighbors."""
        if "max_neighbors" not in self._cache:
            self._max_neighbors = max(self.cardinalities.values())
            self._cache["max_neighbors"] = self._max_neighbors
        return self._max_neighbors

    @property
    def mean_neighbors(self):
        """Average number of neighbors."""
        if "mean_neighbors" not in self._cache:
            self._mean_neighbors = np.mean(list(self.cardinalities.values()))
            self._cache["mean_neighbors"] = self._mean_neighbors
        return self._mean_neighbors

    @property
    def min_neighbors(self):
        """Minimum number of neighbors."""
        if "min_neighbors" not in self._cache:
            self._min_neighbors = min(self.cardinalities.values())
            self._cache["min_neighbors"] = self._min_neighbors
        return self._min_neighbors

    @property
    def nonzero(self):
        """Number of nonzero weights."""
        if "nonzero" not in self._cache:
            self._nonzero = self.sparse.nnz
            self._cache["nonzero"] = self._nonzero
        return self._nonzero

    @property
    def sd(self):
        """Standard deviation of number of neighbors."""
        if "sd" not in self._cache:
            self._sd = np.std(list(self.cardinalities.values()))
            self._cache["sd"] = self._sd
        return self._sd

    @property
    def asymmetries(self):
        """List of id pairs with asymmetric weights
        sorted in ascending *index location* order.
        """
        if "asymmetries" not in self._cache:
            self._asymmetries = self.asymmetry()
            self._cache["asymmetries"] = self._asymmetries
        return self._asymmetries

    @property
    def islands(self):
        """List of ids without any neighbors."""
        if "islands" not in self._cache:
            self._islands = [i for i, c in list(self.cardinalities.items()) if c == 0]
            self._cache["islands"] = self._islands
        return self._islands

    @property
    def histogram(self):
        """Cardinality histogram as a dictionary where key is the id and
        value is the number of neighbors for that unit.
        """
        if "histogram" not in self._cache:
            ct, bin = np.histogram(
                list(self.cardinalities.values()),
                list(range(self.min_neighbors, self.max_neighbors + 2)),
            )
            self._histogram = list(zip(bin, ct))
            self._cache["histogram"] = self._histogram
        return self._histogram

    def __getitem__(self, key):
        """Allow a dictionary like interaction with the weights class.

        Examples
        --------
        >>> from libpysal.weights import lat2W
        >>> w = lat2W()

        >>> w[0] == dict({1: 1.0, 5: 1.0})
        True
        """
        return dict(list(zip(self.neighbors[key], self.weights[key])))

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
        for i in self._id_order:
            yield i, dict(list(zip(self.neighbors[i], self.weights[i])))

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

        old_ids = self._id_order
        if len(old_ids) != len(new_ids):
            raise Exception(
                "W.remap_ids: length of `old_ids` does not match             that of"
                " new_ids"
            )
        if len(set(new_ids)) != len(new_ids):
            raise Exception("W.remap_ids: list `new_ids` contains duplicates")
        else:
            new_neighbors = {}
            new_weights = {}
            old_transformations = self.transformations["O"].copy()
            new_transformations = {}
            for o, n in zip(old_ids, new_ids):
                o_neighbors = self.neighbors[o]
                o_weights = self.weights[o]
                n_neighbors = [new_ids[old_ids.index(j)] for j in o_neighbors]
                new_neighbors[n] = n_neighbors
                new_weights[n] = o_weights[:]
                new_transformations[n] = old_transformations[o]
            self.neighbors = new_neighbors
            self.weights = new_weights
            self.transformations["O"] = new_transformations

            id_order = [self._id_order.index(o) for o in old_ids]
            for i, id_ in enumerate(id_order):
                self.id_order[id_] = new_ids[i]

            self._reset()

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
        >>> w.id_order
        [0, 1, 2, 3, 4, 5, 6, 7, 8]
        >>> w.id_order=range(8,-1,-1)
        >>> list(w.id_order)
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

        if set(self._id_order) == set(ordered_ids):
            self._id_order = ordered_ids
            self._id_order_set = True
            self._reset()
        else:
            raise Exception("ordered_ids do not align with W ids")

    def __get_id_order(self):
        """Returns the ids for the observations in the order in which they
        would be encountered if iterating over the weights.
        """
        return self._id_order

    id_order = property(__get_id_order, __set_id_order)

    @property
    def id_order_set(self):
        """Returns ``True`` if user has set ``id_order``, ``False`` if not.

        Examples
        --------
        >>> from libpysal.weights import lat2W
        >>> w=lat2W()
        >>> w.id_order_set
        True
        """
        return self._id_order_set

    @property
    def neighbor_offsets(self):
        """
        Given the current ``id_order``, ``neighbor_offsets[id]`` is the
        offsets of the id's neighbors in ``id_order``.

        Returns
        -------
        neighbor_list : list
            Offsets of the id's neighbors in ``id_order``.

        Examples
        --------
        >>> from libpysal.weights import W
        >>> neighbors={'c': ['b'], 'b': ['c', 'a'], 'a': ['b']}
        >>> weights ={'c': [1.0], 'b': [1.0, 1.0], 'a': [1.0]}
        >>> w=W(neighbors,weights)
        >>> w.id_order = ['a','b','c']
        >>> w.neighbor_offsets['b']
        [2, 0]
        >>> w.id_order = ['b','a','c']
        >>> w.neighbor_offsets['b']
        [2, 1]
        """

        if "neighbors_0" not in self._cache:
            self.__neighbors_0 = {}
            id2i = self.id2i
            for j, neigh_list in list(self.neighbors.items()):
                self.__neighbors_0[j] = [id2i[neigh] for neigh in neigh_list]
            self._cache["neighbors_0"] = self.__neighbors_0

        neighbor_list = self.__neighbors_0

        return neighbor_list

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
        if value in self.transformations:
            self.weights = self.transformations[value]
            self._reset()
        else:
            if value == "R":
                # row standardized weights
                weights = {}
                self.weights = self.transformations["O"]
                for i in self.weights:
                    wijs = self.weights[i]
                    row_sum = sum(wijs) * 1.0
                    if row_sum == 0.0:
                        if not self.silence_warnings:
                            print(("WARNING: ", i, " is an island (no neighbors)"))
                    weights[i] = [wij / row_sum for wij in wijs]
                weights = weights
                self.transformations[value] = weights
                self.weights = weights
                self._reset()
            elif value == "D":
                # doubly-standardized weights
                # update current chars before doing global sum
                self._reset()
                s0 = self.s0
                ws = 1.0 / s0
                weights = {}
                self.weights = self.transformations["O"]
                for i in self.weights:
                    wijs = self.weights[i]
                    weights[i] = [wij * ws for wij in wijs]
                weights = weights
                self.transformations[value] = weights
                self.weights = weights
                self._reset()
            elif value == "B":
                # binary transformation
                weights = {}
                self.weights = self.transformations["O"]
                for i in self.weights:
                    wijs = self.weights[i]
                    weights[i] = [1.0 for wij in wijs]
                weights = weights
                self.transformations[value] = weights
                self.weights = weights
                self._reset()
            elif value == "V":
                # variance stabilizing
                weights = {}
                q = {}
                k = self.cardinalities
                s = {}
                Q = 0.0
                self.weights = self.transformations["O"]
                for i in self.weights:
                    wijs = self.weights[i]
                    q[i] = math.sqrt(sum([wij * wij for wij in wijs]))
                    s[i] = [wij / q[i] for wij in wijs]
                    Q += sum([si for si in s[i]])
                nQ = self.n / Q
                for i in self.weights:
                    weights[i] = [w * nQ for w in s[i]]
                weights = weights
                self.transformations[value] = weights
                self.weights = weights
                self._reset()
            elif value == "O":
                # put weights back to original transformation
                weights = {}
                original = self.transformations[value]
                self.weights = original
                self._reset()
            else:
                raise Exception("unsupported weights transformation")

    transform = property(get_transform, set_transform)

    def asymmetry(self, intrinsic=True):
        r"""
        Asymmetry check.

        Parameters
        ----------

        intrinsic : bool
            Default is ``True``. Intrinsic symmetry is defined as:

            .. math::

                w_{i,j} == w_{j,i}

            If ``intrinsic`` is ``False`` symmetry is defined as:

            .. math::

                i \in N_j \ \& \ j \in N_i

            where :math:`N_j` is the set of neighbors for :math:`j`.

        Returns
        -------

        asymmetries : list
            Empty if no asymmetries are found. If there are asymmetries,
            then a ``list`` of ``(i,j)`` tuples is returned sorted in
            ascending *index location* order.

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
            wd = self.sparse.transpose() - self.sparse
        else:
            transform = self.transform
            self.transform = "b"
            wd = self.sparse.transpose() - self.sparse
            self.transform = transform

        ids = np.nonzero(wd)

        if len(ids[0]) == 0:
            return []
        else:
            ijs = list(zip(ids[0], ids[1]))
            ijs.sort()

            i2id = {v: k for k, v in self.id2i.items()}
            ijs = [(i2id[i], i2id[j]) for i, j in ijs]

            return ijs

    def symmetrize(self, inplace=False):
        """Construct a symmetric KNN weight. This ensures that the neighbors
        of each focal observation consider the focal observation itself as
        a neighbor. This returns a generic ``W`` object, since the object is no
        longer guaranteed to have ``k`` neighbors for each observation.
        """
        if not inplace:
            neighbors = copy.deepcopy(self.neighbors)
            weights = copy.deepcopy(self.weights)
            out_W = W(neighbors, weights, id_order=self.id_order)
            out_W.symmetrize(inplace=True)
            return out_W
        else:
            for focal, fneighbs in list(self.neighbors.items()):
                for j, neighbor in enumerate(fneighbs):
                    neighb_neighbors = self.neighbors[neighbor]
                    if focal not in neighb_neighbors:
                        self.neighbors[neighbor].append(focal)
                        self.weights[neighbor].append(self.weights[focal][j])
            self._cache = dict()
            return

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
        wfull = self.sparse.toarray()
        keys = list(self.neighbors.keys())
        if self.id_order:
            keys = self.id_order

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
        return WSP(self.sparse, self._id_order)

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

        for idx, neighbors in self.neighbors.items():
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

    def __init__(self, sparse, id_order=None, index=None):
        if not scipy.sparse.issparse(sparse):
            raise ValueError("must pass a scipy sparse object")
        rows, cols = sparse.shape
        if rows != cols:
            raise ValueError("Weights object must be square")
        self.sparse = sparse.tocsr()
        self.n = sparse.shape[0]
        self._cache = {}
        if id_order:
            if len(id_order) != self.n:
                raise ValueError(
                    "Number of values in id_order must match shape of sparse"
                )
            else:
                self._id_order = id_order
                self._cache["id_order"] = self._id_order
        # temp addition of index attribute
        import pandas as pd  # will be removed after refactoring is done

        if index is not None:
            if not isinstance(index, (pd.Index, pd.MultiIndex, pd.RangeIndex)):
                raise TypeError("index must be an instance of pandas.Index dtype")
            if len(index) != self.n:
                raise ValueError("Number of values in index must match shape of sparse")
        else:
            index = pd.RangeIndex(self.n)
        self.index = index

    @property
    def id_order(self):
        """An ordered list of ids, assumed to match the ordering in ``sparse``."""
        # Temporary solution until the refactoring is finished
        if "id_order" not in self._cache:
            if hasattr(self, "index"):
                self._id_order = self.index.tolist()
            else:
                self._id_order = list(range(self.n))
            self._cache["id_order"] = self._id_order
        return self._id_order

    @property
    def s0(self):
        r"""``s0`` is defined as:

        .. math::

               s0=\sum_i \sum_j w_{i,j}

        """
        if "s0" not in self._cache:
            self._s0 = self.sparse.sum()
            self._cache["s0"] = self._s0
        return self._s0

    @property
    def trcWtW_WW(self):
        """Trace of :math:`W^{'}W + WW`."""
        if "trcWtW_WW" not in self._cache:
            self._trcWtW_WW = self.diagWtW_WW.sum()
            self._cache["trcWtW_WW"] = self._trcWtW_WW
        return self._trcWtW_WW

    @property
    def diagWtW_WW(self):
        """Diagonal of :math:`W^{'}W + WW`."""
        if "diagWtW_WW" not in self._cache:
            wt = self.sparse.transpose()
            w = self.sparse
            self._diagWtW_WW = (wt * w + w * w).diagonal()
            self._cache["diagWtW_WW"] = self._diagWtW_WW
        return self._diagWtW_WW

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
        return cls(W.sparse, id_order=W.id_order)

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
        id_order = self.id_order
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
        ids = copy.copy(self.id_order)
        w = W(neighbors, weights, ids, silence_warnings=silence_warnings)
        w._sparse = copy.deepcopy(self.sparse)
        w._cache["sparse"] = w._sparse
        return w

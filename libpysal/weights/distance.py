__all__ = ["KNN", "Kernel", "DistanceBand"]
__author__ = "Sergio J. Rey <srey@asu.edu>, Levi John Wolf <levi.john.wolf@gmail.com>"


from ..cg.kdtree import KDTree
from .weights import W, WSP
from .util import (
    isKDTree,
    get_ids,
    get_points_array_from_shapefile,
    get_points_array,
    WSP2W,
)
import copy
from warnings import warn as Warn
from scipy.spatial import distance_matrix
import scipy.sparse as sp
import numpy as np


def knnW(data, k=2, p=2, ids=None, radius=None, distance_metric="euclidean"):
    """
    This is deprecated. Use the pysal.weights.KNN class instead.
    """
    # Warn('This function is deprecated. Please use pysal.weights.KNN', UserWarning)
    return KNN(data, k=k, p=p, ids=ids, radius=radius, distance_metric=distance_metric)


class KNN(W):
    """
    Creates nearest neighbor weights matrix based on k nearest
    neighbors.

    Parameters
    ----------
    kdtree      : object
                  PySAL KDTree or ArcKDTree where KDtree.data is array (n,k)
                  n observations on k characteristics used to measure
                  distances between the n objects
    k           : int
                  number of nearest neighbors
    p           : float
                  Minkowski p-norm distance metric parameter:
                  1<=p<=infinity
                  2: Euclidean distance
                  1: Manhattan distance
                  Ignored if the KDTree is an ArcKDTree
    ids         : list
                  identifiers to attach to each observation

    Returns
    -------

    w         : W
                instance
                Weights object with binary weights

    Examples
    --------
    >>> import libpysal
    >>> import numpy as np
    >>> points = [(10, 10), (20, 10), (40, 10), (15, 20), (30, 20), (30, 30)]
    >>> kd = libpysal.cg.KDTree(np.array(points))
    >>> wnn2 = libpysal.weights.KNN(kd, 2)
    >>> [1,3] == wnn2.neighbors[0]
    True
    >>> wnn2 = KNN(kd,2)
    >>> wnn2[0]
    {1: 1.0, 3: 1.0}
    >>> wnn2[1]
    {0: 1.0, 3: 1.0}

    now with 1 rather than 0 offset

    >>> wnn2 = libpysal.weights.KNN(kd, 2, ids=range(1,7))
    >>> wnn2[1]
    {2: 1.0, 4: 1.0}
    >>> wnn2[2]
    {1: 1.0, 4: 1.0}
    >>> 0 in wnn2.neighbors
    False

    Notes
    -----

    Ties between neighbors of equal distance are arbitrarily broken.

    Further, if many points occupy the same spatial location (i.e. observations are
    coincident), then you may need to increase k for those observations to
    acquire neighbors at different spatial locations. For example, if five
    points are coincident, then their four nearest neighbors will all
    occupy the same spatial location; only the fifth nearest neighbor will
    result in those coincident points becoming connected to the graph as a
    whole.

    Solutions to this problem include jittering the points (by adding
    a small random value to each observation's location) or by adding
    higher-k neighbors only to the coincident points, using the
    weights.w_sets.w_union() function.

    See Also
    --------
    :class:`libpysal.weights.weights.W`
    """

    def __init__(
        self,
        data,
        k=2,
        p=2,
        ids=None,
        radius=None,
        distance_metric="euclidean",
        **kwargs
    ):
        if radius is not None:
            distance_metric = "arc"
        if isKDTree(data):
            self.kdtree = data
            self.data = self.kdtree.data
        else:
            self.kdtree = KDTree(data, radius=radius, distance_metric=distance_metric)
            self.data = self.kdtree.data
        self.k = k
        self.p = p

        # these are both n x k+1
        distances, indices = self.kdtree.query(self.data, k=k + 1, p=p)
        full_indices = np.arange(self.kdtree.n)

        # if an element in the indices matrix is equal to the corresponding
        # index for that row, we want to mask that site from its neighbors
        not_self_mask = indices != full_indices.reshape(-1, 1)
        # if there are *too many duplicates per site*, then we may get some
        # rows where the site index is not in the set of k+1 neighbors
        # So, we need to know where these sites are
        has_one_too_many = not_self_mask.sum(axis=1) == (k + 1)
        # if a site has k+1 neighbors, drop its k+1th neighbor
        not_self_mask[has_one_too_many, -1] &= False
        not_self_indices = indices[not_self_mask].reshape(self.kdtree.n, -1)

        to_weight = not_self_indices
        if ids is None:
            ids = list(full_indices)
            named_indices = not_self_indices
        else:
            named_indices = np.asarray(ids)[not_self_indices]
        neighbors = {idx: list(indices) for idx, indices in zip(ids, named_indices)}

        W.__init__(self, neighbors, id_order=ids, **kwargs)

    @classmethod
    def from_shapefile(cls, filepath, *args, **kwargs):
        """
        Nearest neighbor weights from a shapefile.

        Parameters
        ----------

        data       : string
                     shapefile containing attribute data.
        k          : int
                     number of nearest neighbors
        p          : float
                     Minkowski p-norm distance metric parameter:
                     1<=p<=infinity
                     2: Euclidean distance
                     1: Manhattan distance
        ids        : list
                     identifiers to attach to each observation
        radius     : float
                     If supplied arc_distances will be calculated
                     based on the given radius. p will be ignored.

        Returns
        -------

        w         : KNN
                    instance; Weights object with binary weights.

        Examples
        --------

        Polygon shapefile
        >>> import libpysal
        >>> from libpysal.weights import KNN
        >>> wc=KNN.from_shapefile(libpysal.examples.get_path("columbus.shp"))
        >>> "%.4f"%wc.pct_nonzero
        '4.0816'
        >>> set([2,1]) == set(wc.neighbors[0])
        True
        >>> wc3=KNN.from_shapefile(libpysal.examples.get_path("columbus.shp"),k=3)
        >>> set(wc3.neighbors[0]) == set([2,1,3])
        True
        >>> set(wc3.neighbors[2]) == set([4,3,0])
        True


        Point shapefile

        >>> w=KNN.from_shapefile(libpysal.examples.get_path("juvenile.shp"))
        >>> w.pct_nonzero
        1.1904761904761905
        >>> w1=KNN.from_shapefile(libpysal.examples.get_path("juvenile.shp"),k=1)
        >>> "%.3f"%w1.pct_nonzero
        '0.595'

        Notes
        -----

        Ties between neighbors of equal distance are arbitrarily broken.

        See Also
        --------
        :class:`libpysal.weights.weights.W`
        """
        return cls(get_points_array_from_shapefile(filepath), *args, **kwargs)

    @classmethod
    def from_array(cls, array, *args, **kwargs):
        """
        Creates nearest neighbor weights matrix based on k nearest
        neighbors.

        Parameters
        ----------
        array       : np.ndarray
                      (n, k) array representing n observations on
                      k characteristics used to measure distances
                      between the n objects
        **kwargs    : keyword arguments, see Rook

        Returns
        -------
        w         : W
                    instance
                    Weights object with binary weights

        Examples
        --------
        >>> from libpysal.weights import KNN
        >>> points = [(10, 10), (20, 10), (40, 10), (15, 20), (30, 20), (30, 30)]
        >>> wnn2 = KNN.from_array(points, 2)
        >>> [1,3] == wnn2.neighbors[0]
        True
        >>> wnn2 = KNN.from_array(points,2)
        >>> wnn2[0]
        {1: 1.0, 3: 1.0}
        >>> wnn2[1]
        {0: 1.0, 3: 1.0}

        now with 1 rather than 0 offset

        >>> wnn2 = KNN.from_array(points, 2, ids=range(1,7))
        >>> wnn2[1]
        {2: 1.0, 4: 1.0}
        >>> wnn2[2]
        {1: 1.0, 4: 1.0}
        >>> 0 in wnn2.neighbors
        False

        Notes
        -----

        Ties between neighbors of equal distance are arbitrarily broken.

        See Also
        --------
        :class:`libpysal.weights.weights.W`
        """
        return cls(array, *args, **kwargs)

    @classmethod
    def from_dataframe(
        cls, df, geom_col=None, ids=None, use_index=True, *args, **kwargs
    ):
        """
        Make KNN weights from a dataframe.

        Parameters
        ----------
        df      :   pandas.dataframe
                    a dataframe with a geometry column that can be used to
                    construct a W object
        geom_col :  string
                    the name of the column in `df` that contains the
                    geometries. Defaults to active geometry column.
        ids     :   list-like, string
                    a list-like of ids to use to index the spatial weights object or
                    the name of the column to use as IDs. If nothing is
                    provided, the dataframe index is used if `use_index=True` or
                    a positional index is used if `use_index=False`.
                    Order of the resulting W is not respected from this list.
        use_index   : bool
                    use index of `df` as `ids` to index the spatial weights object.

        See Also
        --------
        :class:`libpysal.weights.weights.W`
        """
        if geom_col is None:
            geom_col = df.geometry.name
        pts = get_points_array(df[geom_col])
        if ids is None and use_index:
            ids = df.index.tolist()
        elif isinstance(ids, str):
            ids = df[ids].tolist()
        return cls(pts, *args, ids=ids, **kwargs)

    def reweight(self, k=None, p=None, new_data=None, new_ids=None, inplace=True):
        """
        Redo K-Nearest Neighbor weights construction using given parameters

        Parameters
        ----------
        new_data    : np.ndarray
                      an array containing additional data to use in the KNN
                      weight
        new_ids     : list
                      a list aligned with new_data that provides the ids for
                      each new observation
        inplace     : bool
                      a flag denoting whether to modify the KNN object
                      in place or to return a new KNN object
        k           : int
                      number of nearest neighbors
        p           : float
                      Minkowski p-norm distance metric parameter:
                      1<=p<=infinity
                      2: Euclidean distance
                      1: Manhattan distance
                      Ignored if the KDTree is an ArcKDTree

        Returns
        -------
        A copy of the object using the new parameterization, or None if the
        object is reweighted in place.
        """

        if new_data is not None:
            new_data = np.asarray(new_data).reshape(-1, 2)
            data = np.vstack((self.data, new_data)).reshape(-1, 2)
            if new_ids is not None:
                ids = copy.deepcopy(self.id_order)
                ids.extend(list(new_ids))
            else:
                ids = list(range(data.shape[0]))
        elif (new_data is None) and (new_ids is None):
            # If not, we can use the same kdtree we have
            data = self.kdtree
            ids = self.id_order
        elif (new_data is None) and (new_ids is not None):
            Warn("Remapping ids must be done using w.remap_ids")
        if k is None:
            k = self.k
        if p is None:
            p = self.p
        if inplace:
            self._reset()
            self.__init__(data, ids=ids, k=k, p=p)
        else:
            return KNN(data, ids=ids, k=k, p=p)


class Kernel(W):
    """
    Spatial weights based on kernel functions.

    Parameters
    ----------

    data        : array
                  (n,k) or KDTree where KDtree.data is array (n,k)
                  n observations on k characteristics used to measure
                  distances between the n objects
    bandwidth   : float
                  or array-like (optional)
                  the bandwidth :math:`h_i` for the kernel.
    fixed       : binary
                  If true then :math:`h_i=h \\forall i`. If false then
                  bandwidth is adaptive across observations.
    k           : int
                  the number of nearest neighbors to use for determining
                  bandwidth. For fixed bandwidth, :math:`h_i=max(dknn) \\forall i`
                  where :math:`dknn` is a vector of k-nearest neighbor
                  distances (the distance to the kth nearest neighbor for each
                  observation).  For adaptive bandwidths, :math:`h_i=dknn_i`
    diagonal    : boolean
                  If true, set diagonal weights = 1.0, if false (default),
                  diagonals weights are set to value according to kernel
                  function.
    function    : {'triangular','uniform','quadratic','quartic','gaussian'}
                  kernel function defined as follows with

                  .. math::

                      z_{i,j} = d_{i,j}/h_i

                  triangular

                  .. math::

                      K(z) = (1 - |z|) \\ if |z| \\le 1

                  uniform

                  .. math::

                      K(z) = 1/2 \\ if |z| \\le 1

                  quadratic

                  .. math::

                      K(z) = (3/4)(1-z^2) \\ if |z| \\le 1

                  quartic

                  .. math::

                      K(z) = (15/16)(1-z^2)^2 \\ if |z| \\le 1

                  gaussian

                  .. math::

                      K(z) = (2\\pi)^{(-1/2)} exp(-z^2 / 2)

    eps         : float
                  adjustment to ensure knn distance range is closed on the
                  knnth observations

    Attributes
    ----------
    weights : dict
              Dictionary keyed by id with a list of weights for each neighbor

    neighbors : dict
                of lists of neighbors keyed by observation id

    bandwidth : array
                array of bandwidths

    Examples
    --------
    >>> from libpysal.weights import Kernel
    >>> points=[(10, 10), (20, 10), (40, 10), (15, 20), (30, 20), (30, 30)]
    >>> kw=Kernel(points)
    >>> kw.weights[0]
    [1.0, 0.500000049999995, 0.4409830615267465]
    >>> kw.neighbors[0]
    [0, 1, 3]
    >>> kw.bandwidth
    array([[20.000002],
           [20.000002],
           [20.000002],
           [20.000002],
           [20.000002],
           [20.000002]])
    >>> kw15=Kernel(points,bandwidth=15.0)
    >>> kw15[0]
    {0: 1.0, 1: 0.33333333333333337, 3: 0.2546440075000701}
    >>> kw15.neighbors[0]
    [0, 1, 3]
    >>> kw15.bandwidth
    array([[15.],
           [15.],
           [15.],
           [15.],
           [15.],
           [15.]])

    Adaptive bandwidths user specified

    >>> bw=[25.0,15.0,25.0,16.0,14.5,25.0]
    >>> kwa=Kernel(points,bandwidth=bw)
    >>> kwa.weights[0]
    [1.0, 0.6, 0.552786404500042, 0.10557280900008403]
    >>> kwa.neighbors[0]
    [0, 1, 3, 4]
    >>> kwa.bandwidth
    array([[25. ],
           [15. ],
           [25. ],
           [16. ],
           [14.5],
           [25. ]])

    Endogenous adaptive bandwidths

    >>> kwea=Kernel(points,fixed=False)
    >>> kwea.weights[0]
    [1.0, 0.10557289844279438, 9.99999900663795e-08]
    >>> kwea.neighbors[0]
    [0, 1, 3]
    >>> kwea.bandwidth
    array([[11.18034101],
           [11.18034101],
           [20.000002  ],
           [11.18034101],
           [14.14213704],
           [18.02775818]])

    Endogenous adaptive bandwidths with Gaussian kernel

    >>> kweag=Kernel(points,fixed=False,function='gaussian')
    >>> kweag.weights[0]
    [0.3989422804014327, 0.2674190291577696, 0.2419707487162134]
    >>> kweag.bandwidth
    array([[11.18034101],
           [11.18034101],
           [20.000002  ],
           [11.18034101],
           [14.14213704],
           [18.02775818]])

    Diagonals to 1.0

    >>> kq = Kernel(points,function='gaussian')
    >>> kq.weights
    {0: [0.3989422804014327, 0.35206533556593145, 0.3412334260702758], 1: [0.35206533556593145, 0.3989422804014327, 0.2419707487162134, 0.3412334260702758, 0.31069657591175387], 2: [0.2419707487162134, 0.3989422804014327, 0.31069657591175387], 3: [0.3412334260702758, 0.3412334260702758, 0.3989422804014327, 0.3011374490937829, 0.26575287272131043], 4: [0.31069657591175387, 0.31069657591175387, 0.3011374490937829, 0.3989422804014327, 0.35206533556593145], 5: [0.26575287272131043, 0.35206533556593145, 0.3989422804014327]}
    >>> kqd = Kernel(points, function='gaussian', diagonal=True)
    >>> kqd.weights
    {0: [1.0, 0.35206533556593145, 0.3412334260702758], 1: [0.35206533556593145, 1.0, 0.2419707487162134, 0.3412334260702758, 0.31069657591175387], 2: [0.2419707487162134, 1.0, 0.31069657591175387], 3: [0.3412334260702758, 0.3412334260702758, 1.0, 0.3011374490937829, 0.26575287272131043], 4: [0.31069657591175387, 0.31069657591175387, 0.3011374490937829, 1.0, 0.35206533556593145], 5: [0.26575287272131043, 0.35206533556593145, 1.0]}

    """

    def __init__(
        self,
        data,
        bandwidth=None,
        fixed=True,
        k=2,
        function="triangular",
        eps=1.0000001,
        ids=None,
        diagonal=False,
        distance_metric="euclidean",
        radius=None,
        **kwargs
    ):
        if radius is not None:
            distance_metric = "arc"
        if isKDTree(data):
            self.kdtree = data
            self.data = self.kdtree.data
            data = self.data
        else:
            self.kdtree = KDTree(data, distance_metric=distance_metric, radius=radius)
            self.data = self.kdtree.data
        self.k = k + 1
        self.function = function.lower()
        self.fixed = fixed
        self.eps = eps
        if bandwidth:
            try:
                bandwidth = np.array(bandwidth)
                bandwidth.shape = (len(bandwidth), 1)
            except:
                bandwidth = np.ones((len(data), 1), "float") * bandwidth
            self.bandwidth = bandwidth
        else:
            self._set_bw()

        self._eval_kernel()
        neighbors, weights = self._k_to_W(ids)
        if diagonal:
            for i in neighbors:
                weights[i][neighbors[i].index(i)] = 1.0
        W.__init__(self, neighbors, weights, ids, **kwargs)

    @classmethod
    def from_shapefile(cls, filepath, idVariable=None, **kwargs):
        """
        Kernel based weights from shapefile

        Parameters
        ----------
        shapefile   : string
                      shapefile name with shp suffix
        idVariable  : string
                      name of column in shapefile's DBF to use for ids

        Returns
        -------
        Kernel Weights Object

        See Also
        --------
        :class:`libpysal.weights.weights.W`
        """
        points = get_points_array_from_shapefile(filepath)
        if idVariable is not None:
            ids = get_ids(filepath, idVariable)
        else:
            ids = None
        return cls.from_array(points, ids=ids, **kwargs)

    @classmethod
    def from_array(cls, array, **kwargs):
        """
        Construct a Kernel weights from an array. Supports all the same options
        as :class:`libpysal.weights.Kernel`

        See Also
        --------
        :class:`libpysal.weights.weights.W`
        """
        return cls(array, **kwargs)

    @classmethod
    def from_dataframe(cls, df, geom_col=None, ids=None, use_index=True, **kwargs):
        """
        Make Kernel weights from a dataframe.

        Parameters
        ----------
        df      :   pandas.dataframe
                    a dataframe with a geometry column that can be used to
                    construct a W object
        geom_col :  string
                    the name of the column in `df` that contains the
                    geometries. Defaults to active geometry column.
        ids     :   list-like, string
                    a list-like of ids to use to index the spatial weights object or
                    the name of the column to use as IDs. If nothing is
                    provided, the dataframe index is used if `use_index=True` or
                    a positional index is used if `use_index=False`.
                    Order of the resulting W is not respected from this list.
        use_index   : bool
                    use index of `df` as `ids` to index the spatial weights object.

        See Also
        --------
        :class:`libpysal.weights.weights.W`
        """
        if geom_col is None:
            geom_col = df.geometry.name
        pts = get_points_array(df[geom_col])
        if ids is None and use_index:
            ids = df.index.tolist()
        elif isinstance(ids, str):
            ids = df[ids].tolist()
        return cls(pts, ids=ids, **kwargs)

    def _k_to_W(self, ids=None):
        allneighbors = {}
        weights = {}
        if ids:
            ids = np.array(ids)
        else:
            ids = np.arange(len(self.data))
        for i, neighbors in enumerate(self.kernel):
            if len(self.neigh[i]) == 0:
                allneighbors[ids[i]] = []
                weights[ids[i]] = []
            else:
                allneighbors[ids[i]] = list(ids[self.neigh[i]])
                weights[ids[i]] = self.kernel[i].tolist()
        return allneighbors, weights

    def _set_bw(self):
        dmat, neigh = self.kdtree.query(self.data, k=self.k)
        if self.fixed:
            # use max knn distance as bandwidth
            bandwidth = dmat.max() * self.eps
            n = len(dmat)
            self.bandwidth = np.ones((n, 1), "float") * bandwidth
        else:
            # use local max knn distance
            self.bandwidth = dmat.max(axis=1) * self.eps
            self.bandwidth.shape = (self.bandwidth.size, 1)
            # identify knn neighbors for each point
            nnq = self.kdtree.query(self.data, k=self.k)
            self.neigh = nnq[1]

    def _eval_kernel(self):
        # get points within bandwidth distance of each point
        if not hasattr(self, "neigh"):
            kdtq = self.kdtree.query_ball_point
            neighbors = [
                kdtq(self.data[i], r=bwi[0]) for i, bwi in enumerate(self.bandwidth)
            ]
            self.neigh = neighbors
        # get distances for neighbors
        bw = self.bandwidth

        kdtq = self.kdtree.query
        z = []
        for i, nids in enumerate(self.neigh):
            di, ni = kdtq(self.data[i], k=len(nids))
            if not isinstance(di, np.ndarray):
                di = np.asarray([di] * len(nids))
                ni = np.asarray([ni] * len(nids))
            zi = np.array([dict(list(zip(ni, di)))[nid] for nid in nids]) / bw[i]
            z.append(zi)
        zs = z
        # functions follow Anselin and Rey (2010) table 5.4
        if self.function == "triangular":
            self.kernel = [1 - zi for zi in zs]
        elif self.function == "uniform":
            self.kernel = [np.ones(zi.shape) * 0.5 for zi in zs]
        elif self.function == "quadratic":
            self.kernel = [(3.0 / 4) * (1 - zi**2) for zi in zs]
        elif self.function == "quartic":
            self.kernel = [(15.0 / 16) * (1 - zi**2) ** 2 for zi in zs]
        elif self.function == "gaussian":
            c = np.pi * 2
            c = c ** (-0.5)
            self.kernel = [c * np.exp(-(zi**2) / 2.0) for zi in zs]
        else:
            print(("Unsupported kernel function", self.function))


class DistanceBand(W):
    """
    Spatial weights based on distance band.

    Parameters
    ----------

    data        : array
                  (n,k) or KDTree where KDtree.data is array (n,k)
                  n observations on k characteristics used to measure
                  distances between the n objects
    threshold  : float
                 distance band
    p          : float
                 DEPRECATED: use `distance_metric`
                 Minkowski p-norm distance metric parameter:
                 1<=p<=infinity
                 2: Euclidean distance
                 1: Manhattan distance
    binary     : boolean
                 If true w_{ij}=1 if d_{i,j}<=threshold, otherwise w_{i,j}=0
                 If false wij=dij^{alpha}
    alpha      : float
                 distance decay parameter for weight (default -1.0)
                 if alpha is positive the weights will not decline with
                 distance. If binary is True, alpha is ignored

    ids         : list
                  values to use for keys of the neighbors and weights dicts

    build_sp    : boolean
                  DEPRECATED
                  True to build sparse distance matrix and false to build dense
                  distance matrix; significant speed gains may be obtained
                  dending on the sparsity of the of distance_matrix and
                  threshold that is applied
    silent      : boolean
                  By default libpysal will print a warning if the
                  dataset contains any disconnected observations or
                  islands. To silence this warning set this
                  parameter to True.

    Attributes
    ----------
    weights : dict
              of neighbor weights keyed by observation id

    neighbors : dict
                of neighbors keyed by observation id

    Examples
    --------
    >>> import libpysal
    >>> points=[(10, 10), (20, 10), (40, 10), (15, 20), (30, 20), (30, 30)]
    >>> wcheck = libpysal.weights.W({0: [1, 3], 1: [0, 3], 2: [], 3: [0, 1], 4: [5], 5: [4]})

    WARNING: there is one disconnected observation (no neighbors)
    Island id:  [2]
    >>> w=libpysal.weights.DistanceBand(points,threshold=11.2)

    WARNING: there is one disconnected observation (no neighbors)
    Island id:  [2]
    >>> libpysal.weights.util.neighbor_equality(w, wcheck)
    True
    >>> w=libpysal.weights.DistanceBand(points,threshold=14.2)
    >>> wcheck = libpysal.weights.W({0: [1, 3], 1: [0, 3, 4], 2: [4], 3: [1, 0], 4: [5, 2, 1], 5: [4]})
    >>> libpysal.weights.util.neighbor_equality(w, wcheck)
    True

    inverse distance weights

    >>> w=libpysal.weights.DistanceBand(points,threshold=11.2,binary=False)

    WARNING: there is one disconnected observation (no neighbors)
    Island id:  [2]
    >>> w.weights[0]
    [0.1, 0.08944271909999159]
    >>> w.neighbors[0].tolist()
    [1, 3]

    gravity weights

    >>> w=libpysal.weights.DistanceBand(points,threshold=11.2,binary=False,alpha=-2.)

    WARNING: there is one disconnected observation (no neighbors)
    Island id:  [2]
    >>> w.weights[0]
    [0.01, 0.007999999999999998]


    """

    def __init__(
        self,
        data,
        threshold,
        p=2,
        alpha=-1.0,
        binary=True,
        ids=None,
        build_sp=True,
        silence_warnings=False,
        distance_metric="euclidean",
        radius=None,
    ):
        """Casting to floats is a work around for a bug in scipy.spatial.
        See detail in pysal issue #126.

        """
        if ids is not None:
            ids = list(ids)
        if radius is not None:
            distance_metric = "arc"
        self.p = p
        self.threshold = threshold
        self.binary = binary
        self.alpha = alpha
        self.build_sp = build_sp
        self.silence_warnings = silence_warnings

        if isKDTree(data):
            self.kdtree = data
            self.data = self.kdtree.data
        else:
            if self.build_sp:
                try:
                    data = np.asarray(data)
                    if data.dtype.kind != "f":
                        data = data.astype(float)
                    self.kdtree = KDTree(
                        data, distance_metric=distance_metric, radius=radius
                    )
                    self.data = self.kdtree.data
                except:
                    raise ValueError("Could not make array from data")
            else:
                self.data = data
                self.kdtree = None

        self._band()
        neighbors, weights = self._distance_to_W(ids)
        W.__init__(
            self, neighbors, weights, ids, silence_warnings=self.silence_warnings
        )

    @classmethod
    def from_shapefile(cls, filepath, threshold, idVariable=None, **kwargs):
        """
        Distance-band based weights from shapefile

        Parameters
        ----------
        shapefile   : string
                      shapefile name with shp suffix
        idVariable  : string
                      name of column in shapefile's DBF to use for ids

        Returns
        -------
        Kernel Weights Object

        """
        points = get_points_array_from_shapefile(filepath)
        if idVariable is not None:
            ids = get_ids(filepath, idVariable)
        else:
            ids = None
        return cls.from_array(points, threshold, ids=ids, **kwargs)

    @classmethod
    def from_array(cls, array, threshold, **kwargs):
        """
        Construct a DistanceBand weights from an array. Supports all the same options
        as :class:`libpysal.weights.DistanceBand`

        """
        return cls(array, threshold, **kwargs)

    @classmethod
    def from_dataframe(
        cls, df, threshold, geom_col=None, ids=None, use_index=True, **kwargs
    ):

        """
        Make DistanceBand weights from a dataframe.

        Parameters
        ----------
        df      :   pandas.dataframe
                    a dataframe with a geometry column that can be used to
                    construct a W object
        geom_col :  string
                    the name of the column in `df` that contains the
                    geometries. Defaults to active geometry column.
        ids     :   list-like, string
                    a list-like of ids to use to index the spatial weights object or
                    the name of the column to use as IDs. If nothing is
                    provided, the dataframe index is used if `use_index=True` or
                    a positional index is used if `use_index=False`.
                    Order of the resulting W is not respected from this list.
        use_index   : bool
                    use index of `df` as `ids` to index the spatial weights object.

        """
        if geom_col is None:
            geom_col = df.geometry.name
        pts = get_points_array(df[geom_col])
        if ids is None and use_index:
            ids = df.index.tolist()
        elif isinstance(ids, str):
            ids = df[ids].tolist()
        return cls(pts, threshold, ids=ids, **kwargs)

    def _band(self):
        """Find all pairs within threshold."""
        if self.build_sp:
            self.dmat = self.kdtree.sparse_distance_matrix(
                self.kdtree, max_distance=self.threshold, p=self.p
            ).tocsr()
        else:
            if str(self.kdtree).split(".")[-1][0:10] == "Arc_KDTree":
                raise TypeError(
                    "Unable to calculate dense arc distance matrix;"
                    ' parameter "build_sp" must be set to True for arc'
                    " distance type weight"
                )
            self.dmat = self._spdistance_matrix(self.data, self.data, self.threshold)

    def _distance_to_W(self, ids=None):
        if self.binary:
            self.dmat[self.dmat > 0] = 1
            self.dmat.eliminate_zeros()
            tempW = WSP2W(
                WSP(self.dmat, id_order=ids), silence_warnings=self.silence_warnings
            )
            neighbors = tempW.neighbors
            weight_keys = list(tempW.weights.keys())
            weight_vals = list(tempW.weights.values())
            weights = dict(list(zip(weight_keys, list(map(list, weight_vals)))))
            return neighbors, weights
        else:
            weighted = self.dmat.power(self.alpha)
            weighted[weighted == np.inf] = 0
            weighted.eliminate_zeros()
            tempW = WSP2W(
                WSP(weighted, id_order=ids), silence_warnings=self.silence_warnings
            )
            neighbors = tempW.neighbors
            weight_keys = list(tempW.weights.keys())
            weight_vals = list(tempW.weights.values())
            weights = dict(list(zip(weight_keys, list(map(list, weight_vals)))))
            return neighbors, weights

    def _spdistance_matrix(self, x, y, threshold=None):
        dist = distance_matrix(x, y)
        if threshold is not None:
            zeros = dist > threshold
            dist[zeros] = 0
        return sp.csr_matrix(dist)


def _test():
    import doctest

    # the following line could be used to define an alternative to the '<BLANKLINE>' flag
    # doctest.BLANKLINE_MARKER = 'something better than <BLANKLINE>'
    start_suppress = np.get_printoptions()["suppress"]
    np.set_printoptions(suppress=True)
    doctest.testmod()
    np.set_printoptions(suppress=start_suppress)


if __name__ == "__main__":
    _test()

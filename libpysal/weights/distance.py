__all__ = ["KNN", "Kernel", "DistanceBand"]

__author__ = (
    "Sergio J. Rey <sjsrey@gmail.com>, Levi John Wolf <levi.john.wolf@gmail.com>"
)


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
    """This is deprecated. Use the ``libpysal.weights.KNN`` class instead."""
    # Warn("This function is deprecated. Please use libpysal.weights.KNN", UserWarning)
    return KNN(data, k=k, p=p, ids=ids, radius=radius, distance_metric=distance_metric)


class KNN(W):
    """Creates nearest neighbor weights matrix based on `k` nearest neighbors.

    Parameters
    ----------
    data : {libpysal.cg.KDTree, libpysal.cg.ArcKDTree}
        An ``(n,k)`` array of `n` observations on `k` characteristics
        used to measure distances between the `n` objects.
    k : int
        The number of nearest neighbors. Default is ``2``.
    p : {int, float}
        Minkowski `p`-norm distance metric parameter where
        :math:`1<=\mathtt{p}<=\infty`. ``2`` is Euclidean distance and
        ``1`` is Manhattan distance. This parameter is ignored if the
        ``KDTree`` is an ``ArcKDTree``. Default is ``2``.
    ids : list
        Identifiers to attach to each observation. Default is ``None``.
    radius : float
        If supplied arc distances will be calculated based on the given radius
        and ``p`` will be ignored. Default is ``None``.
        See ``libpysal.cg.KDTree`` for more details.
    distance_metric : str
        Either ``'euclidean'`` or ``'arc'``. Default is ``'euclidean'``.
        See ``libpysal.cg.KDTree`` for more details.
    **kwargs : dict
        Keyword arguments for ``libpysal.weights.W``.
    
    Returns
    -------
    w : libpysal.weights.KNN
        A `k` nearest neighbors weights instance.

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

    Now with 1 rather than 0 offset:

    >>> wnn2 = libpysal.weights.KNN(kd, 2, ids=range(1, 7))
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
    
    libpysal.weights.W
    
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

        this_nnq = self.kdtree.query(self.data, k=k + 1, p=p)

        to_weight = this_nnq[1]

        if ids is None:
            ids = list(range(to_weight.shape[0]))

        neighbors = {}
        for i, row in enumerate(to_weight):
            row = row.tolist()
            row.remove(i)
            row = [ids[j] for j in row]
            focal = ids[i]
            neighbors[focal] = row

        W.__init__(self, neighbors, id_order=ids, **kwargs)

    @classmethod
    def from_shapefile(cls, filepath, *args, **kwargs):
        """Nearest neighbor weights from a shapefile.

        Parameters
        ----------
        filepath : str
            The name of polygon shapefile (including the file extension)
            containing attribute data.
        *args : iterable
            Positional arguments for ``libpysal.weights.KNN``.
        **kwargs : dict
            Keyword arguments for ``libpysal.weights.KNN``.

        Returns
        -------
        w : libpysal.weights.KNN
            A `k` nearest neighbors weights instance.

        Examples
        --------

        From a polygon shapefile:
        
        >>> import libpysal
        >>> from libpysal.weights import KNN
        >>> wc = KNN.from_shapefile(libpysal.examples.get_path("columbus.shp"))
        >>> "%.4f"%wc.pct_nonzero
        '4.0816'
        
        >>> set([2,1]) == set(wc.neighbors[0])
        True
        
        >>> wc3 = KNN.from_shapefile(libpysal.examples.get_path("columbus.shp"), k=3)
        >>> set(wc3.neighbors[0]) == set([2, 1, 3])
        True
        
        >>> set(wc3.neighbors[2]) == set([4, 3, 0])
        True

        From a point shapefile:

        >>> w = KNN.from_shapefile(libpysal.examples.get_path("juvenile.shp"))
        >>> w.pct_nonzero
        1.1904761904761905
        
        >>> w1 = KNN.from_shapefile(libpysal.examples.get_path("juvenile.shp"), k=1)
        >>> "%.3f"%w1.pct_nonzero
        '0.595'

        Notes
        -----

        Ties between neighbors of equal distance are arbitrarily broken.

        See Also
        --------
        
        libpysal.weights.W
        
        """

        w = cls(get_points_array_from_shapefile(filepath), *args, **kwargs)

        return w

    @classmethod
    def from_array(cls, array, *args, **kwargs):
        """Creates nearest neighbor weights matrix based on `k` nearest neighbors.

        Parameters
        ----------
        array : numpy.ndarray
            An ``(n, k)`` array representing `n` observations on `k`
            characteristics used to measure distances between the `n` objects.
        *args : iterable
            Positional arguments for ``libpysal.weights.KNN``.
        **kwargs : dict
            Keyword arguments for ``libpysal.weights.KNN``.

        Returns
        -------
        w : libpysal.weights.KNN
            A `k` nearest neighbors weights instance.

        Examples
        --------
        
        >>> from libpysal.weights import KNN
        >>> points = [(10, 10), (20, 10), (40, 10), (15, 20), (30, 20), (30, 30)]
        >>> wnn2 = KNN.from_array(points, 2)
        >>> [1,3] == wnn2.neighbors[0]
        True
        
        >>> wnn2 = KNN.from_array(points, 2)
        >>> wnn2[0]
        {1: 1.0, 3: 1.0}
        
        >>> wnn2[1]
        {0: 1.0, 3: 1.0}

        Now with 1 rather than 0 offset:

        >>> wnn2 = KNN.from_array(points, 2, ids=range(1, 7))
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
        
        libpysal.weights.W
        
        """

        w = cls(array, *args, **kwargs)

        return w

    @classmethod
    def from_dataframe(cls, df, geom_col="geometry", ids=None, *args, **kwargs):
        """Make `KNN` weights from a dataframe.

        Parameters
        ----------
        df : pandas.DataFrame
            A dataframe with a geometry column that can be used
            to construct a `W` object.
        geom_col : string
            The column name of the geometry stored in ``df``.
            Default is ``'geometry'``.
        ids : {str, iterable}
            If string, the column name of the indices from the dataframe.
            If iterable, a list of ids to use for the `W`.
            If ``None``, ``df.index`` is used. Default is ``None``.
        *args : iterable
            Positional arguments for ``libpysal.weights.KNN``.
        **kwargs : dict
            Keyword arguments for ``libpysal.weights.KNN``.

        Returns
        -------
        w : libpysal.weights.KNN
            A `k` nearest neighbors weights instance.
        
        See Also
        --------
        
        libpysal.weights.W
        
        """

        pts = get_points_array(df[geom_col])

        if ids is None:
            ids = df.index.tolist()
        elif isinstance(ids, str):
            ids = df[ids].tolist()

        w = cls(pts, *args, ids=ids, **kwargs)

        return w

    def reweight(self, k=None, p=None, new_data=None, new_ids=None, inplace=True):
        """Redo `K`-nearest neighbor weights construction using given parameters.

        Parameters
        ----------
        k : int
            The number of nearest neighbors. Default is ``None``.
        p : {int, float}
            Minkowski `p`-norm distance metric parameter where
            :math:`1<=\mathtt{p}<=\infty`. ``2`` is Euclidean distance and
            ``1`` is Manhattan distance. This parameter is ignored if the
            ``KDTree`` is an ``ArcKDTree``. Default is ``None``.
        new_data : numpy.ndarray
            An array containing additional data to use in the `KNN` weight.
            Default is ``None``.
        new_ids : list
            A list aligned with ``new_data`` that provides the ids
            for each new observation. Default is ``None``.
        inplace : bool
            A flag denoting whether to modify the `KNN` object 
            in place or to return a new `KNN` object.  Default is ``True``.
        
        Returns
        -------
        w : libpysal.weights.KNN
            A copy of the `k` nearest neighbors weights instance using the
            new parameterization, or ``None`` if the object is reweighted in place.

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
            w = KNN(data, ids=ids, k=k, p=p)

            return w


class Kernel(W):
    """Spatial weights based on kernel functions.

    Parameters
    ----------
    data : {libpysal.cg.KDTree, libpysal.cg.ArcKDTree}
        An :math:`(n,k)` array of :math:`n` observations on :math:`k`
        characteristics used to measure distances between the :math:`n` objects.
    k : int
        The number of nearest neighbors to use for determining the bandwidth. For a
        fixed bandwidth, :math:`h_i = max(dknn) \\forall i` where :math:`dknn` is a
        vector of :math:`k`-nearest neighbor distances (the distance to the
        :math:`k`th nearest neighbor for each observation). For adaptive bandwidths,
        :math:`h_i=dknn_i`. Default is ``2``.
    bandwidth : {float, array-like}
        The bandwidth :math:`h_i` for the kernel. Default is ``None``.
    fixed : bool
        If ``True`` then :math:`h_i = h \\forall i`. If ``False`` then
        bandwidth is adaptive across observations. Default is ``True``.
    diagonal : bool
        If ``True``, set diagonal weights to ``1.0``. If ``False`` diagonal weights
        are set to values according to the kernel function. Default is ``False``.
    eps : float
        The adjustment to ensure the `knn` distance range
        is closed on the `knn`th observations. Default is ``1.0000001``.
    ids : list
        Identifiers to attach to each observation. Default is ``None``.
    radius : float
        If supplied arc distances will be calculated based on the given radius
        and ``p`` will be ignored. Default is ``None``.
        See ``libpysal.cg.KDTree`` for more details.
    distance_metric : str
        Either ``'euclidean'`` or ``'arc'``. Default is ``'euclidean'``.
        See ``libpysal.cg.KDTree`` for more details.
    function : str
        Either ``'triangular'``, ``'uniform'``, ``'quadratic'``, ``'quartic'``,
        or ``'gaussian'``. Default is ``'triangular'``.
        The kernel function is defined as follows with

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

    **kwargs : dict
        Keyword arguments for ``libpysal.weights.W``.

    Attributes
    ----------
    weights : dict
        Dictionary keyed by id with a list of weights for each neighbor.
    neighbors : dict
        Lists of neighbors keyed by observation id.
    bandwidth : array-like
        An array of bandwidths.

    Examples
    --------
    
    >>> from libpysal.weights import Kernel
    >>> points = [(10, 10), (20, 10), (40, 10), (15, 20), (30, 20), (30, 30)]
    >>> kw = Kernel(points)
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
    
    >>> kw15 = Kernel(points,bandwidth=15.0)
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

    Adaptive bandwidths user specified:

    >>> bw = [25.0,15.0,25.0,16.0,14.5,25.0]
    >>> kwa = Kernel(points,bandwidth=bw)
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

    Endogenous adaptive bandwidths:

    >>> kwea = Kernel(points,fixed=False)
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

    Endogenous adaptive bandwidths with Gaussian kernel:

    >>> kweag = Kernel(points,fixed=False,function='gaussian')
    >>> kweag.weights[0]
    [0.3989422804014327, 0.2674190291577696, 0.2419707487162134]
    
    >>> kweag.bandwidth
    array([[11.18034101],
           [11.18034101],
           [20.000002  ],
           [11.18034101],
           [14.14213704],
           [18.02775818]])

    Diagonals to 1.0:

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
        k=2,
        bandwidth=None,
        fixed=True,
        diagonal=False,
        eps=1.0000001,
        ids=None,
        radius=None,
        distance_metric="euclidean",
        function="triangular",
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
        """Construct kernel-based weights from a shapefile.

        Parameters
        ----------
        filepath : str
            The name of polygon shapefile (including the file extension)
            containing attribute data.
        idVariable : str
            The name of the column in shapefile's DBF to use for ids.
            Default is ``None``.
        **kwargs : dict
            Keyword arguments for ``libpysal.weights.Kernel``.

        Returns
        -------
        w : libpysal.weights.Kernel
            A kernel weights instance.

        See Also
        ---------
        
        libpysal.weights.W
        
        """

        points = get_points_array_from_shapefile(filepath)

        if idVariable is not None:
            ids = get_ids(filepath, idVariable)
        else:
            ids = None

        w = cls.from_array(points, ids=ids, **kwargs)

        return w

    @classmethod
    def from_array(cls, array, **kwargs):
        """Construct kernel-based weights from an array.
        
        Parameters
        ----------
        array : numpy.ndarray
            An ``(n, k)`` array representing `n` observations on `k`
            characteristics used to measure distances between the `n` objects.
        **kwargs : dict
            Keyword arguments for ``libpysal.weights.Kernel``.
        
        Returns
        -------
        w : libpysal.weights.Kernel
            A kernel weights instance.
        
        See Also
        --------
        
        libpysal.weights.W
        
        """

        w = cls(array, **kwargs)

        return w

    @classmethod
    def from_dataframe(cls, df, geom_col="geometry", ids=None, **kwargs):
        """Construct kernel-based weights from a dataframe.

        Parameters
        ----------
        df : pandas.DataFrame
            A dataframe with a geometry column that can be used
            to construct a PySAL `W` object.
        geom_col : str
            The column name of the geometry stored in ``df``.
            Default is ``'geometry'``.
        ids : {str, iterable}
            If string, the column name of the indices from the dataframe.
            If iterable, a list of ids to use for the `W`.
            If ``None``, ``df.index`` is used. Default is ``None``.
        **kwargs : dict
            Keyword arguments for ``libpysal.weights.Kernel``.
        
        Returns
        -------
        w : libpysal.weights.Kernel
            A kernel weights instance.

        See Also
        --------
        
        libpysal.weights.W
        
        """

        pts = get_points_array(df[geom_col])

        if ids is None:
            ids = df.index.tolist()
        elif isinstance(ids, str):
            ids = df[ids].tolist()

        w = cls(pts, ids=ids, **kwargs)

        return w

    def _k_to_W(self, ids=None):
        """Internal method for converting `k` neighbors to weights.
        
        Parameters
        ----------
        ids : list
            See ``ids`` in ``libpysal.weights.Kernel``. Default is ``None``.
        
        Returns
        -------
        allneighbors : dict
            Index lookup of all neighbors.
        weights : dict
            Index lookup of neighbor weights.
        
        """

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
        """Internal method for setting binary weights."""

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
        """Internal method for evaluate the kernel function."""

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
            self.kernel = [(3.0 / 4) * (1 - zi ** 2) for zi in zs]
        elif self.function == "quartic":
            self.kernel = [(15.0 / 16) * (1 - zi ** 2) ** 2 for zi in zs]
        elif self.function == "gaussian":
            c = np.pi * 2
            c = c ** (-0.5)
            self.kernel = [c * np.exp(-(zi ** 2) / 2.0) for zi in zs]
        else:
            print(("Unsupported kernel function", self.function))


class DistanceBand(W):
    """Spatial weights based on distance band.

    Parameters
    ----------
    data : {array-like, libpysal.cg.KDTree}
        ``(n,k)`` or ``KDTree`` where ``KDtree.data`` is an ``(n,k)`` array 
        of `n` observations on `k` characteristics used to measure
        distances between the `n` objects.
    threshold : float
        The distance band.
    p : {int, float}
        Minkowski `p`-norm distance metric parameter where :math:`1<=\mathtt{p}<=\infty`.
        ``2`` is Euclidean distance and ``1`` is Manhattan distance.
        This parameter is ignored if the ``KDTree`` is an ``ArcKDTree``.
        Default is ``2``.
    binary : bool
        If set to ``True``, :math:`w_{ij}=1` if :math:`d_{i,j}<=\mathtt{threshold}`,
        otherwise :math:`w_{i,j}=0`. If set to ``False``,
        :math:`w_{ij}=d_{ij}^{\mathtt{alpha}}`. Default is ``True``.
    alpha : float
        The distance decay parameter for weights. Default is ``-1.0``.
        If ``alpha`` is positive the weights will not decline with distance.
        If ``binary`` is set to ``True``, ``alpha`` is ignored.
    ids : list
        Identifiers to attach to each observation. Default is ``None``.
    build_sp : boolean
        Set to ``True`` to build a sparse distance matrix and ``False`` to build dense
        distance matrix. Significant speed gains may be obtained depending on the
        sparsity of the of distance matrix and the ``threshold`` that is applied.
        Default is ``True``.
    silence_warnings : bool
        By default (``False``) libpysal will print a warning if the dataset contains any
        disconnected observations or islands. To silence this warning set to ``True``.
    radius : float
        If supplied arc distances will be calculated based on the given radius
        and ``p`` will be ignored. Default is ``None``.
        See ``libpysal.cg.KDTree`` for more details.
    distance_metric : str
        Either ``'euclidean'`` or ``'arc'``. Default is ``'euclidean'``.
        See ``libpysal.cg.KDTree`` for more details.

    Attributes
    ----------
    weights : dict
        Neighbor weights keyed by observation id.
    neighbors : dict
        Neighbors keyed by observation id.

    Raises
    ------
    Value Error
        An array was unable to be instantiated with ``data``.
    
    Examples
    --------
    
    >>> import libpysal
    >>> points = [(10, 10), (20, 10), (40, 10), (15, 20), (30, 20), (30, 30)]
    >>> wcheck = libpysal.weights.W(
    ...     {0: [1, 3], 1: [0, 3], 2: [], 3: [0, 1], 4: [5], 5: [4]}
    ... )
    UserWarning: The weights matrix is not fully connected: 
    There are 3 disconnected components.
    There is 1 island with id: 2.

    >>> w = libpysal.weights.DistanceBand(points, threshold=11.2)
    UserWarning: The weights matrix is not fully connected: 
    There are 3 disconnected components.
    There is 1 island with id: 2.

    >>> libpysal.weights.util.neighbor_equality(w, wcheck)
    True
    
    >>> w = libpysal.weights.DistanceBand(points, threshold=14.2)
    >>> wcheck = libpysal.weights.W(
    ...     {0: [1, 3], 1: [0, 3, 4], 2: [4], 3: [1, 0], 4: [5, 2, 1], 5: [4]}
    ... )
    >>> libpysal.weights.util.neighbor_equality(w, wcheck)
    True

    Inverse distance weights:

    >>> w = libpysal.weights.DistanceBand(points, threshold=11.2, binary=False)
    UserWarning: The weights matrix is not fully connected: 
    There are 3 disconnected components.
    There is 1 island with id: 2.
    
    >>> w.weights[0]
    [0.1, 0.08944271909999159]
    >>> w.neighbors[0].tolist()
    [1, 3]

    Gravity weights:

    >>> w = libpysal.weights.DistanceBand(points, threshold=11.2, binary=False, alpha=-2.)
    UserWarning: The weights matrix is not fully connected: 
    There are 3 disconnected components.
    There is 1 island with id: 2.
    
    >>> w.weights[0]
    [0.01, 0.007999999999999998]

    Notes
    -----

    This was initially implemented running ``scipy v0.8.0dev`` (in epd 6.1).
    Earlier versions of scipy (0.7.0) have a logic bug in ``scipy/sparse/dok.py``,
    so Serge changed line 221 of that file on sal-dev to fix the logic bug.

    """

    def __init__(
        self,
        data,
        threshold,
        p=2,
        binary=True,
        alpha=-1.0,
        ids=None,
        build_sp=True,
        silence_warnings=False,
        radius=None,
        distance_metric="euclidean",
    ):
        """Casting to floats is a work around for a bug in ``scipy.spatial``.
        See details in `pysal/pysal#126 <https://github.com/pysal/pysal/issues/126>`_.

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
                    raise ValueError("Could not make array from data.")
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
        """Construct a distance band weights object from a shapefile.

        Parameters
        ----------
        filepath : str
            The name of polygon shapefile (including the file extension)
            containing attribute data.
        threshold : float
            The distance band.
        idVariable : str
            The name of the column in shapefile's DBF to use for ids.
            Default is ``None``.
        **kwargs : dict
            Keyword arguments for ``libpysal.weights.DistanceBand``.

        Returns
        -------
        w : libpysal.weights.DistanceBand
            A distance band weights instance.

        """

        points = get_points_array_from_shapefile(filepath)

        if idVariable is not None:
            ids = get_ids(filepath, idVariable)
        else:
            ids = None

        w = cls.from_array(points, threshold, ids=ids, **kwargs)

        return w

    @classmethod
    def from_array(cls, array, threshold, **kwargs):
        """Construct a distance band weights object from an array.
        
        Parameters
        ----------
        array : numpy.ndarray
            An ``(n, k)`` array representing `n` observations on `k`
            characteristics used to measure distances between the `n` objects.
        threshold : float
            The distance band.
        **kwargs : dict
            Keyword arguments for ``libpysal.weights.DistanceBand``.
        
        Returns
        -------
        w : libpysal.weights.DistanceBand
            A distance band weights instance.
        
        """

        w = cls(array, threshold, **kwargs)

        return w

    @classmethod
    def from_dataframe(cls, df, threshold, geom_col="geometry", ids=None, **kwargs):
        """Construct a distance band weights object from a dataframe.

        Parameters
        ----------
        df : pandas.DataFrame
            A dataframe with a geometry column that can be used
            to construct a PySAL `W` object.
        threshold : float
            The distance band.
        geom_col : str
            The column name of the geometry stored in ``df``.
            Default is ``'geometry'``.
        ids : {str, iterable}
            If string, the column name of the indices from the dataframe.
            If iterable, a list of ids to use for the `W`.
            If ``None``, ``df.index`` is used. Default is ``None``.
        **kwargs : dict
            Keyword arguments for ``libpysal.weights.DistanceBand``.
        
        Returns
        -------
        w : libpysal.weights.DistanceBand
            A distance band weights instance.
        
        """

        pts = get_points_array(df[geom_col])

        if ids is None:
            ids = df.index.tolist()
        elif isinstance(ids, str):
            ids = df[ids].tolist()

        w = cls(pts, threshold, ids=ids, **kwargs)

        return w

    def _band(self):
        """Internal function for finding all pairs within the threshold."""

        if self.build_sp:
            self.dmat = self.kdtree.sparse_distance_matrix(
                self.kdtree, max_distance=self.threshold, p=self.p
            ).tocsr()
        else:
            if str(self.kdtree).split(".")[-1][0:10] == "Arc_KDTree":
                raise TypeError(
                    "Unable to calculate dense arc distance matrix;"
                    " parameter 'build_sp' must be set to True for arc"
                    " distance type weight."
                )
            self.dmat = self._spdistance_matrix(self.data, self.data, self.threshold)

    def _distance_to_W(self, ids=None):
        """Internal method for converting distance band neighbors to weights.
        
        Parameters
        ----------
        ids : list
            See ``ids`` in ``libpysal.weights.DistanceBand``. Default is ``None``.
        
        Returns
        -------
        neighbors : dict
            Index lookup of all neighbors.
        weights : dict
            Index lookup of neighbor weights.
        
        """

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
        """Internal method for converting a distance matrix into a CSR matrix.
        
        Parameters
        ----------
        x : array-like
            X values.
        y : array-like
            Y values.
        threshold : float
            See ``threshold`` in ``DistanceBand``. Default is ``None``.
        
        Returns
        -------
        sp_mtx : scipy.sparse.csr_matrix
            A Compressed Sparse Row matrix.
        
        See Also
        --------
        
        scipy.spatial.distance_matrix
        scipy.sparse.csr_matrix
        
        """

        dist = distance_matrix(x, y)

        if threshold is not None:
            zeros = dist > threshold
            dist[zeros] = 0

        sp_mtx = sp.csr_matrix(dist)

        return sp_mtx


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

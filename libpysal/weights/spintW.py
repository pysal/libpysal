"""
Spatial weights for spatial interaction including contiguity OD weights (ODW),
network based weights (netW), and distance-decay based vector weights (vecW).
"""

__author__ = "Taylor Oshan  <tayoshan@gmail.com> "

from scipy.sparse import kron
from .weights import W, WSP
from .distance import DistanceBand
from collections import OrderedDict


def ODW(Wo, Wd, transform="r", silence_warnings=True):
    """Construct an :math:`(o \cdot d)\\times(o \cdot d)`
    origin-destination style spatial weight for :math:`o \cdot d`
    flows using standard spatial weights on :math:`o` origins
    and :math:`d` destinations. Input spatial weights must be
    binary or able to be sutiably transformed to binary.

    Parameters
    ----------

    Wo : libpysal.weights.W
        A `W` object for origin locations as a :math:`o \cdot o`
        spatial weight object amongst :math:`o` origins.
    Wd : libpysal.weights.W
        A `W` object for destination locations as a :math:`d \cdot d`
        spatial weight object amongst :math:`d` destinations
    transform : str
        A transformation for standardization of final the
        `OD` spatial weights. Default is ``'r'`` for row standardized.
    silence_warnings : bool
        By default (``True``) libpysal will silence a warning if the dataset contains any
        disconnected observations or islands. To print this warning set to ``False``.

    Returns
    -------

    Ww : libpysal.weights.WSP
        A sparse spatial contiguity `W` object for assocations between flows
        between :math:`o` origins and :math:`d` destinations,
        :math:`(o \cdot d)\\times(o \cdot d)`.

    Raises
    ------

    AttributeError
        The ``Wo`` argument is not binary.
    AttributeError
        The ``Wd`` argument is not binary.

    Examples
    --------

    >>> import libpysal
    >>> O = libpysal.weights.lat2W(2,2)
    >>> D = libpysal.weights.lat2W(2,2)
    >>> OD = libpysal.weights.ODW(O,D)
    >>> OD.weights[0]
    [0.25, 0.25, 0.25, 0.25]
    >>> OD.neighbors[0]
    [5, 6, 9, 10]
    >>> OD.full()[0][0]
    array([0.  , 0.  , 0.  , 0.  , 0.  , 0.25, 0.25, 0.  , 0.  , 0.25, 0.25,
           0.  , 0.  , 0.  , 0.  , 0.  ])

    """

    if Wo.transform != "b":
        try:
            Wo.tranform = "b"
        except:
            raise AttributeError(
                "Wo is not binary and cannot be transformed to "
                "binary. Wo must be binary or suitably transformed to binary."
            )

    if Wd.transform != "b":
        try:
            Wd.tranform = "b"
        except:
            raise AttributeError(
                "Wd is not binary and cannot be transformed to "
                "binary. Wd must be binary or suitably transformed to binary."
            )

    Wo = Wo.sparse
    Wo.eliminate_zeros()
    Wd = Wd.sparse
    Wd.eliminate_zeros()
    Ww = kron(Wo, Wd, format="csr")
    Ww.eliminate_zeros()
    Ww = WSP(Ww).to_W(silence_warnings=silence_warnings)
    Ww.transform = transform

    return Ww


def netW(link_list, share="A", transform="r", **kwargs):
    """Create a network-contiguity based weights object based
    on different nodal relationships encoded in a network.

    Parameters
    ----------

    link_list : list
        Collection of tuples where each ``tuple`` is of the form :math:`(o,d)`
        where :math:`o` is an origin id and :math:`d` is a destination id.
    share : str
        This denotes how to define the nodal relationship used to determine
        neighboring edges. The default is ``'A'``, for any shared nodes between
        two network edges; options include: ``'O'`` a shared origin node; ``'D'``
        a shared destination node; ``'OD'``; a shared origin or a shared
        destination node; ``'C'`` a shared node that is the destination of
        the first edge  and the origin of the second edge - i.e., a directed
        chain is formed moving from edge one to edge two.
    transform : str
        A transformation for standardization of final the
        `OD` spatial weights. Default is ``'r'`` for row standardized.
    **kwargs : dict
        Optional keyword arguments arguments for ``libpysal.weights.W``

    Returns
    -------

    netW : libpysal.weights.W
        A nodal contiguity `W` object for network edges or
        flows representing the binary adjacency of the network
        edges given a definition of nodal relationships.

    Raises
    ------

    AttributeError
        The ``share`` parameter must be ``'O'``, ``'D'``, ``'OD'``, or ``'C'``.

    Examples
    --------

    >>> import libpysal
    >>> links = [('a','b'), ('a','c'), ('a','d'), ('c','d'), ('c', 'b'), ('c','a')]
    >>> O = libpysal.weights.netW(links, share='O')
    >>> O.neighbors[('a', 'b')]
    [('a', 'c'), ('a', 'd')]
    >>> OD = libpysal.weights.netW(links, share='OD')
    >>> OD.neighbors[('a', 'b')]
    [('a', 'c'), ('a', 'd'), ('c', 'b')]
    >>> any_common = libpysal.weights.netW(links, share='A')
    >>> any_common.neighbors[('a', 'b')]
    [('a', 'c'), ('a', 'd'), ('c', 'b'), ('c', 'a')]

    """

    neighbors = {}
    neighbors = OrderedDict()
    edges = link_list

    for key in edges:
        neighbors[key] = []

        for neigh in edges:
            if key == neigh:
                continue
            if share.upper() == "OD":
                if key[0] == neigh[0] or key[1] == neigh[1]:
                    neighbors[key].append(neigh)
            elif share.upper() == "O":
                if key[0] == neigh[0]:
                    neighbors[key].append(neigh)
            elif share.upper() == "D":
                if key[1] == neigh[1]:
                    neighbors[key].append(neigh)
            elif share.upper() == "C":
                if key[1] == neigh[0]:
                    neighbors[key].append(neigh)
            elif share.upper() == "A":
                if (
                    key[0] == neigh[0]
                    or key[0] == neigh[1]
                    or key[1] == neigh[0]
                    or key[1] == neigh[1]
                ):
                    neighbors[key].append(neigh)
            else:
                raise AttributeError(
                    "Parameter 'share' must be 'O', 'D', 'OD', or 'C'."
                )

    netW = W(neighbors, **kwargs)
    netW.tranform = transform

    return netW


def vecW(
    origin_x,
    origin_y,
    dest_x,
    dest_y,
    threshold,
    p=2,
    alpha=-1.0,
    binary=True,
    ids=None,
    build_sp=False,
    **kwargs
):
    """Distance-based spatial weight for vectors that is computed using a
    4-dimensional distance between the origin x,y-coordinates and the
    destination x,y-coordinates.

    Parameters
    ----------

    origin_x : {list, numpy.ndarray}
        A vector of origin x-coordinates.
    origin_y : {list, numpy.ndarray}
        A vector of origin y-coordinates.
    dest_x : {list, numpy.ndarray}
        A vector of destination x-coordinates.
    dest_y : {list, numpy.ndarray}
        A vector of destination y-coordinates.
    threshold : float
        The distance band.
    p : {int, float}
        Minkowski `p`-norm distance metric parameter where :math:`1<=\mathtt{p}<=\infty`.
        ``2`` is Euclidean distance and ``1`` is Manhattan distance.
        This parameter is ignored if the ``KDTree`` is an ``ArcKDTree``.
        Default is ``2``.
    alpha : float
        The distance decay parameter for weights. Default is ``-1.0``.
        If ``alpha`` is positive the weights will not decline with distance.
        If ``binary`` is set to ``True``, ``alpha`` is ignored.
    binary : bool
        If set to ``True``, :math:`w_{ij}=1` if :math:`d_{i,j}<=\mathtt{threshold}`,
        otherwise :math:`w_{i,j}=0`. If set to ``False``,
        :math:`w_{ij}=d_{ij}^{\mathtt{alpha}}`. Default is ``True``.
    ids : list
        Identifiers to attach to each observation in ``neighbors``
        and ``weights``. Default is ``None``.
    build_sp : boolean
        Set to ``True`` to build a sparse distance matrix and ``False`` to build dense
        distance matrix. Significant speed gains may be obtained depending on the
        sparsity of the of distance matrix and the ``threshold`` that is applied.
        Default is ``True``.
    **kwargs : dict
        Optional keyword arguments arguments for ``libpysal.weights.W``.

    Returns
    -------

    w : libpysal.weights.DistanceBand
        A ``libpysal.weights.DistanceBand`` `W` object that uses 4-dimenional
        distances between vectors of origin and destination coordinates.

    Examples
    --------

    >>> import libpysal
    >>> x1 = [5,6,3]
    >>> y1 = [1,8,5]
    >>> x2 = [2,4,9]
    >>> y2 = [3,6,1]
    >>> W1 = libpysal.weights.vecW(x1, y1, x2, y2, threshold=999)
    >>> list(W1.neighbors[0])
    [1, 2]
    >>> W2 = libpysal.weights.vecW(x1, y2, x1, y2, threshold=8.5)
    >>> list(W2.neighbors[0])
    [1, 2]

    """

    data = list(zip(origin_x, origin_y, dest_x, dest_y))

    w = DistanceBand(
        data,
        threshold=threshold,
        p=p,
        binary=binary,
        alpha=alpha,
        ids=ids,
        build_sp=False,
        **kwargs
    )

    return w


def mat2L(edge_matrix):
    """Convert a matrix denoting network connectivity
    (edges or flows) to a list denoting edges.

    Parameters
    ----------

    edge_matrix : numpy.ndarray
        A matrix where rows denote network edge origins, columns denote
        network edge destinations, and non-zero entries denote the
        existence of an edge between a given origin and destination.

    Raises
    ------

    AttributeError
        The input matrix is not two dimensional.

    Returns
    -------

    edge_list : list
        Collection of tuples where each ``tuple`` is of the form :math:`(o,d)`
        where :math:`o` is an origin id and :math:`d` is a destination id.

    """

    if len(edge_matrix.shape) != 2:
        raise AttributeError(
            "Matrix of network edges should be two dimensions"
            "with edge origins on one axis and edge destinations on the"
            "second axis with non-zero matrix entires denoting an edge"
            "between and origin and destination."
        )
    edge_list = []
    rows, cols = edge_matrix.shape

    for row in range(rows):
        for col in range(cols):
            if edge_matrix[row, col] != 0:
                edge_list.append((row, col))

    return edge_list

"""Spatial lag operations.
"""

__author__ = (
    "Sergio J. Rey <sjsrey@gmail.com>,"
    "David C. Folch <David.Folch@nau.edu>,"
    "Levi John Wolf <levi.john.wolf@gmail.com>"
)

__all__ = ["lag_spatial", "lag_categorical"]

import numpy as np


def lag_spatial(w, y):
    """A spatial lag operator. If ``w`` is row standardized, this function
    returns the average of each observation's neighbors. If it is not, the
    weighted sum of each observation's neighbors is returned.

    Parameters
    ----------
    w : libpysal.weights.weights.W
        A PySAL spatial weights object.
    y : array-like
        A ``numpy`` array with dimensionality conforming to ``w`` (see examples).

    Returns
    -------
    wy : numpy.ndarray
        An array of numeric values for the spatial lag.

    Examples
    --------

    Setup a 9x9 binary spatial weights matrix and vector of data,
    then compute the spatial lag of the vector.

    >>> import libpysal
    >>> import numpy as np
    >>> w = libpysal.weights.lat2W(3, 3)
    >>> y = np.arange(9)
    >>> yl = libpysal.weights.lag_spatial(w, y)
    >>> yl
    array([ 4.,  6.,  6., 10., 16., 14., 10., 18., 12.])

    Row standardize the weights matrix and recompute the spatial lag.

    >>> w.transform = 'r'
    >>> yl = libpysal.weights.lag_spatial(w, y)
    >>> yl
    array([2.        , 2.        , 3.        , 3.33333333, 4.        ,
           4.66666667, 5.        , 6.        , 6.        ])


    Explicitly define data vector as 9x1 and recompute the spatial lag.

    >>> y.shape = (9, 1)
    >>> yl = libpysal.weights.lag_spatial(w, y)
    >>> yl
    array([[2.        ],
           [2.        ],
           [3.        ],
           [3.33333333],
           [4.        ],
           [4.66666667],
           [5.        ],
           [6.        ],
           [6.        ]])


    Take the spatial lag of a 9x2 data matrix.

    >>> yr = np.arange(8, -1, -1)
    >>> yr.shape = (9, 1)
    >>> x = np.hstack((y, yr))
    >>> yl = libpysal.weights.lag_spatial(w, x)
    >>> yl
    array([[2.        , 6.        ],
           [2.        , 6.        ],
           [3.        , 5.        ],
           [3.33333333, 4.66666667],
           [4.        , 4.        ],
           [4.66666667, 3.33333333],
           [5.        , 3.        ],
           [6.        , 2.        ],
           [6.        , 2.        ]])

    """

    return w.sparse * y


def lag_categorical(w, y, ties="tryself"):
    """A spatial lag operator for categorical variables. This function
    constructs the most common categories of neighboring observations
    weighted by their weight strength.

    Parameters
    ----------
    w : libpysal.weights.weights.W
        PySAL spatial weights object.
    y : iterable
        An iterable collection of categories (either ``int`` or ``str``)
        with dimensionality conforming to ``w`` (see examples).
    ties : str
        The method to use when resolving ties. By default, the option is
        ``'tryself'``, and the category of the focal observation is included
        with its neighbors to try and break a tie. If this does not resolve
        the tie, a winner is chosen randomly. To just use random choice to
        break ties, pass ``'random'`` instead. 
        The following are supported options
        
        * ``'tryself'`` -- Use the focal observation's label to tiebreak. If this doesn't successfully break the tie, which only occurs if it induces a new tie, decide randomly.;
        * ``'random'`` -- Resolve the tie randomly amongst winners.;
        * ``'lowest'`` -- Pick the lowest-value label amongst winners.;
        * ``'highest'`` -- Pick the highest-value label amongst winners.
    
    Returns
    -------
    output : numpy.ndarray
        An :math:`(n \cdot k)` column vector containing
        the most common neighboring observation.

    Notes
    -----
    
    This works on any array where the number of unique elements
    along the column axis is less than the number of elements in
    the array, for any ``dtype``. That means the routine should
    work on any ``dtype`` that ``numpy.unique()`` can compare.

    Examples
    --------

    Set up a 9x9 weights matrix describing a 3x3 regular lattice.
    Lag one list of categorical variables with no ties.

    >>> import libpysal
    >>> import numpy as np
    >>> np.random.seed(12345)
    >>> w = libpysal.weights.lat2W(3, 3)
    >>> y = ['a','b','a','b','c','b','c','b','c']
    >>> y_l = libpysal.weights.lag_categorical(w, y)
    >>> np.array_equal(y_l, np.array(['b', 'a', 'b', 'c', 'b', 'c', 'b', 'c', 'b']))
    True

    Explicitly reshape ``y`` into a (9x1) array and calculate lag again.

    >>> yvect = np.array(y).reshape(9,1)
    >>> yvect_l = libpysal.weights.lag_categorical(w,yvect)
    >>> check = np.array( [ [i] for i in  ['b', 'a', 'b', 'c', 'b', 'c', 'b', 'c', 'b']] )
    >>> np.array_equal(yvect_l, check)
    True

    Compute the lag of a 9x2 matrix of categories.

    >>> y2 = ['a', 'c', 'c', 'd', 'b', 'a', 'd', 'd', 'c']
    >>> ym = np.vstack((y,y2)).T
    >>> ym_lag = libpysal.weights.lag_categorical(w,ym)
    >>> check = np.array([['b', 'd'], ['a', 'c'], ['b', 'c'], ['c', 'd'], ['b', 'd'], ['c', 'c'], ['b', 'd'], ['c', 'd'], ['b', 'c']])
    >>> np.array_equal(check, ym_lag)
    True

    """

    if isinstance(y, list):
        y = np.array(y)
    orig_shape = y.shape

    if len(orig_shape) > 1:
        if orig_shape[1] > 1:
            output = np.vstack([lag_categorical(w, col) for col in y.T]).T
            return output

    y = y.flatten()
    output = np.zeros_like(y)
    labels = np.unique(y)
    normalized_labels = np.zeros(y.shape, dtype=np.int)

    for i, label in enumerate(labels):
        normalized_labels[y == label] = i
    for focal_name, neighbors in w:
        focal_idx = w.id2i[focal_name]
        neighborhood_tally = np.zeros(labels.shape)
        for neighb_name, weight in list(neighbors.items()):
            neighb_idx = w.id2i[neighb_name]
            neighb_label = normalized_labels[neighb_idx]
            neighborhood_tally[neighb_label] += weight
        out_label_idx = _resolve_ties(
            focal_idx, normalized_labels, neighborhood_tally, neighbors, ties, w
        )
        output[focal_idx] = labels[out_label_idx]

    output = output.reshape(orig_shape)

    return output


def _resolve_ties(idx, normalized_labels, tally, neighbors, method, w):
    """Helper function to resolve ties if lag is multimodal. First, if this function
    gets called when there's actually no tie, then the correct value will be picked.
    If ``'random'`` is selected as the method, a random tiebeaker is picked. If
    ``'tryself'`` is selected, then the observation's own value will be used in an
    attempt to break the tie, but if it fails, a random tiebreaker will be selected.

    Parameters
    ---------
    idx : int
        The index (aligned with ``normalized_labels``) of
        the current observation being resolved.
    normalized_labels : numpy.ndarray
        A :math:`(n,)` normalized array of labels for each observation.
    tally : numpy.ndarray
        The current tally of :math:`(p,)` neighbors' labels around ``idx`` to resolve.
    neighbors : dict of (neighbor_name : weight)
        The elements of the weights object (identical to ``w[idx]``)
        in the form ``{neighbor_name : weight}``.
    method : str
        The configuration option to use a specific tiebreaking method.
        See ``lag_categorical()`` for all supported options.
    w : libpysal.weights.weights.W
        A PySAL weights object aligned with ``normalized_labels``.

    Returns
    -------
    label : int
        An integer denoting which label to use to label the observation.
    
    Raises
    ------
    KeyError
        The tie-breaking method for categorical lag is not recognized.
    
    """

    m = method.lower()

    # returns a tuple for flat arrays
    (ties,) = np.where(tally == tally.max())

    # no tie, pick the highest
    if len(tally[tally == tally.max()]) <= 1:
        label = np.argmax(tally).astype(int)
    # choose randomly from tally
    elif m == "random":
        label = np.random.choice(np.squeeze(ties)).astype(int)
    # pick lowest tied value
    elif m == "lowest":
        label = ties[0].astype(int)
    # pick highest tied value
    elif m == "highest":
        label = ties[-1].astype(int)
    # add self-label as observation, try again, random if fail
    elif m == "tryself":
        mean_neighbor_value = np.mean(list(neighbors.values()))
        tally[normalized_labels[idx]] += mean_neighbor_value
        label = _resolve_ties(idx, normalized_labels, tally, neighbors, "random", w)
    else:
        msg = "Tie-breaking method for categorical lag not recognized: %s" % m
        raise KeyError(msg)

    return label

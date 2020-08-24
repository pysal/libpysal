import numpy as np


def adjlist_apply(X, W=None, alist=None, func=np.subtract, skip_verify=False):
    """Apply a function to an adajcency list, getting an adjacency list and result.

    Parameters
    ----------
    X : iterable
        An :math:`(N,P)`-length iterable to apply ``func`` to. If :math:`(N,1)`,
        then ``func`` must take 2 arguments and return a single reduction.
        If :math:`P`>1`, then ``func`` must take two :math:`P`-length arrays
        and return a single reduction of them.
    W : libpysal.weights.W
        A weights object that provides adjacency information. Default is ``None``.
    alist : pandas.DataFrame
        A table containing an adajacency list representation of a `W` matrix.
        Default is ``None``.
    func : callable
        A function taking two arguments and returning a single argument. This will
        be evaluated for every (focal, neighbor) pair, or each row of the adjacency
        list. If ``X`` has more than one column, this function should take two arrays
        and provide a single scalar in return. Default is ``np.subtract``.
        Example scalars include:
            ``lambda x,y: x < y, np.subtract``
        Example multivariates: 
            ``lambda (x,y): np.all(x < y)``
            ``lambda (x,y): np.sum((x-y)**2)``
            ``sklearn.metrics.euclidean_distance``
    skip_verify: bool
        Whether or not to skip verifying that the `W` is the same as an adjacency
        list. Do this if you are certain the adjacency list and `W` agree and
        would like to avoid re-instantiating a `W` from the adjacency list. 
        Default is ``False``.

    Returns
    -------
    alist_atts : list
        An adjacency list (or modifies ``alist`` inplace)
        with the function applied to each row.
    
    """

    try:
        import pandas as pd
    except ImportError:
        raise ImportError("Pandas must be installed to use this function.")

    W, alist = _get_W_and_alist(W, alist, skip_verify=skip_verify)

    if len(X.shape) > 1:
        if X.shape[-1] > 1:
            return _adjlist_mvapply(
                X, W=W, alist=alist, func=func, skip_verify=skip_verify
            )
    else:
        vec = np.asarray(X).flatten()

    ids = np.asarray(W.id_order)[:, None]
    table = pd.DataFrame(ids, columns=["id"])
    table = pd.concat((table, pd.DataFrame(vec[:, None], columns=("att",))), axis=1)
    alist_atts = pd.merge(alist, table, how="left", left_on="focal", right_on="id")
    alist_atts = pd.merge(
        alist_atts,
        table,
        how="left",
        left_on="neighbor",
        right_on="id",
        suffixes=("_focal", "_neighbor"),
    )
    alist_atts.drop(["id_focal", "id_neighbor"], axis=1, inplace=True)
    alist_atts[func.__name__] = alist_atts[["att_focal", "att_neighbor"]].apply(
        lambda x: func(x.att_focal, x.att_neighbor), axis=1
    )

    return alist_atts


def _adjlist_mvapply(X, W=None, alist=None, func=None, skip_verify=False):
    """This function is used when ``X`` is multi-dimensional.
    See ``libpysal.weights.adjtools.adjlist_apply()``
    for parameters and returns information.
    
    """

    try:
        import pandas as pd
    except ImportError:
        raise ImportError("Pandas must be installed to use this function.")

    assert len(X.shape) == 2, "Data is not two-dimensional."

    W, alist = _get_W_and_alist(W=W, alist=alist, skip_verify=skip_verify)

    assert X.shape[0] == W.n, "The number of samples in X does not match W."

    try:
        names = X.columns.tolist()
    except AttributeError:
        names = list(map(str, list(range(X.shape[1]))))

    ids = np.asarray(W.id_order)[:, None]
    table = pd.DataFrame(ids, columns=["id"])
    table = pd.concat((table, pd.DataFrame(X, columns=names)), axis=1)
    alist_atts = pd.merge(alist, table, how="left", left_on="focal", right_on="id")
    alist_atts = pd.merge(
        alist_atts,
        table,
        how="left",
        left_on="neighbor",
        right_on="id",
        suffixes=("_focal", "_neighbor"),
    )

    alist_atts.drop(["id_focal", "id_neighbor"], axis=1, inplace=True)
    alist_atts[func.__name__] = list(
        map(
            func,
            list(
                zip(
                    alist_atts.filter(like="_focal").values,
                    alist_atts.filter(like="_neighbor").values,
                )
            ),
        )
    )

    return alist_atts


def _get_W_and_alist(W, alist, skip_verify=False):
    """ Either (1) compute a `W` from an ``alist``; (2) compute an adjacency list
    from a `W`; (3) raise a ``ValueError`` if neither are provided; or (4) raise an
    ``AssertionError`` if both `W` and ``adjlist`` are provided and don't match.
    If this completes successfully, the `W` and ``adjlist`` will both be returned and
    are checked for equality. See ``libpysal.weights.adjtools.adjlist_apply()``
    for parameters and returns information.
    
    """

    if (alist is None) and (W is not None):
        alist = W.to_adjlist()

    elif (W is None) and (alist is not None):
        from .weights import W

        W = W.from_adjlist(alist)

    elif (W is None) and (alist is None):
        raise ValueError("Either W or Adjacency List must be provided.")

    elif (W is not None) and (alist is not None) and (not skip_verify):
        from .weights import W as W_

        np.testing.assert_allclose(
            W.sparse.toarray(), W_.from_adjlist(alist).sparse.toarray()
        )

    return W, alist


def adjlist_map(
    data,
    funcs=(np.subtract,),
    W=None,
    alist=None,
    focal_col="focal",
    neighbor_col="neighbor",
):
    """Map a set of functions over a `W` or an adjacency list.

    Parameters
    ----------
    data : {numpy.ndarray, pandas.Dataframe}
        `N x P` array of `N` observations and `P` covariates. 
    funcs : iterable or callable
        A function to apply to each of the `P` columns in ``data``, or a
        list of functions to apply to each column of `P`. This function
        must take two arguments, compare them, and return a value. Examples
        may be ``lambda x,y: x < y`` or ``np.subtract``.
         Default is ``(np.subtract,)``.
    W : libpysal.weights.W
        A PySAL weights object. If not provided, one is
        constructed from the given adjacency list. Default is ``None``.
    alist : pandas.Dataframe
        An adjacency list representation of a weights matrix. If not
        provided, one is constructed from the weights object. If both are
        provided, they are validated against one another to ensure they
        provide identical weights matrices. Default is ``None``.
    focal_col : str
        The name of column in ``alist`` containing the focal observation ids.
        Default is ``'focal'``.
    neighbor_col : str
        The name of column in ``alist`` containing the neighboring observation ids.
        Default is ``'neighbor'``.

    Returns
    -------
    alist : list
        An adjacency list (or modifies one if provided) with each function
        applied to the column of the data.
    
    """

    try:
        import pandas as pd
    except ImportError:
        raise ImportError("Pandas must be installed to use this function.")

    if isinstance(data, pd.DataFrame):
        names = data.columns
        data = data.values
    else:
        names = [str(i) for i in range(data.shape[1])]

    assert (
        data.shape[0] == W.n
    ), "The shape of 'data' does not match the shape of 'adjacency'."

    if callable(funcs):
        funcs = (funcs,)

    if len(funcs) == 1:
        funcs = [funcs[0] for _ in range(data.shape[1])]

    assert data.shape[1] == len(
        funcs
    ), "The shape of 'data' does not match the number of functions provided."
    W, alist = _get_W_and_alist(W, alist)

    fnames = set([f.__name__ for f in funcs])

    for i, (column, function) in enumerate(zip(data.T, funcs)):
        alist = adjlist_apply(X=column, W=W, alist=alist, skip_verify=True)
        alist.drop(["att_focal", "att_neighbor"], axis=1, inplace=True)
        alist = alist.rename(
            columns={function.__name__: "_".join((function.__name__, names[i]))}
        )
        fnames.update((function.__name__,))

    return alist


def filter_adjlist(adjlist, focal_col="focal", neighbor_col="neighbor"):
    """This deduplicates an adjacency list by examining both `(a,b)` and `(b,a)`
    when `(a,b)` is encountered. The removal is done in order of the iteration
    order of the input adjacency list. So, if a special order of removal is
    desired, you need to sort the list before this function.

    Parameters
    ----------
    adjlist : pandas.DataFrame
        A dataframe that contains focal and neighbor columns.
    focal_col : str
        The name of the column with the focal observation id. Default is ``'focal'``.
    neighbor_col : str
        The name of the column with the neighbor observation id.
        Default is ``'neighbor'``.

    Returns
    -------
    adjlist : pandas.DataFrame
        An adjacency table with reversible entries removed.
    
    """

    edges = adjlist.loc[:, [focal_col, neighbor_col]]
    undirected = set()
    to_remove = []

    for index, *edge in edges.itertuples(name=None):
        edge = tuple(edge)
        if edge in undirected or edge[::-1] in undirected:
            to_remove.append(index)
        else:
            undirected.add(edge)
            undirected.add(edge[::-1])
    adjlist = adjlist.drop(to_remove)

    return adjlist

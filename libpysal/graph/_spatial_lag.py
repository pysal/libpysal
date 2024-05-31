import numpy as np
import pandas as pd


def _lag_spatial(graph, y, categorical=False, ties='raise'):
    """Spatial lag operator

    Constructs spatial lag based on neighbor relations of the graph.


    Parameters
    ----------
    graph : Graph
        libpysal.graph.Graph
    y : array
        numpy array with dimensionality conforming to w
    categorical : bool
        True if y is categorical, False if y is continuous.
    ties : {'raise', 'random', 'tryself'}, optional
        Policy on how to break ties when a focal unit has multiple
        modes for a categorical lag.
        - 'raise': This will raise an exception if ties are
          encountered to alert the user (Default).
        - 'random': Will break ties randomly.
        - 'tryself': Add a self-weight to attempt to break the tie
          with the focal label. If the self-weight does not break a
          tie, or the self-weight induces a tie, the tie will be be
          broken randomly.


    Returns
    -------
    numpy.array
        array of numeric|categorical values for the spatial lag


    Examples
    --------
    >>> from libpysal.graph._spatial_lag import _lag_spatial
    >>> import numpy as np
    >>> from libpysal.weights.util import lat2W
    >>> from libpysal.graph import Graph
    >>> graph = Graph.from_W(lat2W(3,3))
    >>> y = np.arange(9)
    >>> _lag_spatial(graph, y)
    array([ 4.,  6.,  6., 10., 16., 14., 10., 18., 12.])

    Row standardization
    >>> w = lat2W(3,3)
    >>> w.transform = 'r'
    >>> graph = Graph.from_W(w)
    >>> y = np.arange(9)
    >>> _lag_spatial(graph, y)
    array([2.        , 2.        , 3.        , 3.33333333, 4.        ,
           4.66666667, 5.        , 6.        , 6.        ])


    Categorical Lag (no ties)
    >>> y = np.array([*'ababcbcbc'])
    >>> _lag_spatial(graph, y, categorical=True)
    array(['b', 'a', 'b', 'c', 'b', 'c', 'b', 'c', 'b'], dtype=object)

    Handling ties
    >>> y[3] = 'a'
    >>> np.random.seed(12345)
    >>> _lag_spatial(graph, y, categorical=True, ties='random')
    array(['a', 'a', 'b', 'c', 'b', 'c', 'b', 'c', 'b'], dtype=object)
    >>> _lag_spatial(graph, y, categorical=True, ties='random')
    array(['b', 'a', 'b', 'c', 'b', 'c', 'b', 'c', 'b'], dtype=object)
    >>> _lag_spatial(graph, y, categorical=True, ties='tryself')
    array(['a', 'a', 'b', 'a', 'b', 'c', 'b', 'c', 'b'], dtype=object)

    """
    sp = graph.sparse
    if len(y) != sp.shape[0]:
        raise ValueError(
            "The length of `y` needs to match the number of observations "
            f"in Graph. Expected {sp.shape[0]}, got {len(y)}."
        )

    if categorical:
        if ties == 'tryself':
            graph = graph.assign_self_weight()
        df = pd.DataFrame(data=graph.adjacency)
        df['neighbor_label'] = y[graph.adjacency.index.get_level_values(1)]
        gb = df.groupby(['focal', 'neighbor_label']
                        ).count().groupby(level='focal')
        n_ties = gb.apply(_check_ties).sum()
        if n_ties and ties == 'raise':
            raise ValueError(
                f"There are {n_ties} ties that must be broken "
                f"to define the categorical "
                "spatial lag for these observations. To address this "
                "issue, consider setting `ties='tryself'` "
                "or `ties='random'` or consult the documentation "
                "about ties and the categorical spatial lag."
            )
        elif ties in ('random', 'tryself', 'raise'):
            return gb.apply(_get_categorical_lag).values

        else:
            raise ValueError(
                f"Recieved option ties='{ties}', but only options "
                "'raise','random','tryself' are supported."
            )

    return sp @ y


def _check_ties(focal):
    """Reduction to determine if a focal unit has multiple modes for neighbor labels.

    Parameters
    ----------
    focal: row from pandas Dataframe
          Data is a Graph with an additional column having the labels for the neighbors

    Returns
    -------
    bool
    """

    max_count = focal.weight.max()
    if (focal.weight == max_count).sum() > 1:
        return True
    return False


def _get_categorical_lag(focal):
    """Reduction to determine categorical spatial lag for a focal
    unit.

    Parameters
    ----------
    focal: row from pandas Dataframe
          Data is a Graph with an additional column having the labels for the neighbors

    Returns
    -------
    str
      Label for the value of the categorical lag
    """
    filtered = focal[focal.weight == focal.weight.max()]
    max_label = filtered.index.get_level_values('neighbor_label').values
    return np.random.choice(max_label, 1)[0]

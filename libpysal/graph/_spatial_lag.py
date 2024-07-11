import numpy as np
import pandas as pd


def _lag_spatial(graph, y, categorical=False, ties="raise"):
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
        - 'random': modal label ties Will be broken randomly.
        - 'tryself': check if focal label breaks the tie between label
          modes.  If the focal label does not break the modal tie, the
          tie will be be broken randomly. If the focal unit has a
          self-weight, focal label is not used to break any tie,
          rather any tie will be broken randomly.


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
    array(['a', 'a', 'b', 'c', 'b', 'c', 'a', 'c', 'b'], dtype=object)

    """
    sp = graph.sparse
    if len(y) != sp.shape[0]:
        raise ValueError(
            "The length of `y` needs to match the number of observations "
            f"in Graph. Expected {sp.shape[0]}, got {len(y)}."
        )

    # coerce list to array
    if isinstance(y, list):
        y = np.array(y)

    if (
        isinstance(y.dtype, pd.CategoricalDtype)
        or pd.api.types.is_object_dtype(y.dtype)
        or pd.api.types.is_bool_dtype(y.dtype)
        or pd.api.types.is_string_dtype(y.dtype)
    ):
        categorical = True
    if categorical:
        if isinstance(y, np.ndarray):
            y = pd.Series(y, index=graph.unique_ids)

        df = pd.DataFrame(data=graph.adjacency)
        df["neighbor_label"] = y.loc[graph.adjacency.index.get_level_values(1)].values
        df["own_label"] = y.loc[graph.adjacency.index.get_level_values(0)].values
        df["neighbor_idx"] = df.index.get_level_values(1)
        df["focal_idx"] = df.index.get_level_values(0)
        gb = df.groupby(["focal", "neighbor_label"]).count().groupby(level="focal")
        n_ties = gb.apply(_check_ties).sum()
        if n_ties and ties == "raise":
            raise ValueError(
                f"There are {n_ties} ties that must be broken "
                f"to define the categorical "
                "spatial lag for these observations. To address this "
                "issue, consider setting `ties='tryself'` "
                "or `ties='random'` or consult the documentation "
                "about ties and the categorical spatial lag."
            )
        # either there are ties and random|tryself specified or
        # there are no ties
        gb = df.groupby(by=["focal"])
        if ties == "random" or ties == "raise":
            return gb.apply(_get_categorical_lag).values
        elif ties == "tryself" or ties == "raise":
            return gb.apply(_get_categorical_lag, ties="tryself").values
        else:
            raise ValueError(
                f"Received option ties='{ties}', but only options "
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
    return (focal.weight == max_count).sum() > 1


def _get_categorical_lag(focal, ties="random"):
    """Reduction to determine categorical spatial lag for a focal unit.

    Parameters
    ----------
    focal: row from pandas Dataframe
          Data is a Graph with an additional column having the labels for the neighbors

    ties : {'raise', 'random', 'tryself'}, optional
        Policy on how to break ties when a focal unit has multiple
        modes for a categorical lag.
        - 'raise': This will raise an exception if ties are
          encountered to alert the user (Default).
        - 'random': Will break ties randomly.
        - 'tryself': check if focal label breaks the tie between label
          modes.  If the focal label does not break the modal tie, the
          tie will be be broken randomly. If the focal unit has a
          self-weight, focal label is not used to break any tie,
          rather any tie will be broken randomly.


    Returns
    -------
    str|int|float:
      Label for the value of the categorical lag
    """
    self_weight = focal.focal_idx.values[0] in focal.neighbor_idx.values
    labels, counts = np.unique(focal.neighbor_label, return_counts=True)
    node_label = labels[counts == counts.max()]
    if ties == "random" or (ties == "tryself" and self_weight):
        return np.random.choice(node_label, 1)[0]
    elif ties == "tryself" and not self_weight:
        self_label = focal.own_label.values[0]
        if self_label in node_label:  # focal breaks tie
            return self_label
        else:
            return np.random.choice(node_label, 1)[0]

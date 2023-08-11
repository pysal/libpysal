def _lag_spatial(graph, y):
    """Spatial lag operator

    If w is row standardized, returns the average of each observation's neighbors;
    if not, returns the weighted sum of each observation's neighbors.

    Parameters
    ----------
    graph : Graph
        libpysal.graph.Graph
    y : array
        numpy array with dimensionality conforming to w

    Returns
    -------
    numpy.array
        array of numeric values for the spatial lag
    """
    sp = graph.sparse
    if len(y) != sp.shape[0]:
        raise ValueError("The length of `y` needs to match the number of observations "
                         f"in Graph. Expected {sp.shape[0]}, got {len(y)}.")
    return graph.sparse @ y

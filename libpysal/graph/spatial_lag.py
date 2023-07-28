def lag_spatial(graph, y):
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
    return graph.sparse @ y

def lag_spatial(w, y):
    """Spatial lag operator

    If w is row standardized, returns the average of each observation's neighbors;
    if not, returns the weighted sum of each observation's neighbors.

    Parameters
    ----------
    w : W
        libpysal.weights.experimental.W
    y : array
        numpy array with dimensionality conforming to w

    Returns
    -------
    numpy.array
        array of numeric values for the spatial lag
    """
    return w.sparse @ y

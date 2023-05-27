import pandas as pd
import numpy as np
import geopandas # this is geopandas base
import shapely

def _neighbor_dict_to_edges(neighbors, weights=None):
    idxs = pd.Series(neighbors).explode()
    heads, tails = idxs.index.values, idxs.values
    tails = tails.astype(heads.dtype)
    if weights is not None:
        data_array = pd.Series(weights).explode().values
        if not pd.api.types.is_numeric_dtype(data_array):
            data_array = pd.to_numeric(data_array)
    else:
        data_array = np.ones(idxs.shape[0], dtype=int)
        data_array[heads == tails] = 0
    return heads, tails, data_array



def _validate_geometry_input(geoms, ids=None, valid_geometry_types=None):
    if isinstance(geoms, (geopandas.GeoSeries, geopandas.GeoDataFrame)):
        geoms = geoms.geometry
        if ids is None:
            ids = geoms.index
        ids = np.asarray(ids)
        geom_types = set(geoms.geom_type)
        if valid_geometry_types is not None:
            if isinstance(valid_geometry_types, str):
                valid_geometry_types = (valid_geometry_types,)
            valid_geometry_types = set(valid_geometry_types)
            if not geom_types <= valid_geometry_types:
                raise ValueError(
                    f"this W type is only well-defined for geom_types: {valid_geometry_types}."
                )
        coordinates = shapely.get_coordinates(geoms)
        geoms = geoms.copy()
        geoms.index = ids
        return coordinates, ids, geoms
    elif isinstance(geoms.dtype, geopandas.array.GeometryDtype):
        return _validate_geometry_input(geopandas.GeoSeries(geoms), ids=ids, valid_geometry_types=valid_geometry_types)
    else:
        if (geoms.ndim == 2) and (geoms.shape[1] == 2):
            return _validate_geometry_input(geopandas.points_from_xy(*geoms.T), ids=ids, valid_geometry_types=valid_geometry_types)
    raise ValueError(
        "input geometry type is not supported. Input must either be a geopandas.GeoSeries, geopandas.GeoDataFrame, a numpy array with a geometry dtype, or an array of coordinates."
    )

def lat2W(nrows=5, ncols=5, rook=True, id_type="int"):
    """
    Create a W object for a regular lattice.

    Parameters
    ----------

    nrows      : int
                 number of rows
    ncols      : int
                 number of columns
    rook       : boolean
                 type of contiguity. Default is rook. For queen, rook =False
    id_type    : string
                 string defining the type of IDs to use in the final W object;
                 options are 'int' (0, 1, 2 ...; default), 'float' (0.0,
                 1.0, 2.0, ...) and 'string' ('id0', 'id1', 'id2', ...)
    **kwargs   : keyword arguments
                 optional arguments for :class:`pysal.weights.W`


    Returns
    -------

    w : W
        instance of spatial weights class W

    Notes
    -----

    Observations are row ordered: first k observations are in row 0, next k in row 1, and so on.

    Examples
    --------

    >>> from libpysal.weights import lat2W
    >>> w9 = lat2W(3,3)
    >>> "%.3f"%w9.pct_nonzero
    '29.630'
    >>> w9[0] == {1: 1.0, 3: 1.0}
    True
    >>> w9[3] == {0: 1.0, 4: 1.0, 6: 1.0}
    True
    """
    from .base import W

    n = nrows * ncols
    r1 = nrows - 1
    c1 = ncols - 1
    rid = [i // ncols for i in range(n)]  # must be floor!
    cid = [i % ncols for i in range(n)]
    w = {}
    r = below = 0
    for i in range(n - 1):
        if rid[i] < r1:
            below = rid[i] + 1
            r = below * ncols + cid[i]
            w[i] = w.get(i, []) + [r]
            w[r] = w.get(r, []) + [i]
        if cid[i] < c1:
            right = cid[i] + 1
            c = rid[i] * ncols + right
            w[i] = w.get(i, []) + [c]
            w[c] = w.get(c, []) + [i]
        if not rook:
            # southeast bishop
            if cid[i] < c1 and rid[i] < r1:
                r = (rid[i] + 1) * ncols + 1 + cid[i]
                w[i] = w.get(i, []) + [r]
                w[r] = w.get(r, []) + [i]
            # southwest bishop
            if cid[i] > 0 and rid[i] < r1:
                r = (rid[i] + 1) * ncols - 1 + cid[i]
                w[i] = w.get(i, []) + [r]
                w[r] = w.get(r, []) + [i]

    neighbors = {}
    weights = {}
    for key in w:
        weights[key] = [1.0] * len(w[key])
    ids = list(range(n))
    if id_type == "string":
        ids = ["id" + str(i) for i in ids]
    elif id_type == "float":
        ids = [i * 1.0 for i in ids]
    if id_type == "string" or id_type == "float":
        id_dict = dict(list(zip(list(range(n)), ids)))
        alt_w = {}
        alt_weights = {}
        for i in w:
            values = [id_dict[j] for j in w[i]]
            key = id_dict[i]
            alt_w[key] = values
            alt_weights[key] = weights[i]
        w = alt_w
        weights = alt_weights

    return W.from_dicts(weights)

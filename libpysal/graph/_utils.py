import geopandas
import numpy as np
import pandas as pd
import shapely
from itertools import permutations

def _jitter_geoms(coordinates, geoms, seed=None):
    """
    Jitter geometries based on the smallest required movements to induce 
    uniqueness. For each point, this samples a radius and angle uniformly 
    at random from the unit circle and displaces the point. For a 
    non-euclidean geometry, like latitude longitude coordinates,
    this will distort according to a plateé carree prjection, jittering 
    slightly more in the x direction than the y direction. 
    """
    rng = np.random.default_rng(seed=seed)
    dtype = coordinates.dtype
    if dtype not in (np.float32, np.float64):
        # jittering requires us to cast ints to float
        # and the rng.random generator only works with float32 and float64
        dtype = np.float32
    # the resolultion is the approximate difference between two floats
    # that can be resolved at the given dtype.
    resolution = np.finfo(dtype).resolution
    r = rng.random(size=coordinates.shape[0], dtype=dtype)**.5 * resolution
    theta = rng.random(size=coordinates.shape[0], dtype=dtype) * np.pi * 2
    # converting from polar to cartesian
    dx = r + np.sin(theta)
    dy = r + np.cos(theta)
    # then adding the displacements
    coordinates = coordinates + np.column_stack((dx,dy))
    geoms = geopandas.GeoSeries(geopandas.points_from_xy(*coordinates.T, crs=geoms.crs))
    return coordinates, geoms

def _induce_cliques(adjtable, clique_to_members, fill_value=1):
    """
    induce cliques into the input graph. This connects the 
    """
    adj_across_clique = adjtable.merge(
        clique_to_members['input_index'], left_index=True, right_index=True
    ).explode('input_index').rename(
        columns=dict(input_index='subclique_focal')
    ).merge(
        clique_to_members['input_index'], left_on='neighbor', right_index=True
    ).explode('input_index').rename(
        columns=dict(input_index='subclique_neighbor')
    ).reset_index().drop(
        ['focal','neighbor', 'index'], axis=1
    ).rename(
        columns=dict(subclique_focal="focal", subclique_neighbor='neighbor')
    )
    is_multimember_clique = clique_to_members['input_index'].str.len()>1
    adj_within_clique = clique_to_members[
        is_multimember_clique
    ]['input_index'].apply(
        lambda x: list(permutations(x, 2))
    ).explode().apply(
        pd.Series
    ).rename(columns={0:"focal", 1:"neighbor"}).assign(weight=fill_value)

    new_adj = pd.concat(
        (adj_across_clique, adj_within_clique),
        ignore_index=True, axis=0
    ).reset_index(drop=True)

    return new_adj


def _neighbor_dict_to_edges(neighbors, weights=None):
    """
    Convert a neighbor dict to a set of (head, tail, weight) edges, assuming
    that the any self-loops have a weight of zero.
    """
    idxs = pd.Series(neighbors).explode()
    idxs = idxs.fillna(pd.Series(idxs.index, index=idxs.index))  # self-loops
    heads, tails = idxs.index.values, idxs.values
    tails = tails.astype(heads.dtype)
    if weights is not None:
        data_array = pd.Series(weights).explode().fillna(0).values
        if not pd.api.types.is_numeric_dtype(data_array):
            data_array = pd.to_numeric(data_array)
    else:
        data_array = np.ones(idxs.shape[0], dtype=int)
        data_array[heads == tails] = 0
    return heads, tails, data_array


def _build_coincidence_lookup(geoms):
    """
    Identify coincident points and create a look-up table for the coincident geometries. 
    """
    valid_coincident_geom_types = set(("Point",))
    if not set(geoms.geom_type) <= valid_coincident_geom_types:
        raise ValueError(
            f"coindicence checks are only well-defined for geom_types: {valid_coincident_geom_types}"
            )
    max_coincident = geoms.geometry.duplicated().sum()
    lut = geoms.to_frame("geometry").reset_index().groupby("geometry")['index'].agg(list).reset_index()
    lut = geopandas.GeoDataFrame(lut)
    return max_coincident, lut.rename(columns=dict(index='input_index'))

def _validate_geometry_input(geoms, ids=None, valid_geometry_types=None):
    """
    Ensure that input geometries are always aligned to (and refer back to)
    inputted geometries. Geoms can be a GeoSeries, GeoDataFrame, numpy.array
    with a geometry dtype, or a point array.

    is will always align to geoms.

    the returned coordinates will always pertain to geoms, but may be
    longer than geoms (such as when geoms represents polygons).
    """
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
                    "this Graph type is only well-defined for "
                    f"geom_types: {valid_geometry_types}."
                )
        coordinates = shapely.get_coordinates(geoms)
        geoms = geoms.copy()
        geoms.index = ids
        return coordinates, ids, geoms
    elif isinstance(geoms.dtype, geopandas.array.GeometryDtype):
        return _validate_geometry_input(
            geopandas.GeoSeries(geoms),
            ids=ids,
            valid_geometry_types=valid_geometry_types,
        )
    else:
        if (geoms.ndim == 2) and (geoms.shape[1] == 2):
            return _validate_geometry_input(
                geopandas.points_from_xy(*geoms.T),
                ids=ids,
                valid_geometry_types=valid_geometry_types,
            )
    raise ValueError(
        "input geometry type is not supported. Input must either be a "
        "geopandas.GeoSeries, geopandas.GeoDataFrame, a numpy array with a geometry "
        "dtype, or an array of coordinates."
    )


def lat2Graph(nrows=5, ncols=5, rook=True, id_type="int"):
    """
    Create a Graph object for a regular lattice.

    Parameters
    ----------

    nrows      : int
                 number of rows
    ncols      : int
                 number of columns
    rook       : boolean
                 type of contiguity. Default is rook. For queen, rook =False
    id_type    : string
                 string defining the type of IDs to use in the final Graph object;
                 options are 'int' (0, 1, 2 ...; default), 'float' (0.0,
                 1.0, 2.0, ...) and 'string' ('id0', 'id1', 'id2', ...)

    Returns
    -------

    Graph
        libpysal.graph.Graph

    Notes
    -----

    Observations are row ordered: first k observations are in row 0, next k in row 1,
    and so on.

    Examples
    --------

    >>> from libpysal.weights import lat2Graph
    >>> w9 = lat2Graph(3,3)
    >>> "%.3f"%w9.pct_nonzero
    '29.630'
    >>> w9[0] == {1: 1.0, 3: 1.0}
    True
    >>> w9[3] == {0: 1.0, 4: 1.0, 6: 1.0}
    True
    """
    # TODO: this seems to be broken now
    from .base import Graph

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

    return Graph.from_dicts(weights)


def _vec_euclidean_distances(X,Y):
    """
    compute the euclidean distances along corresponding rows of two arrays
    """
    return ((X-Y)**2).sum(axis=1)**.5


def _evaluate_index(data):
    """Helper to get ids from any input."""
    return (
        data.index
        if isinstance(data, (pd.Series, pd.DataFrame))
        else pd.RangeIndex(0, len(data))
    )

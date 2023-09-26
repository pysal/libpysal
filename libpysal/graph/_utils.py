import geopandas
import numpy as np
import pandas as pd
import shapely
from itertools import permutations
from packaging.version import Version

GPD_013 = Version(geopandas.__version__) >= Version("0.13")


def _sparse_to_arrays(sparray, ids=None):
    """Convert sparse array to arrays of adjacency"""
    sparray = sparray.tocoo(copy=False)
    if ids is not None:
        ids = np.asarray(ids)
        if sparray.shape[0] != ids.shape[0]:
            raise ValueError(
                f"The length of ids ({ids.shape[0]}) does not match "
                f"the shape of sparse {sparray.shape}."
            )

        sorter = sparray.row.argsort()
        head = ids[sparray.row][sorter]
        tail = ids[sparray.col][sorter]
        data = sparray.data[sorter]
    else:
        sorter = sparray.row.argsort()
        head = sparray.row[sorter]
        tail = sparray.col[sorter]
        data = sparray.data[sorter]
        ids = np.arange(sparray.shape[0], dtype=int)
    return _resolve_islands(head, tail, ids, data)


def _jitter_geoms(coordinates, geoms, seed=None):
    """
    Jitter geometries based on the smallest required movements to induce
    uniqueness. For each point, this samples a radius and angle uniformly
    at random from the unit circle, rescales it to a circle of values that
    are extremely small relative to the precision of the input, and
    then displaces the point. For a non-euclidean geometry, like latitude
    longitude coordinates, this will distort according to a plateé carree
    projection, jittering slightly more in the x direction than the y direction.
    """
    rng = np.random.default_rng(seed=seed)
    dtype = coordinates.dtype
    if dtype not in (np.float32, np.float64):
        # jittering requires us to cast ints to float
        # and the rng.random generator only works with float32 and float64
        dtype = np.float32
    # the resolution is the approximate difference between two floats
    # that can be resolved at the given dtype.
    resolution = np.finfo(dtype).resolution
    r = rng.random(size=coordinates.shape[0], dtype=dtype) ** 0.5 * resolution
    theta = rng.random(size=coordinates.shape[0], dtype=dtype) * np.pi * 2
    # converting from polar to cartesian
    dx = r + np.sin(theta)
    dy = r + np.cos(theta)
    # then adding the displacements
    coordinates = coordinates + np.column_stack((dx, dy))
    geoms = geopandas.GeoSeries(geopandas.points_from_xy(*coordinates.T, crs=geoms.crs))
    return coordinates, geoms


def _induce_cliques(adjtable, clique_to_members, fill_value=1):
    """
    induce cliques into the input graph. This connects everything within a
    clique together, as well as connecting all things outside of the clique
    to all members of the clique.

    This does not guarantee/understand ordering of the *output* adjacency table.
    """
    adj_across_clique = (
        adjtable.merge(
            clique_to_members["input_index"], left_index=True, right_index=True
        )
        .explode("input_index")
        .rename(columns=dict(input_index="subclique_focal"))
        .merge(clique_to_members["input_index"], left_on="neighbor", right_index=True)
        .explode("input_index")
        .rename(columns=dict(input_index="subclique_neighbor"))
        .reset_index()
        .drop(["focal", "neighbor", "index"], axis=1)
        .rename(columns=dict(subclique_focal="focal", subclique_neighbor="neighbor"))
    )
    is_multimember_clique = clique_to_members["input_index"].str.len() > 1
    adj_within_clique = (
        clique_to_members[is_multimember_clique]["input_index"]
        .apply(lambda x: list(permutations(x, 2)))
        .explode()
        .apply(pd.Series)
        .rename(columns={0: "focal", 1: "neighbor"})
        .assign(weight=fill_value)
    )

    new_adj = pd.concat(
        (adj_across_clique, adj_within_clique), ignore_index=True, axis=0
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
            "coindicence checks are only well-defined for "
            f"geom_types: {valid_coincident_geom_types}"
        )
    max_coincident = geoms.geometry.duplicated().sum()
    if GPD_013:
        lut = (
            geoms.to_frame("geometry")
            .reset_index()
            .groupby("geometry")["index"]
            .agg(list)
            .reset_index()
        )
    else:
        lut = (
            geoms.to_wkb()
            .to_frame("geometry")
            .reset_index()
            .groupby("geometry")["index"]
            .agg(list)
            .reset_index()
        )
        lut["geometry"] = geopandas.GeoSeries.from_wkb(lut["geometry"])

    lut = geopandas.GeoDataFrame(lut)
    return max_coincident, lut.rename(columns=dict(index="input_index"))


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
                    "This Graph type is only well-defined for "
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


def _validate_sparse_input(sparse, ids=None):
    assert (
        sparse.shape[0] == sparse.shape[1]
    ), "coordinates should represent a distance matrix if metric='precomputed'"
    return _sparse_to_arrays(sparse, ids)


def _vec_euclidean_distances(X, Y):
    """
    compute the euclidean distances along corresponding rows of two arrays
    """
    return ((X - Y) ** 2).sum(axis=1) ** 0.5


def _evaluate_index(data):
    """Helper to get ids from any input."""
    return (
        data.index
        if isinstance(data, (pd.Series, pd.DataFrame))
        else pd.RangeIndex(0, len(data))
    )


def _resolve_islands(heads, tails, ids, weights):
    """
    Induce self-loops for a collection of ids and links describing a
    contiguity graph. Induced self-loops will have zero weight.
    """
    islands = np.setdiff1d(ids, heads)
    if islands.shape != (0,):
        heads = np.hstack((heads, islands))
        tails = np.hstack((tails, islands))
        weights = np.hstack((weights, np.zeros_like(islands, dtype=int)))

    # ensure proper order after adding isolates to the end
    adjacency = pd.Series(weights, index=pd.MultiIndex.from_arrays([heads, tails]))
    adjacency = adjacency.reindex(ids, level=0).reindex(ids, level=1)
    return (
        adjacency.index.get_level_values(0),
        adjacency.index.get_level_values(1),
        adjacency.values,
    )

from warnings import warn

import numpy as np
import pandas as pd

from ..weights.raster import _da2wsp
from ._utils import (
    _sparse_to_arrays,
)


def _raster_contiguity(
    da,
    criterion="queen",
    z_value=None,
    coords_labels=None,
    k=1,
    include_nodata=False,
    n_jobs=1,
):
    """
    Create an input for Graph from xarray.DataArray.

    Parameters
    ----------
    da : xarray.DataArray
        Input 2D or 3D DataArray with shape=(z, y, x)
    criterion : {"rook", "queen"}
        Type of contiguity. Default is queen.
    z_value : int/string/float
        Select the z_value of 3D DataArray with multiple layers.
    coords_labels : dictionary
        Pass dimension labels for coordinates and layers if they do not
        belong to default dimensions, which are (band/time, y/lat, x/lon)
        e.g. coords_labels = {"y_label": "latitude",
        "x_label": "longitude", "z_label": "year"}
        Default is {} empty dictionary.
    k : int
        Order of contiguity, this will select all neighbors upto kth order.
        Default is 1.
    include_nodata : boolean
        If True, missing values will be assumed as non-missing when
        selecting higher_order neighbors, Default is False
    n_jobs : int
        Number of cores to be used in the sparse weight construction. If -1,
        all available cores are used. Default is 1.

    Returns
    -------
    (head, tail, weight, index_names)


    """
    try:
        import numba  # noqa: F401

        use_numba = True
        include_nodata = False
    except (ModuleNotFoundError, ImportError):
        warn(
            "numba cannot be imported, parallel processing "
            "and include_nodata functionality will be disabled. "
            "falling back to slower method",
            stacklevel=2,
        )
        use_numba = False

    if coords_labels is None:
        coords_labels = {}

    if use_numba:
        (weight, (head, tail)), ser, _ = _da2wsp(
            da=da,
            criterion=criterion,
            z_value=z_value,
            coords_labels=coords_labels,
            k=k,
            include_nodata=include_nodata,
            n_jobs=n_jobs,
            use_numba=use_numba,
        )
        order = np.lexsort((tail, head))
        head = head[order]
        tail = tail[order]
        weight = weight[order]

        head = ser.index.to_numpy()[head]
        tail = ser.index.to_numpy()[tail]
    else:
        sw, ser = _da2wsp(
            da=da,
            criterion=criterion,
            z_value=z_value,
            coords_labels=coords_labels,
            k=k,
            include_nodata=include_nodata,
            n_jobs=n_jobs,
            use_numba=use_numba,
        )
        head, tail, weight = _sparse_to_arrays(sw, ser.index.to_numpy())

    return (
        head,
        tail,
        weight,
        ser.index,
    )


def _generate_da(g, y):
    """Creates xarray.DataArray object from passed data aligned with the Graph.

    Parameters
    ----------
    g : Graph
        Graph, ideally generated using _raster_contiguity builder to ensure it
        contains _xarray_index_names attribute.
    y : array_like
        flat array that shall be reshaped into a DataArray with dimensionality
        conforming to Graph

    Returns
    -------
    xarray.DataArray
        instance of xarray.DataArray that can be aligned with the DataArray from which
        Graph was built
    """
    if hasattr(g, "_xarray_index_names"):
        names = g._xarray_index_names
    else:
        warn(
            UserWarning,
            "Graph does not store xarray index names."
            "The output may not align with the original DataArray.",
            stacklevel=3,
        )
        names = None
    return pd.Series(
        y,
        index=pd.MultiIndex.from_tuples(g.unique_ids, names=names),
    ).to_xarray()

import numpy as np
import pandas as pd

from ..weights.raster import _da2wsp


def _raster_contiguity(
    da,
    criterion="queen",
    z_value=None,
    coords_labels=None,
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
    # TODO: figure out K and include_nodata - do we need k here?
    if coords_labels is None:
        coords_labels = {}

    (weight, (head, tail)), ser, _ = _da2wsp(
        da=da,
        criterion=criterion,
        z_value=z_value,
        coords_labels=coords_labels,
        k=1,
        include_nodata=include_nodata,
        n_jobs=n_jobs,
    )

    order = np.lexsort((tail, head))
    head = head[order]
    tail = tail[order]
    weight = weight[order]

    return (
        ser.index[head].to_numpy(),
        ser.index[tail].to_numpy(),
        weight,
        ser.index.names,
    )


def _generate_da(g, y):
    names = g._xarray_index_names if hasattr(g, "_xarray_index_names") else None
    return pd.Series(
        y,
        index=pd.MultiIndex.from_tuples(g.unique_ids, names=names),
    ).to_xarray()

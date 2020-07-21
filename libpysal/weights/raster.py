from .util import lat2SW
from .weights import WSP
import numpy as np
from warnings import warn

try:
    from xarray import DataArray
except ImportError:
    raise ImportError(
        "xarray must be installed to use this functionality")


__all__ = ['da_checker', 'da2W', 'da2WSP', 'w2da']


def da2W(da, criterion="rook", band=None, **kwargs):
    """
    Create a W object from rasters(xarray.DataArray)

    Parameters
    ----------
    da : xarray.DataArray
       raster file accessed using xarray.open_rasterio method
    criterion : {"rook", "queen"}
       type of contiguity. Default is rook.
    band : int
       select band for da with multiple bands. Default is 1
    **kwargs : keyword arguments
        optional arguments for :class:`libpysal.weights.W`

    Returns
    -------
    w : libpysal.weights.W
       instance of spatial weights class W
    """
    wsp = da2WSP(da, criterion, band)
    w = wsp.to_W(**kwargs)
    # temp adding index attribute until the W constructor is redesigned
    w.index = wsp.index
    return w


def da2WSP(da, criterion="rook", band=None):
    """
    Generate a WSP object from rasters(xarray.DataArray)

    Parameters
    ----------
    da : xarray.DataArray
       raster file accessed using xarray.open_rasterio method
    criterion : {"rook", "queen"}
       type of contiguity. Default is rook.
    band : int
        select band for da with multiple bands. Default is 1

    Returns
    -------
    wsp : libpysal.weights.WSP
        instance of spatial weights class WSP
    """
    band = da_checker(da, band)
    da = da[band-1:band]
    sw = lat2SW(*da[0].shape, criterion)
    ser = da.to_series()
    mask = (ser != da.nodatavals[0]).to_numpy()
    sw = sw[mask]
    sw = sw[:, mask]
    ser = ser[ser != da.nodatavals[0]]
    index = ser.index
    wsp = WSP(sw, index=index)
    return wsp


def w2da(data, w, attrs={}, coords=None):
    """
    Creates DataArray object from passed data

    Arguments
    ---------
    data : array/list/pd.Series
        1d array-like data with dimensionality conforming to w
    w : libpysal.weights.W
        Spatial weights object aligned with passed data
    attrs : Dictionary
        Attributes stored in dict related to DataArray, e.g. da.attrs
    coords : Dictionary/xarray.core.coordinates.DataArrayCoordinates
        coordinates corresponding to DataArray, e.g. da.coords

    Returns
    -------
    da : xarray.DataArray
        instance of xarray.DataArray
    """
    data = np.array(data).flatten()
    if coords is None:
        idx = w.index
        dims = idx.names
        shape = tuple(lev.size for lev in idx.levels)
        indexer = tuple(idx.codes)
        missing = np.prod(shape) > idx.shape[0]
        if missing:
            if attrs:
                fill_value = attrs["nodatavals"][0]
            else:
                fill_value = np.floor(np.min(data)) - 1
                attrs["nodatavals"] = tuple([fill_value])
            data_complete = np.full(shape, fill_value, data.dtype)
        else:
            data_complete = np.empty(shape, data.dtype)
        data_complete[indexer] = data
        coords = {}
        for dim, lev in zip(dims, idx.levels):
            coords[dim] = lev.to_numpy()
    else:
        shape = tuple(len(value) for value in coords.values())
        dims = tuple(key for key in coords.keys())
        data_complete = np.array(data).reshape(shape)
    da = DataArray(data_complete, coords=coords, dims=dims, attrs=attrs)
    return da


def da_checker(da, band):
    """
    xarray dataarray checker
    Parameters
    ----------
    da : xarray.DataArray
        raster file accessed using xarray.open_rasterio method
    band : int
        user selected band

    Returns
    -------
    band : int
        return default band value
    """
    if not isinstance(da, DataArray):
        raise TypeError("da must be an instance of xarray.DataArray")
    if da[0].ndim != 2:
        raise ValueError("raster must be 2D")
    if not (issubclass(da.values.dtype.type, np.integer) or
            issubclass(da.values.dtype.type, np.float)):
        raise ValueError("da must be an array of integers or float")
    if band is None:
        if da.sizes['band'] != 1:
            warn('Multiple bands detected in da. Using band 1 as default band')
        band = 1
    return band

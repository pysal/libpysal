from .util import lat2SW
from .weights import WSP
import numpy as np
import pandas as pd
from warnings import warn

try:
    from xarray import DataArray
    from affine import Affine
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
    nodata = np.where(ser == da.nodatavals[0])[0]
    mask = np.ones((sw.shape[0],), dtype=np.bool)
    mask[nodata] = False
    sw = sw[mask]
    sw = sw[:, mask]
    ser = ser[ser != da.nodatavals[0]]
    index = ser.index
    wsp = WSP(sw, index=index)
    return wsp


def w2da(data, w, attrs, coords=None):
    """
    Creates DataArray object from passed data

    Arguments
    ---------
    data : array/list
       numpy 1d array or list with dimensionality conforming to w
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
    shape = attrs["shape"]
    dims = w.index.names
    if coords is not None:
        shape = tuple(len(value) for value in coords.values())
        dims = tuple(key for key in coords.keys())
    else:
        coords = {}
        nx, ny = shape[2], shape[1]
        transform = Affine(*attrs["transform"])
        x, _ = transform * (np.arange(nx) + 0.5, np.zeros(nx) + 0.5)
        _, y = transform * (np.zeros(ny) + 0.5, np.arange(ny) + 0.5)
        coords["band"] = np.ones(1)
        coords["y"] = y
        coords["x"] = x
    og = pd.MultiIndex.from_product([i for i in coords.values()], names=dims)
    ser = pd.Series(attrs["nodatavals"][0], index=og)
    ser[w.index] = data
    data = ser.to_numpy().reshape(shape)
    da = DataArray(data, coords=coords, dims=dims, attrs=attrs)
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
    da.attrs["shape"] = da.shape
    return band

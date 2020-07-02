from .set_operations import w_subset
from .util import lat2W, lat2SW
from .weights import WSP
import numpy as np
from warnings import warn

try:
    from xarray import DataArray
except ImportError:
    raise ImportError(
        "xarray must be installed to use this functionality")

__all__ = ['da_checker', 'da2W', 'da2WSP', 'w2da']


def da2W(da, criterion, **kwargs):
    """
    Create a W object from rasters(xarray.DataArray)

    Parameters
    ----------
    da         : xarray.DataArray
                 raster file accessed using xarray.open_rasterio method
    criterion  : {"rook", "queen"}
                 option for which kind of contiguity to build
    Returns
    -------
    w    : libpysal.weights.W
           instance of spatial weights class W
    """
    if criterion != 'rook':
        rook = False
    else:
        rook = True
    w = lat2W(*da[0].shape, rook, **kwargs)
    ser = da.to_series()
    id_order = np.where(ser != da.nodatavals[0])[0]
    w = w_subset(w, id_order)
    ser = ser[ser != da.nodatavals[0]]
    w.coords = ser.index
    attrs = da.attrs
    attrs["dims"] = da.dims
    attrs["shape"] = da.shape
    w.attrs = attrs
    return w


def da2WSP(da, criterion, **kwargs):
    """
    Generate a WSP object from rasters(xarray.DataArray)

    Parameters
    ----------
    da         : xarray.DataArray
                 raster file accessed using xarray.open_rasterio method
    criterion  : {"rook", "queen"}
                 option for which kind of contiguity to build
    Returns
    -------
    sw    : libpysal.weights.WSP
           instance of spatial weights class WSP
    """
    sw = lat2SW(*da[0].shape, criterion)
    ser = da.to_series()
    id_order = np.where(ser != da.nodatavals[0])[0]
    indices = np.where(ser == da.nodatavals[0])[0]
    mask = np.ones((sw.shape[0],), dtype=np.bool)
    mask[indices] = False
    for _ in range(2):
        sw = sw[mask]
        sw = sw.transpose()
        sw = sw.tocsr()
    sw = WSP(sw)
    sw.id_order = id_order
    ser = ser[ser != da.nodatavals[0]]
    sw.coords = ser.index
    attrs = da.attrs
    attrs["dims"] = da.dims
    attrs["shape"] = da.shape
    sw.attrs = attrs
    return sw


def w2da(data, w, coords=None):
    """
    converts calculated results to a DataArray

    Arguments
    ---------
    data    :   array
                data values stored in 1d array
    w       :   list/array
                Ordered sequence of IDs of weight object
    coords  :   Dictionary/xarray.core.coordinates.DataArrayCoordinates
                coordinates from original DataArray
    Returns
    -------

    da : xarray.DataArray
         instance of xarray.DataArray
    """
    attrs = w.attrs
    dims = attrs.pop('dims')
    shape = attrs.pop('shape')
    if coords is not None:
        shape = tuple(len(value) for value in coords.values())
        dims = tuple(key for key in coords.keys())
    else:
        try:
            from affine import Affine
        except ImportError:
            raise ImportError(
                "affine must be installed to use this functionality")
        coords = {}
        nx, ny = shape[2], shape[1]
        transform = Affine(*attrs["transform"])
        x, _ = transform * (np.arange(nx) + 0.5, np.zeros(nx) + 0.5)
        _, y = transform * (np.zeros(ny) + 0.5, np.arange(ny) + 0.5)
        coords["band"] = np.ones(1)
        coords["y"] = y
        coords["x"] = x
    n = shape[1]*shape[2]
    temp = np.full((n), attrs['nodatavals'][0])
    temp[w.id_order] = data
    data = temp.reshape((shape))
    da = DataArray(data=data, dims=dims, coords=coords, attrs=attrs)
    return da


def da_checker(da, band):
    """
    xarray dataarray checker
    Parameters
    ----------
    da      : xarray.DataArray
              raster file accessed using xarray.open_rasterio method
    band    : int
              user selected band
    Returns
    -------
    band    : int
              return default band value
    """
    if not isinstance(da, DataArray):
        raise TypeError("`da` must be an instance of xarray.DataArray")
    if da.ndim != 2:
        raise ValueError("`raster` must be 2D")
    if not (issubclass(da.values.dtype.type, np.integer) or
            issubclass(da.values.dtype.type, np.float)):
        raise ValueError("`da` must be an array of integers or float")
    if band is None:
        if da.sizes['band'] != 1:
            warn('Multiple bands detected in da. Using band 1 as default')
        band = 1
    return band

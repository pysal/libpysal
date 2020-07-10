from .set_operations import w_subset
from .util import lat2W, lat2SW
from .weights import WSP
from scipy import sparse
import numpy as np
from warnings import warn

try:
    from numba import njit, jit
except (ImportError, ModuleNotFoundError):

    def jit(*dec_args, **dec_kwargs):
        """
        decorator mimicking numba.jit
        """

        def intercepted_function(f, *f_args, **f_kwargs):
            return f

        return intercepted_function

    njit = jit

try:
    from xarray import DataArray
except ImportError:
    raise ImportError(
        "xarray must be installed to use this functionality")

__all__ = ['da_checker', 'da2W', 'da2WSP', 'w2da', 'da2W2', 'da2WSP2']


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


def da2W2(da, criterion, **kwargs):
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
    ser = da.to_series()
    indices = np.where(da.data.flatten() == da.nodatavals[0])[0]
    id_order = np.where(da.data.flatten() != da.nodatavals[0])[0]
    sw = _raster2SW(*da[0].shape, indices, criterion)
    w = WSP(sw, id_order.tolist()).to_W(**kwargs)
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


def da2WSP2(da, criterion):
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
    ser = da.to_series()
    indices = np.where(da.data.flatten() == da.nodatavals[0])[0]
    id_order = np.where(da.data.flatten() != da.nodatavals[0])[0]
    sw = _raster2SW(*da[0].shape, indices, criterion)
    sw = WSP(sw, id_order.tolist())
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


def _raster2SW(nrows, ncols, indices, criterion="rook"):
    """
    Create a sparse W matrix for a regular lattice.

    Parameters
    ----------

    nrows     : int
                number of rows
    ncols     : int
                number of columns
    criterion : {"rook", "queen", "bishop"}
                type of contiguity. Default is rook.
    indices   : list/np.array
                indices of missing values in flattened DataArray 

    Returns
    -------

    w : scipy.sparse.dia_matrix
        instance of a scipy sparse matrix
    """
    n = nrows * ncols
    data, offsets = _buildsw(nrows, ncols, criterion="rook")
    m = sparse.dia_matrix((data, offsets), shape=(n, n), dtype=np.int8)
    m = m + m.T
    mask = np.ones((m.shape[0],), dtype=np.bool)
    mask[indices] = False
    m = m[mask]
    m = m[:, mask]
    return m


@njit(fastmath=True)
def _buildsw(nrows, ncols, criterion="rook"):
    n = nrows * ncols
    diagonals = np.empty((4, n), dtype=np.int8)
    offsets = []
    if criterion == "rook" or criterion == "queen":
        d = np.ones(n, dtype=np.int8)
        for i in range(ncols - 1, n, ncols):
            d[i] = 0
        diagonals[0] = (d)
        offsets.append(-1)

        d = np.ones(n, dtype=np.int8)
        diagonals[1] = (d)
        offsets.append(-ncols)

    if criterion == "queen" or criterion == "bishop":
        d = np.ones(n, dtype=np.int8)
        for i in range(0, n, ncols):
            d[i] = 0
        diagonals[2] = (d)
        offsets.append(-(ncols - 1))

        d = np.ones(n, dtype=np.int8)
        for i in range(ncols - 1, n, ncols):
            d[i] = 0
        diagonals[3] = (d)
        offsets.append(-(ncols + 1))
    if criterion == "queen":
        data = diagonals
    else:
        data = diagonals[:2]
    offsets = np.array(offsets)
    return data, offsets

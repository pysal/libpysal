from .util import lat2SW
from .weights import WSP, W
import numpy as np
from warnings import warn

try:
    from xarray import DataArray
except ImportError:
    raise ImportError(
        "xarray must be installed to use this functionality")


__all__ = ['da2W', 'da2WSP', 'w2da', 'wsp2da', 'testDataArray']


def da2W(da, criterion="queen", layer=None, dims=None, **kwargs):
    """
    Create a W object from xarray.DataArray

    Parameters
    ----------
    da : xarray.DataArray
       Input 2D or 3D DataArray with shape=(layer, height, width)
    criterion : {"rook", "queen"}
       Type of contiguity. Default is queen.
    layer : int/string/float
       Select the layer of 3D DataArray with multiple layers
    dims : dictionary
       Pass custom dimensions for coordinates and layers if they
       do not belong to default dimensions, which are (band/time, y/lat, x/lon)
       e.g. dims = {"lat": "latitude", "lon": "longitude", "layer": "year"}
    **kwargs : keyword arguments
        Optional arguments for :class:`libpysal.weights.W`

    Returns
    -------
    w : libpysal.weights.W
       instance of spatial weights class W

    Examples
    --------
    >>> from libpysal.raster import da2W, testDataArray
    >>> da = testDataArray(time=False)
    >>> w = da2W(da, layer=2)
    >>> da.shape
    (3, 4, 4)
    >>> "%.3f"%w.pct_nonzero
    '28.571'
    >>> w[3] == {5: 1, 4: 1, 2: 1}
    True
    >>> w[6] == {5: 1}
    True
    >>> len(w.index)
    7

    See Also
    --------
    :class:`libpysal.weights.weights.W`
    """
    wsp = da2WSP(da, criterion, layer, dims)
    w = wsp.to_W(**kwargs)
    w.index = wsp.index
    return w


def da2WSP(da, criterion="queen", layer=None, dims=None):
    """
    Create a WSP object from xarray.DataArray

    Parameters
    ----------
    da : xarray.DataArray
       Input 2D or 3D DataArray with shape=(layer, height, width)
    criterion : {"rook", "queen"}
       Type of contiguity. Default is queen.
    layer : int/string/float
       Select the layer of 3D DataArray with multiple layers
    dims : dictionary
       Pass custom dimensions for coordinates and layers if they
       do not belong to default dimensions, which are (band/time, y/lat, x/lon)
       e.g. dims = {"lat": "latitude", "lon": "longitude", "layer": "year"}

    Returns
    -------
    w : libpysal.weights.WSP
       instance of spatial weights class WSP

    Examples
    --------
    >>> from libpysal.raster import da2WSP, testDataArray
    >>> da = testDataArray(time=False)
    >>> da.shape
    (3, 4, 4)
    >>> wsp = da2WSP(da, layer=2)
    >>> wsp.n
    7
    >>> pct_sp = wsp.sparse.nnz *1. / wsp.n**2
    >>> "%.3f"%pct_sp
    '0.286'
    >>> print(wsp.sparse[4].todense())
    [[0 0 0 1 0 1 0]]
    >>> len(w.index)
    7

    See Also
    --------
    :class:`libpysal.weights.weights.WSP`
    """
    layer_id = _da_checker(da, layer, dims)[0]
    shape = da.shape
    if layer_id:
        da = da[layer_id-1:layer_id]
        shape = da[0].shape
    sw = lat2SW(*shape, criterion)
    ser = da.to_series()
    if 'nodatavals' in da.attrs:
        mask = (ser != da.nodatavals[0]).to_numpy()
        sw = sw[mask]
        sw = sw[:, mask]
        ser = ser[ser != da.nodatavals[0]]
    index = ser.index
    wsp = WSP(sw, index=index)
    return wsp


def w2da(data, w, attrs={}, coords=None):
    """
    Creates DataArray object from passed data aligned with W object.

    Parameters
    ---------
    data : array/list/pd.Series
        1d array-like data with dimensionality conforming to w
    w : libpysal.weights.W
        Spatial weights object aligned with passed data
    attrs : Dictionary
        Attributes stored in dict related to DataArray, e.g. da.attrs
    coords : Dictionary/xarray.core.coordinates.DataArrayCoordinates
        Coordinates corresponding to DataArray, e.g. da.coords

    Returns
    -------
    da : xarray.DataArray
        instance of xarray.DataArray
    """
    if not isinstance(w, W):
        raise TypeError("w must be an instance of weights.W")
    if hasattr(w, 'index'):
        da = _index2da(data, w.index, attrs, coords)
    else:
        raise AttributeError("Cannot convert deprecated W with no index attribute")
    return da


def wsp2da(data, wsp, attrs={}, coords=None):
    """
    Creates DataArray object from passed data aligned with WSP object.

    Parameters
    ---------
    data : array/list/pd.Series
        1d array-like data with dimensionality conforming to w
    wsp : libpysal.weights.WSP
        Sparse weights object aligned with passed data
    attrs : Dictionary
        Attributes stored in dict related to DataArray, e.g. da.attrs
    coords : Dictionary/xarray.core.coordinates.DataArrayCoordinates
        coordinates corresponding to DataArray, e.g. da.coords

    Returns
    -------
    da : xarray.DataArray
        instance of xarray.DataArray
    """
    if not isinstance(wsp, WSP):
        raise TypeError("wsp must be an instance of weights.WSP")
    if hasattr(wsp, 'index'):
        da = _index2da(data, wsp.index, attrs, coords)
    else:
        raise AttributeError("Cannot convert deprecated wsp object with no index attribute")
    return da


def testDataArray(shape=(3, 4, 4), time=False, rand=False, missing_vals=True):
    """
    Creates 3D test DataArray object

    Parameters
    ---------
    shape : tuple
        Tuple containing shape of the DataArray aligned with
        following dimension= (layer, y, x)
    time : boolean
        Type of layer, if True then layer=time else layer=band
    rand : boolean
        If True, creates a DataArray filled with unique and random data.
        Default is false (generates seeded random data)
    missing_vals : boolean
        Create a DataArray filled with missing values. Default is True.

    Returns
    -------
    da : xarray.DataArray
        instance of xarray.DataArray
    """
    if not rand:
        np.random.seed(123)
    r1 = np.random.randint(100)
    r2 = np.random.randint(100)
    layer = "time" if time else "band"
    dims = (layer, 'y', 'x')
    if time:
        layers = np.arange(np.datetime64('2020-07-30'),
                           shape[0], dtype='datetime64[D]')
    else:
        layers = np.arange(1, shape[0]+1)
    coords = {
        dims[0]: layers,
        dims[1]: np.linspace(r1+shape[1]*0.1, r1+0.1, shape[1]),
        dims[2]: np.linspace(r2+0.1, r2+shape[2]*0.1, shape[2])
    }
    data = np.random.randint(0, 255, shape)
    attrs = {'nodatavals': (-32768.0,)}
    if missing_vals:
        miss_ids = np.where(np.random.randint(2, size=shape) == 1)
        data[miss_ids] = attrs['nodatavals'][0]
    da = DataArray(data, coords, dims, attrs=attrs)
    return da


def _da_checker(da, layer, dims):
    """
    xarray.dataarray checker for raster interface

    Parameters
    ----------
    da : xarray.DataArray
       Input 2D or 3D DataArray with shape=(layer, height, width)
    layer : int/string/float
       Select the layer of 3D DataArray with multiple layers
    dims : dictionary
       Pass custom dimensions for coordinates and layers if they
       do not belong to default dimensions, which are (band/time, y/lat, x/lon)
       e.g. dims = {"lat": "latitude", "lon": "longitude", "layer": "year"}

    Returns
    -------
    layer_id : int
        Returns the index of layer
    dims : dictionary
        Mapped dimensions of the DataArray
    """
    if not isinstance(da, DataArray):
        raise TypeError("da must be an instance of xarray.DataArray")
    if da.ndim not in [2, 3]:
        raise ValueError("da must be 2D or 3D")
    if dims is None:
        # default dimensions
        dims = {
            "lon": "x" if hasattr(da, "x") else "lon",
            "lat": "y" if hasattr(da, "y") else "lat"
        }
        if da.ndim == 3:
            dims["layer"] = "band" if hasattr(da, "band") else "time"
    if da.ndim == 3:
        layer_id = 1
        if layer is None:
            if da.sizes[dims["layer"]] != 1:
                warn('Multiple layers detected in da. Using first layer as default.')
        else:
            layer_id += tuple(da[dims["layer"]]).index(layer)
    else:
        layer_id = None
    return layer_id, dims


def _index2da(data, index, attrs, coords):
    """
    Creates DataArray object from passed data

    Parameters
    ---------
    data : array/list/pd.Series
        1d array-like data with dimensionality conforming to w
    w : libpysal.weights.W
        Spatial weights object aligned with passed data
    attrs : Dictionary
        Attributes stored in dict related to DataArray, e.g. da.attrs
    coords : Dictionary/xarray.core.coordinates.DataArrayCoordinates
        coordinates corresponding to DataArray, e.g. da[n-1:n].coords

    Returns
    -------
    da : xarray.DataArray
        instance of xarray.DataArray
    """
    data = np.array(data).flatten()
    idx = index
    indexer = tuple(idx.codes)
    if coords is None:
        dims = idx.names
        shape = tuple(lev.size for lev in idx.levels)
        missing = np.prod(shape) > idx.shape[0]
        if missing:
            if 'nodatavals' in attrs:
                fill_value = attrs["nodatavals"][0]
            else:
                min_data = np.min(data)
                fill_value = min_data - 1 if min_data < 0 else -1
                attrs["nodatavals"] = tuple([fill_value])
            data_complete = np.full(shape, fill_value, data.dtype)
        else:
            data_complete = np.empty(shape, data.dtype)
        data_complete[indexer] = data
        coords = {}
        for dim, lev in zip(dims, idx.levels):
            coords[dim] = lev.to_numpy()
    else:
        shape = tuple(value.size for value in coords.values())
        dims = tuple(key for key in coords.keys())
        data_complete = np.full(shape, attrs["nodatavals"][0], data.dtype)
        data_complete[indexer] = data
    da = DataArray(data_complete, coords=coords, dims=dims, attrs=attrs)
    return da

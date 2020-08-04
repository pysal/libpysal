from .util import lat2SW
from .weights import WSP, W
import numpy as np
from warnings import warn

__all__ = ['da2W', 'da2WSP', 'w2da', 'wsp2da', 'testDataArray']


def da2W(da, criterion="queen", layer=None, dims={}, **kwargs):
    """
    Create a W object from xarray.DataArray

    Parameters
    ----------
    da : xarray.DataArray
        Input 2D or 3D DataArray with shape=(layer, lat, lon)
    criterion : {"rook", "queen"}
        Type of contiguity. Default is queen.
    layer : int/string/float
        Select the layer of 3D DataArray with multiple layers.
    dims : dictionary
        Pass dimensions for coordinates and layers if they do not
        belong to default dimensions, which are (band/time, y/lat, x/lon)
        e.g. dims = {"lat": "latitude", "lon": "longitude", "layer": "year"}
        Default is {} empty dictionary.
    **kwargs : keyword arguments
        Optional arguments for :class:`libpysal.weights.W`

    Returns
    -------
    w : libpysal.weights.W
       instance of spatial weights class W

    Examples
    --------
    >>> from libpysal.raster import da2W, testDataArray
    >>> da = testDataArray().rename(
            {'band': 'layer', 'x': 'longitude', 'y': 'latitude'})
    >>> da.dims
    ('layer', 'latitude', 'longitude')
    >>> dims = {"layer": "layer", "lat": "latitude", "lon": "longitude"}
    >>> w = da2W(da, layer=2, dims=dims)
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


def da2WSP(da, criterion="queen", layer=None, dims={}):
    """
    Create a WSP object from xarray.DataArray

    Parameters
    ----------
    da : xarray.DataArray
        Input 2D or 3D DataArray with shape=(layer, lat, lon)
    criterion : {"rook", "queen"}
        Type of contiguity. Default is queen.
    layer : int/string/float
        Select the layer of 3D DataArray with multiple layers.
    dims : dictionary
        Pass dimensions for coordinates and layers if they do not
        belong to default dimensions, which are (band/time, y/lat, x/lon)
        e.g. dims = {"lat": "latitude", "lon": "longitude", "layer": "year"}
        Default is {} empty dictionary.

    Returns
    -------
    w : libpysal.weights.WSP
       instance of spatial weights class WSP

    Examples
    --------
    >>> from libpysal.raster import da2WSP, testDataArray
    >>> da = testDataArray().rename(
            {'band': 'layer', 'x': 'longitude', 'y': 'latitude'})
    >>> da.dims
    ('layer', 'latitude', 'longitude')
    >>> da.shape
    (3, 4, 4)
    >>> dims = {"layer": "layer", "lat": "latitude", "lon": "longitude"}
    >>> wsp = da2WSP(da, layer=2, dims=dims)
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
    id_order = np.arange(len(ser))
    if 'nodatavals' in da.attrs:
        mask = (ser != da.nodatavals[0]).to_numpy()
        # temp
        id_order = np.where(mask)[0]
        sw = sw[mask]
        sw = sw[:, mask]
        ser = ser[ser != da.nodatavals[0]]
    index = ser.index
    wsp = WSP(sw, id_order=id_order.tolist(), index=index)
    return wsp


def w2da(data, w, attrs={}, coords=None):
    """
    Creates xarray.DataArray object from passed data aligned with W object.

    Parameters
    ---------
    data : array/list/pd.Series
        1d array-like data with dimensionality conforming to w
    w : libpysal.weights.W
        Spatial weights object aligned with passed data
    attrs : Dictionary
        Attributes stored in dict related to DataArray, e.g. da.attrs
        Default is {} empty dictionary.
    coords : Dictionary/xarray.core.coordinates.DataArrayCoordinates
        Coordinates corresponding to DataArray, e.g. da.coords

    Returns
    -------
    da : xarray.DataArray
        instance of xarray.DataArray

    Examples
    --------
    >>> from libpysal.raster import da2W, testDataArray, w2da
    >>> da = testDataArray()
    >>> da.shape
    (3, 4, 4)
    >>> w = da2W(da, layer=2)
    >>> data = np.random.randint(0, 255, len(w.index))
    >>> da1 = w2da(data, w)

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
    Creates xarray.DataArray object from passed data aligned with WSP object.

    Parameters
    ---------
    data : array/list/pd.Series
        1d array-like data with dimensionality conforming to wsp
    wsp : libpysal.weights.WSP
        Sparse weights object aligned with passed data
    attrs : Dictionary
        Attributes stored in dict related to DataArray, e.g. da.attrs
        Default is {} empty dictionary.
    coords : Dictionary/xarray.core.coordinates.DataArrayCoordinates
        coordinates corresponding to DataArray, e.g. da.coords

    Returns
    -------
    da : xarray.DataArray
        instance of xarray.DataArray

    Examples
    --------
    >>> from libpysal.raster import da2WSP, testDataArray, wsp2da
    >>> da = testDataArray()
    >>> da.shape
    (3, 4, 4)
    >>> wsp = da2WSP(da, layer=2)
    >>> data = np.random.randint(0, 255, len(wsp.index))
    >>> da1 = w2da(data, wsp)

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
    Creates 2 or 3 dimensional test xarray.DataArray object

    Parameters
    ---------
    shape : tuple
        Tuple containing shape of the DataArray aligned with
        following dimension = (lat, lon) or (layer, lat, lon)
        Default shape = (3, 4, 4)
    time : boolean
        Type of layer, if True then layer=time else layer=band
        Default is False.
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
    try:
        from xarray import DataArray
    except ImportError:
        raise ModuleNotFoundError(
            "xarray must be installed to use this functionality")
    if not rand:
        np.random.seed(12345)
    coords = {}
    n = len(shape)
    if n != 2:
        layer = "time" if time else "band"
        dims = (layer, 'y', 'x')
        if time:
            layers = np.arange(
                np.datetime64('2020-07-30'),
                shape[0], dtype='datetime64[D]'
            )
        else:
            layers = np.arange(1, shape[0]+1)
        coords[dims[-3]] = layers
    coords[dims[-2]] = np.linspace(90, -90, shape[-2])
    coords[dims[-1]] = np.linspace(-180, 180, shape[-1])
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
        Input 2D or 3D DataArray with shape=(layer, lat, lon)
    layer : int/string/float
        Select the layer of 3D DataArray with multiple layers
    dims : dictionary
        Pass dimensions for coordinates and layers if they do not
        belong to default dimensions, which are (band/time, y/lat, x/lon)
        e.g. dims = {"lat": "latitude", "lon": "longitude", "layer": "year"}

    Returns
    -------
    layer_id : int
        Returns the index of layer
    dims : dictionary
        Mapped dimensions of the DataArray
    """
    try:
        from xarray import DataArray
    except ImportError:
        raise ModuleNotFoundError(
            "xarray must be installed to use this functionality")
    if not isinstance(da, DataArray):
        raise TypeError("da must be an instance of xarray.DataArray")
    if da.ndim not in [2, 3]:
        raise ValueError("da must be 2D or 3D")
    if not (np.issubdtype(da.values.dtype, np.integer) or
            np.issubdtype(da.values.dtype, np.floating)):
        raise ValueError(
            "da must be an array of integers or float")
    # default dimensions
    def_dims = {
        "lon": dims["lon"] if 'lon' in dims else (
            "x" if hasattr(da, "x") else "lon"),
        "lat": dims["lat"] if 'lat' in dims else (
            "y" if hasattr(da, "y") else "lat")
    }
    if da.ndim == 3:
        def_dims["layer"] = dims["layer"] if 'layer' in dims else (
            "band" if hasattr(da, "band") else "time")
    if da.ndim == 3:
        layer_id = 1
        if layer is None:
            if da.sizes[def_dims["layer"]] != 1:
                warn('Multiple layers detected. Using first layer as default.')
        else:
            layer_id += tuple(da[def_dims["layer"]]).index(layer)
    else:
        layer_id = None
    return layer_id, def_dims


def _index2da(data, index, attrs, coords):
    """
    Creates xarray.DataArray object from passed data

    Parameters
    ---------
    data : array/list/pd.Series
        1d array-like data with dimensionality conforming to index
    index : pd.MultiIndex
        indices of the DataArray when converted to pd.Series
    attrs : Dictionary
        Attributes stored in dict related to DataArray, e.g. da.attrs
    coords : Dictionary/xarray.core.coordinates.DataArrayCoordinates
        coordinates corresponding to DataArray, e.g. da[n-1:n].coords

    Returns
    -------
    da : xarray.DataArray
        instance of xarray.DataArray
    """
    try:
        from xarray import DataArray
    except ImportError:
        raise ModuleNotFoundError(
            "xarray must be installed to use this functionality")
    data = np.array(data).flatten()
    idx = index
    dims = idx.names
    indexer = tuple(idx.codes)
    shape = tuple(lev.size for lev in idx.levels)
    if coords is None:
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
        fill = attrs["nodatavals"][0] if 'nodatavals' in attrs else 0
        data_complete = np.full(shape, fill, data.dtype)
        data_complete[indexer] = data
        data_complete = data_complete[:, ::-1]
    da = DataArray(data_complete, coords=coords, dims=dims, attrs=attrs)
    return da.sortby(dims[-2], False)

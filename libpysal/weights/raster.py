import numpy as np
from warnings import warn
from numba import njit
import os
from scipy import sparse
from .weights import WSP, W

__all__ = ['da2W', 'da2WSP', 'w2da', 'wsp2da', 'testDataArray']


def da2W(da, criterion="queen", layer=None, dims={}, k=1, n_jobs=1, **kwargs):
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
    k : int
        Order of queen contiguity.
    n_jobs : int
        Number of cores to be used in the sparse weight construction. If -1,
        all available cores are used.
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
    wsp = da2WSP(da, criterion, layer, dims, k, n_jobs)
    w = wsp.to_W(**kwargs)
    w.index = wsp.index
    return w


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


def da2WSP(da, criterion="queen", layer=None, dims={}, k=1, n_jobs=1):
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
    k : int
        Order of queen contiguity, if k=1 max neighbors of a point
        will be 8 just like queen contiguity, for k=2 it'll be 8+16 and so on
        Default is 1
    n_jobs : int
        Number of cores to be used in the sparse weight construction. If -1,
        all available cores are used.

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
    ser = da.to_series()
    mask = (ser != da.nodatavals[0]).to_numpy()
    ids = np.where(mask)[0]
    dtype = np.int32 if (shape[0] * shape[1]) < 46340**2 else np.int64
    n = len(ids)
    ser = ser[ser != da.nodatavals[0]]
    index = ser.index

    if n_jobs != 1:
        try:
            import joblib
        except (ModuleNotFoundError, ImportError):
            warn(
                f"Parallel processing is requested (n_jobs={n_jobs}),"
                f" but joblib cannot be imported. n_jobs will be set"
                f" to 1.",
                stacklevel=2,
            )
            n_jobs = 1
    if n_jobs == 1:
        wsp = WSP(
            sparse.coo_matrix(
                _SWbuilder(
                    *shape,
                    ids,
                    _idmap(ids, mask, dtype),
                    criterion,
                    k,
                    dtype,
                ),
                shape=(n, n),
                dtype=np.int8,
            ),
            index=index,
            # temp
            id_order=ids.tolist()
        )
    else:
        if n_jobs == -1:
            n_jobs = os.cpu_count()
        # Parallel implementation
        wsp = WSP(
            sparse.coo_matrix(
                _parSWbuilder(
                    *shape,
                    ids,
                    _idmap(ids, mask, dtype),
                    criterion,
                    k,
                    dtype,
                    n_jobs,
                ),
                shape=(n, n),
                dtype=np.int8,
            ),
            index=index,
            # temp
            id_order=ids.tolist()
        )

    return wsp


@njit(fastmath=True)
def _idmap(ids, mask, dtype):
    """
    Utility function computes id_map of non-missing raster data

    Parameters
    ----------
    ids : ndarray
        1D array containing ids of non-missing raster data
    mask : ndarray
        1D array mask array
    dtype : type
        Data type of the id_map array

    Returns
    -------
    id_map : ndarray
        1D array containing id_maps of non-missing raster data
    """
    id_map = mask * 1
    id_map[ids] = np.arange(len(ids), dtype=dtype)
    return id_map


@njit(fastmath=True)
def _SWbuilder(
    nrows,
    ncols,
    ids,
    id_map,
    criterion,
    k,
    dtype,
):
    """
    Computes data and orders rows, cols, data for a single chunk

    Parameters
    ----------
    nrows : int
        Number of rows in the raster data
    ncols : int
        Number of columns in the raster data
    ids : ndarray
        1D array containing ids of non-missing raster data
    id_map : ndarray
        1D array containing id_maps of non-missing raster data
    criterion : str
        Type of contiguity.
    k : int
        Order of queen contiguity, if k=1 max neighbors of a point
        will be 8 just like queen contiguity, for k=2 it'll be 8+16 and so on
        Default is 1
    dtype : type
        Data type of the id_map array

    Returns
    -------
    data : ndarray
        1D ones array containing weight of each neighbor
    rows : ndarray
        1D ones array containing row value of each id
        in the sparse weight object
    cols : ndarray
        1D ones array containing columns value of each id
        in the sparse weight object
    """
    rows, cols = compute_chunk(
        nrows,
        ncols,
        ids,
        id_map,
        criterion,
        k,
        dtype,
    )
    data = np.ones_like(rows, dtype=np.int8)
    return (data, (rows, cols))


@njit(fastmath=True, nogil=True)
def compute_chunk(
    nrows,
    ncols,
    ids,
    id_map,
    criterion,
    k,
    dtype,
):
    """
    Computes rows cols for a single chunk

    Parameters
    ----------
    nrows : int
        Number of rows in the raster data
    ncols : int
        Number of columns in the raster data
    ids : ndarray
        1D array containing ids of non-missing raster data
    id_map : ndarray
        1D array containing id_maps of non-missing raster data
    criterion : str
        Type of contiguity.
    k : int
        Order of queen contiguity, if k=1 max neighbors of a point
        will be 8 just like queen contiguity, for k=2 it'll be 8+16 and so on
        Default is 1
    dtype : type
        Data type of the id_map array

    Returns
    -------
    rows : ndarray
        1D ones array containing row value of each id
        in the sparse weight object
    cols : ndarray
        1D ones array containing columns value of each id
        in the sparse weight object
    """
    n = len(ids)
    d = 4 if criterion == "rook" else 8  # -> used for row, col preallocation
    if k > 1:
        # only queen contiguity is supported right now
        # this will capture all the circling neighbors
        # of order <= K.
        d = int((k/2)*(2*8+(k-1)*8))
    # preallocating rows and cols
    rows = np.empty(d*n, dtype=dtype)
    cols = np.empty_like(rows)
    ni = 0
    for order in range(1, k+1):
        for i in range(n):
            id_i = ids[i]
            og_id = id_map[id_i]
            if ((id_i+order) % ncols) >= order:
                # east neighbor
                id_neighbor = id_map[id_i + order]
                if id_neighbor:
                    rows[ni], cols[ni] = og_id, id_neighbor
                    ni += 1
                    rows[ni], cols[ni] = id_neighbor, og_id
                    ni += 1
                for j in range(order-1):
                    if (id_i // ncols) < (nrows - j - 1):
                        id_neighbor = id_map[(id_i+order)+(ncols*(j+1))]
                        if id_neighbor:
                            rows[ni], cols[ni] = og_id, id_neighbor
                            ni += 1
                            rows[ni], cols[ni] = id_neighbor, og_id
                            ni += 1
                    if (id_i // ncols) >= j+1:
                        id_neighbor = id_map[(id_i+order)-(ncols*(j+1))]
                        if id_neighbor:
                            rows[ni], cols[ni] = og_id, id_neighbor
                            ni += 1
                            rows[ni], cols[ni] = id_neighbor, og_id
                            ni += 1
            if (id_i // ncols) < (nrows - order):
                # south neighbor
                id_neighbor = id_map[id_i+(ncols*order)]
                if id_neighbor:
                    rows[ni], cols[ni] = og_id, id_neighbor
                    ni += 1
                    rows[ni], cols[ni] = id_neighbor, og_id
                    ni += 1
                for j in range(order-1):
                    if (id_i % ncols) >= j+1:
                        id_neighbor = id_map[id_i+(ncols*order)-j-1]
                        if id_neighbor:
                            rows[ni], cols[ni] = og_id, id_neighbor
                            ni += 1
                            rows[ni], cols[ni] = id_neighbor, og_id
                            ni += 1
                    if ((id_i+j+1) % ncols) >= j+1:
                        id_neighbor = id_map[id_i+(ncols*order)+j+1]
                        if id_neighbor:
                            rows[ni], cols[ni] = og_id, id_neighbor
                            ni += 1
                            rows[ni], cols[ni] = id_neighbor, og_id
                            ni += 1
                if d != 4:
                    if (id_i % ncols) >= order:
                        # south-west neighbor
                        id_neighbor = id_map[id_i+(ncols*order)-order]
                        if id_neighbor:
                            rows[ni], cols[ni] = og_id, id_neighbor
                            ni += 1
                            rows[ni], cols[ni] = id_neighbor, og_id
                            ni += 1
                    if ((id_i+order) % ncols) >= order:
                        # south-east neighbor
                        id_neighbor = id_map[id_i+(ncols*order)+order]
                        if id_neighbor:
                            rows[ni], cols[ni] = og_id, id_neighbor
                            ni += 1
                            rows[ni], cols[ni] = id_neighbor, og_id
                            ni += 1
    ni_arr = np.arange(ni, dtype=dtype)
    return rows[ni_arr, ], cols[ni_arr, ]


@njit(fastmath=True)
def chunk_generator(
    n_jobs,
    starts,
    ids,
):
    """
    Construct chunks to iterate over within numba in parallel

    Parameters
    ----------
    n_jobs : int
        Number of cores to be used in the sparse weight construction. If -1,
        all available cores are used.
    starts : ndarray
        (n_chunks+1,) array of positional starts for ids chunk
    ids : ndarray
        1D array containing ids of non-missing raster data

    Yields
    ------
    ids_chunk : numpy.ndarray
        (n_chunk,) array containing the chunk of non-missing raster data
    """
    chunk_size = starts[1] - starts[0]
    for i in range(n_jobs):
        start = starts[i]
        ids_chunk = ids[start: (start + chunk_size)]
        yield (ids_chunk,)


def _parSWbuilder(
    nrows,
    ncols,
    ids,
    id_map,
    criterion,
    k,
    dtype,
    n_jobs,
):
    """
    Computes data and orders rows, cols, data in parallel using numba

    Parameters
    ----------
    nrows : int
        Number of rows in the raster data
    ncols : int
        Number of columns in the raster data
    ids : ndarray
        1D array containing ids of non-missing raster data
    id_map : ndarray
        1D array containing id_maps of non-missing raster data
    criterion : str
        Type of contiguity.
    k : int
        Order of queen contiguity, if k=1 max neighbors of a point
        will be 8 just like queen contiguity, for k=2 it'll be 8+16 and so on
        Default is 1
    dtype : type
        Data type of the id_map array
    n_jobs : int
        Number of cores to be used in the sparse weight construction. If -1,
        all available cores are used.

    Returns
    -------
    data : ndarray
        1D ones array containing weight of each neighbor
    rows : ndarray
        1D ones array containing row value of each id
        in the sparse weight object
    cols : ndarray
        1D ones array containing columns value of each id
        in the sparse weight object
    """
    from joblib import Parallel, delayed, parallel_backend

    n = len(ids)
    chunk_size = n // n_jobs + 1
    starts = np.arange(n_jobs + 1) * chunk_size
    chunk = chunk_generator(n_jobs, starts, ids)
    with parallel_backend("threading"):
        worker_out = Parallel(n_jobs=n_jobs)(
            delayed(compute_chunk)(
                nrows,
                ncols,
                *pars,
                id_map,
                criterion,
                k,
                dtype
            )
            for pars in chunk
        )
    rows = np.concatenate([i[0] for i in worker_out])
    cols = np.concatenate([i[1] for i in worker_out])
    data = np.ones_like(rows, dtype=np.int8)
    return (data, (rows, cols))

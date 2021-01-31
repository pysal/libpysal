from .util import lat2SW
from .weights import WSP, W
import numpy as np
from warnings import warn
import os
import sys
from scipy import sparse

if os.path.basename(sys.argv[0]) in ("pytest", "py.test"):

    def jit(*dec_args, **dec_kwargs):
        """
        decorator mimicking numba.jit
        """

        def intercepted_function(f, *f_args, **f_kwargs):
            return f

        return intercepted_function


else:
    from ..common import jit

__author__ = "Mragank Shekhar <yesthisismrshekhar@gmail.com>"

__all__ = ["da2W", "da2WSP", "w2da", "wsp2da", "testDataArray"]


def da2W(
    da,
    criterion="queen",
    z_value=None,
    coords_labels={},
    k=1,
    include_nodata=False,
    n_jobs=1,
    **kwargs,
):
    """
    Create a W object from xarray.DataArray with an additional
    attribute index containing coordinate values of the raster
    in the form of Pandas.Index/MultiIndex.

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
        e.g. coords_labels = {"y_label": "latitude", "x_label": "longitude", "z_label": "year"}
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
    **kwargs : keyword arguments
        Optional arguments for :class:`libpysal.weights.W`

    Returns
    -------
    w : libpysal.weights.W
       instance of spatial weights class W with an index attribute

    Notes
    -----
    1. Lower order contiguities are also selected.
    2. Returned object contains `index` attribute that includes a
    `Pandas.MultiIndex` object from the DataArray.

    Examples
    --------

    >>> from libpysal.weights.raster import da2W, testDataArray
    >>> da = testDataArray().rename(
            {'band': 'layer', 'x': 'longitude', 'y': 'latitude'})
    >>> da.dims
    ('layer', 'latitude', 'longitude')
    >>> da.shape
    (3, 4, 4)
    >>> da.coords
    Coordinates:
        * layer      (layer) int64 1 2 3
        * latitude   (latitude) float64 90.0 30.0 -30.0 -90.0
        * longitude  (longitude) float64 -180.0 -60.0 60.0 180.0
    >>> da.attrs
    {'nodatavals': (-32768.0,)}
    >>> coords_labels = {
        "z_label": "layer",
        "y_label": "latitude",
        "x_label": "longitude"
    }
    >>> w = da2W(da, z_value=2, coords_labels=coords_labels)
    >>> "%.3f"%w.pct_nonzero
    '30.000'
    >>> w[(2, 90.0, 180.0)] == {(2, 90.0, 60.0): 1, (2, 30.0, 180.0): 1}
    True
    >>> len(w.index)
    10
    >>> w.index[:2]
    MultiIndex([(2, 90.0,  60.0),
                (2, 90.0, 180.0)],
               names=['layer', 'latitude', 'longitude'])

    See Also
    --------
    :class:`libpysal.weights.weights.W`
    """
    warn(
        "You are trying to build a full W object from "
        "xarray.DataArray (raster) object. This computation "
        "can be very slow and not scale well. It is recommended, "
        "if possible, to instead build WSP object, which is more "
        "efficient and faster. You can do this by using da2WSP method."
    )
    wsp = da2WSP(da, criterion, z_value, coords_labels, k, include_nodata, n_jobs)
    w = wsp.to_W(**kwargs)

    # temp addition of index attribute
    w.index = wsp.index
    return w


def da2WSP(
    da,
    criterion="queen",
    z_value=None,
    coords_labels={},
    k=1,
    include_nodata=False,
    n_jobs=1,
):
    """
    Create a WSP object from xarray.DataArray with an additional
    attribute index containing coordinate values of the raster
    in the form of Pandas.Index/MultiIndex.

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
        e.g. coords_labels = {"y_label": "latitude", "x_label": "longitude", "z_label": "year"}
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
    wsp : libpysal.weights.WSP
       instance of spatial weights class WSP with an index attribute

    Notes
    -----
    1. Lower order contiguities are also selected.
    2. Returned object contains `index` attribute that includes a
    `Pandas.MultiIndex` object from the DataArray.

    Examples
    --------
    >>> from libpysal.weights.raster import da2WSP, testDataArray
    >>> da = testDataArray().rename(
            {'band': 'layer', 'x': 'longitude', 'y': 'latitude'})
    >>> da.dims
    ('layer', 'latitude', 'longitude')
    >>> da.shape
    (3, 4, 4)
    >>> da.coords
    Coordinates:
        * layer      (layer) int64 1 2 3
        * latitude   (latitude) float64 90.0 30.0 -30.0 -90.0
        * longitude  (longitude) float64 -180.0 -60.0 60.0 180.0
    >>> da.attrs
    {'nodatavals': (-32768.0,)}
    >>> coords_labels = {
        "z_label": "layer",
        "y_label": "latitude",
        "x_label": "longitude"
    }
    >>> wsp = da2WSP(da, z_value=2, coords_labels=coords_labels)
    >>> wsp.n
    10
    >>> pct_sp = wsp.sparse.nnz *1. / wsp.n**2
    >>> "%.3f"%pct_sp
    '0.300'
    >>> print(wsp.sparse[4].todense())
    [[0 0 1 0 0 1 1 1 0 0]]
    >>> wsp.index[:2]
    MultiIndex([(2, 90.0,  60.0),
                (2, 90.0, 180.0)],
               names=['layer', 'latitude', 'longitude'])

    See Also
    --------
    :class:`libpysal.weights.weights.WSP`
    """
    z_id, coords_labels = _da_checker(da, z_value, coords_labels)
    shape = da.shape
    if z_id:
        slice_dict = {}
        slice_dict[coords_labels["z_label"]] = 0
        shape = da[slice_dict].shape
        slice_dict[coords_labels["z_label"]] = slice(z_id - 1, z_id)
        da = da[slice_dict]

    ser = da.to_series()
    dtype = np.int32 if (shape[0] * shape[1]) < 46340 ** 2 else np.int64
    if "nodatavals" in da.attrs and da.attrs["nodatavals"]:
        mask = (ser != da.attrs["nodatavals"][0]).to_numpy()
        ids = np.where(mask)[0]
        id_map = _idmap(ids, mask, dtype)
        ser = ser[ser != da.attrs["nodatavals"][0]]
    else:
        ids = np.arange(len(ser), dtype=dtype)
        id_map = ids.copy()

    n = len(ids)

    try:
        import numba
    except (ModuleNotFoundError, ImportError):
        warn(
            "numba cannot be imported, parallel processing "
            "and include_nodata functionality will be disabled. "
            "falling back to slower method"
        )
        include_nodata = False
        # Fallback method to build sparse matrix
        sw = lat2SW(*shape, criterion)
        if "nodatavals" in da.attrs and da.attrs["nodatavals"]:
            sw = sw[mask]
            sw = sw[:, mask]

    else:
        k_nas = k if include_nodata else 1

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
            sw_tup = _SWbuilder(
                *shape, ids, id_map, criterion, k_nas, dtype
            )  # -> (data, (row, col))
        else:
            if n_jobs == -1:
                n_jobs = os.cpu_count()
            # Parallel implementation
            sw_tup = _parSWbuilder(
                *shape, ids, id_map, criterion, k_nas, dtype, n_jobs
            )  # -> (data, (row, col))

        sw = sparse.csr_matrix(sw_tup, shape=(n, n), dtype=np.int8,)

    # Higher_order functionality, this uses idea from
    # libpysal#313 for adding higher order neighbors.
    # Since diagonal elements are also added in the result,
    # this method set the diagonal elements to zero and
    # then eliminate zeros from the data. This changes the
    # sparcity of the csr_matrix !!
    if k > 1 and not include_nodata:
        sw = sum(map(lambda x: sw ** x, range(1, k + 1)))
        sw.setdiag(0)
        sw.eliminate_zeros()
        sw.data[:] = np.ones_like(sw.data, dtype=np.int8)

    index = ser.index
    wsp = WSP(sw, index=index)
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
    >>> w = da2W(da, z_value=2)
    >>> data = np.random.randint(0, 255, len(w.index))
    >>> da1 = w2da(data, w)

    """
    if not isinstance(w, W):
        raise TypeError("w must be an instance of weights.W")
    if hasattr(w, "index"):
        da = _index2da(data, w.index, attrs, coords)
    else:
        raise AttributeError(
            "This method requires `w` object to include `index` attribute that is built as a `pandas.MultiIndex` object."
        )
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
    >>> wsp = da2WSP(da, z_value=2)
    >>> data = np.random.randint(0, 255, len(wsp.index))
    >>> da1 = w2da(data, wsp)

    """
    if not isinstance(wsp, WSP):
        raise TypeError("wsp must be an instance of weights.WSP")
    if hasattr(wsp, "index"):
        da = _index2da(data, wsp.index, attrs, coords)
    else:
        raise AttributeError(
            "This method requires `wsp` object to include `index` attribute that is built as a `pandas.MultiIndex` object."
        )
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
        raise ModuleNotFoundError("xarray must be installed to use this functionality")
    if not rand:
        np.random.seed(12345)
    coords = {}
    n = len(shape)
    if n != 2:
        layer = "time" if time else "band"
        dims = (layer, "y", "x")
        if time:
            layers = np.arange(
                np.datetime64("2020-07-30"), shape[0], dtype="datetime64[D]"
            )
        else:
            layers = np.arange(1, shape[0] + 1)
        coords[dims[-3]] = layers
    else:
        dims = ("y", "x")
    coords[dims[-2]] = np.linspace(90, -90, shape[-2])
    coords[dims[-1]] = np.linspace(-180, 180, shape[-1])
    data = np.random.randint(0, 255, shape)
    attrs = {}
    if missing_vals:
        attrs["nodatavals"] = (-32768.0,)
        miss_ids = np.where(np.random.randint(2, size=shape) == 1)
        data[miss_ids] = attrs["nodatavals"][0]
    da = DataArray(data, coords, dims, attrs=attrs)
    return da


def _da_checker(da, z_value, coords_labels):
    """
    xarray.dataarray checker for raster interface

    Parameters
    ----------
    da : xarray.DataArray
        Input 2D or 3D DataArray with shape=(z, y, x)
    z_value : int/string/float
        Select the z_value of 3D DataArray with multiple layers.
    coords_labels : dictionary
        Pass dimension labels for coordinates and layers if they do not
        belong to default dimensions, which are (band/time, y/lat, x/lon)
        e.g. coords_labels = {"y_label": "latitude", "x_label": "longitude", "z_label": "year"}
        Default is {} empty dictionary.

    Returns
    -------
    z_id : int
        Returns the index of layer
    dims : dictionary
        Mapped dimensions of the DataArray
    """
    try:
        from xarray import DataArray
    except ImportError:
        raise ModuleNotFoundError("xarray must be installed to use this functionality")

    if not isinstance(da, DataArray):
        raise TypeError("da must be an instance of xarray.DataArray")
    if da.ndim not in [2, 3]:
        raise ValueError("da must be 2D or 3D")
    if not (
        np.issubdtype(da.values.dtype, np.integer)
        or np.issubdtype(da.values.dtype, np.floating)
    ):
        raise ValueError("da must be an array of integers or float")

    # default dimensions
    def_labels = {
        "x_label": coords_labels["x_label"]
        if "x_label" in coords_labels
        else ("x" if hasattr(da, "x") else "lon"),
        "y_label": coords_labels["y_label"]
        if "y_label" in coords_labels
        else ("y" if hasattr(da, "y") else "lat"),
    }

    if da.ndim == 3:
        def_labels["z_label"] = (
            coords_labels["z_label"]
            if "z_label" in coords_labels
            else ("band" if hasattr(da, "band") else "time")
        )

        z_id = 1
        if z_value is None:
            if da.sizes[def_labels["z_label"]] != 1:
                warn("Multiple layers detected. Using first layer as default.")
        else:
            z_id += tuple(da[def_labels["z_label"]]).index(z_value)
    else:
        z_id = None
    return z_id, def_labels


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
        raise ModuleNotFoundError("xarray must be installed to use this functionality")

    data = np.array(data).flatten()
    idx = index
    dims = idx.names
    indexer = tuple(idx.codes)
    shape = tuple(lev.size for lev in idx.levels)

    if coords is None:
        missing = np.prod(shape) > idx.shape[0]
        if missing:
            if "nodatavals" in attrs:
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
        fill = attrs["nodatavals"][0] if "nodatavals" in attrs else 0
        data_complete = np.full(shape, fill, data.dtype)
        data_complete[indexer] = data
        data_complete = data_complete[:, ::-1]

    da = DataArray(data_complete, coords=coords, dims=dims, attrs=attrs)
    return da.sortby(dims[-2], False)


@jit(nopython=True, fastmath=True)
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


@jit(nopython=True, fastmath=True)
def _SWbuilder(
    nrows, ncols, ids, id_map, criterion, k, dtype,
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
        Order of contiguity, Default is 1
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
    rows, cols = _compute_chunk(nrows, ncols, ids, id_map, criterion, k, dtype)
    data = np.ones_like(rows, dtype=np.int8)
    return (data, (rows, cols))


@jit(nopython=True, fastmath=True, nogil=True)
def _compute_chunk(
    nrows, ncols, ids, id_map, criterion, k, dtype,
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
        Order of contiguity, Default is 1
    dtype : type
        Data type of the rows and cols array

    Returns
    -------
    rows : ndarray
        1D ones array containing row value of each id
        in the sparse weight object
    cols : ndarray
        1D ones array containing columns value of each id
        in the sparse weight object
    ni : int
        Number of rows and cols
    """
    n = len(ids)
    # Setting d which is used for row, col preallocation
    d = 4 if criterion == "rook" else 8
    if k > 1:
        d = int((k / 2) * (2 * 8 + (k - 1) * 8))
    rows = np.empty(d * n, dtype=dtype)
    cols = np.empty_like(rows)
    ni = 0  # -> Pointer to store rows and cols in array
    for order in range(1, k + 1):
        condition = (
            (order - 1)
            if criterion == "queen"
            else ((k - order) if ((k - order) < order) else (order - 1))
        )
        for i in range(n):
            id_i = ids[i]
            og_id = id_map[id_i]

            if ((id_i + order) % ncols) >= order:
                # east neighbor
                id_neighbor = id_map[id_i + order]
                if id_neighbor:
                    rows[ni], cols[ni] = og_id, id_neighbor
                    ni += 1
                    rows[ni], cols[ni] = id_neighbor, og_id
                    ni += 1
                # north-east to south-east neighbors
                for j in range(condition):
                    if (id_i // ncols) < (nrows - j - 1):
                        id_neighbor = id_map[(id_i + order) + (ncols * (j + 1))]
                        if id_neighbor:
                            rows[ni], cols[ni] = og_id, id_neighbor
                            ni += 1
                            rows[ni], cols[ni] = id_neighbor, og_id
                            ni += 1
                    if (id_i // ncols) >= j + 1:
                        id_neighbor = id_map[(id_i + order) - (ncols * (j + 1))]
                        if id_neighbor:
                            rows[ni], cols[ni] = og_id, id_neighbor
                            ni += 1
                            rows[ni], cols[ni] = id_neighbor, og_id
                            ni += 1

            if (id_i // ncols) < (nrows - order):
                # south neighbor
                id_neighbor = id_map[id_i + (ncols * order)]
                if id_neighbor:
                    rows[ni], cols[ni] = og_id, id_neighbor
                    ni += 1
                    rows[ni], cols[ni] = id_neighbor, og_id
                    ni += 1
                # south-west to south-east neighbors
                for j in range(condition):
                    if (id_i % ncols) >= j + 1:
                        id_neighbor = id_map[id_i + (ncols * order) - j - 1]
                        if id_neighbor:
                            rows[ni], cols[ni] = og_id, id_neighbor
                            ni += 1
                            rows[ni], cols[ni] = id_neighbor, og_id
                            ni += 1
                    if ((id_i + j + 1) % ncols) >= j + 1:
                        id_neighbor = id_map[id_i + (ncols * order) + j + 1]
                        if id_neighbor:
                            rows[ni], cols[ni] = og_id, id_neighbor
                            ni += 1
                            rows[ni], cols[ni] = id_neighbor, og_id
                            ni += 1

                if criterion == "queen" or ((k / order) >= 2.0):
                    if (id_i % ncols) >= order:
                        # south-west neighbor
                        id_neighbor = id_map[id_i + (ncols * order) - order]
                        if id_neighbor:
                            rows[ni], cols[ni] = og_id, id_neighbor
                            ni += 1
                            rows[ni], cols[ni] = id_neighbor, og_id
                            ni += 1
                    if ((id_i + order) % ncols) >= order:
                        # south-east neighbor
                        id_neighbor = id_map[id_i + (ncols * order) + order]
                        if id_neighbor:
                            rows[ni], cols[ni] = og_id, id_neighbor
                            ni += 1
                            rows[ni], cols[ni] = id_neighbor, og_id
                            ni += 1
    return rows[:ni], cols[:ni]


@jit(nopython=True, fastmath=True)
def _chunk_generator(
    n_jobs, starts, ids,
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
        ids_chunk = ids[start : (start + chunk_size)]
        yield (ids_chunk,)


def _parSWbuilder(
    nrows, ncols, ids, id_map, criterion, k, dtype, n_jobs,
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
        Order of contiguity, Default is 1
    dtype : type
        Data type of the rows and cols array
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
    chunk = _chunk_generator(n_jobs, starts, ids)
    with parallel_backend("threading"):
        worker_out = Parallel(n_jobs=n_jobs)(
            delayed(_compute_chunk)(nrows, ncols, *ids, id_map, criterion, k, dtype)
            for ids in chunk
        )
    rows, cols = zip(*worker_out)
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.ones_like(rows, dtype=np.int8)
    return (data, (rows, cols))

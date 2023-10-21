import itertools
import warnings

import numpy

from ..cg import voronoi_frames
from ..io.fileio import FileIO
from ._contW_lists import ContiguityWeightsLists
from .util import get_ids, get_points_array
from .weights import WSP, W
from .raster import da2W, da2WSP

try:
    from shapely.geometry import Point as shapely_point
    from ..cg.shapes import Point as pysal_point

    point_type = (shapely_point, pysal_point)
except ImportError:
    from ..cg.shapes import Point as point_type

WT_TYPE = {"rook": 2, "queen": 1}  # for _contW_Binning

__author__ = "Sergio J. Rey <srey@asu.edu> , Levi John Wolf <levi.john.wolf@gmail.com>"

__all__ = ["Rook", "Queen", "Voronoi"]


class Rook(W):
    """
    Construct a weights object from a collection of pysal polygons that share at least one edge.

    Parameters
    ----------
    polygons    : list
                a collection of PySAL shapes to build weights from
    ids         : list
                a list of names to use to build the weights
    **kw        : keyword arguments
                optional arguments for :class:`pysal.weights.W`

    See Also
    --------
    :class:`libpysal.weights.weights.W`
    """

    def __init__(self, polygons, **kw):
        criterion = "rook"
        ids = kw.pop("ids", None)
        polygons, backup = itertools.tee(polygons)
        first_shape = next(iter(backup))
        if isinstance(first_shape, point_type):
            polygons, vertices = voronoi_frames(get_points_array(polygons))
            polygons = list(polygons.geometry)
        neighbors, ids = _build(polygons, criterion=criterion, ids=ids)
        W.__init__(self, neighbors, ids=ids, **kw)

    @classmethod
    def from_shapefile(cls, filepath, idVariable=None, full=False, **kwargs):
        """
        Rook contiguity weights from a polygon shapefile.

        Parameters
        ----------

        shapefile : string
                    name of polygon shapefile including suffix.
        sparse    : boolean
                    If True return WSP instance
                    If False return W instance

        Returns
        -------

        w          : W
                     instance of spatial weights

        Examples
        --------
        >>> from libpysal.weights import Rook
        >>> import libpysal
        >>> wr=Rook.from_shapefile(libpysal.examples.get_path("columbus.shp"), "POLYID")
        >>> "%.3f"%wr.pct_nonzero
        '8.330'
        >>> wr=Rook.from_shapefile(libpysal.examples.get_path("columbus.shp"), sparse=True)
        >>> pct_sp = wr.sparse.nnz *1. / wr.n**2
        >>> "%.3f"%pct_sp
        '0.083'

        Notes
        -----

        Rook contiguity defines as neighbors any pair of polygons that share a
        common edge in their polygon definitions.

        See Also
        --------
        :class:`libpysal.weights.weights.W`
        :class:`libpysal.weights.contiguity.Rook`
        """
        sparse = kwargs.pop("sparse", False)
        if idVariable is not None:
            ids = get_ids(filepath, idVariable)
        else:
            ids = None
        w = cls(FileIO(filepath), ids=ids, **kwargs)
        w.set_shapefile(filepath, idVariable=idVariable, full=full)
        if sparse:
            w = w.to_WSP()
        return w

    @classmethod
    def from_iterable(cls, iterable, sparse=False, **kwargs):
        """
        Construct a weights object from a collection of arbitrary polygons. This
        will cast the polygons to PySAL polygons, then build the W.

        Parameters
        ----------
        iterable    : iterable
                      a collection of of shapes to be cast to PySAL shapes. Must
                      support iteration. Can be either Shapely or PySAL shapes.
        **kw        : keyword arguments
                      optional arguments for  :class:`pysal.weights.W`
        See Also
        --------
        :class:`libpysal.weights.weights.W`
        :class:`libpysal.weights.contiguity.Rook`
        """
        new_iterable = iter(iterable)
        w = cls(new_iterable, **kwargs)
        if sparse:
            w = WSP.from_W(w)
        return w

    @classmethod
    def from_dataframe(
        cls,
        df,
        geom_col=None,
        idVariable=None,
        ids=None,
        id_order=None,
        use_index=None,
        **kwargs,
    ):
        """
        Construct a weights object from a (geo)pandas dataframe with a geometry
        column. This will cast the polygons to PySAL polygons, then build the W
        using ids from the dataframe.

        Parameters
        ----------
        df          : DataFrame
                      a :class: `pandas.DataFrame` containing geometries to use
                      for spatial weights
        geom_col    : string
                      the name of the column in `df` that contains the
                      geometries. Defaults to active geometry column.
        idVariable  : string
                      DEPRECATED - use `ids` instead.
                      the name of the column to use as IDs. If nothing is
                      provided, the dataframe index is used
        ids         : list-like, string
                      a list-like of ids to use to index the spatial weights object or
                      the name of the column to use as IDs. If nothing is
                      provided, the dataframe index is used if `use_index=True` or
                      a positional index is used if `use_index=False`.
                      Order of the resulting W is not respected from this list.
        id_order    : list
                      DEPRECATED - argument is deprecated and will be removed.
                      An ordered list of ids to use to index the spatial weights
                      object. If used, the resulting weights object will iterate
                      over results in the order of the names provided in this
                      argument.
        use_index   : bool
                      use index of `df` as `ids` to index the spatial weights object.
                      Defaults to False but in future will default to True.

        See Also
        --------
        :class:`libpysal.weights.weights.W`
        :class:`libpysal.weights.contiguity.Rook`
        """
        if geom_col is None:
            geom_col = df.geometry.name

        if id_order is not None:
            warnings.warn(
                "`id_order` is deprecated and will be removed in future.",
                FutureWarning,
                stacklevel=2,
            )
            if id_order is True and ((idVariable is not None) or (ids is not None)):
                # if idVariable is None, we want ids. Otherwise, we want the
                # idVariable column
                id_order = list(df.get(idVariable, ids))
            else:
                id_order = df.get(id_order, ids)

        if idVariable is not None:
            if ids is None:
                warnings.warn(
                    "`idVariable` is deprecated and will be removed in future. "
                    "Use `ids` instead.",
                    FutureWarning,
                    stacklevel=2,
                )
                ids = idVariable
            else:
                warnings.warn(
                    "Both `idVariable` and `ids` passed, using `ids`.",
                    UserWarning,
                    stacklevel=2,
                )

        if ids is None:
            if use_index is None:
                warnings.warn(
                    "`use_index` defaults to False but will default to True in future. "
                    "Set True/False directly to control this behavior and silence this "
                    "warning",
                    FutureWarning,
                    stacklevel=2,
                )
                use_index = False
            if use_index:
                ids = df.index.tolist()

        else:
            if isinstance(ids, str):
                ids = df[ids]

            if not isinstance(ids, list):
                ids = ids.tolist()

            if len(ids) != len(df):
                raise ValueError("The length of `ids` does not match the length of df.")

        if id_order is None:
            id_order = ids

        return cls.from_iterable(
            df[geom_col].tolist(), ids=ids, id_order=id_order, **kwargs
        )

    @classmethod
    def from_xarray(
        cls,
        da,
        z_value=None,
        coords_labels={},
        k=1,
        include_nodata=False,
        n_jobs=1,
        sparse=True,
        **kwargs,
    ):
        """
        Construct a weights object from a xarray.DataArray with an additional
        attribute index containing coordinate values of the raster
        in the form of Pandas.Index/MultiIndex.

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
        sparse : boolean
            type of weight object. Default is True. For libpysal.weights.W, sparse = False
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
            optional arguments passed when sparse = False

        Returns
        -------
        w : libpysal.weights.W/libpysal.weights.WSP
            instance of spatial weights class W or WSP with an index attribute

        Notes
        -----
        1. Lower order contiguities are also selected.
        2. Returned object contains `index` attribute that includes a
           `Pandas.MultiIndex` object from the DataArray.

        See Also
        --------
        :class:`libpysal.weights.weights.W`
        :class:`libpysal.weights.weights.WSP`
        """
        if sparse:
            w = da2WSP(da, "rook", z_value, coords_labels, k, include_nodata)
        else:
            w = da2W(da, "rook", z_value, coords_labels, k, include_nodata, **kwargs)
        return w


class Queen(W):
    """
    Construct a weights object from a collection of pysal polygons that share at least one vertex.

    Parameters
    ----------
    polygons    : list
                  a collection of PySAL shapes to build weights from
    ids         : list
                  a list of names to use to build the weights
    **kw        : keyword arguments
                  optional arguments for :class:`pysal.weights.W`

    See Also
    --------
    :class:`libpysal.weights.weights.W`
    """

    def __init__(self, polygons, **kw):
        criterion = "queen"
        ids = kw.pop("ids", None)
        polygons, backup = itertools.tee(polygons)
        first_shape = next(iter(backup))
        if isinstance(first_shape, point_type):
            polygons, vertices = voronoi_frames(get_points_array(polygons))
            polygons = list(polygons.geometry)
        neighbors, ids = _build(polygons, criterion=criterion, ids=ids)
        W.__init__(self, neighbors, ids=ids, **kw)

    @classmethod
    def from_shapefile(cls, filepath, idVariable=None, full=False, **kwargs):
        """
        Queen contiguity weights from a polygon shapefile.

        Parameters
        ----------

        shapefile   : string
                      name of polygon shapefile including suffix.
        idVariable  : string
                      name of a column in the shapefile's DBF to use for ids.
        sparse      : boolean
                      If True return WSP instance
                      If False return W instance
        Returns
        -------

        w            : W
                       instance of spatial weights

        Examples
        --------
        >>> from libpysal.weights import Queen
        >>> import libpysal
        >>> wq=Queen.from_shapefile(libpysal.examples.get_path("columbus.shp"))
        >>> "%.3f"%wq.pct_nonzero
        '9.829'
        >>> wq=Queen.from_shapefile(libpysal.examples.get_path("columbus.shp"),"POLYID")
        >>> "%.3f"%wq.pct_nonzero
        '9.829'
        >>> wq=Queen.from_shapefile(libpysal.examples.get_path("columbus.shp"), sparse=True)
        >>> pct_sp = wq.sparse.nnz *1. / wq.n**2
        >>> "%.3f"%pct_sp
        '0.098'

        Notes

        Queen contiguity defines as neighbors any pair of polygons that share at
        least one vertex in their polygon definitions.

        See Also
        --------
        :class:`libpysal.weights.weights.W`
        :class:`libpysal.weights.contiguity.Queen`
        """
        sparse = kwargs.pop("sparse", False)
        if idVariable is not None:
            ids = get_ids(filepath, idVariable)
        else:
            ids = None
        w = cls(FileIO(filepath), ids=ids, **kwargs)
        w.set_shapefile(filepath, idVariable=idVariable, full=full)
        if sparse:
            w = w.to_WSP()
        return w

    @classmethod
    def from_iterable(cls, iterable, sparse=False, **kwargs):
        """
        Construct a weights object from a collection of arbitrary polygons. This
        will cast the polygons to PySAL polygons, then build the W.

        Parameters
        ----------
        iterable    : iterable
                      a collection of of shapes to be cast to PySAL shapes. Must
                      support iteration. Contents may either be a shapely or PySAL shape.
        **kw        : keyword arguments
                      optional arguments for  :class:`pysal.weights.W`
        See Also
        ---------
        :class:`libpysal.weights.weights.W`
        :class:`libpysal.weights.contiguiyt.Queen`
        """
        new_iterable = iter(iterable)
        w = cls(new_iterable, **kwargs)
        if sparse:
            w = WSP.from_W(w)
        return w

    @classmethod
    def from_dataframe(
        cls,
        df,
        geom_col=None,
        idVariable=None,
        ids=None,
        id_order=None,
        use_index=None,
        **kwargs,
    ):
        """
        Construct a weights object from a (geo)pandas dataframe with a geometry
        column. This will cast the polygons to PySAL polygons, then build the W
        using ids from the dataframe.

        Parameters
        ----------
        df          : DataFrame
                      a :class: `pandas.DataFrame` containing geometries to use
                      for spatial weights
        geom_col    : string
                      the name of the column in `df` that contains the
                      geometries. Defaults to active geometry column.
        idVariable  : string
                      DEPRECATED - use `ids` instead.
                      the name of the column to use as IDs. If nothing is
                      provided, the dataframe index is used
        ids         : list-like, string
                      a list-like of ids to use to index the spatial weights object or
                      the name of the column to use as IDs. If nothing is
                      provided, the dataframe index is used if `use_index=True` or
                      a positional index is used if `use_index=False`.
                      Order of the resulting W is not respected from this list.
        id_order    : list
                      DEPRECATED - argument is deprecated and will be removed.
                      An ordered list of ids to use to index the spatial weights
                      object. If used, the resulting weights object will iterate
                      over results in the order of the names provided in this
                      argument.
        use_index   : bool
                      use index of `df` as `ids` to index the spatial weights object.
                      Defaults to False but in future will default to True.

        See Also
        --------
        :class:`libpysal.weights.weights.W`
        :class:`libpysal.weights.contiguity.Queen`
        """
        if geom_col is None:
            geom_col = df.geometry.name

        if id_order is not None:
            warnings.warn(
                "`id_order` is deprecated and will be removed in future.",
                FutureWarning,
                stacklevel=2,
            )
            if id_order is True and ((idVariable is not None) or (ids is not None)):
                # if idVariable is None, we want ids. Otherwise, we want the
                # idVariable column
                id_order = list(df.get(idVariable, ids))
            else:
                id_order = df.get(id_order, ids)

        if idVariable is not None:
            if ids is None:
                warnings.warn(
                    "`idVariable` is deprecated and will be removed in future. "
                    "Use `ids` instead.",
                    FutureWarning,
                    stacklevel=2,
                )
                ids = idVariable
            else:
                warnings.warn(
                    "Both `idVariable` and `ids` passed, using `ids`.",
                    UserWarning,
                    stacklevel=2,
                )

        if ids is None:
            if use_index is None:
                warnings.warn(
                    "`use_index` defaults to False but will default to True in future. "
                    "Set True/False directly to control this behavior and silence this "
                    "warning",
                    FutureWarning,
                    stacklevel=2,
                )
                use_index = False
            if use_index:
                ids = df.index.tolist()

        else:
            if isinstance(ids, str):
                ids = df[ids]

            if not isinstance(ids, list):
                ids = ids.tolist()

            if len(ids) != len(df):
                raise ValueError("The length of `ids` does not match the length of df.")

        if id_order is None:
            id_order = ids

        return cls.from_iterable(
            df[geom_col].tolist(), ids=ids, id_order=id_order, **kwargs
        )

    @classmethod
    def from_xarray(
        cls,
        da,
        z_value=None,
        coords_labels={},
        k=1,
        include_nodata=False,
        n_jobs=1,
        sparse=True,
        **kwargs,
    ):
        """
        Construct a weights object from a xarray.DataArray with an additional
        attribute index containing coordinate values of the raster
        in the form of Pandas.Index/MultiIndex.

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
        sparse : boolean
            type of weight object. Default is True. For libpysal.weights.W, sparse = False
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
            optional arguments passed when sparse = False

        Returns
        -------
        w : libpysal.weights.W/libpysal.weights.WSP
            instance of spatial weights class W or WSP with an index attribute

        Notes
        -----
        1. Lower order contiguities are also selected.
        2. Returned object contains `index` attribute that includes a
           `Pandas.MultiIndex` object from the DataArray.

        See Also
        --------
        :class:`libpysal.weights.weights.W`
        :class:`libpysal.weights.weights.WSP`
        """
        if sparse:
            w = da2WSP(da, "queen", z_value, coords_labels, k, include_nodata)
        else:
            w = da2W(da, "queen", z_value, coords_labels, k, include_nodata, **kwargs)
        return w


def Voronoi(points, criterion="rook", clip="ahull", **kwargs):
    """
    Voronoi weights for a 2-d point set


    Points are Voronoi neighbors if their polygons share an edge or vertex.


    Parameters
    ----------

    points      : array
                  (n,2)
                  coordinates for point locations
    kwargs      : arguments to pass to Rook, the underlying contiguity class.

    Returns
    -------

    w           : W
                  instance of spatial weights

    Examples
    --------
    >>> import numpy as np
    >>> from libpysal.weights import Voronoi
    >>> np.random.seed(12345)
    >>> points= np.random.random((5,2))*10 + 10
    >>> w = Voronoi(points)
    >>> w.neighbors
    {0: [2, 3, 4], 1: [2], 2: [0, 1, 4], 3: [0, 4], 4: [0, 2, 3]}
    """
    from ..cg.voronoi import voronoi_frames

    region_df, _ = voronoi_frames(points, clip=clip)
    if criterion.lower() == "queen":
        cls = Queen
    elif criterion.lower() == "rook":
        cls = Rook
    else:
        raise ValueError(
            "Contiguity criterion {} not supported. "
            'Only "rook" and "queen" are supported.'.format(criterion)
        )
    return cls.from_dataframe(region_df, **kwargs)


def _from_dataframe(df, **kwargs):
    """
    Construct a voronoi contiguity weight directly from a dataframe.
    Note that if criterion='rook', this is identical to the delaunay
    graph for the points if no clipping of the voronoi cells is applied.

    If the input dataframe is of any other geometry type than "Point",
    a value error is raised.

    Parameters
    ----------
    df          :   pandas.DataFrame
                    dataframe containing point geometries for a
                    voronoi diagram.

    Returns
    -------
    w           :   W
                    instance of spatial weights.
    """
    try:
        x, y = df.geometry.x.values, df.geometry.y.values
    except ValueError:
        raise NotImplementedError(
            "Voronoi weights are only"
            " implemented for point geometries. "
            "You may consider using df.centroid."
        )
    coords = numpy.column_stack((x, y))
    return Voronoi(coords, **kwargs)


Voronoi.from_dataframe = _from_dataframe


def _build(polygons, criterion="rook", ids=None):
    """
    This is a developer-facing function to construct a spatial weights object.

    Parameters
    ----------
    polygons    : list
                  list of pysal polygons to use to build contiguity
    criterion   : string
                  option of which kind of contiguity to build. Is either "rook" or "queen"
    ids         : list
                  list of ids to use to index the neighbor dictionary

    Returns
    -------
    tuple containing (neighbors, ids), where neighbors is a dictionary
    describing contiguity relations and ids is the list of ids used to index
    that dictionary.

    NOTE: this is different from the prior behavior of buildContiguity, which
          returned an actual weights object. Since this just dispatches for the
          classes above, this returns the raw ingredients for a spatial weights
          object, not the object itself.
    """
    if ids and len(ids) != len(set(ids)):
        raise ValueError(
            "The argument to the ids parameter contains duplicate entries."
        )

    wttype = WT_TYPE[criterion.lower()]
    geo = polygons
    if issubclass(type(geo), FileIO):
        geo.seek(0)  # Make sure we read from the beginning of the file.

    neighbor_data = ContiguityWeightsLists(polygons, wttype=wttype).w

    neighbors = {}
    # weights={}
    if ids:
        for key in neighbor_data:
            ida = ids[key]
            if ida not in neighbors:
                neighbors[ida] = set()
            neighbors[ida].update([ids[x] for x in neighbor_data[key]])
        for key in neighbors:
            neighbors[key] = set(neighbors[key])
    else:
        for key in neighbor_data:
            neighbors[key] = set(neighbor_data[key])
    return (
        dict(
            list(zip(list(neighbors.keys()), list(map(list, list(neighbors.values())))))
        ),
        ids,
    )


def buildContiguity(polygons, criterion="rook", ids=None):
    """
    This is a deprecated function.

    It builds a contiguity W from the polygons provided. As such, it is now
    identical to calling the class constructors for Rook or Queen.
    """
    # Warn('This function is deprecated. Please use the Rook or Queen classes',
    #        UserWarning)
    if criterion.lower() == "rook":
        return Rook(polygons, ids=ids)
    elif criterion.lower() == "queen":
        return Queen(polygons, ids=ids)
    else:
        raise Exception('Weights criterion "{}" was not found.'.format(criterion))

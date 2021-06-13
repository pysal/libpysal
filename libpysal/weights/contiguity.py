import itertools
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

__author__ = (
    "Sergio J. Rey <sjsrey@gmail.com> , Levi John Wolf <levi.john.wolf@gmail.com>"
)

__all__ = ["Rook", "Queen", "Voronoi"]


class Rook(W):
    """Construct a weights object from a collection of
    PySAL polygons that share at least one edge.

    Parameters
    ----------
    polygons : list
        A collection of PySAL shapes from which to build weights.
    **kwargs : dict
        Keyword arguments for ``libpysal.weights.W``. The parameter ``ids``,
        a list of names to use to build the weights, should be included here.

    See Also
    --------

    libpysal.weights.W

    """

    def __init__(self, polygons, **kwargs):

        criterion = "rook"
        ids = kwargs.pop("ids", None)
        polygons, backup = itertools.tee(polygons)
        first_shape = next(iter(backup))

        if isinstance(first_shape, point_type):
            polygons, vertices = voronoi_frames(get_points_array(polygons))
            polygons = list(polygons.geometry)

        neighbors, ids = _build(polygons, criterion=criterion, ids=ids)

        W.__init__(self, neighbors, ids=ids, **kwargs)

    @classmethod
    def from_shapefile(cls, filepath, idVariable=None, full=False, **kwargs):
        """`Rook` contiguity weights from a polygon shapefile.

        Parameters
        ----------
        filepath : str
            The name of polygon shapefile including the file extension.
        idVariable : str
            The name of the attribute in the shapefile to associate
            with ids in the weights. Default is ``None``.
        full : bool
            Write out the entire path for a shapefile (``True``) or
            only the base of the shapefile without extension (``False``).
            Default is ``False``.
        **kwargs : dict
            Keyword arguments for ``libpysal.weights.Rook``. ``'sparse'``
            should be included here.  If ``True`` return `WSP` instance.
            If ``False`` return `W` instance.

        Returns
        -------
        w : libpysal.weights.Rook
            A rook-style instance of spatial weights.

        Examples
        --------

        >>> from libpysal.weights import Rook
        >>> import libpysal
        >>> wr = Rook.from_shapefile(libpysal.examples.get_path("columbus.shp"), "POLYID")
        >>> "%.3f"%wr.pct_nonzero
        '8.330'

        >>> wr = Rook.from_shapefile(
        ...     libpysal.examples.get_path("columbus.shp"), sparse=True
        ... )
        >>> pct_sp = wr.sparse.nnz *1. / wr.n**2
        >>> "%.3f"%pct_sp
        '0.083'

        Notes
        -----

        `Rook` contiguity defines as neighbors any pair of polygons
        that share a common edge in their polygon definitions.

        See Also
        --------

        libpysal.weights.W
        libpysal.weights.Rook

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
        """Construct a weights object from a collection of arbitrary polygons.
        This will cast the polygons to PySAL polygons, then build the `W`.

        Parameters
        ----------
        iterable : iterable
            A collection of of shapes to be cast to PySAL shapes. Must
            support iteration. Can be either Shapely or PySAL shapes.
        sparse : bool
            Generate a `WSP` object. Default is ``False``.
        **kwargs : dict
            Keyword arguments for ``libpysal.weights.Rook``.

        Returns
        -------
        w : libpysal.weights.Rook
            A rook-style instance of spatial weights.

        See Also
        --------

        libpysal.weights.W
        libpysal.weights.WSP
        libpysal.weights.Rook

        """

        new_iterable = iter(iterable)
        w = cls(new_iterable, **kwargs)

        if sparse:
            w = WSP.from_W(w)

        return w

    @classmethod
    def from_dataframe(
        cls, df, geom_col=None, idVariable=None, ids=None, id_order=None, **kwargs
    ):
        """Construct a weights object from a ``pandas.DataFrame`` with a geometry
        column. This will cast the polygons to PySAL polygons, then build the `W`
        using ids from the dataframe.

        Parameters
        ----------
        df : pandas.DataFrame
            A ``pandas.DataFrame`` containing geometries to use for spatial weights.
        geom_col : {None, str}
            The name of the column in ``df`` that contains the
            geometries. Defaults to the active geometry column.
        idVariable : str
            The name of the column to use as IDs. If nothing is provided, the
            dataframe index is used. Default is ``None``.
        ids : list
            A list of ids to use to index the spatial weights object.
            Order is not respected from this list. Default is ``None``.
        id_order : list
            An ordered list of ids to use to index the spatial weights object. If
            used, the resulting weights object will iterate over results in the
            order of the names provided in this argument. Default is ``None``.
        **kwargs : dict
            Keyword arguments for ``libpysal.weights.Rook``.

        Returns
        -------
        w : w : libpysal.weights.Rook
            A rook-style instance of spatial weights.

        See Also
        --------

        libpysal.weights.W
        libpysal.weights.Rook

        """

        if geom_col is None:
            geom_col = df.geometry.name

        if id_order is not None:
            if id_order is True and ((idVariable is not None) or (ids is not None)):
                # if ``idVariable`` is ``None``, we want ids.
                # Otherwise, we want the ``idVariable`` column.
                id_order = list(df.get(idVariable, ids))
            else:
                id_order = df.get(id_order, ids)
        elif idVariable is not None:
            ids = df.get(idVariable).tolist()
        elif isinstance(ids, str):
            ids = df.get(ids).tolist()

        w = cls.from_iterable(
            df[geom_col].tolist(), ids=ids, id_order=id_order, **kwargs
        )

        return w

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
        """Construct a weights object from a ``xarray.DataArray`` with an additional
        attribute index containing coordinate values of the raster
        in the form of ``Pandas.Index``/``MultiIndex``.

        Parameters
        ----------
        da : xarray.DataArray
            Input 2D or 3D DataArray with shape=(z, y, x).
        z_value : {int, str, float}
            Select the z_value of 3D DataArray with multiple layers.
        coords_labels : dict
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
            Optional arguments passed when ``sparse=False``.

        Returns
        -------
        w : {libpysal.weights.W, libpysal.weights.WSP}
            An instance of spatial weights class `W` or `WSP`.

        See Also
        --------

        libpysal.weights.weights.W
        libpysal.weights.weights.WSP

        Notes
        -----
        1. Lower order contiguities are also selected.
        2. Returned object contains `index` attribute that includes a 
        `Pandas.MultiIndex` object from the DataArray.

        """

        if sparse:
            w = da2WSP(da, "rook", z_value, coords_labels, k, include_nodata)
        else:
            w = da2W(da, "rook", z_value, coords_labels, k, include_nodata, **kwargs)
        return w


class Queen(W):
    """Construct a weights object from a collection of PySAL
    polygons that share at least one vertex.

    Parameters
    ----------
    polygons : list
        A collection of PySAL shapes from which to build weights.
    **kwargs : dict
        Keyword arguments for ``pysal.weights.W``. The parameter ``ids``,
        a list of names to use to build the weights, should be included here.

    See Also
    --------

    libpysal.weights.W

    """

    def __init__(self, polygons, **kwargs):

        criterion = "queen"
        ids = kwargs.pop("ids", None)
        polygons, backup = itertools.tee(polygons)
        first_shape = next(iter(backup))

        if isinstance(first_shape, point_type):
            polygons, vertices = voronoi_frames(get_points_array(polygons))
            polygons = list(polygons.geometry)

        neighbors, ids = _build(polygons, criterion=criterion, ids=ids)

        W.__init__(self, neighbors, ids=ids, **kwargs)

    @classmethod
    def from_shapefile(cls, filepath, idVariable=None, full=False, **kwargs):
        """`Queen` contiguity weights from a polygon shapefile.

        Parameters
        ----------
        filepath : str
            The name of polygon shapefile including the file extension.
        idVariable : str
            The name of the attribute in the shapefile to associate
            with ids in the weights. Default is ``None``.
        full : bool
            Write out the entire path for a shapefile (``True``) or
            only the base of the shapefile without extension (``False``).
            Default is ``False``.
        **kwargs : dict
            Keyword arguments for ``libpysal.weights.Queen``. ``'sparse'``
            should be included here.  If ``True`` return `WSP` instance.
            If ``False`` return `W` instance.

        Returns
        -------
        w : libpysal.weights.Queen
            A queen-style instance of spatial weights.

        Examples
        --------

        >>> from libpysal.weights import Queen
        >>> import libpysal
        >>> wq = Queen.from_shapefile(libpysal.examples.get_path("columbus.shp"))
        >>> "%.3f"%wq.pct_nonzero
        '9.829'

        >>> wq = Queen.from_shapefile(
        ...     libpysal.examples.get_path("columbus.shp"), "POLYID"
        ... )
        >>> "%.3f"%wq.pct_nonzero
        '9.829'

        >>> wq = Queen.from_shapefile(
        ...     libpysal.examples.get_path("columbus.shp"), sparse=True
        ... )
        >>> pct_sp = wq.sparse.nnz *1. / wq.n**2
        >>> "%.3f"%pct_sp
        '0.098'

        Notes
        -----

        `Queen` contiguity defines as neighbors any pair of polygons that share at
        least one vertex in their polygon definitions.

        See Also
        --------

        libpysal.weights.W
        libpysal.weights.Queen

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
        """Construct a weights object from a collection of arbitrary polygons.
        This will cast the polygons to PySAL polygons, then build the `W`.

        Parameters
        ----------
        iterable : iterable
            A collection of of shapes to be cast to PySAL shapes. Must
            support iteration. Can be either Shapely or PySAL shapes.
        sparse : bool
            Generate a `WSP` object. Default is ``False``.
        **kwargs : dict
            Keyword arguments for ``libpysal.weights.Queen``.

        Returns
        -------
        w : libpysal.weights.Queen
            A queen-style instance of spatial weights.

        See Also
        --------

        libpysal.weights.W
        libpysal.weights.WSP
        libpysal.weights.Queen

        """

        new_iterable = iter(iterable)
        w = cls(new_iterable, **kwargs)

        if sparse:
            w = WSP.from_W(w)

        return w

    @classmethod
    def from_dataframe(cls, df, geom_col=None, **kwargs):
        """Construct a weights object from a ``pandas.DataFrame`` with a geometry
        column. This will cast the polygons to PySAL polygons, then build the `W`
        using ids from the dataframe.

        Parameters
        ----------
        df : pandas.DataFrame
            A ``pandas.DataFrame`` containing geometries to use for spatial weights.
        geom_col : {None, str}
            The name of the column in ``df`` that contains the
            geometries. Defaults to the active geometry column.
        **kwargs : dict
            Keyword arguments for ``libpysal.weights.Queen``.

        Returns
        -------
        w : libpysal.weights.Queen
            A queen-style instance of spatial weights.

        See Also
        --------

        libpysal.weights.W
        libpysal.weights.Queen

        """

        idVariable = kwargs.pop("idVariable", None)
        ids = kwargs.pop("ids", None)
        id_order = kwargs.pop("id_order", None)

        if geom_col is None:
            geom_col = df.geometry.name

        if id_order is not None:
            if id_order is True and ((idVariable is not None) or (ids is not None)):
                # if idVariable is None, we want ids. Otherwise, we want the
                # idVariable column
                ids = list(df.get(idVariable, ids))
                id_order = ids
            elif isinstance(id_order, str):
                ids = df.get(id_order, ids)
                id_order = ids
        elif idVariable is not None:
            ids = df.get(idVariable).tolist()
        elif isinstance(ids, str):
            ids = df.get(ids).tolist()

        w = cls.from_iterable(
            df[geom_col].tolist(), ids=ids, id_order=id_order, **kwargs
        )

        return w

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
        Construct a weights object from a ``xarray.DataArray`` with an additional
        attribute index containing coordinate values of the raster
        in the form of ``Pandas.Index``/``MultiIndex``.

        Parameters
        ----------
        da : xarray.DataArray
            Input 2D or 3D DataArray with shape=(z, y, x).
        z_value : {int, str, float}
            Select the z_value of 3D DataArray with multiple layers.
        coords_labels : dict
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
            Optional arguments passed when ``sparse=False``.

        Returns
        -------
        w : {libpysal.weights.W, libpysal.weights.WSP}
            An instance of spatial weights class `W` or `WSP`.

        See Also
        --------

        libpysal.weights.weights.W
        libpysal.weights.weights.WSP

        Notes
        -----
        1. Lower order contiguities are also selected.
        2. Returned object contains `index` attribute that includes a 
        `Pandas.MultiIndex` object from the DataArray.

        """

        if sparse:
            w = da2WSP(da, "queen", z_value, coords_labels, k, include_nodata)
        else:
            w = da2W(da, "queen", z_value, coords_labels, k, include_nodata, **kwargs)
        return w


def Voronoi(points, criterion="rook", clip="ahull", **kwargs):
    """Voronoi weights for a 2-d point set. Points are Voronoi neighbors
    if their polygons share an edge or vertex.

    Parameters
    ----------
    points : array-like
        An array-like ``(n,2)`` object of coordinates for point locations.
    criterion : str
        The weight criterion, either ``'rook'`` or ``'queen'``. Default is ``'rook'``.
    clip :  : str, shapely.geometry.Polygon
        An overloaded option about how to clip the voronoi cells. Default is ``'ahull'``.
        See ``libpysal.cg.voronoi_frames()`` for more explanation.
    **kwargs : dict
        Keyword arguments to pass to ``libpysal.weights.Voronoi``.

    Returns
    -------
    w : libpysal.weights.Voronoi
        A voronoi-style instance of spatial weights.

    Raises
    ------
    ValueError
        An unsupported value of ``criterion`` was passed in.

    Examples
    --------

    >>> import numpy as np
    >>> from libpysal.weights import Voronoi
    >>> np.random.seed(12345)
    >>> points = np.random.random((5,2))*10 + 10
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

    w = cls.from_dataframe(region_df, **kwargs)

    return w


def _from_dataframe(df, **kwargs):
    """Construct Voronoi contiguity weights directly from a dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe containing point geometries for a Voronoi diagram.
    **kwargs : dict
        Keyword arguments to pass to ``libpysal.weights.Voronoi``.

    Returns
    -------
    w : libpysal.weights.Vornoi
        A voronoi-style instance of spatial weights.

    Notes
    -----

    If ``criterion='rook'``, this is identical to the Delaunay graph for the points.

    Raises
    ------
    NotImplementedError
        If the input dataframe is of any other geometry type than ``Point``,
        a ``ValueError`` is caught and raised as a ``NotImplementedError``.

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

    w = Voronoi(coords, **kwargs)

    return w


Voronoi.from_dataframe = _from_dataframe


def _build(polygons, criterion="rook", ids=None):
    """This is a developer-facing function to construct a spatial weights object.

    Parameters
    ----------
    polygons : list
        A list of PySAL polygons to use to build contiguity.
    criterion : str
        Option of which kind of contiguity to build, either ``'rook'`` or ``'queen'``.
        Default is ``'rook'``.
    ids : list
        A list of ids to use to index the neighbor dictionary. Default is ``None``.

    Returns
    -------
    neighbor_result : tuple
        The contents are ``(neighbors, ids)``, where ``neighbors`` is
        a dictionary describing contiguity relations and ``ids`` is the
        list of ids used to index that dictionary.

    Raises
    ------
    ValueError
        The argument to the ``ids`` parameter contains duplicate entries.

    Notes
    -----

    This is different from the prior behavior of ``buildContiguity``, which returned an
    actual weights object. Since this just dispatches for the classes above, this returns
    the raw ingredients for a spatial weights object, not the object itself.

    """
    if ids and len(ids) != len(set(ids)):
        raise ValueError(
            "The argument to the ids parameter contains duplicate entries."
        )

    wttype = WT_TYPE[criterion.lower()]
    geo = polygons
    if issubclass(type(geo), FileIO):
        # Make sure we read from the beginning of the file.
        geo.seek(0)

    neighbor_data = ContiguityWeightsLists(polygons, wttype=wttype).w

    neighbors = {}

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

    neighbor_result = (
        dict(
            list(zip(list(neighbors.keys()), list(map(list, list(neighbors.values())))))
        ),
        ids,
    )

    return neighbor_result


def buildContiguity(polygons, criterion="rook", ids=None):
    """This is a deprecated function. It builds a contiguity `W` from the
    polygons provided. As such, it is now identical to calling the class
    constructors for `Rook` or `Queen`.

    """
    # Warn('This function is deprecated. Please use the Rook or Queen classes', UserWarning)

    if criterion.lower() == "rook":
        return Rook(polygons, ids=ids)
    elif criterion.lower() == "queen":
        return Queen(polygons, ids=ids)
    else:
        raise Exception('Weights criterion "{}" was not found.'.format(criterion))

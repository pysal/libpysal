from ..io.fileio import FileIO
from ..cg import voronoi_frames
from .weights import W, WSP
from ._contW_lists import ContiguityWeightsLists
from .util import get_ids, get_points_array
import numpy 
import itertools

try:
    from shapely.geometry import Point as shapely_point
    from ..cg.shapes import Point as pysal_point
    point_type = (shapely_point, pysal_point)
except ImportError:
    from ..cg.shapes import Point as point_type


WT_TYPE = {'rook': 2, 'queen': 1}  # for _contW_Binning

__author__ = "Sergio J. Rey <srey@asu.edu> , Levi John Wolf <levi.john.wolf@gmail.com>"

__all__ = ['Rook', 'Queen', 'Voronoi']

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
    ---------
    :class:`libpysal.weights.weights.W`
    """

    def __init__(self, polygons, **kw):
        criterion = 'rook'
        ids = kw.pop('ids', None) 
        polygons, backup = itertools.tee(polygons)
        first_shape = next(iter(backup))
        if isinstance(first_shape, point_type):
            polygons, vertices = voronoi_frames(get_points_array(polygons))
            polygons = list(polygons.geometry)
        neighbors, ids = _build(polygons, criterion=criterion, 
                                ids=ids)
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
        sparse = kwargs.pop('sparse', False)
        if idVariable is not None:
            ids = get_ids(filepath, idVariable) 
        else:
            ids = None
        w = cls(FileIO(filepath), ids=ids,**kwargs)
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
        ---------
        iterable    : iterable
                      a collection of of shapes to be cast to PySAL shapes. Must
                      support iteration. Can be either Shapely or PySAL shapes.
        **kw        : keyword arguments
                      optional arguments for  :class:`pysal.weights.W`
        See Also
        ----------
        :class:`libpysal.weights.weights.W`
        :class:`libpysal.weights.contiguity.Rook`
        """
        new_iterable = iter(iterable)
        w = cls(new_iterable, **kwargs)
        if sparse:
            w = WSP.from_W(w)
        return w

    @classmethod
    def from_dataframe(cls, df, geom_col='geometry', 
                       idVariable=None, ids=None, id_order=None, **kwargs):
        """
        Construct a weights object from a pandas dataframe with a geometry
        column. This will cast the polygons to PySAL polygons, then build the W
        using ids from the dataframe.

        Parameters
        ---------
        df          : DataFrame
                      a :class: `pandas.DataFrame` containing geometries to use
                      for spatial weights
        geom_col    : string
                      the name of the column in `df` that contains the
                      geometries. Defaults to `geometry`
        idVariable  : string
                      the name of the column to use as IDs. If nothing is
                      provided, the dataframe index is used
        ids         : list
                      a list of ids to use to index the spatial weights object.
                      Order is not respected from this list.
        id_order    : list
                      an ordered list of ids to use to index the spatial weights
                      object. If used, the resulting weights object will iterate
                      over results in the order of the names provided in this
                      argument. 

        See Also
        ---------
        :class:`libpysal.weights.weights.W`
        :class:`libpysal.weights.contiguity.Rook`
        """
        if idVariable is not None:
            ids = df.get(idVariable).tolist() #  idVariable takes precedent 
            if id_order is None:
                id_order = ids
        else:
            ids = df.index.tolist()
            id_order = ids
        return cls.from_iterable(df[geom_col].tolist(), ids=ids,
                                 id_order=id_order, **kwargs)

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
    ---------
    :class:`libpysal.weights.weights.W`
    """

    def __init__(self, polygons, **kw):
        criterion = 'queen'
        ids = kw.pop('ids', None)
        polygons, backup = itertools.tee(polygons)
        first_shape = next(iter(backup))
        if isinstance(first_shape, point_type):
            polygons, vertices = voronoi_frames(get_points_array(polygons))
            polygons = list(polygons.geometry)
        neighbors, ids = _build(polygons, criterion=criterion, 
                                ids=ids)
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
        sparse = kwargs.pop('sparse', False)
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
        ---------
        iterable    : iterable
                      a collection of of shapes to be cast to PySAL shapes. Must
                      support iteration. Contents may either be a shapely or PySAL shape.
        **kw        : keyword arguments
                      optional arguments for  :class:`pysal.weights.W`
        See Also
        ----------
        :class:`libpysal.weights.weights.W`
        :class:`libpysal.weights.contiguiyt.Queen`
        """
        new_iterable = iter(iterable) 
        w = cls(new_iterable, **kwargs) 
        if sparse:
            w = WSP.from_W(w)
        return w

    @classmethod
    def from_dataframe(cls, df, geom_col='geometry', **kwargs):
        """
        Construct a weights object from a pandas dataframe with a geometry
        column. This will cast the polygons to PySAL polygons, then build the W
        using ids from the dataframe.

        Parameters
        ---------
        df          : DataFrame
                      a :class: `pandas.DataFrame` containing geometries to use
                      for spatial weights
        geom_col    : string
                      the name of the column in `df` that contains the
                      geometries. Defaults to `geometry`
        idVariable  : string
                      the name of the column to use as IDs. If nothing is
                      provided, the dataframe index is used
        ids         : list
                      a list of ids to use to index the spatial weights object.
                      Order is not respected from this list.
        id_order    : list
                      an ordered list of ids to use to index the spatial weights
                      object. If used, the resulting weights object will iterate
                      over results in the order of the names provided in this
                      argument. 

        See Also
        ---------
        :class:`libpysal.weights.weights.W`
        :class:`libpysal.weights.contiguity.Queen`
        """
        idVariable = kwargs.pop('idVariable', None)
        ids = kwargs.pop('ids', None)
        id_order = kwargs.pop('id_order', None)
        if idVariable is not None:
            ids = df.get(idVariable).tolist() #  idVariable takes precedent 
            if id_order is None:
                id_order = ids
        else:
            ids = df.index.tolist()
            id_order = ids
        return cls.from_iterable(df[geom_col].tolist(), ids=ids,
                                 id_order=id_order, **kwargs)


def Voronoi(points, criterion='rook', clip='ahull', **kwargs):
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
    if criterion.lower() == 'queen':
        cls = Queen
    elif criterion.lower() == 'rook':
        cls = Rook
    else:
        raise ValueError('Contiguity criterion {} not supported. '
                         'Only "rook" and "queen" are supported.'.format(criterion))
    return cls.from_dataframe(region_df, **kwargs)

def _from_dataframe(df, **kwargs):
    """
    Construct a voronoi contiguity weight directly from a dataframe. 
    Note that if criterion='rook', this is identical to the delaunay
    graph for the points. 

    If the input dataframe is of any other geometry type than "Point",
    a value error is raised. 

    Arguments
    ---------
    df          :   pandas.DataFrame
                    dataframe containing point geometries for a 
                    voronoi diagram.

    Returns
    -------
    w           :   W
                    instance of spatial weights.
    """
    try:
        x,y = df.geometry.x.values, df.geometry.y.values
    except ValueError:
        raise NotImplementedError('Voronoi weights are only'
                                  ' implemented for point geometries. '
                                  'You may consider using df.centroid.')
    coords = numpy.column_stack((x,y))
    return Voronoi(coords, **kwargs) 
    
Voronoi.from_dataframe = _from_dataframe


def _build(polygons, criterion="rook", ids=None):
    """
    This is a developer-facing function to construct a spatial weights object. 

    Parameters
    ---------
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
        raise ValueError("The argument to the ids parameter contains duplicate entries.")

    wttype = WT_TYPE[criterion.lower()]
    geo = polygons
    if issubclass(type(geo), FileIO):
        geo.seek(0)  # Make sure we read from the beginning of the file.

    neighbor_data = ContiguityWeightsLists(polygons, wttype=wttype).w

    neighbors = {}
    #weights={}
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
    return dict(list(zip(list(neighbors.keys()),list(map(list, list(neighbors.values())))))), ids

def buildContiguity(polygons, criterion="rook", ids=None):
    """
    This is a deprecated function.

    It builds a contiguity W from the polygons provided. As such, it is now
    identical to calling the class constructors for Rook or Queen. 
    """
    #Warn('This function is deprecated. Please use the Rook or Queen classes',
    #        UserWarning)
    if criterion.lower() == 'rook':
        return Rook(polygons, ids=ids)
    elif criterion.lower() == 'queen':
        return Queen(polygons, ids=ids)
    else:
        raise Exception('Weights criterion "{}" was not found.'.format(criterion))

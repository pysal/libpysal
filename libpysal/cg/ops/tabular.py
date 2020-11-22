from ...common import requires as _requires
from ...io.geotable.utils import to_gdf, to_df
from warnings import warn as _Warn
import functools as _f
import sys as _sys

try:
    import pandas as _pd

    @_requires("pandas")
    @_f.wraps(_pd.merge)
    def join(*args, **kwargs):
        return _pd.merge(*args, **kwargs)


except ImportError:
    pass


@_requires("geopandas")
def spatial_join(
    df1, df2, left_geom_col="geometry", right_geom_col="geometry", **kwargs
):
    """Perform a spatial join between two ``pandas.DataFrames``
    datasets by calling out to ``geopandas``.

    Parameters
    ----------
    df1 : pandas.DataFrame
        The first dataset. It must have a 'MultiPolygon'
        or 'Polygon' geometry column.
    df2 : pandas.DataFrame
        The second dataset. It must have a 'MultiPolygon'
        or 'Polygon' geometry column.
    left_geom_col : str
        The left (``df1``) dataset's geometry column name.
        Default is ``'geometry'``.
    right_geom_col : str
        The right (``df2``) dataset's geometry column name.
        Default is ``'geometry'``.
    **kwargs : dict
        Optional keyword arguments passed in ``geopandas.tools.sjoin``.
        These may include (1) ``'how'`` (the method of spatial join),
        with valid values including ``'left'`` (use keys from ``df1``
        and retain only the ``df1`` geometry column), ``'right'`` (use
        keys from ``df2`` and retain only the ``df2`` geometry column,
        and ``'inner'`` (use the intersection of keys from both ``df1``
        & ``df2`` and retain only the `df2`` geometry column.;
        (2) ``'op'`` (defaults to ``'intersects'``), with other valid
        values including ``'contains'`` and ``'within'``. See the
        `Shapely docs <http://toblerity.org/shapely/manual.html#binary-predicates>`_
        for more information.; (3) ``'lsuffix'`` (defaults to ``left'``),
        the suffix to apply to overlapping column names from ``df1``.;
        and (4) ``'rsuffix'`` defaults to ``right'``),
        the suffix to apply to overlapping column names from ``df2``.
    
    Returns
    -------
    df : pandas.DataFrame
        A pandas.DataFrame with a new set of polygons
        and attributes resulting from the overlay.
    
    """

    import geopandas as gpd

    gdf1 = to_gdf(df1, geom_col=left_geom_col)
    gdf2 = to_gdf(df2, geom_col=right_geom_col)
    out = gpd.tools.sjoin(gdf1, gdf2, **kwargs)

    df = to_df(out)

    return df


@_requires("geopandas")
def spatial_overlay(
    df1, df2, how, left_geom_col="geometry", right_geom_col="geometry", **kwargs
):
    """Perform a spatial overlay between two polygonal
    datasets by calling out to ``geopandas``. It currently
    only supports data (``pandas.DataFrames``) with polygons
    and implements several methods that are all effectively
    subsets of the union.

    Parameters
    ----------
    df1 : pandas.DataFrame
        The first dataset. It must have a 'MultiPolygon'
        or 'Polygon' geometry column.
    df2 : pandas.DataFrame
        The second dataset. It must have a 'MultiPolygon'
        or 'Polygon' geometry column.
    how : str
        The method of spatial overlay. Options are inculde
        ``'intersection'``, ``'union', ``'identity'``,
        ``'symmetric_difference'`` or ``'difference'``.
    left_geom_col : str
        The left (``df1``) dataset's geometry column name.
        Default is ``'geometry'``.
    right_geom_col : str
        The right (``df2``) dataset's geometry column name.
        Default is ``'geometry'``.
    **kwargs : dict
        Optional keyword arguments passed in ``geopandas.tools.overlay``.
    
    Returns
    -------
    df : pandas.DataFrame
        A pandas.DataFrame with a new set of polygons
        and attributes resulting from the overlay.
    
    """

    import geopandas as gpd

    gdf1 = to_gdf(df1, geom_col=left_geom_col)
    gdf2 = to_gdf(df2, geom_col=right_geom_col)
    out = gpd.tools.overlay(gdf1, gdf2, how, **kwargs)

    df = to_df(out)

    return df


@_requires("shapely")
def dissolve(df, by="", **groupby_kws):
    from ._shapely import cascaded_union as union

    return union(df, by=by, **groupby_kws)


def clip(return_exterior=False):
    # return modified entries of the df that are within an envelope
    # provide an option to null out the geometries instead of not returning
    raise NotImplementedError


def erase(return_interior=True):
    # return modified entries of the df that are outside of an envelope
    # provide an option to null out the geometries instead of not returning
    raise NotImplementedError


@_requires("shapely")
def union(df, **kws):
    if "by" in kws:
        warn("When a 'by' argument is provided, you should be using 'dissolve'.")
        return dissolve(df, **kws)
    from ._shapely import cascaded_union as union

    return union(df)


@_requires("shapely")
def intersection(df, **kws):
    from ._shapely import cascaded_intersection as intersection

    return intersection(df, **kws)


def symmetric_difference():
    raise NotImplementedError


def difference():
    raise NotImplementedError

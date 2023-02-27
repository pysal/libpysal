from ...cg.shapes import asShape as pShape
from ...common import requires as _requires
from warnings import warn


@_requires("geopandas")
def to_df(df, geom_col="geometry", **kw):
    """Convert a ``geopandas.GeoDataFrame`` into a normal
    ``pandas.DataFrame`` with a column containing PySAL shapes.

    Parameters
    ----------
    df : geopandas.GeoDataFrame
        A ``geopandas.GeoDataFrame`` (or ``pandas.DataFrame``)
        with a column containing geo-interfaced shapes.
    geom_col : str
        The column name in ``df`` contains the geometry. Default is ``'geometry'``.
    **kw : dict
        Optional keyword arguments for ``pandas.DataFrame()``.

    Returns
    -------
    df : pandas.DataFrame
        The data converted into a ``pandas.DataFrame`` object.

    See Also
    --------

    pandas.DataFrame

    """

    import pandas as pd
    from geopandas import GeoDataFrame, GeoSeries

    df[geom_col] = df[geom_col].apply(pShape)

    if isinstance(df, (GeoDataFrame, GeoSeries)):
        df = pd.DataFrame(df, **kw)

    return df


@_requires("geopandas")
def to_gdf(df, geom_col="geometry", **kw):
    """Convert a ``pandas.DataFrame`` with
    geometry column to a ``geopandas.GeoDataFrame``.

    Parameters
    ----------
    df : pandas.DataFrame
        A ``pandas.DataFrame`` with a column containing geo-interfaced shapes.
    geom_col : str
        The column name in ``df`` contains the geometry. Default is ``'geometry'``.
    **kw : dict
        Optional keyword arguments for ``geopandas.GeoDataFrame()``.

    Returns
    -------
    gdf : geopandas.GeoDataFrame
        The data converted into a ``geopandas.GeoDataFrame`` object.

    See Also
    --------

    geopandas.GeoDataFrame

    """

    from geopandas import GeoDataFrame
    from shapely.geometry import shape

    df[geom_col] = df[geom_col].apply(shape)

    gdf = GeoDataFrame(df, geometry=geom_col, **kw)

    return gdf


def insert_metadata(df, obj, name=None, inplace=True, overwrite=False):
    """Insert/update metadata for a dataframe."""

    if not inplace:
        new = df.copy(deep=True)
        insert_metadata(new, obj, name=name, inplace=True)
        return new

    if name is None:
        name = type(obj).__name__

    if hasattr(df, name):
        if overwrite:
            warn("Overwriting attribute {}! This may break the dataframe!".format(name))
        else:
            raise Exception(
                "Dataframe already has attribute {}. Cowardly refusing "
                "to break dataframe.".format(name)
            )

    df._metadata.append(name)
    df.__setattr__(name, obj)

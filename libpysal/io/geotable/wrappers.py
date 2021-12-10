from ...common import requires
from ...cg import asShape
from .file import read_files, write_files
import shapely.geometry as sgeom
import pandas as pd


@requires("geopandas")
def geopandas(filename, **kw):
    """Wrapper for ``geopandas.read_file()``.
    
    Parameters
    ----------
    filename : str
        Path to the file.
    **kw : dict
        Optional keyword arguments for ``geopandas.read_file()``.
        
    Returns
    -------
    gdf : geopandas.GeoDataFrame
        The shapefile read in as a ``geopandas.GeoDataFrame``.
    
    """

    import geopandas

    gdf = geopandas.read_file(filename, **kw)

    return gdf


@requires("fiona")
def fiona(filename, geom_type="shapely", **kw):
    """Open a file with ``fiona`` and convert to a ``pandas.DataFrame``.
    
    Parameters
    ----------
    filename : str
        Path to the file.
    geom_type : str
        Package/method to use from creating geometries. Default is ``'shapely'``.
    **kw : dict
        Optional keyword arguments for ``fiona.open()``.
    
    Returns
    -------
    df : pandas.DataFrame
        The file read in as a ``pandas.DataFrame``.

    """

    if geom_type == "shapely":
        converter = sgeom.shape
    elif geom_type is None:
        converter = lambda x: x
    else:
        converter = asShape

    import fiona

    props = {}
    with fiona.open(filename, **kw) as f:
        for i, feat in enumerate(f):
            idx = feat.get("id", i)
            try:
                idx = int(idx)
            except ValueError:
                pass
            props.update({idx: feat.get("properties", dict())})
            props[idx].update({"geometry": converter(feat["geometry"])})

    df = pd.DataFrame().from_dict(props).T

    return df


_readers = {"read_shapefile": read_files, "read_fiona": fiona}
_writers = {"to_shapefile": write_files}

_pandas_readers = {
    k: v for k, v in list(pd.io.api.__dict__.items()) if k.startswith("read_")
}

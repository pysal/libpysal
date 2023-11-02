import os

from ...weights.contiguity import Queen, Rook
from ..fileio import FileIO
from .dbf import dbf2df, df2dbf
from .shp import series2shp, shp2series
from .utils import insert_metadata


def read_files(filepath, **kwargs):
    """Reads a ``.dbf``/``.shp`` pair, squashing geometries into a 'geometry' column.

    Parameters
    ----------
    filepath : str
        The file path.
    **kwargs : dict
        Optional keyword arguments for ``dbf2df()``.

    Returns
    -------
    df : pandas.DataFrame
        The results dataframe returned from ``dbf2df()``.
    """

    # keyword arguments wrapper will strip all around dbf2df's required arguments
    geomcol = kwargs.pop("geomcol", "geometry")
    weights = kwargs.pop("weights", "")

    dbf_path, shp_path = _pairpath(filepath)

    df = dbf2df(dbf_path, **kwargs)
    df[geomcol] = shp2series(shp_path)

    if weights != "" and isinstance(weights, str):
        if weights.lower() in ["rook", "queen"]:
            if weights.lower() == "rook":
                w = Rook.from_dataframe(df)
            else:
                w = Queen.from_dataframe(df)
            insert_metadata(df, w, name="W", inplace=True)
        else:
            try:
                w_path = os.path.splitext(dbf_path)[0] + "." + weights
                w = FileIO(w_path).read()
                insert_metadata(df, w, name="W", inplace=True)
            except OSError:
                print("Weights construction failed! Passing on weights.")

    return df


def write_files(df, filepath, **kwargs):
    """Writes dataframes with potential geometric components out to files.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to write out.
    filepath : str
        The file path.
    **kwargs : dict
        Optional keyword arguments for ``df2dbf()``.

    Returns
    -------
    dbf_path : str
        Path to the output ``.dbf``
    paths : tuple
        The file paths for ``dbf_out``, ``shp_out``, ``W_path``.
    """

    geomcol = kwargs.pop("geomcol", "geometry")
    weights = kwargs.pop("weights", "gal")

    dbf_path, shp_path = _pairpath(filepath)

    if geomcol not in df.columns:
        dbf_path = df2dbf(df, dbf_path, **kwargs)
        return dbf_path
    else:
        shp_out = series2shp(df[geomcol], shp_path)
        not_geom = [x for x in df.columns if x != geomcol]
        dbf_out = df2dbf(df[not_geom], dbf_path, **kwargs)

        if hasattr(df, "W"):
            w_path = os.path.splitext(filepath)[0] + "." + weights
            FileIO(w_path, "w").write(df.W)
        else:
            w_path = "no weights written"

        paths = dbf_out, shp_out, w_path

        return paths


def _pairpath(filepath: str) -> tuple:
    """Return ``.dbf``/``.shp`` paths for any ``.shp``,
    ``.dbf``, or basepath passed to function.
    """

    base = os.path.splitext(filepath)[0]
    paths = base + ".dbf", base + ".shp"

    return paths

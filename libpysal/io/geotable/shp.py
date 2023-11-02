import pandas as pd

from ..fileio import FileIO


def shp2series(filepath):
    """Reads a shapefile, stuffing each shape into an element of a ``pandas.Series``.

    Parameters
    ----------
    filepath : str
        Path to the file.

    Returns
    -------
    s : pandas.Series
        The data cast a ``pandas.Series`` object.
    """

    f = FileIO(filepath)
    s = pd.Series(poly for poly in f)
    f.close()

    return s


def series2shp(series, filepath):
    """Writes a ``pandas.Series`` of PySAL polygons to a file

    Parameters
    ----------
    series : pandas.Series
        The data to write out.

    Returns
    -------
    filepath : str
        Path to the file.
    """

    f = FileIO(filepath, "w")

    for poly in series:
        f.write(poly)
    f.close()

    return filepath

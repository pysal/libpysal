"""Miscellaneous file manipulation utilities.
"""

import numpy as np
import pandas as pd
from ..fileio import FileIO as ps_open


def check_dups(li):
    """Checks duplicates in a list of ID values. ID values must be read in as a list.
    Author(s) -- Luc Anselin <anselin@uchicago.edu>

    Parameters
    ----------
    li : list
        A collection of ID values.

    Returns
    -------
    dups : list
        The duplicate IDs.
        
    """

    dups = list(set([x for x in li if li.count(x) > 1]))

    return dups


def dbfdups(dbfpath, idvar):
    """Checks duplicates in a ``.dBase`` file ID variable must be specified correctly.
    Author(s) -- Luc Anselin <anselin@uchicago.edu>
    
    Parameters
    ----------
    dbfpath : str
        The file path to ``.dBase`` file.
    idvar : str
        The ID variable in ``.dBase`` file.

    Returns
    -------
    dups : list
        The duplicate IDs.
        
    """
    db = ps_open(dbfpath, "r")
    li = db.by_col(idvar)

    dups = list(set([x for x in li if li.count(x) > 1]))

    return dups


def df2dbf(df, dbf_path, my_specs=None):
    """Convert a ``pandas.DataFrame`` into a ``.dbf``. Author(s) --
    Dani Arribas-Bel <D.Arribas-Bel@liverpool.ac.uk>, Luc Anselin <anselin@uchicago.edu>

    Parameters
    ----------
    df : pandas.DataFrame
        Pandas dataframe object to be entirely written out to a ``.dbf``.
    dbf_path : str
        Path to the output ``.dbf``. It is also returned by the function.
    my_specs : list
        A list with the ``field_specs`` to use for each column.
        Defaults to ``None`` and applies the following scheme.
        * ``int`` -- ``('N', 14, 0)``
          * for all ``int`` types
        * ``float`` -- ``('N', 14, 14)``
          * for all ``float`` types
        * ``str`` -- ``('C', 14, 0)``
          * for ``str``, ``object``, and category
          * for all variants for different type sizes

    Returns
    -------
    dbf_path : str
        Path to the output ``.dbf``
    
    Notes
    -----
    
    Use of ``dtypes.name`` may not be fully robust, but preferred
    approach of using ``isinstance`` seems too clumsy.
    
    """
    if my_specs:
        specs = my_specs
    else:
        # new approach using dtypes.name to avoid numpy name issue in type
        type2spec = {
            "int": ("N", 20, 0),
            "int8": ("N", 20, 0),
            "int16": ("N", 20, 0),
            "int32": ("N", 20, 0),
            "int64": ("N", 20, 0),
            "float": ("N", 36, 15),
            "float32": ("N", 36, 15),
            "float64": ("N", 36, 15),
            "str": ("C", 14, 0),
            "object": ("C", 14, 0),
            "category": ("C", 14, 0),
        }
        types = [df[i].dtypes.name for i in df.columns]
        specs = [type2spec[t] for t in types]

    db = ps_open(dbf_path, "w")
    db.header = list(df.columns)
    db.field_spec = specs

    for i, row in list(df.T.items()):
        db.write(row)
    db.close()

    return dbf_path


def dbf2df(dbf_path, index=None, cols=False, incl_index=False):
    """Read a ``.dbf`` file as a ``pandas.DataFrame``, optionally
    selecting the index variable and which columns are to be loaded.
    Author(s) -- Dani Arribas-Bel <D.Arribas-Bel@liverpool.ac.uk>

    Parameters
    ----------
    dbf_path : str
        Path to the ``.dbf`` file to be read.
    index : str
        Name of the column to be used as the index of the ``pandas.DataFrame``.
    cols : list
        List with the names of the columns to be read into the ``pandas.DataFrame``.
        Defaults to ``False``, which reads the whole ``.dbf``
    incl_index : bool
        If ``True`` index is included in the ``pandas.DataFrame``
        as a column too. Defaults to ``False``.

    Returns
    -------
    df : pandas.DataFrame
        The resultant ``pandas.DataFrame`` object.
    
    """

    db = ps_open(dbf_path)

    if cols:
        if incl_index:
            cols.append(index)
        vars_to_read = cols
    else:
        vars_to_read = db.header

    data = dict([(var, db.by_col(var)) for var in vars_to_read])

    if index:
        index = db.by_col(index)
        db.close()
        return pd.DataFrame(data, index=index, columns=vars_to_read)
    else:
        db.close()
        return pd.DataFrame(data, columns=vars_to_read)


def dbfjoin(dbf1_path, dbf2_path, out_path, joinkey1, joinkey2):
    """Wrapper function to merge two ``.dbf`` files into a new
    ``.dbf`` file. Uses ``dbf2df`` and ``df2dbf`` to read and
    write the ``.dbf`` files into a ``pandas.DataFrame``. Uses
    all default settings for ``dbf2df`` and ``df2dbf`` (see docs
    for specifics). Author(s) --  Luc Anselin <anselin@uchicago.edu>

    Parameters
    ----------
    dbf1_path : str
        Path to the first (left) ``.dbf`` file.
    dbf2_path : str
        Path to the second (right) ``.dbf`` file.
    out_path : str
        Path to the output ``.dbf`` file (returned by the function).
    joinkey1 : str
        Variable name for the key in the first ``.dbf``.
        Must be specified. Key must take unique values.
    joinkey2 : str
        Variable name for the key in the second ``.dbf``.
        Must be specified. Key must take unique values.

    Returns
    -------
    dp : str
        Path to output file.
    
    """

    df1 = dbf2df(dbf1_path, index=joinkey1)
    df2 = dbf2df(dbf2_path, index=joinkey2)
    dfbig = pd.merge(df1, df2, left_on=joinkey1, right_on=joinkey2, sort=False)
    dp = df2dbf(dfbig, out_path)

    return dp


def dta2dbf(dta_path, dbf_path):
    """Wrapper function to convert a stata ``.dta`` file into a ``.dbf``
    file.  Uses ``df2dbf`` to write the ``.dbf`` files from a ``pandas.DataFrame``.
    Uses all default settings for ``df2dbf`` (see docs for specifics).
    Author(s) -- Luc Anselin <anselin@uchicago.edu>

    Parameters
    ----------
    dta_path : str
        Path to the Stata ``.dta`` file.
    dbf_path : str
        Path to the output ``.dbf`` file.

    Returns
    -------
    dp : str
        path to output file
    
    """

    db = pd.read_stata(dta_path)
    dp = df2dbf(db, dbf_path)

    return dp

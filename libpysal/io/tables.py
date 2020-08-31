__all__ = ["DataTable"]

from . import fileio
from ..common import requires
from warnings import warn
import numpy as np

__author__ = "Charles R Schmidt <schmidtc@gmail.com>"


class DataTable(fileio.FileIO):
    """`DataTable` provides additional functionality to `FileIO`
    for data table file tables `FileIO` handlers that provide data
    tables should subclass this instead of `FileIO`.
    """

    class _By_Col:
        def __init__(self, parent):

            self.p = parent

        def __repr__(self) -> str:

            return "keys: " + self.p.header.__repr__()

        def __getitem__(self, key):

            return self.p._get_col(key)

        def __setitem__(self, key, val):

            self.p.cast(key, val)

        def __call__(self, key):

            return self.p._get_col(key)

    def __init__(self, *args, **kwargs):

        fileio.FileIO.__init__(self, *args, **kwargs)

    def __repr__(self) -> str:

        return "DataTable: %s" % self.dataPath

    def __len__(self):
        """__len__ should be implemented by `DataTable` subclasses."""

        raise NotImplementedError

    @property
    def by_col(self):
        return self._By_Col(self)

    def _get_col(self, key):
        """Returns the column vector.
        
        Raises
        ------
        AttributeError
            Raised when the header is not set.
        AttributeError
            Raised when a field does not exist.
            
        """

        if not self.header:
            raise AttributeError("Please set the header.")

        if key in self.header:
            return self[:, self.header.index(key)]
        else:
            raise AttributeError("Field: %s does not exist in header." % key)

    def by_col_array(self, *args):
        """Return columns of table as a ``numpy.ndarray``.

        Parameters
        ----------
        *args : iterable
            Any number of strings of length :math:`k` names of variables to extract.

        Returns
        -------
        results : numpy.ndarray
            An array of shape :math:`(n,k)`.

        Notes
        -----

        If the variables are not all of the same data type, then ``numpy`` rules
        for casting will result in a uniform type applied to all variables. If only
        strings are passed to the function, then an array with those columns will be
        constructed. If only one list of strings is passed, the output is identical
        to those strings being passed. If at least one list is passed and other
        strings or lists are passed, this returns a tuple containing arrays
        constructed from each positional argument.

        Examples
        --------

        >>> import libpysal
        >>> dbf = libpysal.io.open(libpysal.examples.get_path('NAT.dbf'))
        >>> hr = dbf.by_col_array('HR70', 'HR80')
        >>> hr[0:5]
        array([[ 0.        ,  8.85582713],
               [ 0.        , 17.20874204],
               [ 1.91515848,  3.4507747 ],
               [ 1.28864319,  3.26381409],
               [ 0.        ,  7.77000777]])
        
        >>> hr = dbf.by_col_array(['HR80', 'HR70'])
        >>> hr[0:5]
        array([[ 8.85582713,  0.        ],
               [17.20874204,  0.        ],
               [ 3.4507747 ,  1.91515848],
               [ 3.26381409,  1.28864319],
               [ 7.77000777,  0.        ]])
        
        >>> hr = dbf.by_col_array(['HR80'])
        >>> hr[0:5]
        array([[ 8.85582713],
               [17.20874204],
               [ 3.4507747 ],
               [ 3.26381409],
               [ 7.77000777]])
        
        Numpy only supports homogeneous arrays. See Notes above.

        >>> hr = dbf.by_col_array('STATE_NAME', 'HR80')
        >>> hr[0:5]
        array([['Minnesota', '8.8558271343'],
               ['Washington', '17.208742041'],
               ['Washington', '3.4507746989'],
               ['Washington', '3.2638140931'],
               ['Washington', '7.77000777']], dtype='<U20')

        >>> y, X = dbf.by_col_array('STATE_NAME', ['HR80', 'HR70'])
        >>> y[0:5]
        array([['Minnesota'],
               ['Washington'],
               ['Washington'],
               ['Washington'],
               ['Washington']], dtype='<U20')
        
        >>> X[0:5]
        array([[ 8.85582713,  0.        ],
               [17.20874204,  0.        ],
               [ 3.4507747 ,  1.91515848],
               [ 3.26381409,  1.28864319],
               [ 7.77000777,  0.        ]])
        """

        if any([isinstance(arg, list) for arg in args]):
            results = []
            for namelist in args:
                if isinstance(namelist, str):
                    results.append([self._get_col(namelist)])
                else:
                    results.append([self._get_col(vbl) for vbl in namelist])
            if len(results) == 1:
                results = np.array(results[0]).T
            else:
                results = tuple(np.array(lst).T for lst in results)
        else:
            results = np.array([self._get_col(name) for name in args]).T

        return results

    def __getitem__(self, key) -> list:
        """DataTables fully support slicing in 2D. To provide slicing, handlers
        must provide ``__len__``. Slicing accepts up to two arguments. For example,
        
        * ``table[row]``
        * ``table[row, col]``
        * ``table[row_start:row_stop]``
        * ``table[row_start:row_stop:row_step]``
        * ``table[:, col]``
        * ``table[:, col_start:col_stop]``
        * etc.

        ALL indices are Zero-Offsets. For example,
        
        * ``>>> assert index in range(0, len(table))``
        
        Raises
        ------
        TypeError
            Raised when two dimensions are not provided for slicing.
        TypeError
            Raised when an unknown key is present.
        
        """

        prevPos = self.tell()

        if issubclass(type(key), str):
            raise TypeError("index should be int or slice")

        if issubclass(type(key), int) or isinstance(key, slice):
            rows = key
            cols = None
        elif len(key) > 2:
            raise TypeError(
                "DataTables support two dimmensional slicing, % d slices provided."
                % len(key)
            )
        elif len(key) == 2:
            rows, cols = key
        else:
            raise TypeError("Key: % r, is confusing me. I don't know what to do." % key)

        if isinstance(rows, slice):
            row_start, row_stop, row_step = rows.indices(len(self))
            self.seek(row_start)
            data = [next(self) for i in range(row_start, row_stop, row_step)]
        else:
            self.seek(slice(rows).indices(len(self))[1])
            data = [next(self)]

        if cols is not None:
            if isinstance(cols, slice):
                col_start, col_stop, col_step = cols.indices(len(data[0]))
                data = [r[col_start:col_stop:col_step] for r in data]
            else:
                # col_start, col_stop, col_step = cols, cols+1, 1
                data = [r[cols] for r in data]
        self.seek(prevPos)

        return data

    @requires("pandas")
    def to_df(self, n=-1, read_shp=False, **df_kws):
        """Convert a ``libpysal.DataTable`` to a ``pandas.DataFrame``.
        
        Parameters
        ----------
        n : int
            Lines to read from file. Default is ``-1``.
        read_shp : bool
            Read in from a shapefile (``True``). Default is ``False``.
        **df_kws : dict
            Optional keyword arguments to pass into ``pandas.DataFrame()``.
        
        Returns
        -------
        df : pandas.DataFrame
            Pandas dataframe representation of the data.

        """

        import pandas as pd

        self.seek(0)
        header = self.header
        records = self.read(n)
        df = pd.DataFrame(records, columns=header, **df_kws)
        if read_shp is not False:
            if read_shp is True or self.dataPath.endswith(".dbf"):
                read_shp = self.dataPath[:-3] + "shp"
            try:
                from .geotable.shp import shp2series

                df["geometry"] = shp2series(self.dataPath[:-3] + "shp")
            except IOError as e:
                warn(
                    "Encountered the following error in attempting to read"
                    " the shapefile {}. Proceeding with read, but the error"
                    " will be reproduced below:\n"
                    " {}".format(self.dataPath[:-3] + "shp", e)
                )
        return df


def _test():
    import doctest

    doctest.testmod(verbose=True)


if __name__ == "__main__":
    _test()

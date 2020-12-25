import os.path
from .. import fileio as FileIO
from ...weights.weights import W
from warnings import warn

__author__ = "Charles R Schmidt <schmidtc@gmail.com>"
__all__ = ["GwtIO"]


class unique_filter(object):
    """(Util function) When a new instance is passed as an arugment to the
    builtin filter it will remove duplicate entries without changing the
    order of the list. Be sure to ceate a new instance everytime, unless
    you want a global filter.

    Examples
    --------
    
    >>> l = ['a', 'a', 'b', 'a', 'c', 'v', 'd', 'a', 'v', 'd']
    >>> list(filter(unique_filter(),l))
    ['a', 'b', 'c', 'v', 'd']
    
    """

    def __init__(self):
        self.exclude = set()

    def __call__(self, x) -> bool:
        if x in self.exclude:
            return False
        else:
            self.exclude.add(x)
            return True


class GwtIO(FileIO.FileIO):

    FORMATS = ["kwt", "gwt"]
    MODES = ["r", "w"]

    def __init__(self, *args, **kwargs):
        self._varName = "Unknown"
        self._shpName = "Unknown"
        FileIO.FileIO.__init__(self, *args, **kwargs)
        self.file = open(self.dataPath, self.mode)

    def _set_varName(self, val):
        if issubclass(type(val), str):
            self._varName = val

    def _get_varName(self) -> str:
        return self._varName

    varName = property(fget=_get_varName, fset=_set_varName)

    def _set_shpName(self, val):
        if issubclass(type(val), str):
            self._shpName = val

    def _get_shpName(self) -> str:
        return self._shpName

    shpName = property(fget=_get_shpName, fset=_set_shpName)

    def read(self, n=-1):
        """
        
        Parameters
        ----------
        n : int
            Read at most ``n`` objects. Default is ``-1``.
        
        Returns
        -------
        w : libpysal.weights.W
            A PySAL `W` object.
        
        """

        self._complain_ifclosed(self.closed)

        w = self._read()

        return w

    def seek(self, pos):
        if pos == 0:
            self.file.seek(0)
            self.pos = 0

    def _readlines(self, id_type, ret_ids=False):
        """Reads the main body of gwt-like weights files into two dictionaries
        containing weights and neighbors. This code part is repeatedly used for
        many weight file formats. Header lines, however, are different from
        format to format. So, for code reusability, this part is separated out
        from the ``_read()`` function by Myunghwa Hwang.
        
        Parameters
        ----------
        id_type : type
            Cast IDs as this type.
        ret_ids : bool
            Return IDs (``True``). Default is ``False``.
        
        Returns
        -------
        weights : dict
            Dictionary of weight values.
        neighbors : dict
            Dictionary of neighbor ID values.
        ids : list
            List of ID values.
        
        """

        data = [row.strip().split() for row in self.file.readlines()]
        ids = list(filter(unique_filter(), [x[0] for x in data]))
        ids = list(map(id_type, ids))
        WN = {}

        # note: fromkeys is no good here, all keys end up sharing the say dict value
        for id in ids:
            WN[id] = {}

        for i, j, v in data:
            i = id_type(i)
            j = id_type(j)
            WN[i][j] = float(v)
        weights = {}
        neighbors = {}

        for i in WN:
            weights[i] = list(WN[i].values())
            neighbors[i] = list(WN[i].keys())
        if ret_ids:
            return weights, neighbors, ids
        else:
            return weights, neighbors

    def _read(self):
        """Reads ``.gwt`` file.
        
        Returns
        -------
        w : libpysal.weights.W
            A PySAL `W` object.
        
        Raises
        ------
        StopIteration
            Raised at the EOF.
        
        Examples
        --------

        Type ``dir(f)`` at the interpreter to see what methods are supported.
        Open ``.gwt`` file and read it into a PySAL weights object.

        >>> import libpysal
        >>> f = libpysal.io.open(libpysal.examples.get_path('juvenile.gwt'), 'r').read()

        Get the number of observations from the header.

        >>> f.n
        168

        Get the mean number of neighbors.

        >>> f.mean_neighbors
        16.678571428571427

        Get neighbor distances for a single observation.

        >>> f[1]
        {2: 14.1421356}

        """

        if self.pos > 0:
            raise StopIteration

        flag, n, shp, id_var = self.file.readline().strip().split()
        self.shpName = shp
        self.varName = id_var
        id_order = None
        id_type = str

        try:
            base = os.path.split(self.dataPath)[0]
            dbf = os.path.join(base, self.shpName.replace(".shp", "") + ".dbf")
            if os.path.exists(dbf):
                db = FileIO.FileIO(dbf, "r")
                if id_var in db.header:
                    id_order = db.by_col(id_var)
                    id_type = type(id_order[0])
                else:
                    msg = "ID_VAR:'%s' was in in the DBF header, "
                    msg += "proceeding with unordered string IDs."
                    msg = msg % id_var
                    warn(msg, RuntimeWarning)
            else:
                msg = "DBF relating to GWT was not found, "
                msg += "proceeding with unordered string IDs."
                warn(msg, RuntimeWarning)
        except:
            msg = "Exception occurred will reading DBF, "
            msg += "proceeding with unordered string IDs."
            warn(msg, RuntimeWarning)

        self.flag = flag
        self.n = n
        self.shp = shp
        self.id_var = id_var

        if id_order is None:
            weights, neighbors, id_order = self._readlines(id_type, True)
        else:
            weights, neighbors = self._readlines(id_type)

        self.pos += 1
        w = W(neighbors, weights, id_order)
        # w.transform = 'b'

        # set meta data
        w._shpName = self.shpName
        w._varName = self.varName

        # msg = "Weights have been converted to binary. "
        # msg += "To retrieve original values use w.transform='o'"
        # warn(msg, RuntimeWarning)

        return w

    def _writelines(self, obj):
        """Writes the main body of gwt-like weights files. This code part is
        repeatedly used for many weight file formats. Header lines, however,
        are different from format to format. So, for code reusability, this
        part is separated out from write function by Myunghwa Hwang.
        
        Parameters
        ----------
        obj : libpysal.weights.W
            A PySAL `W` object.
        
        """

        for id in obj.id_order:
            neighbors = list(zip(obj.neighbors[id], obj.weights[id]))
            str_id = "_".join(str(id).split())
            for neighbor, weight in neighbors:
                neighbor = "_".join(str(neighbor).split())

                self.file.write("%s %s %6G\n" % (str_id, neighbor, weight))
                self.pos += 1

    def write(self, obj):
        """Write a weights object to the opened `GWT` file.

        Parameters
        ----------
        obj : libpysal.weights.W
            A PySAL `W` object.
        
        Raises
        ------
        TypeError
            Raised when the input ``obj`` is not a PySAL `W`.

        Examples
        --------

        >>> import tempfile, libpysal, os
        >>> testfile = libpysal.io.open(libpysal.examples.get_path('juvenile.gwt'), 'r')
        >>> w = testfile.read()

        Create a temporary file for this example.

        >>> f = tempfile.NamedTemporaryFile(suffix='.gwt')

        Reassign to a new variable.

        >>> fname = f.name

        Close the temporary named file.

        >>> f.close()

        Open the new file in write mode.

        >>> o = libpysal.io.open(fname, 'w')

        Write the weights object into the open file.

        >>> o.write(w)
        >>> o.close()

        Read in the newly created ``.gwt`` file.

        >>> wnew =  libpysal.io.open(fname, 'r').read()

        Compare values from old to new.

        >>> wnew.pct_nonzero == w.pct_nonzero
        True

        Clean up the temporary file created for this example.

        >>> os.remove(fname)
        
        """

        self._complain_ifclosed(self.closed)

        if issubclass(type(obj), W):
            # transform = obj.transform
            # obj.transform = 'o'

            if hasattr(obj, "_shpName"):
                self.shpName = obj._shpName

            if hasattr(obj, "_varName"):
                self.varName = obj._varName

            header = "%s %i %s %s\n" % ("0", obj.n, self.shpName, self.varName)
            self.file.write(header)
            # obj.transform = transform

            self._writelines(obj)

        else:
            raise TypeError("Expected a PySAL weights object, got: %s." % (type(obj)))

    def close(self):
        self.file.close()
        FileIO.FileIO.close(self)

    @staticmethod
    def __zero_offset(neighbors: dict, weights: dict, original_ids=None) -> dict:

        if not original_ids:
            original_ids = list(neighbors.keys())

        old_weights = weights
        new_weights = {}
        new_ids = {}
        old_ids = {}
        new_neighbors = {}

        for i in original_ids:
            new_i = original_ids.index(i)
            new_ids[new_i] = i
            old_ids[i] = new_i
            neighbors_i = neighbors[i]
            new_neighbors_i = [original_ids.index(j) for j in neighbors_i]
            new_neighbors[new_i] = new_neighbors_i
            new_weights[new_i] = weights[i]

        info = {}
        info["new_ids"] = new_ids
        info["old_ids"] = old_ids
        info["new_neighbors"] = new_neighbors
        info["new_weights"] = new_weights

        return info

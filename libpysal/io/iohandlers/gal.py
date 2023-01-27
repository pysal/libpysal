from .. import fileio
from ...weights.weights import W, WSP
from scipy import sparse
import numpy as np

__author__ = "Charles R Schmidt <schmidtc@gmail.com>"
__all__ = ["GalIO"]


class GalIO(fileio.FileIO):
    """Opens, reads, and writes file objects in `GAL` format."""

    FORMATS = ["gal"]
    MODES = ["r", "w"]

    def __init__(self, *args, **kwargs):
        self._typ = str
        fileio.FileIO.__init__(self, *args, **kwargs)
        self.file = open(self.dataPath, self.mode)

    def read(self, n=-1, sparse=False):
        """Read in a ``.gal`` file.
        
        Parameters
        ----------
        n : int
            Read at most ``n`` objects. Default is ``-1``.
        sparse: bool
            If ``True`` return a ``scipy`` sparse object. If ``False``
            return PySAL `W` object. Default is ``False``.
        
        Returns
        -------
        w : {libpysal.weights.W, libpysal.weights.WSP}
            A PySAL `W` object or a thin PySAL `WSP`.

        """
        self._sparse = sparse
        self._complain_ifclosed(self.closed)

        w = self._read()

        return w

    def seek(self, pos):
        if pos == 0:
            self.file.seek(0)
            self.pos = 0

    def _get_data_type(self):
        return self._typ

    def _set_data_type(self, typ):
        """
        
        Raises
        ------
        TypeError
            Raised when ``typ`` is not a callable.
        
        """
        if callable(typ):
            self._typ = typ
        else:
            raise TypeError("Expecting a callable.")

    data_type = property(fset=_set_data_type, fget=_get_data_type)

    def _read(self):
        """Reads in a `GalIO` object.

        Returns
        -------
        w : {libpysal.weights.W, libpysal.weights.WSP}
            A PySAL `W` object or a thin PySAL `WSP`.
        
        Raises
        ------
        StopIteration
            Raised at the EOF.
        
        Examples
        --------

        >>> import tempfile, libpysal, os

        Read in a file `GAL` file.

        >>> testfile = libpysal.io.open(libpysal.examples.get_path('sids2.gal'), 'r')

        Return a `W` object.

        >>> w = testfile.read()
        >>> w.n == 100
        True
        
        >>> print(round(w.sd,6))
        1.515124
        
        >>> testfile = libpysal.io.open(libpysal.examples.get_path('sids2.gal'), 'r')

        Return a sparse matrix for the `W` information.

        >>> wsp = testfile.read(sparse=True)
        >>> wsp.sparse.nnz
        462

        """

        if self._sparse:

            if self.pos > 0:
                raise StopIteration

            header = self.file.readline().strip().split()
            header_n = len(header)
            n = int(header[0])

            if header_n > 1:
                n = int(header[1])

            ids = []
            idsappend = ids.append
            row = []
            extend = row.extend  # avoid dot in loops
            col = []
            append = col.append
            counter = 0
            typ = self.data_type

            for i in range(n):
                id, n_neighbors = self.file.readline().strip().split()
                id = typ(id)
                n_neighbors = int(n_neighbors)
                neighbors_i = list(map(typ, self.file.readline().strip().split()))
                nn = len(neighbors_i)
                extend([id] * nn)
                counter += nn

                for id_neigh in neighbors_i:
                    append(id_neigh)
                idsappend(id)

            self.pos += 1
            row = np.array(row)
            col = np.array(col)
            data = np.ones(counter)
            ids = np.unique(row)
            row = np.array([np.where(ids == j)[0] for j in row]).flatten()
            col = np.array([np.where(ids == j)[0] for j in col]).flatten()
            spmat = sparse.csr_matrix((data, (row, col)), shape=(n, n))

            w = WSP(spmat)

        else:

            if self.pos > 0:
                raise StopIteration

            neighbors = {}
            ids = []

            # handle case where more than n is specified in first line
            header = self.file.readline().strip().split()
            header_n = len(header)
            n = int(header[0])

            if header_n > 1:
                n = int(header[1])

            w = {}
            typ = self.data_type

            for i in range(n):
                id, n_neighbors = self.file.readline().strip().split()
                id = typ(id)
                n_neighbors = int(n_neighbors)
                neighbors_i = list(map(typ, self.file.readline().strip().split()))
                neighbors[id] = neighbors_i
                ids.append(id)

            self.pos += 1

            w = W(neighbors, id_order=ids)

        return w

    def write(self, obj):
        """Write a weights object to the opened `GAL` file.

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
        >>> testfile = libpysal.io.open(libpysal.examples.get_path('sids2.gal'), 'r')
        >>> w = testfile.read()

        Create a temporary file for this example.

        >>> f = tempfile.NamedTemporaryFile(suffix='.gal')

        Reassign to the new variable.

        >>> fname = f.name

        Close the temporary named file.

        >>> f.close()

        Open the new file in write mode.

        >>> o = libpysal.io.open(fname, 'w')

        Write the weights object into the open file.

        >>> o.write(w)
        >>> o.close()

        Read in the newly created gal file.

        >>> wnew =  libpysal.io.open(fname, 'r').read()

        Compare values from old to new.

        >>> wnew.pct_nonzero == w.pct_nonzero
        True

        Clean up the temporary file created for this example.

        >>> os.remove(fname)
        
        """

        self._complain_ifclosed(self.closed)

        if issubclass(type(obj), W):
            IDS = obj.id_order
            self.file.write("%d\n" % (obj.n))

            for id in IDS:
                neighbors = obj.neighbors[id]
                self.file.write("%s %d\n" % (str(id), len(neighbors)))
                self.file.write(" ".join(map(str, neighbors)) + "\n")
            self.pos += 1
        else:
            raise TypeError("Expected a PySAL weights object, got: %s." % (type(obj)))

    def close(self):
        self.file.close()
        fileio.FileIO.close(self)

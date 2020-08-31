import scipy.io as sio
from .. import fileio
from ...weights.weights import W, WSP

__author__ = "Myunghwa Hwang <mhwang4@gmail.com>"
__all__ = ["MtxIO"]


class MtxIO(fileio.FileIO):
    """
    Opens, reads, and writes weights file objects in Matrix Market ``.mtx`` format.
    The Matrix Market MTX format is used to facilitate the exchange of matrix data.
    In PySAL, it is being tested as a new file format for delivering the weights
    information of a spatial weights matrix. Although the MTX format supports both
    full and sparse matrices with different data types, it is assumed that spatial
    weights files in the ``.mtx``. format always use the sparse (or coordinate)
    format with real data values. For now, no additional assumption (e.g.,
    symmetry) is made of the structure of a weights matrix.

    With the above assumptions, the structure of a MTX file containing a spatial
    weights matrix can be defined as follows:
    
    ```
    %%MatrixMarket matrix coordinate real general <--- header 1 (constant)
    % Comments starts                             <---
    % ....                                           | 0 or more comment lines
    % Comments ends                               <---
    M    N    L                                   <--- header 2, rows, columns, entries
    I1   J1   A(I1,J1)                            <---
    ...                                              | L entry lines
    IL   JL   A(IL,JL)                            <---
    ```
    
    In the MTX format, the index for rows or columns starts with 1.

    PySAL uses ``mtx`` tools in
    `scipy.io <https://docs.scipy.org/doc/scipy/reference/tutorial/io.html>`_.
    Thus, it is subject to all limits that ``scipy`` currently has. Reengineering
    may be required, since ``scipy`` reads in the entire entry into memory.

    References
    ----------
    
    `MTX format specification <http://math.nist.gov/MatrixMarket/formats.html>`_

    `Matrix Market files
    <https://docs.scipy.org/doc/scipy/reference/tutorial/io.html#matrix-market-files>`_
    in ``scipy``.

    """

    FORMATS = ["mtx"]
    MODES = ["r", "w"]

    def __init__(self, *args, **kwargs):
        fileio.FileIO.__init__(self, *args, **kwargs)
        self.file = open(self.dataPath, self.mode + "b")

    def read(self, n=-1, sparse=False):
        """
        
        Parameters
        ----------
        n : int
            Read at most ``n`` objects. Default is ``-1``.
        sparse : bool
            Flag for returning a sparse weights matrix (``True``).
            Default is ``False``.
        
        Returns
        -------
        w : libpysal.weights.W
            A PySAL `W` object.
        
        """

        self._sparse = sparse
        self._complain_ifclosed(self.closed)
        return self._read()

    def seek(self, pos):
        if pos == 0:
            self.file.seek(0)
            self.pos = 0

    def _read(self):
        """Reads MatrixMarket ``.mtx`` file.
        
        Returns
        -------
        w : {libpysal.weights.W, libpysal.weights.WSP}
            A PySAL `W` object.
        
        Raises
        ------
        StopIteration
            Raised at the EOF.
        
        Examples
        --------

        Type ``dir(w)`` at the interpreter to see what methods are supported.
        Open a MatrixMarket ``.mtx`` file and read it into a PySAL weights object.

        >>> import libpysal
        >>> f = libpysal.io.open(libpysal.examples.get_path('wmat.mtx'), 'r')

        >>> w = f.read()

        Get the number of observations from the header.

        >>> w.n
        49

        Get the mean number of neighbors.

        >>> w.mean_neighbors
        4.73469387755102

        Get neighbor weights for a single observation.

        >>> w[1]
        {2: 0.3333, 5: 0.3333, 6: 0.3333}

        >>> f.close()

        >>> f = libpysal.io.open(libpysal.examples.get_path('wmat.mtx'), 'r')

        >>> wsp = f.read(sparse=True)

        Get the number of observations from the header.

        >>> wsp.n
        49

        Get a row from the weights matrix. Note that the first row in
        the sparse matrix (the 0th row) corresponds to ID 1 from the
        original ``.mtx`` file read in.

        >>> print(wsp.sparse[0].todense())
        [[0.     0.3333 0.     0.     0.3333 0.3333 0.     0.     0.     0.
          0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
          0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
          0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
          0.     0.     0.     0.     0.     0.     0.     0.     0.    ]]
        
        """

        if self.pos > 0:
            raise StopIteration

        mtx = sio.mmread(self.file)
        # matrix market indexes start at one
        ids = list(range(1, mtx.shape[0] + 1))
        wsp = WSP(mtx, ids)

        if self._sparse:
            w = wsp
        else:
            w = wsp.to_W()
        self.pos += 1

        return w

    def write(self, obj):
        """Write a weights object to the opened MatrixMarket ``.mtx`` file.

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
        >>> testfile = libpysal.io.open(libpysal.examples.get_path('wmat.mtx'), 'r')
        >>> w = testfile.read()

        Create a temporary file for this example.

        >>> f = tempfile.NamedTemporaryFile(suffix='.mtx')

        Reassign to a new variable.

        >>> fname = f.name

        Close the temporary named file.

        >>> f.close()

        Open the new file in write mode.

        >>> o = libpysal.io.open(fname, 'w')

        Write the weights object into the open file.

        >>> o.write(w)
        >>> o.close()

        Read in the newly created mtx file.

        >>> wnew = libpysal.io.open(fname, 'r').read()

        Compare values from old to new.

        >>> wnew.pct_nonzero == w.pct_nonzero
        True

        Clean up temporary file created for this example.

        >>> os.remove(fname)

        Go to the beginning of the test file.

        >>> testfile.seek(0)

        Create a sparse weights instance from the test file.

        >>> wsp = testfile.read(sparse=True)

        Open the new file in write mode.

        >>> o = libpysal.io.open(fname, 'w')

        Write the sparse weights object into the open file.

        >>> o.write(wsp)
        >>> o.close()

        Read in the newly created ``.mtx`` file.

        >>> wsp_new =  libpysal.io.open(fname, 'r').read(sparse=True)

        Compare values from old to new.

        >>> wsp_new.s0 == wsp.s0
        True

        Clean up the temporary file created for this example.

        >>> os.remove(fname)

        """

        self._complain_ifclosed(self.closed)
        if issubclass(type(obj), W) or issubclass(type(obj), WSP):
            w = obj.sparse
            sio.mmwrite(
                self.file, w, comment="Generated by PySAL", field="real", precision=7
            )
            self.pos += 1
        else:
            raise TypeError("Expected a PySAL weights object, got: %s." % (type(obj)))

    def close(self):
        self.file.close()
        fileio.FileIO.close(self)

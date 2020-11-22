import scipy.io as sio
from .. import fileio
from ...weights import W
from ...weights.util import full, full2W

__author__ = "Myunghwa Hwang <mhwang4@gmail.com>"
__all__ = ["MatIO"]


class MatIO(fileio.FileIO):
    """Opens, reads, and writes weights file objects in MATLAB Level 4-5 MAT format.
    ``.mat`` files are used in Dr. LeSage's MATLAB Econometrics library. The ``.mat``
    file format can handle both full and sparse matrices, and it allows for a matrix
    dimension greater than 256. In PySAL, row and column headers of a MATLAB array
    are ignored.

    PySAL uses `scipy.io <https://docs.scipy.org/doc/scipy/reference/tutorial/io.html>`_.
    Thus, it is subject to all limits of ``scipy.io.loadmat`` and ``scipy.io.savemat``.

    Notes
    -----
    
    If a given weights object contains too many observations to write it out as
    a full matrix, PySAL writes out the object as a sparse matrix.

    References
    ----------
    
    `MathWorks <http://www.mathworks.com/help/pdf_doc/matlab/matfile_format.pdf>`_
    (2011) "MATLAB 7 MAT-File Format."

    """

    FORMATS = ["mat"]
    MODES = ["r", "w"]

    def __init__(self, *args, **kwargs):
        self._varName = "Unknown"
        fileio.FileIO.__init__(self, *args, **kwargs)
        self.file = open(self.dataPath, self.mode + "b")

    def _set_varName(self, val):
        if issubclass(type(val), str):
            self._varName = val

    def _get_varName(self) -> str:
        return self._varName

    varName = property(fget=_get_varName, fset=_set_varName)

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

    def _read(self):
        """Reads MATLAB ``.mat`` file.
        
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

        Type ``dir(w)`` at the interpreter to see what methods are supported.
        Open a MATLAB ``.mat`` file and read it into a PySAL weights object.

        >>> import libpysal
        >>> w = libpysal.io.open(libpysal.examples.get_path(
        ...     'spat-sym-us.mat'), 'r'
        ... ).read()

        Get the number of observations from the header.

        >>> w.n
        46

        Get the mean number of neighbors.

        >>> w.mean_neighbors
        4.086956521739131

        Get neighbor distances for a single observation.

        >>> w[1] == dict({25: 1, 3: 1, 28: 1, 39: 1})
        True

        """

        if self.pos > 0:
            raise StopIteration

        mat = sio.loadmat(self.file)
        mat_keys = [k for k in mat if not k.startswith("_")]
        full_w = mat[mat_keys[0]]

        self.pos += 1

        w = full2W(full_w)

        return w

    def write(self, obj):
        """Write a weights object to the opened MATLAB ``.mat`` file.

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
        >>> testfile = libpysal.io.open(
        ...     libpysal.examples.get_path('spat-sym-us.mat'), 'r'
        ... )
        >>> w = testfile.read()

        Create a temporary file for this example.

        >>> f = tempfile.NamedTemporaryFile(suffix='.mat')

        Reassign to a new variable.

        >>> fname = f.name

        Close the temporary named file.

        >>> f.close()

        Open the new file in write mode.

        >>> o = libpysal.io.open(fname, 'w')

        Write the weights object into the open file.

        >>> o.write(w)
        >>> o.close()

        Read in the newly created ``.mat`` file.

        >>> wnew =  libpysal.io.open(fname, 'r').read()

        Compare values from old to new.

        >>> wnew.pct_nonzero == w.pct_nonzero
        True

        Clean up temporary file created for this example.

        >>> os.remove(fname)

        """

        self._complain_ifclosed(self.closed)

        if issubclass(type(obj), W):
            try:
                w = full(obj)[0]
            except ValueError:
                w = obj.sparse
            sio.savemat(self.file, {"WEIGHT": w})
            self.pos += 1
        else:
            raise TypeError("Expected a PySAL weights object, got: %s." % (type(obj)))

    def close(self):
        self.file.close()
        fileio.FileIO.close(self)

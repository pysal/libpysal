from . import gwt
from ...weights import W

__author__ = "Myunghwa Hwang <mhwang4@gmail.com>"
__all__ = ["DatIO"]


class DatIO(gwt.GwtIO):
    """Opens, reads, and writes file objects in ``.dat`` format. Spatial weights
    objects in ``.dat`` format are used in Dr. LeSage's MatLab Econ library.
    This ``.dat`` format is a simple text file with a ``.DAT`` or ``.dat``
    extension. Without a header line, it includes three data columns
    for origin ID, destination ID, and weight values as follows:
    
    ```
    [Line 1]    2    1    0.25
    [Line 2]    5    1    0.50
    ```

    Origin/destination IDs in this file format are simply record
    numbers starting with 1. IDs are not necessarily integers.
    Data values for all columns should be numeric.

    """

    FORMATS = ["dat"]
    MODES = ["r", "w"]

    def _read(self):
        """Reads in a ``.dat`` file as a PySAL `W` object.
        
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
        Open ``.dat`` file and read it into a PySAL weights object,

        >>> import libpysal
        >>> w = libpysal.io.open(libpysal.examples.get_path('wmat.dat'), 'r').read()

        Get the number of observations from the header.

        >>> w.n
        49

        Get the mean number of neighbors.

        >>> w.mean_neighbors
        4.73469387755102

        Get neighbor distances for a single observation.

        >>> w[1] == dict({2.0: 0.3333, 5.0: 0.3333, 6.0: 0.3333})
        True

        """

        if self.pos > 0:
            raise StopIteration

        id_type = float
        weights, neighbors = self._readlines(id_type)

        self.pos += 1

        w = W(neighbors, weights)

        return w

    def write(self, obj):
        """Write a weights object to the opened ``.dat`` file.

        Parameters
        ----------
        obj : libpysal.weights.W
            A PySAL `W` object.

        Examples
        --------

        >>> import tempfile, libpysal, os
        >>> testfile = libpysal.io.open(libpysal.examples.get_path('wmat.dat'), 'r')
        >>> w = testfile.read()

        Create a temporary file for this example.

        >>> f = tempfile.NamedTemporaryFile(suffix='.dat')

        Reassign to a new variable.

        >>> fname = f.name

        Close the temporary named file.

        >>> f.close()

        Open the new file in write mode.

        >>> o = libpysal.io.open(fname, 'w')

        Write the weights object into the open file.

        >>> o.write(w)
        >>> o.close()

        Read in the newly created ``.dat`` file.

        >>> wnew =  libpysal.io.open(fname, 'r').read()

        Compare values from old to new.

        >>> wnew.pct_nonzero == w.pct_nonzero
        True

        Clean up the temporary file created for this example.

        >>> os.remove(fname)
        
        """

        self._complain_ifclosed(self.closed)

        if issubclass(type(obj), W):
            self._writelines(obj)
        else:

            raise TypeError("Expected a PySAL weights object, got: %s." % (type(obj)))

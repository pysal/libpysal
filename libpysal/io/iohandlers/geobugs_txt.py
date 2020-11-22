from .. import fileio
from ...weights import W

__author__ = "Myunghwa Hwang <mhwang4@gmail.com>"
__all__ = ["GeoBUGSTextIO"]


class GeoBUGSTextIO(fileio.FileIO):
    """Opens, reads, and writes weights file objects in the text format
    used in `GeoBUGS <http://www.openbugs.net/Manuals/GeoBUGS/Manual.html>`_.
    `GeoBUGS` generates a spatial weights matrix as an R object and writes
    it out as an ASCII text representation of the R object. An exemplary
    `GeoBUGS` text file is as follows.
    
    ```
    list([CARD], [ADJ], [WGT], [SUMNUMNEIGH])
    ```
    
    where ``[CARD]`` and ``[ADJ]`` are required but the others are optional.
    PySAL assumes ``[CARD]`` and ``[ADJ]`` always exist in an input text file.
    It can read a `GeoBUGS` text file, even when its content is not written
    in the order of ``[CARD]``, ``[ADJ]``, ``[WGT]``, and ``[SUMNUMNEIGH]``.
    It always writes all of ``[CARD]``, ``[ADJ]``, ``[WGT]``, and ``[SUMNUMNEIGH]``.
    PySAL does not apply text wrapping during file writing.

    In the above example,

    ```
    [CARD]:
        num = c([a list of comma-splitted neighbor cardinalities])

    [ADJ]:
        adj = c ([a list of comma-splitted neighbor IDs])
        
        If caridnality is zero, neighbor IDs are skipped. The ordering of
        observations is the same in both ``[CARD]`` and ``[ADJ]``.
        Neighbor IDs are record numbers starting from one.

    [WGT]:
        weights = c([a list of comma-splitted weights])
        The restrictions for [ADJ] also apply to ``[WGT]``.

    [SUMNUMNEIGH]:
        sumNumNeigh = [The total number of neighbor pairs]
        the total number of neighbor pairs  is an integer
        value and the same as the sum of neighbor cardinalities.
    ```

    Notes
    -----
    
    For the files generated from R the ``spdep``, ``nb2WB``, and ``dput``
    functions. It is assumed that the value for the control parameter of
    the ``dput`` function is ``NULL``. Please refer to R ``spdep`` and
    ``nb2WB`` functions help files.

    References
    ----------
    
    * **Thomas, A., Best, N., Lunn, D., Arnold, R., and Spiegelhalter, D.**
        (2004) GeoBUGS User Manual. R spdep nb2WB function help file.

    """

    FORMATS = ["geobugs_text"]
    MODES = ["r", "w"]

    def __init__(self, *args, **kwargs):
        args = args[:2]
        fileio.FileIO.__init__(self, *args, **kwargs)
        self.file = open(self.dataPath, self.mode)

    def read(self, n=-1):
        """Read a GeoBUGS text file.

        Returns
        -------
        w : libpysal.weights.W
            A PySAL `W` object.

        Examples
        --------

        Type ``dir(w)`` at the interpreter to see what methods are supported.
        Open a `GeoBUGS` text file and read it into a PySAL weights object.

        >>> import libpysal
        >>> w = libpysal.io.open(
        ...     libpysal.examples.get_path('geobugs_scot'), 'r', 'geobugs_text'
        ... ).read()

        Get the number of observations from the header.

        >>> w.n
        56

        Get the mean number of neighbors.

        >>> w.mean_neighbors
        4.178571428571429

        Get neighbor distances for a single observation.

        >>> w[1] == dict({9: 1.0, 19: 1.0, 5: 1.0})
        True

        """

        self._complain_ifclosed(self.closed)

        w = self._read()

        return w

    def seek(self, pos) -> int:
        if pos == 0:
            self.file.seek(0)
            self.pos = 0

    def _read(self):
        """Reads in a `GeoBUGSTextIO` object.
        
        Raises
        ------
        StopIteration
            Raised at the EOF.
        
        Returns
        -------
        w : libpysal.weights.W
            A PySAL `W` object.
        
        """

        if self.pos > 0:
            raise StopIteration

        fbody = self.file.read()
        body_structure = {}

        for i in ["num", "adj", "weights", "sumNumNeigh"]:
            i_loc = fbody.find(i)

            if i_loc != -1:
                body_structure[i] = (i_loc, i)

        body_sequence = sorted(body_structure.values())
        body_sequence.append((-1, "eof"))

        for i in range(len(body_sequence) - 1):
            part, next_part = body_sequence[i], body_sequence[i + 1]
            start, end = part[0], next_part[0]
            part_text = fbody[start:end]

            part_length, start, end = len(part_text), 0, -1

            for c in range(part_length):
                if part_text[c].isdigit():
                    start = c
                    break

            for c in range(part_length - 1, 0, -1):
                if part_text[c].isdigit():
                    end = c + 1
                    break

            part_text = part_text[start:end]
            part_text = part_text.replace("\n", "")
            value_type = int

            if part[1] == "weights":
                value_type = float

            body_structure[part[1]] = [value_type(v) for v in part_text.split(",")]

        cardinalities = body_structure["num"]
        adjacency = body_structure["adj"]
        raw_weights = [1.0] * int(sum(cardinalities))

        if "weights" in body_structure and isinstance(body_structure["weights"], list):
            raw_weights = body_structure["weights"]

        no_obs = len(cardinalities)
        neighbors = {}
        weights = {}
        pos = 0

        for i in range(no_obs):
            neighbors[i + 1] = []
            weights[i + 1] = []
            no_nghs = cardinalities[i]

            if no_nghs > 0:
                neighbors[i + 1] = adjacency[pos : pos + no_nghs]
                weights[i + 1] = raw_weights[pos : pos + no_nghs]

            pos += no_nghs

        self.pos += 1

        w = W(neighbors, weights)

        return w

    def write(self, obj):
        """Writes a weights object to the opened text file.

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
        ...     libpysal.examples.get_path('geobugs_scot'), 'r', 'geobugs_text'
        ... )
        >>> w = testfile.read()

        Create a temporary file for this example.

        >>> f = tempfile.NamedTemporaryFile(suffix='')

        Reassign to a new variable.

        >>> fname = f.name

        Close the temporary named file.

        >>> f.close()

        Open the new file in write mode.

        >>> o = libpysal.io.open(fname, 'w', 'geobugs_text')

        Write the Weights object into the open file. 

        >>> o.write(w)
        >>> o.close()

        Read in the newly created text file.

        >>> wnew =  libpysal.io.open(fname, 'r', 'geobugs_text').read()

        Compare values from old to new.

        >>> wnew.pct_nonzero == w.pct_nonzero
        True

        Clean up the temporary file created for this example.

        >>> os.remove(fname)

        """

        self._complain_ifclosed(self.closed)

        if issubclass(type(obj), W):

            cardinalities, neighbors, weights = [], [], []
            for i in obj.id_order:
                cardinalities.append(obj.cardinalities[i])
                neighbors.extend(obj.neighbors[i])
                weights.extend(obj.weights[i])

            self.file.write("list(")
            self.file.write("num=c(%s)," % ",".join(map(str, cardinalities)))
            self.file.write("adj=c(%s)," % ",".join(map(str, neighbors)))
            self.file.write("sumNumNeigh=%i)" % sum(cardinalities))
            self.pos += 1

        else:
            raise TypeError("Expected a PySAL weights object, got: %s." % (type(obj)))

    def close(self):
        self.file.close()
        fileio.FileIO.close(self)

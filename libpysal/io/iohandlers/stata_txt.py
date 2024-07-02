# ruff: noqa: SIM115

from ...weights import W
from .. import fileio

__author__ = "Myunghwa Hwang <mhwang4@gmail.com>"
__all__ = ["StataTextIO"]


class StataTextIO(fileio.FileIO):
    """Opens, reads, and writes weights file objects in STATA text format.

    Spatial weights objects in the STATA text format are used in STATA ``sppack``
    library through the ``spmat`` command. This format is a simple text file
    delimited by a whitespace. The ``spmat`` command does not specify which file
    extension to use. But, ``.txt`` seems the default file extension, which is
    assumed in PySAL.

    The first line of the STATA text file is a header including the number of
    observations. After this header line, it includes at least one data column
    that contains unique IDs or record numbers of observations. When an ID
    variable is not specified for the original spatial weights matrix in STATA,
    record numbers are used to identify individual observations, and the record
    numbers start with 1. The ``spmat`` command seems to allow only integer IDs,
    which is also assumed in PySAL.

    A STATA text file can have one of the following structures according to
    its export options in STATA.

    Structure 1: Encoding using the list of neighbor IDs.

    ```
    [Line 1]    [Number_of_Observations]
    [Line 2]    [ID_of_Obs_1] [ID_of_Neighbor_1_of_Obs_1] [ID_of_Neighbor_2_of_Obs_1] ... [ID_of_Neighbor_m_of_Obs_1]
    [Line 3]    [ID_of_Obs_2]
    [Line 4]    [ID_of_Obs_3] [ID_of_Neighbor_1_of_Obs_3] [ID_of_Neighbor_2_of_Obs_3]
    ...
    ```

    Note that for island observations their IDs are still recorded.

    Structure 2: Encoding using a full matrix format.

    ```
    [Line 1]    [Number_of_Observations]
    [Line 2]    [ID_of_Obs_1] [w_11] [w_12] ... [w_1n]
    [Line 3]    [ID_of_Obs_2] [w_21] [w_22] ... [w_2n]
    [Line 4]    [ID_of_Obs_3] [w_31] [w_32] ... [w_3n]
    ...
    [Line n+1]  [ID_of_Obs_n] [w_n1] [w_n2] ... [w_nn]
    ```

    where :math:`w_{ij}` can be a form of general weight. That is, :math:`w_ij`
    can be both a binary value or a general numeric value. If an observation
    is an island, all of its ``w`` columns contain 0.

    References
    ----------

    Drukker D.M., Peng H., Prucha I.R., and Raciborski R. (2011)
    "Creating and managing spatial-weighting matrices using the spmat command"

    Notes
    -----

    The ``spmat`` command allows users to add any note to a spatial weights
    matrix object in STATA. However, all those notes are lost when the matrix
    is exported. PySAL also does not take care of those notes.

    """  # noqa: E501

    FORMATS = ["stata_text"]
    MODES = ["r", "w"]

    def __init__(self, *args, **kwargs):
        args = args[:2]
        fileio.FileIO.__init__(self, *args, **kwargs)
        self.file = open(self.dataPath, self.mode)

    def read(self, n=-1):  # noqa: ARG002
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
        return self._read()

    def seek(self, pos):
        if pos == 0:
            self.file.seek(0)
            self.pos = 0

    def _read(self):
        """Reads STATA Text file
        Returns a pysal.weights.weights.W object

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
        Open a text file and read it into a PySAL weights object.

        >>> import libpysal
        >>> w = libpysal.io.open(
        ...     libpysal.examples.get_path('stata_sparse.txt'), 'r', 'stata_text'
        ... ).read()

        Get the number of observations from the header.

        >>> w.n
        56

        Get the mean number of neighbors.

        >>> w.mean_neighbors
        4.0

        Get neighbor distances for a single observation.

        >>> w[1] == dict({53: 1.0, 51: 1.0, 45: 1.0, 54: 1.0, 7: 1.0})
        True

        """

        if self.pos > 0:
            raise StopIteration

        n = int(self.file.readline().strip())
        line1 = self.file.readline().strip()
        obs_01 = line1.split(" ")
        matrix_form = False

        if len(obs_01) == 1 or float(obs_01[1]) != 0.0:

            def line2wgt(line):
                row = [int(i) for i in line.strip().split(" ")]
                return row[0], row[1:], [1.0] * len(row[1:])

        else:
            matrix_form = True

            def line2wgt(line):
                row = line.strip().split(" ")
                obs = int(float(row[0]))
                ngh, wgt = [], []
                for i in range(n):
                    w = float(row[i + 1])
                    if w > 0:
                        ngh.append(i)
                        wgt.append(w)
                return obs, ngh, wgt

        id_order = []
        weights, neighbors = {}, {}
        l_ = line1

        for _ in range(n):
            obs, ngh, wgt = line2wgt(l_)
            id_order.append(obs)
            neighbors[obs] = ngh
            weights[obs] = wgt
            l_ = self.file.readline()

        if matrix_form:
            for obs in neighbors:
                neighbors[obs] = [id_order[ngh] for ngh in neighbors[obs]]

        self.pos += 1

        w = W(neighbors, weights)

        return w

    def write(self, obj, matrix_form=False):
        """Write a weights object to an opened text file.

        Parameters
        ----------
        obj : libpysal.weights.W
            A PySAL `W` object.
        matrix_form : bool
            Flag for matrix form (``True``). Default is ``False``.

        Raises
        ------
        TypeError
            Raised when the input ``obj`` is not a PySAL `W`.

        Examples
        --------

        >>> import tempfile, libpysal, os
        >>> testfile = libpysal.io.open(
        ...     libpysal.examples.get_path('stata_sparse.txt'), 'r', 'stata_text'
        ... )
        >>> w = testfile.read()

        Create a temporary file for this example.

        >>> f = tempfile.NamedTemporaryFile(suffix='.txt')

        Reassign to a new variable.

        >>> fname = f.name

        Close the temporary named file.

        >>> f.close()

        Open the new file in write mode.

        >>> o = libpysal.io.open(fname, 'w', 'stata_text')

        Write the weights object into the open file.

        >>> o.write(w)
        >>> o.close()

        Read in the newly created text file.

        >>> wnew =  libpysal.io.open(fname, 'r', 'stata_text').read()

        Compare values from old to new.

        >>> wnew.pct_nonzero == w.pct_nonzero
        True

        Clean up the temporary file created for this example.

        >>> os.remove(fname)

        """

        self._complain_ifclosed(self.closed)

        if issubclass(type(obj), W):
            header = f"{obj.n}\n"
            self.file.write(header)
            if matrix_form:

                def wgt2line(obs_id, neighbor, weight):
                    w = ["0.0"] * obj.n
                    for ngh, wgt in zip(neighbor, weight, strict=True):
                        w[obj.id2i[ngh]] = str(wgt)
                    return [str(obs_id)] + w

            else:

                def wgt2line(obs_id, neighbor, _):
                    return [str(obs_id)] + [str(ngh) for ngh in neighbor]

            for id_ in obj.id_order:
                line = wgt2line(id_, obj.neighbors[id_], obj.weights[id_])
                self.file.write("{}\n".format(" ".join(line)))
        else:
            raise TypeError(f"Expected a PySAL weights object, got: {type(obj)}.")

    def close(self):
        self.file.close()
        fileio.FileIO.close(self)

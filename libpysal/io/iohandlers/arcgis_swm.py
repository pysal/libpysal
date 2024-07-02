# ruff: noqa: N802, N806, N815, SIM115

from struct import pack, unpack

import numpy as np

from ...weights import W
from ...weights.util import remap_ids
from .. import fileio

__author__ = "Myunghwa Hwang <mhwang4@gmail.com>"
__all__ = ["ArcGISSwmIO"]


class ArcGISSwmIO(fileio.FileIO):
    """Opens, reads, and writes weights file objects in ArcGIS ``.swm`` format.
    Spatial weights objects in the ArcGIS ``.swm`` format are used in ArcGIS
    Spatial Statistics tools. Particularly, this format can be directly used
    with the tools under the category of Mapping Clusters.

    The values for``ORG_i`` and ``DST_i`` should be integers, as ArcGIS Spatial
    Statistics tools support only unique integer IDs. For the case where a
    weights object uses non-integer IDs, `ArcGISSwmIO` allows users to use
    internal IDs corresponding to record numbers, instead of original IDs.
    The specifics of each part of the above structure is as follows.

    .. table:: ArcGIS SWM Components
    ============ ============ ==================================== ================================
        Part      Data type           Description                               Length
    ============ ============ ==================================== ================================
     ID_VAR_NAME  ASCII TEXT  ID variable name                     Flexible (Up to the 1st ;)
     ESRI_SRS     ASCII TEXT  ESRI spatial reference system        Flexible (Btw the 1st ; and \\n)
     NO_OBS       l.e. int    Number of observations               4
     ROW_STD      l.e. int    Whether or not row-standardized      4
     WGT_i
     ORG_i        l.e. int    ID of observaiton i                  4
     NO_NGH_i     l.e. int    Number of neighbors for obs. i (m)   4
     NGHS_i
     DSTS_i       l.e. int    IDs of all neighbors of obs. i       4*m
     WS_i         l.e. float  Weights for obs. i and its neighbors 8*m
     W_SUM_i      l.e. float  Sum of weights for "                 8
    ============ ============ ==================================== ================================

    """  # noqa: E501

    FORMATS = ["swm"]
    MODES = ["r", "w"]

    def __init__(self, *args, **kwargs):
        self._varName = "Unknown"
        self._srs = "Unknown"
        fileio.FileIO.__init__(self, *args, **kwargs)
        self.file = open(self.dataPath, self.mode + "b")

    def _set_varName(self, val):
        if issubclass(type(val), str):
            self._varName = val

    def _get_varName(self):
        return self._varName

    varName = property(fget=_get_varName, fset=_set_varName)

    def _set_srs(self, val):
        if issubclass(type(val), str):
            self._srs = val

    def _get_srs(self):
        return self._srs

    srs = property(fget=_get_srs, fset=_set_srs)

    def read(self, n=-1):  # noqa: ARG002
        self._complain_ifclosed(self.closed)
        return self._read()

    def seek(self, pos):
        if pos == 0:
            self.file.seek(0)
            self.pos = 0

    def _read(self):
        """Read an ArcGIS ``.swm`` file.

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
        Open an ArcGIS ``.swm`` file and read it into a PySAL weights object.

        >>> import libpysal
        >>> w = libpysal.io.open(libpysal.examples.get_path('ohio.swm'), 'r').read()

        Get the number of observations from the header,

        >>> w.n
        88

        Get the mean number of neighbors.

        >>> w.mean_neighbors
        5.25

        Get neighbor distances for a single observation.

        >>> w[1] == dict({2: 1.0, 11: 1.0, 6: 1.0, 7: 1.0})
        True

        """

        if self.pos > 0:
            raise StopIteration

        header = self.file.readline()
        header = header.decode()

        if header.upper().strip().startswith("VERSION@"):
            #  deal with the new SWM version
            w = self.read_new_version(header)
        else:
            #  deal with the old SWM version
            w = self.read_old_version(header)

        return w

    def read_old_version(self, header):
        """Read the old version of ArcGIS(<10.1) ``.swm`` file.

        Parameters
        ----------
        header : str
            The first line of the ``.swm`` file.

        Returns
        -------
        w : libpysal.weights.W
            A PySAL `W` object.

        """

        id_var, srs = header[:-1].split(";")
        self.varName = id_var
        self.srs = srs
        self.header_len = len(header) + 8
        no_obs, row_std = tuple(unpack("<2l", self.file.read(8)))
        neighbors = {}
        weights = {}

        for _ in range(no_obs):
            origin, no_nghs = tuple(unpack("<2l", self.file.read(8)))
            neighbors[origin] = []
            weights[origin] = []

            if no_nghs > 0:
                neighbors[origin] = list(
                    unpack("<%il" % no_nghs, self.file.read(4 * no_nghs))
                )
                weights[origin] = list(
                    unpack("<%id" % no_nghs, self.file.read(8 * no_nghs))
                )
                _ = list(unpack("<d", self.file.read(8)))[0]

        self.pos += 1

        w = W(neighbors, weights)

        return w

    def read_new_version(self, header_line):
        """Read the new version of ArcGIS(<10.1) ``.swm`` file, which contains
        more parameters and records weights in two ways, fixed or variable.

        Parameters
        ----------
        header_line : str
            The first line of the ``.swm`` file, which contains a lot of
            parameters. The parameters are divided by semicolons (';') and
            the key-value of each parameter is divided by at marks ('@').

        Returns
        -------
        w : libpysal.weights.W
            A PySAL `W` object.

        """

        headerDict = {}
        for item in header_line.split(";"):
            key, value = item.split("@")
            headerDict[key] = value

        # for the reader, in order to generate the PySAL Weight class,
        # only a few of the parameters are needed.
        self.varName = headerDict["UNIQUEID"]
        self.srs = headerDict["SPATIALREFNAME"]

        fixedWeights = False
        if "FIXEDWEIGHTS" in headerDict:
            fixedWeights = headerDict["FIXEDWEIGHTS"].upper().strip() == "TRUE"

        no_obs, row_std = tuple(unpack("<2l", self.file.read(8)))

        neighbors = {}
        weights = {}

        for _ in range(no_obs):
            origin, no_nghs = tuple(unpack("<2l", self.file.read(8)))
            neighbors[origin] = []
            weights[origin] = []

            if no_nghs > 0:
                neighbors[origin] = list(
                    unpack("<%il" % no_nghs, self.file.read(4 * no_nghs))
                )

                if fixedWeights:
                    weights[origin] = list(unpack("<d", self.file.read(8))) * no_nghs
                else:
                    weights[origin] = list(
                        unpack("<%id" % no_nghs, self.file.read(8 * no_nghs))
                    )
                _ = list(unpack("<d", self.file.read(8)))[0]

        self.pos += 1

        w = W(neighbors, weights)

        return w

    def write(self, obj, useIdIndex=False):  # noqa: N803
        """Writes a spatial weights matrix data file in ``.swm`` format.

        Parameters
        ----------
        obj : libpysal.weights.W
            A PySAL `W` object.
        useIdIndex : bool
            Use the `W` IDs and remap (``True``). Default is ``False``.

        Raises
        ------
        TypeError
            Raised when the input ``obj`` is not a PySAL `W`.
        TypeError
            Raised when the IDs in input ``obj`` are not integers.

        Examples
        --------

        >>> import tempfile, libpysal, os
        >>> testfile = libpysal.io.open(libpysal.examples.get_path('ohio.swm'), 'r')
        >>> w = testfile.read()

        Create a temporary file for this example.

        >>> f = tempfile.NamedTemporaryFile(suffix='.swm')

        Reassign to a new variable.

        >>> fname = f.name

        Close the temporary named file.

        >>> f.close()

        Open the new file in write mode.

        >>> o = libpysal.io.open(fname, 'w')

        Add properities to the file to write.

        >>> o.varName = testfile.varName
        >>> o.srs = testfile.srs

        Write the weights object into the open file.

        >>> o.write(w)
        >>> o.close()

        Read in the newly created text file.

        >>> wnew = libpysal.io.open(fname, 'r').read()

        Compare values from old to new.

        >>> wnew.pct_nonzero == w.pct_nonzero
        True

        Clean up the temporary file created for this example.

        >>> os.remove(fname)

        """

        self._complain_ifclosed(self.closed)

        if not issubclass(type(obj), W):
            raise TypeError(f"Expected a PySAL weights object, got: {type(obj)}.")

        if (type(obj.id_order[0]) not in (np.int32, np.int64, int)) and not useIdIndex:
            raise TypeError("ArcGIS SWM files support only integer IDs.")

        if useIdIndex:
            id2i = obj.id2i
            obj = remap_ids(obj, id2i)

        unk = str(f"{self.varName};{self.srs}\n").encode()
        self.file.write(unk)
        self.file.write(pack("<l", obj.n))
        self.file.write(pack("<l", obj.transform.upper() == "R"))

        for obs in obj.weights:
            self.file.write(pack("<l", obs))
            no_nghs = len(obj.weights[obs])
            self.file.write(pack("<l", no_nghs))
            self.file.write(pack(f"<{no_nghs}l", *obj.neighbors[obs]))
            self.file.write(pack(f"<{no_nghs}d", *obj.weights[obs]))
            self.file.write(pack("<d", sum(obj.weights[obs])))

        self.pos += 1

    def close(self):
        self.file.close()
        fileio.FileIO.close(self)

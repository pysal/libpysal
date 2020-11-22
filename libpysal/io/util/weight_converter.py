import os
from ..fileio import FileIO as psopen
from warnings import warn

__author__ = "Myunghwa Hwang <mhwang4@gmail.com>"
__all__ = ["weight_convert"]


class WeightConverter(object):
    """Opens and reads a weights file in a format, then writes
    the file in other formats.

    `WeightConverter` can read a weights file in the following formats:
    `GAL`, `GWT`, `ArcGIS DBF/SWM/Text`, `DAT`, `MAT`, `MTX`, `WK1`,
    `GeoBUGS Text`, and `STATA Text`. It can convert the input file into
    all of the formats listed above, except `GWT`. Currently, PySAL does
    not support writing a weights object in the `GWT` format.

    When an input weight file includes multiple islands and the format of
    an output weight file is `ArcGIS DBF/SWM/TEXT`, `DAT`, or `WK1`, the
    number of observations in the new weights file will be the original
    number of observations substracted by the number of islands. This is
    because `ArcGIS DBF/SWM/TEXT`, `DAT`, `WK1` formats ignore islands.

    """

    def __init__(self, inputPath, dataFormat=None):

        warn("WeightConverter will be deprecated in PySAL 3.1.", DeprecationWarning)

        self.inputPath = inputPath
        self.inputDataFormat = dataFormat
        self._setW()

    def _setW(self):
        """Reads a weights file and sets a
        ``pysal.weights.W`` object as an attribute.
        
        Raises
        ------
        IOError
            Raised when there is a problem reading in the file.
        RuntimeError
            Raised when there is a problem creating the weights object.
        
        Examples
        --------

        Create a WeightConvert object.

        >>> import libpysal
        >>> wc = WeightConverter(
        ...     libpysal.examples.get_path('arcgis_ohio.dbf'), dataFormat='arcgis_dbf'
        ... )

        Check whether or not the `W` object is set as an attribute.

        >>> wc.w_set()
        True

        Get the number of observations included in the `W` object.

        >>> wc.w.n
        88

        """

        try:
            if self.inputDataFormat:
                f = psopen(self.inputPath, "r", self.inputDataFormat)
            else:
                f = psopen(self.inputPath, "r")
        except:
            raise IOError("A problem occurred while reading the input file.")
        else:
            try:
                self.w = f.read()
            except:
                raise RuntimeError(
                    "A problem occurred while creating a weights object."
                )
            finally:
                f.close()

    def w_set(self) -> bool:
        """Checks if a source `W` object is set."""

        return hasattr(self, "w")

    def write(self, outputPath, dataFormat=None, useIdIndex=True, matrix_form=True):
        """
        
        Parameters
        ----------
        outputPath : str
            The path to the output weights file.
        dataFormat : str
            The type of data format. Options include: ``'arcgis_dbf'`` for the
            `ArcGIS DBF` format, ``'arcgis_text'`` for the `ArcGIS Text` format,
            ``'geobugs_text'`` for the `GeoBUGS Text` format, and ``'stata_text'``
            for the `STATA Text` format. Default is ``None``.
        useIdIndex : bool
            Applies only to `ArcGIS DBF/SWM/Text` formats. Default is ``True``.
        matrix_form : bool
            Applies only to the `STATA Text` format. Default is ``True``.

        Raises
        ------
        RuntimeError
            Raised when there is no weights file passed in.
        IOError
            Raised when there is a problem creating the file.
        RuntimeError
            Raised when there is a problem writing the weights object.

        Examples
        --------
        
        >>> import tempfile, os, libpysal

        Create a `WeightConverter` object.

        >>> wc = WeightConverter(libpysal.examples.get_path('sids2.gal'))

        Check whether or not the `W` object is set as an attribute.

        >>> wc.w_set()
        True

        Create a temporary file for this example.

        >>> f = tempfile.NamedTemporaryFile(suffix='.dbf')

        Reassign to the new variable.

        >>> fname = f.name

        Close the temporary named file.

        >>> f.close()

        Write the input ``.gal`` file in the `ArcGIS` ``.dbf`` format.

        >>> wc.write(fname, dataFormat='arcgis_dbf', useIdIndex=True)

        Create a new weights object from the converted ``.dbf`` file.

        >>> wnew = psopen(fname, 'r', 'arcgis_dbf').read()

        Compare the number of observations in the two `W` objects.

        >>> wc.w.n == wnew.n
        True

        Clean up the temporary file.

        >>> os.remove(fname)

        """

        ext = os.path.splitext(outputPath)[1]
        ext = ext.replace(".", "")
        # if ext.lower() == "gwt":
        #    msg = "Currently, PySAL does not support writing "
        #    msg += "a weights object into a gwt file."
        #    raise TypeError(msg)

        if not self.w_set():
            raise RuntimeError("There is no weights object to write out.")

        try:
            if dataFormat:
                o = psopen(outputPath, "w", dataFormat)
            else:
                o = psopen(outputPath, "w")
        except:
            raise IOError("A problem occurred while creating the output file.")
        else:
            try:
                if dataFormat in ["arcgis_text", "arcgis_dbf"] or ext == "swm":
                    o.write(self.w, useIdIndex=useIdIndex)
                elif dataFormat == "stata_text":
                    o.write(self.w, matrix_form=matrix_form)
                else:
                    o.write(self.w)
            except:
                raise RuntimeError(
                    "A problem occurred while writing out the weights object."
                )
            finally:
                o.close()


def weight_convert(
    inPath,
    outPath,
    inDataFormat=None,
    outDataFormat=None,
    useIdIndex=True,
    matrix_form=True,
):
    """
    A utility function for directly converting a given weight
    file into the format specified in ``outPath``.

    Parameters
    ----------
    inPath : str
        The path to the input weights file.
    outPath : str
        The path to the output weights file.
    indataFormat : str
        The type of data format. Options include: ``'arcgis_dbf'`` for the
        `ArcGIS DBF` format, ``'arcgis_text'`` for the `ArcGIS Text` format,
        ``'geobugs_text'`` for the `GeoBUGS Text` format, and ``'stata_text'``
        for the `STATA Text` format. Default is ``None``.
    outdataFormat : str
        The type of data format. Options include: ``'arcgis_dbf'`` for the
        `ArcGIS DBF` format, ``'arcgis_text'`` for the `ArcGIS Text` format,
        ``'geobugs_text'`` for the `GeoBUGS Text` format, and ``'stata_text'``
        for the `STATA Text` format. Default is ``None``.
    useIdIndex : bool
        Applies only to `ArcGIS DBF/SWM/Text` formats. Default is ``True``.
    matrix_form : bool
        Applies only to the `STATA Text` format. Default is ``True``.

    Examples
    --------
    
    >>> import tempfile, os, libpysal

    Create a temporary file for this example.

    >>> f = tempfile.NamedTemporaryFile(suffix='.dbf')

    Reassign to the new variable.

    >>> fname = f.name

    Close the temporary named file.

    >>> f.close()

    Create a `WeightConverter` object.

    >>> weight_convert(
    ...     libpysal.examples.get_path('sids2.gal'),
    ...     fname,
    ...     outDataFormat='arcgis_dbf',
    ...     useIdIndex=True
    ... )

    Create a new weights object from the ``.gal`` file.

    >>> wold = libpysal.io.open(libpysal.examples.get_path('sids2.gal'), 'r').read()

    Create a new weights object from the converted ``.dbf`` file.

    >>> wnew = libpysal.io.open(fname, 'r', 'arcgis_dbf').read()

    Compare the number of observations in the two `W` objects.

    >>> wold.n == wnew.n
    True

    Clean up the temporary file.

    >>> os.remove(fname)

    """

    converter = WeightConverter(inPath, dataFormat=inDataFormat)
    converter.write(
        outPath,
        dataFormat=outDataFormat,
        useIdIndex=useIdIndex,
        matrix_form=matrix_form,
    )

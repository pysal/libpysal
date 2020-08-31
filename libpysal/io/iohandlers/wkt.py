from .. import fileio
from ..util import WKTParser

__author__ = "Charles R Schmidt <schmidtc@gmail.com>"
__all__ = ["WKTReader"]

#####################################################################
## ToDo: Add Well-Known-Binary support...
##       * WKB spec:
##  http://webhelp.esri.com/arcgisserver/9.3/dotNet/index.htm#geodatabases/the_ogc_103951442.htm
#####################################################################


class WKTReader(fileio.FileIO):
    """Reads Well-Known Text into PySAL polygon objects.

    Examples
    --------
    
    Read in WKT-formatted file.

    >>> import libpysal
    >>> f = libpysal.io.open(libpysal.examples.get_path('stl_hom.wkt'), 'r')

    Convert ``wkt`` to PySAL polygons.

    >>> polys = f.read()

    Check length.

    >>> len(polys)
    78

    Return centroid of polygon at index 1.

    >>> polys[1].centroid
    (-91.19578469430738, 39.990883050220845)

    Type ``dir(polys[1])`` at the python interpreter to get a list of supported methods.

    """

    MODES = ["r"]
    FORMATS = ["wkt"]

    def __init__(self, *args, **kwargs):
        fileio.FileIO.__init__(self, *args, **kwargs)
        self.__idx = {}
        self.__pos = 0
        self.__open()

    def open(self):
        self.__open()

    def __open(self):
        self.dataObj = open(self.dataPath, self.mode)
        self.wkt = WKTParser()

    def _read(self):
        """

        Returns
        -------
        shape : libpysal.cg.Polygon
            Geometric shape.

        """

        fileio.FileIO._complain_ifclosed(self.closed)
        if self.__pos not in self.__idx:
            self.__idx[self.__pos] = self.dataObj.tell()
        line = self.dataObj.readline()

        if line:
            shape = self.wkt.fromWKT(line)
            shape.id = self.pos
            self.__pos += 1
            self.pos += 1
            return shape
        else:
            self.seek(0)
            return None

    def seek(self, n):
        """

        Raises
        ------
        IndexError
            Raised when an incorrect index is used.

        """

        fileio.FileIO.seek(self, n)
        pos = self.pos
        if pos in self.__idx:
            self.dataObj.seek(self.__idx[pos])
            self.__pos = pos
        else:
            while pos not in self.__idx:
                s = self._read()
                if not s:
                    msg = "%d not in range(0,%d)." % (pos, max(self.__idx.keys()))
                    raise IndexError(msg)
            self.pos = pos
            self.__pos = pos
            self.dataObj.seek(self.__idx[pos])

    def close(self):
        self.dataObj.close()
        fileio.FileIO.close(self)

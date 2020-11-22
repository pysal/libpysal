from .. import tables
import csv

from typing import Union

__author__ = "Charles R Schmidt <schmidtc@gmail.com>"
__all__ = ["csvWrapper"]


class csvWrapper(tables.DataTable):
    """Read a ``.csv`` file in a `DataTable` object.

    Examples
    --------
    
    >>> import libpysal
    >>> stl = libpysal.examples.load_example('stl')
    >>> file_name = stl.get_path('stl_hom.csv')
    >>> f = libpysal.io.open(file_name,'r')
    >>> y = f.read()
    >>> f.header
    ['WKT',
     'NAME',
     'STATE_NAME',
     'STATE_FIPS',
     'CNTY_FIPS',
     'FIPS',
     'FIPSNO',
     'HR7984',
     'HR8488',
     'HR8893',
     'HC7984',
     'HC8488',
     'HC8893',
     'PO7984',
     'PO8488',
     'PO8893',
     'PE77',
     'PE82',
     'PE87',
     'RDAC80',
     'RDAC85',
     'RDAC90']
    
    >>> f._spec
    [str,
     str,
     str,
     int,
     int,
     int,
     int,
     float,
     float,
     float,
     int,
     int,
     int,
     int,
     int,
     int,
     float,
     float,
     float,
     float,
     float,
     float]
    
    """

    __doc__ = tables.DataTable.__doc__
    FORMATS = ["csv"]
    READ_MODES = ["r", "Ur", "rU", "U"]
    MODES = READ_MODES[:]

    def __init__(self, *args, **kwargs):

        tables.DataTable.__init__(self, *args, **kwargs)
        self.__idx = {}
        self.__len = None
        self._open()

    def __len__(self):
        return self.__len

    def _open(self):

        self.fileObj = open(self.dataPath, self.mode)
        if self.mode in self.READ_MODES:
            self.dataObj = csv.reader(self.fileObj)
            data = list(self.dataObj)
            if self._determineHeader(data):
                self.header = data.pop(0)
            else:
                self.header = ["field_%d" % i for i in range(len(data[0]))]
            self._spec = self._determineSpec(data)
            self.data = data
            self.fileObj.close()
            self.__len = len(data)

    def _determineHeader(self, data: list) -> bool:

        HEADER = True

        headSpec = self._determineSpec([data[0]])
        restSpec = self._determineSpec(data[1:])

        if headSpec == restSpec:
            HEADER = False

        return HEADER

    @staticmethod
    def _determineSpec(data: list) -> list:

        cols = len(data[0])
        spec = []

        for j in range(cols):
            isInt = True
            isFloat = True

            for row in data:
                val = row[j]
                if not val.strip().replace("-", "").replace(".", "").isdigit():
                    isInt = False
                    isFloat = False
                    break
                else:
                    if isInt and "." in val:
                        isInt = False

            if isInt:
                spec.append(int)
            elif isFloat:
                spec.append(float)
            else:
                spec.append(str)

        return spec

    def _read(self) -> Union[list, None]:

        if self.pos < len(self):
            row = self.data[self.pos]
            self.pos += 1
            return row
        else:
            return None

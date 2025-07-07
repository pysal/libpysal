"""A Pure Python Shapefile Reader and Writer

This module is self-contained and does not require pysal. It returns and expects
dictionary-based data structures. It should be wrapped into your native data structures.

Contact:
    Charles Schmidt
    GeoDa Center
    Arizona State University
    Tempe, AZ
    http://geodacenter.asu.edu

"""

# ruff: noqa: N801, N802, N803, N806, SIM115

__author__ = "Charles R Schmidt <schmidtc@gmail.com>"

import array
import io
import sys
from itertools import islice
from struct import calcsize, pack, unpack

SYS_BYTE_ORDER = "<" if sys.byteorder == "little" else ">"

STRUCT_ITEMSIZE = {}
STRUCT_ITEMSIZE["i"] = calcsize("i")
STRUCT_ITEMSIZE["d"] = calcsize("d")

__all__ = ["shp_file", "shx_file"]

# SHAPEFILE Globals


def struct2arrayinfo(struct: tuple) -> list:
    struct = list(struct)
    lname, ltype, lorder = struct.pop(0)
    groups = {}
    g = 0
    groups[g] = {
        "names": [lname],
        "size": STRUCT_ITEMSIZE[ltype],
        "fmt": ltype,
        "order": lorder,
    }

    while struct:
        name, type_, order = struct.pop(0)

        if order == lorder:
            groups[g]["names"].append(name)
            groups[g]["size"] += STRUCT_ITEMSIZE[type_]
            groups[g]["fmt"] += type_
        else:
            g += 1
            groups[g] = {
                "names": [name],
                "size": STRUCT_ITEMSIZE[type_],
                "fmt": type_,
                "order": order,
            }
        lname, ltype, lorder = name, type_, order

    return [groups[x] for x in range(g + 1)]


HEADERSTRUCT = (
    ("File Code", "i", ">"),
    ("Unused0", "i", ">"),
    ("Unused1", "i", ">"),
    ("Unused2", "i", ">"),
    ("Unused3", "i", ">"),
    ("Unused4", "i", ">"),
    ("File Length", "i", ">"),
    ("Version", "i", "<"),
    ("Shape Type", "i", "<"),
    ("BBOX Xmin", "d", "<"),
    ("BBOX Ymin", "d", "<"),
    ("BBOX Xmax", "d", "<"),
    ("BBOX Ymax", "d", "<"),
    ("BBOX Zmin", "d", "<"),
    ("BBOX Zmax", "d", "<"),
    ("BBOX Mmin", "d", "<"),
    ("BBOX Mmax", "d", "<"),
)
UHEADERSTRUCT = struct2arrayinfo(HEADERSTRUCT)
RHEADERSTRUCT = (("Record Number", "i", ">"), ("Content Length", "i", ">"))
URHEADERSTRUCT = struct2arrayinfo(RHEADERSTRUCT)


def noneMax(a: float | None, b: float | None) -> float:
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


def noneMin(a: float | None, b: float | None) -> float:
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


def _unpackDict(structure, fileObj):
    """Utility function that requires a tuple of tuples that
    describe the element structure.

    Parameters
    ----------
    structure : tuple
        A tuple of tuples in the form:
        ``(('FieldName 1','type','byteOrder'),('FieldName 2','type','byteOrder'))``.
    fileObj : file
        An open file at the correct position.

    Returns
    -------
    d : dict
        Dictionary in the form: ``{'FieldName 1': value, 'FieldName 2': value}``.

    Notes
    -----
    The file is at new position.

    Examples
    --------
    >>> import libpysal
    >>> _unpackDict(
    ...     UHEADERSTRUCT,
    ...     open(
    ...         libpysal.examples.get_path('10740.shx'), 'rb')
    ... ) == \
    ... {
    ...     'BBOX Xmax': -105.29012,
    ...     'BBOX Ymax': 36.219799000000002,
    ...     'BBOX Mmax': 0.0,
    ...     'BBOX Zmin': 0.0,
    ...     'BBOX Mmin': 0.0,
    ...     'File Code': 9994,
    ...     'BBOX Ymin': 34.259672000000002,
    ...     'BBOX Xmin': -107.62651,
    ...     'Unused0': 0,
    ...     'Unused1': 0,
    ...     'Unused2': 0,
    ...     'Unused3': 0,
    ...     'Unused4': 0,
    ...     'Version': 1000,
    ...     'BBOX Zmax': 0.0,
    ...     'Shape Type': 5,
    ...     'File Length': 830
    ... }
    True
    """

    d = {}

    for struct in structure:
        items = unpack(struct["order"] + struct["fmt"], fileObj.read(struct["size"]))
        for i, name in enumerate(struct["names"]):
            d[name] = items[i]

    return d


def _unpackDict2(d, structure, fileObj):
    """Utility Function, used arrays instead from struct.

    Parameters
    ----------
    d : dict
        Dictionary in to be updated.
    structure : tuple
        A tuple of tuples in the form:
        ``(('FieldName 1','type','byteOrder'),('FieldName 2','type','byteOrder'))``.
    fileObj : file
        An open file at the correct position.

    Returns
    -------
    d : dict
        The updated dictionary.
    """

    for name, dtype, order in structure:
        dtype, n = dtype
        result = array.array(dtype)
        result.frombytes(fileObj.read(result.itemsize * n))
        if order != SYS_BYTE_ORDER:
            result.byteswap()
        d[name] = result.tolist()

    return d


def _packDict(structure, d) -> str:
    """Utility Function for packing a dictionary with byte strings.

    Parameters
    ----------
    structure : tuple
        A tuple of tuples in the form:
        ``(('FieldName 1','type','byteOrder'),('FieldName 2','type','byteOrder'))``.
    d : dict
        Dictionary in the form: ``{'FieldName 1': value, 'FieldName 2': value}``.

    Examples
    --------
    >>> s = _packDict(
    ...     (('FieldName 1', 'i', '<'), ('FieldName 2', 'i', '<')),
    ...     {'FieldName 1': 1, 'FieldName 2': 2}
    ... )
    >>> s == pack('<ii', 1, 2)
    True

    >>> unpack('<ii', s)
    (1, 2)
    """

    string = b""

    for name, dtype, order in structure:
        if len(dtype) > 1:
            string += pack(order + dtype, *d[name])
        else:
            string += pack(order + dtype, d[name])

    return string


class shp_file:
    """Reads and writes the SHP compenent of a shapefile.

    Parameters
    ----------
    filename : str
        The name of the file to create.
    mode : str
        The mode for file interaction, either ``'r'`` (read)
        or ``'w'`` (write). Default is ``'r'``.
    shape_type : str
        Must be one of the following: ``'POINT'``, ``'POINTZ'``, ``'POINTM'``,
        ``'ARC'``, ``'ARCZ'``, ``'ARCM'``, ``'POLYGON'``, ``'POLYGONZ'``,
        ``'POLYGONM'``, ``'MULTIPOINT'``, ``'MULTIPOINTZ'``, ``'MULTIPOINTM'``,
        ``'MULTIPATCH'``. Default is ``None``.

    Attributes
    ----------
    header : dict
        Contents of the SHP header. For contents see ``HEADERSTRUCT``.
    shape : int
        See ``SHAPE_TYPES`` and ``TYPE_DISPATCH``.

    Examples
    --------
    >>> import libpysal
    >>> shp = shp_file(libpysal.examples.get_path('10740.shp'))
    >>> shp.header == {
    ...     'BBOX Xmax': -105.29012,
    ...     'BBOX Ymax': 36.219799000000002,
    ...     'BBOX Mmax': 0.0,
    ...     'BBOX Zmin': 0.0,
    ...     'BBOX Mmin': 0.0,
    ...     'File Code': 9994,
    ...     'BBOX Ymin': 34.259672000000002,
    ...     'BBOX Xmin': -107.62651,
    ...     'Unused0': 0,
    ...     'Unused1': 0,
    ...     'Unused2': 0,
    ...     'Unused3': 0,
    ...     'Unused4': 0,
    ...     'Version': 1000,
    ...     'BBOX Zmax': 0.0,
    ...     'Shape Type': 5,
    ...     'File Length': 260534
    ... }
    True

    >>> len(shp)
    195

    Notes
    -----
    The header of both the SHP and SHX files are indentical.
    """

    SHAPE_TYPES = {
        "POINT": 1,
        "ARC": 3,
        "POLYGON": 5,
        "MULTIPOINT": 8,
        "POINTZ": 11,
        "ARCZ": 13,
        "POLYGONZ": 15,
        "MULTIPOINTZ": 18,
        "POINTM": 21,
        "ARCM": 23,
        "POLYGONM": 25,
        "MULTIPOINTM": 28,
        "MULTIPATCH": 31,
    }

    def __iswritable(self) -> bool:
        """

        Raises
        ------
        IOError
            Raised when a bad file name is passed in.
        """

        try:
            assert self.__mode == "w"
        except AssertionError:
            raise OSError("[Errno 9] Bad file descriptor.") from None
        return True

    def __isreadable(self) -> bool:
        """

        Raises
        ------
        IOError
            Raised when a bad file name is passed in.
        """

        try:
            assert self.__mode == "r"
        except AssertionError:
            raise OSError("[Errno 9] Bad file descriptor.") from None
        return True

    def __init__(self, fileName, mode="r", shape_type=None):
        """

        Raises
        ------
        Exception
            Raised when an invalid shape type is passed in.
        Exception
            Raised when an invalid mode is passed in.
        """

        self.__mode = mode

        if (
            fileName.lower().endswith(".shp")
            or fileName.lower().endswith(".shx")
            or fileName.lower().endswith(".dbf")
        ):
            fileName = fileName[:-4]

        self.fileName = fileName

        if mode == "r":
            self._open_shp_file()
        elif mode == "w":
            if shape_type not in self.SHAPE_TYPES:
                raise Exception("Attempt to create shp/shx file of invalid type.")
            self._create_shp_file(shape_type)
        else:
            raise Exception("Only 'w' and 'r' modes are supported.")

    def _open_shp_file(self):
        """Opens a shp/shx file."""

        self.__isreadable()
        fileName = self.fileName
        self.fileObj = open(fileName + ".shp", "rb")
        self._shx = shx_file(fileName)
        self.header = _unpackDict(UHEADERSTRUCT, self.fileObj)
        self.shape = TYPE_DISPATCH[self.header["Shape Type"]]
        self.__lastShape = 0

        # localizing for convenience
        self.__numRecords = self._shx.numRecords

        # constructing bounding box from header
        h = self.header
        self.bbox = [h["BBOX Xmin"], h["BBOX Ymin"], h["BBOX Xmax"], h["BBOX Ymax"]]
        self.shapeType = self.header["Shape Type"]

    def _create_shp_file(self, shape_type: str):
        """Creates a shp/shx file.

        Examples
        --------
        >>> import libpysal, os
        >>> shp = shp_file('test', 'w', 'POINT')
        >>> p = shp_file(libpysal.examples.get_path('Point.shp'))
        >>> for pt in p:
        ...   shp.add_shape(pt)
        >>> shp.close()
        >>> open('test.shp','rb').read() == open(
        ...     libpysal.examples.get_path('Point.shp'), 'rb'
        ... ).read()
        True

        >>> open('test.shx', 'rb').read() == open(
        ...     libpysal.examples.get_path('Point.shx'), 'rb'
        ... ).read()
        True

        >>> os.remove('test.shx')
        >>> os.remove('test.shp')
        """

        self.__iswritable()
        fileName = self.fileName
        self.fileObj = open(fileName + ".shp", "wb")
        self._shx = shx_file(fileName, "w")
        self.header = {}
        self.header["Shape Type"] = self.SHAPE_TYPES[shape_type]
        self.header["Version"] = 1000
        self.header["Unused0"] = 0
        self.header["Unused1"] = 0
        self.header["Unused2"] = 0
        self.header["Unused3"] = 0
        self.header["Unused4"] = 0
        self.header["File Code"] = 9994
        self.__file_Length = 100
        self.header["File Length"] = 0
        self.header["BBOX Xmax"] = None
        self.header["BBOX Ymax"] = None
        self.header["BBOX Mmax"] = None
        self.header["BBOX Zmax"] = None
        self.header["BBOX Xmin"] = None
        self.header["BBOX Ymin"] = None
        self.header["BBOX Mmin"] = None
        self.header["BBOX Zmin"] = None
        self.shape = TYPE_DISPATCH[self.header["Shape Type"]]
        # self.__numRecords = self._shx.numRecords

    def __len__(self) -> int:
        return self.__numRecords

    def __iter__(self):
        return self

    def type(self) -> str:  # noqa: A003
        return self.shape.String_Type

    def __next__(self) -> int:
        """Returns the next shape in the shapefile.

        Raises
        ------
        StopIteration
            Raised at the EOF.

        Examples
        --------
        >>> import libpysal
        >>> list(shp_file(libpysal.examples.get_path('Point.shp'))) == [
        ...     {
        ...         'Y': -0.25904661905760773,
        ...         'X': -0.00068176617532103578,
        ...         'Shape Type': 1
        ...     },
        ...     {
        ...         'Y': -0.25630328607387354,
        ...         'X': 0.11697145363360706,
        ...         'Shape Type': 1
        ...     },
        ...     {
        ...         'Y': -0.33930131004366804,
        ...         'X': 0.05043668122270728,
        ...         'Shape Type': 1
        ...     },
        ...     {
        ...         'Y': -0.41266375545851519,
        ...         'X': -0.041266375545851552,
        ...         'Shape Type': 1
        ...     },
        ...     {
        ...         'Y': -0.44017467248908293,
        ...         'X': -0.011462882096069604,
        ...         'Shape Type': 1
        ...     },
        ...     {
        ...         'Y': -0.46080786026200882,
        ...         'X': 0.027510917030567628,
        ...         'Shape Type': 1
        ...     },
        ...     {
        ...         'Y': -0.45851528384279472,
        ...         'X': 0.075655021834060809,
        ...         'Shape Type': 1
        ...     },
        ...     {
        ...         'Y': -0.43558951965065495,
        ...         'X': 0.11233624454148461,
        ...         'Shape Type': 1
        ...     },
        ...     {
        ...         'Y': -0.40578602620087334,
        ...         'X': 0.13984716157205224,
        ...         'Shape Type': 1
        ...     }
        ... ]
        True
        """

        self.__isreadable()
        nextShape = self.__lastShape
        if nextShape == self._shx.numRecords:
            self.__lastShape = 0
            raise StopIteration
        else:
            self.__lastShape = nextShape + 1
            return self.get_shape(nextShape)

    def __seek(self, pos: int):
        if pos != self.fileObj.tell():
            self.fileObj.seek(pos)

    def __read(self, pos: int, size: int):
        self.__isreadable()
        if pos != self.fileObj.tell():
            self.fileObj.seek(pos)
        return self.fileObj.read(size)

    def get_shape(self, shpId: int) -> dict:
        self.__isreadable()
        if shpId + 1 > self.__numRecords:
            raise IndexError
        fPosition, byts = self._shx.index[shpId]
        self.__seek(fPosition)

        # the index does not include the 2 byte record header
        # (which contains, Record ID and Content Length)

        rec_id, con_len = _unpackDict(URHEADERSTRUCT, self.fileObj)

        return self.shape.unpack(io.BytesIO(self.fileObj.read(byts)))
        # return self.shape.unpack(self.fileObj.read(bytes))

    def __update_bbox(self, s: dict):
        h = self.header
        if s.get("Shape Type") == 1:
            h["BBOX Xmax"] = noneMax(h["BBOX Xmax"], s.get("X"))
            h["BBOX Ymax"] = noneMax(h["BBOX Ymax"], s.get("Y"))
            h["BBOX Mmax"] = noneMax(h["BBOX Mmax"], s.get("M"))
            h["BBOX Zmax"] = noneMax(h["BBOX Zmax"], s.get("Z"))
            h["BBOX Xmin"] = noneMin(h["BBOX Xmin"], s.get("X"))
            h["BBOX Ymin"] = noneMin(h["BBOX Ymin"], s.get("Y"))
            h["BBOX Mmin"] = noneMin(h["BBOX Mmin"], s.get("M"))
            h["BBOX Zmin"] = noneMin(h["BBOX Zmin"], s.get("Z"))
        else:
            h["BBOX Xmax"] = noneMax(h["BBOX Xmax"], s.get("BBOX Xmax"))
            h["BBOX Ymax"] = noneMax(h["BBOX Ymax"], s.get("BBOX Ymax"))
            h["BBOX Mmax"] = noneMax(h["BBOX Mmax"], s.get("BBOX Mmax"))
            h["BBOX Zmax"] = noneMax(h["BBOX Zmax"], s.get("BBOX Zmax"))
            h["BBOX Xmin"] = noneMin(h["BBOX Xmin"], s.get("BBOX Xmin"))
            h["BBOX Ymin"] = noneMin(h["BBOX Ymin"], s.get("BBOX Ymin"))
            h["BBOX Mmin"] = noneMin(h["BBOX Mmin"], s.get("BBOX Mmin"))
            h["BBOX Zmin"] = noneMin(h["BBOX Zmin"], s.get("BBOX Zmin"))

        if not self.shape.HASM:
            self.header["BBOX Mmax"] = 0.0
            self.header["BBOX Mmin"] = 0.0
        if not self.shape.HASZ:
            self.header["BBOX Zmax"] = 0.0
            self.header["BBOX Zmin"] = 0.0

    def add_shape(self, s: dict):
        self.__iswritable()
        self.__update_bbox(s)
        rec = self.shape.pack(s)
        con_len = len(rec)
        self.__file_Length += con_len + 8
        rec_id, pos = self._shx.add_record(con_len)
        self.__seek(pos)
        self.fileObj.write(pack(">ii", rec_id, con_len // 2))
        self.fileObj.write(rec)

    def close(self):
        self._shx.close(self.header)
        if self.__mode == "w":
            self.header["File Length"] = self.__file_Length // 2
            self.__seek(0)
            self.fileObj.write(_packDict(HEADERSTRUCT, self.header))
        self.fileObj.close()


class shx_file:
    """Reads and writes the SHX compenent of a shapefile.

    Parameters
    ----------
    filename : str
        The name of the file to create. Default is ``None``.
        The extension is optional, will remove ``'.dbf'``,
        ``'.shx'``, ``'.shp'`` and append ``'.shx'``.
    mode : str
        The mode for file interaction. Must be ``'r'`` (read).

    Attributes
    ----------
    index : list
        Contains the file offset and length of each recond in the SHP component.
    numRecords : int
        The number of records.

    Examples
    --------
    >>> import libpysal
    >>> shx = shx_file(libpysal.examples.get_path('10740.shx'))
    >>> shx._header == {
    ...     'BBOX Xmax': -105.29012,
    ...     'BBOX Ymax': 36.219799000000002,
    ...     'BBOX Mmax': 0.0,
    ...     'BBOX Zmin': 0.0,
    ...     'BBOX Mmin': 0.0,
    ...     'File Code': 9994,
    ...     'BBOX Ymin': 34.259672000000002,
    ...     'BBOX Xmin': -107.62651,
    ...     'Unused0': 0,
    ...     'Unused1': 0,
    ...     'Unused2': 0,
    ...     'Unused3': 0,
    ...     'Unused4': 0,
    ...     'Version': 1000,
    ...     'BBOX Zmax': 0.0,
    ...     'Shape Type': 5,
    ...     'File Length': 830
    ... }
    True

    >>> len(shx.index)
    195

    >>> shx = shx_file(libpysal.examples.get_path('Point.shx'))
    >>> isinstance(shx, shx_file)
    True
    """

    def __iswritable(self) -> bool:
        """

        Raises
        ------
        IOError
            Raised when a bad file name is passed in.
        """

        try:
            assert self.__mode == "w"
        except AssertionError:
            raise OSError("[Errno 9] Bad file descriptor.") from None
        return True

    def __isreadable(self) -> bool:
        """

        Raises
        ------
        IOError
            Raised when a bad file name is passed in.
        """

        try:
            assert self.__mode == "r"
        except AssertionError:
            raise OSError("[Errno 9] Bad file descriptor.") from None
        return True

    def __init__(self, fileName=None, mode="r"):
        self.__mode = mode
        if (
            fileName.endswith(".shp")
            or fileName.endswith(".shx")
            or fileName.endswith(".dbf")
        ):
            fileName = fileName[:-4]
        self.fileName = fileName

        if mode == "r":
            self._open_shx_file()
        elif mode == "w":
            self._create_shx_file()

    def _open_shx_file(self):
        """Opens the SHX file."""

        self.__isreadable()
        self.fileObj = open(self.fileName + ".shx", "rb")
        self._header = _unpackDict(UHEADERSTRUCT, self.fileObj)
        self.numRecords = numRecords = (self._header["File Length"] - 50) // 4
        fmt = f">{2 * numRecords}i"
        size = calcsize(fmt)
        dat = unpack(fmt, self.fileObj.read(size))
        self.index = [(dat[i] * 2, dat[i + 1] * 2) for i in range(0, len(dat), 2)]

    def _create_shx_file(self):
        """Creates the SHX file."""

        self.__iswritable()
        self.fileObj = open(self.fileName + ".shx", "wb")
        self.numRecords = 0
        self.index = []

        # length of header
        self.__offset = 100
        # record IDs start at 1
        self.__next_rid = 1

    def add_record(self, size: int):
        """Add a record to the shx index.

        Parameters
        ----------
        size : int
            The length of the record in bytes NOT including the 8-byte record header.

        Returns
        -------
        rec_id : int
            The sequential record ID, 1-based.
        pos : int
            See ``self.__offset`` in ``_create_shx_file``.

        Notes
        -----
        The SHX records contain (Offset, Length) in 16-bit words.

        Examples
        --------
        >>> import libpysal, os
        >>> shx = shx_file(libpysal.examples.get_path('Point.shx'))
        >>> shx.index
        [(100, 20),
         (128, 20),
         (156, 20),
         (184, 20),
         (212, 20),
         (240, 20),
         (268, 20),
         (296, 20),
         (324, 20)]

        >>> shx2 = shx_file('test', 'w')
        >>> [shx2.add_record(rec[1]) for rec in shx.index]
        [(1, 100),
         (2, 128),
         (3, 156),
         (4, 184),
         (5, 212),
         (6, 240),
         (7, 268),
         (8, 296),
         (9, 324)]

        >>> shx2.index == shx.index
        True

        >>> shx2.close(shx._header)
        >>> open('test.shx', 'rb').read() == open(
        ...     libpysal.examples.get_path('Point.shx'), 'rb'
        ... ).read()
        True

        >>> os.remove('test.shx')
        """

        self.__iswritable()
        pos = self.__offset
        rec_id = self.__next_rid
        self.index.append((self.__offset, size))

        # the 8-byte record header.
        self.__offset += size + 8
        self.numRecords += 1
        self.__next_rid += 1

        return rec_id, pos

    def close(self, header: dict):
        if self.__mode == "w":
            self.__iswritable()
            header["File Length"] = (self.numRecords * calcsize(">ii") + 100) // 2
            self.fileObj.seek(0)
            self.fileObj.write(_packDict(HEADERSTRUCT, header))
            fmt = f">{2 * self.numRecords}i"
            values = []
            for off, size in self.index:
                values.extend([off // 2, size // 2])
            self.fileObj.write(pack(fmt, *values))

        self.fileObj.close()


class NullShape:
    Shape_Type = 0
    STRUCT = ("Shape Type", "i", "<")

    def unpack(self) -> None:
        return None

    def pack(self, x=None) -> str:  # noqa: ARG002
        return pack("<i", 0)


class Point:
    """Packs and unpacks a shapefile Point type.

    Examples
    --------
    >>> import libpysal
    >>> shp = shp_file(libpysal.examples.get_path('Point.shp'))
    >>> rec = shp.get_shape(0)
    >>> rec == (
    ...     {'Y': -0.25904661905760773, 'X': -0.00068176617532103578, 'Shape Type': 1}
    ... )
    True

    >>> # +8 byte record header
    >>> pos = shp.fileObj.seek(shp._shx.index[0][0] + 8)
    >>> dat = shp.fileObj.read(shp._shx.index[0][1])
    >>> dat == Point.pack(rec)
    True
    """

    Shape_Type = 1
    String_Type = "POINT"
    HASZ = False
    HASM = False
    STRUCT = (("Shape Type", "i", "<"), ("X", "d", "<"), ("Y", "d", "<"))
    USTRUCT = [
        {"fmt": "idd", "order": "<", "names": ["Shape Type", "X", "Y"], "size": 20}
    ]

    @classmethod
    def unpack(cls, dat) -> dict:
        """
        Parameters
        ----------
        dat : file
            An open file at the correct position.
        """

        return _unpackDict(cls.USTRUCT, dat)

    @classmethod
    def pack(cls, record: dict) -> str:
        rheader = _packDict(cls.STRUCT, record)
        return rheader


class PointZ(Point):
    Shape_Type = 11
    String_Type = "POINTZ"
    HASZ = True
    HASM = True
    STRUCT = (
        ("Shape Type", "i", "<"),
        ("X", "d", "<"),
        ("Y", "d", "<"),
        ("Z", "d", "<"),
        ("M", "d", "<"),
    )
    USTRUCT = [
        {
            "fmt": "idddd",
            "order": "<",
            "names": ["Shape Type", "X", "Y", "Z", "M"],
            "size": 36,
        }
    ]


class PolyLine:
    """Packs and unpacks a shapefile PolyLine type.

    Examples
    --------
    >>> import libpysal
    >>> shp = shp_file(libpysal.examples.get_path('Line.shp'))
    >>> rec = shp.get_shape(0)
    >>> rec == {
    ...     'BBOX Ymax': -0.25832280562918325,
    ...     'NumPoints': 3,
    ...     'BBOX Ymin': -0.25895877033237352,
    ...     'NumParts': 1,
    ...     'Vertices': [
    ...         (-0.0090539248870159517, -0.25832280562918325),
    ...         (0.0074811573959305822, -0.25895877033237352),
    ...         (0.0074811573959305822, -0.25895877033237352)
    ...     ],
    ...     'BBOX Xmax': 0.0074811573959305822,
    ...     'BBOX Xmin': -0.0090539248870159517,
    ...     'Shape Type': 3,
    ...     'Parts Index': [0]
    ... }
    True

    >>> # +8 byte record header
    >>> pos = shp.fileObj.seek(shp._shx.index[0][0] + 8)
    >>> dat = shp.fileObj.read(shp._shx.index[0][1])
    >>> dat == PolyLine.pack(rec)
    True
    """

    HASZ = False
    HASM = False
    String_Type = "ARC"
    STRUCT = (
        ("Shape Type", "i", "<"),
        ("BBOX Xmin", "d", "<"),
        ("BBOX Ymin", "d", "<"),
        ("BBOX Xmax", "d", "<"),
        ("BBOX Ymax", "d", "<"),
        ("NumParts", "i", "<"),
        ("NumPoints", "i", "<"),
    )
    USTRUCT = [
        {
            "fmt": "iddddii",
            "order": "<",
            "names": [
                "Shape Type",
                "BBOX Xmin",
                "BBOX Ymin",
                "BBOX Xmax",
                "BBOX Ymax",
                "NumParts",
                "NumPoints",
            ],
            "size": 44,
        }
    ]

    @classmethod
    def unpack(cls, dat) -> dict:
        """

        Parameters
        ----------
        dat : file
            An open file at the correct position.
        """

        record = _unpackDict(cls.USTRUCT, dat)
        content_struct = (
            ("Parts Index", ("i", record["NumParts"]), "<"),
            ("Vertices", ("d", 2 * record["NumPoints"]), "<"),
        )
        _unpackDict2(record, content_struct, dat)

        # record['Vertices'] = [
        #    (record['Vertices'][i], record['Vertices'][i+1])
        #    for i in range(0, record['NumPoints']*2, 2)
        # ]
        verts = record["Vertices"]

        # Next line is equivalent to: zip(verts[::2],verts[1::2])
        record["Vertices"] = list(
            zip(islice(verts, 0, None, 2), islice(verts, 1, None, 2), strict=True)
        )
        if not record["Parts Index"]:
            record["Parts Index"] = [0]

        return record
        # partsIndex = list(partsIndex)
        # partsIndex.append(None)
        # parts = [
        #     vertices[partsIndex[i]:partsIndex[i+1]] for i in range(header['NumParts'])
        # ]

    @classmethod
    def pack(cls, record: dict) -> str:
        rheader = _packDict(cls.STRUCT, record)
        content_struct = (
            ("Parts Index", f"{record['NumParts']}i", "<"),
            ("Vertices", f"{2 * record['NumPoints']}d", "<"),
        )
        content = {}
        content["Parts Index"] = record["Parts Index"]
        verts = []
        [verts.extend(vert) for vert in record["Vertices"]]
        content["Vertices"] = verts
        content = _packDict(content_struct, content)

        return rheader + content


class PolyLineZ:
    HASZ = True
    HASM = True
    String_Type = "ARC"
    STRUCT = (
        ("Shape Type", "i", "<"),
        ("BBOX Xmin", "d", "<"),
        ("BBOX Ymin", "d", "<"),
        ("BBOX Xmax", "d", "<"),
        ("BBOX Ymax", "d", "<"),
        ("NumParts", "i", "<"),
        ("NumPoints", "i", "<"),
    )
    USTRUCT = [
        {
            "fmt": "iddddii",
            "order": "<",
            "names": [
                "Shape Type",
                "BBOX Xmin",
                "BBOX Ymin",
                "BBOX Xmax",
                "BBOX Ymax",
                "NumParts",
                "NumPoints",
            ],
            "size": 44,
        }
    ]

    @classmethod
    def unpack(cls, dat) -> dict:
        """

        Parameters
        ----------
        dat : file
            An open file at the correct position.
        """

        record = _unpackDict(cls.USTRUCT, dat)
        content_struct = (
            ("Parts Index", ("i", record["NumParts"]), "<"),
            ("Vertices", ("d", 2 * record["NumPoints"]), "<"),
            ("Zmin", ("d", 1), "<"),
            ("Zmax", ("d", 1), "<"),
            ("Zarray", ("d", record["NumPoints"]), "<"),
            ("Mmin", ("d", 1), "<"),
            ("Mmax", ("d", 1), "<"),
            ("Marray", ("d", record["NumPoints"]), "<"),
        )

        _unpackDict2(record, content_struct, dat)
        verts = record["Vertices"]
        record["Vertices"] = list(
            zip(islice(verts, 0, None, 2), islice(verts, 1, None, 2), strict=True)
        )

        if not record["Parts Index"]:
            record["Parts Index"] = [0]
        record["Zmin"] = record["Zmin"][0]
        record["Zmax"] = record["Zmax"][0]
        record["Mmin"] = record["Mmin"][0]
        record["Mmax"] = record["Mmax"][0]

        return record

    @classmethod
    def pack(cls, record: dict) -> str:
        rheader = _packDict(cls.STRUCT, record)
        content_struct = (
            ("Parts Index", f"{record['NumParts']}i", "<"),
            ("Vertices", f"{2 * record['NumPoints']}d", "<"),
            ("Zmin", "d", "<"),
            ("Zmax", "d", "<"),
            ("Zarray", f"{record['NumPoints']}d", "<"),
            ("Mmin", "d", "<"),
            ("Mmax", "d", "<"),
            ("Marray", f"{record['NumPoints']}d", "<"),
        )

        content = {}
        content.update(record)
        content["Parts Index"] = record["Parts Index"]
        verts = []
        [verts.extend(vert) for vert in record["Vertices"]]
        content["Vertices"] = verts
        content = _packDict(content_struct, content)

        return rheader + content


class Polygon(PolyLine):
    """Packs and unpacks a shapefile Polygon type identical to PolyLine.

    Examples
    --------
    >>> import libpysal
    >>> shp = shp_file(libpysal.examples.get_path('Polygon.shp'))
    >>> rec = shp.get_shape(1)
    >>> rec == {
    ...     'BBOX Ymax': -0.3126531125455273,
    ...     'NumPoints': 7,
    ...     'BBOX Ymin': -0.35957259110238166,
    ...     'NumParts': 1,
    ...     'Vertices': [
    ...         (0.05396439570183631, -0.3126531125455273),
    ...         (0.051473095955454629, -0.35251390848763364),
    ...         (0.059777428443393454, -0.34254870950210703),
    ...         (0.063099161438568974, -0.34462479262409174),
    ...         (0.048981796209073003, -0.35957259110238166),
    ...         (0.046905713087088297, -0.3126531125455273),
    ...         (0.05396439570183631, -0.3126531125455273)
    ...     ],
    ...     'BBOX Xmax': 0.063099161438568974,
    ...     'BBOX Xmin': 0.046905713087088297,
    ...     'Shape Type': 5,
    ...     'Parts Index': [0]
    ...     }
    True

    >>> # +8 byte record header
    >>> pos = shp.fileObj.seek(shp._shx.index[1][0] + 8)
    >>> dat = shp.fileObj.read(shp._shx.index[1][1])
    >>> dat == Polygon.pack(rec)
    True
    """

    String_Type = "POLYGON"


class MultiPoint:
    String_Type = "MULTIPOINT"

    def __init__(self):
        raise NotImplementedError("No MultiPoint support at this time.")


class PolygonZ(PolyLineZ):
    String_Type = "POLYGONZ"


class MultiPointZ:
    String_Type = "MULTIPOINTZ"

    def __init__(self):
        raise NotImplementedError("No MultiPointZ support at this time.")


class PointM:
    String_Type = "POINTM"

    def __init__(self):
        raise NotImplementedError("No PointM support at this time.")


class PolyLineM:
    String_Type = "ARCM"

    def __init__(self):
        raise NotImplementedError("No PolyLineM support at this time.")


class PolygonM:
    String_Type = "POLYGONM"

    def __init__(self):
        raise NotImplementedError("No PolygonM support at this time.")


class MultiPointM:
    String_Type = "MULTIPOINTM"

    def __init__(self):
        raise NotImplementedError("No MultiPointM support at this time.")


class MultiPatch:
    String_Type = "MULTIPATCH"

    def __init__(self):
        raise NotImplementedError("No MultiPatch support at this time.")


TYPE_DISPATCH = {
    0: NullShape,
    1: Point,
    3: PolyLine,
    5: Polygon,
    8: MultiPoint,
    11: PointZ,
    13: PolyLineZ,
    15: PolygonZ,
    18: MultiPointZ,
    21: PointM,
    23: PolyLineM,
    25: PolygonM,
    28: MultiPointM,
    31: MultiPatch,
    "POINT": Point,
    "POINTZ": PointZ,
    "POINTM": PointM,
    "ARC": PolyLine,
    "ARCZ": PolyLineZ,
    "ARCM": PolyLineM,
    "POLYGON": Polygon,
    "POLYGONZ": PolygonZ,
    "POLYGONM": PolygonM,
    "MULTIPOINT": MultiPoint,
    "MULTIPOINTZ": MultiPointZ,
    "MULTIPOINTM": MultiPointM,
    "MULTIPATCH": MultiPatch,
}

from .. import tables
from ...common import MISSINGVALUE
import datetime
import struct
import os
import time

from typing import Union

__author__ = "Charles R Schmidt <schmidtc@gmail.com>"
__all__ = ["DBF"]


class DBF(tables.DataTable):
    """PySAL DBF Reader/Writer. This DBF handler implements the PySAL DataTable
    interface and initializes an instance of the PySAL's DBF handler.
    
    Parameters
    -----------
    dataPath : str
        Path to file, including file name and extension.
    mode : str
        Mode for file interaction; either ``'r'`` or ``'w'``.
    
    Attributes
    ----------
    header : list
        A list of field names. The header is a python list of strings. Each string
        is a field name and field name must not be longer than 10 characters.
    field_spec : list
        A list describing the data types of each field. It is comprised of a list
        of tuples, each tuple describing a field. The format for the tuples is
        ``('Type', len, precision)``. Valid values for ``'Type'`` are ``'C'``
        for characters, ``'L'`` for bool, ``'D'`` for data, and ``'N'`` or
        ``'F'`` for number.

    Examples
    --------

    >>> import libpysal
    >>> dbf = libpysal.io.open(libpysal.examples.get_path('juvenile.dbf'), 'r')
    >>> dbf.header
    ['ID', 'X', 'Y']
    
    >>> dbf.field_spec
    [('N', 9, 0), ('N', 9, 0), ('N', 9, 0)]

    """

    FORMATS = ["dbf"]
    MODES = ["r", "w"]

    def __init__(self, *args, **kwargs):

        tables.DataTable.__init__(self, *args, **kwargs)

        if self.mode == "r":
            self.f = f = open(self.dataPath, "rb")
            # from dbf file standards
            numrec, lenheader = struct.unpack("<xxxxLH22x", f.read(32))

            # each field is 32 bytes
            numfields = (lenheader - 33) // 32
            self.n_records = numrec
            self.n_fields = numfields
            self.field_info = [("DeletionFlag", "C", 1, 0)]
            record_size = 1

            # each record is a string
            fmt = "s"

            self._col_index = {}
            idx = 0

            for fieldno in range(numfields):
                # again, check struct for fmt def.
                name, typ, size, deci = struct.unpack("<11sc4xBB14x", f.read(32))
                # forces to unicode in 2, to str in 3
                name = name.decode()
                typ = typ.decode()
                # same as NULs, \x00
                name = name.replace("\0", "")
                # eliminate NULs from string
                self._col_index[name] = (idx, record_size)
                idx += 1
                # alt: str(size) + 's'
                fmt += "%ds" % size

                record_size += size
                self.field_info.append((name, typ, size, deci))

            terminator = f.read(1).decode()

            assert terminator == "\r"
            self.header_size = self.f.tell()
            self.record_size = record_size
            self.record_fmt = fmt
            self.pos = 0
            self.header = [fInfo[0] for fInfo in self.field_info[1:]]
            field_spec = []

            for fname, ftype, flen, fpre in self.field_info[1:]:
                field_spec.append((ftype, flen, fpre))

            self.field_spec = field_spec

            # self.spec = [types[fInfo[0]] for fInfo in self.field_info]

        elif self.mode == "w":
            self.f = open(self.dataPath, "wb")
            self.header = None
            self.field_spec = None
            self.numrec = 0
            self.FIRST_WRITE = True

    def __len__(self) -> int:
        """
        
        Raises
        ------
        IOError
            Raised when a file is open ``'w'`` mode.
        
        """

        if self.mode != "r":
            msg = "Invalid operation, cannot read from a file opened in 'w' mode."
            raise IOError(msg)

        return self.n_records

    def seek(self, i):
        self.f.seek(self.header_size + (self.record_size * i))
        self.pos = i

    def _get_col(self, key: str) -> list:
        """Return the column vector.
        
        Raises
        ------
        AttributeError
            Raised when a field does not exist in the header.
        
        """

        if key not in self._col_index:
            raise AttributeError("Field: %s does not exist in header." % key)

        prevPos = self.tell()
        idx, offset = self._col_index[key]
        typ, size, deci = self.field_spec[idx]
        gap = self.record_size - size
        f = self.f
        f.seek(self.header_size + offset)
        col = [0] * self.n_records

        for i in range(self.n_records):
            value = f.read(size)
            value = value.decode()
            f.seek(gap, 1)

            if typ == "N":
                value = value.replace("\0", "").lstrip()
                if value == "":
                    value = MISSINGVALUE
                elif deci:
                    try:
                        value = float(value)
                    except ValueError:
                        value = MISSINGVALUE
                else:
                    try:
                        value = int(value)
                    except ValueError:
                        value = MISSINGVALUE
            elif typ == "D":
                try:
                    y, m, d = int(value[:4]), int(value[4:6]), int(value[6:8])
                    value = datetime.date(y, m, d)
                except ValueError:
                    value = MISSINGVALUE
            elif typ == "L":
                value = (value in "YyTt" and "T") or (value in "NnFf" and "F") or "?"
            elif typ == "F":
                value = value.replace("\0", "").lstrip()
                if value == "":
                    value = MISSINGVALUE
                else:
                    value = float(value)
            if isinstance(value, str) or isinstance(value, str):
                value = value.rstrip()
            col[i] = value

        self.seek(prevPos)

        return col

    def read_record(self, i: int) -> list:

        self.seek(i)

        rec = list(struct.unpack(self.record_fmt, self.f.read(self.record_size)))
        rec = [entry.decode() for entry in rec]

        if rec[0] != " ":
            return self.read_record(i + 1)
        result = []

        for (name, typ, size, deci), value in zip(self.field_info, rec):
            if name == "DeletionFlag":
                continue
            if typ == "N":
                value = value.replace("\0", "").lstrip()
                if value == "":
                    value = MISSINGVALUE
                elif deci:
                    try:
                        value = float(value)
                    except ValueError:
                        value = MISSINGVALUE
                else:
                    try:
                        value = int(value)
                    except ValueError:
                        value = MISSINGVALUE
            elif typ == "D":
                try:
                    y, m, d = int(value[:4]), int(value[4:6]), int(value[6:8])
                    value = datetime.date(y, m, d)
                except ValueError:
                    # value = datetime.date.min#NULL Date: See issue 114
                    value = MISSINGVALUE
            elif typ == "L":
                value = (value in "YyTt" and "T") or (value in "NnFf" and "F") or "?"
            elif typ == "F":
                value = value.replace("\0", "").lstrip()
                if value == "":
                    value = MISSINGVALUE
                else:
                    value = float(value)
            if isinstance(value, str) or isinstance(value, str):
                value = value.rstrip()
            result.append(value)

        return result

    def _read(self) -> Union[list, None]:
        """
        
        Raises
        ------
        IOError
            Raised when a file is open ``'w'`` mode.
        
        """

        if self.mode != "r":
            msg = "Invalid operation, cannot read from a file opened in 'w' mode."
            raise IOError(msg)

        if self.pos < len(self):
            rec = self.read_record(self.pos)
            self.pos += 1
            return rec
        else:
            return None

    def write(self, obj: list):
        """
        
        Raises
        ------
        IOError
            Raised when a file is open ``'r'`` mode.
        TypeError
            Raised when a row length and header length are not equivalent.
        
        """

        self._complain_ifclosed(self.closed)

        if self.mode != "w":
            msg = "Invalid operation, cannot read from a file opened in 'r' mode."
            raise IOError(msg)

        if self.FIRST_WRITE:
            self._firstWrite()

        if len(obj) != len(self.header):
            raise TypeError("Rows must contains %d fields." % len(self.header))

        self.numrec += 1

        # deletion flag
        self.f.write(" ".encode())

        for (typ, size, deci), value in zip(self.field_spec, obj):
            if value is None:
                if typ == "C":
                    value = " " * size
                else:
                    value = "\0" * size
            elif typ == "N" or typ == "F":
                v = str(value).rjust(size, " ")
                # if len(v) == size:
                #    value = v
                # else:
                value = (("%" + "%d.%d" % (size, deci) + "f") % (value))[:size]
            elif typ == "D":
                value = value.strftime("%Y%m%d")
            elif typ == "L":
                value = str(value)[0].upper()
            else:
                value = str(value)[:size].ljust(size, " ")
            try:
                assert len(value) == size
            except:
                print(value, len(value), size)
                raise
            self.f.write(value.encode())
            self.pos += 1

    def flush(self):
        self._complain_ifclosed(self.closed)
        self._writeHeader()
        self.f.flush()

    def close(self):

        if self.mode == "w":
            self.flush()
            # End of file
            self.f.write("\x1A".encode())
        self.f.close()

        tables.DataTable.close(self)

    def _firstWrite(self):
        """
        
        Raises
        ------
        IOError
            Raised when there is no specified header.
        IOError
            Raised when there is no field specification.
        
        """

        if not self.header:
            raise IOError("No header, DBF files require a header.")
        if not self.field_spec:
            raise IOError("No field_spec, DBF files require a specification.")

        self._writeHeader()

        self.FIRST_WRITE = False

    def _writeHeader(self):
        """Modified from:
        http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/362715
        """

        POS = self.f.tell()
        self.f.seek(0)
        ver = 3
        now = datetime.datetime.utcfromtimestamp(
            int(os.environ.get("SOURCE_DATE_EPOCH", time.time())),
        )

        yr, mon, day = now.year - 1900, now.month, now.day
        numrec = self.numrec
        numfields = len(self.header)
        lenheader = numfields * 32 + 33
        lenrecord = sum(field[1] for field in self.field_spec) + 1

        hdr = struct.pack(
            "<BBBBLHH20x", ver, yr, mon, day, numrec, lenheader, lenrecord
        )
        self.f.write(hdr)

        # field specs
        for name, (typ, size, deci) in zip(self.header, self.field_spec):
            typ = typ.encode()
            name = name.ljust(11, "\x00")
            name = name.encode()
            fld = struct.pack("<11sc4xBB14x", name, typ, size, deci)
            self.f.write(fld)

        # terminator
        term = "\r".encode()
        self.f.write(term)
        if self.f.tell() != POS and not self.FIRST_WRITE:
            self.f.seek(POS)


if __name__ == "__main__":
    import libpysal

    file_name = libpysal.examples.get_path("10740.dbf")
    f = libpysal.open(file_name, "r")
    newDB = libpysal.open("copy.dbf", "w")

    newDB.header = f.header
    newDB.field_spec = f.field_spec
    print(f.header)
    for row in f:
        print(row)
        newDB.write(row)
    newDB.close()
    copy = libpysal.open("copy.dbf", "r")

    f.seek(0)
    print("HEADER: ", copy.header == f.header)
    print("SPEC: ", copy.field_spec == f.field_spec)
    print("DATA: ", list(copy) == list(f))

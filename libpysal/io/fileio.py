"""
FileIO: Module for reading and writing various file types in a Pythonic way.
This module should not be used directly, instead...

```
import pysal.core.FileIO as FileIO
```

Readers and Writers will mimic python file objects.
    .seek(n) seeks to the n'th object
    .read(n) reads n objects, default == all
    .next() reads the next object

"""


# ruff: noqa: ARG002, N801, N802, N803, N806, SIM115

__author__ = "Charles R Schmidt <schmidtc@gmail.com>"

__all__ = ["FileIO"]

import os.path
from warnings import warn

from ..common import MISSINGVALUE


class FileIO_MetaCls(type):
    """This meta class is instantiated when the class is first defined. All
    subclasses of `FileIO` also inherit this meta class, which registers
    their abilities with the FileIO registry. Subclasses must contain
    ``FORMATS`` and ``MODES`` (both are ``type(list)``).

    Raises
    ------
    TypeError
        FileIO subclasses must have ``FORMATS`` and ``MODES`` defined.
    """

    def __new__(cls, name, bases, dict_):
        cls = type.__new__(cls, name, bases, dict_)

        if name != "FileIO" and name != "DataTable":
            if "FORMATS" in dict_ and "MODES" in dict_:
                # msg = "Registering %s with FileIO.\n\tFormats: %r\n\tModes: %r"
                # msg = msg % (name, dict["FORMATS"], dict["MODES"])
                FileIO._register(cls, dict_["FORMATS"], dict_["MODES"])
            else:
                raise TypeError(
                    "FileIO subclasses must have 'FORMATS' and 'MODES' defined."
                )

        return cls


class FileIO(metaclass=FileIO_MetaCls):  # should be a type?
    """Metaclass for supporting spatial data file read and write.

    How this works:

    ``FileIO.open(\\*args) == FileIO(\\*args)``

    When creating a new instance of `FileIO` the ``.__new__`` method intercepts.
    ``.__new__`` parses the filename to determine the ``fileType``. Next,
    ``.__registry`` and checked for that type. Each type supports one or more modes
    (``['r', 'w', 'a', etc.]``). If we support the type and mode, an instance of the
    appropriate handler is created and returned. All handlers must inherit from this
    class, and by doing so are automatically added to the ``.__registry`` and are
    forced to conform to the prescribed API. The metaclass takes care of the
    registration by parsing the class definition. It doesn't make much sense to
    treat weights in the same way as shapefiles and dbfs, so...

    * ... for now we'll just return an instance of `W` on ``mode='r'``.
    * ... on ``mode='w'``, ``.write`` will expect an instance of `W`.
    """

    __registry = {}  # {'shp':{'r':[OGRshpReader,pysalShpReader]}}

    def __new__(cls, dataPath="", mode="r", dataFormat=None):
        """Intercepts the instantiation of ``FileIO`` and dispatches
        to the correct handler. If no suitable handler is found a
        python file object is returned.
        """

        if cls is FileIO:
            try:
                newCls = object.__new__(
                    cls.__registry[cls.getType(dataPath, mode, dataFormat)][mode][0]
                )
            except KeyError:
                return open(dataPath, mode)
            return newCls
        else:
            return object.__new__(cls)

    @staticmethod
    def getType(dataPath: str, mode: str, dataFormat=None) -> str:
        """Parse the ``dataPath`` and return the data type."""

        if dataFormat:
            ext = dataFormat
        else:
            ext = os.path.splitext(dataPath)[1]
            ext = ext.replace(".", "")
            ext = ext.lower()
        if ext == "txt":
            with open(dataPath, mode) as f:
                l1 = f.readline()
                l2 = f.readline()
                try:
                    n, k = l1.split(",")
                    n, k = int(n), int(k)
                    fields = l2.split(",")
                    assert len(fields) == k
                    return "geoda_txt"
                except AssertionError:
                    return ext

        return ext

    @classmethod
    def _register(cls, parser, formats, modes):
        """This method is called automatically via the Metaclass of `FileIO` subclasses
        This should be private, but that hides it from the Metaclass.
        """

        assert cls is FileIO

        for format_ in formats:
            if format_ not in cls.__registry:
                cls.__registry[format_] = {}
            for mode in modes:
                if mode not in cls.__registry[format_]:
                    cls.__registry[format_][mode] = []
                cls.__registry[format_][mode].append(parser)
        # cls.check()

    @classmethod
    def check(cls):
        """Prints the contents of the registry."""

        print("PySAL File I/O understands the following file extensions:")

        for key, val in list(cls.__registry.items()):
            print(f"Ext: '.{key}', Modes: {list(val.keys())!r}")

    @classmethod
    def open(cls, *args, **kwargs):  # noqa A001
        """Alias for ``FileIO()``."""

        return cls(*args, **kwargs)

    class _By_Row:
        def __init__(self, parent):
            self.p = parent

        def __repr__(self) -> str:
            if not self.p.ids:
                return "keys: range(0,n)"
            else:
                return "keys: " + list(self.p.ids.keys()).__repr__()

        def __getitem__(self, key) -> list | str:
            if isinstance(key, list):
                r = []
                if self.p.ids:
                    for k in key:
                        r.append(self.p.get(self.p.ids[k]))
                else:
                    for k in key:
                        r.append(self.p.get(k))
                return r
            if self.p.ids:
                return self.p.get(self.p.ids[key])
            else:
                return self.p.get(key)

        __call__ = __getitem__

    def __init__(self, dataPath="", mode="r", dataFormat=None):
        self.dataPath = dataPath
        self.dataObj = ""
        self.mode = mode
        # pos Should ALWAYS be in the range 0,...,n
        # for custom IDs set the ids property.
        self.pos = 0
        self.__ids = None  # {'id':n}
        self.__rIds = None
        self.closed = False
        self._spec = []
        self.header = []

    def __getitem__(self, key):
        return self.by_row.__getitem__(key)

    @property
    def by_row(self):
        return self._By_Row(self)

    def __getIds(self):
        return self.__ids

    def __setIds(self, ids: list | (dict | None)):
        """Property method for ``.ids``. Takes a list of ids and maps then
        to a 0-based index. Need to provide a method to set ID's based on
        a ``fieldName`` preferably without reading the whole file.

        Raises
        ------
        AssertionError
            Raised when IDs are not unique.
        """

        if isinstance(ids, list):
            try:
                assert len(ids) == len(set(ids))
            except AssertionError:
                raise KeyError("IDs must be unique.") from None
            # keys: ID values: i
            self.__ids = {}
            # keys: i values: ID
            self.__rIds = {}
            for i, id_ in enumerate(ids):
                self.__ids[id] = i
                self.__rIds[i] = id_
        elif isinstance(ids, dict):
            self.__ids = ids
            self.__rIds = {}
            for id_, n in list(ids.items()):
                self.__rIds[n] = id_
        elif not ids:
            self.__ids = None
            self.__rIds = None

    ids = property(fget=__getIds, fset=__setIds)

    @property
    def rIds(self) -> dict | None:
        return self.__rIds

    def __iter__(self):
        self.seek(0)
        return self

    @staticmethod
    def _complain_ifclosed(closed):
        """From `StringIO`.

        Raises
        ------
        ValueError
            Raised when a file is already closed.
        """
        if closed:
            raise ValueError("I/O operation on closed file.")

    def cast(self, key, typ):
        """Cast ``key`` as ``typ``.

        Raises
        ------
        TypeError
            Raised when a cast object in not callable.
        KeyError
            Raised when a key is not present.
        """
        if key in self.header:
            if not self._spec:
                self._spec = [lambda x: x for k in self.header]
            if typ is None:
                self._spec[self.header.index(key)] = lambda x: x
            else:
                try:
                    assert callable(typ)
                    self._spec[self.header.index(key)] = typ
                except AssertionError:
                    raise TypeError("Cast objects must be callable.") from None
        else:
            raise KeyError("%s" % key)

    def _cast(self, row) -> list:
        """

        Raises
        ------
        ValueError
            Raised when a value could not be cast a particular type.
        """
        if self._spec and row:
            try:
                return [f(v) for f, v in zip(self._spec, row, strict=True)]
            except ValueError:
                r = []
                for f, v in zip(self._spec, row, strict=True):
                    try:
                        if not v and f != str:
                            raise ValueError
                        r.append(f(v))
                    except ValueError:
                        msg = "Value '%r' could not be cast to %s, "
                        msg += "value set to MISSINGVALUE."
                        msg = msg % (v, str(f))
                        warn(msg, RuntimeWarning, stacklevel=2)
                        r.append(MISSINGVALUE)
                return r

        else:
            return row

    def __next__(self) -> list:
        """A `FileIO` object is its own iterator, see `StringIO`.

        Raises
        ------
        StopIteration
            Raised at the EOF.
        """

        self._complain_ifclosed(self.closed)
        r = self.__read()
        if r is None:
            raise StopIteration

        return r

    def close(self):
        """Subclasses should clean themselves up and then call this method."""

        if not self.closed:
            self.closed = True
            del self.dataObj, self.pos

    def get(self, n: int) -> list:
        """Seeks the file to ``n`` and returns ``n``. If ``.ids`` is set
        ``n`` should be an id, else, ``n`` should be an offset.
        """

        prev_pos = self.tell()
        self.seek(n)
        obj = self.__read()
        self.seek(prev_pos)

        return obj

    def seek(self, n: int):
        """Seek the `FileObj` to the beginning of the ``n``'th record.
        If IDs are set, seeks to the beginning of the record at ID, ``n``.
        """

        self._complain_ifclosed(self.closed)
        self.pos = n

    def tell(self) -> int:
        """Return ID (or offset) of next object."""

        self._complain_ifclosed(self.closed)

        return self.pos

    def read(self, n=-1) -> list | None:
        """Read at most ``n`` objects, less if read hits EOF.
        If size is negative or omitted read all objects until EOF.
        Returns ``None`` if EOF is reached before any objects.

        Raises
        ------
        StopIteration
            Raised at the EOF.
        """

        self._complain_ifclosed(self.closed)

        if n < 0:
            # return list(self)
            result = []
            while 1:
                try:
                    result.append(self.__read())
                except StopIteration:
                    break
            return result
        elif n == 0:
            return None
        else:
            result = []
            for _i in range(0, n):
                try:
                    result.append(self.__read())
                except StopIteration:
                    break
            return result

    def __read(self) -> list:
        """Gets one row from the file handler, and if necessary casts it's objects.

        Raises
        ------
        StopIteration
            Raised at the EOF.
        """

        row = self._read()
        if row is None:
            raise StopIteration
        row = self._cast(row)

        return row

    def _read(self):
        """Must be implemented by subclasses that support 'r' subclasses.
        Should increment ``.pos`` and redefine this doc string.

        Raises
        ------
        NotImplementedError
        """

        self._complain_ifclosed(self.closed)
        raise NotImplementedError

    def truncate(self, size=None):
        """Should be implemented by subclasses and redefine this doc string.

        Raises
        ------
        NotImplementedError
        """

        self._complain_ifclosed(self.closed)
        raise NotImplementedError

    def write(self, obj):
        """Must be implemented by subclasses that support 'w' subclasses
        Should increment ``.pos``. Subclasses should also check if ``obj``
        is an instance of type(list) and redefine this doc string.

        Raises
        ------
        NotImplementedError
        """

        self._complain_ifclosed(self.closed)
        "Write obj to dataObj"
        raise NotImplementedError

    def flush(self):
        """

        Raises
        ------
        NotImplementedError
        """

        self._complain_ifclosed(self.closed)
        raise NotImplementedError

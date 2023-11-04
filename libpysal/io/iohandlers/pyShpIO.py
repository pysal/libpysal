"""
PySAL ShapeFile Reader and Writer based on pure python shapefile module.

"""

# ruff: noqa: N802, N806, N999

__author__ = "Charles R Schmidt <schmidtc@gmail.com>"
__credits__ = "Copyright (c) 2009 Charles R. Schmidt"
__all__ = ["PurePyShpWrapper"]

from warnings import warn

from ... import cg
from .. import fileio
from ..util import shp_file

STRING_TO_TYPE = {
    "POLYGON": cg.Polygon,
    "POINT": cg.Point,
    "POINTM": cg.Point,
    "POINTZ": cg.Point,
    "ARC": cg.Chain,
    "POLYGONZ": cg.Polygon,
}

# build the reverse map
# for key,value in STRING_TO_TYPE.items():
#    TYPE_TO_STRING[value] = key

TYPE_TO_STRING = {
    cg.Polygon: "POLYGON",
    cg.Point: "POINT",
    cg.Chain: "ARC",
}


class PurePyShpWrapper(fileio.FileIO):
    """FileIO handler for ESRI ShapeFiles.

    Notes
    -----

    This class wraps ``_pyShpIO``'s ``shp_file`` class with the PySAL `FileIO` API.
    shp_file can be used without PySAL.

    Examples
    --------

    >>> import tempfile
    >>> f = tempfile.NamedTemporaryFile(suffix='.shp')
    >>> fname = f.name
    >>> f.close()

    >>> import libpysal
    >>> i = libpysal.io.open(libpysal.examples.get_path('10740.shp'),'r')
    >>> o = libpysal.io.open(fname,'w')

    >>> for shp in i:
    ...     o.write(shp)
    >>> o.close()

    >>> one = libpysal.io.open(libpysal.examples.get_path('10740.shp'),'rb').read()
    >>> two = libpysal.io.open(fname,'rb').read()
    >>> one[0].centroid == two[0].centroid
    True

    >>> one = libpysal.io.open(libpysal.examples.get_path('10740.shx'),'rb').read()
    >>> two = libpysal.io.open(fname[:-1]+'x','rb').read()
    >>> one[0].centroid == two[0].centroid
    True

    >>> import os
    >>> os.remove(fname); os.remove(fname.replace('.shp','.shx'))

    """

    FORMATS = ["shp", "shx"]
    MODES = ["w", "r", "wb", "rb"]

    def __init__(self, *args, **kwargs):
        fileio.FileIO.__init__(self, *args, **kwargs)
        self.dataObj = None
        if self.mode == "r" or self.mode == "rb":
            self.__open()
        elif self.mode == "w" or self.mode == "wb":
            self.__create()

    def __len__(self) -> int:
        if self.dataObj is not None:
            return len(self.dataObj)
        else:
            return 0

    def __open(self):
        """

        Raises
        ------
        TypeError
            Raised when an invalid shape is passed in.

        """

        self.dataObj = shp_file(self.dataPath)
        self.header = self.dataObj.header
        self.bbox = self.dataObj.bbox

        try:
            self.type = STRING_TO_TYPE[self.dataObj.type()]
        except KeyError:
            msg = "%s does not support shapes of type: %s."
            msg = msg % (self.__class__.__name__, self.dataObj.type())
            raise TypeError(msg) from None

    def __create(self):
        self.write = self.__firstWrite

    def __firstWrite(self, shape):
        """

        Parameters
        ----------
        shape : libpysal.cg.{Point, Chain, Polygon}
            Geometric shape.

        """

        self.type = TYPE_TO_STRING[type(shape)]
        if self.type == "POINT":
            if len(shape) == 3:
                self.type = "POINTM"
            if len(shape) == 4:
                self.type = "POINTZ"
        self.dataObj = shp_file(self.dataPath, "w", self.type)
        self.write = self.__writer
        self.write(shape)

    def __writer(self, shape):
        """

        Parameters
        ----------
        shape : libpysal.cg.{Point, Chain, Polygon}
            Geometric shape.

        Raises
        ------
        TypeError
            Raised when an invalid shape is passed in.

        """

        if TYPE_TO_STRING[type(shape)] != self.type:
            raise TypeError("This file only supports %s type shapes." % self.type)

        rec = {}
        rec["Shape Type"] = shp_file.SHAPE_TYPES[self.type]

        if self.type == "POINT":
            rec["X"] = shape[0]
            rec["Y"] = shape[1]
            if len(shape) > 2:
                rec["M"] = shape[2]
            if len(shape) > 3:
                rec["Z"] = shape[3]
            shape = rec
        else:
            rec["BBOX Xmin"] = shape.bounding_box.left
            rec["BBOX Ymin"] = shape.bounding_box.lower
            rec["BBOX Xmax"] = shape.bounding_box.right
            rec["BBOX Ymax"] = shape.bounding_box.upper
            if self.type == "POLYGON":
                holes = [hole[::-1] for hole in shape.holes if hole]
                # holes should be in CCW order
                rec["NumParts"] = len(shape.parts) + len(holes)
                all_parts = shape.parts + holes
            else:
                rec["NumParts"] = len(shape.parts)
                all_parts = shape.parts
            partsIndex = [0]
            for l_ in [len(part) for part in all_parts][:-1]:
                partsIndex.append(partsIndex[-1] + l_)
            rec["Parts Index"] = partsIndex
            verts = sum(all_parts, [])
            verts = list(verts)
            rec["NumPoints"] = len(verts)
            rec["Vertices"] = verts
        self.dataObj.add_shape(rec)
        self.pos += 1

    def _read(self):
        """

        Returns
        -------
        shape : libpysal.cg.{Point, Chain, Polygon}
            Geometric shape.

        """
        try:
            rec = self.dataObj.get_shape(self.pos)
        except IndexError:
            return None

        self.pos += 1

        if self.dataObj.type() == "POINT":
            shp = self.type((rec["X"], rec["Y"]))
        elif self.dataObj.type() == "POINTZ":
            shp = self.type((rec["X"], rec["Y"]))
            shp.Z = rec["Z"]
            shp.M = rec["M"]
        else:
            if rec["NumParts"] > 1:
                partsIndex = list(rec["Parts Index"])
                partsIndex.append(None)
                parts = [
                    rec["Vertices"][partsIndex[i] : partsIndex[i + 1]]
                    for i in range(rec["NumParts"])
                ]
                if self.dataObj.type() == "POLYGON":
                    is_cw = [cg.is_clockwise(part) for part in parts]
                    vertices = [
                        part for part, cw in zip(parts, is_cw, strict=True) if cw
                    ]
                    holes = [
                        part for part, cw in zip(parts, is_cw, strict=True) if not cw
                    ]
                    if not holes:
                        holes = None
                    shp = self.type(vertices, holes)
                else:
                    vertices = parts
                    shp = self.type(vertices)
            elif rec["NumParts"] == 1:
                vertices = rec["Vertices"]
                if self.dataObj.type() == "POLYGON" and not cg.is_clockwise(vertices):
                    # SHAPEFILE WARNING:
                    # Polygon %d topology has been fixed. (ccw -> cw)
                    msg = "SHAPEFILE WARNING: Polygon %d "
                    msg += "topology has been fixed. (ccw -> cw)."
                    msg = msg % self.pos
                    warn(msg, RuntimeWarning, stacklevel=2)
                    print(msg)

                shp = self.type(vertices)
            else:
                warn(
                    "Polygon %d has zero parts." % self.pos,
                    RuntimeWarning,
                    stacklevel=2,
                )
                shp = self.type([[]])
                # raise ValueError, "Polygon %d has zero parts"%self.pos

        if self.ids:
            # shp IDs start at 1.
            shp.id = self.rIds[self.pos - 1]
        else:
            # shp IDs start at 1.
            shp.id = self.pos

        return shp

    def close(self):
        self.dataObj.close()
        fileio.FileIO.close(self)

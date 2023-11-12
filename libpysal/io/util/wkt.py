# ruff: noqa: N802, N815

import re

from ... import cg

__author__ = "Charles R Schmidt <schmidtc@gmail.com>"
__all__ = ["WKTParser"]


class WKTParser:
    """Class to represent `OGC WKT`, supports reading and writing.
    Modified from...
    # URL: http://dev.openlayers.org/releases/OpenLayers-2.7/lib/OpenLayers/Format/WKT.js
    # Reg Ex Strings copied from OpenLayers.Format.WKT

    Examples
    --------
    >>> import libpysal

    Create some Well-Known Text objects.

    >>> p = 'POLYGON((1 1, 5 1, 5 5, 1 5, 1 1), (2 2, 3 2, 3 3, 2 3, 2 2))'
    >>> pt = 'POINT(6 10)'
    >>> l = 'LINESTRING(3 4, 10 50, 20 25)'

    Instantiate the parser.

    >>> parser = libpysal.io.wkt.WKTParser()

    Inspect our WKT polygon.

    >>> parser(p).parts
    [[(1.0, 1.0), (1.0, 5.0), (5.0, 5.0), (5.0, 1.0), (1.0, 1.0)],
     [(2.0, 2.0), (2.0, 3.0), (3.0, 3.0), (3.0, 2.0), (2.0, 2.0)]]

    >>> parser(p).centroid
    (2.9705882352941178, 2.9705882352941178)

    >>> parser(p).area
    17.0

    Inspect ``pt``, our WKT point object.

    >>> parser(pt)
    (6.0, 10.0)

    Inspect our WKT linestring.

    >>> parser(l).len
    73.45538453219989

    >>> parser(l).parts
    [[(3.0, 4.0), (10.0, 50.0), (20.0, 25.0)]]

    Read in WKT from a file.

    >>> f = libpysal.io.open(libpysal.examples.get_path('stl_hom.wkt'))
    >>> f.mode
    'r'

    >>> f.header
    []

    See local doctest output for the items not tested.
    """

    regExes = {
        "typeStr": re.compile(r"^\s*([\w\s]+)\s*\(\s*(.*)\s*\)\s*$"),
        "spaces": re.compile(r"\s+"),
        "parenComma": re.compile(r"\)\s*,\s*\("),
        "doubleParenComma": re.compile(r"\)\s*\)\s*,\s*\(\s*\("),  # can't use {2} here
        "trimParens": re.compile(r"^\s*\(?(.*?)\)?\s*$"),
    }

    def __init__(self):
        self.parsers = p = {}
        p["point"] = self.Point
        p["linestring"] = self.LineString
        p["polygon"] = self.Polygon

    def Point(self, geo_str):
        """Returns a ``libpysal.cg.Point`` object."""
        coords = self.regExes["spaces"].split(geo_str.strip())
        return cg.Point((coords[0], coords[1]))

    def LineString(self, geo_str):
        """Returns a ``libpysal.cg.Chain`` object."""
        points = geo_str.strip().split(",")
        points = list(map(self.Point, points))
        return cg.Chain(points)

    def Polygon(self, geo_str):
        """Returns a ``libpysal.cg.Polygon`` object."""
        rings = self.regExes["parenComma"].split(geo_str.strip())
        for i, ring in enumerate(rings):
            ring = self.regExes["trimParens"].match(ring).groups()[0]
            ring = self.LineString(ring).vertices
            rings[i] = ring
        return cg.Polygon(rings)

    def fromWKT(self, wkt):
        """Returns geometric representation from WKT or ``None``.

        Raises
        ------
        NotImplementedError
            Raised when a unknown/unsupported format is passed in.
        """

        matches = self.regExes["typeStr"].match(wkt)
        if matches:
            geo_type, geo_str = matches.groups()
            geo_type = geo_type.lower().strip()
            try:
                return self.parsers[geo_type](geo_str)
            except KeyError:
                raise NotImplementedError(
                    f"Unsupported WKT Type: {geo_type}."
                ) from None
        else:
            return None

    __call__ = fromWKT


# if __name__ == "__main__":
#
#     p = "POLYGON((1 1,5 1,5 5,1 5,1 1),(2 2, 3 2, 3 3, 2 3,2 2))"
#     pt = "POINT(6 10)"
#     l_ = "LINESTRING(3 4,10 50,20 25)"
#     wktExamples = [
#         "POINT(6 10)",
#         "LINESTRING(3 4,10 50,20 25)",
#         "POLYGON((1 1,5 1,5 5,1 5,1 1),(2 2, 3 2, 3 3, 2 3,2 2))",
#         "MULTIPOINT(3.5 5.6,4.8 10.5)",
#         "MULTILINESTRING((3 4,10 50,20 25),(-5 -8,-10 -8,-15 -4))",
#         (
#             "MULTIPOLYGON(((1 1,5 1,5 5,1 5,1 1),"
#             "(2 2, 3 2, 3 3, 2 3,2 2)),((3 3,6 2,6 4,3 3)))"
#         ),
#         "GEOMETRYCOLLECTION(POINT(4 6),LINESTRING(4 6,7 10))",
#         "POINT ZM (1 1 5 60)",
#         "POINT M (1 1 80)",
#         "POINT EMPTY",
#         "MULTIPOLYGON EMPTY",
#     ]
#     wkt = WKTParser()

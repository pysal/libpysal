import pytest

from ....cg.shapes import Chain, Point, Polygon
from ..wkt import WKTParser


class TesttestWKTParser:
    def setup_method(self):
        # Create some Well-Known Text objects
        self.wktPOINT = "POINT(6 10)"
        self.wktLINESTRING = "LINESTRING(3 4,10 50,20 25)"
        self.wktPOLYGON = "POLYGON((1 1,5 1,5 5,1 5,1 1),(2 2, 3 2, 3 3, 2 3,2 2))"
        self.unsupported = [
            "MULTIPOINT(3.5 5.6,4.8 10.5)",
            "MULTILINESTRING((3 4,10 50,20 25),(-5 -8,-10 -8,-15 -4))",
            (
                "MULTIPOLYGON(((1 1,5 1,5 5,1 5,1 1),"
                "(2 2, 3 2, 3 3, 2 3,2 2)),((3 3,6 2,6 4,3 3)))"
            ),
            "GEOMETRYCOLLECTION(POINT(4 6),LINESTRING(4 6,7 10))",
            "POINT ZM (1 1 5 60)",
            "POINT M (1 1 80)",
        ]
        self.empty = ["POINT EMPTY", "MULTIPOLYGON EMPTY"]
        self.parser = WKTParser()

    def test_point(self):
        pt = self.parser(self.wktPOINT)
        assert issubclass(type(pt), Point)
        assert pt[:] == (6.0, 10.0)

    def test_line_string(self):
        line = self.parser(self.wktLINESTRING)
        assert issubclass(type(line), Chain)
        parts = [[pt[:] for pt in part] for part in line.parts]
        assert parts == [[(3.0, 4.0), (10.0, 50.0), (20.0, 25.0)]]
        assert line.len == 73.455384532199886

    def test_polygon(self):
        poly = self.parser(self.wktPOLYGON)
        assert issubclass(type(poly), Polygon)
        parts = [[pt[:] for pt in part] for part in poly.parts]
        assert parts == [
            [(1.0, 1.0), (1.0, 5.0), (5.0, 5.0), (5.0, 1.0), (1.0, 1.0)],
            [(2.0, 2.0), (2.0, 3.0), (3.0, 3.0), (3.0, 2.0), (2.0, 2.0)],
        ]
        assert poly.centroid == (2.9705882352941178, 2.9705882352941178)
        assert poly.area == 17.0

    def test_from_wkt(self):
        for wkt in self.unsupported:
            pytest.raises(NotImplementedError, self.parser.fromWKT, wkt)
        for wkt in self.empty:
            assert self.parser.fromWKT(wkt) is None
        assert self.parser.__call__ == self.parser.fromWKT

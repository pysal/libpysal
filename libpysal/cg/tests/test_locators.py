"""locators Unittest."""
from ..locators import *

# ruff: noqa: F403, F405
from ..shapes import *


class TestPolygonLocator:
    def setup_method(self):
        p1 = Polygon([Point((0, 1)), Point((4, 5)), Point((5, 1))])
        p2 = Polygon([Point((3, 9)), Point((6, 7)), Point((1, 1))])
        p3 = Polygon([Point((7, 1)), Point((8, 7)), Point((9, 1))])
        self.polygons = [p1, p2, p3]
        self.pl = PolygonLocator(self.polygons)

        pt = Point
        pg = Polygon
        polys = []
        for i in range(5):
            l_ = i * 10
            r = l_ + 10
            b = 10
            t = 20
            sw = pt((l_, b))
            se = pt((r, b))
            ne = pt((r, t))
            nw = pt((l_, t))
            polys.append(pg([sw, se, ne, nw]))
        self.pl2 = PolygonLocator(polys)

    def test_polygon_locator(self):
        qr = Rectangle(3, 7, 5, 8)
        res = self.pl.inside(qr)
        assert len(res) == 0

    def test_inside(self):
        qr = Rectangle(3, 3, 5, 5)
        res = self.pl.inside(qr)
        assert len(res) == 0
        qr = Rectangle(0, 0, 5, 5)
        res = self.pl.inside(qr)
        assert len(res) == 1

    def test_overlapping(self):
        qr = Rectangle(3, 3, 5, 5)
        res = self.pl.overlapping(qr)
        assert len(res) == 2
        qr = Rectangle(8, 3, 10, 10)
        res = self.pl.overlapping(qr)
        assert len(res) == 1

        qr = Rectangle(2, 12, 35, 15)
        res = self.pl2.overlapping(qr)
        assert len(res) == 4

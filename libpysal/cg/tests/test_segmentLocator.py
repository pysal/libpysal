"""Segment Locator Unittest."""

# ruff: noqa: F403, F405, N999

from ..segmentLocator import *
from ..shapes import *


class TestSegmentGrid:
    def setup_method(self):
        # 10x10 grid with four line segments, one for each edge of the grid.
        self.grid = SegmentGrid(Rectangle(0, 0, 10, 10), 1)
        self.grid.add(LineSegment(Point((0.0, 0.0)), Point((0.0, 10.0))), 0)
        self.grid.add(LineSegment(Point((0.0, 10.0)), Point((10.0, 10.0))), 1)
        self.grid.add(LineSegment(Point((10.0, 10.0)), Point((10.0, 0.0))), 2)
        self.grid.add(LineSegment(Point((10.0, 0.0)), Point((0.0, 0.0))), 3)

    def test_nearest_1(self):
        # Center
        assert self.grid.nearest(Point((5.0, 5.0))) == [0, 1, 2, 3]
        # Left Edge
        assert self.grid.nearest(Point((0.0, 5.0))) == [0]
        # Top Edge
        assert self.grid.nearest(Point((5.0, 10.0))) == [1]
        # Right Edge
        assert self.grid.nearest(Point((10.0, 5.0))) == [2]
        # Bottom Edge
        assert self.grid.nearest(Point((5.0, 0.0))) == [3]

    def test_nearest_2(self):
        # Left Edge
        assert self.grid.nearest(Point((-100000.0, 5.0))) == [0, 1, 3]
        # Right Edge
        assert self.grid.nearest(Point((100000.0, 5.0))) == [1, 2, 3]
        # Bottom Edge
        assert self.grid.nearest(Point((5.0, -100000.0))) == [0, 2, 3]
        # Top Edge
        assert self.grid.nearest(Point((5.0, 100000.0))) == [0, 1, 2]

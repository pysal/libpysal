"""Segment Locator Unittest."""

# ruff: noqa: F403, F405

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
        assert [0, 1, 2, 3] == self.grid.nearest(Point((5.0, 5.0)))
        # Left Edge
        assert [0] == self.grid.nearest(Point((0.0, 5.0)))
        # Top Edge
        assert [1] == self.grid.nearest(Point((5.0, 10.0)))
        # Right Edge
        assert [2] == self.grid.nearest(Point((10.0, 5.0)))
        # Bottom Edge
        assert [3] == self.grid.nearest(Point((5.0, 0.0)))

    def test_nearest_2(self):
        # Left Edge
        assert [0, 1, 3] == self.grid.nearest(Point((-100000.0, 5.0)))
        # Right Edge
        assert [1, 2, 3] == self.grid.nearest(Point((100000.0, 5.0)))
        # Bottom Edge
        assert [0, 2, 3] == self.grid.nearest(Point((5.0, -100000.0)))
        # Top Edge
        assert [0, 1, 2] == self.grid.nearest(Point((5.0, 100000.0)))

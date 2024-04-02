import pytest

from ..shapes import Chain, Line, LineSegment, Point, Polygon, Ray, Rectangle


class TesttestPoint:
    def test___init__1(self):
        """Tests whether points are created without issue."""

        for l_ in [(-5.0, 10.0), (0.0, -6.0), (1e300, -1e300)]:
            Point(l_)

    def test___str__1(self):
        """Tests whether the string produced is valid for corner cases."""

        for l_ in [(-5, 10), (0, -6.0), (1e300, -1e300)]:
            p = Point(l_)
            # Recast to floats like point does
            assert str(p) == str((float(l_[0]), float(l_[1])))


class TesttestLineSegment:
    def test_is_ccw1(self):
        """Test corner cases for horizontal segment starting at origin."""

        ls = LineSegment(Point((0, 0)), Point((5, 0)))

        # At positive boundary beyond segment
        assert not ls.is_ccw(Point((10, 0)))
        # On segment
        assert not ls.is_ccw(Point((3, 0)))
        # At negative boundary beyond segment
        assert not ls.is_ccw(Point((-10, 0)))
        # Endpoint of segment
        assert not ls.is_ccw(Point((0, 0)))
        # Endpoint of segment
        assert not ls.is_ccw(Point((5, 0)))

    def test_is_ccw2(self):
        """Test corner cases for vertical segment ending at origin."""

        ls = LineSegment(Point((0, -5)), Point((0, 0)))

        # At positive boundary beyond segment
        assert not ls.is_ccw(Point((0, 10)))
        # On segment
        assert not ls.is_ccw(Point((0, -3)))
        # At negative boundary beyond segment
        assert not ls.is_ccw(Point((0, -10)))
        # Endpoint of segment
        assert not ls.is_ccw(Point((0, -5)))
        # Endpoint of segment
        assert not ls.is_ccw(Point((0, 0)))

    def test_is_ccw3(self):
        """Test corner cases for non-axis-aligned segment not through origin."""

        ls = LineSegment(Point((0, 1)), Point((5, 6)))

        # At positive boundary beyond segment
        assert not ls.is_ccw(Point((10, 11)))
        # On segment
        assert not ls.is_ccw(Point((3, 4)))
        # At negative boundary beyond segment
        assert not ls.is_ccw(Point((-10, -9)))
        # Endpoint of segment
        assert not ls.is_ccw(Point((0, 1)))
        # Endpoint of segment
        assert not ls.is_ccw(Point((5, 6)))

    def test_is_cw1(self):
        """Test corner cases for horizontal segment starting at origin."""

        ls = LineSegment(Point((0, 0)), Point((5, 0)))

        # At positive boundary beyond segment
        assert not ls.is_cw(Point((10, 0)))
        # On segment
        assert not ls.is_cw(Point((3, 0)))
        # At negative boundary beyond segment
        assert not ls.is_cw(Point((-10, 0)))
        # Endpoint of segment
        assert not ls.is_cw(Point((0, 0)))
        # Endpoint of segment
        assert not ls.is_cw(Point((5, 0)))

    def test_is_cw2(self):
        """Test corner cases for vertical segment ending at origin."""

        ls = LineSegment(Point((0, -5)), Point((0, 0)))

        # At positive boundary beyond segment
        assert not ls.is_cw(Point((0, 10)))
        # On segment
        assert not ls.is_cw(Point((0, -3)))
        # At negative boundary beyond segment
        assert not ls.is_cw(Point((0, -10)))
        # Endpoint of segment
        assert not ls.is_cw(Point((0, -5)))
        # Endpoint of segment
        assert not ls.is_cw(Point((0, 0)))

    def test_is_cw3(self):
        """Test corner cases for non-axis-aligned segment not through origin."""

        ls = LineSegment(Point((0, 1)), Point((5, 6)))

        # At positive boundary beyond segment
        assert not ls.is_cw(Point((10, 11)))
        # On segment
        assert not ls.is_cw(Point((3, 4)))
        # At negative boundary beyond segment
        assert not ls.is_cw(Point((-10, -9)))
        # Endpoint of segment
        assert not ls.is_cw(Point((0, 1)))
        # Endpoint of segment
        assert not ls.is_cw(Point((5, 6)))

    def test_get_swap1(self):
        """Tests corner cases."""

        ls = LineSegment(Point((0, 0)), Point((10, 0)))
        swap = ls.get_swap()
        assert ls.p1 == swap.p2
        assert ls.p2 == swap.p1

        ls = LineSegment(Point((-5, 0)), Point((5, 0)))
        swap = ls.get_swap()
        assert ls.p1 == swap.p2
        assert ls.p2 == swap.p1

        ls = LineSegment(Point((0, 0)), Point((0, 0)))
        swap = ls.get_swap()
        assert ls.p1 == swap.p2
        assert ls.p2 == swap.p1

        ls = LineSegment(Point((5, 5)), Point((5, 5)))
        swap = ls.get_swap()
        assert ls.p1 == swap.p2
        assert ls.p2 == swap.p1

    def test_bounding_box(self):
        """Tests corner cases."""

        ls = LineSegment(Point((0, 0)), Point((0, 10)))
        assert ls.bounding_box.left == 0
        assert ls.bounding_box.lower == 0
        assert ls.bounding_box.right == 0
        assert ls.bounding_box.upper == 10

        ls = LineSegment(Point((0, 0)), Point((-3, -4)))
        assert ls.bounding_box.left == -3
        assert ls.bounding_box.lower == -4
        assert ls.bounding_box.right == 0
        assert ls.bounding_box.upper == 0

        ls = LineSegment(Point((-5, 0)), Point((3, 0)))
        assert ls.bounding_box.left == -5
        assert ls.bounding_box.lower == 0
        assert ls.bounding_box.right == 3
        assert ls.bounding_box.upper == 0

    def test_len1(self):
        """Tests corner cases."""

        ls = LineSegment(Point((0, 0)), Point((0, 0)))
        assert ls.len == 0

        ls = LineSegment(Point((0, 0)), Point((-3, 0)))
        assert ls.len == 3

    def test_line1(self):
        """Tests corner cases."""

        import math

        ls = LineSegment(Point((0, 0)), Point((1, 0)))
        assert ls.line.m == 0
        assert ls.line.b == 0

        ls = LineSegment(Point((0, 0)), Point((0, 1)))
        assert ls.line.m == float("inf")
        assert math.isnan(ls.line.b)

        ls = LineSegment(Point((0, 0)), Point((0, -1)))
        assert ls.line.m == float("inf")
        assert math.isnan(ls.line.b)

        ls = LineSegment(Point((0, 0)), Point((0, 0)))
        assert ls.line is None

        ls = LineSegment(Point((5, 0)), Point((10, 0)))
        ls1 = LineSegment(Point((5, 0)), Point((10, 1)))
        assert ls.intersect(ls1)
        ls2 = LineSegment(Point((5, 1)), Point((10, 1)))
        assert not ls.intersect(ls2)
        ls2 = LineSegment(Point((7, -1)), Point((7, 2)))
        assert ls.intersect(ls2)


class TesttestLine:
    def test___init__1(self):
        """Tests a variety of generic cases."""

        for m, b in [(4, 0.0), (-140, 5), (0, 0)]:
            _ = Line(m, b)

    def test_y1(self):
        """Tests a variety of generic and special cases (+-infinity)."""

        l_ = Line(0, 0)
        assert l_.y(0) == 0
        assert l_.y(-1e600) == 0
        assert l_.y(1e600) == 0

        l_ = Line(1, 1)
        assert l_.y(2) == 3
        assert l_.y(-1e600) == -1e600
        assert l_.y(1e600) == 1e600

        l_ = Line(-1, 1)
        assert l_.y(2) == -1
        assert l_.y(-1e600) == 1e600
        assert l_.y(1e600) == -1e600

    def test_x1(self):
        """Tests a variety of generic and special cases (+-infinity)."""

        l_ = Line(0, 0)

        # self.assertEquals(l.x(0), 0)
        with pytest.raises(ArithmeticError):
            l_.x(0)
        with pytest.raises(ArithmeticError):
            l_.x(-1e600)
        with pytest.raises(ArithmeticError):
            l_.x(1e600)

        l_ = Line(1, 1)
        assert l_.x(3) == 2
        assert l_.x(-1e600) == -1e600
        assert l_.x(1e600) == 1e600

        l_ = Line(-1, 1)
        assert l_.x(2) == -1
        assert l_.x(-1e600) == 1e600
        assert l_.x(1e600) == -1e600


class TesttestRay:
    def test___init__1(self):
        """Tests generic cases."""

        _ = Ray(Point((0, 0)), Point((1, 1)))
        _ = Ray(Point((8, -3)), Point((-5, 9)))


class TesttestChain:
    def test___init__1(self):
        """Generic testing that no exception is thrown."""

        _ = Chain([Point((0, 0))])
        _ = Chain([[Point((0, 0)), Point((1, 1))], [Point((2, 5))]])

    def test_vertices1(self):
        """Testing for repeated vertices and multiple parts."""

        vertices = [
            Point((0, 0)),
            Point((1, 1)),
            Point((2, 5)),
            Point((0, 0)),
            Point((1, 1)),
            Point((2, 5)),
        ]
        assert Chain(vertices).vertices == vertices

        vertices = [
            [Point((0, 0)), Point((1, 1)), Point((2, 5))],
            [Point((0, 0)), Point((1, 1)), Point((2, 5))],
        ]
        assert Chain(vertices).vertices == vertices[0] + vertices[1]

    def test_parts1(self):
        """Generic testing of parts functionality."""

        vertices = [
            Point((0, 0)),
            Point((1, 1)),
            Point((2, 5)),
            Point((0, 0)),
            Point((1, 1)),
            Point((2, 5)),
        ]
        assert Chain(vertices).parts == [vertices]

        vertices = [
            [Point((0, 0)), Point((1, 1)), Point((2, 5))],
            [Point((0, 0)), Point((1, 1)), Point((2, 5))],
        ]
        assert Chain(vertices).parts == vertices

    def test_bounding_box1(self):
        """Test correctness with multiple parts."""

        vertices = [
            [Point((0, 0)), Point((1, 1)), Point((2, 6))],
            [Point((-5, -5)), Point((0, 0)), Point((2, 5))],
        ]
        bb = Chain(vertices).bounding_box
        assert bb.left == -5
        assert bb.lower == -5
        assert bb.right == 2
        assert bb.upper == 6

    def test_len1(self):
        """Test correctness with multiple parts and
        zero-length point-to-point distances.

        """

        vertices = [
            [Point((0, 0)), Point((1, 0)), Point((1, 5))],
            [Point((-5, -5)), Point((-5, 0)), Point((0, 0)), Point((0, 0))],
        ]
        assert Chain(vertices).len == 6 + 10


class TesttestPolygon:
    def test___init__1(self):
        """Test various input configurations (list vs. lists of lists, holes)."""

        # Input configurations tested (in order of test):
        # one part, no holes
        # multi parts, no holes
        # one part, one hole
        # multi part, one hole
        # one part, multi holes
        # multi part, multi holes

        _ = Polygon([Point((0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))])
        _ = Polygon(
            [
                [Point((0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))],
                [Point((30, 30)), Point((40, 30)), Point((40, 40)), Point((30, 40))],
            ]
        )
        _ = Polygon(
            [Point((0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))],
            holes=[Point((2, 2)), Point((4, 2)), Point((4, 4)), Point((2, 4))],
        )
        _ = Polygon(
            [
                [Point((0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))],
                [Point((30, 30)), Point((40, 30)), Point((40, 40)), Point((30, 40))],
            ],
            holes=[Point((2, 2)), Point((4, 2)), Point((4, 4)), Point((2, 4))],
        )
        _ = Polygon(
            [Point((0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))],
            holes=[
                [Point((2, 2)), Point((4, 2)), Point((4, 4)), Point((2, 4))],
                [Point((6, 6)), Point((6, 8)), Point((8, 8)), Point((8, 6))],
            ],
        )
        _ = Polygon(
            [
                [Point((0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))],
                [Point((30, 30)), Point((40, 30)), Point((40, 40)), Point((30, 40))],
            ],
            holes=[
                [Point((2, 2)), Point((4, 2)), Point((4, 4)), Point((2, 4))],
                [Point((6, 6)), Point((6, 8)), Point((8, 8)), Point((8, 6))],
            ],
        )

    def test_area1(self):
        """Test multiple parts."""

        p = Polygon(
            [
                [Point((0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))],
                [Point((30, 30)), Point((40, 30)), Point((40, 40)), Point((30, 40))],
            ]
        )
        assert p.area == 200

    def test_area2(self):
        """Test holes."""

        p = Polygon(
            [Point((0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))],
            holes=[Point((2, 2)), Point((4, 2)), Point((4, 4)), Point((2, 4))],
        )
        assert p.area == 100 - 4

        p = Polygon(
            [Point((0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))],
            holes=[
                [Point((2, 2)), Point((4, 2)), Point((4, 4)), Point((2, 4))],
                [Point((6, 6)), Point((6, 8)), Point((8, 8)), Point((8, 6))],
            ],
        )
        assert p.area == 100 - (4 + 4)

        p = Polygon(
            [
                [Point((0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))],
                [Point((30, 30)), Point((40, 30)), Point((40, 40)), Point((30, 40))],
            ],
            holes=[
                [Point((2, 2)), Point((4, 2)), Point((4, 4)), Point((2, 4))],
                [Point((36, 36)), Point((36, 38)), Point((38, 38)), Point((38, 36))],
            ],
        )
        assert p.area == 200 - (4 + 4)

    def test_area4(self):
        """Test polygons with vertices in both orders (cw, ccw)."""

        p = Polygon([Point((0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))])
        assert p.area == 100

        p = Polygon([Point((0, 0)), Point((0, 10)), Point((10, 10)), Point((10, 0))])
        assert p.area == 100

    def test_bounding_box1(self):
        """Test polygons with multiple parts."""

        p = Polygon(
            [
                [Point((0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))],
                [Point((30, 30)), Point((40, 30)), Point((40, 40)), Point((30, 40))],
            ]
        )
        bb = p.bounding_box
        assert bb.left == 0
        assert bb.lower == 0
        assert bb.right == 40
        assert bb.upper == 40

    def test_centroid1(self):
        """Test polygons with multiple parts of the same size."""

        p = Polygon(
            [
                [Point((0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))],
                [Point((30, 30)), Point((40, 30)), Point((40, 40)), Point((30, 40))],
            ]
        )
        c = p.centroid
        assert c[0] == 20
        assert c[1] == 20

    def test_centroid2(self):
        """Test polygons with multiple parts of different size."""

        p = Polygon(
            [
                [Point((0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))],
                [Point((30, 30)), Point((35, 30)), Point((35, 35)), Point((30, 35))],
            ]
        )
        c = p.centroid
        assert c[0] == 10.5
        assert c[1] == 10.5

    def test_holes1(self):
        """Test for correct vertex values/order."""

        p = Polygon(
            [Point((0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))],
            holes=[Point((2, 2)), Point((4, 2)), Point((4, 4)), Point((2, 4))],
        )
        assert len(p.holes) == 1
        e_holes = [Point((2, 2)), Point((2, 4)), Point((4, 4)), Point((4, 2))]
        assert p.holes[0] in [
            e_holes,
            [e_holes[-1]] + e_holes[:3],
            e_holes[-2:] + e_holes[:2],
            e_holes[-3:] + [e_holes[0]],
        ]

    def test_holes2(self):
        """Test for multiple holes."""

        p = Polygon(
            [Point((0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))],
            holes=[
                [Point((2, 2)), Point((4, 2)), Point((4, 4)), Point((2, 4))],
                [Point((6, 6)), Point((6, 8)), Point((8, 8)), Point((8, 6))],
            ],
        )
        holes = p.holes
        assert len(holes) == 2

    def test_parts1(self):
        """Test for correct vertex values/order."""

        p = Polygon(
            [
                [Point((0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))],
                [Point((30, 30)), Point((40, 30)), Point((30, 40))],
            ]
        )
        assert len(p.parts) == 2

        part1 = [Point((0, 0)), Point((0, 10)), Point((10, 10)), Point((10, 0))]
        part2 = [Point((30, 30)), Point((30, 40)), Point((40, 30))]
        if len(p.parts[0]) == 4:
            assert p.parts[0] in [
                part1,
                part1[-1:] + part1[:3],
                part1[-2:] + part1[:2],
                part1[-3:] + part1[:1],
            ]
            assert p.parts[1] in [part2, part2[-1:] + part2[:2], part2[-2:] + part2[:1]]
        elif len(p.parts[0]) == 3:
            assert p.parts[0] in [part2, part2[-1:] + part2[:2], part2[-2:] + part2[:1]]
            assert p.parts[1] in [
                part1,
                part1[-1:] + part1[:3],
                part1[-2:] + part1[:2],
                part1[-3:] + part1[:1],
            ]
        else:
            pytest.fail()

    def test_perimeter1(self):
        """Test with multiple parts."""

        p = Polygon(
            [
                [Point((0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))],
                [Point((30, 30)), Point((40, 30)), Point((40, 40)), Point((30, 40))],
            ]
        )
        assert p.perimeter == 80

    def test_perimeter2(self):
        """Test with holes."""

        p = Polygon(
            [
                [Point((0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))],
                [Point((30, 30)), Point((40, 30)), Point((40, 40)), Point((30, 40))],
            ],
            holes=[
                [Point((2, 2)), Point((4, 2)), Point((4, 4)), Point((2, 4))],
                [Point((6, 6)), Point((6, 8)), Point((8, 8)), Point((8, 6))],
            ],
        )
        assert p.perimeter == 80 + 16

    def test_vertices1(self):
        """Test for correct values/order of vertices."""

        p = Polygon([Point((0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))])
        assert len(p.vertices) == 4
        e_verts = [Point((0, 0)), Point((0, 10)), Point((10, 10)), Point((10, 0))]
        assert p.vertices in [
            e_verts,
            e_verts[-1:] + e_verts[:3],
            e_verts[-2:] + e_verts[:2],
            e_verts[-3:] + e_verts[:1],
        ]

    def test_vertices2(self):
        """Test for multiple parts."""

        p = Polygon(
            [
                [Point((0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))],
                [Point((30, 30)), Point((40, 30)), Point((40, 40)), Point((30, 40))],
            ]
        )
        assert len(p.vertices) == 8

    def test_contains_point(self):
        p = Polygon(
            [Point((0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))],
            [Point((1, 2)), Point((2, 2)), Point((2, 1)), Point((1, 1))],
        )
        assert p.contains_point((0, 0)) == 0
        assert p.contains_point((1, 1)) == 0
        assert p.contains_point((5, 5)) == 1
        assert p.contains_point((10, 10)) == 0


class TesttestRectangle:
    def test___init__1(self):
        """Test exceptions are thrown correctly."""

        try:
            # right < left
            _ = Rectangle(1, 1, -1, 5)
        except ArithmeticError:
            pass
        else:
            pytest.fail()

        try:
            # upper < lower
            _ = Rectangle(1, 1, 5, -1)
        except ArithmeticError:
            pass
        else:
            pytest.fail()

    def test_set_centroid1(self):
        """Test with rectangles of zero width or height."""

        # Zero width
        r = Rectangle(5, 5, 5, 10)
        r.set_centroid(Point((0, 0)))
        assert r.left == 0
        assert r.lower == -2.5
        assert r.right == 0
        assert r.upper == 2.5

        # Zero height
        r = Rectangle(10, 5, 20, 5)
        r.set_centroid(Point((40, 40)))
        assert r.left == 35
        assert r.lower == 40
        assert r.right == 45
        assert r.upper == 40

        # Zero width and height
        r = Rectangle(0, 0, 0, 0)
        r.set_centroid(Point((-4, -4)))
        assert r.left == -4
        assert r.lower == -4
        assert r.right == -4
        assert r.upper == -4

    def test_set_scale1(self):
        """Test repeated scaling."""

        r = Rectangle(2, 2, 4, 4)

        r.set_scale(0.5)
        assert r.left == 2.5
        assert r.lower == 2.5
        assert r.right == 3.5
        assert r.upper == 3.5

        r.set_scale(2)
        assert r.left == 2
        assert r.lower == 2
        assert r.right == 4
        assert r.upper == 4

    def test_set_scale2(self):
        """Test scaling of rectangles with zero width/height."""

        # Zero width
        r = Rectangle(5, 5, 5, 10)
        r.set_scale(2)
        assert r.left == 5
        assert r.lower == 2.5
        assert r.right == 5
        assert r.upper == 12.5

        # Zero height
        r = Rectangle(10, 5, 20, 5)
        r.set_scale(2)
        assert r.left == 5
        assert r.lower == 5
        assert r.right == 25
        assert r.upper == 5

        # Zero width and height
        r = Rectangle(0, 0, 0, 0)
        r.set_scale(100)
        assert r.left == 0
        assert r.lower == 0
        assert r.right == 0
        assert r.upper == 0

        # Zero width and height
        r = Rectangle(0, 0, 0, 0)
        r.set_scale(0.01)
        assert r.left == 0
        assert r.lower == 0
        assert r.right == 0
        assert r.upper == 0

    def test_area1(self):
        """Test rectangles with zero width/height."""

        # Zero width
        r = Rectangle(5, 5, 5, 10)
        assert r.area == 0

        # Zero height
        r = Rectangle(10, 5, 20, 5)
        assert r.area == 0

        # Zero width and height
        r = Rectangle(0, 0, 0, 0)
        assert r.area == 0

    def test_height1(self):
        """Test rectangles with zero height."""

        # Zero height
        r = Rectangle(10, 5, 20, 5)
        assert r.height == 0

    def test_width1(self):
        """Test rectangles with zero width."""

        # Zero width
        r = Rectangle(5, 5, 5, 10)
        assert r.width == 0

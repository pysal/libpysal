# ruff: noqa: F403, F405

import math

import numpy as np
import pytest

from ..shapes import *
from ..standalone import *


class TestBbcommon:
    def test_bbcommon(self):
        b0 = [0, 0, 10, 10]
        b1 = [5, 5, 15, 15]
        assert bbcommon(b0, b1) == 1

    def test_bbcommon_same(self):
        b0 = [0, 0, 10, 10]
        b1 = [0, 0, 10, 10]
        assert bbcommon(b0, b1) == 1

    def test_bbcommon_nested(self):
        b0 = [0, 0, 10, 10]
        b1 = [1, 1, 9, 9]
        assert bbcommon(b0, b1) == 1

    def test_bbcommon_top(self):
        b0 = [0, 0, 10, 10]
        b1 = [3, 5, 6, 15]
        assert bbcommon(b0, b1) == 1

    def test_bbcommon_shared_edge(self):
        b0 = [0, 0, 10, 10]
        b1 = [0, 10, 10, 20]
        assert bbcommon(b0, b1) == 1

    def test_bbcommon_shared_corner(self):
        b0 = [0, 0, 10, 10]
        b1 = [10, 10, 20, 20]
        assert bbcommon(b0, b1) == 1

    def test_bbcommon_floats(self):
        b0 = [0.0, 0.0, 0.1, 0.1]
        b1 = [0.05, 0.05, 0.15, 0.15]
        assert bbcommon(b0, b1) == 1


class TestGetBoundingBox:
    def test_get_bounding_box(self):
        items = [Point((-1, 5)), Rectangle(0, 6, 11, 12)]
        expected = [-1, 5, 11, 12]
        assert expected == get_bounding_box(items)[:]


class TestGetAngleBetween:
    def test_get_angle_between(self):
        ray1 = Ray(Point((0, 0)), Point((1, 0)))
        ray2 = Ray(Point((0, 0)), Point((1, 0)))
        assert get_angle_between(ray1, ray2) == 0.0

    def test_get_angle_between_expect45(self):
        ray1 = Ray(Point((0, 0)), Point((1, 0)))
        ray2 = Ray(Point((0, 0)), Point((1, 1)))
        assert math.degrees(get_angle_between(ray1, ray2)) == 45.0

    def test_get_angle_between_expect90(self):
        ray1 = Ray(Point((0, 0)), Point((1, 0)))
        ray2 = Ray(Point((0, 0)), Point((0, 1)))
        assert math.degrees(get_angle_between(ray1, ray2)) == 90.0


class TestIsCollinear:
    def test_is_collinear(self):
        assert is_collinear(Point((0, 0)), Point((1, 1)), Point((5, 5)))

    def test_is_collinear_expect_false(self):
        assert not is_collinear(Point((0, 0)), Point((1, 1)), Point((5, 0)))

    def test_is_collinear_along_x(self):
        assert is_collinear(Point((0, 0)), Point((1, 0)), Point((5, 0)))

    def test_is_collinear_along_y(self):
        assert is_collinear(Point((0, 0)), Point((0, 1)), Point((0, -1)))

    def test_is_collinear_small_float(self):
        """
        Given: p1 = (0.1, 0.2), p2 = (0.2, 0.3), p3 = (0.3, 0.4)

        Line(p1, p2):  y = mx + b
            m = (0.3-0.2) / (0.2-0.1) = .1/.1 = 1
            y - mx = b
            b = 0.3 - 1*0.2 = 0.1
            b = 0.2 - 1*0.1 = 0.1

            y = 1*x + 0.1

        Line(p2, p3): y = mx + b
            m = (0.4-0.3) / (0.3-0.2) = .1/.1 = 1
            y - mx = b
            b = 0.4 - 1*0.3 = 0.1
            b = 0.4 - 1*0.2 = 0.1

            y = 1*x + 0.1

        Line(p1, p2) == Line(p2 ,p3)

        Therefore ``p1, p2, p3`` are collinear.

        Due to floating point rounding areas the standard test,

            ((p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0])) == 0

        will fail. To get around this we use an epsilon. The ``numpy.finfo``
        function return an smallest epsilon for the given data types such that,

            (numpy.finfo(float).eps + 1.0) != 1.0

        Therefore if

            abs(
                (p2[0]-p1[0]) * (p3[1]-p1[1]) - (p2[1]-p1[1]) * (p3[0]-p1[0])
            ) < numpy.finfo(p1[0]).eps

        The points are collinear.

        """

        assert is_collinear(Point((0.1, 0.2)), Point((0.2, 0.3)), Point((0.3, 0.4)))

    def test_is_collinear_random(self):
        for i in range(10):
            a, b, c = np.random.random(3) * 10 ** (i)
            assert is_collinear(Point((a, a)), Point((b, b)), Point((c, c)))

    def test_is_collinear_random2(self):
        for _i in range(1000):
            a, b, c = np.random.random(3)
            assert is_collinear(Point((a, a)), Point((b, b)), Point((c, c)))


class TestGetSegmentsIntersect:
    def test_get_segments_intersect(self):
        seg1 = LineSegment(Point((0, 0)), Point((0, 10)))
        seg2 = LineSegment(Point((-5, 5)), Point((5, 5)))
        assert get_segments_intersect(seg1, seg2)[:] == (0.0, 5.0)

    def test_get_segments_intersect_shared_vert(self):
        seg1 = LineSegment(Point((0, 0)), Point((0, 10)))
        seg2 = LineSegment(Point((-5, 5)), Point((0, 10)))
        assert get_segments_intersect(seg1, seg2)[:] == (0.0, 10.0)

    def test_get_segments_intersect_floats(self):
        seg1 = LineSegment(Point((0, 0)), Point((0, 0.10)))
        seg2 = LineSegment(Point((-0.5, 0.05)), Point((0.5, 0.05)))
        assert get_segments_intersect(seg1, seg2)[:] == (0.0, 0.05)

    def test_get_segments_intersect_angles(self):
        seg1 = LineSegment(Point((0, 0)), Point((1, 1)))
        seg2 = LineSegment(Point((1, 0)), Point((0, 1)))
        assert get_segments_intersect(seg1, seg2)[:] == (0.5, 0.5)

    def test_get_segments_intersect_no_intersect(self):
        seg1 = LineSegment(Point((-5, 5)), Point((5, 5)))
        seg2 = LineSegment(Point((100, 100)), Point((100, 101)))
        assert None is get_segments_intersect(seg1, seg2)

    def test_get_segments_intersect_overlap(self):
        seg1 = LineSegment(Point((0.1, 0.1)), Point((0.6, 0.6)))
        seg2 = LineSegment(Point((0.3, 0.3)), Point((0.9, 0.9)))
        expected = LineSegment(Point((0.3, 0.3)), Point((0.6, 0.6)))
        assert expected == get_segments_intersect(seg1, seg2)

    def test_get_segments_intersect_same(self):
        seg1 = LineSegment(Point((-5, 5)), Point((5, 5)))
        assert seg1 == get_segments_intersect(seg1, seg1)

    def test_get_segments_intersect_nested(self):
        seg1 = LineSegment(Point((0.1, 0.1)), Point((0.9, 0.9)))
        seg2 = LineSegment(Point((0.3, 0.3)), Point((0.6, 0.6)))
        assert seg2 == get_segments_intersect(seg1, seg2)


class TestGetSegmentPointIntersect:
    def test_get_segment_point_intersect(self):
        seg = LineSegment(Point((0, 0)), Point((0, 10)))
        pt = Point((0, 5))
        assert pt == get_segment_point_intersect(seg, pt)

    def test_get_segment_point_intersect_left_end(self):
        seg = LineSegment(Point((0, 0)), Point((0, 10)))
        pt = seg.p1
        assert pt == get_segment_point_intersect(seg, pt)

    def test_get_segment_point_intersect_right_end(self):
        seg = LineSegment(Point((0, 0)), Point((0, 10)))
        pt = seg.p2
        assert pt == get_segment_point_intersect(seg, pt)

    def test_get_segment_point_intersect_angle(self):
        seg = LineSegment(Point((0, 0)), Point((1, 1)))
        pt = Point((0.1, 0.1))
        assert pt == get_segment_point_intersect(seg, pt)

    def test_get_segment_point_intersect_no_intersect(self):
        seg = LineSegment(Point((0, 0)), Point((0, 10)))
        pt = Point((5, 5))
        assert None is get_segment_point_intersect(seg, pt)

    def test_get_segment_point_intersect_no_intersect_collinear(self):
        seg = LineSegment(Point((0, 0)), Point((0, 10)))
        pt = Point((0, 20))
        assert None is get_segment_point_intersect(seg, pt)

    def test_get_segment_point_intersect_floats_1(self):
        seg = LineSegment(Point((0.3, 0.3)), Point((0.9, 0.9)))
        pt = Point((0.5, 0.5))
        assert pt == get_segment_point_intersect(seg, pt)

    def test_get_segment_point_intersect_floats_2(self):
        seg = LineSegment(
            Point((0.0, 0.0)), Point((2.7071067811865475, 2.7071067811865475))
        )
        pt = Point((1.0, 1.0))
        assert pt == get_segment_point_intersect(seg, pt)

    def test_get_segment_point_intersect_floats_no_intersect(self):
        seg = LineSegment(Point((0.3, 0.3)), Point((0.9, 0.9)))
        pt = Point((0.1, 0.1))
        assert None is get_segment_point_intersect(seg, pt)


class TestGetPolygonPointIntersect:
    def test_get_polygon_point_intersect(self):
        poly = Polygon([Point((0, 0)), Point((1, 0)), Point((1, 1)), Point((0, 1))])
        pt = Point((0.5, 0.5))
        assert pt == get_polygon_point_intersect(poly, pt)

    def test_get_polygon_point_intersect_on_edge(self):
        poly = Polygon([Point((0, 0)), Point((1, 0)), Point((1, 1)), Point((0, 1))])
        pt = Point((1.0, 0.5))
        assert pt == get_polygon_point_intersect(poly, pt)

    def test_get_polygon_point_intersect_on_vertex(self):
        poly = Polygon([Point((0, 0)), Point((1, 0)), Point((1, 1)), Point((0, 1))])
        pt = Point((1.0, 1.0))
        assert pt == get_polygon_point_intersect(poly, pt)

    def test_get_polygon_point_intersect_outside(self):
        poly = Polygon([Point((0, 0)), Point((1, 0)), Point((1, 1)), Point((0, 1))])
        pt = Point((2.0, 2.0))
        assert None is get_polygon_point_intersect(poly, pt)


class TestGetRectanglePointIntersect:
    def test_get_rectangle_point_intersect(self):
        rect = Rectangle(0, 0, 5, 5)
        pt = Point((1, 1))
        assert pt == get_rectangle_point_intersect(rect, pt)

    def test_get_rectangle_point_intersect_on_edge(self):
        rect = Rectangle(0, 0, 5, 5)
        pt = Point((2.5, 5))
        assert pt == get_rectangle_point_intersect(rect, pt)

    def test_get_rectangle_point_intersect_on_vertex(self):
        rect = Rectangle(0, 0, 5, 5)
        pt = Point((5, 5))
        assert pt == get_rectangle_point_intersect(rect, pt)

    def test_get_rectangle_point_intersect_outside(self):
        rect = Rectangle(0, 0, 5, 5)
        pt = Point((10, 10))
        assert None is get_rectangle_point_intersect(rect, pt)


class TestGetRaySegmentIntersect:
    def test_get_ray_segment_intersect(self):
        ray = Ray(Point((0, 0)), Point((0, 1)))
        seg = LineSegment(Point((-1, 10)), Point((1, 10)))
        assert get_ray_segment_intersect(ray, seg)[:] == (0.0, 10.0)

    def test_get_ray_segment_intersect_orgin(self):
        ray = Ray(Point((0, 0)), Point((0, 1)))
        seg = LineSegment(Point((-1, 0)), Point((1, 0)))
        assert get_ray_segment_intersect(ray, seg)[:] == (0.0, 0.0)

    def test_get_ray_segment_intersect_edge(self):
        ray = Ray(Point((0, 0)), Point((0, 1)))
        seg = LineSegment(Point((0, 2)), Point((2, 2)))
        assert get_ray_segment_intersect(ray, seg)[:] == (0.0, 2.0)

    def test_get_ray_segment_intersect_no_intersect(self):
        ray = Ray(Point((0, 0)), Point((0, 1)))
        seg = LineSegment(Point((10, 10)), Point((10, 11)))
        assert None is get_ray_segment_intersect(ray, seg)

    def test_get_ray_segment_intersect_segment(self):
        ray = Ray(Point((0, 0)), Point((5, 5)))
        seg = LineSegment(Point((1, 1)), Point((2, 2)))
        assert seg == get_ray_segment_intersect(ray, seg)


class TestGetRectangleRectangleIntersection:
    def test_get_rectangle_rectangle_intersection_leftright(self):
        r0 = Rectangle(0, 4, 6, 9)
        r1 = Rectangle(4, 0, 9, 7)
        expected = [4.0, 4.0, 6.0, 7.0]
        assert expected == get_rectangle_rectangle_intersection(r0, r1)[:]

    def test_get_rectangle_rectangle_intersection_topbottom(self):
        r0 = Rectangle(0, 0, 4, 4)
        r1 = Rectangle(2, 1, 6, 3)
        expected = [2.0, 1.0, 4.0, 3.0]
        assert expected == get_rectangle_rectangle_intersection(r0, r1)[:]

    def test_get_rectangle_rectangle_intersection_nested(self):
        r0 = Rectangle(0, 0, 4, 4)
        r1 = Rectangle(2, 1, 3, 2)
        assert r1 == get_rectangle_rectangle_intersection(r0, r1)

    def test_get_rectangle_rectangle_intersection_shared_corner(self):
        r0 = Rectangle(0, 0, 4, 4)
        r1 = Rectangle(4, 4, 8, 8)
        assert Point((4, 4)) == get_rectangle_rectangle_intersection(r0, r1)

    def test_get_rectangle_rectangle_intersection_shared_edge(self):
        r0 = Rectangle(0, 0, 4, 4)
        r1 = Rectangle(0, 4, 4, 8)
        assert LineSegment(
            Point((0, 4)), Point((4, 4))
        ) == get_rectangle_rectangle_intersection(r0, r1)

    def test_get_rectangle_rectangle_intersection_shifted_edge(self):
        r0 = Rectangle(0, 0, 4, 4)
        r1 = Rectangle(2, 4, 6, 8)
        assert LineSegment(
            Point((2, 4)), Point((4, 4))
        ) == get_rectangle_rectangle_intersection(r0, r1)

    def test_get_rectangle_rectangle_intersection_no_intersect(self):
        r0 = Rectangle(0, 0, 4, 4)
        r1 = Rectangle(5, 5, 8, 8)
        assert None is get_rectangle_rectangle_intersection(r0, r1)


class TestGetPolygonPointDist:
    def test_get_polygon_point_dist(self):
        poly = Polygon([Point((0, 0)), Point((1, 0)), Point((1, 1)), Point((0, 1))])
        pt = Point((2, 0.5))
        expected = 1.0
        assert expected == get_polygon_point_dist(poly, pt)

    def test_get_polygon_point_dist_inside(self):
        poly = Polygon([Point((0, 0)), Point((1, 0)), Point((1, 1)), Point((0, 1))])
        pt = Point((0.5, 0.5))
        expected = 0.0
        assert expected == get_polygon_point_dist(poly, pt)

    def test_get_polygon_point_dist_on_vertex(self):
        poly = Polygon([Point((0, 0)), Point((1, 0)), Point((1, 1)), Point((0, 1))])
        pt = Point((1.0, 1.0))
        expected = 0.0
        assert expected == get_polygon_point_dist(poly, pt)

    def test_get_polygon_point_dist_on_edge(self):
        poly = Polygon([Point((0, 0)), Point((1, 0)), Point((1, 1)), Point((0, 1))])
        pt = Point((0.5, 1.0))
        expected = 0.0
        assert expected == get_polygon_point_dist(poly, pt)


class TestGetPointsDist:
    def test_get_points_dist(self):
        pt1 = Point((0.5, 0.5))
        pt2 = Point((0.5, 0.5))
        assert get_points_dist(pt1, pt2) == 0

    def test_get_points_dist_diag(self):
        pt1 = Point((0, 0))
        pt2 = Point((1, 1))
        assert get_points_dist(pt1, pt2) == 2 ** (0.5)

    def test_get_points_dist_along_x(self):
        pt1 = Point((-1000, 1 / 3.0))
        pt2 = Point((1000, 1 / 3.0))
        assert get_points_dist(pt1, pt2) == 2000

    def test_get_points_dist_along_y(self):
        pt1 = Point((1 / 3.0, -500))
        pt2 = Point((1 / 3.0, 500))
        assert get_points_dist(pt1, pt2) == 1000


class TestGetSegmentPointDist:
    def test_get_segment_point_dist(self):
        seg = LineSegment(Point((0, 0)), Point((10, 0)))
        pt = Point((5, 5))
        assert get_segment_point_dist(seg, pt) == (5.0, 0.5)

    def test_get_segment_point_dist_on_end_point(self):
        seg = LineSegment(Point((0, 0)), Point((10, 0)))
        pt = Point((0, 0))
        assert get_segment_point_dist(seg, pt) == (0.0, 0.0)

    def test_get_segment_point_dist_on_middle(self):
        seg = LineSegment(Point((0, 0)), Point((10, 0)))
        pt = Point((5, 0))
        assert get_segment_point_dist(seg, pt) == (0.0, 0.5)

    def test_get_segment_point_diag(self):
        seg = LineSegment(Point((0, 0)), Point((10, 10)))
        pt = Point((5, 5))
        assert pytest.approx(get_segment_point_dist(seg, pt)[0]) == 0.0
        assert pytest.approx(get_segment_point_dist(seg, pt)[1]) == 0.5

    def test_get_segment_point_diag_with_dist(self):
        seg = LineSegment(Point((0, 0)), Point((10, 10)))
        pt = Point((0, 10))
        assert pytest.approx(get_segment_point_dist(seg, pt)[0]) == 50 ** (0.5)
        assert pytest.approx(get_segment_point_dist(seg, pt)[1]) == 0.5


class TestGetPointAtAngleAndDist:
    def test_get_point_at_angle_and_dist(self):
        ray = Ray(Point((0, 0)), Point((1, 0)))
        pt = get_point_at_angle_and_dist(ray, math.pi, 1.0)
        assert pytest.approx(pt[0]) == -1.0
        assert pytest.approx(pt[1]) == 0.0

    def test_get_point_at_angle_and_dist_diag(self):
        ray = Ray(Point((0, 0)), Point((1, 1)))
        pt = get_point_at_angle_and_dist(ray, math.pi, 2 ** (0.5))
        assert pytest.approx(pt[0]) == -1.0
        assert pytest.approx(pt[1]) == -1.0

    def test_get_point_at_angle_and_dist_diag_90(self):
        ray = Ray(Point((0, 0)), Point((1, 1)))
        pt = get_point_at_angle_and_dist(ray, -math.pi / 2.0, 2 ** (0.5))
        assert pytest.approx(pt[0]) == 1.0
        assert pytest.approx(pt[1]) == -1.0

    def test_get_point_at_angle_and_dist_diag_45(self):
        ray = Ray(Point((0, 0)), Point((1, 1)))
        pt = get_point_at_angle_and_dist(ray, -math.pi / 4.0, 1)
        assert pytest.approx(pt[0]) == 1.0
        assert pytest.approx(pt[1]) == 0.0


class TestConvexHull:
    def test_convex_hull(self):
        points = [Point((0, 0)), Point((4, 4)), Point((4, 0)), Point((3, 1))]
        assert [Point((0.0, 0.0)), Point((4.0, 0.0)), Point((4.0, 4.0))] == convex_hull(
            points
        )


class TestIsClockwise:
    def test_is_clockwise(self):
        vertices = [Point((0, 0)), Point((0, 10)), Point((10, 0))]
        assert True is is_clockwise(vertices)

    def test_is_clockwise_expect_false(self):
        vertices = [Point((0, 0)), Point((10, 0)), Point((0, 10))]
        assert False is is_clockwise(vertices)

    def test_is_clockwise_big(self):
        vertices = [
            (-106.57798, 35.174143999999998),
            (-106.583412, 35.174141999999996),
            (-106.58417999999999, 35.174143000000001),
            (-106.58377999999999, 35.175542999999998),
            (-106.58287999999999, 35.180543),
            (-106.58263099999999, 35.181455),
            (-106.58257999999999, 35.181643000000001),
            (-106.58198299999999, 35.184615000000001),
            (-106.58148, 35.187242999999995),
            (-106.58127999999999, 35.188243),
            (-106.58138, 35.188243),
            (-106.58108, 35.189442999999997),
            (-106.58104, 35.189644000000001),
            (-106.58028, 35.193442999999995),
            (-106.580029, 35.194541000000001),
            (-106.57974399999999, 35.195785999999998),
            (-106.579475, 35.196961999999999),
            (-106.57922699999999, 35.198042999999998),
            (-106.578397, 35.201665999999996),
            (-106.57827999999999, 35.201642999999997),
            (-106.57737999999999, 35.201642999999997),
            (-106.57697999999999, 35.201543000000001),
            (-106.56436599999999, 35.200311999999997),
            (-106.56058, 35.199942999999998),
            (-106.56048, 35.197342999999996),
            (-106.56048, 35.195842999999996),
            (-106.56048, 35.194342999999996),
            (-106.56048, 35.193142999999999),
            (-106.56048, 35.191873999999999),
            (-106.56048, 35.191742999999995),
            (-106.56048, 35.190242999999995),
            (-106.56037999999999, 35.188642999999999),
            (-106.56037999999999, 35.187242999999995),
            (-106.56037999999999, 35.186842999999996),
            (-106.56037999999999, 35.186552999999996),
            (-106.56037999999999, 35.185842999999998),
            (-106.56037999999999, 35.184443000000002),
            (-106.56037999999999, 35.182943000000002),
            (-106.56037999999999, 35.181342999999998),
            (-106.56037999999999, 35.180433000000001),
            (-106.56037999999999, 35.179943000000002),
            (-106.56037999999999, 35.178542999999998),
            (-106.56037999999999, 35.177790999999999),
            (-106.56037999999999, 35.177143999999998),
            (-106.56037999999999, 35.175643999999998),
            (-106.56037999999999, 35.174444000000001),
            (-106.56037999999999, 35.174043999999995),
            (-106.560526, 35.174043999999995),
            (-106.56478, 35.174043999999995),
            (-106.56627999999999, 35.174143999999998),
            (-106.566541, 35.174144999999996),
            (-106.569023, 35.174157000000001),
            (-106.56917199999999, 35.174157999999998),
            (-106.56938, 35.174143999999998),
            (-106.57061499999999, 35.174143999999998),
            (-106.57097999999999, 35.174143999999998),
            (-106.57679999999999, 35.174143999999998),
            (-106.57798, 35.174143999999998),
        ]
        assert True is is_clockwise(vertices)


class TestPointTouchesRectangle:
    def test_point_touches_rectangle_inside(self):
        rect = Rectangle(0, 0, 10, 10)
        point = Point((5, 5))
        assert point_touches_rectangle(point, rect)

    def test_point_touches_rectangle_on_edge(self):
        rect = Rectangle(0, 0, 10, 10)
        point = Point((10, 5))
        assert point_touches_rectangle(point, rect)

    def test_point_touches_rectangle_on_corner(self):
        rect = Rectangle(0, 0, 10, 10)
        point = Point((10, 10))
        assert point_touches_rectangle(point, rect)

    def test_point_touches_rectangle_outside(self):
        rect = Rectangle(0, 0, 10, 10)
        point = Point((11, 11))
        assert not point_touches_rectangle(point, rect)


class TestGetSharedSegments:
    def test_get_shared_segments(self):
        poly1 = Polygon([Point((0, 0)), Point((0, 1)), Point((1, 1)), Point((1, 0))])
        poly2 = Polygon([Point((1, 0)), Point((1, 1)), Point((2, 1)), Point((2, 0))])
        poly3 = Polygon([Point((0, 1)), Point((0, 2)), Point((1, 2)), Point((1, 1))])
        poly4 = Polygon([Point((1, 1)), Point((1, 2)), Point((2, 2)), Point((2, 1))])
        assert True is get_shared_segments(poly1, poly2, bool_ret=True)
        assert True is get_shared_segments(poly1, poly3, bool_ret=True)
        assert True is get_shared_segments(poly3, poly4, bool_ret=True)
        assert True is get_shared_segments(poly4, poly2, bool_ret=True)

        assert False is get_shared_segments(poly1, poly4, bool_ret=True)
        assert False is get_shared_segments(poly3, poly2, bool_ret=True)

    def test_get_shared_segments_non_bool(self):
        poly1 = Polygon([Point((0, 0)), Point((0, 1)), Point((1, 1)), Point((1, 0))])
        poly2 = Polygon([Point((1, 0)), Point((1, 1)), Point((2, 1)), Point((2, 0))])
        poly3 = Polygon([Point((0, 1)), Point((0, 2)), Point((1, 2)), Point((1, 1))])
        poly4 = Polygon([Point((1, 1)), Point((1, 2)), Point((2, 2)), Point((2, 1))])
        assert (
            LineSegment(Point((1, 0)), Point((1, 1)))
            == get_shared_segments(poly1, poly2)[0]
        )
        assert (
            LineSegment(Point((0, 1)), Point((1, 1)))
            == get_shared_segments(poly1, poly3)[0]
        )
        assert (
            LineSegment(Point((1, 2)), Point((1, 1)))
            == get_shared_segments(poly3, poly4)[0]
        )
        assert (
            LineSegment(Point((2, 1)), Point((1, 1)))
            == get_shared_segments(poly4, poly2)[0]
        )
        # expected =  [LineSegment(Point((1, 1)), Point((1, 0)))]
        # assert expected == get_shared_segments(poly1, poly3)
        # expected =  [LineSegment(Point((1, 1)), Point((1, 0)))]
        # assert expected == get_shared_segments(poly3, poly4)
        # expected =  [LineSegment(Point((1, 1)), Point((1, 0)))]
        # assert expected == get_shared_segments(poly4, poly2)


class TestDistanceMatrix:
    def test_distance_matrix(self):
        points = [(10, 10), (20, 10), (40, 10), (15, 20), (30, 20), (30, 30)]
        dist = distance_matrix(np.array(points), 2)
        for i in range(0, len(points)):
            for j in range(i, len(points)):
                x, y = points[i]
                _x, _y = points[j]
                d = ((x - _x) ** 2 + (y - _y) ** 2) ** (0.5)
                assert dist[i, j] == d

"""
Helper functions for computational geometry in PySAL.

"""

__author__ = "Sergio J. Rey, Xinyue Ye, Charles Schmidt, Andrew Winslow"
__credits__ = "Copyright (c) 2005-2009 Sergio J. Rey"

import doctest
import math
import copy
import random
from .shapes import *
from itertools import islice
import scipy.spatial
import numpy as np

EPSILON_SCALER = 3


__all__ = [
    "bbcommon",
    "get_bounding_box",
    "get_angle_between",
    "is_collinear",
    "get_segments_intersect",
    "get_segment_point_intersect",
    "get_polygon_point_intersect",
    "get_rectangle_point_intersect",
    "get_ray_segment_intersect",
    "get_rectangle_rectangle_intersection",
    "get_polygon_point_dist",
    "get_points_dist",
    "get_segment_point_dist",
    "get_point_at_angle_and_dist",
    "convex_hull",
    "is_clockwise",
    "point_touches_rectangle",
    "get_shared_segments",
    "distance_matrix",
]


def bbcommon(bb, bbother):
    """Old Stars method for bounding box overlap testing.
    Also defined in ``pysal.weights._cont_binning``.

    Parameters
    ----------
    bb : list
        A bounding box.
    bbother : list
        The bounding box to test against.
    
    Returns
    -------
    chflag : int
        ``1`` if ``bb`` overlaps ``bbother``, otherwise ``0``.

    Examples
    --------

    >>> b0 = [0, 0, 10, 10]
    >>> b1 = [10, 0, 20, 10]
    >>> bbcommon(b0, b1)
    1
    
    """

    chflag = 0

    if not ((bbother[2] < bb[0]) or (bbother[0] > bb[2])):
        if not ((bbother[3] < bb[1]) or (bbother[1] > bb[3])):
            chflag = 1

    return chflag


def get_bounding_box(items):
    """Find bounding box for a list of geometries.

    Parameters
    ----------
    items : list
        PySAL shapes.

    Returns
    -------
    rect = libpysal.cg.Rectangle
        The bounding box for a list of geometries.

    Examples
    --------
    
    >>> bb = get_bounding_box([Point((-1, 5)), Rectangle(0, 6, 11, 12)])
    >>> bb.left
    -1.0
    
    >>> bb.lower
    5.0
    
    >>> bb.right
    11.0
    
    >>> bb.upper
    12.0
    
    """

    def left(o):
        # Polygon, Ellipse
        if hasattr(o, "bounding_box"):
            return o.bounding_box.left
        # Rectangle
        elif hasattr(o, "left"):
            return o.left
        # Point
        else:
            return o[0]

    def right(o):
        # Polygon, Ellipse
        if hasattr(o, "bounding_box"):
            return o.bounding_box.right
        # Rectangle
        elif hasattr(o, "right"):
            return o.right
        # Point
        else:
            return o[0]

    def lower(o):
        # Polygon, Ellipse
        if hasattr(o, "bounding_box"):
            return o.bounding_box.lower
        # Rectangle
        elif hasattr(o, "lower"):
            return o.lower
        # Point
        else:
            return o[1]

    def upper(o):
        # Polygon, Ellipse
        if hasattr(o, "bounding_box"):
            return o.bounding_box.upper
        # Rectangle
        elif hasattr(o, "upper"):
            return o.upper
        # Point
        else:
            return o[1]

    rect = Rectangle(
        min(list(map(left, items))),
        min(list(map(lower, items))),
        max(list(map(right, items))),
        max(list(map(upper, items))),
    )

    return rect


def get_angle_between(ray1, ray2):
    """Returns the angle formed between a pair of rays which share an origin.

    Parameters
    ----------
    ray1 : libpysal.cg.Ray
        A ray forming the beginning of the angle measured.
    ray2 : libpysal.cg.Ray
        A ray forming the end of the angle measured.
    
    Returns
    -------
    angle : float
        The angle between ``ray1`` and ``ray2``.
    
    Raises
    ------
    ValueError
        Raised when rays do not have the same origin.
    
    Examples
    --------
    
    >>> get_angle_between(
    ...     Ray(Point((0, 0)), Point((1, 0))),
    ...     Ray(Point((0, 0)), Point((1, 0)))
    ... )
    0.0
    
    """

    if ray1.o != ray2.o:
        raise ValueError("Rays must have the same origin.")

    vec1 = (ray1.p[0] - ray1.o[0], ray1.p[1] - ray1.o[1])
    vec2 = (ray2.p[0] - ray2.o[0], ray2.p[1] - ray2.o[1])

    rot_theta = -math.atan2(vec1[1], vec1[0])
    rot_matrix = [
        [math.cos(rot_theta), -math.sin(rot_theta)],
        [math.sin(rot_theta), math.cos(rot_theta)],
    ]

    rot_vec2 = (
        rot_matrix[0][0] * vec2[0] + rot_matrix[0][1] * vec2[1],
        rot_matrix[1][0] * vec2[0] + rot_matrix[1][1] * vec2[1],
    )

    angle = math.atan2(rot_vec2[1], rot_vec2[0])

    return angle


def is_collinear(p1, p2, p3):
    """Returns whether a triplet of points is collinear.

    Parameters
    ----------
    p1 : libpysal.cg.Point
        A point.
    p2 : libpysal.cg.Point
        A point.
    p3 : libpysal.cg.Point
        A point.

    Returns
    -------
    collinear : bool
        ``True`` if ``{p1, p2, p3}`` are collinear, otherwise ``False``.

    Examples
    --------
    
    >>> is_collinear(Point((0, 0)), Point((1, 1)), Point((5, 5)))
    True
    
    >>> is_collinear(Point((0, 0)), Point((1, 1)), Point((5, 0)))
    False
    
    """

    eps = np.finfo(type(p1[0])).eps

    slope_diff = abs(
        (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
    )

    very_small_dist = EPSILON_SCALER * eps

    collinear = slope_diff < very_small_dist

    return collinear


def get_segments_intersect(seg1, seg2):
    """Returns the intersection of two segments if one exists.

    Parameters
    ----------
    seg1 : libpysal.cg.LineSegment
        A segment to check for an intersection.
    seg2 : libpysal.cg.LineSegment
        The segment to check against ``seg1`` for an intersection.

    Returns
    -------
    intersection : {libpysal.cg.Point, libpysal.cg.LineSegment, None}
        The intersecting point or line between ``seg1`` and
        ``seg2`` if an intersection exists or ``None`` if
        ``seg1`` and ``seg2`` do not intersect.

    Examples
    --------
    
    >>> seg1 = LineSegment(Point((0, 0)), Point((0, 10)))
    >>> seg2 = LineSegment(Point((-5, 5)), Point((5, 5)))
    >>> i = get_segments_intersect(seg1, seg2)
    >>> isinstance(i, Point)
    True
    
    >>> str(i)
    '(0.0, 5.0)'
    
    >>> seg3 = LineSegment(Point((100, 100)), Point((100, 101)))
    >>> i = get_segments_intersect(seg2, seg3)
    
    """

    p1 = seg1.p1
    p2 = seg1.p2
    p3 = seg2.p1
    p4 = seg2.p2
    a = p2[0] - p1[0]
    b = p3[0] - p4[0]
    c = p2[1] - p1[1]
    d = p3[1] - p4[1]
    det = float(a * d - b * c)

    intersection = None

    if det == 0:
        if seg1 == seg2:
            intersection = LineSegment(seg1.p1, seg1.p2)
        else:
            a = get_segment_point_intersect(seg2, seg1.p1)
            b = get_segment_point_intersect(seg2, seg1.p2)
            c = get_segment_point_intersect(seg1, seg2.p1)
            d = get_segment_point_intersect(seg1, seg2.p2)
            if a and b:  # seg1 in seg2
                intersection = LineSegment(seg1.p1, seg1.p2)
            if c and d:  # seg2 in seg1
                intersection = LineSegment(seg2.p1, seg2.p2)
            if (a or b) and (c or d):
                p1 = a if a else b
                p2 = c if c else d
                intersection = LineSegment(p1, p2)
    else:
        a_inv = d / det
        b_inv = -b / det
        c_inv = -c / det
        d_inv = a / det
        m = p3[0] - p1[0]
        n = p3[1] - p1[1]
        x = a_inv * m + b_inv * n
        y = c_inv * m + d_inv * n
        intersect_exists = 0 <= x <= 1 and 0 <= y <= 1

        if intersect_exists:
            intersection = Point(
                (p1[0] + x * (p2[0] - p1[0]), p1[1] + x * (p2[1] - p1[1]))
            )

    return intersection


def get_segment_point_intersect(seg, pt):
    """Returns the intersection of a segment and point.

    Parameters
    ----------
    seg : libpysal.cg.LineSegment
        A segment to check for an intersection.
    pt : libpysal.cg.Point
        A point to check ``seg`` for an intersection.

    Returns
    -------
    pt : {libpysal.cg.Point, None}
        The intersection of a ``seg`` and ``pt`` if one exists, otherwise ``None``.

    Examples
    --------
    
    >>> seg = LineSegment(Point((0, 0)), Point((0, 10)))
    >>> pt = Point((0, 5))
    >>> i = get_segment_point_intersect(seg, pt)
    >>> str(i)
    '(0.0, 5.0)'
    
    >>> pt2 = Point((5, 5))
    >>> get_segment_point_intersect(seg, pt2)
    
    """

    eps = np.finfo(type(pt[0])).eps

    if is_collinear(pt, seg.p1, seg.p2):
        if get_segment_point_dist(seg, pt)[0] < EPSILON_SCALER * eps:
            pass
        else:
            pt = None
    else:
        vec1 = (pt[0] - seg.p1[0], pt[1] - seg.p1[1])
        vec2 = (seg.p2[0] - seg.p1[0], seg.p2[1] - seg.p1[1])

        if abs(vec1[0] * vec2[1] - vec1[1] * vec2[0]) < eps:
            pass
        else:
            pt = None

    return pt


def get_polygon_point_intersect(poly, pt):
    """Returns the intersection of a polygon and point.

    Parameters
    ----------
    poly : libpysal.cg.Polygon
        A polygon to check for an intersection.
    pt : libpysal.cg.Point
        A point to check ``poly`` for an intersection.

    Returns
    -------
    ret : {libpysal.cg.Point, None}
        The intersection of a ``poly`` and ``pt`` if one exists, otherwise ``None``.

    Examples
    --------
    
    >>> poly = Polygon([Point((0, 0)), Point((1, 0)), Point((1, 1)), Point((0, 1))])
    >>> pt = Point((0.5, 0.5))
    >>> i = get_polygon_point_intersect(poly, pt)
    >>> str(i)
    '(0.5, 0.5)'
    
    >>> pt2 = Point((2, 2))
    >>> get_polygon_point_intersect(poly, pt2)
    
    """

    def pt_lies_on_part_boundary(p, vx):
        vx_range = range(-1, len(vx) - 1)
        seg = lambda i: LineSegment(vx[i], vx[i + 1])
        return [i for i in vx_range if get_segment_point_dist(seg(i), p)[0] == 0] != []

    ret = None

    # Weed out points that aren't even close
    if get_rectangle_point_intersect(poly.bounding_box, pt) is None:
        pass
    else:
        if [vxs for vxs in poly._vertices if pt_lies_on_part_boundary(pt, vxs)] != []:
            ret = pt
        elif [vxs for vxs in poly._vertices if _point_in_vertices(pt, vxs)] != []:
            ret = pt
        if poly._holes != [[]]:
            if [vxs for vxs in poly.holes if pt_lies_on_part_boundary(pt, vxs)] != []:
                # pt lies on boundary of hole.
                pass
            if [vxs for vxs in poly.holes if _point_in_vertices(pt, vxs)] != []:
                # pt lines inside a hole.
                ret = None
            # raise NotImplementedError,
            # 'Cannot compute containment for polygon with holes'

    return ret


def get_rectangle_point_intersect(rect, pt):
    """Returns the intersection of a rectangle and point.

    Parameters
    ----------
    rect : libpysal.cg.Rectangle
        A rectangle to check for an intersection.
    pt : libpysal.cg.Point
        A point to check ``rect`` for an intersection.

    Returns
    -------
    pt : {libpysal.cg.Point, None}
        The intersection of a ``rect`` and ``pt`` if one exists, otherwise ``None``.

    Examples
    --------
    
    >>> rect = Rectangle(0, 0, 5, 5)
    >>> pt = Point((1, 1))
    >>> i = get_rectangle_point_intersect(rect, pt)
    >>> str(i)
    '(1.0, 1.0)'
    
    >>> pt2 = Point((10, 10))
    >>> get_rectangle_point_intersect(rect, pt2)
    
    """

    if rect.left <= pt[0] <= rect.right and rect.lower <= pt[1] <= rect.upper:
        pass
    else:
        pt = None

    return pt


def get_ray_segment_intersect(ray, seg):
    """Returns the intersection of a ray and line segment.

    Parameters
    ----------
    ray : libpysal.cg.Ray
        A ray to check for an intersection.
    seg : libpysal.cg.LineSegment
        A segment to check for an intersection against ``ray``.

    Returns
    -------
    intersection : {libpysal.cg.Point, libpysal.cg.LineSegment, None}
        The intersecting point or line between ``ray`` and
        ``seg`` if an intersection exists or ``None`` if
        ``ray`` and ``seg`` do not intersect.
    
    See Also
    --------
    
    libpysal.cg.get_segments_intersect
    
    
    Examples
    --------
    
    >>> ray = Ray(Point((0, 0)), Point((0, 1)))
    >>> seg = LineSegment(Point((-1, 10)), Point((1, 10)))
    >>> i = get_ray_segment_intersect(ray, seg)
    >>> isinstance(i, Point)
    True
    
    >>> str(i)
    '(0.0, 10.0)'
    
    >>> seg2 = LineSegment(Point((10, 10)), Point((10, 11)))
    >>> get_ray_segment_intersect(ray, seg2)
    
    """

    # Upper bound on origin to segment dist (+1)
    d = (
        max(
            math.hypot(seg.p1[0] - ray.o[0], seg.p1[1] - ray.o[1]),
            math.hypot(seg.p2[0] - ray.o[0], seg.p2[1] - ray.o[1]),
        )
        + 1
    )
    ratio = d / math.hypot(ray.o[0] - ray.p[0], ray.o[1] - ray.p[1])
    ray_seg = LineSegment(
        ray.o,
        Point(
            (
                ray.o[0] + ratio * (ray.p[0] - ray.o[0]),
                ray.o[1] + ratio * (ray.p[1] - ray.o[1]),
            )
        ),
    )

    intersection = get_segments_intersect(seg, ray_seg)

    return intersection


def get_rectangle_rectangle_intersection(r0, r1, checkOverlap=True):
    """Returns the intersection between two rectangles.

    Parameters
    ----------
    r0 : libpysal.cg.Rectangle
        A rectangle to check for an intersection.
    r1 : libpysal.cg.Rectangle
        A rectangle to check for an intersection against ``r0``.
    checkOverlap : bool
        Call ``bbcommon(r0, r1)`` prior to complex geometry
        checking. Default is ``True``. Prior to setting as
        ``False`` see the Notes section. 
    
    Returns
    -------
    intersection : {libpysal.cg.Point, libpysal.cg.LineSegment, libpysal.cg.Rectangle, None}
        The intersecting point, line, or rectangle between 
        `r0`` and ``r1`` if an intersection exists or ``None``
        if ``r0`` and ``r1`` do not intersect.

    Notes
    -----
    
    The algorithm assumes the rectangles overlap. The keyword
    ``checkOverlap=False`` should be used with extreme caution.

    Examples
    --------
    
    >>> r0 = Rectangle(0,4,6,9)
    >>> r1 = Rectangle(4,0,9,7)
    >>> ri = get_rectangle_rectangle_intersection(r0,r1)
    >>> ri[:]
    [4.0, 4.0, 6.0, 7.0]
    
    >>> r0 = Rectangle(0,0,4,4)
    >>> r1 = Rectangle(2,1,6,3)
    >>> ri = get_rectangle_rectangle_intersection(r0,r1)
    >>> ri[:]
    [2.0, 1.0, 4.0, 3.0]
    
    >>> r0 = Rectangle(0,0,4,4)
    >>> r1 = Rectangle(2,1,3,2)
    >>> ri = get_rectangle_rectangle_intersection(r0,r1)
    >>> ri[:] == r1[:]
    True
    
    """

    intersection = None
    common_bb = True

    if checkOverlap:
        if not bbcommon(r0, r1):
            # raise ValueError, "Rectangles do not intersect"
            common_bb = False

    if common_bb:
        left = max(r0.left, r1.left)
        lower = max(r0.lower, r1.lower)
        right = min(r0.right, r1.right)
        upper = min(r0.upper, r1.upper)

        if upper == lower and left == right:
            intersection = Point((left, lower))
        elif upper == lower:
            intersection = LineSegment(Point((left, lower)), Point((right, lower)))
        elif left == right:
            intersection = LineSegment(Point((left, lower)), Point((left, upper)))
        else:
            intersection = Rectangle(left, lower, right, upper)

    return intersection


def get_polygon_point_dist(poly, pt):
    """Returns the distance between a polygon and point.

    Parameters
    ----------
    poly : libpysal.cg.Polygon
        A polygon to compute distance from.
    
    pt : libpysal.cg.Point
        a point to compute distance from

    Returns
    -------
    dist : float
        The distance between ``poly`` and ``point``.
    
    Examples
    --------
    
    >>> poly = Polygon([Point((0, 0)), Point((1, 0)), Point((1, 1)), Point((0, 1))])
    >>> pt = Point((2, 0.5))
    >>> get_polygon_point_dist(poly, pt)
    1.0
    
    >>> pt2 = Point((0.5, 0.5))
    >>> get_polygon_point_dist(poly, pt2)
    0.0
    
    """

    if get_polygon_point_intersect(poly, pt) is not None:
        dist = 0.0
    else:
        part_prox = []
        for vertices in poly._vertices:
            vx_range = range(-1, len(vertices) - 1)
            seg = lambda i: LineSegment(vertices[i], vertices[i + 1])
            _min_dist = min([get_segment_point_dist(seg(i), pt)[0] for i in vx_range])
            part_prox.append(_min_dist)
        dist = min(part_prox)

    return dist


def get_points_dist(pt1, pt2):
    """Returns the distance between a pair of points.

    Parameters
    ----------
    pt1 : libpysal.cg.Point
        A point.
    
    pt2 : libpysal.cg.Point
        The other point.
    
    Returns
    -------
    dist : float
        The distance between ``pt1`` and ``pt2``.
    
    Examples
    --------
    
    >>> get_points_dist(Point((4, 4)), Point((4, 8)))
    4.0
    
    >>> get_points_dist(Point((0, 0)), Point((0, 0)))
    0.0
    
    """

    dist = math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])

    return dist


def get_segment_point_dist(seg, pt):
    """Returns (1) the distance between a line segment and point
    and (2) the distance along the segment to the closest location on the
    segment from the point as a ratio of the length of the segment.

    Parameters
    ----------
    seg : libpysal.cg.LineSegment
        A line segment to compute distance from.
    pt : libpysal.cg.Point
        A point to compute distance from.
    
    Returns
    -------
    dist : float
        The distance between ``seg`` and ``pt``.
    ratio : float
        The distance along ``seg`` to the closest location on
        ``seg`` from ``pt`` as a ratio of the length of ``seg``.

    Examples
    --------
    
    >>> seg = LineSegment(Point((0, 0)), Point((10, 0)))
    >>> pt = Point((5, 5))
    >>> get_segment_point_dist(seg, pt)
    (5.0, 0.5)
    
    >>> pt2 = Point((0, 0))
    >>> get_segment_point_dist(seg, pt2)
    (0.0, 0.0)
    
    """

    src_p = seg.p1
    dest_p = seg.p2

    # Shift line to go through origin
    points_0 = pt[0] - src_p[0]
    points_1 = pt[1] - src_p[1]
    points_2 = 0
    points_3 = 0
    points_4 = dest_p[0] - src_p[0]
    points_5 = dest_p[1] - src_p[1]

    segment_length = get_points_dist(src_p, dest_p)

    # Meh, robustness...
    # maybe should incorporate this into a more general approach later
    if segment_length == 0:
        dist, ratio = get_points_dist(pt, src_p), 0

    else:
        u_x = points_4 / segment_length
        u_y = points_5 / segment_length

        inter_x = u_x * u_x * points_0 + u_x * u_y * points_1
        inter_y = u_x * u_y * points_0 + u_y * u_y * points_1

        src_proj_dist = get_points_dist((0, 0), (inter_x, inter_y))
        dest_proj_dist = get_points_dist((inter_x, inter_y), (points_4, points_5))

        if src_proj_dist > segment_length or dest_proj_dist > segment_length:
            src_pt_dist = get_points_dist((points_2, points_3), (points_0, points_1))
            dest_pt_dist = get_points_dist((points_4, points_5), (points_0, points_1))

            if src_pt_dist < dest_pt_dist:
                dist, ratio = src_pt_dist, 0
            else:
                dist, ratio = dest_pt_dist, 1
        else:
            dist = get_points_dist((inter_x, inter_y), (points_0, points_1))
            ratio = src_proj_dist / segment_length

    return dist, ratio


def get_point_at_angle_and_dist(ray, angle, dist):
    """Returns the point at a distance and angle relative to the origin of a ray.

    Parameters
    ----------
    ray : libpysal.cg.Ray
        The ray to which ``angle`` and ``dist`` are relative.
    angle : float
        The angle relative to ``ray`` at which ``point`` is located.
    dist : float
        The distance from the origin of ``ray`` at which ``point`` is located.
    
    Returns
    -------
    point : libpysal.cg.Point
        The point at ``dist`` and ``angle`` relative to the origin of ``ray``.

    Examples
    --------
    
    >>> ray = Ray(Point((0, 0)), Point((1, 0)))
    >>> pt = get_point_at_angle_and_dist(ray, math.pi, 1.0)
    >>> isinstance(pt, Point)
    True
    
    >>> round(pt[0], 8)
    -1.0
    
    >>> round(pt[1], 8)
    0.0
    
    """

    v = (ray.p[0] - ray.o[0], ray.p[1] - ray.o[1])
    cur_angle = math.atan2(v[1], v[0])
    dest_angle = cur_angle + angle

    point = Point(
        (ray.o[0] + dist * math.cos(dest_angle), ray.o[1] + dist * math.sin(dest_angle))
    )

    return point


def convex_hull(points):
    """Returns the convex hull of a set of points.
    
    Parameters
    ----------
    points : list
        A list of points for computing the convex hull.

    Returns
    -------
    stack : list
        A list of points representing the convex hull.
    
    Examples
    --------
    
    >>> points = [Point((0, 0)), Point((4, 4)), Point((4, 0)), Point((3, 1))]
    >>> convex_hull(points)
    [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0)]
    
    """

    def right_turn(p1, p2, p3) -> bool:
        """Returns if ``p1`` -> ``p2`` -> ``p3`` forms a 'right turn'."""
        vec1 = (p2[0] - p1[0], p2[1] - p1[1])
        vec2 = (p3[0] - p2[0], p3[1] - p2[1])
        _rt = vec2[0] * vec1[1] - vec2[1] * vec1[0] >= 0
        return _rt

    points = copy.copy(points)
    lowest = min(points, key=lambda p: (p[1], p[0]))

    points.remove(lowest)
    points.sort(key=lambda p: math.atan2(p[1] - lowest[1], p[0] - lowest[0]))

    stack = [lowest]

    for p in points:
        stack.append(p)
        while len(stack) > 3 and right_turn(stack[-3], stack[-2], stack[-1]):
            stack.pop(-2)

    return stack


def is_clockwise(vertices):
    """Returns whether a list of points describing
    a polygon are clockwise or counterclockwise.
    
    Parameters
    ----------
    vertices : list
        A list of points that form a single ring.

    Returns
    -------
    clockwise : bool
        ``True`` if ``vertices`` are clockwise, otherwise ``False``.
    
    See Also
    --------
    
    libpysal.cg.ccw
    
    Examples
    --------
    
    >>> is_clockwise([Point((0, 0)), Point((10, 0)), Point((0, 10))])
    False
    
    >>> is_clockwise([Point((0, 0)), Point((0, 10)), Point((10, 0))])
    True
    
    >>> v = [
    ...     (-106.57798, 35.174143999999998),
    ...     (-106.583412, 35.174141999999996),
    ...     (-106.58417999999999, 35.174143000000001),
    ...     (-106.58377999999999, 35.175542999999998),
    ...     (-106.58287999999999, 35.180543),
    ...     (-106.58263099999999, 35.181455),
    ...     (-106.58257999999999, 35.181643000000001),
    ...     (-106.58198299999999, 35.184615000000001),
    ...     (-106.58148, 35.187242999999995),
    ...     (-106.58127999999999, 35.188243),
    ...     (-106.58138, 35.188243),
    ...     (-106.58108, 35.189442999999997),
    ...     (-106.58104, 35.189644000000001),
    ...     (-106.58028, 35.193442999999995),
    ...     (-106.580029, 35.194541000000001),
    ...     (-106.57974399999999, 35.195785999999998),
    ...     (-106.579475, 35.196961999999999),
    ...     (-106.57922699999999, 35.198042999999998),
    ...     (-106.578397, 35.201665999999996),
    ...     (-106.57827999999999, 35.201642999999997),
    ...     (-106.57737999999999, 35.201642999999997),
    ...     (-106.57697999999999, 35.201543000000001),
    ...     (-106.56436599999999, 35.200311999999997),
    ...     (-106.56058, 35.199942999999998),
    ...     (-106.56048, 35.197342999999996),
    ...     (-106.56048, 35.195842999999996),
    ...     (-106.56048, 35.194342999999996),
    ...     (-106.56048, 35.193142999999999),
    ...     (-106.56048, 35.191873999999999),
    ...     (-106.56048, 35.191742999999995),
    ...     (-106.56048, 35.190242999999995),
    ...     (-106.56037999999999, 35.188642999999999),
    ...     (-106.56037999999999, 35.187242999999995),
    ...     (-106.56037999999999, 35.186842999999996),
    ...     (-106.56037999999999, 35.186552999999996),
    ...     (-106.56037999999999, 35.185842999999998),
    ...     (-106.56037999999999, 35.184443000000002),
    ...     (-106.56037999999999, 35.182943000000002),
    ...     (-106.56037999999999, 35.181342999999998),
    ...     (-106.56037999999999, 35.180433000000001),
    ...     (-106.56037999999999, 35.179943000000002),
    ...     (-106.56037999999999, 35.178542999999998),
    ...     (-106.56037999999999, 35.177790999999999),
    ...     (-106.56037999999999, 35.177143999999998),
    ...     (-106.56037999999999, 35.175643999999998),
    ...     (-106.56037999999999, 35.174444000000001),
    ...     (-106.56037999999999, 35.174043999999995),
    ...     (-106.560526, 35.174043999999995),
    ...     (-106.56478, 35.174043999999995),
    ...     (-106.56627999999999, 35.174143999999998),
    ...     (-106.566541, 35.174144999999996),
    ...     (-106.569023, 35.174157000000001),
    ...     (-106.56917199999999, 35.174157999999998),
    ...     (-106.56938, 35.174143999999998),
    ...     (-106.57061499999999, 35.174143999999998),
    ...     (-106.57097999999999, 35.174143999999998),
    ...     (-106.57679999999999, 35.174143999999998),
    ...     (-106.57798, 35.174143999999998)
    ... ]
    >>> is_clockwise(v)
    True
    
    """

    clockwise = True

    if not len(vertices) < 3:
        area = 0.0
        ax, ay = vertices[0]
        for bx, by in vertices[1:]:
            area += ax * by - ay * bx
            ax, ay = bx, by
        bx, by = vertices[0]
        area += ax * by - ay * bx

        clockwise = area < 0.0

    return clockwise


def ccw(vertices):
    """Returns whether a list of points is counterclockwise.
    
    Parameters
    ----------
    vertices : list
        A list of points that form a single ring.

    Returns
    -------
    counter_clockwise : bool
        ``True`` if ``vertices`` are counter clockwise, otherwise ``False``.
    
    See Also
    --------
    
    libpysal.cg.is_clockwise
    
    Examples
    --------
    
    >>> ccw([Point((0, 0)), Point((10, 0)), Point((0, 10))])
    True
    
    >>> ccw([Point((0, 0)), Point((0, 10)), Point((10, 0))])
    False
    
    """

    counter_clockwise = True

    if is_clockwise(vertices):
        counter_clockwise = False

    return counter_clockwise


def seg_intersect(a, b, c, d):
    """Tests if two segments (a,b) and (c,d) intersect.
    
    Parameters
    ----------
    a : libpysal.cg.Point
        The first vertex for the first segment.
    b : libpysal.cg.Point
        The second vertex for the first segment.
    c : libpysal.cg.Point
        The first vertex for the second segment.
    d : libpysal.cg.Point
        The second vertex for the second segment.
    
    Returns
    -------
    segments_intersect : bool
        ``True`` if segments ``(a,b)`` and ``(c,d)``, otherwise ``False``.
    
    Examples
    --------
    
    >>> a = Point((0,1))
    >>> b = Point((0,10))
    >>> c = Point((-2,5))
    >>> d = Point((2,5))
    >>> e = Point((-3,5))
    >>> seg_intersect(a, b, c, d)
    True
    
    >>> seg_intersect(a, b, c, e)
    False
    
    """

    segments_intersect = True

    acd_bcd = ccw([a, c, d]) == ccw([b, c, d])

    abc_abd = ccw([a, b, c]) == ccw([a, b, d])

    if acd_bcd or abc_abd:
        segments_intersect = False

    return segments_intersect


def _point_in_vertices(pt, vertices):
    """**HELPER METHOD. DO NOT CALL.** Returns whether a point
    is contained in a polygon specified by a sequence of vertices.

    Parameters
    ----------
    pt : libpysal.cg.Point
        A point.
    vertices : list
        A list of vertices representing as polygon.
    
    Returns
    -------
    pt_in_poly : bool
        ``True`` if ``pt`` is contained in ``vertices``, otherwise ``False``.

    Examples
    --------
    
    >>> _point_in_vertices(
    ...     Point((1, 1)),
    ...     [Point((0, 0)), Point((10, 0)), Point((0, 10))]
    ... )
    True
    
    """

    def neg_ray_intersect(p1, p2, p3) -> bool:
        """Returns whether a ray in the negative-x
        direction from ``p3`` intersects the segment between.
        """

        if not min(p1[1], p2[1]) <= p3[1] <= max(p1[1], p2[1]):
            nr_inters = False
        else:
            if p1[1] > p2[1]:
                vec1 = (p2[0] - p1[0], p2[1] - p1[1])
            else:
                vec1 = (p1[0] - p2[0], p1[1] - p2[1])

            vec2 = (p3[0] - p1[0], p3[1] - p1[1])

            nr_inters = vec1[0] * vec2[1] - vec2[0] * vec1[1] >= 0

        return nr_inters

    vert_y_set = set([v[1] for v in vertices])
    while pt[1] in vert_y_set:
        # Perturb the location very slightly
        pt = pt[0], pt[1] + -1e-14 + random.random() * 2e-14

    inters = 0
    for i in range(-1, len(vertices) - 1):
        v1 = vertices[i]
        v2 = vertices[i + 1]
        if neg_ray_intersect(v1, v2, pt):
            inters += 1

    pt_in_poly = inters % 2 == 1

    return pt_in_poly


def point_touches_rectangle(point, rect):
    """Returns ``True`` (``1``) if the point is in the rectangle
    or touches it's boundary, otherwise ``False`` (``0``).

    Parameters
    ----------
    point : {libpysal.cg.Point, tuple}
        A point or point coordinates.
    rect : libpysal.cg.Rectangle
        A rectangle.
    
    Returns
    -------
    chflag : int
        ``1`` if ``point`` is in (or touches
        boundary of) ``rect``, otherwise ``0``.
    
    Examples
    --------
    
    >>> rect = Rectangle(0, 0, 10, 10)
    >>> a = Point((5, 5))
    >>> b = Point((10, 5))
    >>> c = Point((11, 11))
    >>> point_touches_rectangle(a, rect)
    1
    
    >>> point_touches_rectangle(b, rect)
    1
    
    >>> point_touches_rectangle(c, rect)
    0
    
    """

    chflag = 0
    if point[0] >= rect.left and point[0] <= rect.right:
        if point[1] >= rect.lower and point[1] <= rect.upper:
            chflag = 1

    return chflag


def get_shared_segments(poly1, poly2, bool_ret=False):
    """Returns the line segments in common to both polygons.

    Parameters
    ----------
    poly1 : libpysal.cg.Polygon
        A Polygon.
    poly2 : libpysal.cg.Polygon
        A Polygon.
    bool_ret : bool
        Return only a ``bool``. Default is ``False``.
    
    Returns
    -------
    common : list
        The shared line segments between ``poly1`` and ``poly2``.
    _ret_bool : bool
        Whether ``poly1`` and ``poly2`` share a
        segment (``True``) or not (``False``).

    Examples
    --------
    
    >>> from libpysal.cg.shapes import Polygon
    >>> x = [0, 0, 1, 1]
    >>> y = [0, 1, 1, 0]
    >>> poly1 = Polygon(list(map(Point, zip(x, y))) )
    >>> x = [a+1 for a in x]
    >>> poly2 = Polygon(list(map(Point, zip(x, y))) )
    >>> get_shared_segments(poly1, poly2, bool_ret=True)
    True

    """

    # get_rectangle_rectangle_intersection inlined for speed.
    r0 = poly1.bounding_box
    r1 = poly2.bounding_box
    wLeft = max(r0.left, r1.left)
    wLower = max(r0.lower, r1.lower)
    wRight = min(r0.right, r1.right)
    wUpper = min(r0.upper, r1.upper)

    segmentsA = set()
    common = list()
    partsA = poly1.parts

    for part in poly1.parts + [p for p in poly1.holes if p]:
        if part[0] != part[-1]:  # not closed
            part = part[:] + part[0:1]
        a = part[0]

        for b in islice(part, 1, None):
            # inlining point_touches_rectangle for speed
            x, y = a
            # check if point a is in the bounding box intersection
            if x >= wLeft and x <= wRight and y >= wLower and y <= wUpper:
                x, y = b
                # check if point b is in the bounding box intersection
                if x >= wLeft and x <= wRight and y >= wLower and y <= wUpper:
                    if a > b:
                        segmentsA.add((b, a))
                    else:
                        segmentsA.add((a, b))
            a = b

    _ret_bool = False

    for part in poly2.parts + [p for p in poly2.holes if p]:
        if part[0] != part[-1]:  # not closed
            part = part[:] + part[0:1]
        a = part[0]

        for b in islice(part, 1, None):
            # inlining point_touches_rectangle for speed
            x, y = a
            if x >= wLeft and x <= wRight and y >= wLower and y <= wUpper:
                x, y = b
                if x >= wLeft and x <= wRight and y >= wLower and y <= wUpper:
                    if a > b:
                        seg = (b, a)
                    else:
                        seg = (a, b)
                    if seg in segmentsA:
                        common.append(LineSegment(*seg))
                        if bool_ret:
                            _ret_bool = True
                            return _ret_bool
            a = b

    if bool_ret:
        if len(common) > 0:
            _ret_bool = True
        return _ret_bool

    return common


def distance_matrix(X, p=2.0, threshold=5e7):
    """Calculate a distance matrix.

    Parameters
    ----------
    X : numpy.ndarray
        An :math:`n \\times k` array where :math:`n` is the number
        of observations and :math:`k` is the number of dimensions
        (2 for :math:`x,y`).
    p : float
        Minkowski `p`-norm distance metric parameter where
        :math:`1<=\mathtt{p}<=\infty`. ``2`` is Euclidean distance and
        ``1`` is Manhattan distance. Default is ``2.0``.
    threshold : int
        If :math:`(\mathtt{n}**2)*32 > \mathtt{threshold}` use
        ``scipy.spatial.distance_matrix`` instead of working in RAM,
        this is roughly the amount of RAM (in bytes) that will be used.
        Must be positive. Default is ``5e7``.

    Returns
    -------
    D : numpy.ndarray
        An n by :math:`m` :math:`p`-norm distance matrix.
    
    Raises
    ------
    TypeError
        Raised when an invalid dimensional array is passed in.
    
    Notes
    -----
    
    Needs optimization/integration with other weights in PySAL.
    
    Examples
    --------
    
    >>> x, y = [r.flatten() for r in np.indices((3, 3))]
    >>> data = np.array([x, y]).T
    >>> d = distance_matrix(data)
    >>> np.array(d)
    array([[0.        , 1.        , 2.        , 1.        , 1.41421356,
            2.23606798, 2.        , 2.23606798, 2.82842712],
           [1.        , 0.        , 1.        , 1.41421356, 1.        ,
            1.41421356, 2.23606798, 2.        , 2.23606798],
           [2.        , 1.        , 0.        , 2.23606798, 1.41421356,
            1.        , 2.82842712, 2.23606798, 2.        ],
           [1.        , 1.41421356, 2.23606798, 0.        , 1.        ,
            2.        , 1.        , 1.41421356, 2.23606798],
           [1.41421356, 1.        , 1.41421356, 1.        , 0.        ,
            1.        , 1.41421356, 1.        , 1.41421356],
           [2.23606798, 1.41421356, 1.        , 2.        , 1.        ,
            0.        , 2.23606798, 1.41421356, 1.        ],
           [2.        , 2.23606798, 2.82842712, 1.        , 1.41421356,
            2.23606798, 0.        , 1.        , 2.        ],
           [2.23606798, 2.        , 2.23606798, 1.41421356, 1.        ,
            1.41421356, 1.        , 0.        , 1.        ],
           [2.82842712, 2.23606798, 2.        , 2.23606798, 1.41421356,
            1.        , 2.        , 1.        , 0.        ]])
    
    """

    if X.ndim == 1:
        X.shape = (X.shape[0], 1)

    if X.ndim > 2:
        msg = "Should be 2D point coordinates: %s dimensions present." % X.ndim
        raise TypeError(msg)

    n, k = X.shape

    if (n ** 2) * 32 > threshold:
        D = scipy.spatial.distance_matrix(X, X, p)
    else:
        M = np.ones((n, n))
        D = np.zeros((n, n))
        for col in range(k):
            x = X[:, col]
            xM = x * M
            dx = xM - xM.T
            if p % 2 != 0:
                dx = np.abs(dx)
            dx2 = dx ** p
            D += dx2
        D = D ** (1.0 / p)

    return D

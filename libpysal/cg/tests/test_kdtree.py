# Tests derived from Arc_KDTree docstring examples to get the expected behavior


import math

import pytest

from libpysal.cg import sphere
from libpysal.cg.kdtree import KDTree


def test_arc_kdtree_basic_query():
    pts = [(0, 90), (0, 0), (180, 0), (0, -90)]
    kd = KDTree(pts, distance_metric="Arc", radius=sphere.RADIUS_EARTH_KM)

    d, i = kd.query((90, 0), k=4)

    assert len(d) == 4
    assert len(i) == 4


def test_arc_distance_quarter_circumference():
    pts = [(0, 90), (0, 0), (180, 0), (0, -90)]
    kd = KDTree(pts, distance_metric="Arc", radius=sphere.RADIUS_EARTH_KM)

    d, _ = kd.query((90, 0), k=4)

    expected = 2 * math.pi * sphere.RADIUS_EARTH_KM / 4
    assert abs(d[0] - expected) < 1e-5


def test_arc_query_ball_point_raises_for_large_radius():
    pts = [(0, 0), (90, 0)]
    kd = KDTree(pts, distance_metric="Arc", radius=sphere.RADIUS_EARTH_KM)

    too_large_radius = kd.circumference

    with pytest.raises(ValueError):
        kd.query_ball_point((0, 0), r=too_large_radius)

"""
Computational geometry code for PySAL: Python Spatial Analysis Library.
"""

__author__ = "Sergio J. Rey, Xinyue Ye, Charles Schmidt, Andrew Winslow"
__credits__ = "Copyright (c) 2005-2011 Sergio J. Rey"

# ruff: noqa: B028, F403, F405

import copy
import math
import warnings

from .rtree import *
from .shapes import *
from .standalone import *

__all__ = ["Grid", "BruteForcePointLocator", "PointLocator", "PolygonLocator"]

dep_msg = "is deprecated and will be removed in a future version of libpysal"


class Grid:
    """
    Representation of a binning data structure.
    """

    def __init__(self, bounds, resolution):
        """
        Returns a grid with specified properties.

        __init__(Rectangle, number) -> Grid

        Parameters
        ----------
        bounds      : the area for the grid to encompass
        resolution  : the diameter of each bin

        Examples
        --------
        TODO: complete this doctest
        >>> g = Grid(Rectangle(0, 0, 10, 10), 1)
        """
        warnings.warn("Grid " + dep_msg, FutureWarning)
        if resolution == 0:
            raise Exception("Cannot create grid with resolution 0")
        self.res = resolution
        self.hash = {}
        self.x_range = (bounds.left, bounds.right)
        self.y_range = (bounds.lower, bounds.upper)
        try:
            self.i_range = int(
                math.ceil((self.x_range[1] - self.x_range[0]) / self.res)
            )
            self.j_range = int(
                math.ceil((self.y_range[1] - self.y_range[0]) / self.res)
            )
        except Exception as e:
            raise Exception(
                "Invalid arguments for Grid(): ("
                + str(x_range)
                + ", "
                + str(y_range)
                + ", "
                + str(res)
                + ")"
            ) from e

    def in_grid(self, loc):
        """
        Returns whether a 2-tuple location _loc_ lies inside the grid bounds.

        Test tag: <tc>#is#Grid.in_grid</tc>
        """
        return (
            self.x_range[0] <= loc[0] <= self.x_range[1]
            and self.y_range[0] <= loc[1] <= self.y_range[1]
        )

    def __grid_loc(self, loc):
        i = min(self.i_range, max(int((loc[0] - self.x_range[0]) / self.res), 0))
        j = min(self.j_range, max(int((loc[1] - self.y_range[0]) / self.res), 0))
        return (i, j)

    def add(self, item, pt):
        """
        Adds an item to the grid at a specified location.

        add(x, Point) -> x

        Parameters
        ----------
        item  : the item to insert into the grid
        pt : the location to insert the item at

        Examples
        --------
        >>> g = Grid(Rectangle(0, 0, 10, 10), 1)
        >>> g.add('A', Point((4.2, 8.7)))
        'A'
        """
        if not self.in_grid(pt):
            raise Exception(
                "Attempt to insert item at location outside grid bounds: " + str(pt)
            )
        grid_loc = self.__grid_loc(pt)
        if grid_loc in self.hash:
            self.hash[grid_loc].append((pt, item))
        else:
            self.hash[grid_loc] = [(pt, item)]
        return item

    def remove(self, item, pt):
        """
        Removes an item from the grid at a specified location.

        remove(x, Point) -> x

        Parameters
        ----------
        item : the item to remove from the grid
        pt : the location the item was added at

        Examples
        --------
        >>> g = Grid(Rectangle(0, 0, 10, 10), 1)
        >>> g.add('A', Point((4.2, 8.7)))
        'A'
        >>> g.remove('A', Point((4.2, 8.7)))
        'A'
        """
        if not self.in_grid(pt):
            raise Exception(
                "Attempt to remove item at location outside grid bounds: " + str(pt)
            )
        grid_loc = self.__grid_loc(pt)
        self.hash[grid_loc].remove((pt, item))
        if self.hash[grid_loc] == []:
            del self.hash[grid_loc]
        return item

    def bounds(self, bounds):
        """
        Returns a list of items found in the grid within the bounds specified.

        bounds(Rectangle) -> x list

        Parameters
        ----------
        item     : the item to remove from the grid
        pt       : the location the item was added at

        Examples
        --------
        >>> g = Grid(Rectangle(0, 0, 10, 10), 1)
        >>> g.add('A', Point((1.0, 1.0)))
        'A'
        >>> g.add('B', Point((4.0, 4.0)))
        'B'
        >>> g.bounds(Rectangle(0, 0, 3, 3))
        ['A']
        >>> g.bounds(Rectangle(2, 2, 5, 5))
        ['B']
        >>> sorted(g.bounds(Rectangle(0, 0, 5, 5)))
        ['A', 'B']
        """
        x_range = (bounds.left, bounds.right)
        y_range = (bounds.lower, bounds.upper)
        items = []
        lower_left = self.__grid_loc((x_range[0], y_range[0]))
        upper_right = self.__grid_loc((x_range[1], y_range[1]))
        for i in range(lower_left[0], upper_right[0] + 1):
            for j in range(lower_left[1], upper_right[1] + 1):
                if (i, j) in self.hash:
                    items.extend(
                        [
                            item[1]
                            for item in [
                                item
                                for item in self.hash[(i, j)]
                                if x_range[0] <= item[0][0] <= x_range[1]
                                and y_range[0] <= item[0][1] <= y_range[1]
                            ]
                        ]
                    )
        return items

    def proximity(self, pt, r):
        """
        Returns a list of items found in the grid
        within a specified distance of a point.

        proximity(Point, number) -> x list

        Parameters
        ----------
        pt : the location to search around
        r  : the distance to search around the point

        Examples
        --------
        >>> g = Grid(Rectangle(0, 0, 10, 10), 1)
        >>> g.add('A', Point((1.0, 1.0)))
        'A'
        >>> g.add('B', Point((4.0, 4.0)))
        'B'
        >>> g.proximity(Point((2.0, 1.0)), 2)
        ['A']
        >>> g.proximity(Point((6.0, 5.0)), 3.0)
        ['B']
        >>> sorted(g.proximity(Point((4.0, 1.0)), 4.0))
        ['A', 'B']
        """
        items = []
        lower_left = self.__grid_loc((pt[0] - r, pt[1] - r))
        upper_right = self.__grid_loc((pt[0] + r, pt[1] + r))
        for i in range(lower_left[0], upper_right[0] + 1):
            for j in range(lower_left[1], upper_right[1] + 1):
                if (i, j) in self.hash:
                    items.extend(
                        [
                            item[1]
                            for item in [
                                item
                                for item in self.hash[(i, j)]
                                if get_points_dist(pt, item[0]) <= r
                            ]
                        ]
                    )
        return items

    def nearest(self, pt):
        """
        Returns the nearest item to a point.

        nearest(Point) -> x

        Parameters
        ----------
        pt : the location to search near

        Examples
        --------
        >>> g = Grid(Rectangle(0, 0, 10, 10), 1)
        >>> g.add('A', Point((1.0, 1.0)))
        'A'
        >>> g.add('B', Point((4.0, 4.0)))
        'B'
        >>> g.nearest(Point((2.0, 1.0)))
        'A'
        >>> g.nearest(Point((7.0, 5.0)))
        'B'
        """
        search_size = self.res
        while self.proximity(pt, search_size) == [] and (
            get_points_dist((self.x_range[0], self.y_range[0]), pt) > search_size
            or get_points_dist((self.x_range[1], self.y_range[0]), pt) > search_size
            or get_points_dist((self.x_range[0], self.y_range[1]), pt) > search_size
            or get_points_dist((self.x_range[1], self.y_range[1]), pt) > search_size
        ):
            search_size = 2 * search_size
        items = []
        lower_left = self.__grid_loc((pt[0] - search_size, pt[1] - search_size))
        upper_right = self.__grid_loc((pt[0] + search_size, pt[1] + search_size))
        for i in range(lower_left[0], upper_right[0] + 1):
            for j in range(lower_left[1], upper_right[1] + 1):
                if (i, j) in self.hash:
                    items.extend(
                        [
                            (get_points_dist(pt, item[0]), item[1])
                            for item in self.hash[(i, j)]
                        ]
                    )
        if items == []:
            return None
        return min(items)[1]


class BruteForcePointLocator:
    """
    A class which does naive linear search on a set of Point objects.
    """

    def __init__(self, points):
        """
        Creates a naive index of the points specified.

        __init__(Point list) -> BruteForcePointLocator

        Parameters
        ----------
        points : a list of points to index (Point list)

        Examples
        --------
        >>> pl = BruteForcePointLocator([Point((0, 0)), Point((5, 0)), Point((0, 10))])
        """
        warnings.warn("BruteForcePointLocator " + dep_msg, FutureWarning)
        self._points = points

    def nearest(self, query_point):
        """
        Returns the nearest point indexed to a query point.

        nearest(Point) -> Point

        Parameters
        ----------
        query_point : a point to find the nearest indexed point to

        Examples
        --------
        >>> points = [Point((0, 0)), Point((1, 6)), Point((5.4, 1.4))]
        >>> pl = BruteForcePointLocator(points)
        >>> n = pl.nearest(Point((1, 1)))
        >>> str(n)
        '(0.0, 0.0)'
        """
        return min(self._points, key=lambda p: get_points_dist(p, query_point))

    def region(self, region_rect):
        """
        Returns the indexed points located inside a rectangular query region.

        region(Rectangle) -> Point list

        Parameters
        ----------
        region_rect : the rectangular range to find indexed points in

        Examples
        --------
        >>> points = [Point((0, 0)), Point((1, 6)), Point((5.4, 1.4))]
        >>> pl = BruteForcePointLocator(points)
        >>> pts = pl.region(Rectangle(-1, -1, 10, 10))
        >>> len(pts)
        3
        """
        return [
            p
            for p in self._points
            if get_rectangle_point_intersect(region_rect, p) is not None
        ]

    def proximity(self, origin, r):
        """
        Returns the indexed points located within some distance of an origin point.

        proximity(Point, number) -> Point list

        Parameters
        ----------
        origin  : the point to find indexed points near
        r       : the maximum distance to find indexed point from the origin point

        Examples
        --------
        >>> points = [Point((0, 0)), Point((1, 6)), Point((5.4, 1.4))]
        >>> pl = BruteForcePointLocator(points)
        >>> neighs = pl.proximity(Point((1, 0)), 2)
        >>> len(neighs)
        1
        >>> p = neighs[0]
        >>> isinstance(p, Point)
        True
        >>> str(p)
        '(0.0, 0.0)'
        """
        return [p for p in self._points if get_points_dist(p, origin) <= r]


class PointLocator:
    """
    An abstract representation of a point indexing data structure.
    """

    def __init__(self, points):
        """
        Returns a point locator object.

        __init__(Point list) -> PointLocator

        Parameters
        ----------
        points : a list of points to index

        Examples
        --------
        >>> points = [Point((0, 0)), Point((1, 6)), Point((5.4, 1.4))]
        >>> pl = PointLocator(points)
        """
        warnings.warn("PointLocator " + dep_msg, FutureWarning)
        self._locator = BruteForcePointLocator(points)

    def nearest(self, query_point):
        """
        Returns the nearest point indexed to a query point.

        nearest(Point) -> Point

        Parameters
        ----------
        query_point : a point to find the nearest indexed point to

        Examples
        --------
        >>> points = [Point((0, 0)), Point((1, 6)), Point((5.4, 1.4))]
        >>> pl = PointLocator(points)
        >>> n = pl.nearest(Point((1, 1)))
        >>> str(n)
        '(0.0, 0.0)'
        """
        return self._locator.nearest(query_point)

    def region(self, region_rect):
        """
        Returns the indexed points located inside a rectangular query region.

        region(Rectangle) -> Point list

        Parameters
        ----------
        region_rect : the rectangular range to find indexed points in

        Examples
        --------
        >>> points = [Point((0, 0)), Point((1, 6)), Point((5.4, 1.4))]
        >>> pl = PointLocator(points)
        >>> pts = pl.region(Rectangle(-1, -1, 10, 10))
        >>> len(pts)
        3
        """
        return self._locator.region(region_rect)

    overlapping = region

    def polygon(self, polygon):
        """
        Returns the indexed points located inside a polygon
        """

        # get points in polygon bounding box

        # for points in bounding box, check for inclusion in polygon

    def proximity(self, origin, r):
        """
        Returns the indexed points located within some distance of an origin point.

        proximity(Point, number) -> Point list

        Parameters
        ----------
        origin  : the point to find indexed points near
        r       : the maximum distance to find indexed point from the origin point

        Examples
        --------
        >>> points = [Point((0, 0)), Point((1, 6)), Point((5.4, 1.4))]
        >>> pl = PointLocator(points)
        >>> len(pl.proximity(Point((1, 0)), 2))
        1
        """
        return self._locator.proximity(origin, r)


class PolygonLocator:
    """
    An abstract representation of a polygon indexing data structure.
    """

    def __init__(self, polygons):
        """
        Returns a polygon locator object.

        __init__(Polygon list) -> PolygonLocator

        Parameters
        ----------
        polygons : a list of polygons to index

        Examples
        --------
        >>> p1 = Polygon([Point((0, 1)), Point((4, 5)), Point((5, 1))])
        >>> p2 = Polygon([Point((3, 9)), Point((6, 7)), Point((1, 1))])
        >>> pl = PolygonLocator([p1, p2])
        >>> isinstance(pl, PolygonLocator)
        True
        """
        warnings.warn("PolygonLocator " + dep_msg, FutureWarning)
        self._locator = polygons
        # create and rtree
        self._rtree = RTree()
        for polygon in polygons:
            x = polygon.bounding_box.left
            y = polygon.bounding_box.lower
            X = polygon.bounding_box.right
            Y = polygon.bounding_box.upper
            self._rtree.insert(polygon, Rect(x, y, X, Y))

    def inside(self, query_rectangle):
        """
        Returns polygons that are inside query_rectangle

        Examples
        --------
        >>> p1 = Polygon([Point((0, 1)), Point((4, 5)), Point((5, 1))])
        >>> p2 = Polygon([Point((3, 9)), Point((6, 7)), Point((1, 1))])
        >>> p3 = Polygon([Point((7, 1)), Point((8, 7)), Point((9, 1))])
        >>> pl = PolygonLocator([p1, p2, p3])
        >>> qr = Rectangle(0, 0, 5, 5)
        >>> res = pl.inside( qr )
        >>> len(res)
        1
        >>> qr = Rectangle(3, 7, 5, 8)
        >>> res = pl.inside( qr )
        >>> len(res)
        0
        >>> qr = Rectangle(10, 10, 12, 12)
        >>> res = pl.inside( qr )
        >>> len(res)
        0
        >>> qr = Rectangle(0, 0, 12, 12)
        >>> res = pl.inside( qr )
        >>> len(res)
        3

        Notes
        -----
        inside means the intersection of the query rectangle and a
        polygon is not empty and is equal to the area of the polygon
        """
        left = query_rectangle.left
        right = query_rectangle.right
        upper = query_rectangle.upper
        lower = query_rectangle.lower

        # rtree rect
        qr = Rect(left, lower, right, upper)
        # bb overlaps
        res = [r.leaf_obj() for r in self._rtree.query_rect(qr) if r.is_leaf()]

        qp = Polygon(
            [
                Point((left, lower)),
                Point((right, lower)),
                Point((right, upper)),
                Point((left, upper)),
            ]
        )
        ip = []
        GPPI = get_polygon_point_intersect
        for poly in res:
            lower = poly.bounding_box.lower
            right = poly.bounding_box.right
            upper = poly.bounding_box.upper
            left = poly.bounding_box.left
            p1 = Point((left, lower))
            p2 = Point((right, upper))
            if GPPI(qp, p1) and GPPI(qp, p2):
                ip.append(poly)
        return ip

    def overlapping(self, query_rectangle):
        """
        Returns list of polygons that overlap query_rectangle

        Examples
        --------
        >>> p1 = Polygon([Point((0, 1)), Point((4, 5)), Point((5, 1))])
        >>> p2 = Polygon([Point((3, 9)), Point((6, 7)), Point((1, 1))])
        >>> p3 = Polygon([Point((7, 1)), Point((8, 7)), Point((9, 1))])
        >>> pl = PolygonLocator([p1, p2, p3])
        >>> qr = Rectangle(0, 0, 5, 5)
        >>> res = pl.overlapping( qr )
        >>> len(res)
        2
        >>> qr = Rectangle(3, 7, 5, 8)
        >>> res = pl.overlapping( qr )
        >>> len(res)
        1
        >>> qr = Rectangle(10, 10, 12, 12)
        >>> res = pl.overlapping( qr )
        >>> len(res)
        0
        >>> qr = Rectangle(0, 0, 12, 12)
        >>> res = pl.overlapping( qr )
        >>> len(res)
        3
        >>> qr = Rectangle(8, 3, 9, 4)
        >>> p1 = Polygon([Point((2, 1)), Point((2, 3)), Point((4, 3)), Point((4,1))])
        >>> p2 = Polygon([Point((7, 1)), Point((7, 5)), Point((10, 5)), Point((10, 1))])
        >>> pl = PolygonLocator([p1, p2])
        >>> res = pl.overlapping(qr)
        >>> len(res)
        1

        Notes
        -----
        overlapping means the intersection of the query rectangle and a
        polygon is not empty and is no larger than the area of the polygon
        """
        left = query_rectangle.left
        right = query_rectangle.right
        upper = query_rectangle.upper
        lower = query_rectangle.lower

        # rtree rect
        qr = Rect(left, lower, right, upper)

        # bb overlaps
        res = [r.leaf_obj() for r in self._rtree.query_rect(qr) if r.is_leaf()]
        # have to check for polygon overlap using segment intersection

        # add polys whose bb contains at least one of the corners of the query
        # rectangle

        sw = (left, lower)
        se = (right, lower)
        ne = (right, upper)
        nw = (left, upper)
        pnts = [sw, se, ne, nw]
        cs = []
        for pnt in pnts:
            c = [r.leaf_obj() for r in self._rtree.query_point(pnt) if r.is_leaf()]
            cs.extend(c)

        cs = list(set(cs))

        overlapping = []

        # first find polygons with at least one vertex inside query rectangle
        remaining = copy.copy(res)
        for polygon in res:
            vertices = polygon.vertices
            for vertex in vertices:
                xb = vertex[0] >= left
                xb *= vertex[0] < right
                yb = vertex[1] >= lower
                yb *= vertex[1] < upper
                if xb * yb:
                    overlapping.append(polygon)
                    remaining.remove(polygon)
                    break

        # for remaining polys in bb overlap check if vertex chains intersect
        # segments of the query rectangle
        left_edge = LineSegment(Point((left, lower)), Point((left, upper)))
        right_edge = LineSegment(Point((right, lower)), Point((right, upper)))
        lower_edge = LineSegment(Point((left, lower)), Point((right, lower)))
        upper_edge = LineSegment(Point((left, upper)), Point((right, upper)))
        for polygon in remaining:
            vertices = copy.copy(polygon.vertices)
            if vertices[-1] != vertices[0]:
                vertices.append(vertices[0])  # put on closed cartographic form
            nv = len(vertices)
            for i in range(nv - 1):
                head = vertices[i]
                tail = vertices[i + 1]
                edge = LineSegment(head, tail)
                li = get_segments_intersect(edge, left_edge)
                if (
                    li
                    or get_segments_intersect(edge, right_edge)
                    or get_segments_intersect(edge, lower_edge)
                    or get_segments_intersect(edge, upper_edge)
                ):
                    overlapping.append(polygon)
                    break
        # check remaining for explicit containment of the bounding rectangle
        # cs has candidates for this check
        sw = Point(sw)
        se = Point(se)
        ne = Point(ne)
        nw = Point(nw)
        for polygon in cs:
            if (
                get_polygon_point_intersect(polygon, sw)
                or get_polygon_point_intersect(polygon, se)
                or get_polygon_point_intersect(polygon, ne)
                or get_polygon_point_intersect(polygon, nw)
            ):
                overlapping.append(polygon)
                break
        return list(set(overlapping))

    def nearest(self, query_point, rule="vertex"):
        """
        Returns the nearest polygon indexed to a query point based on
        various rules.

        nearest(Polygon) -> Polygon

        Parameters
        ----------
        query_point  : a point to find the nearest indexed polygon to

        rule         : representative point for polygon in nearest query.
                 vertex -- measures distance between vertices and query_point
                 centroid -- measures distance between centroid and
                 query_point
                 edge   -- measures the distance between edges and query_point

        Examples
        --------
        >>> p1 = Polygon([Point((0, 1)), Point((4, 5)), Point((5, 1))])
        >>> p2 = Polygon([Point((3, 9)), Point((6, 7)), Point((1, 1))])
        >>> pl = PolygonLocator([p1, p2])
        >>> try: n = pl.nearest(Point((-1, 1)))
        ... except NotImplementedError: print("future test: str(min(n.vertices())) == (0.0, 1.0)")
        future test: str(min(n.vertices())) == (0.0, 1.0)
        """  # noqa E501
        raise NotImplementedError

    def region(self, region_rect):
        """
        Returns the indexed polygons located inside a rectangular query region.

        region(Rectangle) -> Polygon list

        Parameters
        ----------
        region_rect  : the rectangular range to find indexed polygons in

        Examples
        --------
        >>> p1 = Polygon([Point((0, 1)), Point((4, 5)), Point((5, 1))])
        >>> p2 = Polygon([Point((3, 9)), Point((6, 7)), Point((1, 1))])
        >>> pl = PolygonLocator([p1, p2])
        >>> n = pl.region(Rectangle(0, 0, 4, 10))
        >>> len(n)
        2
        """
        n = self._locator
        for polygon in n:
            points = polygon.vertices
            pl = BruteForcePointLocator(points)
            pts = pl.region(region_rect)
            if len(pts) == 0:
                n.remove(polygon)
        return n

    def contains_point(self, point):
        """
        Returns polygons that contain point

        Parameters
        ----------
        point: point (x,y)

        Returns
        -------
        list of polygons containing point

        Examples
        --------
        >>> p1 = Polygon([Point((0,0)), Point((6,0)), Point((4,4))])
        >>> p2 = Polygon([Point((1,2)), Point((4,0)), Point((4,4))])
        >>> p1.contains_point((2,2))
        1
        >>> p2.contains_point((2,2))
        1
        >>> pl = PolygonLocator([p1, p2])
        >>> len(pl.contains_point((2,2)))
        2
        >>> p2.contains_point((1,1))
        0
        >>> p1.contains_point((1,1))
        1
        >>> len(pl.contains_point((1,1)))
        1
        >>> p1.centroid
        (3.3333333333333335, 1.3333333333333333)
        >>> pl.contains_point((1,1))[0].centroid
        (3.3333333333333335, 1.3333333333333333)
        """
        # bbounding box containment
        res = [r.leaf_obj() for r in self._rtree.query_point(point) if r.is_leaf()]
        # explicit containment check for candidate polygons needed
        return [poly for poly in res if poly.contains_point(point)]

    def proximity(self, origin, r, rule="vertex"):
        """
        Returns the indexed polygons located within some distance of an
        origin point based on various rules.

        proximity(Polygon, number) -> Polygon list

        Parameters
        ----------
        origin  : the point to find indexed polygons near
        r       : the maximum distance to find indexed polygon from the origin point

        rule    : representative point for polygon in nearest query.
                vertex -- measures distance between vertices and query_point
                centroid -- measures distance between centroid and
                query_point
                edge   -- measures the distance between edges and query_point

        Examples
        --------
        >>> p1 = Polygon([Point((0, 1)), Point((4, 5)), Point((5, 1))])
        >>> p2 = Polygon([Point((3, 9)), Point((6, 7)), Point((1, 1))])
        >>> pl = PolygonLocator([p1, p2])
        >>> try:
        ...     len(pl.proximity(Point((0, 0)), 2))
        ... except NotImplementedError:
        ...     print("future test: len(pl.proximity(Point((0, 0)), 2)) == 2")
        future test: len(pl.proximity(Point((0, 0)), 2)) == 2
        """
        raise NotImplementedError

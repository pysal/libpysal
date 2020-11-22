"""
Computational geometry code for PySAL: Python Spatial Analysis Library.

"""

__author__ = "Sergio J. Rey, Xinyue Ye, Charles Schmidt, Andrew Winslow, Hu Shao"

import math
from .sphere import arcdist

from typing import Union

__all__ = [
    "Point",
    "LineSegment",
    "Line",
    "Ray",
    "Chain",
    "Polygon",
    "Rectangle",
    "asShape",
]


def asShape(obj):
    """Returns a PySAL shape object from ``obj``, which
    must support the ``__geo_interface__``.
    
    Parameters
    ----------
    obj : {libpysal.cg.{Point, LineSegment, Line, Ray, Chain, Polygon}
        A geometric representation of an object.
    
    Raises
    ------
    TypeError
        Raised when ``obj`` is not a supported shape.
    NotImplementedError
        Raised when ``geo_type`` is not a supported type.
    
    Returns
    -------
    obj : {libpysal.cg.{Point, LineSegment, Line, Ray, Chain, Polygon}
        A new geometric representation of the object.
    
    """

    if isinstance(obj, (Point, LineSegment, Line, Ray, Chain, Polygon)):
        pass
    else:
        if hasattr(obj, "__geo_interface__"):
            geo = obj.__geo_interface__
        else:
            geo = obj

        if hasattr(geo, "type"):
            raise TypeError("%r does not appear to be a shape object." % (obj))

        geo_type = geo["type"].lower()

        # if geo_type.startswith('multi'):
        #    raise NotImplementedError, "%s are not supported at this time."%geo_type

        if geo_type in _geoJSON_type_to_Pysal_type:

            obj = _geoJSON_type_to_Pysal_type[geo_type].__from_geo_interface__(geo)
        else:
            raise NotImplementedError("%s is not supported at this time." % geo_type)

    return obj


class Geometry(object):
    """A base class to help implement ``is_geometry``
    and make geometric types extendable.
    
    """

    def __init__(self):
        pass


class Point(Geometry):
    """Geometric class for point objects.

    Parameters
    ----------
    loc : tuple
        The point's location (number :math:`x`-tuple, :math:`x` > 1).
    
    Examples
    --------
    
    >>> p = Point((1, 3))
    
    """

    def __init__(self, loc):

        self.__loc = tuple(map(float, loc))

    @classmethod
    def __from_geo_interface__(cls, geo):
        return cls(geo["coordinates"])

    @property
    def __geo_interface__(self):
        return {"type": "Point", "coordinates": self.__loc}

    def __lt__(self, other) -> bool:
        """Tests if the point is less than another object.

        Parameters
        ----------
        other : libpysal.cg.Point
            An object to test equality against.

        Examples
        --------
        
        >>> Point((0, 1)) < Point((0, 1))
        False
        
        >>> Point((0, 1)) < Point((1, 1))
        True
        
        """

        return (self.__loc) < (other.__loc)

    def __le__(self, other) -> bool:
        """Tests if the point is less than or equal to another object.

        Parameters
        ----------
        other : libpysal.cg.Point
            An object to test equality against.

        Examples
        --------
        
        >>> Point((0, 1)) <= Point((0, 1))
        True
        
        >>> Point((0, 1)) <= Point((1, 1))
        True
        
        """

        return (self.__loc) <= (other.__loc)

    def __eq__(self, other) -> bool:
        """Tests if the point is equal to another object.

        Parameters
        ----------
        other : libpysal.cg.Point
            An object to test equality against.
        
        Examples
        --------
        
        >>> Point((0, 1)) == Point((0, 1))
        True
        
        >>> Point((0, 1)) == Point((1, 1))
        False
        
        """

        try:
            return (self.__loc) == (other.__loc)
        except AttributeError:
            return False

    def __ne__(self, other) -> bool:
        """Tests if the point is not equal to another object.

        Parameters
        ----------
        other : libpysal.cg.Point
            An object to test equality against.

        Examples
        --------
        
        >>> Point((0, 1)) != Point((0, 1))
        False
        
        >>> Point((0, 1)) != Point((1, 1))
        True
        
        """

        try:
            return (self.__loc) != (other.__loc)
        except AttributeError:
            return True

    def __gt__(self, other) -> bool:
        """Tests if the point is greater than another object.

        Parameters
        ----------
        other : libpysal.cg.Point
            An object to test equality against.

        Examples
        --------
        
        >>> Point((0, 1)) > Point((0, 1))
        False
        
        >>> Point((0, 1)) > Point((1, 1))
        False
        
        """

        return (self.__loc) > (other.__loc)

    def __ge__(self, other) -> bool:
        """Tests if the point is greater than or equal to another object.

        Parameters
        ----------
        other : libpysal.cg.Point
            An object to test equality against.

        Examples
        --------
        
        >>> Point((0, 1)) >= Point((0, 1))
        True
        
        >>> Point((0, 1)) >= Point((1, 1))
        False
        
        """

        return (self.__loc) >= (other.__loc)

    def __hash__(self) -> int:
        """Returns the hash of the point's location.

        Examples
        --------
        
        >>> hash(Point((0, 1))) == hash(Point((0, 1)))
        True
        
        >>> hash(Point((0, 1))) == hash(Point((1, 1)))
        False
        
        """

        return hash(self.__loc)

    def __getitem__(self, *args) -> Union[int, float]:
        """Return the coordinate for the given dimension.
        
        Parameters
        ----------
        *args : tuple
            A singleton tuple of :math:`(i)` with :math:`i`
            as the index of the desired dimension.

        Examples
        --------
        
        >>> p = Point((5.5, 4.3))
        >>> p[0] == 5.5
        True
        >>> p[1] == 4.3
        True
        
        """

        return self.__loc.__getitem__(*args)

    def __getslice__(self, *args) -> slice:
        """Return the coordinates for the given dimensions.

        Parameters
        ----------
        *args : tuple
            A tuple of :math:`(i,j)` with :math:`i` as the index to the start
            slice and :math:`j` as the index to end the slice (excluded).

        Examples
        --------
        
        >>> p = Point((3, 6, 2))
        >>> p[:2] == (3, 6)
        True
        
        >>> p[1:2] == (6,)
        True
        
        """

        return self.__loc.__getslice__(*args)

    def __len__(self) -> int:
        """ Returns the dimensions of the point.

        Examples
        --------
        
        >>> len(Point((1, 2)))
        2
        
        """

        return len(self.__loc)

    def __repr__(self) -> str:
        """Returns the string representation of the ``Point``.

        Examples
        --------
        
        >>> Point((0, 1))
        (0.0, 1.0)
        
        """

        return str(self)

    def __str__(self) -> str:
        """Returns a string representation of a ``Point`` object.

        Examples
        --------
        
        >>> p = Point((1, 3))
        >>> str(p)
        '(1.0, 3.0)'
        
        """

        return str(self.__loc)
        # return "POINT ({} {})".format(*self.__loc)


class LineSegment(Geometry):
    """Geometric representation of line segment objects.

    Parameters
    ----------
    start_pt : libpysal.cg.Point
        The point where the segment begins.
    end_pt : libpysal.cg.Point
        The point where the segment ends.

    Attributes
    ----------
    p1 : libpysal.cg.Point
        The starting point of the line segment.
    p2 : Point
        The ending point of the line segment.
    bounding_box : libpysal.cg.Rectangle
        The bounding box of the segment.
    len : float
        The length of the segment.
    line : libpysal.cg.Line
        The line on which the segment lies.

    Examples
    --------
    
    >>> ls = LineSegment(Point((1, 2)), Point((5, 6)))

    """

    def __init__(self, start_pt, end_pt):

        self._p1 = start_pt
        self._p2 = end_pt
        self._reset_props()

    def __str__(self):
        return "LineSegment(" + str(self._p1) + ", " + str(self._p2) + ")"
        # return "LINESTRING ({} {}, {} {})".format(
        #    self._p1[0], self._p1[1], self._p2[0], self._p2[1]
        # )

    def __eq__(self, other) -> bool:
        """Returns ``True`` if ``self`` and ``other`` are the same line segment.

        Examples
        --------
        
        >>> l1 = LineSegment(Point((1, 2)), Point((5, 6)))
        >>> l2 = LineSegment(Point((5, 6)), Point((1, 2)))
        >>> l1 == l2
        True
        
        >>> l2 == l1
        True
        
        """

        eq = False

        if not isinstance(other, self.__class__):
            pass
        else:
            if other.p1 == self._p1 and other.p2 == self._p2:
                eq = True
            elif other.p2 == self._p1 and other.p1 == self._p2:
                eq = True

        return eq

    def intersect(self, other) -> bool:
        """Test whether segment intersects with other segment (``True``) or
        not (``False``). Handles endpoints of segments being on other segment.
        
        Parameters
        ----------
        other : libpysal.cg.LineSegment
            Another line segment to check against.
        
        Examples
        --------

        >>> ls = LineSegment(Point((5, 0)), Point((10, 0)))
        >>> ls1 = LineSegment(Point((5, 0)), Point((10, 1)))
        >>> ls.intersect(ls1)
        True
        
        >>> ls2 = LineSegment(Point((5, 1)), Point((10, 1)))
        >>> ls.intersect(ls2)
        False
        
        >>> ls2 = LineSegment(Point((7, -1)), Point((7, 2)))
        >>> ls.intersect(ls2)
        True
        
        """

        ccw1 = self.sw_ccw(other.p2)
        ccw2 = self.sw_ccw(other.p1)
        ccw3 = other.sw_ccw(self.p1)
        ccw4 = other.sw_ccw(self.p2)

        intersects = ccw1 * ccw2 <= 0 and ccw3 * ccw4 <= 0

        return intersects

    def _reset_props(self):
        """**HELPER METHOD. DO NOT CALL.**
        Resets attributes which are functions of other attributes.
        The getters for these attributes (implemented as properties)
        then recompute their values if they have been reset since
        the last call to the getter.

        Examples
        --------
        
        >>> ls = LineSegment(Point((1, 2)), Point((5, 6)))
        >>> ls._reset_props()
        
        """

        self._bounding_box = None
        self._len = None
        self._line = False

    def _get_p1(self):
        """**HELPER METHOD. DO NOT CALL.**
        Returns the ``p1`` attribute of the line segment.

        Returns
        -------
        self._p1 : libpysal.cg.Point
            The ``_p1`` attribute.

        Examples
        --------
        
        >>> ls = LineSegment(Point((1, 2)), Point((5, 6)))
        >>> r = ls._get_p1()
        >>> r == Point((1, 2))
        True
        
        """

        return self._p1

    def _set_p1(self, p1):
        """**HELPER METHOD. DO NOT CALL.**
        Sets the ``p1`` attribute of the line segment.
        
        Parameters
        ----------
        p1 : libpysal.cg.Point
            A point.
        
        Returns
        -------
        self._p1 : libpysal.cg.Point
            The reset ``p1`` attribute.

        Examples
        --------
        
        >>> ls = LineSegment(Point((1, 2)), Point((5, 6)))
        >>> r = ls._set_p1(Point((3, -1)))
        >>> r == Point((3.0, -1.0))
        True
        
        """

        self._p1 = p1
        self._reset_props()

        return self._p1

    p1 = property(_get_p1, _set_p1)

    def _get_p2(self):
        """**HELPER METHOD. DO NOT CALL.**
        Returns the ``p2`` attribute of the line segment.

        Returns
        -------
        self._p2 : libpysal.cg.Point
            The ``_p2`` attribute.

        Examples
        --------
        
        >>> ls = LineSegment(Point((1, 2)), Point((5, 6)))
        >>> r = ls._get_p2()
        >>> r == Point((5, 6))
        True
        
        """

        return self._p2

    def _set_p2(self, p2):
        """**HELPER METHOD. DO NOT CALL.**
        Sets the ``p2`` attribute of the line segment.

        Parameters
        ----------
        p2 : libpysal.cg.Point
            A point.
        
        Returns
        -------
        self._p2 : libpysal.cg.Point
            The reset ``p2`` attribute.

        Examples
        --------
        
        >>> ls = LineSegment(Point((1, 2)), Point((5, 6)))
        >>> r = ls._set_p2(Point((3, -1)))
        >>> r == Point((3.0, -1.0))
        True
        
        """

        self._p2 = p2
        self._reset_props()

        return self._p2

    p2 = property(_get_p2, _set_p2)

    def is_ccw(self, pt) -> bool:
        """Returns whether a point is counterclockwise of the
        segment (``True``) or not (``False``). Exclusive.

        Parameters
        ----------
        pt : libpysal.cg.Point
            A point lying ccw or cw of a segment.

        Examples
        --------
        
        >>> ls = LineSegment(Point((0, 0)), Point((5, 0)))
        >>> ls.is_ccw(Point((2, 2)))
        True
        
        >>> ls.is_ccw(Point((2, -2)))
        False
        
        """

        v1 = (self._p2[0] - self._p1[0], self._p2[1] - self._p1[1])
        v2 = (pt[0] - self._p1[0], pt[1] - self._p1[1])

        return v1[0] * v2[1] - v1[1] * v2[0] > 0

    def is_cw(self, pt) -> bool:
        """Returns whether a point is clockwise of the
        segment (``True``) or not (``False``). Exclusive.

        Parameters
        ----------
        pt : libpysal.cg.Point
            A point lying ccw or cw of a segment.

        Examples
        --------
        
        >>> ls = LineSegment(Point((0, 0)), Point((5, 0)))
        >>> ls.is_cw(Point((2, 2)))
        False
        
        >>> ls.is_cw(Point((2, -2)))
        True
        
        """

        v1 = (self._p2[0] - self._p1[0], self._p2[1] - self._p1[1])
        v2 = (pt[0] - self._p1[0], pt[1] - self._p1[1])

        return v1[0] * v2[1] - v1[1] * v2[0] < 0

    def sw_ccw(self, pt):
        """Sedgewick test for ``pt`` being ccw of segment.

        Returns
        -------
        is_ccw : bool
            ``1`` if turn from ``self.p1`` to ``self.p2`` to ``pt`` is ccw.
            ``-1`` if turn from ``self.p1`` to ``self.p2`` to ``pt`` is cw.
            ``-1`` if the points are collinear and ``self.p1`` is in the middle.
            ``1`` if the points are collinear and ``self.p2`` is in the middle.
            ``0`` if the points are collinear and ``pt`` is in the middle.

        """

        p0 = self.p1
        p1 = self.p2
        p2 = pt

        dx1 = p1[0] - p0[0]
        dy1 = p1[1] - p0[1]
        dx2 = p2[0] - p0[0]
        dy2 = p2[1] - p0[1]

        if dy1 * dx2 < dy2 * dx1:
            is_ccw = 1
        elif dy1 * dx2 > dy2 * dx1:
            is_ccw = -1
        elif dx1 * dx2 < 0 or dy1 * dy2 < 0:
            is_ccw = -1
        elif dx1 * dx1 + dy1 * dy1 >= dx2 * dx2 + dy2 * dy2:
            is_ccw = 0
        else:
            is_ccw = 1

        return is_ccw

    def get_swap(self):
        """Returns a ``LineSegment`` object which has its endpoints swapped.

        Returns
        -------
        line_seg : libpysal.cg.LineSegment
            The ``LineSegment`` object which has its endpoints swapped.

        Examples
        --------
        
        >>> ls = LineSegment(Point((1, 2)), Point((5, 6)))
        >>> swap = ls.get_swap()
        >>> swap.p1[0]
        5.0
        
        >>> swap.p1[1]
        6.0
        
        >>> swap.p2[0]
        1.0
        
        >>> swap.p2[1]
        2.0
        
        """

        line_seg = LineSegment(self._p2, self._p1)

        return line_seg

    @property
    def bounding_box(self):
        """Returns the minimum bounding box of a ``LineSegment`` object.
        
        Returns
        -------
        self._bounding_box : libpysal.cg.Rectangle
            The bounding box of the line segment.
        
        Examples
        --------
        
        >>> ls = LineSegment(Point((1, 2)), Point((5, 6)))
        >>> ls.bounding_box.left
        1.0
        
        >>> ls.bounding_box.lower
        2.0
        
        >>> ls.bounding_box.right
        5.0
        
        >>> ls.bounding_box.upper
        6.0
        
        """

        # If LineSegment attributes p1, p2 changed, recompute
        if self._bounding_box is None:
            self._bounding_box = Rectangle(
                min([self._p1[0], self._p2[0]]),
                min([self._p1[1], self._p2[1]]),
                max([self._p1[0], self._p2[0]]),
                max([self._p1[1], self._p2[1]]),
            )
        return Rectangle(
            self._bounding_box.left,
            self._bounding_box.lower,
            self._bounding_box.right,
            self._bounding_box.upper,
        )

    @property
    def len(self) -> float:
        """Returns the length of a ``LineSegment`` object.

        Examples
        --------
        
        >>> ls = LineSegment(Point((2, 2)), Point((5, 2)))
        >>> ls.len
        3.0
        
        """

        # If LineSegment attributes p1, p2 changed, recompute
        if self._len is None:
            self._len = math.hypot(self._p1[0] - self._p2[0], self._p1[1] - self._p2[1])

        return self._len

    @property
    def line(self):
        """Returns a ``Line`` object of the line on which the segment lies.

        Returns
        -------
        self._line : libpysal.cg.Line
            The ``Line`` object of the line on which the segment lies.

        Examples
        --------
        
        >>> ls = LineSegment(Point((2, 2)), Point((3, 3)))
        >>> l = ls.line
        >>> l.m
        1.0
        
        >>> l.b
        0.0
        
        """

        if self._line == False:
            dx = self._p1[0] - self._p2[0]
            dy = self._p1[1] - self._p2[1]

            if dx == 0 and dy == 0:
                self._line = None
            elif dx == 0:
                self._line = VerticalLine(self._p1[0])
            else:
                m = dy / float(dx)
                # y - mx
                b = self._p1[1] - m * self._p1[0]
                self._line = Line(m, b)

        return self._line


class VerticalLine(Geometry):
    """Geometric representation of verticle line objects.
    
    Parameters
    ----------
    x : {int, float}
        The :math:`x`-intercept of the line. ``x`` is also an attribute.
    
    Examples
    --------
    
    >>> ls = VerticalLine(0)
    >>> ls.m
    inf
    
    >>> ls.b
    nan
    
    """

    def __init__(self, x):

        self._x = float(x)
        self.m = float("inf")
        self.b = float("nan")

    def x(self, y) -> float:
        """Returns the :math:`x`-value of the line at a particular :math:`y`-value.

        Parameters
        ----------
        y : {int, float}
            The :math:`y`-value at which to compute :math:`x`.

        Examples
        --------
        
        >>> l = VerticalLine(0)
        >>> l.x(0.25)
        0.0
        
        """

        return self._x

    def y(self, x) -> float:
        """Returns the :math:`y`-value of the line at a particular :math:`x`-value.

        Parameters
        ----------
        x : {int, float}
            The :math:`x`-value at which to compute :math:`y`.

        Examples
        --------
        
        >>> l = VerticalLine(1)
        >>> l.y(1)
        nan
        
        """

        return float("nan")


class Line(Geometry):
    """Geometric representation of line objects.

    Parameters
    ----------
    m : {int, float}
        The slope of the line. ``m`` is also an attribute.
    b : {int, float}
        The :math:`y`-intercept of the line. ``b`` is also an attribute.
    
    Raises
    ------
    ArithmeticError
        Raised when infinity is passed in as the slope.
    
    Examples
    --------
    
    >>> ls = Line(1, 0)
    >>> ls.m
    1.0
    
    >>> ls.b
    0.0
    
    """

    def __init__(self, m, b):

        if m == float("inf"):
            raise ArithmeticError("Slope cannot be infinite.")

        self.m = float(m)
        self.b = float(b)

    def x(self, y: Union[int, float]) -> float:
        """Returns the :math:`x`-value of the line at a particular :math:`y`-value.

        Parameters
        ----------
        y : {int, float}
            The :math:`y`-value at which to compute :math:`x`.
        
        Raises
        ------
        ArithmeticError
            Raised when ``0.`` is passed in as the slope.
        
        Examples
        --------
        
        >>> l = Line(0.5, 0)
        >>> l.x(0.25)
        0.5
        
        """

        if self.m == 0:
            raise ArithmeticError("Cannot solve for 'x' when slope is zero.")

        return (y - self.b) / self.m

    def y(self, x: Union[int, float]) -> float:
        """Returns the :math:`y`-value of the line at a particular :math:`x`-value.

        Parameters
        ----------
        x : {int, float}
            The :math:`x`-value at which to compute :math:`y`.

        Examples
        --------
        
        >>> l = Line(1, 0)
        >>> l.y(1)
        1.0
        
        """

        if self.m == 0:
            return self.b

        return self.m * x + self.b


class Ray:
    """Geometric representation of ray objects.

    Parameters
    ----------
    origin : libpysal.cg.Point
        The point where the ray originates.
    second_p :
        The second point specifying the ray (not ``origin``.)

    Attributes
    ----------
    o : libpysal.cg.Point
        The origin (point where ray originates). See ``origin``.
    p : libpysal.cg.Point
        The second point on the ray (not the point where the
        ray originates). See ``second_p``.
    
    Examples
    --------
    
    >>> l = Ray(Point((0, 0)), Point((1, 0)))
    >>> str(l.o)
    '(0.0, 0.0)'
    
    >>> str(l.p)
    '(1.0, 0.0)'
    
    """

    def __init__(self, origin, second_p):

        self.o = origin
        self.p = second_p


class Chain(Geometry):
    """Geometric representation of a chain, also known as a polyline.
    
    Parameters
    ----------
    vertices : list
        A point list or list of point lists.

    Attributes
    ----------
    vertices : list
        The list of points of the vertices of the chain in order.
    len : float
        The geometric length of the chain.
    
    Examples
    --------
    
    >>> c = Chain([Point((0, 0)), Point((1, 0)), Point((1, 1)), Point((2, 1))])
    
    """

    def __init__(self, vertices: list):

        if isinstance(vertices[0], list):
            self._vertices = [part for part in vertices]
        else:
            self._vertices = [vertices]
        self._reset_props()

    @classmethod
    def __from_geo_interface__(cls, geo: dict):
        if geo["type"].lower() == "linestring":
            verts = [Point(pt) for pt in geo["coordinates"]]
        elif geo["type"].lower() == "multilinestring":
            verts = [list(map(Point, part)) for part in geo["coordinates"]]
        else:
            raise TypeError("%r is not a Chain." % geo)
        return cls(verts)

    @property
    def __geo_interface__(self) -> dict:
        if len(self.parts) == 1:
            return {"type": "LineString", "coordinates": self.vertices}
        else:
            return {"type": "MultiLineString", "coordinates": self.parts}

    def _reset_props(self):
        """**HELPER METHOD. DO NOT CALL.** Resets attributes which are
        functions of other attributes. The ``getter``s for these attributes
        (implemented as ``properties``) then recompute their values if they
        have been reset since the last call to the ``getter``.

        """

        self._len = None
        self._arclen = None
        self._bounding_box = None

    @property
    def vertices(self) -> list:
        """Returns the vertices of the chain in clockwise order.

        Examples
        --------
        
        >>> c = Chain([Point((0, 0)), Point((1, 0)), Point((1, 1)), Point((2, 1))])
        >>> verts = c.vertices
        >>> len(verts)
        4
        
        """

        return sum([part for part in self._vertices], [])

    @property
    def parts(self) -> list:
        """Returns the parts (lists of ``libpysal.cg.Point`` objects) of the chain.

        Examples
        --------
       
        >>> c = Chain(
        ...     [
        ...         [Point((0, 0)), Point((1, 0)), Point((1, 1)), Point((0, 1))],
        ...         [Point((2, 1)), Point((2, 2)), Point((1, 2)), Point((1, 1))]
        ...     ]
        ... )
        >>> len(c.parts)
        2
        
        """

        return [[v for v in part] for part in self._vertices]

    @property
    def bounding_box(self):
        """Returns the bounding box of the chain.

        Returns
        -------
        self._bounding_box : libpysal.cg.Rectangle
            The bounding box of the chain.
        
        Examples
        --------
        
        >>> c = Chain([Point((0, 0)), Point((2, 0)), Point((2, 1)), Point((0, 1))])
        >>> c.bounding_box.left
        0.0
        
        >>> c.bounding_box.lower
        0.0
        
        >>> c.bounding_box.right
        2.0
        
        >>> c.bounding_box.upper
        1.0
        
        """

        if self._bounding_box is None:
            vertices = self.vertices
            self._bounding_box = Rectangle(
                min([v[0] for v in vertices]),
                min([v[1] for v in vertices]),
                max([v[0] for v in vertices]),
                max([v[1] for v in vertices]),
            )

        return self._bounding_box

    @property
    def len(self) -> int:
        """Returns the geometric length of the chain.

        Examples
        --------
        
        >>> c = Chain([Point((0, 0)), Point((1, 0)), Point((1, 1)), Point((2, 1))])
        >>> c.len
        3.0
        
        >>> c = Chain(
        ...     [
        ...         [Point((0, 0)), Point((1, 0)), Point((1, 1))],
        ...         [Point((10, 10)), Point((11, 10)), Point((11, 11))]
        ...     ]
        ... )
        >>> c.len
        4.0
        
        """

        def dist(v1: tuple, v2: tuple) -> Union[int, float]:
            return math.hypot(v1[0] - v2[0], v1[1] - v2[1])

        def part_perimeter(p: list) -> Union[int, float]:
            return sum([dist(p[i], p[i + 1]) for i in range(len(p) - 1)])

        if self._len is None:
            self._len = sum([part_perimeter(part) for part in self._vertices])

        return self._len

    @property
    def arclen(self) -> Union[int, float]:
        """Returns the geometric length of the chain
        computed using 'arcdistance' (meters).
        
        """

        def part_perimeter(p: list) -> Union[int, float]:
            return sum([arcdist(p[i], p[i + 1]) * 1000.0 for i in range(len(p) - 1)])

        if self._arclen is None:
            self._arclen = sum([part_perimeter(part) for part in self._vertices])

        return self._arclen

    @property
    def segments(self) -> list:
        """Returns the segments that compose the chain."""

        return [
            [LineSegment(a, b) for (a, b) in zip(part[:-1], part[1:])]
            for part in self._vertices
        ]


class Ring(Geometry):
    """Geometric representation of a linear ring. Linear rings must be
    closed, the first and last point must be the same. Open rings will
    be closed. This class exists primarily as a geometric primitive to
    form complex polygons with multiple rings and holes. The ordering
    of the vertices is ignored and will not be altered.

    Parameters
    ----------
    vertices : list
        A list of vertices.

    Attributes
    ----------
    vertices : list
        A list of points with the vertices of the ring.
    len : int
        The number of vertices.
    perimeter : float
        The geometric length of the perimeter of the ring.
    bounding_box : libpysal.cg.Rectangle
        The bounding box of the ring.
    area : float
        The area enclosed by the ring.
    centroid : {tuple, libpysal.cg.Point}
        The centroid of the ring defined by the 'center of gravity'
        or 'center or mass'.
    _quad_tree_structure : libpysal.cg.QuadTreeStructureSingleRing
        The quad tree structure for the ring. This structure helps
        test if a point is inside the ring.
    
    """

    def __init__(self, vertices):
        if vertices[0] != vertices[-1]:
            vertices = vertices[:] + vertices[0:1]
            # msg = "Supplied vertices do not form a closed ring, "
            # msg += "the first and last vertices are not the same."
            # raise ValueError(msg)

        self.vertices = tuple(vertices)
        self._perimeter = None
        self._bounding_box = None
        self._area = None
        self._centroid = None
        self._quad_tree_structure = None

    def __len__(self) -> int:
        return len(self.vertices)

    @property
    def len(self) -> int:
        return len(self)

    @staticmethod
    def dist(v1, v2) -> Union[int, float]:

        return math.hypot(v1[0] - v2[0], v1[1] - v2[1])

    @property
    def perimeter(self) -> Union[int, float]:

        if self._perimeter is None:
            dist = self.dist
            v = self.vertices
            self._perimeter = sum(
                [dist(v[i], v[i + 1]) for i in range(-1, len(self) - 1)]
            )
        return self._perimeter

    @property
    def bounding_box(self):
        """Returns the bounding box of the ring.

        Returns
        -------
        self._bounding_box : libpysal.cg.Rectangle
            The bounding box of the ring.
        
        Examples
        --------
        
        >>> r = Ring(
        ...     [
        ...         Point((0, 0)),
        ...         Point((2, 0)),
        ...         Point((2, 1)),
        ...         Point((0, 1)),
        ...         Point((0, 0))
        ...     ]
        ... )
        
        >>> r.bounding_box.left
        0.0
        
        >>> r.bounding_box.lower
        0.0
        
        >>> r.bounding_box.right
        2.0
        
        >>> r.bounding_box.upper
        1.0
        
        """

        if self._bounding_box is None:
            vertices = self.vertices
            x = [v[0] for v in vertices]
            y = [v[1] for v in vertices]
            self._bounding_box = Rectangle(min(x), min(y), max(x), max(y))

        return self._bounding_box

    @property
    def area(self) -> Union[int, float]:
        """Returns the area of the ring.

        Examples
        --------
        
        >>> r = Ring(
        ...     [
        ...         Point((0, 0)),
        ...         Point((2, 0)),
        ...         Point((2, 1)),
        ...         Point((0, 1)),
        ...         Point((0, 0))
        ...     ]
        ... )
        >>> r.area
        2.0
        
        """

        return abs(self.signed_area)

    @property
    def signed_area(self) -> Union[int, float]:
        if self._area is None:
            vertices = self.vertices
            x = [v[0] for v in vertices]
            y = [v[1] for v in vertices]
            N = len(self)

            A = 0.0
            for i in range(N - 1):
                A += (x[i] + x[i + 1]) * (y[i] - y[i + 1])
            A = A * 0.5
            self._area = -A

        return self._area

    @property
    def centroid(self):
        """Returns the centroid of the ring.

        Returns
        -------
        self._centroid : libpysal.cg.Point
            The ring's centroid.

        Notes
        -----
        
        The centroid returned by this method is the geometric centroid.
        Also known as the 'center of gravity' or 'center of mass'.

        Examples
        --------
        
        >>> r = Ring(
        ...     [
        ...         Point((0, 0)),
        ...         Point((2, 0)),
        ...         Point((2, 1)),
        ...         Point((0, 1)),
        ...         Point((0, 0))
        ...     ]
        ... )
        >>> str(r.centroid)
        '(1.0, 0.5)'
        
        """

        if self._centroid is None:
            vertices = self.vertices
            x = [v[0] for v in vertices]
            y = [v[1] for v in vertices]
            A = self.signed_area
            N = len(self)
            cx = 0
            cy = 0
            for i in range(N - 1):
                f = x[i] * y[i + 1] - x[i + 1] * y[i]
                cx += (x[i] + x[i + 1]) * f
                cy += (y[i] + y[i + 1]) * f
            cx = 1.0 / (6 * A) * cx
            cy = 1.0 / (6 * A) * cy
            self._centroid = Point((cx, cy))

        return self._centroid

    def build_quad_tree_structure(self):
        """Build the quad tree structure for this polygon. Once
        the structure is built, speed for testing if a point is
        inside the ring will be increased significantly.
        
        """

        self._quad_tree_structure = QuadTreeStructureSingleRing(self)

    def contains_point(self, point):
        """Point containment using winding number. The implementation is based on
        `this <http://www.engr.colostate.edu/~dga/dga/papers/point_in_polygon.pdf>`_.
        
        Parameters
        ----------
        point : libpysal.cg.Point
            The point to test for containment.
        
        Returns
        -------
        point_contained : bool
            ``True`` if ``point`` is contained within the polygon, otherwise ``False``.
        
        """

        point_contained = False

        if self._quad_tree_structure is None:
            x, y = point

            # bbox checks
            bbleft = x < self.bounding_box.left
            bbright = x > self.bounding_box.right
            bblower = y < self.bounding_box.lower
            bbupper = y > self.bounding_box.upper

            if bbleft or bbright or bblower or bbupper:
                pass
            else:
                rn = len(self.vertices)
                xs = [self.vertices[i][0] - point[0] for i in range(rn)]
                ys = [self.vertices[i][1] - point[1] for i in range(rn)]
                w = 0

                for i in range(len(self.vertices) - 1):
                    yi = ys[i]
                    yj = ys[i + 1]
                    xi = xs[i]
                    xj = xs[i + 1]
                    if yi * yj < 0:
                        r = xi + yi * (xj - xi) / (yi - yj)
                        if r > 0:
                            if yi < 0:
                                w += 1
                            else:
                                w -= 1
                    elif yi == 0 and xi > 0:
                        if yj > 0:
                            w += 0.5
                        else:
                            w -= 0.5
                    elif yj == 0 and xj > 0:
                        if yi < 0:
                            w += 0.5
                        else:
                            w -= 0.5
                if w == 0:
                    pass
                else:
                    point_contained = True
        else:
            point_contained = self._quad_tree_structure.contains_point(point)

        return point_contained


class Polygon(Geometry):
    """Geometric representation of polygon objects.
    Returns a polygon created from the objects specified.

    Parameters
    ----------
    vertices : list
        A list of vertices or a list of lists of vertices.
    holes : list
        A list of sub-polygons to be considered as holes.
        Default is ``None``.

    Attributes
    ----------
    vertices : list
        A list of points with the vertices of the polygon in clockwise order.
    len : int
        The number of vertices including holes.
    perimeter : float
        The geometric length of the perimeter of the polygon.
    bounding_box : libpysal.cg.Rectangle
        The bounding box of the polygon.
    bbox : list
        A list representation of the bounding box in the 
        form ``[left, lower, right, upper]``.
    area : float
        The area enclosed by the polygon.
    centroid : tuple
        The 'center of gravity', i.e. the mean point of the polygon.
    
    Examples
    --------
    
    >>> p1 = Polygon([Point((0, 0)), Point((1, 0)), Point((1, 1)), Point((0, 1))])
    
    """

    def __init__(self, vertices, holes=None):

        self._part_rings = []
        self._hole_rings = []

        def clockwise(part: list) -> list:
            if standalone.is_clockwise(part):
                return part[:]
            else:
                return part[::-1]

        vl = list(vertices)
        if isinstance(vl[0], list):
            self._part_rings = list(map(Ring, vertices))
            self._vertices = [clockwise(part) for part in vertices]
        else:
            self._part_rings = [Ring(vertices)]
            self._vertices = [clockwise(vertices)]
        if holes is not None and holes != []:
            if isinstance(holes[0], list):
                self._hole_rings = list(map(Ring, holes))
                self._holes = [clockwise(hole) for hole in holes]
            else:
                self._hole_rings = [Ring(holes)]
                self._holes = [clockwise(holes)]
        else:
            self._holes = [[]]
        self._reset_props()

    @classmethod
    def __from_geo_interface__(cls, geo: dict):
        """While PySAL does not differentiate polygons and multipolygons
        GEOS, Shapely, and geoJSON do. In GEOS, etc, polygons may only
        have a single exterior ring, all other parts are holes.
        MultiPolygons are simply a list of polygons.
        
        """

        geo_type = geo["type"].lower()
        if geo_type == "multipolygon":
            parts = []
            holes = []
            for polygon in geo["coordinates"]:
                verts = [[Point(pt) for pt in part] for part in polygon]
                parts += verts[0:1]
                holes += verts[1:]
            if not holes:
                holes = None
            return cls(parts, holes)
        else:
            verts = [[Point(pt) for pt in part] for part in geo["coordinates"]]
            return cls(verts[0:1], verts[1:])

    @property
    def __geo_interface__(self) -> dict:
        """Return ``__geo_interface__`` information lookup."""

        if len(self.parts) > 1:
            geo = {
                "type": "MultiPolygon",
                "coordinates": [[part] for part in self.parts],
            }
            if self._holes[0]:
                geo["coordinates"][0] += self._holes
            return geo
        if self._holes[0]:
            return {"type": "Polygon", "coordinates": self._vertices + self._holes}
        else:
            return {"type": "Polygon", "coordinates": self._vertices}

    def _reset_props(self):
        """Resets the geometric properties of the polygon."""
        self._perimeter = None
        self._bounding_box = None
        self._bbox = None
        self._area = None
        self._centroid = None
        self._len = None

    def __len__(self) -> int:
        return self.len

    @property
    def len(self) -> int:
        """Returns the number of vertices in the polygon.

        Examples
        --------
        
        >>> p1 = Polygon([Point((0, 0)), Point((0, 1)), Point((1, 1)), Point((1, 0))])
        >>> p1.len
        4
        
        >>> len(p1)
        4
        
        """

        if self._len is None:
            self._len = len(self.vertices)
        return self._len

    @property
    def vertices(self) -> list:
        """Returns the vertices of the polygon in clockwise order.

        Examples
        --------
        
        >>> p1 = Polygon([Point((0, 0)), Point((0, 1)), Point((1, 1)), Point((1, 0))])
        >>> len(p1.vertices)
        4
        
        """

        return sum([part for part in self._vertices], []) + sum(
            [part for part in self._holes], []
        )

    @property
    def holes(self) -> list:
        """Returns the holes of the polygon in clockwise order.

        Examples
        --------
        
        >>> p = Polygon(
        ...     [Point((0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))],
        ...     [Point((1, 2)), Point((2, 2)), Point((2, 1)), Point((1, 1))]
        ... )
        >>> len(p.holes)
        1
        
        """

        return [[v for v in part] for part in self._holes]

    @property
    def parts(self) -> list:
        """Returns the parts of the polygon in clockwise order.

        Examples
        --------
        
        >>> p = Polygon(
        ...     [
        ...         [Point((0, 0)), Point((1, 0)), Point((1, 1)), Point((0, 1))],
        ...         [Point((2, 1)), Point((2, 2)), Point((1, 2)), Point((1, 1))]
        ...     ]
        ... )
        >>> len(p.parts)
        2
        
        """

        return [[v for v in part] for part in self._vertices]

    @property
    def perimeter(self) -> Union[int, float]:
        """Returns the perimeter of the polygon.

        Examples
        --------
        
        >>> p = Polygon([Point((0, 0)), Point((1, 0)), Point((1, 1)), Point((0, 1))])
        >>> p.perimeter
        4.0
        
        """

        def dist(v1: Union[int, float], v2: Union[int, float]) -> float:
            return math.hypot(v1[0] - v2[0], v1[1] - v2[1])

        def part_perimeter(part) -> Union[int, float]:
            return sum([dist(part[i], part[i + 1]) for i in range(-1, len(part) - 1)])

        sum_perim = lambda part_type: sum([part_perimeter(part) for part in part_type])

        if self._perimeter is None:
            self._perimeter = sum_perim(self._vertices) + sum_perim(self._holes)

        return self._perimeter

    @property
    def bbox(self):
        """Returns the bounding box of the polygon as a list.
        
        Returns
        -------
        self._bbox : list
            The bounding box of the polygon as a list.
        
        See Also
        --------
        
        libpysal.cg.bounding_box
        
        """

        if self._bbox is None:
            self._bbox = [
                self.bounding_box.left,
                self.bounding_box.lower,
                self.bounding_box.right,
                self.bounding_box.upper,
            ]
        return self._bbox

    @property
    def bounding_box(self):
        """Returns the bounding box of the polygon.

        Returns
        -------
        self._bounding_box : libpysal.cg.Rectangle
            The bounding box of the polygon.

        Examples
        --------
        
        >>> p = Polygon([Point((0, 0)), Point((2, 0)), Point((2, 1)), Point((0, 1))])
        >>> p.bounding_box.left
        0.0
        
        >>> p.bounding_box.lower
        0.0
        
        >>> p.bounding_box.right
        2.0
        
        >>> p.bounding_box.upper
        1.0
        
        """

        if self._bounding_box is None:
            vertices = self.vertices
            self._bounding_box = Rectangle(
                min([v[0] for v in vertices]),
                min([v[1] for v in vertices]),
                max([v[0] for v in vertices]),
                max([v[1] for v in vertices]),
            )
        return self._bounding_box

    @property
    def area(self) -> float:
        """Returns the area of the polygon.

        Examples
        --------
        
        >>> p = Polygon([Point((0, 0)), Point((1, 0)), Point((1, 1)), Point((0, 1))])
        >>> p.area
        1.0
        
        >>> p = Polygon(
        ...     [Point((0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))],
        ...     [Point((2, 1)), Point((2, 2)), Point((1, 2)), Point((1, 1))]
        ... )
        >>> p.area
        99.0
        
        """

        def part_area(pv: list) -> float:
            __area = 0
            for i in range(-1, len(pv) - 1):
                __area += (pv[i][0] + pv[i + 1][0]) * (pv[i][1] - pv[i + 1][1])
            __area = __area * 0.5
            if __area < 0:
                __area = -area
            return __area

        sum_area = lambda part_type: sum([part_area(part) for part in part_type])
        _area = sum_area(self._vertices) - sum_area(self._holes)

        return _area

    @property
    def centroid(self) -> tuple:
        """Returns the centroid of the polygon.

        Notes
        -----
        
        The centroid returned by this method is the geometric
        centroid and respects multipart polygons with holes.
        Also known as the 'center of gravity' or 'center of mass'.

        Examples
        --------
        
        >>> p = Polygon(
        ...     [Point((0, 0)), Point((10, 0)), Point((10, 10)), Point((0, 10))],
        ...     [Point((1, 1)), Point((1, 2)), Point((2, 2)), Point((2, 1))]
        ... )
        >>> p.centroid
        (5.0353535353535355, 5.0353535353535355)
        
        """

        CP = [ring.centroid for ring in self._part_rings]
        AP = [ring.area for ring in self._part_rings]
        CH = [ring.centroid for ring in self._hole_rings]
        AH = [-ring.area for ring in self._hole_rings]

        A = AP + AH
        cx = sum([pt[0] * area for pt, area in zip(CP + CH, A)]) / sum(A)
        cy = sum([pt[1] * area for pt, area in zip(CP + CH, A)]) / sum(A)

        return cx, cy

    def build_quad_tree_structure(self):
        """Build the quad tree structure for this polygon. Once
        the structure is built, speed for testing if a point is
        inside the ring will be increased significantly.
        
        """

        for ring in self._part_rings:
            ring.build_quad_tree_structure()
        for ring in self._hole_rings:
            ring.build_quad_tree_structure()
        self.is_quad_tree_structure_built = True

    def contains_point(self, point):
        """Test if a polygon contains a point.
        
        Parameters
        ----------
        point : libpysal.cg.Point
            A point to test for containment.
        
        Returns
        -------
        contains : bool
            ``True`` if the polygon contains ``point`` otherwise ``False``.
        
        Examples
        --------
        
        >>> p = Polygon(
        ...     [Point((0,0)), Point((4,0)), Point((4,5)), Point((2,3)), Point((0,5))]
        ... )
        >>> p.contains_point((3,3))
        1
        
        >>> p.contains_point((0,6))
        0
        
        >>> p.contains_point((2,2.9))
        1
        
        >>> p.contains_point((4,5))
        0
        
        >>> p.contains_point((4,0))
        0

        Handles holes.

        >>> p = Polygon(
        ...     [Point((0, 0)), Point((0, 10)), Point((10, 10)), Point((10, 0))],
        ...     [Point((2, 2)), Point((4, 2)), Point((4, 4)), Point((2, 4))]
        ... )
        >>> p.contains_point((3.0, 3.0))
        False
        
        >>> p.contains_point((1.0, 1.0))
        True

        Notes
        -----
        
        Points falling exactly on polygon edges may yield unpredictable results.
        
        """

        searching = True

        for ring in self._hole_rings:
            if ring.contains_point(point):
                contains = False
                searching = False
                break

        if searching:
            for ring in self._part_rings:
                if ring.contains_point(point):
                    contains = True
                    searching = False
                    break
            if searching:
                contains = False

        return contains


class Rectangle(Geometry):
    """Geometric representation of rectangle objects.

    Attributes
    ----------
    left : float
        Minimum x-value of the rectangle.
    lower : float
        Minimum y-value of the rectangle.
    right : float
        Maximum x-value of the rectangle.
    upper : float
        Maximum y-value of the rectangle.
    
    Examples
    --------
    
    >>> r = Rectangle(-4, 3, 10, 17)
    >>> r.left #minx
    -4.0
    
    >>> r.lower #miny
    3.0
    
    >>> r.right #maxx
    10.0
    
    >>> r.upper #maxy
    17.0
    
    """

    def __init__(self, left, lower, right, upper):

        if right < left or upper < lower:
            raise ArithmeticError("Rectangle must have positive area.")
        self.left = float(left)
        self.lower = float(lower)
        self.right = float(right)
        self.upper = float(upper)

    def __bool__(self):
        """Rectangles will evaluate to False if they have zero area.
        ``___nonzero__`` is used "to implement truth value
        testing and the built-in operation ``bool()``"
        ``-- http://docs.python.org/reference/datamodel.html

        Examples
        --------
        
        >>> r = Rectangle(0, 0, 0, 0)
        >>> bool(r)
        False
        
        >>> r = Rectangle(0, 0, 1, 1)
        >>> bool(r)
        True
        
        """

        return bool(self.area)

    def __eq__(self, other):
        if other:
            return self[:] == other[:]
        return False

    def __add__(self, other):
        x, y, X, Y = self[:]
        x1, y2, X1, Y1 = other[:]

        return Rectangle(
            min(self.left, other.left),
            min(self.lower, other.lower),
            max(self.right, other.right),
            max(self.upper, other.upper),
        )

    def __getitem__(self, key):
        """
        
        Examples
        --------
        
        >>> r = Rectangle(-4, 3, 10, 17)
        >>> r[:]
        [-4.0, 3.0, 10.0, 17.0]
        
        """

        l = [self.left, self.lower, self.right, self.upper]

        return l.__getitem__(key)

    def set_centroid(self, new_center):
        """Moves the rectangle center to a new specified point.

        Parameters
        ----------
        new_center : libpysal.cg.Point
            The new location of the centroid of the polygon.

        Examples
        --------
        
        >>> r = Rectangle(0, 0, 4, 4)
        >>> r.set_centroid(Point((4, 4)))
        >>> r.left
        2.0
        
        >>> r.right
        6.0
        
        >>> r.lower
        2.0
        
        >>> r.upper
        6.0
        
        """

        shift = (
            new_center[0] - (self.left + self.right) / 2,
            new_center[1] - (self.lower + self.upper) / 2,
        )

        self.left = self.left + shift[0]
        self.right = self.right + shift[0]
        self.lower = self.lower + shift[1]
        self.upper = self.upper + shift[1]

    def set_scale(self, scale):
        """Rescales the rectangle around its center.

        Parameters
        ----------
        scale : int, float
            The ratio of the new scale to the old
            scale (e.g. 1.0 is current size).

        Examples
        --------
        
        >>> r = Rectangle(0, 0, 4, 4)
        >>> r.set_scale(2)
        >>> r.left
        -2.0
        >>> r.right
        6.0
        >>> r.lower
        -2.0
        >>> r.upper
        6.0
        
        """

        center = ((self.left + self.right) / 2, (self.lower + self.upper) / 2)

        self.left = center[0] + scale * (self.left - center[0])
        self.right = center[0] + scale * (self.right - center[0])
        self.lower = center[1] + scale * (self.lower - center[1])
        self.upper = center[1] + scale * (self.upper - center[1])

    @property
    def area(self) -> Union[int, float]:
        """Returns the area of the Rectangle.

        Examples
        --------
        
        >>> r = Rectangle(0, 0, 4, 4)
        >>> r.area
        16.0
        
        """

        return (self.right - self.left) * (self.upper - self.lower)

    @property
    def width(self) -> Union[int, float]:
        """Returns the width of the Rectangle.

        Examples
        --------
        
        >>> r = Rectangle(0, 0, 4, 4)
        >>> r.width
        4.0
        
        """

        return self.right - self.left

    @property
    def height(self) -> Union[int, float]:
        """Returns the height of the Rectangle.

        Examples
        --------
        
        >>> r = Rectangle(0, 0, 4, 4)
        >>> r.height
        4.0
        
        """

        return self.upper - self.lower


_geoJSON_type_to_Pysal_type = {
    "point": Point,
    "linestring": Chain,
    "multilinestring": Chain,
    "polygon": Polygon,
    "multipolygon": Polygon,
}

# moving this to top breaks unit tests !
from . import standalone
from .polygonQuadTreeStructure import QuadTreeStructureSingleRing

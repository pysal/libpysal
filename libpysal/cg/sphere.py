"""
sphere: Tools for working with spherical geometry.

Author(s):
    Charles R Schmidt schmidtc@gmail.com
    Luc Anselin luc.anselin@asu.edu
    Xun Li xun.li@asu.edu

"""

__author__ = (
    "Charles R Schmidt <schmidtc@gmail.com>,"
    "Luc Anselin <luc.anselin@asu.edu,"
    "Xun Li <xun.li@asu.edu"
)

import math
import numpy
import scipy.spatial
import scipy.constants
from scipy.spatial.distance import euclidean
from math import pi, cos, sin


__all__ = [
    "RADIUS_EARTH_KM",
    "RADIUS_EARTH_MILES",
    "arcdist",
    "arcdist2linear",
    "brute_knn",
    "fast_knn",
    "fast_threshold",
    "linear2arcdist",
    "toLngLat",
    "toXYZ",
    "lonlat",
    "harcdist",
    "geointerpolate",
    "geogrid",
]


RADIUS_EARTH_KM = 6371.0
RADIUS_EARTH_MILES = (RADIUS_EARTH_KM * scipy.constants.kilo) / scipy.constants.mile


def arcdist(pt0, pt1, radius=RADIUS_EARTH_KM):
    """Arc distance between two points on a sphere.

    Parameters
    ----------
    pt0 : tuple
        A point assumed to be in form (longitude,latitude).
    pt1 : tuple
        A point assumed to be in form (longitude,latitude).
    radius : float
        The radius of a sphere. Default is Earth's radius in
        kilometers, ``RADIUS_EARTH_KM`` (``6371.0``). Earth's
        radius in miles, ``RADIUS_EARTH_MILES`` (``3958.76``)
        is also an option.
        Source: http://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html

    Returns
    -------
    dist : float
        The arc distance between ``pt0`` and ``pt1`` using supplied ``radius``.

    Examples
    --------
    
    >>> pt0 = (0, 0)
    >>> pt1 = (180, 0)
    >>> d = arcdist(pt0, pt1, RADIUS_EARTH_MILES)
    >>> d == math.pi * RADIUS_EARTH_MILES
    True
    
    """

    dist = linear2arcdist(euclidean(toXYZ(pt0), toXYZ(pt1)), radius)

    return dist


def arcdist2linear(arc_dist, radius=RADIUS_EARTH_KM):
    """Convert an arc distance (spherical earth)
    to a linear distance (R3) in the unit sphere.
    
    Parameters
    ----------
    arc_dist : float
        The arc distance to convert.
    radius : float
        The radius of a sphere. Default is Earth's radius in
        kilometers, ``RADIUS_EARTH_KM`` (``6371.0``). Earth's
        radius in miles, ``RADIUS_EARTH_MILES`` (``3958.76``)
        is also an option.
        Source: http://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html
    
    Returns
    -------
    linear_dist : float
        The linear distance conversion of ``arc_dist``.
    
    Examples
    --------
    
    >>> pt0 = (0, 0)
    >>> pt1 = (180, 0)
    >>> d = arcdist(pt0, pt1, RADIUS_EARTH_MILES)
    >>> d == math.pi * RADIUS_EARTH_MILES
    True
    
    >>> arcdist2linear(d, RADIUS_EARTH_MILES)
    2.0
    
    """

    circumference = 2 * math.pi * radius
    linear_dist = (
        2 - (2 * math.cos(math.radians((arc_dist * 360.0) / circumference)))
    ) ** (0.5)

    return linear_dist


def linear2arcdist(linear_dist, radius=RADIUS_EARTH_KM):
    """Convert a linear distance in the unit sphere
    (R3) to an arc distance based on supplied radius.
    
    Parameters
    ----------
    linear_dist : float
        The linear distance to convert.
    radius : float
        The radius of a sphere. Default is Earth's radius in
        kilometers, ``RADIUS_EARTH_KM`` (``6371.0``). Earth's
        radius in miles, ``RADIUS_EARTH_MILES`` (``3958.76``)
        is also an option.
        Source: http://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html
    
    Returns
    -------
    arc_dist : float
        The arc distance conversion of ``linear_dist``.
    
    Raises
    ------
    ValueError
        Raised when ``linear_dist`` exceeds the diameter of the unit sphere.
    
    Examples
    --------
    
    >>> pt0 = (0, 0)
    >>> pt1 = (180, 0)
    >>> d = arcdist(pt0, pt1, RADIUS_EARTH_MILES)
    >>> d == linear2arcdist(2.0, radius=RADIUS_EARTH_MILES)
    True
    
    """

    if linear_dist == float("inf"):
        arc_dist = linear_dist
    elif linear_dist > 2.0:
        msg = "'linear_dist', must not exceed the diameter of the unit sphere, 2.0."
        raise ValueError(msg)
    else:
        circumference = 2 * math.pi * radius
        a2 = linear_dist ** 2
        theta = math.degrees(math.acos((2 - a2) / (2.0)))
        arc_dist = (theta * circumference) / 360.0

    return arc_dist


def toXYZ(pt):
    """Convert a point's latitude and longitude to x,y,z.
    
    Parameters
    ----------
    pt : tuple
        A point assumed to be in form (lng,lat).

    Returns
    -------
    x, y, z : tuple
        A point in form (x, y, z).
    
    """

    phi, theta = list(map(math.radians, pt))
    phi, theta = phi + pi, theta + (pi / 2)
    x = 1 * sin(theta) * cos(phi)
    y = 1 * sin(theta) * sin(phi)
    z = 1 * cos(theta)

    return x, y, z


def toLngLat(xyz):
    """Convert a point's x,y,z to latitude and longitude.
    
    Parameters
    ----------
    xyz : tuple
        A point assumed to be in form (x,y,z).
    
    Returns
    -------
    phi, theta : tuple
        A point in form (phi, theta) [y,x].
    
    """

    x, y, z = xyz
    if z == -1 or z == 1:
        phi = 0
    else:
        phi = math.atan2(y, x)
        if phi > 0:
            phi = phi - math.pi
        elif phi < 0:
            phi = phi + math.pi
    theta = math.acos(z) - (math.pi / 2)

    return phi, theta


def brute_knn(pts, k, mode="arc", radius=RADIUS_EARTH_KM):
    """Computes a brute-force :math:`k` nearest neighbors.
    
    Parameters
    ----------
    pts : list
        A list of :math:`x,y` pairs.
    k :  int
        The number of points to query.
    mode : str
        The mode of distance. Valid modes are ``'arc'``
        and ``'xyz'``. Default is ``'arc'``.
    radius : float
        The radius of a sphere. Default is Earth's radius in
        kilometers, ``RADIUS_EARTH_KM`` (``6371.0``). Earth's
        radius in miles, ``RADIUS_EARTH_MILES`` (``3958.76``)
        is also an option.
        Source: http://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html
    
    Returns
    -------
    w : dict
        A neighbor ID lookup.
    
    """

    n = len(pts)
    full = numpy.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            if mode == "arc":
                lng0, lat0 = pts[i]
                lng1, lat1 = pts[j]
                dist = arcdist(pts[i], pts[j], radius=radius)
            elif mode == "xyz":
                dist = euclidean(pts[i], pts[j])
            full[i, j] = dist
            full[j, i] = dist

    w = {}
    for i in range(n):
        w[i] = full[i].argsort()[1 : k + 1].tolist()

    return w


def fast_knn(pts, k, return_dist=False, radius=RADIUS_EARTH_KM):
    """Computes :math:`k` nearest neighbors on a sphere.

    Parameters
    ----------
    pts : list
        A list of :math:`x,y` pairs.
    k :  int
        The number of points to query.
    return_dist : bool
        Return distances in the ``wd`` container object (``True``).
        Default is ``False``.
    radius : float
        The radius of a sphere. Default is Earth's radius in
        kilometers, ``RADIUS_EARTH_KM`` (``6371.0``). Earth's
        radius in miles, ``RADIUS_EARTH_MILES`` (``3958.76``)
        is also an option.
        Source: http://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html

    Returns
    -------
    wn : dict
        A neighbor ID lookup.
    wd : dict
        A neighbor distance lookup (optional).

    """

    pts = numpy.array(pts)
    kd = scipy.spatial.KDTree(pts)
    d, w = kd.query(pts, k + 1)
    w = w[:, 1:]
    wn = {}

    for i in range(len(pts)):
        wn[i] = w[i].tolist()

    if return_dist:
        d = d[:, 1:]
        wd = {}
        for i in range(len(pts)):
            wd[i] = [linear2arcdist(x, radius=radius) for x in d[i].tolist()]
        return wn, wd
    return wn


def fast_threshold(pts, dist, radius=RADIUS_EARTH_KM):
    """Find all neighbors on a sphere within a threshold distance.

    Parameters
    ----------
    pointslist : list
        A list of lat-lon tuples. This **must** be a list, even for one point.
    dist: float
        The threshold distance.
    radius : float
        The radius of a sphere. Default is Earth's radius in
        kilometers, ``RADIUS_EARTH_KM`` (``6371.0``). Earth's
        radius in miles, ``RADIUS_EARTH_MILES`` (``3958.76``)
        is also an option.
        Source: http://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html

    Returns
    -------
    wd : dict
        A neighbor distance lookup where the key is the ID
        of a point and the value is a list of IDs for other
        points within ``dist`` of the key point,

    """

    d = arcdist2linear(dist, radius)
    kd = scipy.spatial.KDTree(pts)
    r = kd.query_ball_tree(kd, d)
    wd = {}

    for i in range(len(pts)):
        l = r[i]
        l.remove(i)
        wd[i] = l

    return wd


def lonlat(pointslist):
    """Converts point order from lat-lon tuples to lon-lat (x,y) tuples.

    Parameters
    ----------
    pointslist : list
        A list of lat-lon tuples. This **must** be a list, even for one point.

    Returns
    -------
    newpts : list
        A list with tuples of points in lon-lat order.

    Examples
    --------
    
    >>> points = [
    ...     (41.981417, -87.893517), (41.980396, -87.776787), (41.980906, -87.696450)
    ... ]
    >>> newpoints = lonlat(points)
    >>> newpoints
    [(-87.893517, 41.981417), (-87.776787, 41.980396), (-87.69645, 41.980906)]

    """

    newpts = [(i[1], i[0]) for i in pointslist]

    return newpts


def haversine(x):
    """Computes the haversine formula.

    Parameters
    ----------
    x : float
        The angle in radians.

    Returns
    -------
    haversine_dist : float
        The square of sine of half the radian (the haversine formula).

    Examples
    --------
    
    >>> haversine(math.pi) # is 180 in radians, hence sin of 90 = 1
    1.0

    """

    x = math.sin(x / 2)

    haversine_dist = x * x

    return haversine_dist


# Lambda functions

# degree to radian conversion
d2r = lambda x: x * math.pi / 180.0

# radian to degree conversion
r2d = lambda x: x * 180.0 / math.pi


def radangle(p0, p1):
    """Radian angle between two points on a sphere in lon-lat (x,y).

    Parameters
    ----------
    p0 : tuple
        The first point in (lon,lat) format.
    p1 : tuple
        The second point in (lon,lat) format.

    Returns
    -------
    d : float
        Radian angle in radians.

    Examples
    --------
    
    >>> p0 = (-87.893517, 41.981417)
    >>> p1 = (-87.519295, 41.657498)
    >>> radangle(p0, p1)
    0.007460167953189258

    Notes
    -----
    
    Uses haversine formula, function haversine and degree to radian
    conversion lambda function ``d2r``.

    """

    x0, y0 = d2r(p0[0]), d2r(p0[1])
    x1, y1 = d2r(p1[0]), d2r(p1[1])
    d = 2.0 * math.asin(
        math.sqrt(haversine(y1 - y0) + math.cos(y0) * math.cos(y1) * haversine(x1 - x0))
    )

    return d


def harcdist(p0, p1, lonx=True, radius=RADIUS_EARTH_KM):
    """Alternative the arc distance function, uses the haversine formula.

    Parameters
    ----------
    p0 : tuple
        The first point decimal degrees.
    p1 : tuple
        The second point decimal degrees.
    lonx : bool
        The method to assess the order of the coordinates.
        ``True`` for (lon,lat); ``False`` for (lat,lon).
        Default is ``True``.
    radius : float
        The radius of a sphere. Default is Earth's radius in
        kilometers, ``RADIUS_EARTH_KM`` (``6371.0``). Earth's
        radius in miles, ``RADIUS_EARTH_MILES`` (``3958.76``)
        is also an option. Set to ``None`` for radians.
        Source: http://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html

    Returns
    -------
    harc_dist : harc_dist
        The distance in units specified, km, miles or radians.

    Examples
    --------
    
    >>> p0 = (-87.893517, 41.981417)
    >>> p1 = (-87.519295, 41.657498)
    >>> harcdist(p0, p1)
    47.52873002976876
    
    >>> harcdist(p0, p1, radius=None)
    0.007460167953189258

    Notes
    -----
    
    Uses the ``radangle`` function to compute radian angle.

    """

    if not (lonx):
        p = lonlat([p0, p1])
        p0 = p[0]
        p1 = p[1]

    harc_dist = radangle(p0, p1)

    if radius is not None:
        harc_dist = harc_dist * radius

    return harc_dist


def geointerpolate(p0, p1, t, lonx=True):
    """Finds a point on a sphere along the great circle distance between
    two points on a sphere also known as a way point in great circle navigation.

    Parameters
    ----------
    p0 : tuple
        The first point decimal degrees.
    p1 : tuple
        The second point decimal degrees.
    t : float
        The proportion along great circle distance between ``p0``
        and ``p1`` (e.g., :math:`\mathtt{t}=0.5` would find the mid-point).
    lonx : bool
        The method to assess the order of the coordinates.
        ``True`` for (lon,lat); ``False`` for (lat,lon).
        Default is ``True``.

    Returns
    -------
    newpx, newpy : tuple
        The new point in decimal degrees of (lon-lat) by
        default or (lat-lon) if ``lonx`` is set to ``False``.

    Examples
    --------
    
    >>> p0 = (-87.893517, 41.981417)
    >>> p1 = (-87.519295, 41.657498)
    >>> geointerpolate(p0, p1, 0.1)             # using lon-lat
    (-87.85592403438788, 41.949079912574796)
    
    >>> p3 = (41.981417, -87.893517)
    >>> p4 = (41.657498, -87.519295)
    >>> geointerpolate(p3, p4, 0.1, lonx=False) # using lat-lon
    (41.949079912574796, -87.85592403438788)

    """

    if not (lonx):
        p = lonlat([p0, p1])
        p0 = p[0]
        p1 = p[1]

    d = radangle(p0, p1)
    k = 1.0 / math.sin(d)
    t = t * d
    A = math.sin(d - t) * k
    B = math.sin(t) * k

    x0, y0 = d2r(p0[0]), d2r(p0[1])
    x1, y1 = d2r(p1[0]), d2r(p1[1])

    x = A * math.cos(y0) * math.cos(x0) + B * math.cos(y1) * math.cos(x1)
    y = A * math.cos(y0) * math.sin(x0) + B * math.cos(y1) * math.sin(x1)
    z = A * math.sin(y0) + B * math.sin(y1)

    newpx = r2d(math.atan2(y, x))
    newpy = r2d(math.atan2(z, math.sqrt(x * x + y * y)))

    if not lonx:
        return newpy, newpx

    return newpx, newpy


def geogrid(pup, pdown, k, lonx=True):
    """Computes a :math:`k+1` by :math:`k+1` set of grid
    points for a bounding box in lat-lon. Uses ``geointerpolate``.

    Parameters
    ----------
    pup : tuple
        The lat-lon or lon-lat for the upper left corner of the bounding box.
    pdown : tuple
        The lat-lon or lon-lat for The lower right corner of The bounding box.
    k : int
        The number of grid cells (grid points will be one more).
    lonx : bool
        The method to assess the order of the coordinates.
        ``True`` for (lon,lat); ``False`` for (lat,lon).
        Default is ``True``.

    Returns
    -------
    grid : list
        A list of tuples with (lat-lon) or (lon-lat) for grid points,
        row by row, starting with the top row and moving to the bottom;
        coordinate tuples are returned in same order as input.

    Examples
    --------
    
    >>> pup = (42.023768, -87.946389)       # Arlington Heights, IL
    >>> pdown = (41.644415, -87.524102)     # Hammond, IN
    >>> geogrid(pup,pdown, 3, lonx=False)
    [(42.023768, -87.946389),
     (42.02393997819538, -87.80562679358316),
     (42.02393997819538, -87.66486420641684),
     (42.023768, -87.524102),
     (41.897317, -87.94638900000001),
     (41.8974888973743, -87.80562679296166),
     (41.8974888973743, -87.66486420703835),
     (41.897317, -87.524102),
     (41.770866000000005, -87.94638900000001),
     (41.77103781320412, -87.80562679234043),
     (41.77103781320412, -87.66486420765956),
     (41.770866000000005, -87.524102),
     (41.644415, -87.946389),
     (41.64458672568646, -87.80562679171955),
     (41.64458672568646, -87.66486420828045),
     (41.644415, -87.524102)]

    """

    if lonx:
        corners = [pup, pdown]
    else:
        corners = lonlat([pup, pdown])

    tpoints = [float(i) / k for i in range(k)[1:]]
    leftcorners = [corners[0], (corners[0][0], corners[1][1])]
    rightcorners = [(corners[1][0], corners[0][1]), corners[1]]
    leftside = [leftcorners[0]]
    rightside = [rightcorners[0]]

    for t in tpoints:
        newpl = geointerpolate(leftcorners[0], leftcorners[1], t)
        leftside.append(newpl)
        newpr = geointerpolate(rightcorners[0], rightcorners[1], t)
        rightside.append(newpr)
    leftside.append(leftcorners[1])
    rightside.append(rightcorners[1])

    grid = []
    for i in range(len(leftside)):
        grid.append(leftside[i])
        for t in tpoints:
            newp = geointerpolate(leftside[i], rightside[i], t)
            grid.append(newp)
        grid.append(rightside[i])
    if not (lonx):
        grid = lonlat(grid)

    return grid

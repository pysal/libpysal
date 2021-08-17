"""
Computation of alpha shape algorithm in 2-D based on original implementation
by Tim Kittel (@timkittel) available at:

    https://github.com/timkittel/alpha-shapes

Author(s):
    Dani Arribas-Bel daniel.arribas.bel@gmail.com
"""

import numpy as np
import scipy.spatial as spat

from ..common import requires, jit, HAS_JIT

if not HAS_JIT:
    from warnings import warn

    NUMBA_WARN = (
        "Numba not imported, so alpha shape construction may be slower than expected."
    )

try:
    import pygeos

    HAS_PYGEOS = True
except ModuleNotFoundError:
    HAS_PYGEOS = False


EPS = np.finfo(float).eps

__all__ = ["alpha_shape", "alpha_shape_auto"]


@jit
def nb_dist(x, y):
    """numba implementation of distance between points `x` and `y`

    Parameters
    ----------

    x : ndarray
        Coordinates of point `x`

    y : ndarray
        Coordinates of point `y`

    Returns
    -------

    dist : float
        Distance between `x` and `y`

    Examples
    --------

    >>> x = np.array([0, 0])
    >>> y = np.array([1, 1])
    >>> dist = nb_dist(x, y)
    >>> dist
    1.4142135623730951

    """
    sum = 0
    for x_i, y_i in zip(x, y):
        sum += (x_i - y_i) ** 2
    dist = np.sqrt(sum)
    return dist


@jit(nopython=True)
def r_circumcircle_triangle_single(a, b, c):
    """Computation of the circumcircle of a single triangle

    Parameters
    ----------

    a : ndarray
        (2,) Array with coordinates of vertex `a` of the triangle
    b : ndarray
        (2,) Array with coordinates of vertex `b` of the triangle
    c : ndarray
        (2,) Array with coordinates of vertex `c` of the triangle

    Returns
    -------

    r : float
        Circumcircle of the triangle

    Notes
    -----

    Source for equations:

    > https://www.mathopenref.com/trianglecircumcircle.html

    [Last accessed July 11th. 2018]

    Examples
    --------

    >>> a = np.array([0, 0])
    >>> b = np.array([0.5, 0])
    >>> c = np.array([0.25, 0.25])
    >>> r = r_circumcircle_triangle_single(a, b, c)
    >>> r
    0.2500000000000001

    """
    ab = nb_dist(a, b)
    bc = nb_dist(b, c)
    ca = nb_dist(c, a)

    num = ab * bc * ca
    den = np.sqrt((ab + bc + ca) * (bc + ca - ab) * (ca + ab - bc) * (ab + bc - ca))
    if den == 0:
        return np.array([ab, bc, ca]).max() / 2.0
    else:
        return num / den


@jit(nopython=True)
def r_circumcircle_triangle(a_s, b_s, c_s):
    """Computation of circumcircles for a series of triangles

    Parameters
    ----------

    a_s : ndarray
        (N, 2) array with coordinates of vertices `a` of the triangles
    b_s : ndarray
        (N, 2) array with coordinates of vertices `b` of the triangles
    c_s : ndarray
        (N, 2) array with coordinates of vertices `c` of the triangles

    Returns
    -------

    radii : ndarray
        (N,) array with circumcircles for every triangle

    Examples
    --------

    >>> a_s = np.array([[0, 0], [2, 1], [3, 2]])
    >>> b_s = np.array([[1, 0], [5, 1], [2, 4]])
    >>> c_s = np.array([[0, 7], [1, 3], [4, 2]])
    >>> rs = r_circumcircle_triangle(a_s, b_s, c_s)
    >>> rs
    array([3.53553391, 2.5       , 1.58113883])

    """
    len_a = len(a_s)
    r2 = np.zeros((len_a,))
    for i in range(len_a):
        r2[i] = r_circumcircle_triangle_single(a_s[i], b_s[i], c_s[i])
    return r2


@jit
def get_faces(triangle):
    """Extract faces from a single triangle

    Parameters
    ----------

    triangles : ndarray
        (3,) array with the vertex indices for a triangle

    Returns
    -------

    faces : ndarray
        (3, 2) array with a row for each face containing the indices of the two
        points that make up the face

    Examples
    --------

    >>> triangle = np.array([3, 1, 4], dtype=np.int32)
    >>> faces = get_faces(triangle)
    >>> faces
    array([[3., 1.],
           [1., 4.],
           [4., 3.]])

    """
    faces = np.zeros((3, 2))
    for i, (i0, i1) in enumerate([(0, 1), (1, 2), (2, 0)]):
        faces[i] = triangle[i0], triangle[i1]
    return faces


@jit
def build_faces(faces, triangles_is, num_triangles, num_faces_single):
    """Build facing triangles

    Parameters
    ----------

    faces : ndarray
        (num_triangles * num_faces_single, 2) array of zeroes in int form

    triangles_is : ndarray
        (D, 3) array, where D is the number of Delaunay triangles, with the
        vertex indices for each triangle

    num_triangles : int
        Number of triangles

    num_faces_single : int
        Number of faces a triangle has (i.e. 3)

    Returns
    -------

    faces : ndarray
        Two dimensional array with a row for every facing segment containing
        the indices of the coordinate points

    Examples
    --------

    >>> import scipy.spatial as spat
    >>> pts = np.array([[0, 1], [3, 5], [4, 1], [6, 7], [9, 3]])
    >>> triangulation = spat.Delaunay(pts)
    >>> triangulation.simplices
    array([[3, 1, 4],
           [1, 2, 4],
           [2, 1, 0]], dtype=int32)
    >>> num_faces_single = 3
    >>> num_triangles = triangulation.simplices.shape[0]
    >>> num_faces = num_triangles * num_faces_single
    >>> faces = np.zeros((num_faces, 2), dtype=np.int_)
    >>> mask = np.ones((num_faces,), dtype=np.bool_)
    >>> faces = build_faces(faces, triangulation.simplices, num_triangles, num_faces_single)
    >>> faces
    array([[3, 1],
           [1, 4],
           [4, 3],
           [1, 2],
           [2, 4],
           [4, 1],
           [2, 1],
           [1, 0],
           [0, 2]])

    """
    for i in range(num_triangles):
        from_i = num_faces_single * i
        to_i = num_faces_single * (i + 1)
        faces[from_i:to_i] = get_faces(triangles_is[i])
    return faces


@jit
def nb_mask_faces(mask, faces):
    """ Run over each row in `faces`, if the face in the following row is the
    same, then mark both as False on `mask`

    Parameters
    ----------

    mask : ndarray
        One-dimensional boolean array set to True with as many observations as
        rows in `faces`

    faces : ndarray
        Sorted sequence of faces for all triangles (ie. triangles split by each
        segment)

    Returns
    -------

    masked : ndarray
         Sequence of outward-facing faces

    Examples
    --------

    >>> import numpy as np
    >>> faces = np.array([[0, 1], [0, 2], [1, 2], [1, 2], [1, 3], [1, 4], [1, 4], [2, 4], [3, 4]])
    >>> mask = np.ones((faces.shape[0], ), dtype=np.bool_)
    >>> masked = nb_mask_faces(mask, faces)
    >>> masked
    array([[0, 1],
           [0, 2],
           [1, 3],
           [2, 4],
           [3, 4]])

    """
    for k in range(faces.shape[0] - 1):
        if mask[k]:
            if np.all(faces[k] == faces[k + 1]):
                mask[k] = False
                mask[k + 1] = False
    return faces[mask]


def get_single_faces(triangles_is):
    """Extract outward facing edges from collection of triangles

    Parameters
    ----------

    triangles_is : ndarray
        (D, 3) array, where D is the number of Delaunay triangles, with the
        vertex indices for each triangle

    Returns
    -------

    single_faces : ndarray

    Examples
    --------

    >>> import scipy.spatial as spat
    >>> pts = np.array([[0, 1], [3, 5], [4, 1], [6, 7], [9, 3]])
    >>> alpha = 0.33
    >>> triangulation = spat.Delaunay(pts)
    >>> triangulation.simplices
    array([[3, 1, 4],
           [1, 2, 4],
           [2, 1, 0]], dtype=int32)
    >>> get_single_faces(triangulation.simplices)
    array([[0, 1],
           [0, 2],
           [1, 3],
           [2, 4],
           [3, 4]])

    """
    num_faces_single = 3
    num_triangles = triangles_is.shape[0]
    num_faces = num_triangles * num_faces_single
    faces = np.zeros((num_faces, 2), dtype=np.int_)
    mask = np.ones((num_faces,), dtype=np.bool_)

    faces = build_faces(faces, triangles_is, num_triangles, num_faces_single)

    orderlist = ["x{}".format(i) for i in range(faces.shape[1])]
    dtype_list = [(el, faces.dtype.str) for el in orderlist]
    # Arranging each face so smallest vertex is first
    faces.sort(axis=1)
    # Arranging faces in ascending way
    faces.view(dtype_list).sort(axis=0)
    # Masking
    single_faces = nb_mask_faces(mask, faces)
    return single_faces


@requires("geopandas", "shapely")
def alpha_geoms(alpha, triangles, radii, xys):
    """Generate alpha-shape polygon(s) from `alpha` value, vertices of
    `triangles`, the `radii` for all points, and the points themselves

    Parameters
    ----------

    alpha : float
        Alpha value to delineate the alpha-shape

    triangles : ndarray
         (D, 3) array, where D is the number of Delaunay triangles, with the
         vertex indices for each triangle

    radii : ndarray
        (N,) array with circumcircles for every triangle

    xys : ndarray
        (N, 2) array with one point per row and coordinates structured as X and Y

    Returns
    -------

    geoms : GeoSeries
        Polygon(s) resulting from the alpha shape algorithm. The GeoSeries
        object remains so even if only a single polygon is returned. There is
        no CRS included in the object.

    Examples
    --------

    >>> import scipy.spatial as spat
    >>> pts = np.array([[0, 1], [3, 5], [4, 1], [6, 7], [9, 3]])
    >>> alpha = 0.33
    >>> triangulation = spat.Delaunay(pts)
    >>> triangles = pts[triangulation.simplices]
    >>> triangles
    array([[[6, 7],
            [3, 5],
            [9, 3]],
    <BLANKLINE>
           [[3, 5],
            [4, 1],
            [9, 3]],
    <BLANKLINE>
           [[4, 1],
            [3, 5],
            [0, 1]]])
    >>> a_pts = triangles[:, 0, :]
    >>> b_pts = triangles[:, 1, :]
    >>> c_pts = triangles[:, 2, :]
    >>> radii = r_circumcircle_triangle(a_pts, b_pts, c_pts)
    >>> geoms = alpha_geoms(alpha, triangulation.simplices, radii, pts)
    >>> geoms
    0    POLYGON ((0.00000 1.00000, 3.00000 5.00000, 4....
    dtype: geometry

    """
    from shapely.geometry import LineString
    from shapely.ops import polygonize
    from geopandas import GeoSeries

    triangles_reduced = triangles[radii < 1 / alpha]
    outer_triangulation = get_single_faces(triangles_reduced)
    face_pts = xys[outer_triangulation]
    geoms = GeoSeries(list(polygonize(list(map(LineString, face_pts)))))
    return geoms


@requires("geopandas", "shapely")
def alpha_shape(xys, alpha):
    """Alpha-shape delineation (Edelsbrunner, Kirkpatrick & Seidel, 1983) from a collection of points

    Parameters
    ----------

    xys : ndarray
        (N, 2) array with one point per row and coordinates structured as X and
        Y

    alpha : float
        Alpha value to delineate the alpha-shape

    Returns
    -------

    shapes : GeoSeries
         Polygon(s) resulting from the alpha shape algorithm. The GeoSeries
         object remains so even if only a single polygon is returned. There is
         no CRS included in the object.

    Examples
    --------

    >>> pts = np.array([[0, 1], [3, 5], [4, 1], [6, 7], [9, 3]])
    >>> alpha = 0.1
    >>> poly = alpha_shape(pts, alpha)
    >>> poly
    0    POLYGON ((0.00000 1.00000, 3.00000 5.00000, 6....
    dtype: geometry
    >>> poly.centroid
    0    POINT (4.69048 3.45238)
    dtype: geometry


    References
    ----------

    Edelsbrunner, H., Kirkpatrick, D., & Seidel, R. (1983). On the shape of
        a set of points in the plane. IEEE Transactions on information theory,
        29(4), 551-559.

    """
    if not HAS_JIT:
        warn(NUMBA_WARN)
    if xys.shape[0] < 4:
        from shapely import ops, geometry as geom

        return ops.cascaded_union([geom.Point(xy) for xy in xys]).convex_hull.buffer(0)
    triangulation = spat.Delaunay(xys)
    triangles = xys[triangulation.simplices]
    a_pts = triangles[:, 0, :]
    b_pts = triangles[:, 1, :]
    c_pts = triangles[:, 2, :]
    radii = r_circumcircle_triangle(a_pts, b_pts, c_pts)
    del triangles, a_pts, b_pts, c_pts
    geoms = alpha_geoms(alpha, triangulation.simplices, radii, xys)
    return geoms


def _valid_hull(geoms, points):
    """Sanity check within ``alpha_shape_auto()`` to verify the generated alpha
    shape actually contains the original set of points (xys).

    Parameters
    ----------

    geoms : GeoSeries
        See alpha_geoms()

    points : list
        xys parameter cast as shapely.geometry.Point objects

    Returns
    -------

    flag : bool
        Valid hull for alpha shape [True] or not [False]

    """
    flag = True
    # if there is not exactly one polygon
    if geoms.shape[0] != 1:
        return False
    # if any (xys) points do not intersect the polygon
    if HAS_PYGEOS:
        return pygeos.intersects(pygeos.from_shapely(geoms[0]), points).all()
    else:
        for point in points:
            if not point.intersects(geoms[0]):
                return False
        return True


@requires("geopandas", "shapely")
def alpha_shape_auto(
    xys, step=1, verbose=False, return_radius=False, return_circles=False
):
    """Computation of alpha-shape delineation with automated selection of alpha.

    This method uses the algorithm proposed by  Edelsbrunner, Kirkpatrick &
    Seidel (1983) to return the tightest polygon that contains all points in
    `xys`. The algorithm ranks every point based on its radious and iterates
    over each point, checking whether the maximum alpha that would keep the
    point and all the other ones in the set with smaller radii results in a
    single polygon. If that is the case, it moves to the next point;
    otherwise, it retains the previous alpha value and returns the polygon
    as `shapely` geometry.

    Parameters
    ----------

    xys : ndarray
        Nx2 array with one point per row and coordinates structured as X and Y

    step : int
        [Optional. Default=1] Number of points in `xys` to jump ahead after
        checking whether the largest possible alpha that includes the point and
        all the other ones with smaller radii

    verbose : Boolean
        [Optional. Default=False] If True, it prints alpha values being tried at every step.

    Returns
    -------
    poly : shapely.Polygon
         Tightest alpha-shape polygon containing all points in `xys`

    Examples
    --------

    >>> pts = np.array([[0, 1], [3, 5], [4, 1], [6, 7], [9, 3]])
    >>> poly = alpha_shape_auto(pts)
    >>> poly.bounds
    (0.0, 1.0, 9.0, 7.0)
    >>> poly.centroid.x, poly.centroid.y
    (4.690476190476191, 3.4523809523809526)

    References
    ----------

    Edelsbrunner, H., Kirkpatrick, D., & Seidel, R. (1983). On the shape of
        a set of points in the plane. IEEE Transactions on information theory,
        29(4), 551-559.

    """
    if not HAS_JIT:
        warn(NUMBA_WARN)
    from shapely import geometry as geom

    if return_circles:
        return_radius = True
    if xys.shape[0] < 4:
        from shapely import ops

        if xys.shape[0] == 3:
            multipoint = ops.cascaded_union([geom.Point(xy) for xy in xys])
            alpha_shape = multipoint.convex_hull.buffer(0)
        else:
            alpha_shape = geom.Polygon([])
        if xys.shape[0] == 1:
            if return_radius:
                if return_circles:
                    out = [alpha_shape, 0, alpha_shape]
                return alpha_shape, 0
            return alpha_shape
        elif xys.shape[0] == 2:
            if return_radius:
                r = spat.distance.euclidean(xys[0], xys[1]) / 2
                if return_circles:
                    circle = _construct_centers(xys[0], xys[1], r)
                    return [alpha_shape, r, circle]
                return [alpha_shape, r]
            return alpha_shape
        elif return_radius:  # this handles xys.shape[0] == 3
            radius = r_circumcircle_triangle_single(xys[0], xys[1], xys[2])
            if return_circles:
                circles = construct_bounding_circles(alpha_shape, radius)
                return [alpha_shape, radius, circles]
            return [alpha_shape, radius]
        return alpha_shape
    triangulation = spat.Delaunay(xys)
    triangles = xys[triangulation.simplices]
    a_pts = triangles[:, 0, :]
    b_pts = triangles[:, 1, :]
    c_pts = triangles[:, 2, :]
    radii = r_circumcircle_triangle(a_pts, b_pts, c_pts)
    radii[np.isnan(radii)] = 0  # "Line" triangles to be kept for sure
    del triangles, a_pts, b_pts, c_pts
    radii_sorted_i = radii.argsort()
    triangles = triangulation.simplices[radii_sorted_i][::-1]
    radii = radii[radii_sorted_i][::-1]
    geoms_prev = alpha_geoms((1 / radii.max()) - EPS, triangles, radii, xys)
    if HAS_PYGEOS:
        points = pygeos.points(xys)
    else:
        points = [geom.Point(pnt) for pnt in xys]
    if verbose:
        print("Step set to %i" % step)
    for i in range(0, len(radii), step):
        radi = radii[i]
        alpha = (1 / radi) - EPS
        if verbose:
            print("%.2f%% | Trying a = %f" % ((i + 1) / radii.shape[0], alpha))
        geoms = alpha_geoms(alpha, triangles, radii, xys)
        if _valid_hull(geoms, points):
            geoms_prev = geoms
            radi_prev = radi
        else:
            break
    if verbose:
        print(geoms_prev.shape)
    if return_radius:
        out = [geoms_prev[0], radi_prev]
        if return_circles:
            out.append(construct_bounding_circles(out[0], radi_prev))
        return out
    # Return a shapely polygon
    return geoms_prev[0]


def construct_bounding_circles(alpha_shape, radius):
    """Construct the bounding circles for an alpha shape, given the radius
    computed from the `alpha_shape_auto` method.

    Arguments
    ---------
    alpha_shape : shapely.Polygon
        An alpha-hull with the input radius.

    radius : float
        The radius of the input alpha_shape.

    Returns
    -------
    center : numpy.ndarray of shape (n,2)
        The centers of the circles defining the alpha_shape.

    """
    coordinates = list(alpha_shape.boundary.coords)
    n_coordinates = len(coordinates)
    centers = []
    for i in range(n_coordinates - 1):
        a, b = coordinates[i], coordinates[i + 1]
        centers.append(_construct_centers(a, b, radius))
    return centers


@jit(nopython=True)
def _construct_centers(a, b, radius):
    midpoint_x = (a[0] + b[0]) * 0.5
    midpoint_y = (a[1] + b[1]) * 0.5
    d = ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
    if b[0] - a[0] == 0:
        m = np.inf
        axis_rotation = np.pi / 2
    else:
        m = (b[1] - a[1]) / (b[0] - a[0])
        axis_rotation = np.arctan(m)
    # altitude is perpendicular bisector of AB
    interior_angle = np.arccos(0.5 * d / radius)
    chord = np.sin(interior_angle) * radius

    dx = chord * np.sin(axis_rotation)
    dy = chord * np.cos(axis_rotation)

    up_x = midpoint_x - dx
    up_y = midpoint_y + dy
    down_x = midpoint_x + dx
    down_y = midpoint_y - dy

    # sign gives us direction of point, since
    # shapely shapes are clockwise-defined
    sign = np.sign((b[0] - a[0]) * (up_y - a[1]) - (b[1] - a[1]) * (up_x - a[0]))
    if sign == 1:
        return up_x, up_y
    else:
        return down_x, down_y


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import time
    import geopandas as gpd

    plt.close("all")
    xys = np.random.random((1000, 2))
    t0 = time.time()
    geoms = alpha_shape_auto(xys, 1)
    t1 = time.time()
    print("%.2f Seconds to run algorithm" % (t1 - t0))
    f, ax = plt.subplots(1)
    gpd.GeoDataFrame({"geometry": [geoms]}).plot(ax=ax, color="orange", alpha=0.5)
    ax.scatter(xys[:, 0], xys[:, 1], s=0.1)
    plt.show()

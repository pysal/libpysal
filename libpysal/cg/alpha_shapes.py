import numba as nb
import numpy as np
import scipy.spatial as spat
from shapely.geometry import LineString
from shapely.ops import polygonize
from geopandas import GeoSeries

@nb.jit
def nb_dist(x, y):
    sum = 0
    for x_i, y_i in zip(x, y):
        sum += (x_i - y_i)**2
    dist = np.sqrt(sum)
    return dist

@nb.jit
def r_circumcircle_triangle_single(a, b, c):
    # https://www.mathopenref.com/trianglecircumcircle.html
    ab = nb_dist(a, b)
    bc = nb_dist(b, c)
    ca = nb_dist(c, a)

    num = ab * bc * ca
    den = np.sqrt( (ab + bc + ca) * \
                   (bc + ca - ab) * \
                   (ca + ab - bc) * \
                   (ab + bc - ca) )
    if den == 0:
        print(("Warning: Zero denominator found when calculating"\
               " circumcircles"))
        return 0
    else:
        r = num / den
        return r

@nb.jit(nopython=True)
def r_circumcircle_triangle(a_s, b_s, c_s):
    len_a = len(a_s)
    r2 = np.zeros( (len_a,) )
    for i in range(len_a):
        r2[i] = r_circumcircle_triangle_single(a_s[i], 
                                               b_s[i], 
                                               c_s[i])
    return r2

@nb.jit
def get_faces(triangle):
    faces = np.zeros((3, 2))
    for i, (i0, i1) in enumerate([(0, 1), (1, 2), (2, 0)]):
        faces[i] = triangle[i0], triangle[i1]
    return faces

@nb.jit
def build_faces(faces, triangles_is, 
        num_triangles, num_faces_single):
    for i in range(num_triangles):
        from_i = num_faces_single * i
        to_i = num_faces_single * (i+1)
        faces[from_i: to_i] = get_faces(triangles_is[i])
    return faces

@nb.jit
def nb_mask_faces(mask, faces):
    for k in range(faces.shape[0]-1):
        if mask[k]:
            if np.all(faces[k] == faces[k+1]):
                mask[k] = False
                mask[k+1] = False
    return faces[mask]

def get_single_faces(triangles_is):
    num_faces_single = 3
    num_triangles = triangles_is.shape[0]
    num_faces = num_triangles * num_faces_single
    faces = np.zeros((num_faces, 2), np.int_)
    mask = np.ones((num_faces,), np.bool_)

    faces = build_faces(faces, triangles_is, 
                        num_triangles, num_faces_single)

    orderlist = ["x{}".format(i) for i in range(faces.shape[1])]
    dtype_list = [(el, faces.dtype.str) for el in orderlist]
    faces.sort(axis=1)                  # Not sure why is required to
    faces.view(dtype_list).sort(axis=0) # sort by rows first in 2D
    single_faces = nb_mask_faces(mask, faces)
    return single_faces

def alpha_geoms(alpha, triangles, radii, xys):
    triangles_reduced = triangles[radii < 1/alpha]
    outer_triangulation = get_single_faces(triangles_reduced)
    face_pts = xys[outer_triangulation]
    geoms = GeoSeries(list(polygonize(list(map(LineString, 
                                                   face_pts)))))
    return geoms

def alpha_shape(xys, alpha):
    '''
    Alpha-shape delineation from a collection of points
    ...

    Arguments
    ---------
    xys     : ndarray
              Nx2 array with one point per row and coordinates structured as X
              and Y
    alpha   : float
              Alpha value to delineate the alpha-shape

    Returns
    -------
    shapes  : GeoSeries
              Polygon(s) resulting from the alpha shape algorithm. The
              GeoSeries object remains so even if only a single polygon is
              returned. There is no CRS included in the object.

    Example
    -------

    >>> pts = np.array([[0, 1],
                        [3, 5],
                        [4, 1],
                        [6, 7],
                        [9, 3]])
    >>> alpha = 0.1
    >>> poly = alpha_shape(pts, alpha)
    >>> poly
    0    POLYGON ((0 1, 3 5, 6 7, 9 3, 4 1, 0 1))
    dtype: object
    >>> poly.centroid
    0    POINT (4.690476190476191 3.452380952380953)
    dtype: object
    '''
    triangulation = spat.Delaunay(xys)
    triangles = xys[triangulation.simplices]
    a_pts = triangles[:, 0, :]
    b_pts = triangles[:, 1, :]
    c_pts = triangles[:, 2, :]
    radii = r_circumcircle_triangle(a_pts, b_pts, c_pts)
    del triangles, a_pts, b_pts, c_pts
    geoms = alpha_geoms(alpha, triangulation.simplices, radii, xys)
    return geoms

def alpha_shape_auto(xys, step=1, verbose=False):
    triangulation = spat.Delaunay(xys)
    triangles = xys[triangulation.simplices]
    a_pts = triangles[:, 0, :]
    b_pts = triangles[:, 1, :]
    c_pts = triangles[:, 2, :]
    radii = r_circumcircle_triangle(a_pts, b_pts, c_pts)
    radii[np.isnan(radii)] = 0 # "Line" triangles to be kept for sure
    del triangles, a_pts, b_pts, c_pts
    radii_sorted_i = radii.argsort()
    triangles = triangulation.simplices[radii_sorted_i][::-1]
    radii = radii[radii_sorted_i][::-1]
    geoms_prev = alpha_geoms((1/radii.max())-1e-10, triangles, radii, xys)
    xys_bb = np.array([*xys.min(axis=0), *xys.max(axis=0)])
    if verbose:
        print('Step set to %i'%step)
    for i in range(0, len(radii), step):
        radi = radii[i]
        alpha = (1 / radi) - 1e-100
        if verbose:
            print('%.2f%% | Trying a = %f'\
		  %((i+1)/radii.shape[0], alpha))
        geoms = alpha_geoms(alpha, triangles, radii, xys)
        if (geoms.shape[0] != 1) or not (np.all(xys_bb == geoms.total_bounds)):
            break
        else:
            geoms_prev = geoms
    return geoms_prev[0] # Return a shapely polygon

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import geopandas as gpd
    import time
    plt.close('all')
    xys = np.random.random((1000, 2))
    t0 = time.time()
    geoms = alpha_shape_auto(xys, 1)
    t1 = time.time()
    print('%.2f Seconds to run algorithm'%(t1-t0))
    f, ax = plt.subplots(1)
    geoms.plot(ax=ax, color='orange', alpha=0.5)
    ax.scatter(xys[:, 0], xys[:, 1], s=0.1)
    plt.show()


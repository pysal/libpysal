"""
Convenience functions for the construction of spatial weights based on
contiguity and distance criteria.
"""

__author__ = "Sergio J. Rey <sjsrey@gmail.com> "

from .util import get_points_array_from_shapefile, min_threshold_distance
from ..io.fileio import FileIO as ps_open
from .. import cg
import numpy as np

__all__ = [
    "min_threshold_dist_from_shapefile",
    "build_lattice_shapefile",
    "spw_from_gal",
]


def spw_from_gal(galfile):
    """Sparse ``scipy`` matrix for w from a ``.gal`` file.

    Parameters
    ----------
    galfile : str
        The name of a ``.gal`` file including the file extension.

    Returns
    -------
    spw : libpysal.weights.weights.WSP
        The sparse matrix in CSR format (``scipy.sparse.csr.csr_matrix``) can
        be accessed through ``spw.sparse``.

    Examples
    --------
   
    >>> import libpysal
    >>> spw = libpysal.weights.spw_from_gal(libpysal.examples.get_path("sids2.gal"))
    
    The number of all stored values in ``spw``:
    
    >>> spw.sparse.nnz
    462

    """

    return ps_open(galfile, "r").read(sparse=True)


def min_threshold_dist_from_shapefile(shapefile, radius=None, p=2):
    """Get the maximum nearest neighbor distance
    between observations in the shapefile.

    Parameters
    ----------
    shapefile  : str
        The shapefile name including the ``.shp`` file extension.
    radius : float
        If supplied ``arc_distances`` will be calculated
        based on the given radius and ``p`` will be ignored.
    p : float
        Minkowski `p`-norm distance metric parameter where ``1<=p<=infinity``.
        ``2`` is Euclidean distance and ``1`` is Manhattan distance. Default is ``2 ``.

    Returns
    -------
    d : float
        The maximum nearest neighbor distance between the ``n`` observations.

    Examples
    --------
   
    >>> import libpysal
    >>> md = libpysal.weights.min_threshold_dist_from_shapefile(libpysal.examples.get_path("columbus.shp"))
    >>> md
    0.6188641580768541
    >>> libpysal.weights.min_threshold_dist_from_shapefile(libpysal.examples.get_path("stl_hom.shp"), libpysal.cg.sphere.RADIUS_EARTH_MILES)
    31.846942936393717

    Notes
    -----
    
    This function supports polygon or point shapefiles. For polygon
    shapefiles, distance is based on polygon centroids. Distances are
    defined using coordinates from the shapefile which are assumed to
    be projected and not geographical coordinates.

    """

    points = get_points_array_from_shapefile(shapefile)
    if radius is not None:
        kdt = cg.kdtree.Arc_KDTree(points, radius=radius)
        nn = kdt.query(kdt.data, k=2)
        nnd = nn[0].max(axis=0)[1]
        return nnd
    return min_threshold_distance(points, p)


def build_lattice_shapefile(nrows, ncols, out_file_name):
    """Build a lattice shapefile with ``nrows`` rows and ``ncols`` columns.

    Parameters
    ----------
    nrows : int
        The number of rows.
    ncols : int
        The number of columns.
    out_file_name : str
        The shapefile name including the ``.shp`` file extension.

    """

    if not out_file_name.endswith(".shp"):
        raise ValueError("``out_file_name`` must end with .shp")
    o = ps_open(out_file_name, "w")
    dbf_name = out_file_name.split(".")[0] + ".dbf"
    d = ps_open(dbf_name, "w")
    d.header = ["ID"]
    d.field_spec = [("N", 8, 0)]
    c = 0
    for i in range(ncols):
        for j in range(nrows):
            ll = i, j
            ul = i, j + 1
            ur = i + 1, j + 1
            lr = i + 1, j
            o.write(cg.Polygon([ll, ul, ur, lr, ll]))
            d.write([c])
            c += 1
    d.close()
    o.close()


def _test():
    import doctest

    # the following line could be used to define an alternative to the '<BLANKLINE>' flag
    # doctest.BLANKLINE_MARKER = 'something better than <BLANKLINE>'

    start_suppress = np.get_printoptions()["suppress"]
    np.set_printoptions(suppress=True)
    doctest.testmod()
    np.set_printoptions(suppress=start_suppress)


if __name__ == "__main__":
    _test()

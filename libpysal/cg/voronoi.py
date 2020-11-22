"""
Voronoi tesslation of 2-d point sets.

Adapted from https://gist.github.com/pv/8036995

"""

import numpy as np
from scipy.spatial import Voronoi

__author__ = "Serge Rey <sjsrey@gmail.com>"

__all__ = ["voronoi_frames"]


def voronoi(points, radius=None):
    """Determine finite Voronoi diagram for a 2-d point set.
    See also ``voronoi_regions()``.

    Parameters
    ----------
    points : array_like
        An nx2 array of points.
    radius : float (optional)
        The distance to 'points at infinity'. Default is ``None.``

    Returns
    -------
    vor : tuple
        A two-element tuple consisting of a list and an array. Each element of
        the list contains the sequence of the indices of Voronoi vertices
        composing a Voronoi polygon (region), whereas the array contains
        the Voronoi vertex coordinates.
    
    Examples
    --------
    
    >>> points = [(10.2, 5.1), (4.7, 2.2), (5.3, 5.7), (2.7, 5.3)]
    >>> regions, coordinates = voronoi(points)
    >>> regions
    [[1, 3, 2], [4, 5, 1, 0], [0, 1, 7, 6], [9, 0, 8]]
    
    >>> coordinates
    array([[  4.21783296,   4.08408578],
           [  7.51956025,   3.51807539],
           [  9.4642193 ,  19.3994576 ],
           [ 14.98210684, -10.63503022],
           [ -9.22691341,  -4.58994414],
           [ 14.98210684, -10.63503022],
           [  1.78491801,  19.89803294],
           [  9.4642193 ,  19.3994576 ],
           [  1.78491801,  19.89803294],
           [ -9.22691341,  -4.58994414]])
    
    """

    vor = voronoi_regions(Voronoi(points), radius=radius)

    return vor


def voronoi_regions(vor, radius=None):
    """Finite voronoi regions for a 2-d point set. See also ``voronoi()``.

    Parameters
    ----------
    vor : scipy.spatial.Voronoi
        A planar Voronoi diagram.
    radius : float (optional)
        Distance to 'points at infinity'. Default is ``None.``
    
    Returns
    -------
    regions_vertices : tuple
        A two-element tuple consisting of a list of finite voronoi regions
        and an array Voronoi vertex coordinates.
    
    """

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max() * 2

    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue

        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue

            t = vor.points[p2] - vor.points[p1]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        new_regions.append(new_region.tolist())

    regions_vertices = new_regions, np.asarray(new_vertices)

    return regions_vertices


def as_dataframes(regions, vertices, points):
    """Helper function to store finite Voronoi regions and
    originator points as ``geopandas`` (or ``pandas``) dataframes.

    Parameters
    ----------
    regions : list
        Each element of the list contains sequence of the indexes of
        voronoi vertices composing a vornoi polygon (region).
    vertices : array_like
        The coordinates of the vornoi vertices.
    points : array_like
        The originator points.

    Returns
    -------
    region_df : geopandas.GeoDataFrame
        Finite Voronoi polygons as geometries.
    points_df : geopandas.GeoDataFrame
        Originator points as geometries.
    
    Raises
    ------
    ImportError
        Raised when ``geopandas`` is not available.
    ImportError
        Raised when ``shapely`` is not available.
    
    """

    try:
        import geopandas as gpd
    except ImportError:
        gpd = None

    try:
        from shapely.geometry import Polygon, Point
    except ImportError:
        from .shapes import Polygon, Point

    if gpd is not None:
        region_df = gpd.GeoDataFrame()
        region_df["geometry"] = [Polygon(vertices[region]) for region in regions]

        point_df = gpd.GeoDataFrame()
        point_df["geometry"] = gpd.GeoSeries(Point(pnt) for pnt in points)
    else:
        import pandas as pd

        region_df = pd.DataFrame()
        region_df["geometry"] = [
            Polygon(vertices[region].tolist()) for region in regions
        ]
        point_df = pd.DataFrame()
        point_df["geometry"] = [Point(pnt) for pnt in points]

    return region_df, point_df


def voronoi_frames(points, radius=None, clip="extent"):
    """Composite helper to return Voronoi regions and
    generator points as individual dataframes.

    Parameters
    ----------
    points : array_like
        The originator points.
    radius : float
        The distance to 'points at infinity' used in building voronoi cells.
        Default is ``None``.
    clip : {str, shapely.geometry.Polygon}
        An overloaded option about how to clip the voronoi cells.
        Default is ``'extent'``. Options are as follows.
        
        * ``'none'``/``None`` -- No clip is applied. Voronoi cells may be arbitrarily larger that the source map. Note that this may lead to cells that are many orders of magnitude larger in extent than the original map. Not recommended.
        * ``'bbox'``/``'extent'``/``'bounding box'`` -- Clip the voronoi cells to the bounding box of the input points.
        * ``'chull``/``'convex hull'`` -- Clip the voronoi cells to the convex hull of the input points.
        * ``'ashape'``/``'ahull'`` -- Clip the voronoi cells to the tightest hull that contains all points (e.g. the smallest alphashape, using ``libpysal.cg.alpha_shape_auto``).
        * Polygon -- Clip to an arbitrary Polygon.
    
    tolerance : float
        The percent of map width to use to buffer the extent of the map,
        if clipping (default: ``.01``, or 1%).

    Returns
    -------
    reg_vtx : tuple
        Two ``geopandas.GeoDataFrame`` (or ``pandas.DataFrame`` if ``geopandas``
        is unavailable) objects--``(region_df, points_df)``--of finite 
        Voronoi polygons and the originator points as geometries.

    Notes
    -----

    If ``geopandas`` is not available the return types will be
    ``pandas.DataFrame`` objects, each with a geometry column populated
    with PySAL shapes. If ``geopandas`` is available, return types are
    ``pandas.GeoDataFrame`` objects with a geometry column populated
    with shapely geometry types.

    Examples
    --------
    
    >>> points = [(10.2, 5.1), (4.7, 2.2), (5.3, 5.7), (2.7, 5.3)]
    >>> regions_df, points_df = voronoi_frames(points)
    >>> regions_df.shape
    (4, 1)
    
    >>> regions_df.shape == points_df.shape
    True

    """

    regions, vertices = voronoi(points, radius=radius)
    regions, vertices = as_dataframes(regions, vertices, points)
    if clip:
        regions = clip_voronoi_frames_to_extent(regions, vertices, clip=clip)

    reg_vtx = regions, vertices
    return reg_vtx


def clip_voronoi_frames_to_extent(regions, vertices, clip="extent"):
    """Generate a geopandas.GeoDataFrame of Voronoi cells clipped to
    a specified extent.
    
    Parameters
    ----------
    regions : geopandas.GeoDataFrame
        A (geo)dataframe containing voronoi cells to clip.
    vertices : geopandas.GeoDataFrame
        A (geo)dataframe containing vertices used to build voronoi cells.
    clip : str, shapely.geometry.Polygon
        An overloaded option about how to clip the voronoi cells.
        The options are:
          - 'none'/None: No clip is applied. Voronoi cells may be arbitrarily
            larger that the source map. Note that this may lead to cells that
            are many orders of magnitude larger in extent than
            the original map. Not recommended.
          - 'bbox'/'extent'/'bounding box': Clip the voronoi cells to the
            bounding box of the input points.
          - 'chull'/'convex hull': Clip the voronoi cells to the
            convex hull of the input points.
          - 'ashape'/'ahull': Clip the voronoi cells to the tightest hull that
            contains all points (e.g. the smallest alphashape,
            using ``libpysal.cg.alpha_shape_auto``).
          - Polygon: Clip to an arbitrary Polygon.
    
    Returns
    -------
    clipped_regions : geopandas.GeoDataFrame
        A ``geopandas.GeoDataFrame`` of clipped voronoi regions.
    
    Raises
    ------
    ImportError
        Raised when ``shapely`` is not available.
    ImportError
        Raised when ``geopandas`` is not available.
    ValueError
        Raised when in invalid value for ``clip`` is passed in.
    
    """
    try:
        from shapely.geometry import Polygon
    except ImportError:
        raise ImportError("Shapely is required to clip voronoi regions.")
    try:
        import geopandas
    except ImportError:
        raise ImportError("Geopandas is required to clip voronoi regions.")

    if isinstance(clip, Polygon):
        clipper = geopandas.GeoDataFrame(geometry=[clip])
    elif clip is None:
        return regions
    elif clip.lower() == "none":
        return regions
    elif clip.lower() in ("bounds", "bounding box", "bbox", "extent"):
        min_x, min_y, max_x, max_y = vertices.total_bounds
        bounding_poly = Polygon(
            [
                (min_x, min_y),
                (min_x, max_y),
                (max_x, max_y),
                (max_x, min_y),
                (min_x, min_y),
            ]
        )
        clipper = geopandas.GeoDataFrame(geometry=[bounding_poly])
    elif clip.lower() in ("chull", "convex hull", "convex_hull"):
        clipper = geopandas.GeoDataFrame(
            geometry=[vertices.geometry.unary_union.convex_hull]
        )
    elif clip.lower() in (
        "ahull",
        "alpha hull",
        "alpha_hull",
        "ashape",
        "alpha shape",
        "alpha_shape",
    ):
        from .alpha_shapes import alpha_shape_auto
        from ..weights.distance import get_points_array

        coordinates = get_points_array(vertices.geometry)
        clipper = geopandas.GeoDataFrame(geometry=[alpha_shape_auto(coordinates)])
    else:
        raise ValueError(
            "Clip type '{}' not understood. Try one "
            " of the supported options: [None, 'extent', "
            "'chull', 'ahull'].".format(clip)
        )
    clipped_regions = geopandas.overlay(regions, clipper, how="intersection")
    return clipped_regions

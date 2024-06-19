"""
Voronoi tesslation of 2-d point sets.

Adapted from https://gist.github.com/pv/8036995

"""

import warnings

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import shapely
from packaging.version import Version
from scipy.spatial import Voronoi

__author__ = "Serge Rey <sjsrey@gmail.com>"

__all__ = ["voronoi_frames"]

GPD_GE_013 = Version(gpd.__version__) >= Version("0.13.0")


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
    warnings.warn(
        "The 'voronoi' function is considered private and will be "
        "removed in a future release.",
        FutureWarning,
        stacklevel=2,
    )

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
    warnings.warn(
        "The 'voronoi_regions' function is considered private and will be "
        "removed in a future release.",
        FutureWarning,
        stacklevel=2,
    )

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = np.ptp(vor.points).max() * 2

    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices, strict=True):
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
    warnings.warn(
        "The 'as_dataframes' function is considered private and will be "
        "removed in a future release.",
        FutureWarning,
        stacklevel=2,
    )

    try:
        import geopandas as gpd
    except ImportError:
        gpd = None

    try:
        from shapely.geometry import Point, Polygon
    except ImportError:
        from .shapes import Point, Polygon

    if gpd is not None:
        region_df = gpd.GeoDataFrame(
            geometry=gpd.GeoSeries(Polygon(vertices[region]) for region in regions)
        )
        point_df = gpd.GeoDataFrame(
            geometry=gpd.GeoSeries(Point(pnt) for pnt in points)
        )
    else:
        import pandas as pd

        region_df = pd.DataFrame()
        region_df["geometry"] = [
            Polygon(vertices[region].tolist()) for region in regions
        ]
        point_df = pd.DataFrame()
        point_df["geometry"] = [Point(pnt) for pnt in points]

    return region_df, point_df


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
    warnings.warn(
        "The 'clip_voronoi_frames_to_extent' function is considered private "
        "and will be removed in a future release.",
        FutureWarning,
        stacklevel=2,
    )

    try:
        from shapely.geometry import Polygon
    except ImportError:
        raise ImportError("Shapely is required to clip voronoi regions.") from None
    try:
        import geopandas
    except ImportError:
        raise ImportError("Geopandas is required to clip voronoi regions.") from None

    if isinstance(clip, Polygon):
        clipper = geopandas.GeoDataFrame(geometry=[clip])
    elif clip is None or clip.lower() == "none":
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
        from ..weights.distance import get_points_array
        from .alpha_shapes import alpha_shape_auto

        coordinates = get_points_array(vertices.geometry)
        clipper = geopandas.GeoDataFrame(geometry=[alpha_shape_auto(coordinates)])
    else:
        raise ValueError(
            f"Clip type '{clip}' not understood. Try one of the supported options: "
            "[None, 'extent', 'chull', 'ahull']."
        )
    clipped_regions = geopandas.overlay(regions, clipper, how="intersection")
    return clipped_regions


def voronoi_frames(
    geometry: gpd.GeoSeries | gpd.GeoDataFrame | npt.ArrayLike,
    radius: float | None = None,
    clip: str | shapely.Geometry | None = "bounding_box",
    shrink: float = 0,
    segment: float = 0,
    grid_size: float = 1e-5,
    return_input: bool | None = None,
    as_gdf: bool | None = None,
) -> gpd.GeoSeries:
    """
    Create Voronoi polygons from a GeoSeries of points, lines, or polygons.

    This is a wrapper around ``shapely.voronoi_polygons`` that handles not only
    points but also lines and polygons through their discretization and dissolution
    of the resulting polygons.

    Parameters
    ----------
    geometry : GeoSeries | GeoDataFrame | array_like
        A GeoSeries of points, lines, or polygons or an array of coordinates.
    radius : float, optional
        Deprecated. Has no effect any longer.
    clip : str, shapely.geometry.Polygon, optional
        Polygon used to clip the Voronoi polygons, by default "bounding_box"
        The options are:

        * ``None`` -- No clip is applied. Voronoi cells may be arbitrarily
          larger that the source map. Note that this may lead to cells that are many
          orders of magnitude larger in extent than the original map. Not recommended.
        * ``'bounding_box'`` -- Clip the voronoi cells to the
          bounding box of the input points.
        * ``'convex_hull'`` -- Clip the voronoi cells to the convex hull of
          the input points.
        * ``'alpha_shape'`` -- Clip the voronoi cells to the tightest hull that
          contains all points (e.g. the smallest alpha shape, using
          :func:`libpysal.cg.alpha_shape_auto`).
        * ``shapely.Polygon`` -- Clip to an arbitrary Polygon.

    shrink : float, optional
        Distance for the negative buffer of polygons required when there are polygons
        sharing portion of their exterior, by default 0
    segment : float, optional
        Distance for the segmentation of lines used to add coordinates to lines or
        polygons prior Voronoi tessellation, by default 0
    grid_size : float, optional
        Grid size precision under which the voronoi algorithm is generated,
        by default 1e-5
    return_input : bool, optional
        Whether to return the input geometry, defaults to True
    as_gdf : bool, optional
        Whether to return the output as a GeoDataFrame (True) or GeoSeries (False),
        defaults to True

    Returns
    -------
    GeoSeries | GeoDataFrame | tuple
        GeoSeries of Voronoi polygons with index allowing to link back to the input
    """
    if radius is not None:
        warnings.warn(
            "The 'radius' parameter is deprecated and will be removed in a future "
            "release. It has no effect any longer.",
            FutureWarning,
            stacklevel=2,
        )

    if isinstance(geometry, gpd.GeoDataFrame | gpd.GeoSeries):
        # Check if the input geometry is in a geographic CRS
        if geometry.crs and geometry.crs.is_geographic:
            raise ValueError(
                "Geometry is in a geographic CRS. "
                "Use 'GeoSeries.to_crs()' to re-project geometries to a "
                "projected CRS before using voronoi_polygons.",
            )

        # Set precision of the input geometry (avoids GEOS precision issues)
        objects = shapely.set_precision(geometry.geometry.copy(), grid_size)

        geom_types = objects.geom_type
        mask_poly = geom_types.isin(["Polygon", "MultiPolygon"])
        mask_line = objects.geom_type.isin(["LineString", "MultiLineString"])

        if mask_poly.any():
            # Shrink polygons if required
            if shrink != 0:
                objects[mask_poly] = objects[mask_poly].buffer(
                    -shrink, cap_style=2, join_style=2
                )
            # Segmentize polygons if required
            if segment != 0:
                objects.loc[mask_poly] = shapely.segmentize(objects[mask_poly], segment)

        if mask_line.any():
            if segment != 0:
                objects.loc[mask_line] = shapely.segmentize(objects[mask_line], segment)

            if not GPD_GE_013:
                raise ImportError(
                    "Voronoi tessellation of lines requires geopandas 0.13.0 or later."
                )

            # Remove duplicate coordinates from lines
            objects.loc[mask_line] = (
                objects.loc[mask_line]
                .get_coordinates(index_parts=True)
                .drop_duplicates(keep=False)
                .groupby(level=0)
                .apply(shapely.multipoints)
                .values
            )
    else:
        geometry = np.asarray(geometry)
        objects = geometry = gpd.GeoSeries.from_xy(geometry[:, 0], geometry[:, 1])
        mask_poly = mask_line = np.array([False])

    limit = _get_limit(objects, clip)

    # Compute Voronoi polygons
    voronoi = shapely.voronoi_polygons(
        shapely.GeometryCollection(objects.values), extend_to=limit
    )
    # Get individual polygons out of the collection
    polygons = gpd.GeoSeries(
        shapely.make_valid(shapely.get_parts(voronoi)), crs=geometry.crs
    )

    # temporary fix for libgeos/geos#1062
    if not (polygons.geom_type == "Polygon").all():
        polygons = polygons.explode(ignore_index=True)
        polygons = polygons[polygons.geom_type == "Polygon"]

    # Assign to each input geometry the corresponding Voronoi polygon
    # TODO: check if we still need indexing after shapely/shapely#1968 is released
    if GPD_GE_013:
        ids_objects, ids_polygons = polygons.sindex.query(
            objects, predicate="intersects"
        )
    else:
        ids_objects, ids_polygons = polygons.sindex.query_bulk(
            objects, predicate="intersects"
        )
    if mask_poly.any() or mask_line.any():
        # Dissolve polygons
        polygons = (
            polygons.iloc[ids_polygons]
            .groupby(objects.index.take(ids_objects))
            .agg(shapely.coverage_union_all)
        )
        if geometry.crs is not None:
            polygons = polygons.set_crs(geometry.crs)
    else:
        polygons = polygons.iloc[ids_polygons].reset_index(drop=True)

    # Clip polygons if limit is provided
    if limit is not None:
        to_be_clipped = polygons.sindex.query(limit.boundary, "intersects")
        polygons.iloc[to_be_clipped] = polygons.iloc[to_be_clipped].intersection(limit)

    if as_gdf is None:
        as_gdf = True
        warnings.warn(
            "The 'as_gdf' parameter currently defaults to True but will "
            "default to False in a future release. Set it explicitly to avoid "
            "this warning.",
            FutureWarning,
            stacklevel=2,
        )

    if as_gdf:
        polygons = polygons.to_frame("geometry")
        geometry = geometry.geometry.to_frame("geometry")

    if return_input is None:
        return_input = True
        warnings.warn(
            "The 'return_input' parameter currently defaults to True but will "
            "default to False in a future release. Set it explicitly to avoid "
            "this warning.",
            FutureWarning,
            stacklevel=2,
        )

    if return_input:
        return polygons, geometry

    return polygons


def _get_limit(points, clip):
    if isinstance(clip, shapely.Geometry):
        return clip
    if clip is None or clip is False:
        return None
    if clip.lower() == "none":
        warnings.warn(
            "The 'none' option for the 'clip' parameter is deprecated and will "
            "be removed in a future release. Use None or False instead.",
            FutureWarning,
            stacklevel=3,
        )
        return None
    if clip.lower() in ("bounding_box", "bounds", "bounding box", "bbox", "extent"):
        if clip.lower() != "bounding_box":
            warnings.warn(
                f"The '{clip}' option for the 'clip' parameter is deprecated and "
                "will be removed in a future release. Use 'bounding_box' instead.",
                FutureWarning,
                stacklevel=3,
            )
        return shapely.box(*points.total_bounds)

    if clip.lower() in ("chull", "convex hull", "convex_hull"):
        if clip.lower() != "convex_hull":
            warnings.warn(
                f"The '{clip}' option for the 'clip' parameter is deprecated and "
                "will be removed in a future release. Use 'convex_hull' instead.",
                FutureWarning,
                stacklevel=3,
            )
        return points.unary_union.convex_hull

    if clip.lower() in (
        "ahull",
        "alpha hull",
        "alpha_hull",
        "ashape",
        "alpha shape",
        "alpha_shape",
    ):
        if clip.lower() != "alpha_shape":
            warnings.warn(
                f"The '{clip}' option for the 'clip' parameter is deprecated and "
                "will be removed in a future release. Use 'alpha_shape' instead.",
                FutureWarning,
                stacklevel=3,
            )
        from .alpha_shapes import alpha_shape_auto

        coordinates = shapely.get_coordinates(points.values)
        return alpha_shape_auto(coordinates)

    raise ValueError(
        f"Clip type '{clip}' not understood. Try one of the supported options: "
        "[None, 'bounding_box', 'convex_hull', 'alpha_shape', shapely.Polygon]."
    )

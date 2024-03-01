import geopandas as gpd
import numpy as np
import pytest
import shapely
from geopandas.testing import assert_geoseries_equal
from packaging.version import Version

from ..voronoi import voronoi, voronoi_frames


class TestVoronoi:
    def setup_method(self):
        self.points = [(10.2, 5.1), (4.7, 2.2), (5.3, 5.7), (2.7, 5.3)]
        self.points2 = [(10, 5), (4, 2), (5, 5)]

        self.vertices = [
            [4.21783295711061, 4.084085778781038],
            [7.519560251284979, 3.518075385494004],
            [9.464219298524961, 19.399457604620512],
            [14.982106844470032, -10.63503022227075],
            [-9.226913414477298, -4.58994413837245],
            [14.982106844470032, -10.63503022227075],
            [1.7849180090475505, 19.898032941190912],
            [9.464219298524961, 19.399457604620512],
            [1.7849180090475505, 19.898032941190912],
            [-9.226913414477298, -4.58994413837245],
        ]

        p1 = shapely.Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
        p2 = shapely.Polygon([(0, 1), (0, 2), (1, 2), (1, 1)])
        p3 = shapely.Polygon([(1, 1), (1, 2), (2, 2), (2, 1)])
        p4 = shapely.Polygon([(1, 0), (1, 1), (2, 1), (2, 0)])
        self.polygons = gpd.GeoSeries([p1, p2, p3, p4], crs="EPSG:3857")

        self.lines = gpd.GeoSeries(
            [
                shapely.LineString([(0, 0), (0, 1)]),
                shapely.LineString([(1, 1), (1, 0)]),
            ],
            crs="EPSG:3857",
        )

    def test_voronoi(self):
        with pytest.warns(
            FutureWarning, match="The 'voronoi' function is considered private"
        ):
            regions, vertices = voronoi(self.points)
        assert regions == [[1, 3, 2], [4, 5, 1, 0], [0, 1, 7, 6], [9, 0, 8]]

        np.testing.assert_array_almost_equal(vertices, self.vertices)

    def test_from_array(self):
        geoms = voronoi_frames(self.points2, as_gdf=False, return_input=False)
        expected = gpd.GeoSeries.from_wkt(
            [
                "POLYGON ((7.5 2.5, 7.5 5, 10 5, 10 2, 7.75 2, 7.5 2.5))",
                "POLYGON ((7.5 2.5, 7.75 2, 4 2, 4 3.66666666, 7.5 2.5))",
                "POLYGON ((7.5 2.5, 4 3.66666666, 4 5, 7.5 5, 7.5 2.5))",
            ],
        )
        assert_geoseries_equal(
            shapely.normalize(geoms),
            shapely.normalize(expected),
            check_less_precise=True,
        )

    def test_from_polygons(self):
        geoms = voronoi_frames(
            self.polygons, as_gdf=False, return_input=False, shrink=0.1
        )
        expected = gpd.GeoSeries.from_wkt(
            [
                "POLYGON ((0.5 1, 1 1, 1 0.5, 1 0.1, 0.1 0.1, 0.1 1, 0.5 1))",
                "POLYGON ((1 1.5, 1 1, 0.5 1, 0.1 1, 0.1 1.9, 1 1.9, 1 1.5))",
                "POLYGON ((1 1.5, 1 1.9, 1.9 1.9, 1.9 1, 1.5 1, 1 1, 1 1.5))",
                "POLYGON ((1 0.5, 1 1, 1.5 1, 1.9 1, 1.9 0.1, 1 0.1, 1 0.5))",
            ],
            crs="EPSG:3857",
        )
        assert_geoseries_equal(
            shapely.normalize(geoms),
            shapely.normalize(expected),
            check_less_precise=True,
        )

    @pytest.mark.skipif(
        Version(gpd.__version__) < Version("0.13.0"),
        reason="requires geopandas>=0.13.0",
    )
    def test_from_lines(self):
        geoms = voronoi_frames(
            self.lines, as_gdf=False, return_input=False, segment=0.1
        )
        expected = gpd.GeoSeries.from_wkt(
            [
                "POLYGON ((0.5 0.95, 0.5 0, 0 0, 0 1, 0.5 0.95))",
                "POLYGON ((0.5 0.05, 0.5 1, 1 1, 1 0, 0.5 0.05))",
            ],
            crs="EPSG:3857",
        )
        assert_geoseries_equal(geoms.simplify(0.1), expected, check_less_precise=True)

    @pytest.mark.skipif(
        Version(gpd.__version__) >= Version("0.13.0"),
        reason="requires geopandas<0.13.0",
    )
    def test_from_lines_import_error(self):
        with pytest.raises(
            ImportError,
            match="Voronoi tessellation of lines requires geopandas 0.13.0 or later.",
        ):
            voronoi_frames(self.lines, as_gdf=False, return_input=False, segment=0.1)

    def test_clip_none(self):
        geoms = voronoi_frames(
            self.points2, as_gdf=False, return_input=False, clip=None
        )
        expected = gpd.GeoSeries.from_wkt(
            [
                "POLYGON ((16 11, 16 -4, 10.75 -4, 7.5 2.5, 7.5 11, 16 11))",
                "POLYGON ((-2 -4, -2 5.6666666, 7.5 2.5, 10.75 -4, -2 -4))",
                "POLYGON ((-2 11, 7.5 11, 7.5 2.5, -2 5.66666666, -2 11))",
            ],
        )
        assert_geoseries_equal(
            shapely.normalize(geoms),
            shapely.normalize(expected),
            check_less_precise=True,
        )

    def test_clip_chull(self):
        geoms = voronoi_frames(
            self.points2, as_gdf=False, return_input=False, clip="convex_hull"
        )
        expected = gpd.GeoSeries.from_wkt(
            [
                "POLYGON ((7.5 5, 10 5, 7.5 3.75, 7.5 5))",
                "POLYGON ((6 3, 4 2, 4.5 3.5, 6 3))",
                "POLYGON ((7.5 3.75, 6 3, 4.5 3.5, 5 5, 7.5 5, 7.5 3.75))",
            ],
        )
        assert_geoseries_equal(
            shapely.normalize(geoms),
            shapely.normalize(expected),
            check_less_precise=True,
        )

    def test_clip_ahull(self):
        geoms = voronoi_frames(
            self.points2, as_gdf=False, return_input=False, clip="alpha_shape"
        )
        expected = gpd.GeoSeries.from_wkt(
            [
                "POLYGON ((7.5 5, 10 5, 7.5 3.75, 7.5 5))",
                "POLYGON ((6 3, 4 2, 4.5 3.5, 6 3))",
                "POLYGON ((7.5 3.75, 6 3, 4.5 3.5, 5 5, 7.5 5, 7.5 3.75))",
            ],
        )
        assert_geoseries_equal(
            shapely.normalize(geoms),
            shapely.normalize(expected),
            check_less_precise=True,
        )

    def test_clip_polygon(self):
        geoms = voronoi_frames(
            self.points2,
            as_gdf=False,
            return_input=False,
            clip=shapely.box(-10, -10, 10, 10),
        )
        expected = gpd.GeoSeries.from_wkt(
            [
                "POLYGON ((7.5 2.5, 7.5 10, 10 10, 10 -2.5, 7.5 2.5))",
                "POLYGON ("
                "(-10 8.333333, 7.5 2.5, 10 -2.5, 10 -10, -10 -10, -10 8.333333))",
                "POLYGON ((7.5 2.5, -10 8.33333333, -10 10, 7.5 10, 7.5 2.5))",
            ],
        )
        assert_geoseries_equal(
            shapely.normalize(geoms),
            shapely.normalize(expected),
            check_less_precise=True,
        )

    def test_clip_error(self):
        with pytest.raises(ValueError, match="Clip type 'invalid' not understood."):
            voronoi_frames(self.points2, clip="invalid")

    def test_as_gdf(self):
        geoms, polys = voronoi_frames(self.polygons, as_gdf=True, return_input=True)
        assert isinstance(geoms, gpd.GeoDataFrame)
        assert isinstance(polys, gpd.GeoDataFrame)

        with pytest.warns(
            FutureWarning,
            match="The 'as_gdf' parameter currently defaults to True but will",
        ):
            voronoi_frames(self.points2, return_input=True)

    def test_return_input(self):
        geoms, polys = voronoi_frames(self.polygons, return_input=True, as_gdf=False)
        assert isinstance(geoms, gpd.GeoSeries)
        assert polys is self.polygons

        with pytest.warns(
            FutureWarning,
            match="The 'return_input' parameter currently defaults to True but will",
        ):
            voronoi_frames(self.points2, as_gdf=True)

    def test_radius(self):
        with pytest.warns(FutureWarning, match="The 'radius' parameter is deprecated"):
            voronoi_frames(self.points2, radius=1)

    @pytest.mark.parametrize("clip", ["none", "bounds", "chull", "ahull"])
    def test_deprecated_clip(self, clip):
        with pytest.warns(
            FutureWarning,
            match=f"The '{clip}' option for the 'clip' parameter is deprecated",
        ):
            voronoi_frames(self.points2, clip=clip)

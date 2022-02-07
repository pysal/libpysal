from unittest import TestCase, skipIf
from ...examples import get_path
from ..alpha_shapes import alpha_shape, alpha_shape_auto
import numpy as np
import os

try:
    import geopandas
    from shapely import geometry

    GEOPANDAS_EXTINCT = False
except ImportError:
    GEOPANDAS_EXTINCT = True

this_directory = os.path.dirname(__file__)


@skipIf(GEOPANDAS_EXTINCT, "Geopandas is missing, so test will not run.")
class Test_Alpha_Shapes(TestCase):
    def setUp(self):
        eberly = geopandas.read_file(get_path("eberly_net.shp"))
        eberly_vertices = eberly.geometry.apply(
            lambda x: np.hstack(x.xy).reshape(2, 2).T
        ).values
        eberly_vertices = np.vstack(eberly_vertices)
        self.vertices = eberly_vertices

        self.a05 = (
            geopandas.read_file(os.path.join(this_directory, "data/alpha_05.gpkg"))
            .geometry.to_numpy()
            .item()
        )
        self.a10 = (
            geopandas.read_file(os.path.join(this_directory, "data/alpha_tenth.gpkg"))
            .geometry.to_numpy()
            .item()
        )
        self.a2 = (
            geopandas.read_file(os.path.join(this_directory, "data/alpha_fifth.gpkg"))
            .geometry.to_numpy()
            .item()
        )
        self.a25 = (
            geopandas.read_file(os.path.join(this_directory, "data/alpha_fourth.gpkg"))
            .geometry.to_numpy()
            .item()
        )
        self.a25 = (
            geopandas.read_file(os.path.join(this_directory, "data/alpha_fourth.gpkg"))
            .geometry.to_numpy()
            .item()
        )
        circles = geopandas.read_file(
            os.path.join(this_directory, "data/eberly_bounding_circles.gpkg")
        )
        self.circle_radii = circles.radius.iloc[0]
        self.circle_verts = np.column_stack(
            (circles.geometry.x.values, circles.geometry.y.values)
        )

        self.autoalpha = geopandas.read_file(
            os.path.join(this_directory, "data/alpha_auto.gpkg")
        ).geometry[0]

    def test_alpha_shapes(self):
        new_a05 = alpha_shape(self.vertices, 0.05).to_numpy().item()
        new_a10 = alpha_shape(self.vertices, 0.10).to_numpy().item()
        new_a2 = alpha_shape(self.vertices, 0.2).to_numpy().item()
        new_a25 = alpha_shape(self.vertices, 0.25).to_numpy().item()

        assert new_a05.equals(self.a05)
        assert new_a10.equals(self.a10)
        assert new_a2.equals(self.a2)
        assert new_a25.equals(self.a25)

    def test_auto(self):
        auto_alpha = alpha_shape_auto(self.vertices, 5)

        assert self.autoalpha.equals(auto_alpha)

    def test_small_n(self):
        new_singleton = alpha_shape(self.vertices[0].reshape(1, -1), 0.5)
        assert isinstance(new_singleton, geometry.Polygon)
        new_duo = alpha_shape(self.vertices[:1], 0.5)
        assert isinstance(new_duo, geometry.Polygon)
        new_triple = alpha_shape(self.vertices[:2], 0.5)
        assert isinstance(new_triple, geometry.Polygon)
        new_triple = alpha_shape_auto(
            self.vertices[0].reshape(1, -1), return_circles=True
        )
        assert isinstance(new_triple[0], geometry.Polygon)
        new_triple = alpha_shape_auto(self.vertices[:1], return_circles=True)
        assert isinstance(new_triple[0], geometry.Polygon)
        new_triple = alpha_shape_auto(self.vertices[:2], return_circles=True)
        assert isinstance(new_triple[0], geometry.Polygon)

    def test_circles(self):
        ashape, radius, centers = alpha_shape_auto(self.vertices, return_circles=True)
        np.testing.assert_allclose(radius, self.circle_radii)
        np.testing.assert_allclose(centers, self.circle_verts)
    
    def test_holes(self):
        np.random.seed(seed=100)
        points = np.random.rand(1000, 2)*100
        inv_alpha = 3.5
        geoms = alpha_shape(points, 1/inv_alpha)
        assert len(geoms) == 1
        holes = geopandas.GeoSeries(geoms.interiors.explode()).reset_index(drop=True)
        assert len(holes) == 30
        # No holes are within the shape (shape has holes already)
        result = geoms.sindex.query_bulk(holes.centroid, predicate='within')
        assert result.shape == (2,0)
        # All holes are within the exterior
        shell = geopandas.GeoSeries(geoms.exterior.apply(geometry.Polygon))
        within, outside = shell.sindex.query_bulk(holes.centroid, predicate='within')
        assert (outside == 0).all()
        np.testing.assert_array_equal(within, np.arange(30))
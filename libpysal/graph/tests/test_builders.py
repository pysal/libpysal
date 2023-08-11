import pytest

import geopandas as gpd
import pandas as pd
import geodatasets
from libpysal import graph

TRIANGULATIONS = ["delaunay", "gabriel", "relative_neighborhood", "voronoi"]

"""
This file tests Graph initialisation from various build_* constructors. The correctness
of the underlying data shall be tested in respective constructor test suites.
"""


class TestContiguity:
    def setup_method(self):
        self.gdf = gpd.read_file(geodatasets.get_path("nybb"))
        self.gdf_str = self.gdf.set_index("BoroName")

    def test_vertex_intids(self):
        G = graph.Graph.build_contiguity(self.gdf)

        assert pd.api.types.is_numeric_dtype(G._adjacency.index.dtype)
        assert pd.api.types.is_numeric_dtype(G._adjacency.neighbor.dtype)
        assert pd.api.types.is_numeric_dtype(G._adjacency.weight.dtype)

    def test_vertex_strids(self):
        G = graph.Graph.build_contiguity(self.gdf_str)

        assert pd.api.types.is_string_dtype(G._adjacency.index.dtype)
        assert pd.api.types.is_string_dtype(G._adjacency.neighbor.dtype)
        assert pd.api.types.is_numeric_dtype(G._adjacency.weight.dtype)

    def test_rook_intids(self):
        G = graph.Graph.build_contiguity(self.gdf, strict=True)

        assert pd.api.types.is_numeric_dtype(G._adjacency.index.dtype)
        assert pd.api.types.is_numeric_dtype(G._adjacency.neighbor.dtype)
        assert pd.api.types.is_numeric_dtype(G._adjacency.weight.dtype)

    def test_rook_strids(self):
        G = graph.Graph.build_contiguity(self.gdf_str, strict=True)

        assert pd.api.types.is_string_dtype(G._adjacency.index.dtype)
        assert pd.api.types.is_string_dtype(G._adjacency.neighbor.dtype)
        assert pd.api.types.is_numeric_dtype(G._adjacency.weight.dtype)

    def test_queen_intids(self):
        G = graph.Graph.build_contiguity(self.gdf, rook=False, strict=True)

        assert pd.api.types.is_numeric_dtype(G._adjacency.index.dtype)
        assert pd.api.types.is_numeric_dtype(G._adjacency.neighbor.dtype)
        assert pd.api.types.is_numeric_dtype(G._adjacency.weight.dtype)

    def test_queen_strids(self):
        G = graph.Graph.build_contiguity(self.gdf_str, rook=False, strict=True)

        assert pd.api.types.is_string_dtype(G._adjacency.index.dtype)
        assert pd.api.types.is_string_dtype(G._adjacency.neighbor.dtype)
        assert pd.api.types.is_numeric_dtype(G._adjacency.weight.dtype)


class TestTriangulation:
    def setup_method(self):
        gdf = gpd.read_file(geodatasets.get_path("geoda liquor_stores")).explode(
            ignore_index=True
        )
        self.gdf = gdf[~gdf.geometry.duplicated()]
        self.gdf_str = self.gdf.set_index("placeid")

    @pytest.mark.parametrize("method", TRIANGULATIONS)
    def test_triangulation_intids(self, method):
        G = graph.Graph.build_triangulation(self.gdf, method)

        assert pd.api.types.is_numeric_dtype(G._adjacency.index.dtype)
        assert pd.api.types.is_numeric_dtype(G._adjacency.neighbor.dtype)
        assert pd.api.types.is_numeric_dtype(G._adjacency.weight.dtype)

    @pytest.mark.parametrize("method", TRIANGULATIONS)
    def test_triangulation_strids(self, method):
        G = graph.Graph.build_triangulation(self.gdf_str, method)

        assert pd.api.types.is_string_dtype(G._adjacency.index.dtype)
        assert pd.api.types.is_string_dtype(G._adjacency.neighbor.dtype)
        assert pd.api.types.is_numeric_dtype(G._adjacency.weight.dtype)

    @pytest.mark.parametrize("method", TRIANGULATIONS)
    def test_triangulation_intids_kernel(self, method):
        G = graph.Graph.build_triangulation(
            self.gdf, method, kernel="parabolic", bandwidth=7500
        )
        assert pd.api.types.is_numeric_dtype(G._adjacency.index.dtype)
        assert pd.api.types.is_numeric_dtype(G._adjacency.neighbor.dtype)
        assert pd.api.types.is_numeric_dtype(G._adjacency.weight.dtype)

    @pytest.mark.parametrize("method", TRIANGULATIONS)
    def test_triangulation_strids_kernel(self, method):
        G = graph.Graph.build_triangulation(
            self.gdf_str, method, kernel="parabolic", bandwidth=7500
        )

        assert pd.api.types.is_string_dtype(G._adjacency.index.dtype)
        assert pd.api.types.is_string_dtype(G._adjacency.neighbor.dtype)
        assert pd.api.types.is_numeric_dtype(G._adjacency.weight.dtype)

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="Method 'invalid' is not supported"):
            graph.Graph.build_triangulation(
                self.gdf, method="invalid", kernel="parabolic", bandwidth=7500
            )


class TestKernel:
    def setup_method(self):
        self.gdf = gpd.read_file(geodatasets.get_path("geoda liquor_stores")).explode(
            ignore_index=True
        )
        self.gdf_str = self.gdf.set_index("placeid")

    def test_kernel_intids(self):
        G = graph.Graph.build_kernel(self.gdf)

        assert pd.api.types.is_numeric_dtype(G._adjacency.index.dtype)
        assert pd.api.types.is_numeric_dtype(G._adjacency.neighbor.dtype)
        assert pd.api.types.is_numeric_dtype(G._adjacency.weight.dtype)

    def test_kernel_strids(self):
        G = graph.Graph.build_kernel(self.gdf_str)

        assert pd.api.types.is_string_dtype(G._adjacency.index.dtype)
        assert pd.api.types.is_string_dtype(G._adjacency.neighbor.dtype)
        assert pd.api.types.is_numeric_dtype(G._adjacency.weight.dtype)

    def test_knn_intids(self):
        G = graph.Graph.build_knn(self.gdf, k=3)

        assert pd.api.types.is_numeric_dtype(G._adjacency.index.dtype)
        assert pd.api.types.is_numeric_dtype(G._adjacency.neighbor.dtype)
        assert pd.api.types.is_numeric_dtype(G._adjacency.weight.dtype)

    def test_knn_strids(self):
        G = graph.Graph.build_kernel(self.gdf_str, k=3)

        assert pd.api.types.is_string_dtype(G._adjacency.index.dtype)
        assert pd.api.types.is_string_dtype(G._adjacency.neighbor.dtype)
        assert pd.api.types.is_numeric_dtype(G._adjacency.weight.dtype)

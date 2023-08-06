import pytest

import geopandas as gpd
import pandas as pd
import geodatasets
from libpysal import graph

TRIANGULATIONS = ["delaunay", "gabriel", "relative_neighborhood", "voronoi"]


class TestTriangulation:
    def setup_method(self):
        self.gdf = gpd.read_file(geodatasets.get_path("geoda liquor_stores")).explode(
            ignore_index=True
        )
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

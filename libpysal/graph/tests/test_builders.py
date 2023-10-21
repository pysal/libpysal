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

        assert pd.api.types.is_numeric_dtype(G._adjacency.index.dtypes["focal"])
        assert pd.api.types.is_numeric_dtype(G._adjacency.index.dtypes["neighbor"])
        assert pd.api.types.is_numeric_dtype(G._adjacency.dtype)

    def test_vertex_strids(self):
        G = graph.Graph.build_contiguity(self.gdf_str)

        assert pd.api.types.is_string_dtype(G._adjacency.index.dtypes["focal"])
        assert pd.api.types.is_string_dtype(G._adjacency.index.dtypes["neighbor"])
        assert pd.api.types.is_numeric_dtype(G._adjacency.dtype)

    def test_rook_intids(self):
        G = graph.Graph.build_contiguity(self.gdf, strict=True)

        assert pd.api.types.is_numeric_dtype(G._adjacency.index.dtypes["focal"])
        assert pd.api.types.is_numeric_dtype(G._adjacency.index.dtypes["neighbor"])
        assert pd.api.types.is_numeric_dtype(G._adjacency.dtype)

    def test_rook_strids(self):
        G = graph.Graph.build_contiguity(self.gdf_str, strict=True)

        assert pd.api.types.is_string_dtype(G._adjacency.index.dtypes["focal"])
        assert pd.api.types.is_string_dtype(G._adjacency.index.dtypes["neighbor"])
        assert pd.api.types.is_numeric_dtype(G._adjacency.dtype)

    def test_queen_intids(self):
        G = graph.Graph.build_contiguity(self.gdf, rook=False, strict=True)

        assert pd.api.types.is_numeric_dtype(G._adjacency.index.dtypes["focal"])
        assert pd.api.types.is_numeric_dtype(G._adjacency.index.dtypes["neighbor"])
        assert pd.api.types.is_numeric_dtype(G._adjacency.dtype)

    def test_queen_strids(self):
        G = graph.Graph.build_contiguity(self.gdf_str, rook=False, strict=True)

        assert pd.api.types.is_string_dtype(G._adjacency.index.dtypes["focal"])
        assert pd.api.types.is_string_dtype(G._adjacency.index.dtypes["neighbor"])
        assert pd.api.types.is_numeric_dtype(G._adjacency.dtype)

    def test_vertex_intids_perimeter(self):
        G = graph.Graph.build_contiguity(self.gdf, by_perimeter=True)

        assert pd.api.types.is_numeric_dtype(G._adjacency.index.dtypes["focal"])
        assert pd.api.types.is_numeric_dtype(G._adjacency.index.dtypes["neighbor"])
        assert pd.api.types.is_numeric_dtype(G._adjacency.dtype)

    def test_vertex_strid_perimeters(self):
        G = graph.Graph.build_contiguity(self.gdf_str, by_perimeter=True)

        assert pd.api.types.is_string_dtype(G._adjacency.index.dtypes["focal"])
        assert pd.api.types.is_string_dtype(G._adjacency.index.dtypes["neighbor"])
        assert pd.api.types.is_numeric_dtype(G._adjacency.dtype)

    def test_rook_intids_perimeter(self):
        G = graph.Graph.build_contiguity(self.gdf, strict=True, by_perimeter=True)

        assert pd.api.types.is_numeric_dtype(G._adjacency.index.dtypes["focal"])
        assert pd.api.types.is_numeric_dtype(G._adjacency.index.dtypes["neighbor"])
        assert pd.api.types.is_numeric_dtype(G._adjacency.dtype)

    def test_rook_strid_perimeters(self):
        G = graph.Graph.build_contiguity(self.gdf_str, strict=True, by_perimeter=True)

        assert pd.api.types.is_string_dtype(G._adjacency.index.dtypes["focal"])
        assert pd.api.types.is_string_dtype(G._adjacency.index.dtypes["neighbor"])
        assert pd.api.types.is_numeric_dtype(G._adjacency.dtype)

    def test_queen_intid_perimeters(self):
        G = graph.Graph.build_contiguity(
            self.gdf, rook=False, strict=True, by_perimeter=True
        )

        assert pd.api.types.is_numeric_dtype(G._adjacency.index.dtypes["focal"])
        assert pd.api.types.is_numeric_dtype(G._adjacency.index.dtypes["neighbor"])
        assert pd.api.types.is_numeric_dtype(G._adjacency.dtype)

    def test_queen_strids_perimeter(self):
        G = graph.Graph.build_contiguity(
            self.gdf_str, rook=False, strict=True, by_perimeter=True
        )

        assert pd.api.types.is_string_dtype(G._adjacency.index.dtypes["focal"])
        assert pd.api.types.is_string_dtype(G._adjacency.index.dtypes["neighbor"])
        assert pd.api.types.is_numeric_dtype(G._adjacency.dtype)

    def test_block_contiguity(self):
        regimes = ["n", "n", "s", "s", "e", "e", "w", "w", "e", "l"]
        G = graph.Graph.build_block_contiguity(regimes)

        assert pd.api.types.is_numeric_dtype(G._adjacency.index.dtypes["focal"])
        assert pd.api.types.is_numeric_dtype(G._adjacency.index.dtypes["neighbor"])
        assert pd.api.types.is_numeric_dtype(G._adjacency.dtype)

        regimes = pd.Series(
            regimes, index=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
        )
        G = graph.Graph.build_block_contiguity(regimes)
        assert pd.api.types.is_string_dtype(G._adjacency.index.dtypes["focal"])
        assert pd.api.types.is_string_dtype(G._adjacency.index.dtypes["neighbor"])
        assert pd.api.types.is_numeric_dtype(G._adjacency.dtype)

    def test_fuzzy_contiguity_intids(self):
        G = graph.Graph.build_fuzzy_contiguity(self.gdf)

        assert pd.api.types.is_numeric_dtype(G._adjacency.index.dtypes["focal"])
        assert pd.api.types.is_numeric_dtype(G._adjacency.index.dtypes["neighbor"])
        assert pd.api.types.is_numeric_dtype(G._adjacency.dtype)

    def test_fuzzy_contiguity_strids(self):
        G = graph.Graph.build_fuzzy_contiguity(self.gdf_str)

        assert pd.api.types.is_string_dtype(G._adjacency.index.dtypes["focal"])
        assert pd.api.types.is_string_dtype(G._adjacency.index.dtypes["neighbor"])
        assert pd.api.types.is_numeric_dtype(G._adjacency.dtype)


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

        assert pd.api.types.is_numeric_dtype(G._adjacency.index.dtypes["focal"])
        assert pd.api.types.is_numeric_dtype(G._adjacency.index.dtypes["neighbor"])
        assert pd.api.types.is_numeric_dtype(G._adjacency.dtype)

    @pytest.mark.parametrize("method", TRIANGULATIONS)
    def test_triangulation_strids(self, method):
        G = graph.Graph.build_triangulation(self.gdf_str, method)

        assert pd.api.types.is_string_dtype(G._adjacency.index.dtypes["focal"])
        assert pd.api.types.is_string_dtype(G._adjacency.index.dtypes["neighbor"])
        assert pd.api.types.is_numeric_dtype(G._adjacency.dtype)

    @pytest.mark.parametrize("method", TRIANGULATIONS)
    def test_triangulation_intids_kernel(self, method):
        G = graph.Graph.build_triangulation(
            self.gdf, method, kernel="parabolic", bandwidth=7500
        )
        assert pd.api.types.is_numeric_dtype(G._adjacency.index.dtypes["focal"])
        assert pd.api.types.is_numeric_dtype(G._adjacency.index.dtypes["neighbor"])
        assert pd.api.types.is_numeric_dtype(G._adjacency.dtype)

    @pytest.mark.parametrize("method", TRIANGULATIONS)
    def test_triangulation_strids_kernel(self, method):
        G = graph.Graph.build_triangulation(
            self.gdf_str, method, kernel="parabolic", bandwidth=7500
        )

        assert pd.api.types.is_string_dtype(G._adjacency.index.dtypes["focal"])
        assert pd.api.types.is_string_dtype(G._adjacency.index.dtypes["neighbor"])
        assert pd.api.types.is_numeric_dtype(G._adjacency.dtype)

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

        assert pd.api.types.is_numeric_dtype(G._adjacency.index.dtypes["focal"])
        assert pd.api.types.is_numeric_dtype(G._adjacency.index.dtypes["neighbor"])
        assert pd.api.types.is_numeric_dtype(G._adjacency.dtype)

    def test_kernel_strids(self):
        G = graph.Graph.build_kernel(self.gdf_str)

        assert pd.api.types.is_string_dtype(G._adjacency.index.dtypes["focal"])
        assert pd.api.types.is_string_dtype(G._adjacency.index.dtypes["neighbor"])
        assert pd.api.types.is_numeric_dtype(G._adjacency.dtype)

    def test_knn_intids(self):
        G = graph.Graph.build_knn(self.gdf, k=3)

        assert pd.api.types.is_numeric_dtype(G._adjacency.index.dtypes["focal"])
        assert pd.api.types.is_numeric_dtype(G._adjacency.index.dtypes["neighbor"])
        assert pd.api.types.is_numeric_dtype(G._adjacency.dtype)

    def test_knn_strids(self):
        G = graph.Graph.build_kernel(self.gdf_str, k=3)

        assert pd.api.types.is_string_dtype(G._adjacency.index.dtypes["focal"])
        assert pd.api.types.is_string_dtype(G._adjacency.index.dtypes["neighbor"])
        assert pd.api.types.is_numeric_dtype(G._adjacency.dtype)


class TestDistanceBand:
    def setup_method(self):
        df = gpd.read_file(geodatasets.get_path("nybb"))
        self.gdf = df.set_geometry(df.centroid)
        self.gdf_str = self.gdf.set_index("BoroName")

    def test_distance_band_intids(self):
        G = graph.Graph.build_distance_band(self.gdf, 50000)

        assert pd.api.types.is_numeric_dtype(G._adjacency.index.dtypes["focal"])
        assert pd.api.types.is_numeric_dtype(G._adjacency.index.dtypes["neighbor"])
        assert pd.api.types.is_numeric_dtype(G._adjacency.dtype)

    def test_distance_band_strids(self):
        G = graph.Graph.build_distance_band(self.gdf_str, 50000)

        assert pd.api.types.is_string_dtype(G._adjacency.index.dtypes["focal"])
        assert pd.api.types.is_string_dtype(G._adjacency.index.dtypes["neighbor"])
        assert pd.api.types.is_numeric_dtype(G._adjacency.dtype)

    def test_distance_band_intids_weighted(self):
        G = graph.Graph.build_distance_band(self.gdf, 50000, binary=False)

        assert pd.api.types.is_numeric_dtype(G._adjacency.index.dtypes["focal"])
        assert pd.api.types.is_numeric_dtype(G._adjacency.index.dtypes["neighbor"])
        assert pd.api.types.is_numeric_dtype(G._adjacency.dtype)

    def test_distance_band_strids_weighted(self):
        G = graph.Graph.build_distance_band(self.gdf_str, 50000, binary=False)

        assert pd.api.types.is_string_dtype(G._adjacency.index.dtypes["focal"])
        assert pd.api.types.is_string_dtype(G._adjacency.index.dtypes["neighbor"])
        assert pd.api.types.is_numeric_dtype(G._adjacency.dtype)

    def test_distance_band_intids_kernel(self):
        G = graph.Graph.build_distance_band(
            self.gdf, 50000, binary=False, kernel="gaussian"
        )

        assert pd.api.types.is_numeric_dtype(G._adjacency.index.dtypes["focal"])
        assert pd.api.types.is_numeric_dtype(G._adjacency.index.dtypes["neighbor"])
        assert pd.api.types.is_numeric_dtype(G._adjacency.dtype)

    def test_distance_band_strids_kernel(self):
        G = graph.Graph.build_distance_band(
            self.gdf_str, 50000, binary=False, kernel="gaussian"
        )

        assert pd.api.types.is_string_dtype(G._adjacency.index.dtypes["focal"])
        assert pd.api.types.is_string_dtype(G._adjacency.index.dtypes["neighbor"])
        assert pd.api.types.is_numeric_dtype(G._adjacency.dtype)


class TestAdjacency:
    def setup_method(self):
        self.gdf = gpd.read_file(geodatasets.get_path("nybb"))
        self.gdf_str = self.gdf.set_index("BoroName")
        self.expected_adjacency_intid = pd.DataFrame(
            {
                "focal": {
                    0: 0,
                    1: 1,
                    2: 1,
                    3: 1,
                    4: 2,
                    5: 2,
                    6: 3,
                    7: 3,
                    8: 3,
                    9: 4,
                    10: 4,
                },
                "neighbor": {
                    0: 0,
                    1: 2,
                    2: 3,
                    3: 4,
                    4: 1,
                    5: 3,
                    6: 1,
                    7: 2,
                    8: 4,
                    9: 1,
                    10: 3,
                },
                "weight": {
                    0: 0,
                    1: 1,
                    2: 1,
                    3: 1,
                    4: 1,
                    5: 1,
                    6: 1,
                    7: 1,
                    8: 1,
                    9: 1,
                    10: 1,
                },
            }
        )
        self.expected_adjacency_strid = pd.DataFrame(
            {
                "focal": {
                    0: "Staten Island",
                    1: "Queens",
                    2: "Queens",
                    3: "Queens",
                    4: "Brooklyn",
                    5: "Brooklyn",
                    6: "Manhattan",
                    7: "Manhattan",
                    8: "Manhattan",
                    9: "Bronx",
                    10: "Bronx",
                },
                "neighbor": {
                    0: "Staten Island",
                    1: "Brooklyn",
                    2: "Manhattan",
                    3: "Bronx",
                    4: "Queens",
                    5: "Manhattan",
                    6: "Queens",
                    7: "Brooklyn",
                    8: "Bronx",
                    9: "Queens",
                    10: "Manhattan",
                },
                "weight": {
                    0: 0,
                    1: 1,
                    2: 1,
                    3: 1,
                    4: 1,
                    5: 1,
                    6: 1,
                    7: 1,
                    8: 1,
                    9: 1,
                    10: 1,
                },
            }
        )

    def test_adjacency_intids(self):
        G = graph.Graph.from_adjacency(
            self.expected_adjacency_intid,
        )

        assert pd.api.types.is_numeric_dtype(G._adjacency.index.dtypes["focal"])
        assert pd.api.types.is_numeric_dtype(G._adjacency.index.dtypes["neighbor"])
        assert pd.api.types.is_numeric_dtype(G._adjacency.dtype)

    def test_adjacency_strids(self):
        G = graph.Graph.from_adjacency(
            self.expected_adjacency_strid,
        )

        assert pd.api.types.is_string_dtype(G._adjacency.index.dtypes["focal"])
        assert pd.api.types.is_string_dtype(G._adjacency.index.dtypes["neighbor"])
        assert pd.api.types.is_numeric_dtype(G._adjacency.dtype)

    def test_adjacency_rename(self):
        adj = self.expected_adjacency_intid
        adj.columns = ["focal", "neighbor", "cost"]  # no longer named weight
        G = graph.Graph.from_adjacency(adj, weight_col="cost")

    def test_adjacency_wrong(self):
        adj = self.expected_adjacency_intid
        adj.columns = ["focal", "neighbor", "cost"]  # no longer named weight
        try:
            G = graph.Graph.from_adjacency(
                adj,
            )
        except AssertionError as e:
            pass

    def test_adjacency_match_contiguity(self):
        contiguity = graph.Graph.build_contiguity(self.gdf)
        built = graph.Graph.from_adjacency(self.expected_adjacency_intid)
        assert contiguity == built

        contiguity_str = graph.Graph.build_contiguity(self.gdf_str)
        built_str = graph.Graph.from_adjacency(self.expected_adjacency_strid)
        assert contiguity_str == built_str

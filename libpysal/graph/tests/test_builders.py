import sys

import geodatasets
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal
from scipy.sparse import csr_matrix
from shapely import get_coordinates

from libpysal import graph

TRIANGULATIONS = ["delaunay", "gabriel", "relative_neighborhood", "voronoi"]

"""
This file tests Graph initialisation from various build_* constructors. The correctness
of the underlying data shall be tested in respective constructor test suites.
"""


@pytest.mark.network
class TestContiguity:
    def setup_method(self):
        self.gdf = gpd.read_file(geodatasets.get_path("nybb"))
        self.gdf_str = self.gdf.set_index("BoroName")

    def test_vertex_intids(self):
        g = graph.Graph.build_contiguity(self.gdf)

        assert pd.api.types.is_numeric_dtype(g._adjacency.index.dtypes["focal"])
        assert pd.api.types.is_numeric_dtype(g._adjacency.index.dtypes["neighbor"])
        assert pd.api.types.is_numeric_dtype(g._adjacency.dtype)

    def test_vertex_strids(self):
        g = graph.Graph.build_contiguity(self.gdf_str)

        assert pd.api.types.is_string_dtype(g._adjacency.index.dtypes["focal"])
        assert pd.api.types.is_string_dtype(g._adjacency.index.dtypes["neighbor"])
        assert pd.api.types.is_numeric_dtype(g._adjacency.dtype)

    def test_rook_intids(self):
        g = graph.Graph.build_contiguity(self.gdf, strict=True)

        assert pd.api.types.is_numeric_dtype(g._adjacency.index.dtypes["focal"])
        assert pd.api.types.is_numeric_dtype(g._adjacency.index.dtypes["neighbor"])
        assert pd.api.types.is_numeric_dtype(g._adjacency.dtype)

    def test_rook_strids(self):
        g = graph.Graph.build_contiguity(self.gdf_str, strict=True)

        assert pd.api.types.is_string_dtype(g._adjacency.index.dtypes["focal"])
        assert pd.api.types.is_string_dtype(g._adjacency.index.dtypes["neighbor"])
        assert pd.api.types.is_numeric_dtype(g._adjacency.dtype)

    def test_queen_intids(self):
        g = graph.Graph.build_contiguity(self.gdf, rook=False, strict=True)

        assert pd.api.types.is_numeric_dtype(g._adjacency.index.dtypes["focal"])
        assert pd.api.types.is_numeric_dtype(g._adjacency.index.dtypes["neighbor"])
        assert pd.api.types.is_numeric_dtype(g._adjacency.dtype)

    def test_queen_strids(self):
        g = graph.Graph.build_contiguity(self.gdf_str, rook=False, strict=True)

        assert pd.api.types.is_string_dtype(g._adjacency.index.dtypes["focal"])
        assert pd.api.types.is_string_dtype(g._adjacency.index.dtypes["neighbor"])
        assert pd.api.types.is_numeric_dtype(g._adjacency.dtype)

    def test_vertex_intids_perimeter(self):
        g = graph.Graph.build_contiguity(self.gdf, by_perimeter=True)

        assert pd.api.types.is_numeric_dtype(g._adjacency.index.dtypes["focal"])
        assert pd.api.types.is_numeric_dtype(g._adjacency.index.dtypes["neighbor"])
        assert pd.api.types.is_numeric_dtype(g._adjacency.dtype)

    def test_vertex_strid_perimeters(self):
        g = graph.Graph.build_contiguity(self.gdf_str, by_perimeter=True)

        assert pd.api.types.is_string_dtype(g._adjacency.index.dtypes["focal"])
        assert pd.api.types.is_string_dtype(g._adjacency.index.dtypes["neighbor"])
        assert pd.api.types.is_numeric_dtype(g._adjacency.dtype)

    def test_rook_intids_perimeter(self):
        g = graph.Graph.build_contiguity(self.gdf, strict=True, by_perimeter=True)

        assert pd.api.types.is_numeric_dtype(g._adjacency.index.dtypes["focal"])
        assert pd.api.types.is_numeric_dtype(g._adjacency.index.dtypes["neighbor"])
        assert pd.api.types.is_numeric_dtype(g._adjacency.dtype)

    def test_rook_strid_perimeters(self):
        g = graph.Graph.build_contiguity(self.gdf_str, strict=True, by_perimeter=True)

        assert pd.api.types.is_string_dtype(g._adjacency.index.dtypes["focal"])
        assert pd.api.types.is_string_dtype(g._adjacency.index.dtypes["neighbor"])
        assert pd.api.types.is_numeric_dtype(g._adjacency.dtype)

    def test_queen_intid_perimeters(self):
        g = graph.Graph.build_contiguity(
            self.gdf, rook=False, strict=True, by_perimeter=True
        )

        assert pd.api.types.is_numeric_dtype(g._adjacency.index.dtypes["focal"])
        assert pd.api.types.is_numeric_dtype(g._adjacency.index.dtypes["neighbor"])
        assert pd.api.types.is_numeric_dtype(g._adjacency.dtype)

    def test_queen_strids_perimeter(self):
        g = graph.Graph.build_contiguity(
            self.gdf_str, rook=False, strict=True, by_perimeter=True
        )

        assert pd.api.types.is_string_dtype(g._adjacency.index.dtypes["focal"])
        assert pd.api.types.is_string_dtype(g._adjacency.index.dtypes["neighbor"])
        assert pd.api.types.is_numeric_dtype(g._adjacency.dtype)

    def test_block_contiguity(self):
        regimes = ["n", "n", "s", "s", "e", "e", "w", "w", "e", "l"]
        g = graph.Graph.build_block_contiguity(regimes)

        assert pd.api.types.is_numeric_dtype(g._adjacency.index.dtypes["focal"])
        assert pd.api.types.is_numeric_dtype(g._adjacency.index.dtypes["neighbor"])
        assert pd.api.types.is_numeric_dtype(g._adjacency.dtype)

        regimes = pd.Series(
            regimes, index=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
        )
        g = graph.Graph.build_block_contiguity(regimes)
        assert pd.api.types.is_string_dtype(g._adjacency.index.dtypes["focal"])
        assert pd.api.types.is_string_dtype(g._adjacency.index.dtypes["neighbor"])
        assert pd.api.types.is_numeric_dtype(g._adjacency.dtype)

    def test_fuzzy_contiguity_intids(self):
        g = graph.Graph.build_fuzzy_contiguity(self.gdf)

        assert pd.api.types.is_numeric_dtype(g._adjacency.index.dtypes["focal"])
        assert pd.api.types.is_numeric_dtype(g._adjacency.index.dtypes["neighbor"])
        assert pd.api.types.is_numeric_dtype(g._adjacency.dtype)

    def test_fuzzy_contiguity_strids(self):
        g = graph.Graph.build_fuzzy_contiguity(self.gdf_str)

        assert pd.api.types.is_string_dtype(g._adjacency.index.dtypes["focal"])
        assert pd.api.types.is_string_dtype(g._adjacency.index.dtypes["neighbor"])
        assert pd.api.types.is_numeric_dtype(g._adjacency.dtype)

    def test_fuzzy_contiguity_kwargs(self):
        g = graph.Graph.build_fuzzy_contiguity(self.gdf_str, resolution=2)

        assert pd.api.types.is_string_dtype(g._adjacency.index.dtypes["focal"])
        assert pd.api.types.is_string_dtype(g._adjacency.index.dtypes["neighbor"])
        assert pd.api.types.is_numeric_dtype(g._adjacency.dtype)


@pytest.mark.network
class TestTriangulation:
    def setup_method(self):
        gdf = gpd.read_file(geodatasets.get_path("geoda liquor_stores")).explode(
            ignore_index=True
        )
        self.gdf = gdf[~gdf.geometry.duplicated()]
        self.gdf_str = self.gdf.set_index("placeid")

    @pytest.mark.parametrize("method", TRIANGULATIONS)
    def test_triangulation_intids(self, method):
        g = graph.Graph.build_triangulation(self.gdf, method)

        assert pd.api.types.is_numeric_dtype(g._adjacency.index.dtypes["focal"])
        assert pd.api.types.is_numeric_dtype(g._adjacency.index.dtypes["neighbor"])
        assert pd.api.types.is_numeric_dtype(g._adjacency.dtype)

    @pytest.mark.parametrize("method", TRIANGULATIONS)
    def test_triangulation_strids(self, method):
        g = graph.Graph.build_triangulation(self.gdf_str, method)

        assert pd.api.types.is_string_dtype(g._adjacency.index.dtypes["focal"])
        assert pd.api.types.is_string_dtype(g._adjacency.index.dtypes["neighbor"])
        assert pd.api.types.is_numeric_dtype(g._adjacency.dtype)

    @pytest.mark.parametrize("method", TRIANGULATIONS)
    def test_triangulation_intids_kernel(self, method):
        g = graph.Graph.build_triangulation(
            self.gdf, method, kernel="parabolic", bandwidth=7500
        )
        assert pd.api.types.is_numeric_dtype(g._adjacency.index.dtypes["focal"])
        assert pd.api.types.is_numeric_dtype(g._adjacency.index.dtypes["neighbor"])
        assert pd.api.types.is_numeric_dtype(g._adjacency.dtype)

    @pytest.mark.parametrize("method", TRIANGULATIONS)
    def test_triangulation_strids_kernel(self, method):
        g = graph.Graph.build_triangulation(
            self.gdf_str, method, kernel="parabolic", bandwidth=7500
        )

        assert pd.api.types.is_string_dtype(g._adjacency.index.dtypes["focal"])
        assert pd.api.types.is_string_dtype(g._adjacency.index.dtypes["neighbor"])
        assert pd.api.types.is_numeric_dtype(g._adjacency.dtype)

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="Method 'invalid' is not supported"):
            graph.Graph.build_triangulation(
                self.gdf, method="invalid", kernel="parabolic", bandwidth=7500
            )

    def test_delunay_subsets(self):
        delaunay = graph.Graph.build_triangulation(
            self.gdf_str, "delaunay", kernel="identity"
        )
        gabriel = graph.Graph.build_triangulation(
            self.gdf_str, "gabriel", kernel="identity"
        )
        rn = graph.Graph.build_triangulation(
            self.gdf_str, "relative_neighborhood", kernel="identity"
        )
        pd.testing.assert_series_equal(
            gabriel.adjacency, delaunay.adjacency.loc[gabriel.adjacency.index]
        )
        pd.testing.assert_series_equal(
            rn.adjacency, delaunay.adjacency.loc[rn.adjacency.index]
        )


@pytest.mark.network
class TestKernel:
    def setup_method(self):
        self.gdf = gpd.read_file(geodatasets.get_path("geoda liquor_stores")).explode(
            ignore_index=True
        )
        self.gdf_str = self.gdf.set_index("placeid")

    def test_kernel_precompute(self):
        sklearn = pytest.importorskip("sklearn")
        df = gpd.read_file(geodatasets.get_path("nybb"))
        df = df.to_crs(df.estimate_utm_crs())
        distmat = csr_matrix(
            sklearn.metrics.pairwise.euclidean_distances(get_coordinates(df.centroid))
        )
        g = graph.Graph.build_kernel(distmat, metric="precomputed")
        expected = np.array(
            [
                0.07131664,
                0.14998932,
                0.09804811,
                0.0402638,
                0.07131664,
                0.18556845,
                0.17529176,
                0.16394507,
                0.14998932,
                0.18556845,
                0.17495794,
                0.11561449,
                0.09804811,
                0.17529176,
                0.17495794,
                0.19116432,
                0.0402638,
                0.16394507,
                0.11561449,
                0.19116432,
            ]
        )

        assert_array_almost_equal(g.adjacency.values, expected, 3)

    def test_kernel_intids(self):
        g = graph.Graph.build_kernel(self.gdf)

        assert pd.api.types.is_numeric_dtype(g._adjacency.index.dtypes["focal"])
        assert pd.api.types.is_numeric_dtype(g._adjacency.index.dtypes["neighbor"])
        assert pd.api.types.is_numeric_dtype(g._adjacency.dtype)

    def test_kernel_strids(self):
        g = graph.Graph.build_kernel(self.gdf_str)

        assert pd.api.types.is_string_dtype(g._adjacency.index.dtypes["focal"])
        assert pd.api.types.is_string_dtype(g._adjacency.index.dtypes["neighbor"])
        assert pd.api.types.is_numeric_dtype(g._adjacency.dtype)

    def test_knn_intids(self):
        g = graph.Graph.build_knn(self.gdf, k=3, coplanar="jitter")

        assert pd.api.types.is_numeric_dtype(g._adjacency.index.dtypes["focal"])
        assert pd.api.types.is_numeric_dtype(g._adjacency.index.dtypes["neighbor"])
        assert pd.api.types.is_numeric_dtype(g._adjacency.dtype)

    def test_knn_strids(self):
        g = graph.Graph.build_kernel(self.gdf_str, k=3, coplanar="jitter")

        assert pd.api.types.is_string_dtype(g._adjacency.index.dtypes["focal"])
        assert pd.api.types.is_string_dtype(g._adjacency.index.dtypes["neighbor"])
        assert pd.api.types.is_numeric_dtype(g._adjacency.dtype)


@pytest.mark.network
class TestDistanceBand:
    def setup_method(self):
        df = gpd.read_file(geodatasets.get_path("nybb"))
        self.gdf = df.set_geometry(df.centroid)
        self.gdf_str = self.gdf.set_index("BoroName")

    def test_distance_band_intids(self):
        g = graph.Graph.build_distance_band(self.gdf, 50000)

        assert pd.api.types.is_numeric_dtype(g._adjacency.index.dtypes["focal"])
        assert pd.api.types.is_numeric_dtype(g._adjacency.index.dtypes["neighbor"])
        assert pd.api.types.is_numeric_dtype(g._adjacency.dtype)

    def test_distance_band_strids(self):
        g = graph.Graph.build_distance_band(self.gdf_str, 50000)

        assert pd.api.types.is_string_dtype(g._adjacency.index.dtypes["focal"])
        assert pd.api.types.is_string_dtype(g._adjacency.index.dtypes["neighbor"])
        assert pd.api.types.is_numeric_dtype(g._adjacency.dtype)

    def test_distance_band_intids_weighted(self):
        g = graph.Graph.build_distance_band(self.gdf, 50000, binary=False)

        assert pd.api.types.is_numeric_dtype(g._adjacency.index.dtypes["focal"])
        assert pd.api.types.is_numeric_dtype(g._adjacency.index.dtypes["neighbor"])
        assert pd.api.types.is_numeric_dtype(g._adjacency.dtype)

    def test_distance_band_strids_weighted(self):
        g = graph.Graph.build_distance_band(self.gdf_str, 50000, binary=False)

        assert pd.api.types.is_string_dtype(g._adjacency.index.dtypes["focal"])
        assert pd.api.types.is_string_dtype(g._adjacency.index.dtypes["neighbor"])
        assert pd.api.types.is_numeric_dtype(g._adjacency.dtype)

    def test_distance_band_intids_kernel(self):
        g = graph.Graph.build_distance_band(
            self.gdf, 50000, binary=False, kernel="gaussian"
        )

        assert pd.api.types.is_numeric_dtype(g._adjacency.index.dtypes["focal"])
        assert pd.api.types.is_numeric_dtype(g._adjacency.index.dtypes["neighbor"])
        assert pd.api.types.is_numeric_dtype(g._adjacency.dtype)

    def test_distance_band_strids_kernel(self):
        g = graph.Graph.build_distance_band(
            self.gdf_str, 50000, binary=False, kernel="gaussian"
        )

        assert pd.api.types.is_string_dtype(g._adjacency.index.dtypes["focal"])
        assert pd.api.types.is_string_dtype(g._adjacency.index.dtypes["neighbor"])
        assert pd.api.types.is_numeric_dtype(g._adjacency.dtype)


@pytest.mark.network
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
        g = graph.Graph.from_adjacency(
            self.expected_adjacency_intid,
        )

        assert pd.api.types.is_numeric_dtype(g._adjacency.index.dtypes["focal"])
        assert pd.api.types.is_numeric_dtype(g._adjacency.index.dtypes["neighbor"])
        assert pd.api.types.is_numeric_dtype(g._adjacency.dtype)

    def test_adjacency_strids(self):
        g = graph.Graph.from_adjacency(
            self.expected_adjacency_strid,
        )

        assert pd.api.types.is_string_dtype(g._adjacency.index.dtypes["focal"])
        assert pd.api.types.is_string_dtype(g._adjacency.index.dtypes["neighbor"])
        assert pd.api.types.is_numeric_dtype(g._adjacency.dtype)

    def test_adjacency_rename(self):
        adj = self.expected_adjacency_intid
        adj.columns = ["focal", "neighbor", "cost"]  # no longer named weight
        _ = graph.Graph.from_adjacency(adj, weight_col="cost")

    def test_adjacency_wrong(self):
        adj = self.expected_adjacency_intid
        adj.columns = ["focal", "neighbor", "cost"]  # no longer named weight
        with pytest.raises(AssertionError, match='"weight" was given for `weight_col`'):
            graph.Graph.from_adjacency(adj)

    def test_adjacency_match_contiguity(self):
        contiguity = graph.Graph.build_contiguity(self.gdf)
        built = graph.Graph.from_adjacency(self.expected_adjacency_intid)
        assert contiguity == built

        contiguity_str = graph.Graph.build_contiguity(self.gdf_str)
        built_str = graph.Graph.from_adjacency(self.expected_adjacency_strid)
        assert contiguity_str == built_str


class TestMatching:
    def setup_method(self):
        pytest.importorskip("pulp")
        self.gdf = gpd.read_file(geodatasets.get_path("nybb"))
        self.gdf = self.gdf.set_geometry(self.gdf.centroid)
        self.gdf_str = self.gdf.set_index("BoroName")
        self.gdf_str = self.gdf_str.set_geometry(self.gdf_str.centroid)

    def test_matching_intids(self):
        g = graph.Graph.build_spatial_matches(self.gdf, k=1)

        assert pd.api.types.is_numeric_dtype(g._adjacency.index.dtypes["focal"])
        assert pd.api.types.is_numeric_dtype(g._adjacency.index.dtypes["neighbor"])
        assert pd.api.types.is_numeric_dtype(g._adjacency.dtype)

    def test_matching_strids(self):
        g = graph.Graph.build_spatial_matches(self.gdf_str, k=2)

        assert pd.api.types.is_string_dtype(g._adjacency.index.dtypes["focal"])
        assert pd.api.types.is_string_dtype(g._adjacency.index.dtypes["neighbor"])
        assert pd.api.types.is_numeric_dtype(g._adjacency.dtype)


@pytest.mark.network
@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="pandana has dtype issues on windows"
)
class TestTravelNetwork:
    def setup_method(self):
        pandana = pytest.importorskip("pandana")
        import pooch

        self.net_path = pooch.retrieve(
            "https://spatial-ucr.s3.amazonaws.com/osm/metro_networks_8k/17140.h5",
            known_hash=None,
        )
        df = gpd.read_file(geodatasets.get_path("geoda cincinnati")).to_crs(4326)
        self.df = df.set_geometry(df.centroid)
        self.network = pandana.Network.from_hdf5(self.net_path)

    def test_build_travel_network(self):
        g = graph.Graph.build_travel_cost(self.df, self.network, 500)
        assert_array_almost_equal(
            g.adjacency.head(10).to_numpy(),
            np.array(
                [
                    418.28601074,
                    228.23899841,
                    196.0269928,
                    0.0,
                    341.73498535,
                    478.47799683,
                    298.91699219,
                    445.60501099,
                    174.64199829,
                    0.0,
                ]
            ),
        )
        assert g.n == self.df.shape[0]

    def test_build_travel_network_kernel(self):
        g = graph.Graph.build_travel_cost(
            self.df, self.network, 500, kernel="triangular"
        )
        assert_array_almost_equal(
            g.adjacency.head(10).to_numpy(),
            np.array(
                [
                    0.16342798,
                    0.543522,
                    0.60794601,
                    1.0,
                    0.31653003,
                    0.04304401,
                    0.40216602,
                    0.10878998,
                    0.650716,
                    1.0,
                ]
            ),
        )
        assert g.n == self.df.shape[0]

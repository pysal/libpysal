import pandas as pd
import pytest

from libpysal import graph


class TestMultiply:
    def setup_method(self):
        # Simple 3-node chain: 0-1-2
        self.adj = pd.Series(
            [1.0, 1.0, 1.0, 1.0],
            index=pd.MultiIndex.from_tuples(
                [(0, 1), (1, 0), (1, 2), (2, 1)], names=["focal", "neighbor"]
            ),
            name="weight",
        )
        self.g = graph.Graph(self.adj)

    def test_multiply_identity(self):
        """Multiplying a graph by itself squares the weights."""
        result = self.g.multiply(self.g)
        pd.testing.assert_series_equal(
            self.g.adjacency**2, result.adjacency, check_dtype=False
        )

    def test_multiply_by_scaled(self):
        """Multiplying by a 0.5-scaled graph halves all weights."""
        half_adj = self.adj * 0.5
        half_g = graph.Graph(half_adj)
        result = self.g.multiply(half_g)
        pd.testing.assert_series_equal(
            self.g.adjacency * 0.5, result.adjacency, check_dtype=False
        )

    def test_multiply_mismatched_structure_raises(self):
        """Multiplying graphs with different edge structures raises ValueError."""
        other_adj = pd.Series(
            [1.0, 1.0],
            index=pd.MultiIndex.from_tuples(
                [(0, 1), (1, 0)], names=["focal", "neighbor"]
            ),
            name="weight",
        )
        other_g = graph.Graph(other_adj)
        with pytest.raises(ValueError, match="same edge structure"):
            self.g.multiply(other_g)

    def test_multiply_produces_new_graph(self):
        """multiply() must return a new Graph, not mutate self."""
        result = self.g.multiply(self.g)
        assert result is not self.g
        # original is unchanged
        assert (self.g.adjacency == 1.0).all()


class TestBuildSpatiotemporal:
    """Integration tests for Graph.build_spatiotemporal()."""

    def setup_method(self):
        pytest.importorskip("sklearn")
        pytest.importorskip("geopandas")
        import geopandas as gpd
        from shapely.geometry import Point

        # 5 points laid out in a line, spaced 100m apart
        self.gdf = gpd.GeoDataFrame(
            geometry=[Point(i * 100, 0) for i in range(5)],
            crs="EPSG:3857",
        )
        # temporal values: each point 10 days apart
        self.t = pd.Series([0, 10, 20, 30, 40])

    def test_returns_graph(self):
        st = graph.Graph.build_spatiotemporal(
            self.gdf,
            t=self.t,
            spatial_bandwidth=250,
            temporal_bandwidth=15,
            kernel="gaussian",
        )
        assert isinstance(st, graph.Graph)

    def test_weights_are_positive(self):
        st = graph.Graph.build_spatiotemporal(
            self.gdf,
            t=self.t,
            spatial_bandwidth=250,
            temporal_bandwidth=15,
            kernel="gaussian",
        )
        assert (st.adjacency >= 0).all()

    def test_temporal_decay_reduces_weights(self):
        """Larger temporal bandwidth = slower decay = higher weights."""
        st_narrow = graph.Graph.build_spatiotemporal(
            self.gdf,
            t=self.t,
            spatial_bandwidth=250,
            temporal_bandwidth=5,
            kernel="gaussian",
        )
        st_wide = graph.Graph.build_spatiotemporal(
            self.gdf,
            t=self.t,
            spatial_bandwidth=250,
            temporal_bandwidth=100,
            kernel="gaussian",
        )
        # wide temporal bandwidth should give higher average weights
        assert st_wide.adjacency.mean() > st_narrow.adjacency.mean()

    def test_same_edge_structure_as_spatial(self):
        """The spatiotemporal graph should have the same edges as the spatial graph."""
        st = graph.Graph.build_spatiotemporal(
            self.gdf,
            t=self.t,
            spatial_bandwidth=250,
            temporal_bandwidth=15,
            kernel="gaussian",
        )
        spatial_g = graph.Graph.build_kernel(self.gdf, bandwidth=250, kernel="gaussian")
        assert st.adjacency.index.equals(spatial_g.adjacency.index)

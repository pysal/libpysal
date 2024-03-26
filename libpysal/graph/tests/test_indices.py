import geopandas as gpd
import pytest
from geodatasets import get_path

from libpysal import graph

pytest.importorskip("h3")
pytest.importorskip("tobler")


class TestH3:
    def setup_method(self):
        from tobler.util import h3fy

        gdf = gpd.read_file(get_path("geoda guerry"))
        h3_geoms = h3fy(gdf, resolution=4)
        self.h3_ids = h3_geoms.index

    def test_h3(self):
        g = graph.Graph.build_h3(self.h3_ids)
        assert g.n == len(self.h3_ids)
        assert g.pct_nonzero == 1.69921875
        assert len(g) == 1740
        assert g.adjacency.max() == 1

    @pytest.mark.parametrize("order", range(2, 6))
    def test_h3_order(self, order):
        g = graph.Graph.build_h3(self.h3_ids, order)
        assert g.n == len(self.h3_ids)
        assert g.adjacency.max() == order

    def test_h3_binary(self):
        g = graph.Graph.build_h3(self.h3_ids, order=4, weight="binary")
        assert g.n == len(self.h3_ids)
        assert g.adjacency.max() == 1

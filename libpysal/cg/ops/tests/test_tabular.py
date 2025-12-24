import numpy as np

from ....common import requires as _requires
from ....examples import get_path
from ....io import geotable as pdio
from ... import ops as GIS  # noqa: N812
from ...shapes import Polygon


class TestTabular:
    def setup_method(self):
        import pandas as pd

        self.columbus = pdio.read_files(get_path("columbus.shp"))
        grid = [
            Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]),
            Polygon([(0, 1), (0, 2), (1, 2), (1, 1)]),
            Polygon([(1, 2), (2, 2), (2, 1), (1, 1)]),
            Polygon([(1, 1), (2, 1), (2, 0), (1, 0)]),
        ]
        regime = [0, 0, 1, 1]
        ids = list(range(4))
        data = np.array((regime, ids)).T
        self.exdf = pd.DataFrame(data, columns=["regime", "ids"])
        self.exdf["geometry"] = grid

    @_requires("geopandas")
    def test_round_trip(self):
        import geopandas as gpd
        import pandas as pd

        geodf = GIS.tabular.to_gdf(self.columbus)
        assert isinstance(geodf, gpd.GeoDataFrame)
        new_df = GIS.tabular.to_df(geodf)
        assert isinstance(new_df, pd.DataFrame)
        for new, old in zip(new_df.geometry, self.columbus.geometry, strict=True):
            assert new == old

    def test_spatial_join(self):
        pass

    def test_spatial_overlay(self):
        pass

    def test_dissolve(self):
        out = GIS.tabular.dissolve(self.exdf, by="regime")
        assert out[0].area == 2.0
        assert out[1].area == 2.0

        answer_vertices0 = {(0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (1, 0), (0, 0)}
        answer_vertices1 = {(2, 1), (2, 0), (1, 0), (1, 1), (1, 2), (2, 2), (2, 1)}

        s0 = {tuple(map(int, t)) for t in out[0].vertices}
        s1 = {tuple(map(int, t)) for t in out[1].vertices}

        assert s0 == answer_vertices0
        assert s1 == answer_vertices1

    def test_clip(self):
        pass

    def test_erase(self):
        pass

    def test_union(self):
        new_geom = GIS.tabular.union(self.exdf)
        assert new_geom.area == 4

    def test_intersection(self):
        pass

    def test_symmetric_difference(self):
        pass

    def test_difference(self):
        pass

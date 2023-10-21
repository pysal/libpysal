import pytest
import geodatasets
import geopandas
import pandas as pd
from libpysal.graph.base import Graph


class Test_Set_Ops:
    def setup_method(self):
        self.grocs = geopandas.read_file(geodatasets.get_path("geoda groceries"))[
            ["OBJECTID", "geometry"]
        ].explode(ignore_index=True)

        self.distance500 = Graph.build_distance_band(self.grocs, 500)
        self.distance2500 = Graph.build_distance_band(self.grocs, 2500)
        self.knn3 = Graph.build_knn(self.grocs, 3)

        self.distance2500_id = Graph.build_distance_band(
            self.grocs.set_index("OBJECTID"), 2500
        )

    def test_intersects(self):
        assert self.distance2500.intersects(self.knn3)
        assert self.knn3.intersects(self.distance2500)
        assert not self.knn3.intersects(self.distance2500_id)
        assert not self.distance2500.intersects(self.distance2500_id)

    def test_intersection(self):
        result = self.distance2500.intersection(self.knn3)
        assert len(result) == 115
        assert result._adjacency.shape[0] == 185
        pd.testing.assert_index_equal(result.unique_ids, self.distance2500.unique_ids)

    def test_symmetric_difference(self):
        result = self.distance2500.symmetric_difference(self.knn3)
        assert len(result) == 334
        assert result._adjacency.shape[0] == 340
        pd.testing.assert_index_equal(result.unique_ids, self.distance2500.unique_ids)

        with pytest.raises(ValueError, match="Cannot do symmetric difference"):
            self.distance2500_id.symmetric_difference(self.knn3)

    def test_union(self):
        result = self.distance2500.union(self.knn3)
        assert len(result) == 449
        assert result._adjacency.shape[0] == 449
        pd.testing.assert_index_equal(result.unique_ids, self.distance2500.unique_ids)

        with pytest.raises(ValueError, match="Cannot do union"):
            self.distance2500_id.union(self.knn3)

    def test_difference(self):
        result = self.distance2500.difference(self.knn3)
        assert len(result) == 5
        assert result._adjacency.shape[0] == 148
        pd.testing.assert_index_equal(result.unique_ids, self.distance2500.unique_ids)

        result = self.knn3.difference(self.distance2500)
        assert len(result) == 329
        assert result._adjacency.shape[0] == 340
        pd.testing.assert_index_equal(result.unique_ids, self.knn3.unique_ids)

    def test_issubgraph(self):
        assert self.distance500.issubgraph(self.distance2500)
        assert not self.distance2500.issubgraph(self.distance500)
        assert not self.knn3.issubgraph(self.distance2500)
        assert self.knn3.issubgraph(self.knn3)

    def test_equals(self):
        assert self.distance2500.equals(self.distance2500.copy())
        assert not self.distance2500.equals(self.distance2500.transform("r"))

    def test_isomorphic(self):
        pytest.importorskip("networkx")
        assert self.distance2500.isomorphic(self.distance2500_id)

    def test___le__(self):
        assert self.distance500 <= self.distance2500
        assert not self.knn3 <= self.distance2500
        assert self.distance2500 <= self.distance2500

    def test___ge__(self):
        assert self.distance2500 >= self.distance500
        assert not self.knn3 >= self.distance2500
        assert self.distance2500 >= self.distance2500

    def test___lt__(self):
        assert self.distance500 < self.distance2500
        assert not self.knn3 < self.distance2500
        assert not self.distance2500 < self.distance2500

    def test___gt__(self):
        assert self.distance2500 > self.distance500
        assert not self.knn3 > self.distance2500
        assert not self.distance2500 > self.distance2500

    def test___eq__(self):
        assert self.distance2500 == self.distance2500.copy()
        assert not self.distance2500 == self.distance2500_id

    def test___ne__(self):
        assert not self.distance2500 != self.distance2500.copy()
        assert self.distance2500 != self.distance2500_id

    def test___and__(self):
        result = self.distance2500 & self.knn3
        assert len(result) == 115
        assert result._adjacency.shape[0] == 185
        pd.testing.assert_index_equal(result.unique_ids, self.distance2500.unique_ids)

    def test___or__(self):
        result = self.distance2500 | self.knn3
        assert len(result) == 449
        assert result._adjacency.shape[0] == 449
        pd.testing.assert_index_equal(result.unique_ids, self.distance2500.unique_ids)

        with pytest.raises(ValueError, match="Cannot do union"):
            self.distance2500_id | self.knn3

    def test___xor__(self):
        result = self.distance2500 ^ self.knn3
        assert len(result) == 334
        assert result._adjacency.shape[0] == 340
        pd.testing.assert_index_equal(result.unique_ids, self.distance2500.unique_ids)

        with pytest.raises(ValueError, match="Cannot do symmetric difference"):
            self.distance2500_id ^ self.knn3

    def test___iand__(self):
        with pytest.raises(TypeError, match="Graphs are immutable."):
            self.distance2500 &= self.knn3

    def test___ior__(self):
        with pytest.raises(TypeError, match="Graphs are immutable."):
            self.distance2500 |= self.knn3

    def test___len__(self):
        assert len(self.distance2500) == 120
        assert len(self.distance500) == 8
        assert len(self.knn3) == len(self.grocs) * 3

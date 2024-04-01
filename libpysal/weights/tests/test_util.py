"""Unit test for util.py"""

import geopandas as gpd
import numpy as np
import pytest

from ... import examples
from ...io.fileio import FileIO
from .. import util
from ..contiguity import Queen, Rook
from ..distance import KNN, DistanceBand
from ..util import fuzzy_contiguity, lat2W, nonplanar_neighbors
from ..weights import WSP, W


class Testutil:
    def setup_method(self):
        self.w = Rook.from_shapefile(examples.get_path("10740.shp"))

        self.rio = examples.load_example("Rio Grande do Sul")

    def test_lat2_w(self):
        w9 = lat2W(3, 3)
        assert w9.pct_nonzero == 29.62962962962963
        assert w9[0] == {1: 1.0, 3: 1.0}
        assert w9[3] == {0: 1.0, 4: 1.0, 6: 1.0}

    def test_lat2_sw(self):
        w9 = util.lat2SW(3, 3)
        rows, cols = w9.shape
        n = rows * cols
        assert w9.nnz == 24
        pct_nonzero = w9.nnz / float(n)
        assert pct_nonzero == 0.29629629629629628
        data = w9.todense().tolist()
        assert data[0] == [0, 1, 0, 1, 0, 0, 0, 0, 0]
        assert data[1] == [1, 0, 1, 0, 1, 0, 0, 0, 0]
        assert data[2] == [0, 1, 0, 0, 0, 1, 0, 0, 0]
        assert data[3] == [1, 0, 0, 0, 1, 0, 1, 0, 0]
        assert data[4] == [0, 1, 0, 1, 0, 1, 0, 1, 0]
        assert data[5] == [0, 0, 1, 0, 1, 0, 0, 0, 1]
        assert data[6] == [0, 0, 0, 1, 0, 0, 0, 1, 0]
        assert data[7] == [0, 0, 0, 0, 1, 0, 1, 0, 1]
        assert data[8] == [0, 0, 0, 0, 0, 1, 0, 1, 0]

    def test_block_weights(self):
        regimes = np.ones(25)
        regimes[list(range(10, 20))] = 2
        regimes[list(range(21, 25))] = 3
        regimes = np.array(
            [
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                1.0,
                3.0,
                3.0,
                3.0,
                3.0,
            ]
        )
        w = util.block_weights(regimes)
        ww0 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        assert w.weights[0] == ww0
        wn0 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 20]
        assert w.neighbors[0] == wn0
        regimes = ["n", "n", "s", "s", "e", "e", "w", "w", "e"]
        w = util.block_weights(regimes)
        wn = {
            0: [1],
            1: [0],
            2: [3],
            3: [2],
            4: [5, 8],
            5: [4, 8],
            6: [7],
            7: [6],
            8: [4, 5],
        }
        assert w.neighbors == wn
        ids = ["id-%i" % i for i in range(len(regimes))]
        w = util.block_weights(regimes, ids=np.array(ids))
        w0 = {"id-1": 1.0}
        assert w["id-0"] == w0
        w = util.block_weights(regimes, ids=ids)
        w0 = {"id-1": 1.0}
        assert w["id-0"] == w0

    def test_comb(self):
        x = list(range(4))
        l_ = []
        for i in util.comb(x, 2):
            l_.append(i)
        lo = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
        assert l_ == lo

    def test_order(self):
        w3 = util.order(self.w, kmax=3)
        w3105 = [1, -1, 1, 2, 1]
        assert w3105 == w3[1][0:5]

    def test_higher_order(self):
        w10 = lat2W(10, 10)
        w10_2 = util.higher_order(w10, 2)
        w10_20 = {2: 1.0, 11: 1.0, 20: 1.0}
        assert w10_20 == w10_2[0]
        w5 = lat2W()
        w50 = {1: 1.0, 5: 1.0}
        assert w50 == w5[0]
        w51 = {0: 1.0, 2: 1.0, 6: 1.0}
        assert w51 == w5[1]
        w5_2 = util.higher_order(w5, 2)
        w5_20 = {2: 1.0, 10: 1.0, 6: 1.0}
        assert w5_20 == w5_2[0]

    def test_higher_order_sp(self):
        w10 = lat2W(10, 10)
        w10_3 = util.higher_order_sp(w10, 3)
        w10_30 = {30: 1.0, 21: 1.0, 12: 1.0, 3: 1.0}
        assert w10_30 == w10_3[0]
        w10_3 = util.higher_order_sp(w10, 3, lower_order=True)
        w10_30 = {
            20: 1.0,
            30: 1.0,
            21: 1.0,
            10: 1.0,
            1: 1.0,
            11: 1.0,
            2: 1.0,
            12: 1.0,
            3: 1.0,
        }
        assert w10_30 == w10_3[0]

    def test_higher_order_classes(self):
        wdb = DistanceBand.from_shapefile(examples.get_path("baltim.shp"), 34)
        wknn = KNN.from_shapefile(examples.get_path("baltim.shp"), 10)
        wrook = Rook.from_shapefile(examples.get_path("columbus.shp"))
        wqueen = Queen.from_shapefile(examples.get_path("columbus.shp"))
        wsparse = wqueen.sparse
        ww = W(wknn.neighbors, wknn.weights)
        util.higher_order(wdb, 2)
        util.higher_order(wknn, 3)
        util.higher_order(wrook, 4)
        util.higher_order(wqueen, 5)
        util.higher_order(wsparse, 2)
        util.higher_order(ww, 2)
        ww.transform = "r"
        _ = wrook.sparse
        util.higher_order(wsparse, 2)
        with pytest.raises(ValueError):
            util.higher_order(ww, 3)

    def test_shimbel(self):
        w5 = lat2W()
        w5_shimbel = util.shimbel(w5)
        w5_shimbel024 = 8
        assert w5_shimbel024 == w5_shimbel[0][24]
        w5_shimbel004 = [-1, 1, 2, 3]
        assert w5_shimbel004 == w5_shimbel[0][0:4]

    def test_full(self):
        neighbors = {
            "first": ["second"],
            "second": ["first", "third"],
            "third": ["second"],
        }
        weights = {"first": [1], "second": [1, 1], "third": [1]}
        w = W(neighbors, weights)
        wf, ids = util.full(w)
        wfo = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        np.testing.assert_array_almost_equal(wfo, wf, decimal=8)
        idso = ["first", "second", "third"]
        assert idso == ids

    def test_full2_w(self):
        a = np.zeros((4, 4))
        for i in range(len(a)):
            for j in range(len(a[i])):
                if i != j:
                    a[i, j] = np.random.random(1)[0]
        w = util.full2W(a)
        np.testing.assert_array_equal(w.full()[0], a)
        ids = ["myID0", "myID1", "myID2", "myID3"]
        w = util.full2W(a, ids=ids)
        np.testing.assert_array_equal(w.full()[0], a)

    def test_wsp2_w(self):
        sp = util.lat2SW(2, 5)
        wsp = WSP(sp)
        w = util.WSP2W(wsp)
        assert w.n == 10
        assert w[0] == {1: 1, 5: 1}
        for weights in w.weights.values():
            assert isinstance(weights, list)
        w = FileIO(examples.get_path("sids2.gal"), "r").read()
        wsp = WSP(w.sparse, w.id_order)
        w = util.WSP2W(wsp)
        assert w.n == 100
        assert w["37135"] == {
            "37001": 1.0,
            "37033": 1.0,
            "37037": 1.0,
            "37063": 1.0,
            "37145": 1.0,
        }

    def test_fill_diagonal(self):
        w1 = util.fill_diagonal(self.w)
        r1 = {0: 1.0, 1: 1.0, 4: 1.0, 101: 1.0, 85: 1.0, 5: 1.0}
        assert w1[0] == r1
        w1 = util.fill_diagonal(self.w, 20)
        r1 = {0: 20, 1: 1.0, 4: 1.0, 101: 1.0, 85: 1.0, 5: 1.0}
        assert w1[0] == r1
        diag = np.arange(100, 100 + self.w.n)
        w1 = util.fill_diagonal(self.w, diag)
        r1 = {0: 100, 1: 1.0, 4: 1.0, 101: 1.0, 85: 1.0, 5: 1.0}
        assert w1[0] == r1

    def test_remap_ids(self):
        w = lat2W(3, 2)
        wid_order = [0, 1, 2, 3, 4, 5]
        assert wid_order == w.id_order
        wneighbors0 = [2, 1]
        assert wneighbors0 == w.neighbors[0]
        old_to_new = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e", 5: "f"}
        w_new = util.remap_ids(w, old_to_new)
        w_newid_order = ["a", "b", "c", "d", "e", "f"]
        assert w_newid_order == w_new.id_order
        w_newdneighborsa = ["c", "b"]
        assert w_newdneighborsa == w_new.neighbors["a"]

    def test_get_ids_shp(self):
        polyids = util.get_ids(examples.get_path("columbus.shp"), "POLYID")
        polyids5 = [1, 2, 3, 4, 5]
        assert polyids5 == polyids[:5]

    def test_get_ids_gdf(self):
        gdf = gpd.read_file(examples.get_path("columbus.shp"))
        polyids = util.get_ids(gdf, "POLYID")
        polyids5 = [1, 2, 3, 4, 5]
        assert polyids5 == polyids[:5]

    def test_get_points_array_from_shapefile(self):
        xy = util.get_points_array_from_shapefile(examples.get_path("juvenile.shp"))
        xy3 = np.array([[94.0, 93.0], [80.0, 95.0], [79.0, 90.0]])
        np.testing.assert_array_almost_equal(xy3, xy[:3], decimal=8)
        xy = util.get_points_array_from_shapefile(examples.get_path("columbus.shp"))
        xy3 = np.array(
            [
                [8.82721847, 14.36907602],
                [8.33265837, 14.03162401],
                [9.01226541, 13.81971908],
            ]
        )
        np.testing.assert_array_almost_equal(xy3, xy[:3], decimal=8)

    def test_min_threshold_distance(self):
        x, y = np.indices((5, 5))
        x.shape = (25, 1)
        y.shape = (25, 1)
        data = np.hstack([x, y])
        mint = 1.0
        assert mint == util.min_threshold_distance(data)

    def test_attach_islands(self):
        w = Rook.from_shapefile(examples.get_path("10740.shp"))
        w_knn1 = KNN.from_shapefile(examples.get_path("10740.shp"), k=1)
        w_attach = util.attach_islands(w, w_knn1)
        assert w_attach.islands == []
        assert w_attach[w.islands[0]] == {166: 1.0}

    def test_nonplanar_neighbors(self):
        df = gpd.read_file(examples.get_path("map_RS_BR.shp"))
        w = Queen.from_dataframe(df)
        assert w.islands == [
            0,
            4,
            23,
            27,
            80,
            94,
            101,
            107,
            109,
            119,
            122,
            139,
            169,
            175,
            223,
            239,
            247,
            253,
            254,
            255,
            256,
            261,
            276,
            291,
            294,
            303,
            321,
            357,
            374,
        ]
        wnp = nonplanar_neighbors(w, df)
        assert wnp.islands == []
        assert w.neighbors[0] == []
        assert wnp.neighbors[0] == [23, 59, 152, 239]
        assert wnp.neighbors[23] == [0, 45, 59, 107, 152, 185, 246]

    def test_fuzzy_contiguity(self):
        rs = examples.get_path("map_RS_BR.shp")
        rs_df = gpd.read_file(rs)
        wf = fuzzy_contiguity(rs_df)
        assert wf.islands == []
        assert set(wf.neighbors[0]) == {239, 59, 152, 23}
        buff = fuzzy_contiguity(rs_df, buffering=True, buffer=0.2)
        assert set(buff.neighbors[0]) == {175, 119, 239, 59, 152, 246, 23, 107}
        rs_index = rs_df.set_index("NM_MUNICIP")
        index_w = fuzzy_contiguity(rs_index)
        assert set(index_w.neighbors["TAVARES"]) == {"SÃO JOSÉ DO NORTE", "MOSTARDAS"}
        wf_pred = fuzzy_contiguity(rs_df, predicate="touches")
        assert set(wf_pred.neighbors[0]) == set()
        assert set(wf_pred.neighbors[1]) == {142, 82, 197, 285, 386, 350}

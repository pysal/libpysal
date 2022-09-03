import os
import tempfile
import warnings

import unittest
from ..weights import W, WSP
from .. import util
from ..util import WSP2W, lat2W
from ..contiguity import Rook
from ...io.fileio import FileIO as psopen
from ... import examples
from ..distance import KNN
import numpy as np

NPTA3E = np.testing.assert_array_almost_equal

try:
    import geopandas

    GEOPANDAS_EXTINCT = False
except ImportError:
    GEOPANDAS_EXTINCT = True

try:
    import matplotlib

    MPL_EXTINCT = False
except ImportError:
    MPL_EXTINCT = True


class TestW(unittest.TestCase):
    def setUp(self):
        self.w = Rook.from_shapefile(
            examples.get_path("10740.shp"), silence_warnings=True
        )

        self.neighbors = {
            0: [3, 1],
            1: [0, 4, 2],
            2: [1, 5],
            3: [0, 6, 4],
            4: [1, 3, 7, 5],
            5: [2, 4, 8],
            6: [3, 7],
            7: [4, 6, 8],
            8: [5, 7],
        }
        self.weights = {
            0: [1, 1],
            1: [1, 1, 1],
            2: [1, 1],
            3: [1, 1, 1],
            4: [1, 1, 1, 1],
            5: [1, 1, 1],
            6: [1, 1],
            7: [1, 1, 1],
            8: [1, 1],
        }

        self.w3x3 = util.lat2W(3, 3)

    def test_W(self):
        w = W(self.neighbors, self.weights, silence_warnings=True)
        self.assertEqual(w.pct_nonzero, 29.62962962962963)

    def test___getitem__(self):
        self.assertEqual(self.w[0], {1: 1.0, 4: 1.0, 101: 1.0, 85: 1.0, 5: 1.0})

    def test___init__(self):
        w = W(self.neighbors, self.weights, silence_warnings=True)
        self.assertEqual(w.pct_nonzero, 29.62962962962963)

    def test___iter__(self):
        w = lat2W(3, 3)
        res = {}
        for i, wi in enumerate(w):
            res[i] = wi
        self.assertEqual(res[0], (0, {1: 1.0, 3: 1.0}))
        self.assertEqual(res[8], (8, {5: 1.0, 7: 1.0}))

    def test_asymmetries(self):
        w = lat2W(3, 3)
        w.transform = "r"
        result = w.asymmetry()
        self.assertEqual(
            result,
            [
                (0, 1),
                (0, 3),
                (1, 0),
                (1, 2),
                (1, 4),
                (2, 1),
                (2, 5),
                (3, 0),
                (3, 4),
                (3, 6),
                (4, 1),
                (4, 3),
                (4, 5),
                (4, 7),
                (5, 2),
                (5, 4),
                (5, 8),
                (6, 3),
                (6, 7),
                (7, 4),
                (7, 6),
                (7, 8),
                (8, 5),
                (8, 7),
            ],
        )

    def test_asymmetry(self):
        w = lat2W(3, 3)
        self.assertEqual(w.asymmetry(), [])
        w.transform = "r"
        self.assertFalse(w.asymmetry() == [])

    def test_cardinalities(self):
        w = lat2W(3, 3)
        self.assertEqual(
            w.cardinalities, {0: 2, 1: 3, 2: 2, 3: 3, 4: 4, 5: 3, 6: 2, 7: 3, 8: 2}
        )

    def test_diagW2(self):
        NPTA3E(
            self.w3x3.diagW2, np.array([2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 2.0, 3.0, 2.0])
        )

    def test_diagWtW(self):
        NPTA3E(
            self.w3x3.diagW2, np.array([2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 2.0, 3.0, 2.0])
        )

    def test_diagWtW_WW(self):
        NPTA3E(
            self.w3x3.diagWtW_WW,
            np.array([4.0, 6.0, 4.0, 6.0, 8.0, 6.0, 4.0, 6.0, 4.0]),
        )

    def test_full(self):
        wf = np.array(
            [
                [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            ]
        )
        ids = list(range(9))

        wf1, ids1 = self.w3x3.full()
        NPTA3E(wf1, wf)
        self.assertEqual(ids1, ids)

    def test_get_transform(self):
        self.assertEqual(self.w3x3.transform, "O")
        self.w3x3.transform = "r"
        self.assertEqual(self.w3x3.transform, "R")
        self.w3x3.transform = "b"

    def test_higher_order(self):
        weights = {
            0: [1.0, 1.0, 1.0],
            1: [1.0, 1.0, 1.0],
            2: [1.0, 1.0, 1.0],
            3: [1.0, 1.0, 1.0],
            4: [1.0, 1.0, 1.0, 1.0],
            5: [1.0, 1.0, 1.0],
            6: [1.0, 1.0, 1.0],
            7: [1.0, 1.0, 1.0],
            8: [1.0, 1.0, 1.0],
        }
        neighbors = {
            0: [4, 6, 2],
            1: [3, 5, 7],
            2: [8, 0, 4],
            3: [7, 1, 5],
            4: [8, 0, 2, 6],
            5: [1, 3, 7],
            6: [4, 0, 8],
            7: [3, 1, 5],
            8: [6, 2, 4],
        }
        wneighbs = {
            k: {neighb: weights[k][i] for i, neighb in enumerate(v)}
            for k, v in list(neighbors.items())
        }
        w2 = util.higher_order(self.w3x3, 2)
        test_wneighbs = {
            k: {ne: weights[k][i] for i, ne in enumerate(v)}
            for k, v in list(w2.neighbors.items())
        }
        self.assertEqual(test_wneighbs, wneighbs)

    def test_histogram(self):
        hist = [
            (0, 1),
            (1, 1),
            (2, 4),
            (3, 20),
            (4, 57),
            (5, 44),
            (6, 36),
            (7, 15),
            (8, 7),
            (9, 1),
            (10, 6),
            (11, 0),
            (12, 2),
            (13, 0),
            (14, 0),
            (15, 1),
        ]
        self.assertEqual(self.w.histogram, hist)

    def test_id2i(self):
        id2i = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8}
        self.assertEqual(self.w3x3.id2i, id2i)

    def test_id_order_set(self):
        w = W(neighbors={"a": ["b"], "b": ["a", "c"], "c": ["b"]})
        self.assertFalse(w.id_order_set)

    def test_islands(self):
        w = W(
            neighbors={"a": ["b"], "b": ["a", "c"], "c": ["b"], "d": []},
            silence_warnings=True,
        )
        self.assertEqual(w.islands, ["d"])
        self.assertEqual(self.w3x3.islands, [])

    def test_max_neighbors(self):
        w = W(
            neighbors={"a": ["b"], "b": ["a", "c"], "c": ["b"], "d": []},
            silence_warnings=True,
        )
        self.assertEqual(w.max_neighbors, 2)
        self.assertEqual(self.w3x3.max_neighbors, 4)

    def test_mean_neighbors(self):
        w = util.lat2W()
        self.assertEqual(w.mean_neighbors, 3.2)

    def test_min_neighbors(self):
        w = util.lat2W()
        self.assertEqual(w.min_neighbors, 2)

    def test_n(self):
        w = util.lat2W()
        self.assertEqual(w.n, 25)

    def test_neighbor_offsets(self):
        d = {
            0: [3, 1],
            1: [0, 4, 2],
            2: [1, 5],
            3: [0, 6, 4],
            4: [1, 3, 7, 5],
            5: [2, 4, 8],
            6: [3, 7],
            7: [4, 6, 8],
            8: [5, 7],
        }

        self.assertEqual(self.w3x3.neighbor_offsets, d)

    def test_nonzero(self):
        self.assertEqual(self.w3x3.nonzero, 24)

    def test_order(self):
        w = util.lat2W(3, 3)
        o = {
            0: [-1, 1, 2, 1, 2, 3, 2, 3, 0],
            1: [1, -1, 1, 2, 1, 2, 3, 2, 3],
            2: [2, 1, -1, 3, 2, 1, 0, 3, 2],
            3: [1, 2, 3, -1, 1, 2, 1, 2, 3],
            4: [2, 1, 2, 1, -1, 1, 2, 1, 2],
            5: [3, 2, 1, 2, 1, -1, 3, 2, 1],
            6: [2, 3, 0, 1, 2, 3, -1, 1, 2],
            7: [3, 2, 3, 2, 1, 2, 1, -1, 1],
            8: [0, 3, 2, 3, 2, 1, 2, 1, -1],
        }
        self.assertEqual(util.order(w), o)

    def test_pct_nonzero(self):
        self.assertEqual(self.w3x3.pct_nonzero, 29.62962962962963)

    def test_s0(self):
        self.assertEqual(self.w3x3.s0, 24.0)

    def test_s1(self):
        self.assertEqual(self.w3x3.s1, 48.0)

    def test_s2(self):
        self.assertEqual(self.w3x3.s2, 272.0)

    def test_s2array(self):
        s2a = np.array(
            [[16.0], [36.0], [16.0], [36.0], [64.0], [36.0], [16.0], [36.0], [16.0]]
        )
        NPTA3E(self.w3x3.s2array, s2a)

    def test_sd(self):
        self.assertEqual(self.w3x3.sd, 0.66666666666666663)

    def test_set_transform(self):
        w = util.lat2W(2, 2)
        self.assertEqual(w.transform, "O")
        self.assertEqual(w.weights[0], [1.0, 1.0])
        w.transform = "r"
        self.assertEqual(w.weights[0], [0.5, 0.5])

    def test_shimbel(self):
        d = {
            0: [-1, 1, 2, 1, 2, 3, 2, 3, 4],
            1: [1, -1, 1, 2, 1, 2, 3, 2, 3],
            2: [2, 1, -1, 3, 2, 1, 4, 3, 2],
            3: [1, 2, 3, -1, 1, 2, 1, 2, 3],
            4: [2, 1, 2, 1, -1, 1, 2, 1, 2],
            5: [3, 2, 1, 2, 1, -1, 3, 2, 1],
            6: [2, 3, 4, 1, 2, 3, -1, 1, 2],
            7: [3, 2, 3, 2, 1, 2, 1, -1, 1],
            8: [4, 3, 2, 3, 2, 1, 2, 1, -1],
        }
        self.assertEqual(util.shimbel(self.w3x3), d)

    def test_sparse(self):
        self.assertEqual(self.w3x3.sparse.nnz, 24)

    def test_trcW2(self):
        self.assertEqual(self.w3x3.trcW2, 24.0)

    def test_trcWtW(self):
        self.assertEqual(self.w3x3.trcWtW, 24.0)

    def test_trcWtW_WW(self):
        self.assertEqual(self.w3x3.trcWtW_WW, 48.0)

    def test_symmetrize(self):
        symm = self.w.symmetrize()
        np.testing.assert_allclose(symm.sparse.toarray(), self.w.sparse.toarray())
        knn = KNN.from_shapefile(
            examples.get_path("baltim.shp"), k=10, silence_warnings=True
        )
        sknn = knn.symmetrize()
        assert not np.allclose(knn.sparse.toarray(), sknn.sparse.toarray())
        np.testing.assert_allclose(sknn.sparse.toarray(), sknn.sparse.toarray().T)
        knn.symmetrize(inplace=True)
        np.testing.assert_allclose(sknn.sparse.toarray(), knn.sparse.toarray())
        np.testing.assert_allclose(knn.sparse.toarray().T, knn.sparse.toarray())

    def test_connected_components(self):
        disco = {0: [1], 1: [0], 2: [3], 3: [2]}
        disco = W(disco)
        assert disco.n_components == 2

    def test_roundtrip_write(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(str(tmpdir), "tmp.gal")
            self.w.to_file(path)
            new = W.from_file(path)
        np.testing.assert_array_equal(self.w.sparse.toarray(), new.sparse.toarray())

    @unittest.skipIf(GEOPANDAS_EXTINCT or MPL_EXTINCT, "Missing dependencies")
    def test_plot(self):
        df = geopandas.read_file(examples.get_path("10740.shp"))
        with warnings.catch_warnings(record=True) as record:
            self.w.plot(df)
        assert len(record) == 0


class Test_WSP_Back_To_W(unittest.TestCase):
    # Test to make sure we get back to the same W functionality
    def setUp(self):
        self.w = Rook.from_shapefile(
            examples.get_path("10740.shp"), silence_warnings=True
        )
        wsp = self.w.to_WSP()
        self.w = wsp.to_W(silence_warnings=True)

        self.neighbors = {
            0: [3, 1],
            1: [0, 4, 2],
            2: [1, 5],
            3: [0, 6, 4],
            4: [1, 3, 7, 5],
            5: [2, 4, 8],
            6: [3, 7],
            7: [4, 6, 8],
            8: [5, 7],
        }
        self.weights = {
            0: [1, 1],
            1: [1, 1, 1],
            2: [1, 1],
            3: [1, 1, 1],
            4: [1, 1, 1, 1],
            5: [1, 1, 1],
            6: [1, 1],
            7: [1, 1, 1],
            8: [1, 1],
        }

        self.w3x3 = util.lat2W(3, 3)
        w3x3 = WSP(self.w3x3.sparse, self.w3x3.id_order)
        self.w3x3 = WSP2W(w3x3)

    def test_W(self):
        w = W(self.neighbors, self.weights, silence_warnings=True)
        self.assertEqual(w.pct_nonzero, 29.62962962962963)

    def test___getitem__(self):
        self.assertEqual(self.w[0], {1: 1.0, 4: 1.0, 101: 1.0, 85: 1.0, 5: 1.0})

    def test___init__(self):
        w = W(self.neighbors, self.weights, silence_warnings=True)
        self.assertEqual(w.pct_nonzero, 29.62962962962963)

    def test___iter__(self):
        w = util.lat2W(3, 3)
        res = {}
        for i, wi in enumerate(w):
            res[i] = wi
        self.assertEqual(res[0], (0, {1: 1.0, 3: 1.0}))
        self.assertEqual(res[8], (8, {5: 1.0, 7: 1.0}))

    def test_asymmetries(self):
        w = util.lat2W(3, 3)
        w.transform = "r"
        result = w.asymmetry()
        self.assertEqual(
            result,
            [
                (0, 1),
                (0, 3),
                (1, 0),
                (1, 2),
                (1, 4),
                (2, 1),
                (2, 5),
                (3, 0),
                (3, 4),
                (3, 6),
                (4, 1),
                (4, 3),
                (4, 5),
                (4, 7),
                (5, 2),
                (5, 4),
                (5, 8),
                (6, 3),
                (6, 7),
                (7, 4),
                (7, 6),
                (7, 8),
                (8, 5),
                (8, 7),
            ],
        )

    def test_asymmetry(self):
        w = util.lat2W(3, 3)
        self.assertEqual(w.asymmetry(), [])
        w.transform = "r"
        self.assertFalse(w.asymmetry() == [])

    def test_cardinalities(self):
        w = util.lat2W(3, 3)
        self.assertEqual(
            w.cardinalities, {0: 2, 1: 3, 2: 2, 3: 3, 4: 4, 5: 3, 6: 2, 7: 3, 8: 2}
        )

    def test_diagW2(self):
        NPTA3E(
            self.w3x3.diagW2, np.array([2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 2.0, 3.0, 2.0])
        )

    def test_diagWtW(self):
        NPTA3E(
            self.w3x3.diagW2, np.array([2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 2.0, 3.0, 2.0])
        )

    def test_diagWtW_WW(self):
        NPTA3E(
            self.w3x3.diagWtW_WW,
            np.array([4.0, 6.0, 4.0, 6.0, 8.0, 6.0, 4.0, 6.0, 4.0]),
        )

    def test_full(self):
        wf = np.array(
            [
                [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            ]
        )
        ids = list(range(9))

        wf1, ids1 = self.w3x3.full()
        NPTA3E(wf1, wf)
        self.assertEqual(ids1, ids)

    def test_get_transform(self):
        self.assertEqual(self.w3x3.transform, "O")
        self.w3x3.transform = "r"
        self.assertEqual(self.w3x3.transform, "R")
        self.w3x3.transform = "b"

    def test_higher_order(self):
        weights = {
            0: [1.0, 1.0, 1.0],
            1: [1.0, 1.0, 1.0],
            2: [1.0, 1.0, 1.0],
            3: [1.0, 1.0, 1.0],
            4: [1.0, 1.0, 1.0, 1.0],
            5: [1.0, 1.0, 1.0],
            6: [1.0, 1.0, 1.0],
            7: [1.0, 1.0, 1.0],
            8: [1.0, 1.0, 1.0],
        }
        neighbors = {
            0: [4, 6, 2],
            1: [3, 5, 7],
            2: [8, 0, 4],
            3: [7, 1, 5],
            4: [8, 0, 2, 6],
            5: [1, 3, 7],
            6: [4, 0, 8],
            7: [3, 1, 5],
            8: [6, 2, 4],
        }
        wneighbs = {
            k: {neighb: weights[k][i] for i, neighb in enumerate(v)}
            for k, v in list(neighbors.items())
        }
        w2 = util.higher_order(self.w3x3, 2)
        test_wneighbs = {
            k: {ne: w2.weights[k][i] for i, ne in enumerate(v)}
            for k, v in list(w2.neighbors.items())
        }
        self.assertEqual(test_wneighbs, wneighbs)

    def test_histogram(self):
        hist = [
            (0, 1),
            (1, 1),
            (2, 4),
            (3, 20),
            (4, 57),
            (5, 44),
            (6, 36),
            (7, 15),
            (8, 7),
            (9, 1),
            (10, 6),
            (11, 0),
            (12, 2),
            (13, 0),
            (14, 0),
            (15, 1),
        ]
        self.assertEqual(self.w.histogram, hist)

    def test_id2i(self):
        id2i = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8}
        self.assertEqual(self.w3x3.id2i, id2i)

    def test_id_order_set(self):
        w = W(neighbors={"a": ["b"], "b": ["a", "c"], "c": ["b"]})
        self.assertFalse(w.id_order_set)

    def test_islands(self):
        w = W(
            neighbors={"a": ["b"], "b": ["a", "c"], "c": ["b"], "d": []},
            silence_warnings=True,
        )
        self.assertEqual(w.islands, ["d"])
        self.assertEqual(self.w3x3.islands, [])

    def test_max_neighbors(self):
        w = W(
            neighbors={"a": ["b"], "b": ["a", "c"], "c": ["b"], "d": []},
            silence_warnings=True,
        )
        self.assertEqual(w.max_neighbors, 2)
        self.assertEqual(self.w3x3.max_neighbors, 4)

    def test_mean_neighbors(self):
        w = util.lat2W()
        self.assertEqual(w.mean_neighbors, 3.2)

    def test_min_neighbors(self):
        w = util.lat2W()
        self.assertEqual(w.min_neighbors, 2)

    def test_n(self):
        w = util.lat2W()
        self.assertEqual(w.n, 25)

    def test_nonzero(self):
        self.assertEqual(self.w3x3.nonzero, 24)

    def test_order(self):
        w = util.lat2W(3, 3)
        o = {
            0: [-1, 1, 2, 1, 2, 3, 2, 3, 0],
            1: [1, -1, 1, 2, 1, 2, 3, 2, 3],
            2: [2, 1, -1, 3, 2, 1, 0, 3, 2],
            3: [1, 2, 3, -1, 1, 2, 1, 2, 3],
            4: [2, 1, 2, 1, -1, 1, 2, 1, 2],
            5: [3, 2, 1, 2, 1, -1, 3, 2, 1],
            6: [2, 3, 0, 1, 2, 3, -1, 1, 2],
            7: [3, 2, 3, 2, 1, 2, 1, -1, 1],
            8: [0, 3, 2, 3, 2, 1, 2, 1, -1],
        }
        self.assertEqual(util.order(w), o)

    def test_pct_nonzero(self):
        self.assertEqual(self.w3x3.pct_nonzero, 29.62962962962963)

    def test_s0(self):
        self.assertEqual(self.w3x3.s0, 24.0)

    def test_s1(self):
        self.assertEqual(self.w3x3.s1, 48.0)

    def test_s2(self):
        self.assertEqual(self.w3x3.s2, 272.0)

    def test_s2array(self):
        s2a = np.array(
            [[16.0], [36.0], [16.0], [36.0], [64.0], [36.0], [16.0], [36.0], [16.0]]
        )
        NPTA3E(self.w3x3.s2array, s2a)

    def test_sd(self):
        self.assertEqual(self.w3x3.sd, 0.66666666666666663)

    def test_set_transform(self):
        w = util.lat2W(2, 2)
        self.assertEqual(w.transform, "O")
        self.assertEqual(w.weights[0], [1.0, 1.0])
        w.transform = "r"
        self.assertEqual(w.weights[0], [0.5, 0.5])

    def test_shimbel(self):
        d = {
            0: [-1, 1, 2, 1, 2, 3, 2, 3, 4],
            1: [1, -1, 1, 2, 1, 2, 3, 2, 3],
            2: [2, 1, -1, 3, 2, 1, 4, 3, 2],
            3: [1, 2, 3, -1, 1, 2, 1, 2, 3],
            4: [2, 1, 2, 1, -1, 1, 2, 1, 2],
            5: [3, 2, 1, 2, 1, -1, 3, 2, 1],
            6: [2, 3, 4, 1, 2, 3, -1, 1, 2],
            7: [3, 2, 3, 2, 1, 2, 1, -1, 1],
            8: [4, 3, 2, 3, 2, 1, 2, 1, -1],
        }
        self.assertEqual(util.shimbel(self.w3x3), d)

    def test_sparse(self):
        self.assertEqual(self.w3x3.sparse.nnz, 24)

    def test_trcW2(self):
        self.assertEqual(self.w3x3.trcW2, 24.0)

    def test_trcWtW(self):
        self.assertEqual(self.w3x3.trcWtW, 24.0)

    def test_trcWtW_WW(self):
        self.assertEqual(self.w3x3.trcWtW_WW, 48.0)


class TestWSP(unittest.TestCase):
    def setUp(self):
        self.w = psopen(examples.get_path("sids2.gal")).read()
        self.wsp = WSP(self.w.sparse, self.w.id_order)
        w3x3 = util.lat2W(3, 3)
        self.w3x3 = WSP(w3x3.sparse)

    def test_WSP(self):
        self.assertEqual(self.w.id_order, self.wsp.id_order)
        self.assertEqual(self.w.n, self.wsp.n)
        np.testing.assert_array_equal(
            self.w.sparse.todense(), self.wsp.sparse.todense()
        )

    def test_diagWtW_WW(self):
        NPTA3E(
            self.w3x3.diagWtW_WW,
            np.array([4.0, 6.0, 4.0, 6.0, 8.0, 6.0, 4.0, 6.0, 4.0]),
        )

    def test_trcWtW_WW(self):
        self.assertEqual(self.w3x3.trcWtW_WW, 48.0)

    def test_s0(self):
        self.assertEqual(self.w3x3.s0, 24.0)


if __name__ == "__main__":
    unittest.main()

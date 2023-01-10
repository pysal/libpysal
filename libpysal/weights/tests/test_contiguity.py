from .. import contiguity as c
from ..weights import W
from .. import util
from .. import raster
from ...common import pandas
from ...io.fileio import FileIO as ps_open
from ...io import geotable as pdio

from ... import examples as pysal_examples
import unittest as ut
import numpy as np
import pytest

PANDAS_EXTINCT = pandas is None
try:
    import geopandas

    GEOPANDAS_EXTINCT = False
except ImportError:
    GEOPANDAS_EXTINCT = True

try:
    import shapely

    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False


class Contiguity_Mixin(object):
    polygon_path = pysal_examples.get_path("columbus.shp")
    point_path = pysal_examples.get_path("baltim.shp")
    da = raster.testDataArray((1, 4, 4), missing_vals=False)
    f = ps_open(polygon_path)  # our file handler
    polygons = f.read()  # our iterable
    f.seek(0)  # go back to head of file
    cls = object  # class constructor
    known_wi = None  # index of known w entry to compare
    known_w = dict()  # actual w entry
    known_name = known_wi
    known_namedw = known_w
    idVariable = None  # id variable from file or column
    known_wspi_da = None
    known_wsp_da = dict()
    known_wi_da = None
    known_w_da = dict()

    def setUp(self):
        self.__dict__.update(
            {
                k: v
                for k, v in list(Contiguity_Mixin.__dict__.items())
                if not k.startswith("_")
            }
        )

    def runTest(self):
        pass

    def test_init(self):
        # basic
        w = self.cls(self.polygons)
        self.assertEqual(w[self.known_wi], self.known_w)

        # sparse
        # w = self.cls(self.polygons, sparse=True)
        # srowvec = ws.sparse[self.known_wi].todense().tolist()[0]
        # this_w = {i:k for i,k in enumerate(srowvec) if k>0}
        # self.assertEqual(this_w, self.known_w)
        # ids = ps.weights2.utils.get_ids(self.polygon_path, self.idVariable)

        # named
        ids = util.get_ids(self.polygon_path, self.idVariable)
        w = self.cls(self.polygons, ids=ids)
        self.assertEqual(w[self.known_name], self.known_namedw)

    def test_from_iterable(self):
        w = self.cls.from_iterable(self.f)
        self.f.seek(0)
        self.assertEqual(w[self.known_wi], self.known_w)

    def test_from_shapefile(self):
        # basic
        w = self.cls.from_shapefile(self.polygon_path)
        self.assertEqual(w[self.known_wi], self.known_w)

        # sparse
        ws = self.cls.from_shapefile(self.polygon_path, sparse=True)
        srowvec = ws.sparse[self.known_wi].todense().tolist()[0]
        this_w = {i: k for i, k in enumerate(srowvec) if k > 0}
        self.assertEqual(this_w, self.known_w)

        # named
        w = self.cls.from_shapefile(self.polygon_path, idVariable=self.idVariable)
        self.assertEqual(w[self.known_name], self.known_namedw)

    def test_from_array(self):
        # test named, sparse from point array
        pass

    @ut.skipIf(PANDAS_EXTINCT, "Missing pandas")
    def test_from_dataframe(self):
        # basic
        df = pdio.read_files(self.polygon_path)
        w = self.cls.from_dataframe(df)
        self.assertEqual(w[self.known_wi], self.known_w)

        # named geometry
        df.rename(columns={"geometry": "the_geom"}, inplace=True)
        w = self.cls.from_dataframe(df, geom_col="the_geom")
        self.assertEqual(w[self.known_wi], self.known_w)

    @ut.skipIf(GEOPANDAS_EXTINCT, "Missing geopandas")
    def test_from_geodataframe(self):
        df = pdio.read_files(self.polygon_path)
        # named active geometry
        df.rename(columns={"geometry": "the_geom"}, inplace=True)
        df = df.set_geometry("the_geom")
        w = self.cls.from_dataframe(df)
        self.assertEqual(w[self.known_wi], self.known_w)

        # named geometry + named obs
        w = self.cls.from_dataframe(df, geom_col="the_geom", ids=self.idVariable)
        self.assertEqual(w[self.known_name], self.known_namedw)

    @ut.skipIf(GEOPANDAS_EXTINCT, "Missing geopandas")
    def test_from_geodataframe_order(self):
        import geopandas

        south = geopandas.read_file(pysal_examples.get_path("south.shp"))
        expected = south.FIPS.iloc[:5].tolist()
        for ids_ in ("FIPS", south.FIPS):
            w = self.cls.from_dataframe(south, ids=ids_)
            self.assertEqual(w.id_order[:5], expected)

    def test_from_xarray(self):
        w = self.cls.from_xarray(self.da, sparse=False, n_jobs=-1)
        self.assertEqual(w[self.known_wi_da], self.known_w_da)
        ws = self.cls.from_xarray(self.da)
        srowvec = ws.sparse[self.known_wspi_da].todense().tolist()[0]
        this_w = {i: k for i, k in enumerate(srowvec) if k > 0}
        self.assertEqual(this_w, self.known_wsp_da)


class Test_Queen(ut.TestCase, Contiguity_Mixin):
    def setUp(self):
        Contiguity_Mixin.setUp(self)

        self.known_wi = 4
        self.known_w = {
            2: 1.0,
            3: 1.0,
            5: 1.0,
            7: 1.0,
            8: 1.0,
            10: 1.0,
            14: 1.0,
            15: 1.0,
        }
        self.cls = c.Queen
        self.idVariable = "POLYID"
        self.known_name = 5
        self.known_namedw = {k + 1: v for k, v in list(self.known_w.items())}
        self.known_wspi_da = 1
        self.known_wsp_da = {0: 1, 2: 1, 4: 1, 5: 1, 6: 1}
        self.known_wi_da = (1, -30.0, -60.0)
        self.known_w_da = {
            (1, -90.0, -180.0): 1,
            (1, -90.0, -60.0): 1,
            (1, -90.0, 60.0): 1,
            (1, -30.0, -180.0): 1,
            (1, -30.0, 60.0): 1,
            (1, 30.0, -180.0): 1,
            (1, 30.0, -60.0): 1,
            (1, 30.0, 60.0): 1,
        }

    @ut.skipIf(GEOPANDAS_EXTINCT, "Missing Geopandas")
    def test_linestrings(self):
        import geopandas

        eberly = geopandas.read_file(pysal_examples.get_path("eberly_net.shp")).iloc[
            0:8
        ]
        eberly_w = {
            0: [1, 2, 3],
            1: [0, 4],
            2: [0, 3, 4, 5],
            3: [0, 2, 7],
            4: [1, 2, 5],
            5: [2, 4, 6],
            6: [5],
            7: [3],
        }
        eberly_w = W(neighbors=eberly_w).sparse.toarray()
        computed = self.cls.from_dataframe(eberly).sparse.toarray()
        np.testing.assert_array_equal(eberly_w, computed)


class Test_Rook(ut.TestCase, Contiguity_Mixin):
    def setUp(self):
        Contiguity_Mixin.setUp(self)

        self.known_w = {2: 1.0, 3: 1.0, 5: 1.0, 7: 1.0, 8: 1.0, 10: 1.0, 14: 1.0}
        self.known_wi = 4
        self.cls = c.Rook
        self.idVariable = "POLYID"
        self.known_name = 5
        self.known_namedw = {k + 1: v for k, v in list(self.known_w.items())}
        self.known_wspi_da = 1
        self.known_wsp_da = {0: 1, 2: 1, 5: 1}
        self.known_wi_da = (1, -30.0, -180.0)
        self.known_w_da = {
            (1, 30.0, -180.0): 1,
            (1, -30.0, -60.0): 1,
            (1, -90.0, -180.0): 1,
        }


class Test_Voronoi(ut.TestCase):
    @pytest.mark.skipif(not HAS_SHAPELY, reason="shapely needed")
    def test_voronoiW(self):
        np.random.seed(12345)
        points = np.random.random((5, 2)) * 10 + 10
        w = c.Voronoi(points)
        self.assertEqual(w.n, 5)
        self.assertEqual(
            w.neighbors, {0: [2, 3, 4], 1: [2], 2: [0, 1, 4], 3: [0, 4], 4: [0, 2, 3]}
        )


q = ut.TestLoader().loadTestsFromTestCase(Test_Queen)
r = ut.TestLoader().loadTestsFromTestCase(Test_Rook)
suite = ut.TestSuite([q, r])
if __name__ == "__main__":
    runner = ut.TextTestRunner()
    runner.run(suite)

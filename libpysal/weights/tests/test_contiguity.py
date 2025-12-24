# ruff: noqa: N815

import numpy as np
import pytest

from ... import examples as pysal_examples
from ...io import geotable as pdio
from ...io.fileio import FileIO
from .. import contiguity as c
from .. import util
from ..weights import W


class ContiguityMixin:
    polygon_path = pysal_examples.get_path("columbus.shp")
    point_path = pysal_examples.get_path("baltim.shp")
    f = FileIO(polygon_path)  # our file handler
    polygons = f.read()  # our iterable
    f.seek(0)  # go back to head of file
    cls = object  # class constructor
    known_wi = None  # index of known w entry to compare
    known_w = {}  # actual w entry
    known_name = known_wi
    known_namedw = known_w
    idVariable = None  # id variable from file or column
    known_wspi_da = None
    known_wsp_da = {}
    known_wi_da = None
    known_w_da = {}
    try:
        from .. import raster

        da = raster.testDataArray((1, 4, 4), missing_vals=False)
    except ImportError:
        da = None

    def setup_method(self):
        self.__dict__.update(
            {
                k: v
                for k, v in list(ContiguityMixin.__dict__.items())
                if not k.startswith("_")
            }
        )

    def test_init(self):
        # basic
        w = self.cls(self.polygons)
        assert w[self.known_wi] == self.known_w

        # sparse
        # w = self.cls(self.polygons, sparse=True)
        # srowvec = ws.sparse[self.known_wi].todense().tolist()[0]
        # this_w = {i:k for i,k in enumerate(srowvec) if k>0}
        # self.assertEqual(this_w, self.known_w)
        # ids = ps.weights2.utils.get_ids(self.polygon_path, self.idVariable)

        # named
        ids = util.get_ids(self.polygon_path, self.idVariable)
        w = self.cls(self.polygons, ids=ids)
        assert w[self.known_name] == self.known_namedw

    def test_from_iterable(self):
        w = self.cls.from_iterable(self.f)
        self.f.seek(0)
        assert w[self.known_wi] == self.known_w

    def test_from_shapefile(self):
        # basic
        w = self.cls.from_shapefile(self.polygon_path)
        assert w[self.known_wi] == self.known_w

        # sparse
        ws = self.cls.from_shapefile(self.polygon_path, sparse=True)
        srowvec = ws.sparse[self.known_wi].todense().tolist()[0]
        this_w = {i: k for i, k in enumerate(srowvec) if k > 0}
        assert this_w == self.known_w

        # named
        w = self.cls.from_shapefile(self.polygon_path, idVariable=self.idVariable)
        assert w[self.known_name] == self.known_namedw

    def test_from_array(self):
        # test named, sparse from point array
        pass

    def test_from_dataframe(self):
        # basic
        df = pdio.read_files(self.polygon_path)
        w = self.cls.from_dataframe(df)
        assert w[self.known_wi] == self.known_w

        # named geometry
        df.rename(columns={"geometry": "the_geom"}, inplace=True)
        w = self.cls.from_dataframe(df, geom_col="the_geom")
        assert w[self.known_wi] == self.known_w

    def test_from_geodataframe(self):
        df = pdio.read_files(self.polygon_path)
        # named active geometry
        df.rename(columns={"geometry": "the_geom"}, inplace=True)
        df = df.set_geometry("the_geom")
        w = self.cls.from_dataframe(df)
        assert w[self.known_wi] == self.known_w

        # named geometry + named obs
        w = self.cls.from_dataframe(df, geom_col="the_geom", ids=self.idVariable)
        assert w[self.known_name] == self.known_namedw

    @pytest.mark.network
    def test_from_geodataframe_order(self):
        import geopandas

        south = geopandas.read_file(pysal_examples.get_path("south.shp"))
        expected = south.FIPS.iloc[:5].tolist()
        for ids_ in ("FIPS", south.FIPS):
            w = self.cls.from_dataframe(south, ids=ids_)
            assert w.id_order[:5] == expected

    def test_from_xarray(self):
        pytest.importorskip("xarray")

        w = self.cls.from_xarray(self.da, sparse=False, n_jobs=-1)
        assert w[self.known_wi_da] == self.known_w_da
        ws = self.cls.from_xarray(self.da)
        srowvec = ws.sparse[self.known_wspi_da].todense().tolist()[0]
        this_w = {i: k for i, k in enumerate(srowvec) if k > 0}
        assert this_w == self.known_wsp_da


class TestQueen(ContiguityMixin):
    def setup_method(self):
        ContiguityMixin.setup_method(self)

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


class TestRook(ContiguityMixin):
    def setup_method(self):
        ContiguityMixin.setup_method(self)

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


class TestVoronoi:
    def test_voronoi_w(self):
        np.random.seed(12345)
        points = np.random.random((5, 2)) * 10 + 10
        w = c.Voronoi(points)
        assert w.n == 5
        assert w.neighbors == {
            0: [2, 3, 4],
            1: [2],
            2: [0, 1, 4],
            3: [0, 4],
            4: [0, 2, 3],
        }

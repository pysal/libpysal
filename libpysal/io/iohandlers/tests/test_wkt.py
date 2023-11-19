import pytest

from .... import examples as pysal_examples
from ..wkt import WKTReader


class TesttestWKTReader:
    def setup_method(self):
        self.test_file = test_file = pysal_examples.get_path("stl_hom.wkt")
        self.obj = WKTReader(test_file, "r")

    def test_close(self):
        f = self.obj
        f.close()
        pytest.raises(ValueError, f.read)
        # w_kt_reader = WKTReader(*args, **kwargs)
        # self.assertEqual(expected, w_kt_reader.close())

    def test_open(self):
        f = self.obj
        expected = ["wkt"]
        assert expected == f.FORMATS

    def test__read(self):
        polys = self.obj.read()
        assert len(polys) == 78
        assert polys[1].centroid == (-91.195784694307383, 39.990883050220845)

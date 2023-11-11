"""GeoDa Text File Reader Unit Tests"""

import pytest

from .... import examples as pysal_examples
from ..geoda_txt import GeoDaTxtReader


class TesttestGeoDaTxtReader:
    def setup_method(self):
        test_file = pysal_examples.get_path("stl_hom.txt")
        self.obj = GeoDaTxtReader(test_file, "r")

    def test___init__(self):
        assert self.obj.header == ["FIPSNO", "HR8488", "HR8893", "HC8488"]

    def test___len__(self):
        expected = 78
        assert expected == len(self.obj)

    def test_close(self):
        f = self.obj
        f.close()
        pytest.raises(ValueError, f.read)

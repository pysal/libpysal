import os
import tempfile

import pytest

from .... import examples as pysal_examples
from ...fileio import FileIO as psopen
from ..arcgis_swm import ArcGISSwmIO


class Testtest_ArcGISSwmIO:
    def setup_method(self):
        self.test_file = test_file = pysal_examples.get_path("ohio.swm")
        self.obj = ArcGISSwmIO(test_file, "r")

    def test_close(self):
        f = self.obj
        f.close()
        pytest.raises(ValueError, f.read)

    def test_read(self):
        w = self.obj.read()
        assert w.n == 88
        assert w.mean_neighbors == 5.25
        assert [1.0, 1.0, 1.0, 1.0] == list(w[1].values())

    def test_seek(self):
        self.test_read()
        pytest.raises(StopIteration, self.obj.read)
        self.obj.seek(0)
        self.test_read()

    def test_write(self):
        w = self.obj.read()
        f = tempfile.NamedTemporaryFile(suffix=".swm")
        fname = f.name
        f.close()
        o = psopen(fname, "w")
        o.write(w)
        o.close()
        wnew = psopen(fname, "r").read()
        assert wnew.pct_nonzero == w.pct_nonzero
        os.remove(fname)

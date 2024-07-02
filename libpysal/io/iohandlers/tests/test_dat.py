import os
import tempfile

import pytest

from .... import examples as pysal_examples
from ...fileio import FileIO
from ..dat import DatIO


class TesttestDatIO:
    def setup_method(self):
        self.test_file = test_file = pysal_examples.get_path("wmat.dat")
        self.obj = DatIO(test_file, "r")

    def test_close(self):
        f = self.obj
        f.close()
        pytest.raises(ValueError, f.read)

    def test_read(self):
        w = self.obj.read()
        assert w.n == 49
        assert w.mean_neighbors == 4.7346938775510203
        assert list(w[5.0].values()) == [0.5, 0.5]

    def test_seek(self):
        self.test_read()
        pytest.raises(StopIteration, self.obj.read)
        self.obj.seek(0)
        self.test_read()

    def test_write(self):
        w = self.obj.read()
        f = tempfile.NamedTemporaryFile(suffix=".dat")
        fname = f.name
        f.close()
        o = FileIO(fname, "w")
        o.write(w)
        o.close()
        wnew = FileIO(fname, "r").read()
        assert wnew.pct_nonzero == w.pct_nonzero
        os.remove(fname)

import os
import tempfile
import warnings

import pytest

from .... import examples as pysal_examples
from ...fileio import FileIO
from ..mat import MatIO


class TesttestMatIO:
    def setup_method(self):
        self.test_file = test_file = pysal_examples.get_path("spat-sym-us.mat")
        self.obj = MatIO(test_file, "r")

    def test_close(self):
        f = self.obj
        f.close()
        pytest.raises(ValueError, f.read)

    def test_read(self):
        w = self.obj.read()
        assert w.n == 46
        assert w.mean_neighbors == 4.0869565217391308
        assert list(w[1].values()) == [1.0, 1.0, 1.0, 1.0]

    def test_seek(self):
        self.test_read()
        pytest.raises(StopIteration, self.obj.read)
        self.obj.seek(0)
        self.test_read()

    def test_write(self):
        w = self.obj.read()
        f = tempfile.NamedTemporaryFile(suffix=".mat")
        fname = f.name
        f.close()
        o = FileIO(fname, "w")
        with warnings.catch_warnings(record=True) as warn:
            warnings.simplefilter("always")
            o.write(w)
            if len(warn) > 0:
                assert issubclass(warn[0].category, FutureWarning)
        o.close()
        wnew = FileIO(fname, "r").read()
        assert wnew.pct_nonzero == w.pct_nonzero
        os.remove(fname)

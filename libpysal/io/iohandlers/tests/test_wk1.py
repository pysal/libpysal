import os
import tempfile

import pytest

from .... import examples as pysal_examples
from ...fileio import FileIO
from ..wk1 import Wk1IO


class TesttestWk1IO:
    def setup_method(self):
        self.test_file = test_file = pysal_examples.get_path("spat-sym-us.wk1")
        self.obj = Wk1IO(test_file, "r")

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
        f = tempfile.NamedTemporaryFile(suffix=".wk1")
        fname = f.name
        f.close()
        o = FileIO(fname, "w")
        o.write(w)
        o.close()
        wnew = FileIO(fname, "r").read()
        assert wnew.pct_nonzero == w.pct_nonzero
        os.remove(fname)

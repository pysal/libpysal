import os
import tempfile

import pytest

from .... import examples as pysal_examples
from ...fileio import FileIO
from ..mtx import MtxIO


class TesttestMtxIO:
    def setup_method(self):
        self.test_file = test_file = pysal_examples.get_path("wmat.mtx")
        self.obj = MtxIO(test_file, "r")

    def test_close(self):
        f = self.obj
        f.close()
        pytest.raises(ValueError, f.read)

    def test_read(self):
        w = self.obj.read()
        assert w.n == 49
        assert w.mean_neighbors == 4.7346938775510203
        assert list(w[1].values()) == [
            0.33329999999999999,
            0.33329999999999999,
            0.33329999999999999,
        ]
        s0 = w.s0
        self.obj.seek(0)
        wsp = self.obj.read(sparse=True)
        assert wsp.n == 49
        assert s0 == wsp.s0

    def test_seek(self):
        self.test_read()
        pytest.raises(StopIteration, self.obj.read)
        self.obj.seek(0)
        self.test_read()

    def test_write(self):
        for i in [False, True]:
            self.obj.seek(0)
            w = self.obj.read(sparse=i)
            f = tempfile.NamedTemporaryFile(suffix=".mtx")
            fname = f.name
            f.close()
            o = FileIO(fname, "w")
            o.write(w)
            o.close()
            wnew = FileIO(fname, "r").read(sparse=i)
            if i:
                assert wnew.s0 == w.s0
            else:
                assert wnew.pct_nonzero == w.pct_nonzero
            os.remove(fname)

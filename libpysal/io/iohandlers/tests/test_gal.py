"""Unit tests for gal.py"""

import tempfile

import pytest

from .... import examples as pysal_examples
from ...fileio import FileIO
from ..gal import GalIO


class TesttestGalIO:
    def setup_method(self):
        self.test_file = test_file = pysal_examples.get_path("sids2.gal")
        self.obj = GalIO(test_file, "r")

    def test___init__(self):
        assert self.obj._typ == str  # noqa: E721

    def test_close(self):
        f = self.obj
        f.close()
        pytest.raises(ValueError, f.read)

    def test_read(self):
        # reading a GAL returns a W
        w = self.obj.read()
        assert w.n == 100
        assert w.sd == pytest.approx(1.5151237573214935)
        assert w.s0 == 462.0
        assert w.s1 == 924.0

    def test_seek(self):
        self.test_read()
        pytest.raises(StopIteration, self.obj.read)
        self.obj.seek(0)
        self.test_read()

    def test_write(self):
        w = self.obj.read()
        f = tempfile.NamedTemporaryFile(suffix=".gal")
        fname = f.name
        f.close()
        o = FileIO(fname, "w")
        o.write(w)
        o.close()
        wnew = FileIO(fname, "r").read()
        assert wnew.pct_nonzero == w.pct_nonzero

import os
import tempfile

import pytest

from .... import examples as pysal_examples
from ...fileio import FileIO
from ..gwt import GwtIO


class TesttestGwtIO:
    def setup_method(self):
        self.test_file = test_file = pysal_examples.get_path("juvenile.gwt")
        self.obj = GwtIO(test_file, "r")

    def test_close(self):
        f = self.obj
        f.close()
        pytest.raises(ValueError, f.read)

    def test_read(self):
        w = self.obj.read()
        assert w.n == 168
        assert w.mean_neighbors == 16.678571428571427
        w.transform = "B"
        assert list(w[1].values()) == [1.0]

    def test_seek(self):
        self.test_read()
        pytest.raises(StopIteration, self.obj.read)
        self.obj.seek(0)
        self.test_read()

    # Commented out by CRS, GWT 'w' mode removed until we
    # can find a good solution for retaining distances.
    # see issue #153.
    # Added back by CRS,
    def test_write(self):
        with pytest.warns(RuntimeWarning, match="DBF relating to GWT was not found"):
            w = self.obj.read()
            f = tempfile.NamedTemporaryFile(suffix=".gwt")
            fname = f.name
            f.close()
            o = FileIO(fname, "w")
            # copy the shapefile and ID variable names from the old gwt.
            # this is only available after the read() method has been called.
            # o.shpName = self.obj.shpName
            # o.varName = self.obj.varName
            o.write(w)
            o.close()
            wnew = FileIO(fname, "r").read()
            assert wnew.pct_nonzero == w.pct_nonzero
            os.remove(fname)

import os
import tempfile
import warnings

import pytest

from .... import examples as pysal_examples
from ...fileio import FileIO
from ..arcgis_dbf import ArcGISDbfIO


class TesttestArcGISDbfIO:
    def setup_method(self):
        self.test_file = test_file = pysal_examples.get_path("arcgis_ohio.dbf")
        self.obj = ArcGISDbfIO(test_file, "r")

    def test_close(self):
        f = self.obj
        f.close()
        pytest.raises(ValueError, f.read)

    def test_read(self):
        with warnings.catch_warnings(record=True) as warn:
            warnings.simplefilter("always")
            w = self.obj.read()
            if len(warn) > 0:
                assert issubclass(warn[0].category, RuntimeWarning)
                assert (
                    "Missing Value Found, setting value to libpysal.MISSINGVALUE."
                    in str(warn[0].message)
                )
        assert w.n == 88
        assert w.mean_neighbors == 5.25
        assert list(w[1].values()) == [1.0, 1.0, 1.0, 1.0]

    def test_seek(self):
        self.test_read()
        pytest.raises(StopIteration, self.obj.read)
        self.obj.seek(0)
        self.test_read()

    def test_write(self):
        with warnings.catch_warnings(record=True) as warn:
            warnings.simplefilter("always")
            w = self.obj.read()
            if len(warn) > 0:
                assert issubclass(warn[0].category, RuntimeWarning)
                assert (
                    "Missing Value Found, setting value to libpysal.MISSINGVALUE."
                    in str(warn[0].message)
                )
        f = tempfile.NamedTemporaryFile(suffix=".dbf")
        fname = f.name
        f.close()
        o = FileIO(fname, "w", "arcgis_dbf")
        o.write(w)
        o.close()
        f = FileIO(fname, "r", "arcgis_dbf")
        wnew = f.read()
        f.close()
        assert wnew.pct_nonzero == w.pct_nonzero
        os.remove(fname)

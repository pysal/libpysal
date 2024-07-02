import os
import tempfile
import warnings

import pytest

from .... import examples as pysal_examples
from ...fileio import FileIO
from ..arcgis_txt import ArcGISTextIO


class TesttestArcGISTextIO:
    def setup_method(self):
        self.test_file = test_file = pysal_examples.get_path("arcgis_txt.txt")
        self.obj = ArcGISTextIO(test_file, "r")

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
                    "DBF relating to ArcGIS TEXT was not found, "
                    "proceeding with unordered string IDs."
                ) in str(warn[0].message)
        assert w.n == 3
        assert w.mean_neighbors == 2.0
        assert list(w[2].values()) == [0.1, 0.05]

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
                    "DBF relating to ArcGIS TEXT was not found, "
                    "proceeding with unordered string IDs."
                ) in str(warn[0].message)
        f = tempfile.NamedTemporaryFile(suffix=".txt")
        fname = f.name
        f.close()
        o = FileIO(fname, "w", "arcgis_text")
        o.write(w)
        o.close()
        with warnings.catch_warnings(record=True) as warn:
            warnings.simplefilter("always")
            wnew = FileIO(fname, "r", "arcgis_text").read()
            if len(warn) > 0:
                assert issubclass(warn[0].category, RuntimeWarning)
                assert (
                    "DBF relating to ArcGIS TEXT was not found, "
                    "proceeding with unordered string IDs."
                ) in str(warn[0].message)
        assert wnew.pct_nonzero == w.pct_nonzero
        os.remove(fname)

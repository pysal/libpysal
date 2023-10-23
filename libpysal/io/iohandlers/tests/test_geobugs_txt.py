import os
import tempfile

import pytest

from .... import examples as pysal_examples
from ...fileio import FileIO as psopen
from ..geobugs_txt import GeoBUGSTextIO


class Testtest_GeoBUGSTextIO:
    def setup_method(self):
        self.test_file_scot = test_file_scot = pysal_examples.get_path("geobugs_scot")
        self.test_file_col = test_file_col = pysal_examples.get_path(
            "spdep_listw2WB_columbus"
        )
        self.obj_scot = GeoBUGSTextIO(test_file_scot, "r")
        self.obj_col = GeoBUGSTextIO(test_file_col, "r")

    def test_close(self):
        for obj in [self.obj_scot, self.obj_col]:
            f = obj
            f.close()
            pytest.raises(ValueError, f.read)

    def test_read(self):
        w_scot = self.obj_scot.read()
        assert w_scot.n == 56
        assert w_scot.mean_neighbors == 4.1785714285714288
        assert [1.0, 1.0, 1.0] == list(w_scot[1].values())

        w_col = self.obj_col.read()
        assert w_col.n == 49
        assert w_col.mean_neighbors == 4.6938775510204085
        assert [0.5, 0.5] == list(w_col[1].values())

    def test_seek(self):
        self.test_read()
        pytest.raises(StopIteration, self.obj_scot.read)
        pytest.raises(StopIteration, self.obj_col.read)
        self.obj_scot.seek(0)
        self.obj_col.seek(0)
        self.test_read()

    def test_write(self):
        for obj in [self.obj_scot, self.obj_col]:
            w = obj.read()
            f = tempfile.NamedTemporaryFile(suffix="")
            fname = f.name
            f.close()
            o = psopen(fname, "w", "geobugs_text")
            o.write(w)
            o.close()
            wnew = psopen(fname, "r", "geobugs_text").read()
            assert wnew.pct_nonzero == w.pct_nonzero
            os.remove(fname)

import os
import tempfile

import pytest

from .... import examples as pysal_examples
from ...fileio import FileIO
from ..stata_txt import StataTextIO


class TesttestStataTextIO:
    def setup_method(self):
        self.test_file_sparse = test_file_sparse = pysal_examples.get_path(
            "stata_sparse.txt"
        )
        self.test_file_full = test_file_full = pysal_examples.get_path("stata_full.txt")
        self.obj_sparse = StataTextIO(test_file_sparse, "r")
        self.obj_full = StataTextIO(test_file_full, "r")

    def test_close(self):
        for obj in [self.obj_sparse, self.obj_full]:
            f = obj
            f.close()
            pytest.raises(ValueError, f.read)

    def test_read(self):
        w_sparse = self.obj_sparse.read()
        assert w_sparse.n == 56
        assert w_sparse.mean_neighbors == 4.0
        assert list(w_sparse[1].values()) == [1.0, 1.0, 1.0, 1.0, 1.0]

        w_full = self.obj_full.read()
        assert w_full.n == 56
        assert w_full.mean_neighbors == 4.0
        assert list(w_full[1].values()) == [0.125, 0.125, 0.125, 0.125, 0.125]

    def test_seek(self):
        self.test_read()
        pytest.raises(StopIteration, self.obj_sparse.read)
        pytest.raises(StopIteration, self.obj_full.read)
        self.obj_sparse.seek(0)
        self.obj_full.seek(0)
        self.test_read()

    def test_write(self):
        for obj in [self.obj_sparse, self.obj_full]:
            w = obj.read()
            f = tempfile.NamedTemporaryFile(suffix=".txt")
            fname = f.name
            f.close()
            o = FileIO(fname, "w", "stata_text")
            if obj == self.obj_sparse:
                o.write(w)
            else:
                o.write(w, matrix_form=True)
            o.close()
            wnew = FileIO(fname, "r", "stata_text").read()
            assert wnew.pct_nonzero == w.pct_nonzero
            os.remove(fname)

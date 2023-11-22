# ruff: noqa: N806, N999, SIM115

import os
import tempfile

from .... import examples as pysal_examples
from ..pyShpIO import PurePyShpWrapper


class TesttestPurePyShpWrapper:
    def setup_method(self):
        test_file = pysal_examples.get_path("10740.shp")
        self.test_file = test_file
        self.shp_obj = PurePyShpWrapper(test_file, "r")
        f = tempfile.NamedTemporaryFile(suffix=".shp")
        shpcopy = f.name
        f.close()
        self.shpcopy = shpcopy
        self.shxcopy = shpcopy.replace(".shp", ".shx")

    def test_len(self):
        assert len(self.shp_obj) == 195

    def test_tell(self):
        assert self.shp_obj.tell() == 0
        self.shp_obj.read(1)
        assert self.shp_obj.tell() == 1
        self.shp_obj.read(50)
        assert self.shp_obj.tell() == 51
        self.shp_obj.read()
        assert self.shp_obj.tell() == 195

    def test_seek(self):
        self.shp_obj.seek(0)
        assert self.shp_obj.tell() == 0
        self.shp_obj.seek(55)
        assert self.shp_obj.tell() == 55
        self.shp_obj.read(1)
        assert self.shp_obj.tell() == 56

    def test_read(self):
        self.shp_obj.seek(0)
        objs = self.shp_obj.read()
        assert len(objs) == 195

        self.shp_obj.seek(0)
        objs_b = list(self.shp_obj)
        assert len(objs_b) == 195

        for shp_a, shp_b in zip(objs, objs_b, strict=True):
            assert shp_a.vertices == shp_b.vertices

    def test_random_access(self):
        self.shp_obj.seek(57)
        shp57 = self.shp_obj.read(1)[0]
        self.shp_obj.seek(32)
        shp32 = self.shp_obj.read(1)[0]

        self.shp_obj.seek(57)
        assert self.shp_obj.read(1)[0].vertices == shp57.vertices
        self.shp_obj.seek(32)
        assert self.shp_obj.read(1)[0].vertices == shp32.vertices

    def test_write(self):
        out = PurePyShpWrapper(self.shpcopy, "w")
        self.shp_obj.seek(0)
        for shp in self.shp_obj:
            out.write(shp)
        out.close()

        orig = open(self.test_file, "rb")
        copy = open(self.shpcopy, "rb")
        assert orig.read() == copy.read()
        orig.close()
        copy.close()

        oshx = open(self.test_file.replace(".shp", ".shx"), "rb")
        cshx = open(self.shxcopy, "rb")
        assert oshx.read() == cshx.read()
        oshx.close()
        cshx.close()

        os.remove(self.shpcopy)
        os.remove(self.shxcopy)

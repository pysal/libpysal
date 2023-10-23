import os
import tempfile

from .... import examples as pysal_examples
from ..pyShpIO import PurePyShpWrapper


class Testtest_PurePyShpWrapper:
    def setup_method(self):
        test_file = pysal_examples.get_path("10740.shp")
        self.test_file = test_file
        self.shpObj = PurePyShpWrapper(test_file, "r")
        f = tempfile.NamedTemporaryFile(suffix=".shp")
        shpcopy = f.name
        f.close()
        self.shpcopy = shpcopy
        self.shxcopy = shpcopy.replace(".shp", ".shx")

    def test_len(self):
        assert len(self.shpObj) == 195

    def test_tell(self):
        assert self.shpObj.tell() == 0
        self.shpObj.read(1)
        assert self.shpObj.tell() == 1
        self.shpObj.read(50)
        assert self.shpObj.tell() == 51
        self.shpObj.read()
        assert self.shpObj.tell() == 195

    def test_seek(self):
        self.shpObj.seek(0)
        assert self.shpObj.tell() == 0
        self.shpObj.seek(55)
        assert self.shpObj.tell() == 55
        self.shpObj.read(1)
        assert self.shpObj.tell() == 56

    def test_read(self):
        self.shpObj.seek(0)
        objs = self.shpObj.read()
        assert len(objs) == 195

        self.shpObj.seek(0)
        objsB = list(self.shpObj)
        assert len(objsB) == 195

        for shpA, shpB in zip(objs, objsB):
            assert shpA.vertices == shpB.vertices

    def test_random_access(self):
        self.shpObj.seek(57)
        shp57 = self.shpObj.read(1)[0]
        self.shpObj.seek(32)
        shp32 = self.shpObj.read(1)[0]

        self.shpObj.seek(57)
        assert self.shpObj.read(1)[0].vertices == shp57.vertices
        self.shpObj.seek(32)
        assert self.shpObj.read(1)[0].vertices == shp32.vertices

    def test_write(self):
        out = PurePyShpWrapper(self.shpcopy, "w")
        self.shpObj.seek(0)
        for shp in self.shpObj:
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

# ruff: noqa: N999

from .... import examples as pysal_examples
from ...util import WKTParser
from .. import csvWrapper


class TesttestCsvWrapper:
    def setup_method(self):
        stl = pysal_examples.load_example("stl")
        self.test_file = test_file = stl.get_path("stl_hom.csv")
        self.obj = csvWrapper.csvWrapper(test_file, "r")

    def test_len(self):
        assert len(self.obj) == 78

    def test_tell(self):
        assert self.obj.tell() == 0
        self.obj.read(1)
        assert self.obj.tell() == 1
        self.obj.read(50)
        assert self.obj.tell() == 51
        self.obj.read()
        assert self.obj.tell() == 78

    def test_seek(self):
        self.obj.seek(0)
        assert self.obj.tell() == 0
        self.obj.seek(55)
        assert self.obj.tell() == 55
        self.obj.read(1)
        assert self.obj.tell() == 56

    def test_read(self):
        self.obj.seek(0)
        objs = self.obj.read()
        assert len(objs) == 78
        self.obj.seek(0)
        objs_b = list(self.obj)
        assert len(objs_b) == 78
        for row_a, row_b in zip(objs, objs_b, strict=True):
            assert row_a == row_b

    def test_casting(self):
        self.obj.cast("WKT", WKTParser())
        verts = [
            (-89.585220336914062, 39.978794097900391),
            (-89.581146240234375, 40.094867706298828),
            (-89.603988647460938, 40.095306396484375),
            (-89.60589599609375, 40.136119842529297),
            (-89.6103515625, 40.3251953125),
            (-89.269027709960938, 40.329566955566406),
            (-89.268562316894531, 40.285579681396484),
            (-89.154655456542969, 40.285774230957031),
            (-89.152763366699219, 40.054969787597656),
            (-89.151618957519531, 39.919403076171875),
            (-89.224777221679688, 39.918678283691406),
            (-89.411857604980469, 39.918041229248047),
            (-89.412437438964844, 39.931644439697266),
            (-89.495201110839844, 39.933486938476562),
            (-89.4927978515625, 39.980186462402344),
            (-89.585220336914062, 39.978794097900391),
        ]
        for i, pt in enumerate(self.obj.__next__()[0].vertices):
            assert pt[:] == verts[i]

    def test_by_col(self):
        for field in self.obj.header:
            assert len(self.obj.by_col[field]) == 78

    def test_slicing(self):
        chunk = self.obj[50:55, 1:3]
        assert chunk[0] == ["Jefferson", "Missouri"]
        assert chunk[1] == ["Jefferson", "Illinois"]
        assert chunk[2] == ["Miller", "Missouri"]
        assert chunk[3] == ["Maries", "Missouri"]
        assert chunk[4] == ["White", "Illinois"]

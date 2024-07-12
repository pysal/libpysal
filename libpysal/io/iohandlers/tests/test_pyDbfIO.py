# ruff: noqa: N999, SIM115

import os
import tempfile

from .... import examples as pysal_examples
from ..pyDbfIO import DBF


class TesttestDBF:
    def setup_method(self):
        self.test_file = test_file = pysal_examples.get_path("10740.dbf")
        self.dbObj = DBF(test_file, "r")

    def test_len(self):
        assert len(self.dbObj) == 195

    def test_tell(self):
        assert self.dbObj.tell() == 0
        self.dbObj.read(1)
        assert self.dbObj.tell() == 1
        self.dbObj.read(50)
        assert self.dbObj.tell() == 51
        self.dbObj.read()
        assert self.dbObj.tell() == 195

    def test_cast(self):
        assert self.dbObj._spec == []
        self.dbObj.cast("FIPSSTCO", float)
        assert self.dbObj._spec[1] == float  # noqa: E721

    def test_seek(self):
        self.dbObj.seek(0)
        assert self.dbObj.tell() == 0
        self.dbObj.seek(55)
        assert self.dbObj.tell() == 55
        self.dbObj.read(1)
        assert self.dbObj.tell() == 56

    def test_read(self):
        self.dbObj.seek(0)
        objs = self.dbObj.read()
        assert len(objs) == 195
        self.dbObj.seek(0)
        objs_b = list(self.dbObj)
        assert len(objs_b) == 195
        for row_a, row_b in zip(objs, objs_b, strict=True):
            assert row_a == row_b

    def test_random_access(self):
        self.dbObj.seek(0)
        db0 = self.dbObj.read(1)[0]
        assert db0 == [1, "35001", "000107", "35001000107", "1.07"]
        self.dbObj.seek(57)
        db57 = self.dbObj.read(1)[0]
        assert db57 == [58, "35001", "001900", "35001001900", "19"]
        self.dbObj.seek(32)
        db32 = self.dbObj.read(1)[0]
        assert db32 == [33, "35001", "000500", "35001000500", "5"]
        self.dbObj.seek(0)
        assert next(self.dbObj) == db0
        self.dbObj.seek(57)
        assert next(self.dbObj) == db57
        self.dbObj.seek(32)
        assert next(self.dbObj) == db32

    def test_write(self):
        f = tempfile.NamedTemporaryFile(suffix=".dbf")
        fname = f.name
        f.close()
        self.dbfcopy = fname
        self.out = DBF(fname, "w")
        self.dbObj.seek(0)
        self.out.header = self.dbObj.header
        self.out.field_spec = self.dbObj.field_spec
        for row in self.dbObj:
            self.out.write(row)
        self.out.close()

        orig = open(self.test_file, "rb")
        copy = open(self.dbfcopy, "rb")
        orig.seek(32)  # self.dbObj.header_size) #skip the header, file date has changed
        copy.seek(32)  # self.dbObj.header_size) #skip the header, file date has changed

        # PySAL writes proper DBF files with a terminator at the end, not everyone does.
        n = self.dbObj.record_size * self.dbObj.n_records  # bytes to read.
        assert orig.read(n) == copy.read(n)
        # self.assertEquals(orig.read(1), copy.read(1)) # last byte may fail
        orig.close()
        copy.close()
        os.remove(self.dbfcopy)

    def test_write_nones(self):
        import datetime
        import time

        f = tempfile.NamedTemporaryFile(suffix=".dbf")
        fname = f.name
        f.close()
        db = DBF(fname, "w")
        db.header = ["recID", "date", "strID", "aFloat"]
        db.field_spec = [("N", 10, 0), ("D", 8, 0), ("C", 10, 0), ("N", 5, 5)]
        records = []
        for i in range(10):
            d = datetime.date(*time.localtime()[:3])
            rec = [i + 1, d, str(i + 1), (i + 1) / 2.0]
            records.append(rec)
        records.append([None, None, "", None])
        records.append(rec)
        for rec in records:
            db.write(rec)
        db.close()
        db2 = DBF(fname, "r")
        assert records == db2.read()
        db2.close()
        os.remove(fname)

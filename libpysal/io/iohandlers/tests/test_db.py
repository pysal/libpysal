import os
import platform

import pytest
import shapely

from .... import examples as pysal_examples
from ... import geotable as pdio
from ...fileio import FileIO

try:
    import sqlalchemy

    missing_sql = False
except ImportError:
    missing_sql = True


windows = platform.system() == "Windows"


@pytest.mark.skipif(windows, reason="Skipping Windows due to `PermissionError`.")
@pytest.mark.skipif(missing_sql, reason="Missing dependency: SQLAlchemy.")
class TestSqliteReader:
    def setup_method(self):
        path = pysal_examples.get_path("new_haven_merged.dbf")
        if path is None:
            pysal_examples.load_example("newHaven")
            path = pysal_examples.get_path("new_haven_merged.dbf")
        df = pdio.read_files(path)
        df["GEOMETRY"] = shapely.to_wkb(shapely.points(df["geometry"].values.tolist()))
        # This is a hack to not have to worry about a custom point type in the DB
        del df["geometry"]
        self.dbf = "iohandlers_test_db.db"
        engine = sqlalchemy.create_engine(f"sqlite:///{self.dbf}")
        self.conn = engine.connect()
        df.to_sql(
            "newhaven",
            self.conn,
            index=True,
            dtype={
                # Should convert the df date into a true date object, just a hack again
                "date": sqlalchemy.types.UnicodeText,
                "dataset": sqlalchemy.types.UnicodeText,
                "street": sqlalchemy.types.UnicodeText,
                "intersection": sqlalchemy.types.UnicodeText,
                "time": sqlalchemy.types.UnicodeText,  # As above re: date
                "GEOMETRY": sqlalchemy.types.BLOB,
            },
        )  # This is converted to TEXT as lowest type common sqlite

    def test_deserialize(self):
        db = FileIO(f"sqlite:///{self.dbf}")
        assert db.tables == ["newhaven"]

        gj = db._get_gjson("newhaven")
        assert gj["type"] == "FeatureCollection"

        self.conn.close()

        os.remove(self.dbf)

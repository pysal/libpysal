from shapely import wkb

from .. import fileio

errmsg = ""

try:
    from sqlalchemy import create_engine
    from sqlalchemy.ext.automap import automap_base
    from sqlalchemy.orm import Session

    nosql_mode = False
except ImportError:
    nosql_mode = True
    errmsg += (
        "No module named sqlalchemy. Please install"
        " sqlalchemy to enable this functionality."
    )


class SQLConnection(fileio.FileIO):
    """Reads an SQL mappable."""

    FORMATS = ["sqlite", "db"]
    MODES = ["r"]

    def __init__(self, *args, **kwargs):
        if errmsg != "":
            raise ImportError(errmsg)

        self._typ = str
        fileio.FileIO.__init__(self, *args, **kwargs)

        self.dbname = args[0]
        self.Base = automap_base()
        self._engine = create_engine(self.dbname)
        self.Base.prepare(autoload_with=self._engine)
        self.metadata = self.Base.metadata

    def read(self, *args, **kwargs):
        return self._get_gjson(*args, **kwargs)

    def seek(self):
        pass

    def __next__(self):
        pass

    def close(self):
        self.file.close()
        fileio.FileIO.close(self)

    def _get_gjson(self, tablename: str, geom_column="GEOMETRY"):
        gjson = {"type": "FeatureCollection", "features": []}

        for row in self.session.query(self.metadata.tables[tablename]):
            feat = {"type": "Feature", "geometry": {}, "properties": {}}
            feat["GEOMETRY"] = wkb.loads(getattr(row, geom_column))

            attributes = row._asdict()
            attributes.pop(geom_column, None)

            feat["properties"] = attributes
            gjson["features"].append(feat)

        return gjson

    @property
    def tables(self) -> list:
        if not hasattr(self, "_tables"):
            self._tables = list(self.metadata.tables.keys())

        return self._tables

    @property
    def session(self):
        """Create an ``sqlalchemy.orm.Session`` instance.

        Returns
        -------
        self._session : sqlalchemy.orm.Session
            An ``sqlalchemy.orm.Session`` instance.

        """

        # What happens if the session is externally closed?  Check for None?
        if not hasattr(self, "_session"):
            self._session = Session(self._engine)
        return self._session

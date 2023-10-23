import numpy as np

# import pysal_examples
from ... import examples as pysal_examples
from ...common import pandas
from ..fileio import FileIO as psopen

PANDAS_EXTINCT = pandas is None


class TestTable:
    def setup_method(self):
        self.filehandler = psopen(pysal_examples.get_path("columbus.dbf"))
        self.df = self.filehandler.to_df()
        self.filehandler.seek(0)
        self.shapefile = psopen(pysal_examples.get_path("columbus.shp"))
        self.csvhandler = psopen(pysal_examples.get_path("usjoin.csv"))
        self.csv_df = self.csvhandler.to_df()
        self.csvhandler.seek(0)

    def test_to_df(self):
        for column in self.csv_df.columns:
            if column.lower() == "name":
                continue
            np.testing.assert_allclose(
                self.csvhandler.by_col(column), self.csv_df[column].values
            )
        for column in self.df.columns:
            if column == "geometry":
                continue
            np.testing.assert_allclose(self.filehandler.by_col(column), self.df[column])

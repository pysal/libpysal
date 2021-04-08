from xarray import DataArray
from scipy import sparse
import numpy
from sklearn.preprocessing import LabelEncoder


class W_XA:
    def __init__(self, adjlist):
        self.xarray = DataArray.from_series(
            adjlist.set_index(["focal", "neighbor"])["weight"], sparse=True
        ).fillna(0)
        self._cache = dict()

    @classmethod
    def from_w(cls, w):
        return cls(w.to_adjlist())

    def neighbors(self, ix):
        # assume that the ix is not iterable, like currently done in W
        (neighbors,) = self.xarray.sel(focal=ix).data.nonzero()
        return self.xarray.neighbor[neighbors]

    def weights(self, ix):
        # assume that the ix is not iterable, like currently done in W
        neighbors = self.neighbors(ix)
        return self.xarray.sel(focal=ix).data[neighbors]

    @property
    def sparse(self):
        try:
            return self._cache["sparse"]
        except KeyError:
            coo = self.xarray.data
            self._cache["sparse"] = sparse.csr_matrix((coo.data, coo.nonzero()))
            return self.sparse


if __name__ == "__main__":
    import geopandas, libpysal

    counties = geopandas.read_file(libpysal.examples.get_path("NAT.shp"))
    w = libpysal.weights.Rook.from_dataframe(counties, ids=counties.FIPS.tolist())
    alist = w.to_adjlist()
    wx = W_XA(alist)

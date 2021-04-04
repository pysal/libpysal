counties = geopandas.read_file("./data/ncovr/ncovr/NAT.gpkg")
w = libpysal.weights.Rook.from_dataframe(counties, ids=counties.FIPS.tolist())
alist = w.to_adjlist()
wx = xarray.DataArray.from_series(
    alist.set_index(["focal", "neighbor"])["weight"], sparse=True
).fillna(0)

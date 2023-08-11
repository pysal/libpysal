"""
import geodatasets
import geopandas
import pytest

columbus = geopandas.read_file(geodatasets.get_path("geoda columbus"))
columbus["intID"] = columbus.POLYID.values
columbus["strID"] = columbus.POLYID.astype(str)
rivers = geopandas.read_file(geodatasets.get_path("eea large_rivers"))
rivers["strID"] = rivers.NAME
rivers["intID"] = rivers.index.values + 1

id_types = [None, "intID", "strID"]
id_type_names = ["no index", "int index", "string index"]

geom_names = ["columbus", "columbus centroids", "rivers"]
geoms = [columbus, columbus.set_geometry(columbus.geometry.centroid), rivers]

shuffle = [False, True]
external_ids = [False, True]
input_type = ["gdf", "gseries", "array"]

parametrize_inputs = pytest.mark.parametrize("geoms", geoms, ids=geom_names)
parametrize_idtypes = pytest.mark.parametrize("ids", id_types, ids=id_type_names)
parametrize_shuffle = pytest.mark.parametrize(
    "shuffle", shuffle, ids=["input order", "shuffled"]
)
parametrize_input_type = pytest.mark.parametrize(
    "input_type", input_type, ids=input_type
)
parametrize_external_ids = pytest.mark.parametrize(
    "external_ids", external_ids, ids=["use set_index", "use id vector"]
)
"""
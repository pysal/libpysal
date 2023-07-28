import geodatasets
import geopandas
import numpy
import pytest
import shapely

from libpysal.graph._contiguity import \
    _VALID_GEOMETRY_TYPES as contiguity_types
from libpysal.graph._kernel import \
    _VALID_GEOMETRY_TYPES as kernel_types
from libpysal.graph._triangulation import \
    _VALID_GEOMETRY_TYPES as triang_types
from libpysal.graph._utils import _validate_geometry_input

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


@parametrize_shuffle
@parametrize_external_ids
@parametrize_inputs
@parametrize_idtypes
@parametrize_input_type
def test_validate_input_geoms(geoms, ids, shuffle, external_ids, input_type):
    """
    Test that all combinations of geometries and ids get aligned correctly
    """
    if ids is not None:
        geoms = geoms.set_index(ids)
    if external_ids:
        input_ids = geoms.index
    else:
        input_ids = None
    if shuffle:
        geoms = geoms.sample(frac=1, replace=False)
    if input_type == "gdf":
        geoms = geoms
        geom_type = geoms.geometry.iloc[0].geom_type
    elif input_type == "gseries":
        geoms = geoms.geometry
        geom_type = geoms.iloc[0].geom_type
    elif input_type == "array":
        geoms = geoms.geometry.values
        geom_type = geoms[0].geom_type
    else:
        raise ValueError(
            'input_type not in supported testing types: "gdf", "gseries", "array"'
        )

    coordinates, ids, out_geoms = _validate_geometry_input(geoms, ids=input_ids)
    assert (out_geoms.index == ids).all(), "validated ids are not equal to input ids"
    if geom_type == "Point":
        assert coordinates.shape[0] == len(
            geoms
        ), "Point inputs should be cast to coordinates, but the output coordinates and input geometries are not equal length"
        assert coordinates.shape[0] == len(
            ids
        ), "Point inputs should be cast to coordinates, but the output coordinates and output ids are not equal length"
        if hasattr(geoms, "geometry"):
            coords = shapely.get_coordinates(geoms.geometry)
        else:
            coords = shapely.get_coordinates(geoms)
        numpy.testing.assert_array_equal(coordinates, coords)


@parametrize_shuffle
@parametrize_idtypes
def test_validate_input_coords(shuffle, ids):
    """
    Test that input coordinate arrays get validated correctly
    """
    if shuffle:
        data = columbus.sample(frac=1, replace=False)
    else:
        data = columbus
    input_coords = shapely.get_coordinates(data.centroid)
    if ids is not None:
        ids = data[ids].values
    coordinates, ids, out_geoms = _validate_geometry_input(input_coords, ids=ids)
    assert coordinates.shape[0] == len(out_geoms)
    assert coordinates.shape[0] == len(ids)


def test_validate_raises(
    kernel_types=kernel_types,
    contiguity_types=contiguity_types,
    triang_types=triang_types,
):
    with pytest.raises(ValueError):  # no lines for kernels
        _validate_geometry_input(rivers, valid_geometry_types=kernel_types)
    with pytest.raises(ValueError):  # no polygons for kernels
        _validate_geometry_input(columbus, valid_geometry_types=kernel_types)
    with pytest.raises(ValueError):  # no point gdf for contiguity
        _validate_geometry_input(
            columbus.set_geometry(columbus.centroid),
            valid_geometry_types=contiguity_types,
        )
    with pytest.raises(ValueError):  # no point gseries for contiguity
        _validate_geometry_input(
            columbus.set_geometry(columbus.centroid).geometry,
            valid_geometry_types=contiguity_types,
        )
    with pytest.raises(ValueError):  # no point arrays for contiguity
        _validate_geometry_input(
            numpy.arange(20).reshape(-1, 2), valid_geometry_types=contiguity_types
        )

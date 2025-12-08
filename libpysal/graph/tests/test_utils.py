# ruff: noqa: N811

import geodatasets
import geopandas
import numpy
import pytest
import shapely

from libpysal.graph._contiguity import _VALID_GEOMETRY_TYPES as contiguity_types
from libpysal.graph._kernel import _VALID_GEOMETRY_TYPES as kernel_types
from libpysal.graph._triangulation import _VALID_GEOMETRY_TYPES as triang_types
from libpysal.graph._utils import _validate_geometry_input


@pytest.fixture(scope="session")
def guerry():
    guerry = geopandas.read_file(geodatasets.get_path("geoda guerry"))
    guerry["intID"] = range(len(guerry))
    guerry["strID"] = guerry.intID.astype(str)
    return guerry


@pytest.fixture(scope="session")
def rivers():
    rivers = geopandas.read_file(geodatasets.get_path("eea large_rivers"))
    rivers["strID"] = rivers.NAME
    rivers["intID"] = rivers.index.values + 1
    return rivers


@pytest.fixture(params=["guerry", "guerry centroids", "rivers"])
def geoms(guerry, rivers, request):
    if request.param == "guerry":
        return guerry
    elif request.param == "guerry centroids":
        return guerry.set_geometry(guerry.geometry.centroid)
    elif request.param == "rivers":
        return rivers
    else:
        raise ValueError(
            "geoms not in supported testing types: "
            "'guerry', 'guerry centroids', 'rivers'"
        )


parametrize_idtypes = pytest.mark.parametrize(
    "ids",
    [None, "intID", "strID"],
    ids=["no index", "int index", "string index"],
)
parametrize_shuffle = pytest.mark.parametrize(
    "shuffle", [False, True], ids=["input order", "shuffled"]
)
parametrize_input_type = pytest.mark.parametrize(
    "input_type",
    ["gdf", "gseries", "array"],
)
parametrize_external_ids = pytest.mark.parametrize(
    "external_ids", [False, True], ids=["use set_index", "use id vector"]
)


@pytest.mark.network
@parametrize_shuffle
@parametrize_external_ids
@parametrize_idtypes
@parametrize_input_type
def test_validate_input_geoms(geoms, ids, shuffle, external_ids, input_type):
    """
    Test that all combinations of geometries and ids get aligned correctly
    """
    if ids is not None:
        geoms = geoms.set_index(ids)
    input_ids = geoms.index if external_ids else None
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
            "input_type not in supported testing types: 'gdf', 'gseries', 'array'"
        )

    coordinates, ids, out_geoms = _validate_geometry_input(geoms, ids=input_ids)
    assert (out_geoms.index == ids).all(), "validated ids are not equal to input ids"
    if geom_type == "Point":
        assert coordinates.shape[0] == len(geoms), (
            "Point inputs should be cast to coordinates, "
            "but the output coordinates and input geometries are not equal length"
        )
        assert coordinates.shape[0] == len(ids), (
            "Point inputs should be cast to coordinates, "
            "but the output coordinates and output ids are not equal length"
        )
        if hasattr(geoms, "geometry"):
            coords = shapely.get_coordinates(geoms.geometry)
        else:
            coords = shapely.get_coordinates(geoms)
        numpy.testing.assert_array_equal(coordinates, coords)


@pytest.mark.network
@parametrize_shuffle
@parametrize_idtypes
def test_validate_input_coords(shuffle, ids, guerry):
    """
    Test that input coordinate arrays get validated correctly
    """
    data = guerry.sample(frac=1, replace=False) if shuffle else guerry
    input_coords = shapely.get_coordinates(data.centroid)
    if ids is not None:
        ids = data[ids].values
    coordinates, ids, out_geoms = _validate_geometry_input(input_coords, ids=ids)
    assert coordinates.shape[0] == len(out_geoms)
    assert coordinates.shape[0] == len(ids)


@pytest.mark.network
def test_validate_raises(
    guerry,
    rivers,
    kernel_types=kernel_types,
    contiguity_types=contiguity_types,
    triang_types=triang_types,
):
    # kernels
    with pytest.raises(ValueError):  # no lines for kernels
        _validate_geometry_input(rivers, valid_geometry_types=kernel_types)
    with pytest.raises(ValueError):  # no polygons for kernels
        _validate_geometry_input(guerry, valid_geometry_types=kernel_types)
    # triangulation
    with pytest.raises(ValueError):  # no lines for triangulation
        _validate_geometry_input(rivers, valid_geometry_types=triang_types)
    with pytest.raises(ValueError):  # no polygons for triangulation
        _validate_geometry_input(guerry, valid_geometry_types=triang_types)
    # contiguity
    with pytest.raises(ValueError):  # no point gdf for contiguity
        _validate_geometry_input(
            guerry.set_geometry(guerry.centroid),
            valid_geometry_types=contiguity_types,
        )
    with pytest.raises(ValueError):  # no point gseries for contiguity
        _validate_geometry_input(
            guerry.set_geometry(guerry.centroid).geometry,
            valid_geometry_types=contiguity_types,
        )
    with pytest.raises(ValueError):  # no point arrays for contiguity
        _validate_geometry_input(
            numpy.arange(20).reshape(-1, 2), valid_geometry_types=contiguity_types
        )


def fetch_map_string(m):
    out = m._parent.render()
    out_str = "".join(out.split())
    return out_str

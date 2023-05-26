import pandas as pd
import numpy as np
import geopandas
import shapely


def _neighbor_dict_to_edges(neighbors, weights=None):
    idxs = pd.Series(neighbors).explode()
    if weights is not None:
        data_array = pd.Series(weights).explode().values
        if not pd.api.types.is_numeric_dtype(data_array):
            data_array = pd.to_numeric(data_array)
    else:
        data_array = np.ones(idxs.shape[0], dtype=int)
    return idxs.index.values, idxs.values, data_array



def _validate_geometry_input(geoms, ids=None, valid_geometry_types=None):
    if isinstance(geoms, (geopandas.GeoSeries, geopandas.GeoDataFrame)):
        geoms = geoms.geometry
        if ids is None:
            ids = geoms.index
        ids = np.asarray(ids)
        geom_types = set(geoms.geom_type)
        if valid_geometry_types is not None:
            if isinstance(valid_geometry_types, str):
                valid_geometry_types = (valid_geometry_types,)
            valid_geometry_types = set(valid_geometry_types)
            if not geom_types <= valid_geometry_types:
                raise ValueError(
                    f"this W type is only well-defined for geom_types: {valid_geometry_types}."
                )
            print(geom_types, valid_geometry_types)
        coordinates = shapely.get_coordinates(geoms)
        geoms = geoms.copy()
        geoms.index = ids
        return coordinates, ids, geoms
    elif isinstance(geoms.dtype, geopandas.array.GeometryDtype):
        return _validate_geometry_input(geopandas.GeoSeries(geoms), ids=ids, valid_geometry_types=valid_geometry_types)
    else:
        if (geoms.ndim == 2) and (geoms.shape[1] == 2):
            return _validate_geometry_input(geopandas.points_from_xy(*geoms.T), ids=ids, valid_geometry_types=valid_geometry_types)
    raise ValueError(
        "input geometry type is not supported. Input must either be a geopandas.GeoSeries, geopandas.GeoDataFrame, a numpy array with a geometry dtype, or an array of coordinates."
    )


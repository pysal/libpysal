from scipy import spatial
import pandas as pd
import numpy as np

from ._utils import _validate_geometry_input

_VALID_GEOMETRY_TYPES = ["Point"]


def _distance_band(coordinates, threshold, binary=True, alpha=-1.0, ids=None):
    """Generate adjacency table based on a distance band

    Parameters
    ----------
    coordinates : numpy.ndarray, geopandas.GeoSeries, geopandas.GeoDataFrame
        geometries over which to compute a kernel. If a geopandas.Geo* object
        is provided, the .geometry attribute is used. If a numpy.ndarray with
        a geometry dtype is used, then the coordinates are extracted and used.
    threshold : float
        distance band
    binary : bool, optional
        If True w_{ij}=1 if d_{i,j}<=threshold, otherwise w_{i,j}=0
        If False wij=dij^{alpha}, by default True.
    alpha : float, optional
        distance decay parameter for weight (default -1.0)
        if alpha is positive the weights will not decline with
        distance. If binary is True, alpha is ignored
    ids : array-like, optional
        ids to use for each sample in coordinates. Generally, construction functions
        that are accessed via Graph.build_kernel() will set this automatically from
        the index of the input. Do not use this argument directly unless you intend
        to set the indices separately from your input data. Otherwise, use
        data.set_index(ids) to ensure ordering is respected. If None, then the index
        from the input coordinates will be used.

    Returns
    -------
    pandas.DataFrame
        adjacency table
    """
    coordinates, ids, _ = _validate_geometry_input(
        coordinates, ids=ids, valid_geometry_types=_VALID_GEOMETRY_TYPES
    )
    tree = spatial.KDTree(coordinates)
    dist = tree.sparse_distance_matrix(tree, threshold, output_type="ndarray")
    adjacency = pd.DataFrame(
        index=pd.Index(ids[dist["i"]], name="focal"),
        data={"neighbor": ids[dist["j"]], "weight": dist["v"]},
    )

    # drop diagnoal
    counts = adjacency.index.value_counts()
    no_isolates = counts[counts > 1]
    adjacency = adjacency[
        ~(
            adjacency.index.isin(no_isolates.index)
            & (adjacency.index == adjacency.neighbor)
        )
    ]

    if binary:
        adjacency["weight"] = adjacency.weight.astype(bool).astype(int)
    else:
        if alpha != 1.0:
            adjacency["weight"] = np.power(adjacency["weight"], alpha)
            adjacency["weight"] = adjacency["weight"].fillna(0)  # handle isolates
    return adjacency

    # TODO: handle co-located points

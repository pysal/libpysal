import numpy as np

from ._utils import _induce_cliques, _validate_geometry_input


def _build_coplanarity_node_lookup(geoms):
    """
    Identify coplanar points and create a look-up table for the coplanar geometries.
    Same function as in ``graph._utils``, but need to keep the index to use as graph ids
    """
    # geoms = geoms.reset_index(drop=True)
    coplanar = []
    nearest = []
    r = geoms.groupby(geoms).groups
    for g in r.values():
        if len(g) == 2:
            coplanar.append(g[0])
            nearest.append(g[1])
        elif len(g) > 2:
            for n in g[1:]:
                coplanar.append(n)
                nearest.append(g[0])
    return np.asarray(coplanar), np.asarray(nearest)


def pdna_to_adj(origins, network, node_ids, threshold):
    """Create an adjacency list of shortest network-based travel between
       origins and destinations in a pandana.Network.

    Parameters
    ----------
    origins : geopandas.GeoDataFrame
        Geodataframe of origin geometries to begin routing.
    network : pandana.Network
        pandana.Network instance that stores the local travel network
    node_ids:
        array of node_ids in the pandana.Network aligned with the input
        observations in ``origins``. This is created via a call like
        ``pandana.Network.get_node_ids(df.geometry.x, df.geometry.y)``
    threshold : int
        maximum travel distance (inclusive)

    Returns
    -------
    pandas.DataFrame
        adjacency list with columns 'origin', 'destination', and 'cost'
    """

    # map node ids in the network to index in the gdf
    mapper = dict(zip(node_ids, origins.index.values, strict=False))

    namer = {"source": "origin", network.impedance_names[0]: "cost"}

    adj = network.nodes_in_range(node_ids, threshold)
    adj = adj.rename(columns=namer)
    # swap osm ids for gdf index
    adj = adj.set_index("destination").rename(index=mapper).reset_index()
    adj = adj.set_index("origin").rename(index=mapper).reset_index()
    adj = adj[adj.destination.isin(origins.index.values)]

    return adj


def build_travel_graph(df, network, threshold, mapping_distance):
    """Compute the shortest path between gdf centroids via a pandana.Network
    and return an adjacency list with weight=cost. Note unlike distance_band,
    :math:`G_{ij}` and :math:`G_{ji}` are often different because travel networks
    may be directed.

    Parameters
    ----------
    df : geopandas.GeoDataFrame
        geodataframe of observations. CRS should be the same as the locations
        of ``node_x`` and ``node_y`` in the pandana.Network (usually 4326 if network
        comes from OSM, but sometimes projected to improve snapping quality).
    network : pandana.Network
        Network that encodes travel costs. See <https://udst.github.io/pandana/>
    threshold : int
        maximum travel cost to consider neighbors
    mapping_distance : int
        snapping tolerance passed to ``pandana.Network.get_node_ids`` that defines
        the maximum range at which observations are snapped to nearest nodes in the
        network. Default is None

    Returns
    -------
    pandas.Series
        adjacency formatted as multiindexed (focal, neighbor) series
    """
    df = df.copy()
    _validate_geometry_input(df.geometry, ids=None, valid_geometry_types="Point")
    df["node_ids"] = network.get_node_ids(
        df.geometry.x, df.geometry.y, mapping_distance
    )

    # depending on density of the graph nodes / observations, it is common to have
    # multiple observations snapped to the same network node, so use the clique
    # expansion logic to handle these cases

    # get indices of multi-observations at unique nodes
    coplanar, nearest = _build_coplanarity_node_lookup(df["node_ids"])
    # create adjacency on unique nodes
    adj = pdna_to_adj(df, network, df["node_ids"].values, threshold)
    # add clique members back to adjacency
    adj_cliques = _induce_cliques(
        adj.rename(
            columns={"origin": "focal", "destination": "neighbor", "cost": "weight"}
        ),
        coplanar=coplanar,
        nearest=nearest,
    )
    # reorder, drop induced dupes, and return
    adj_cliques = (
        adj_cliques.groupby(["focal", "neighbor"])
        .first()
        .reindex(df.index, level=0)
        .reindex(df.index, level=1)
        .reset_index()
    )

    return adj_cliques

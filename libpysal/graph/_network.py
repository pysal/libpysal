import numpy as np

from ._utils import _induce_cliques


def _build_coplanarity_node_lookup(geoms):
    """
    Identify coplanar points and create a look-up table for the coplanar geometries.
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


def pdna_to_adj(origins, network, threshold, reindex=True, drop_nonorigins=True):
    """Create an adjacency list of shortest network-based travel between
       origins and destinations in a pandana.Network.

    Parameters
    ----------
    origins : geopandas.GeoDataFrame
        Geodataframe of origin geometries to begin routing. If geometries are
        polygons, they will be collapsed to centroids
    network : pandana.Network
        pandana.Network instance that stores the local travel network
    threshold : int
        maximum travel distance (inclusive)
    reindex : bool, optional
        if True, use geodataframe index to identify observations in the
        adjlist. If False, the node_id from the OSM node nearest each
        observation will be used. by default True
    drop_nonorigins : bool, optional
        If True, drop any destination nodes that are not also origins,
        by default True

    Returns
    -------
    pandas.DataFrame
        adjacency list with columns 'origin', 'destination', and 'cost'
    """
    node_ids = network.get_node_ids(origins.centroid.x, origins.centroid.y).astype(int)

    # map node ids in the network to index in the gdf
    mapper = dict(zip(node_ids, origins.index.values))

    namer = {"source": "origin", network.impedance_names[0]: "cost"}

    adj = network.nodes_in_range(node_ids, threshold)
    adj = adj.rename(columns=namer)
    # swap osm ids for gdf index
    if reindex:
        adj = adj.set_index("destination").rename(index=mapper).reset_index()
        adj = adj.set_index("origin").rename(index=mapper).reset_index()
    if drop_nonorigins:
        adj = adj[adj.destination.isin(origins.index.values)]

    return adj


def build_travel_graph(
    df,
    network,
    threshold,
):
    df = df.copy()
    df["node_ids"] = network.get_node_ids(
        df.geometry.centroid.x, df.geometry.centroid.y
    )

    coplanar, nearest = _build_coplanarity_node_lookup(df["node_ids"])
    adj = pdna_to_adj(df, network, threshold, reindex=True, drop_nonorigins=True)
    adj_cliques = _induce_cliques(
        adj.rename(
            columns={"origin": "focal", "destination": "neighbor", "cost": "weight"}
        ),
        coplanar=coplanar,
        nearest=nearest,
    )

    adj_cliques = (
        adj_cliques.groupby(["focal", "neighbor"])
        .first()
        .reindex(df.index, level=0)
        .reindex(df.index, level=1)
        .reset_index()
    )

    return adj_cliques

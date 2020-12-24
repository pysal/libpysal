import pandas as pd
import requests

from tqdm.auto import tqdm
from pandas.testing import assert_frame_equal
from .weights import W


def feeds_from_bbox(bbox):
    """Generate a dict of GTFS feeds by querying the transitland api with a bounding box

    Parameters
    ----------
    bbox : list-like
        A list or tuple of bounding box coordinates formatted as (lng_max, lat_min, lng_min, lat_max). e.g. For a geodataframe
        stored in epsg 4326, this can be obtained with geodataframe.total_bounds

    Returns
    -------
    dict
        dictorionary of gtfs feeds formatted as {`provider_name`: `url`}. 
        This dict can be fed directly to ua.gtfsfeeds.feeds.add_feed(
    """
    bbox = ",".join([str(i) for i in bbox])
    q = requests.get(f"https://transit.land/api/v1/feeds?bbox={bbox}&per_page=1000")
    feeds = pd.DataFrame.from_records(q.json()["feeds"])
    # several providers are missing names, but the `onestop_id` field appears to have useful info
    feeds.onestop_id = (
        feeds.onestop_id.str.split("-")
        .apply(lambda x: x[-1])
        .str.split("~")
        .apply(lambda x: x[0])
    )
    feeds.name = feeds.name.fillna(feeds.onestop_id)
    feeds = (
        feeds[["name", "url"]].dropna(subset=["url"]).set_index("name").to_dict()["url"]
    )
    return feeds


class NetworkW(W):
    @classmethod
    def from_dataframe(cls, df=None, network=None, ids=None, max_dist=None, **kwargs):
        """
        Make Network weights from a dataframe.

        Parameters
        ----------
        df      :   pandas.DataFrame
                    a dataframe with a geometry column that can be used to
                    construct a W object
        network:    pandana.Network
                    a pandana Network object (optionally created by `multimodal_from_bbox`). If none, the geodataframe's total_bounds
                    attribute will be used to download an openstreetmap pedestrian network covering the study area.
        ids     :   string or iterable
                    if string, the column name of the indices from the dataframe
                    if iterable, a list of ids to use for the W
                    if None, pandana node_ids are used
        max_dist:   int
                    maximum distance along the road network to be considered neighbors

        See Also
        --------
        :class:`libpysal.weights.weights.W`
        """
        if not network:
            try:
                from pandana.loaders import osm
            except ImportError:
                raise ImportError(
                    "You must have pandana installed to generate a Network"
                )
            assert (
                "epsg:4326" in df.crs.to_string()
            ), "You must pass in an explicit pandana.Network object or provide a geodataframe in geographic (epsg:4326) coordinates"
            network = osm.pdna_network_from_bbox(bbox=tuple(df.total_bounds))

        adj = compute_travel_cost_adjlist(
            df, df, network, index_orig=ids, max_dist=max_dist, index_dest=ids
        )

        return W.from_adjlist(
            adj, focal_col="origin", neighbor_col="destination", weight_col="cost"
        )


def compute_travel_cost_adjlist(
    origins, destinations, network, index_orig=None, index_dest=None, max_dist=None
):
    """Generate travel cost adjacency list.

    Parameters
    ----------
    origins : geopandas.GeoDataFrame
        a geodataframe containing the locations of origin features
    destinations : geopandas.GeoDataFrame
        a geodataframe containing the locations of destination features
    network : pandana.Network
        pandana Network instance for calculating the shortest path between origins and destinations
    index_orig : str, optional
        Unique index on the origins dataframe.
    index_dest : str, optional
        Unique index on the destinations dataframe.
    Returns
    -------
    pandas.DataFrame
        pandas DataFrame containing the shortest-cost distance from each origin feature to each destination feature
    """
    origins = origins.copy()
    destinations = destinations.copy()

    origins["osm_ids"] = network.get_node_ids(
        origins.centroid.x, origins.centroid.y
    ).astype(int)

    try:  #  only do the node lookup once if os and ds are same
        assert_frame_equal(origins.drop(columns=['osm_ids']), destinations)
        destinations["osm_ids"] = origins["osm_ids"].copy()
    except Exception:
        destinations["osm_ids"] = network.get_node_ids(
            destinations.centroid.x, destinations.centroid.y
        ).astype(int)

    ods = []

    if not index_orig:
        origins["idx"] = origins.index.values
        index_orig = "idx"
    if not index_dest:
        destinations["idx"] = destinations.index.values
        index_dest = "idx"

    # I dont think there's a way to do this in parallel, so we can at least show a progress bar
    with tqdm(total=len(origins["osm_ids"])) as pbar:
        for origin in origins["osm_ids"]:
            df = pd.DataFrame()
            df["cost"] = network.shortest_path_lengths(
                [origin for d in destinations["osm_ids"]],
                [d for d in destinations["osm_ids"]],
            )
            df["destination"] = destinations[index_dest].values
            df["origin"] = origins[origins.osm_ids == origin][index_orig].values[0]
            if max_dist:
                df = df[df["cost"] <= max_dist]

            ods.append(df)
            pbar.update(1)

    combined = pd.concat(ods).reset_index()
    return combined


def multimodal_from_bbox(
    bbox,
    gtfs_dir=None,
    save_osm=None,
    save_gtfs=None,
    excluded_feeds=None,
    transit_net_kwargs=None,
    headways=False,
    additional_feeds=None,
):
    """Generate a combined walk/transit pandana Network from a bounding box of latitudes and longitudes

    Parameters
    ----------
    bbox : tuple
        A bounding box formatted as (lng_max, lat_min, lng_min, lat_max). e.g. For a geodataframe
        stored in epsg 4326, this can be obtained with geodataframe.total_bounds
    gtfs_dir : str, optional
        path to directory for storing downloaded GTFS data. If None, the current directory will be used
    save_osm : str, optional
        Path to store the intermediate OSM Network as an h5 file
    save_gtfs : str, optional
        Path to store the intermediate GTFS Network as an h5 file
    excluded_feeds : list, optional
        list of feed names to exclude from the GTFS downloaded
    transit_net_kwargs : dict, optional
        additional keyword arguments to be passed to the urbanaccess GTFS network instantiator.
        defaults to {'day':"monday", 'timerange':["07:00:00", "10:00:00"]}
    headways : bool, optional
        Whether to include headway calculations for the combined network
    additional_feeds : dict, optional
        Dictionary of additional feed locations in case they are not hosted on transitland.
        Should be specified as {transitagencyname: url} 

    Returns
    -------
    pandana.Network
        a multimodal (walk/transit) Network object built from OSM and GTFS data that lie within the bounding box
    """
    try:
        import osmnet
        import pandana as pdna
        import urbanaccess as ua
    except ImportError:
        raise ImportError(
            "You must have osmnet, pandana, and urbanaccess installed to use this function"
        )

    assert bbox is not None, "You must provide a bounding box to collect network data"
    if not gtfs_dir:
        gtfs_dir = "./data/"

    if not transit_net_kwargs:
        transit_net_kwargs = dict(
            day="monday", timerange=["07:00:00", "10:00:00"], calendar_dates_lookup=None
        )

    # Get gtfs data
    feeds = feeds_from_bbox(bbox)

    if excluded_feeds:  # remove problematic feeds if necessary
        for feed in list(feeds.keys()):
            if feed in excluded_feeds:
                feeds.pop(feed)

    if len(ua.gtfsfeeds.feeds.to_dict()["gtfs_feeds"]) > 0:
        ua.gtfsfeeds.feeds.remove_feed(
            remove_all=True
        )  # feeds object is global so reset it if there's anything leftover

    ua.gtfsfeeds.feeds.add_feed(feeds)
    if additional_feeds:
        ua.gtfsfeeds.feeds.add_feed(additional_feeds)

    ua.gtfsfeeds.download(data_folder=gtfs_dir)

    loaded_feeds = ua.gtfs.load.gtfsfeed_to_df(
        f"{gtfs_dir}/gtfsfeed_text/", bbox=bbox, remove_stops_outsidebbox=True
    )
    if save_gtfs:
        ua_to_h5(loaded_feeds, f"{gtfs_dir}/{save_gtfs}")

    # Get OSM data
    nodes, edges = osmnet.network_from_bbox(bbox=bbox)
    osm_network = pdna.Network(
        nodes["x"], nodes["y"], edges["from"], edges["to"], edges[["distance"]]
    )
    if save_osm:
        osm_network.save_hdf5(save_osm)

    # Create the transit network
    ua.create_transit_net(gtfsfeeds_dfs=loaded_feeds, **transit_net_kwargs)
    osm_network.nodes_df["id"] = osm_network.nodes_df.index

    ua.create_osm_net(
        osm_edges=osm_network.edges_df,
        osm_nodes=osm_network.nodes_df,
        travel_speed_mph=3,
    )
    if headways:
        ua.gtfs.headways.headways(
            gtfsfeeds_df=loaded_feeds, headway_timerange=transit_net_kwargs["timerange"]
        )
        ua.network.integrate_network(
            urbanaccess_network=ua.ua_network,
            headways=True,
            urbanaccess_gtfsfeeds_df=loaded_feeds,
            headway_statistic="mean",
        )
    else:
        ua.integrate_network(urbanaccess_network=ua.ua_network, headways=False)

    combined_net = pdna.Network(
        ua.ua_network.net_nodes["x"],
        ua.ua_network.net_nodes["y"],
        ua.ua_network.net_edges["from_int"],
        ua.ua_network.net_edges["to_int"],
        ua.ua_network.net_edges[["weight"]],
    )

    return combined_net

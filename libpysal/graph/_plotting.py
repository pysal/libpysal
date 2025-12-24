import geopandas as gpd
import numpy as np
import pandas as pd
import shapely


def _plot(
    graph_obj,
    gdf,
    focal=None,
    nodes=True,
    color="k",
    edge_kws=None,
    node_kws=None,
    focal_kws=None,
    ax=None,
    figsize=None,
    limit_extent=False,
):
    """Plot edges and nodes of the Graph

    Creates a ``maptlotlib`` plot based on the topology stored in the
    Graph and spatial location defined in ``gdf``.

    Parameters
    ----------
    graph_obj : Graph
        Graph to be plotted
    gdf : geopandas.GeoDataFrame
        Geometries indexed using the same index as Graph. Geometry types other than
        points are converted to representative points encoding start and end point of
        Graph edges.
    focal : hashable | array-like[hashable] | None, optional
        ID or an array-like of IDs of focal geometries whose weights shall be
        plotted. If None, all weights from all focal geometries are plotted.
        By default None
    nodes : bool, optional
        Plot nodes as points. Nodes are plotted using zorder=2 to show them on top of
        the edges. By default True
    color : str, optional
        The color of all objects, by default "k"
    edge_kws : dict, optional
        Keyword arguments dictionary to send to ``LineCollection``,
        which provides fine-grained control over the aesthetics
        of the edges in the plot. By default None
    node_kws : dict, optional
        Keyword arguments dictionary to send to ``ax.scatter``,
        which provides fine-grained control over the aesthetics
        of the nodes in the plot. By default None
    focal_kws : dict, optional
        Keyword arguments dictionary to send to ``ax.scatter``,
        which provides fine-grained control over the aesthetics
        of the focal nodes in the plot on top of generic ``node_kws``.
        Values of ``node_kws`` are updated from ``focal_kws``.
        Ignored if ``focal=None``. By default None
    ax : matplotlib.axes.Axes, optional
        Axis on which to plot the weights. If None, a new figure and axis are
        created. By default None
    figsize : tuple, optional
        figsize used to create a new axis. By default None
    limit_extent : bool, optional
        limit the extent of the axis to the extent of the plotted graph, by default
        False

    Returns
    -------
    matplotlib.axes.Axes
        Axis with the resulting plot
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib import collections

    except (ImportError, ModuleNotFoundError) as err:
        raise ImportError("matplotlib is required for `plot`.") from err

    if ax is None:
        f, ax = plt.subplots(figsize=figsize)

    if node_kws is not None:
        if "color" not in node_kws:
            node_kws["color"] = color
    else:
        node_kws = {"color": color}

    if edge_kws is not None:
        if "color" not in edge_kws:
            edge_kws["color"] = color
    else:
        edge_kws = {"color": color}

    # get array of coordinates in the order reflecting graph_obj._adjacency.index.codes
    # we need to work on int position to allow fast filtering of duplicated edges and
    # cannot rely on gdf remaining in the same order between Graph creation and
    # plotting
    coords = shapely.get_coordinates(
        gdf.reindex(graph_obj.unique_ids).representative_point()
    )

    if focal is not None:
        if not pd.api.types.is_list_like(focal):
            focal = [focal]
        subset = graph_obj._adjacency[focal]
        codes = subset.index.codes

    else:
        codes = graph_obj._adjacency.index.codes

    # avoid plotting both ij and ji
    edges = np.unique(np.sort(np.column_stack([codes]).T, axis=1), axis=0)
    lines = coords[edges]

    ax.add_collection(collections.LineCollection(lines, **edge_kws))

    if limit_extent:
        xm, ym = lines.min(axis=0).min(axis=0)
        xx, yx = lines.max(axis=0).max(axis=0)
        x_margin = (xx - xm) * 0.05
        y_margin = (yx - ym) * 0.05
        ax.set_xlim(xm - x_margin, xx + x_margin)
        ax.set_ylim(ym - y_margin, yx + y_margin)
    else:
        ax.autoscale_view()

    if nodes:
        if focal is not None:
            used_focal = coords[np.unique(subset.index.codes[0])]
            used_neighbor = coords[np.unique(subset.index.codes[1])]
            ax.scatter(used_neighbor[:, 0], used_neighbor[:, 1], **node_kws, zorder=2)
            if focal_kws is None:
                focal_kws = {}
            ax.scatter(
                used_focal[:, 0],
                used_focal[:, 1],
                **dict(node_kws, **focal_kws),
                zorder=3,
            )
        else:
            ax.scatter(coords[:, 0], coords[:, 1], **node_kws, zorder=2)

    return ax


def _explore_graph(
    g,
    gdf,
    focal=None,
    nodes=True,
    color="black",
    edge_kws=None,
    node_kws=None,
    focal_kws=None,
    m=None,
    **kwargs,
):
    """Plot graph as an interactive Folium Map

    Parameters
    ----------
    g : libpysal.Graph
        graph to be plotted
    gdf : geopandas.GeoDataFrame
        geodataframe used to instantiate to Graph
    focal : list, optional
        subset of focal observations to plot in the map, by default None.
        If none, all relationships are plotted
    nodes : bool, optional
        whether to display observations as nodes in the map, by default True
    color : str, optional
        color applied to nodes and edges, by default "black"
    edge_kws : dict, optional
        additional keyword arguments passed to geopandas explore function
        when plotting edges, by default None
    node_kws : dict, optional
        additional keyword arguments passed to geopandas explore function
        when plotting nodes, by default None
    focal_kws : dict, optional
        additional keyword arguments passed to geopandas explore function
        when plotting focal observations, by default None. Only applicable when
        passing a subset of nodes with the `focal` argument
    m : Folilum.Map, optional
        folium map objecto to plot on top of, by default None
    **kwargs : dict, optional
        additional keyword arguments are passed directly to geopandas.explore, when
        ``m=None`` by default None


    Returns
    -------
    folium.Map
        folium map
    """
    geoms = gdf.representative_point().reindex(g.unique_ids)

    if node_kws is not None:
        if "color" not in node_kws:
            node_kws["color"] = color
    else:
        node_kws = {"color": color}

    if focal_kws is not None:
        if "color" not in node_kws:
            focal_kws["color"] = color
    else:
        focal_kws = {"color": color}

    if edge_kws is not None:
        if "color" not in edge_kws:
            edge_kws["color"] = color
    else:
        edge_kws = {"color": color}

    coords = shapely.get_coordinates(geoms)

    if focal is not None:
        if not pd.api.types.is_list_like(focal):
            focal = [focal]
        subset = g._adjacency[focal]
        codes = subset.index.codes
        adj = subset

    else:
        codes = g._adjacency.index.codes
        adj = g._adjacency

    # avoid plotting both ij and ji
    edges, indices = np.unique(
        np.sort(np.column_stack([codes]).T, axis=1), return_index=True, axis=0
    )
    lines = coords[edges]
    lines = gpd.GeoSeries(
        shapely.linestrings(lines),
        crs=gdf.crs,
    )
    adj = adj.iloc[indices].reset_index()
    edges = gpd.GeoDataFrame(adj, geometry=lines)[
        ["focal", "neighbor", "weight", "geometry"]
    ]

    m = (
        edges.explore(m=m, **edge_kws)
        if m is not None
        else edges.explore(**edge_kws, **kwargs)
    )

    if nodes is True:
        if focal is not None:
            # destinations
            geoms.iloc[np.unique(subset.index.codes[1])].explore(m=m, **node_kws)
            if focal_kws is None:
                focal_kws = {}
            # focals
            geoms.iloc[np.unique(subset.index.codes[0])].explore(
                m=m, **dict(node_kws, **focal_kws)
            )
        else:
            geoms.explore(m=m, **node_kws)
    return m

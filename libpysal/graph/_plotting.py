import numpy as np
import shapely


def _plot(
    G, gdf, nodes=True, color="k", edge_kws=None, node_kws=None, ax=None, figsize=None
):
    """Plot edges and nodes of the Graph

    Creates a ``maptlotlib`` plot based on the topology stored in the
    Graph and spatial location defined in ``gdf``.

    Parameters
    ----------
    G : Graph
        Graph to be plotted
    gdf : geopandas.GeoDataFrame
        Geometries indexed using the same index as Graph. Geomtry types other than
        points are converted to centroids encoding start and end point of Graph
        edges.
    nodes : bool, optional
        Plot nodes as points. Nodes are plotted using zorder=2 to show them on top of
        the edges. By default True
    color : str, optional
        The color of all objects, by default "k"
    edge_kws : _type_, optional
        Keyword arguments dictionary to send to ``LineCollection``,
        which provides fine-grained control over the aesthetics
        of the edges in the plot. By default None
    node_kws : _type_, optional
        Keyword arguments dictionary to send to ``ax.scatter``,
        which provides fine-grained control over the aesthetics
        of the nodes in the plot. By default None
    ax : matplotlib.axes.Axes, optional
        Axis on which to plot the weights. If None, a new figure and axis are
        created. By default None
    figsize : tuple, optional
        figsize used to create a new axis. By default None

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
        node_kws = dict(color=color)

    if edge_kws is not None:
        if "color" not in edge_kws:
            edge_kws["color"] = color
    else:
        edge_kws = dict(color=color)

    coords = shapely.get_coordinates(gdf.centroid)
    lines = np.column_stack(
        [
            coords[G._adjacency.index.get_level_values("focal")],
            coords[G._adjacency.index.get_level_values("neighbor")],
        ]
    ).reshape(-1, 2, 2)

    ax.add_collection(collections.LineCollection(lines, **edge_kws))
    ax.autoscale_view()

    if nodes:
        ax.scatter(coords[:, 0], coords[:, 1], **node_kws, zorder=2)

    return ax

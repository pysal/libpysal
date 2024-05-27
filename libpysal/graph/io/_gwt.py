import pandas as pd


def _read_gwt(path):
    """
    Read GWT weights to Graph object

    Parameters
    ----------
    path : str
        path to GWT file

    Returns
    -------
    tuple
        focal, neighbor, weight arrays
    """
    adjacency = pd.read_csv(path, sep=r"\s+", skiprows=1, header=None)
    return adjacency[0].values, adjacency[1].values, adjacency[2].values


def _to_gwt(graph_obj, path):
    """
    Write GWT weights to Graph object

    Parameters
    ----------
    graph_obj : Graph
        Graph object
    path : str
        path to GAL file
    """
    adj = graph_obj._adjacency.reset_index()
    adj["focal"] = adj["focal"].astype(str).str.replace(" ", "_")
    adj["neighbor"] = adj["neighbor"].astype(str).str.replace(" ", "_")
    with open(path, "w") as file:
        file.write(f"0 {graph_obj.n} Unknown Unknown\n")
    adj.to_csv(path, sep=" ", header=False, index=False, mode="a", float_format="%.7f")

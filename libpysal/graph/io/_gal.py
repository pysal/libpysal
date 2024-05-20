import contextlib


def _read_gal(path):
    """Read GAL weights to Graph object

    Parameters
    ----------
    path : str
        path to GAL file

    Returns
    -------
    dict
        neighbors dict
    """
    with open(path) as file:
        neighbors = {}

        # handle case where more than n is specified in first line
        header = file.readline().strip().split()
        header_n = len(header)
        n = int(header[0])

        if header_n > 1:
            n = int(header[1])

        for _ in range(n):
            id_, _ = file.readline().strip().split()
            neighbors_i = file.readline().strip().split()
            neighbors[id_] = neighbors_i

    # try casting to ints to ensure loss-less roundtrip of integer node ids
    with contextlib.suppress(ValueError):
        neighbors = {int(k): list(map(int, v)) for k, v in neighbors.items()}

    return neighbors


def _to_gal(graph_obj, path):
    """Write GAL weights to Graph object

    Parameters
    ----------
    graph_obj : Graph
        Graph object
    path : str
        path to GAL file
    """
    grouper = graph_obj._adjacency.groupby(level=0, sort=False)

    with open(path, "w") as file:
        file.write(f"{graph_obj.n}\n")

        for ix, chunk in grouper:
            if ix in graph_obj.isolates:
                neighbors = []
            else:
                neighbors = (
                    chunk.index.get_level_values("neighbor").astype(str).tolist()
                )

            file.write(f"{ix} {len(neighbors)}\n")
            file.write(" ".join(neighbors) + "\n")

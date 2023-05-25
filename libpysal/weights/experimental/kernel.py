import numpy, pandas, scipy, shapely
from .base import W

coordinates = numpy.random.normal(0, 1, size=(100, 2))


def _triangular(distances, bandwidth):
    u = numpy.clip(distances / bandwidth, 0, 1)
    return 1 - u


def _parabolic(distances, bandwidth):
    u = numpy.clip(distances / bandwidth, 0, 1)
    return 0.75 * (1 - u**2)


def _gaussian(distances, bandwidth):
    u = distances / bandwidth
    return numpy.exp(-((u / 2) ** 2)) / (numpy.sqrt(2) * numpy.pi)


def _bisquare(distances, bandwidth):
    u = numpy.clip(distances / bandwidth, 0, 1)
    return (15 / 16) * (1 - u**2) ** 2


def _cosine(distances, bandwidth):
    u = numpy.clip(distances / bandwidth, 0, 1)
    return (numpy.pi / 4) * numpy.cos(numpy.pi / 2 * u)


def _boxcar(distances, bandwidth):
    return (distances > bandwidth).astype(distances.dtype)


_kernel_functions = dict(
    triangular=_triangular,
    parabolic=_parabolic,
    gaussian=_gaussian,
    bisquare=_bisquare,
    cosine=_cosine,
    boxcar=_boxcar,
    discrete=_boxcar,
)


def kernel(
    coordinates,
    bandwidth=None,
    metric="euclidean",
    kernel="triangular",
    k=None,
    ids=None,
    **kwargs,
):
    if hasattr(coordinates, "geometry"):
        if ids is None:
            ids = coordinates.index
        assert (
            coordinates.geom_type.unique == "Point"
        ).all(), "this graph type is only well-defined for point geometries."
        coordinates = shapely.get_coordinates(coordinates)
    else:
        if ids is None:
            ids = pandas.RangeIndex(n_samples)
    if metric == "precomputed":
        assert (
            coordinates.shape[0] == coordinates.shape[1]
        ), "coordinates should represent a distance matrix if metric='precomputed'"

    if k is not None:
        if metric != "precomputed":
            p = kwargs.get("p", 2)
            if metric == "haversine":
                # sklearn haversine works with (lat,lng) in radians...
                coordinates = numpy.fliplr(coordinates.deg2rad())
            query = _prepare_tree_query(coordinates, metric, p=p)
            D_linear, ixs = query(coordinates, k=k + 1)
            self_ix, neighbor_ix = ixs[:, 0], ixs[:, 1:]
            D_linear = D_linear[:, 1:]
            self_ix_flat = numpy.repeat(self_ix, k)
            neighbor_ix_flat = neighbor_ix.flatten()
            D_linear_flat = D_linear.flatten()
            if metric == "haversine":
                D_linear_flat * 6371  # express haversine distances in kilometers
            D = scipy.sparse.csc_array(
                (D_linear_flat, (self_ix_flat, neighbor_ix_flat))
            )
        else:
            D = coordinates * (coordinates.argsort(axis=1, kind="stable") < (k + 1))
    else:
        if metric != "precomputed":
            D = scipy.spatial.distance.pdist(coordinates, metric=metric)
            D = scipy.sparse.csc_array(scipy.spatial.distance.squareform(D))
        else:
            D = scipy.sparse.csc_array(coordinates)
    if bandwidth is None:
        bandwidth = numpy.percentile(D.data, 25)
    elif bandwidth == "opt":
        bandwidth = _optimize_bandwidth(D, kernel)
    K = D.copy()
    if callable(kernel):
        K.data = kernel(D.data, bandwidth)
    else:
        K.data = _kernel_functions[kernel](D.data, bandwidth)
    return K
    return W.from_sparse(K, ids=ids)


def _prepare_tree_query(coordinates, metric, p=2):
    try:
        from sklearn.neighbors import BallTree as tree

        return tree(coordinates, metric=metric).query
    except ImportError:
        if metric in ("euclidean", "manhattan", "cityblock", "minkowski"):
            from scipy.spatial import KDTree as tree

            tree_ = tree(coordinates)
            p = {"euclidean": 2, "manhattan": 1, "cityblock": 1, "minkowski": p}

            def query(target, k):
                return tree_.query(target, k=k, p=p)

            return query
        else:
            raise ValueError(
                f"metric {metric} is not supported by scipy, and scikit-learn is not able to be imported"
            )


def _from_dataframe(df, **kwargs):
    coordinates = get_points_array(df.geometry)
    indices = df.index
    return kernel(coordinates, indices=indices, **kwargs)


# kernel.from_dataframe = _from_dataframe


def _optimize_bandwidth(D, kernel):
    kernel_function = _kernel_functions[kernel]

    def _loss(bandwidth, D=D, kernel_function=kernel_function):
        Ku = kernel_function(D.data, bandwidth)
        bins, _ = numpy.histogram(Ku, bins=int(D.shape[0] ** 0.5), range=(0, 1))
        return -stats.entropy(bins / bins.sum())

    xopt = minimize_scalar(_loss, bounds=(0, D.data.max() * 2), method="bounded")
    return xopt.x

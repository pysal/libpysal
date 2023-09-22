import numpy
from scipy import sparse, optimize, spatial, stats

from ._utils import (
    _validate_geometry_input,
    _build_coincidence_lookup,
    _induce_cliques,
    _jitter_geoms,
    _sparse_to_arrays,
    _resolve_islands,
)

try:
    from sklearn import neighbors, metrics

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

_VALID_GEOMETRY_TYPES = ["Point"]


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
    r = (distances < bandwidth).astype(int)
    return r


def _identity(distances, bandwidth):
    return distances


_kernel_functions = {
    "triangular": _triangular,
    "parabolic": _parabolic,
    "gaussian": _gaussian,
    "bisquare": _bisquare,
    "cosine": _cosine,
    "boxcar": _boxcar,
    "discrete": _boxcar,
    "identity": _identity,
    None: _identity,
}


def _kernel(
    coordinates,
    bandwidth=None,
    metric="euclidean",
    kernel="gaussian",
    k=None,
    ids=None,
    p=2,
    taper=True,
    coincident="raise",
):
    """
    Compute a kernel function over a distance matrix.

    Paramters
    ---------
    coordinates : numpy.ndarray, geopandas.GeoSeries, geopandas.GeoDataFrame
        geometries over which to compute a kernel. If a geopandas.Geo* object
        is provided, the .geometry attribute is used. If a numpy.ndarray with
        a geometry dtype is used, then the coordinates are extracted and used.
    bandwidth : float (default: None)
        distance to use in the kernel computation. Should be on the same scale as
        the input coordinates.
    metric : string or callable (default: 'euclidean')
        distance function to apply over the input coordinates. Supported options
        depend on whether or not scikit-learn is installed. If so, then any
        distance function supported by scikit-learn is supported here. Otherwise,
        only euclidean, minkowski, and manhattan/cityblock distances are admitted.
    kernel : string or callable (default: 'gaussian')
        kernel function to apply over the distance matrix computed by `metric`.
        The following kernels are supported:
            - triangular:
            - parabolic:
            - gaussian:
            - bisquare:
            - cosine:
            - boxcar/discrete: all distances less than `bandwidth` are 1, and all
                other distances are 0
            - identity/None : do nothing, weight similarity based on raw distance
            - callable : a user-defined function that takes the distance vector and
                the bandwidth and returns the kernel: kernel(distances, bandwidth)
    k : int (default: None)
        number of nearest neighbors used to truncate the kernel. This is assumed
        to be constant across samples. If None, no truncation is conduted.
    ids : numpy.narray (default: None)
        ids to use for each sample in coordinates. Generally, construction functions
        that are accessed via Graph.build_kernel() will set this automatically from
        the index of the input. Do not use this argument directly unless you intend
        to set the indices separately from your input data. Otherwise, use
        data.set_index(ids) to ensure ordering is respected. If None, then the index
        from the input coordinates will be used.
    p : int (default: 2)
        parameter for minkowski metric, ignored if metric != "minkowski".
    taper : bool (default: True)
        remove links with a weight equal to zero
    """
    if metric != "precomputed":
        coordinates, ids, _ = _validate_geometry_input(
            coordinates, ids=ids, valid_geometry_types=_VALID_GEOMETRY_TYPES
        )
    else:
        assert (
            coordinates.shape[0] == coordinates.shape[1]
        ), "coordinates should represent a distance matrix if metric='precomputed'"
        if ids is None:
            ids = numpy.arange(coordinates.shape[0])

    if metric == "haversine":
        if not (
            (coordinates[:, 0] > -180)
            & (coordinates[:, 0] < 180)
            & (coordinates[:, 1] > -90)
            & (coordinates[:, 1] < 90)
        ).all():
            raise ValueError(
                "'haversine' metric is limited to the range of "
                "latitude coordinates (-90, 90) and the range of "
                "longitude coordinates (-180, 180)."
            )

    if k is not None:
        if metric != "precomputed":
            D = _knn(coordinates, k=k, metric=metric, p=p, coincident=coincident)
        else:
            D = coordinates * (coordinates.argsort(axis=1, kind="stable") < (k + 1))
    else:
        if metric != "precomputed":
            dist_kwds = {}
            if metric == "minkowski":
                dist_kwds["p"] = p
            if HAS_SKLEARN:
                sq = metrics.pairwise_distances(
                    coordinates, coordinates, metric=metric, **dist_kwds
                )
            else:
                if metric not in ("euclidean", "manhattan", "cityblock", "minkowski"):
                    raise ValueError(
                        f"metric {metric} is not supported by scipy, and scikit-learn "
                        "could not be imported."
                    )
                D = spatial.distance.pdist(coordinates, metric=metric, **dist_kwds)
                sq = spatial.distance.squareform(D)

            # ensure that self-distance is dropped but 0 between co-located pts not
            # get data and ids for sparse constructor
            data = sq.flatten()
            i = numpy.tile(numpy.arange(sq.shape[0]), sq.shape[0])
            j = numpy.repeat(numpy.arange(sq.shape[0]), sq.shape[0])
            # remove diagonal
            data = numpy.delete(data, numpy.arange(0, data.size, sq.shape[0] + 1))
            i = numpy.delete(i, numpy.arange(0, i.size, sq.shape[0] + 1))
            j = numpy.delete(j, numpy.arange(0, j.size, sq.shape[0] + 1))
            # construct sparse
            D = sparse.csc_array((data, (i, j)))
        else:
            D = sparse.csc_array(coordinates)
    if bandwidth is None:
        bandwidth = numpy.percentile(D.data, 25)
    elif bandwidth == "auto":
        if (kernel == "identity") or (kernel is None):
            bandwidth = numpy.nan  # ignored by identity
        else:
            bandwidth = _optimize_bandwidth(D, kernel)
    if callable(kernel):
        D.data = kernel(D.data, bandwidth)
    else:
        D.data = _kernel_functions[kernel](D.data, bandwidth)

    if taper:
        D.eliminate_zeros()

    heads, tails, weights = _sparse_to_arrays(D, ids=ids)

    return _resolve_islands(heads, tails, ids, weights)


def _knn(coordinates, metric="euclidean", k=1, p=2, coincident="raise"):
    """internal function called only from within _kernel, never directly to build KNN"""
    coordinates, ids, geoms = _validate_geometry_input(
        coordinates, ids=None, valid_geometry_types=_VALID_GEOMETRY_TYPES
    )
    n_coincident, coincident_lut = _build_coincidence_lookup(geoms)
    max_at_one_site = coincident_lut.input_index.str.len().max()
    n_samples, _ = coordinates.shape

    if max_at_one_site <= k:
        if metric == "haversine":
            # sklearn haversine works with (lat,lng) in radians...
            coordinates = numpy.fliplr(numpy.deg2rad(coordinates))
        query = _prepare_tree_query(coordinates, metric, p=p)
        D_linear, ixs = query(coordinates, k=k + 1)
        self_ix, neighbor_ix = ixs[:, 0], ixs[:, 1:]
        D_linear = D_linear[:, 1:]
        self_ix_flat = numpy.repeat(self_ix, k)
        neighbor_ix_flat = neighbor_ix.flatten()
        D_linear_flat = D_linear.flatten()
        if metric == "haversine":
            D_linear_flat * 6371  # express haversine distances in kilometers
        D = sparse.csr_array(
            (D_linear_flat, (self_ix_flat, neighbor_ix_flat)),
            shape=(n_samples, n_samples),
        )
        return D

    else:
        if coincident == "raise":
            raise ValueError(
                f"There are {len(coincident_lut)} "
                f"unique locations in the dataset, but {len(coordinates)} observations. "
                f"At least one of these sites has {max_at_one_site} points, more than the "
                f"{k} nearest neighbors requested. This means there are more than {k} points "
                "in the same location, which makes this graph type undefined. To address "
                "this issue, consider setting `coincident='clique' or consult the "
                "documentation about coincident points."
            )
        if coincident == "jitter":
            # force re-jittering over and over again until the coincidence is broken
            return _knn(
                _jitter_geoms(coordinates, geoms)[-1],
                metric=metric,
                k=k,
                p=p,
                coincident="jitter",
            )

        if coincident == "clique":
            raise NotImplementedError(
                "clique-based resolver of coincident points is not yet implemented."
            )
            # # implicit coincident == "clique"
            # heads, tails, weights = _sparse_to_arrays(
            #     _knn(coincident_lut.geometry, metric=metric, k=k, p=p, coincident="raise")
            # )
            # adjtable = pandas.DataFrame.from_dict(
            #     dict(focal=heads, neighbor=tails, weight=weights)
            # )
            # adjtable = _induce_cliques(adjtable, coincident_lut, fill_value=0)
            # return sparse.csr_array(
            #     adjtable.weight.values,
            #     (adjtable.focal.values, adjtable.neighbor.values),
            #     shape=(n_samples, n_samples),
            # )
        raise ValueError(
            f"'{coincident}' is not a valid option. Use one of "
            "['raise', 'jitter', 'clique']."
        )


def _distance_band(coordinates, threshold, ids=None):
    coordinates, ids, _ = _validate_geometry_input(
        coordinates, ids=ids, valid_geometry_types=_VALID_GEOMETRY_TYPES
    )
    tree = spatial.KDTree(coordinates)
    sp = sparse.csr_array(tree.sparse_distance_matrix(tree, threshold))
    return sp


def _prepare_tree_query(coordinates, metric, p=2):
    """
    Construct a tree query function relevant to the input metric.
    Prefer scikit-learn trees if they are available.
    """
    if HAS_SKLEARN:
        dist_kwds = {}
        if metric == "minkowski":
            dist_kwds["p"] = p
        if metric in neighbors.VALID_METRICS["kd_tree"]:
            tree = neighbors.KDTree
        else:
            tree = neighbors.BallTree
        return tree(coordinates, metric=metric, **dist_kwds).query
    else:
        if metric in ("euclidean", "manhattan", "cityblock", "minkowski"):
            from scipy.spatial import KDTree as tree

            tree_ = tree(coordinates)
            p = {"euclidean": 2, "manhattan": 1, "cityblock": 1, "minkowski": p}[metric]

            def query(target, k):
                return tree_.query(target, k=k, p=p)

            return query
        else:
            raise ValueError(
                f"metric {metric} is not supported by scipy, and scikit-learn could "
                "not be imported"
            )


def _optimize_bandwidth(D, kernel):
    """
    Optimize the bandwidth as a function of entropy for a given kernel function.

    This ensures that the entropy of the kernel is maximized for a given
    distance matrix. This will result in the smoothing that provide the most
    uniform distribution of kernel values, which is a good proxy for a
    "moderate" level of smoothing.
    """
    kernel_function = _kernel_functions.get(kernel, kernel)
    assert callable(
        kernel_function
    ), f"kernel {kernel} was not in supported kernel types {_kernel_functions.keys()} or callable"

    def _loss(bandwidth, D=D, kernel_function=kernel_function):
        Ku = kernel_function(D.data, bandwidth)
        bins, _ = numpy.histogram(Ku, bins=int(D.shape[0] ** 0.5), range=(0, 1))
        return -stats.entropy(bins / bins.sum())

    xopt = optimize.minimize_scalar(
        _loss, bounds=(0, D.data.max() * 2), method="bounded"
    )
    return xopt.x

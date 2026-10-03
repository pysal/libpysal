import warnings

import numpy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array
from sklearn.metrics import pairwise_distances

from ._utils import _validate_geometry_input

_VALID_GEOMETRY_TYPES = ["Point"]


def _spatial_matching(
    x,
    y=None,
    n_matches=5,
    metric="euclidean",
    solver=None,
    return_mip=False,
    allow_partial_match=False,
    **metric_kwargs,
):
    """
    Match locations in one dataset to at least `n_matches`
    locations in another (possibly identical) dataset
    by minimizing the total distance between matched locations.

    Letting :math:`d_{ij}` be

    .. math::

        \\text{minimize} \\sum_i^n \\sum_j^n  d_{ij}m_{ij}

        \\text{subject to}
            \\sum_j^n m_{ij} >= k \\forall i

            m_{ij} \\in {0,1} \\forall ij


    Parameters
    ----------
    x : numpy.ndarray, geopandas.GeoSeries, geopandas.GeoDataFrame
        geometries that need matches. If a geopandas.Geo* object
        is provided, the .geometry attribute is used. If a numpy.ndarray with
        a geometry dtype is used, then the coordinates are extracted and used.
    y : numpy.ndarray, geopandas.GeoSeries, geopandas.GeoDataFrame (default: None)
        geometries that are used as a source for matching. If a geopandas.Geo* object
        is provided, the .geometry attribute is used. If a numpy.ndarray with
        a geometry dtype is used, then the coordinates are extracted and
        used. If none, matches are made within `x`.
    n_matches : int (default: None)
        number of matches
    metric : string or callable (default: 'euclidean')
        distance function to apply over the input coordinates. Supported options
        depend on whether or not scikit-learn is installed. If so, then any
        distance function supported by scikit-learn is supported here. Otherwise,
        only euclidean, minkowski, and manhattan/cityblock distances are admitted.
    solver : solver from pulp (default: None)
        a solver defined by the pulp optimization library. If no solver is
        provided, SciPy's mixed-integer linear programming solver is used,
        unless ``return_mip=True`` requests a PuLP problem.
    return_mip : bool (default: False)
        whether or not to return the instance of the pulp.LpProblem. By
        default, the problem is not returned to the user.
    allow_partial_match : bool (default: False)
        whether to allow for partial matching. A partial match may have
        a weight between zero and one, while a "full" match (by default)
        must have a weight of either zero or one. A partial matching may
        have a shorter total distance, but will result in a weighted
        graph.
    Notes
    -----
    By default, matching uses SciPy and does not require PuLP. Passing a custom
    ``solver`` or requesting ``return_mip=True`` uses the PuLP implementation.
    If SciPy cannot find an optimal solution, matching falls back to PuLP
    and its existing warning behavior. This fallback requires PuLP.

    """
    if metric == "precomputed":
        distance_matrix = numpy.asarray(x)
        x_ids = numpy.arange(distance_matrix.shape[0])
        y_ids = numpy.arange(distance_matrix.shape[1])
        match_between = y is not None
    elif y is not None:
        x, x_ids, _ = _validate_geometry_input(
            x, ids=None, valid_geometry_types=_VALID_GEOMETRY_TYPES
        )
        y, y_ids, _ = _validate_geometry_input(
            y, ids=None, valid_geometry_types=_VALID_GEOMETRY_TYPES
        )
        distance_matrix = pairwise_distances(x, y, metric=metric)
        match_between = True
    else:
        x, x_ids, _ = _validate_geometry_input(
            x, ids=None, valid_geometry_types=_VALID_GEOMETRY_TYPES
        )
        y_ids = x_ids
        distance_matrix = pairwise_distances(x, metric=metric, **metric_kwargs)

        match_between = False

    n_targets, n_sources = distance_matrix.shape

    if match_between:
        row, col = numpy.meshgrid(
            numpy.arange(n_targets), numpy.arange(n_sources), indexing="ij"
        )
        row = row.flatten()
        col = col.flatten()
    else:
        # if we are matching within, we need to
        row, col = numpy.triu_indices(
            n=n_targets, m=n_sources, k=int(not match_between)
        )

    if solver is None and not return_mip:
        # Each column is an edge variable. Between datasets its endpoints have
        # distinct constraints; within a dataset both endpoints share the same
        # degree constraints.
        variables = numpy.arange(len(row))
        constraint_rows = numpy.concatenate(
            [row, col + n_targets if match_between else col]
        )
        constraint_columns = numpy.tile(variables, 2)
        n_constraints = n_targets + n_sources if match_between else n_targets
        incidence = coo_array(
            (numpy.ones(2 * len(row)), (constraint_rows, constraint_columns)),
            shape=(n_constraints, len(row)),
        ).tocsc()
        lower = numpy.full(n_constraints, n_matches, dtype=float)
        upper = numpy.full(n_constraints, numpy.inf)
        if match_between:
            lower[n_targets:] = -numpy.inf
            upper[n_targets:] = n_matches
        elif allow_partial_match:
            upper[:] = n_matches

        result = milp(
            distance_matrix[row, col],
            integrality=0 if allow_partial_match else 1,
            bounds=Bounds(0, 1),
            constraints=LinearConstraint(incidence, lower, upper),
        )
        if result.success:
            values = result.x if allow_partial_match else numpy.rint(result.x)
            selected = values > 0
            heads, tails, weights = row[selected], col[selected], values[selected]
            if not match_between:
                heads, tails = (
                    numpy.concatenate([heads, tails]),
                    numpy.concatenate([tails, heads]),
                )
                weights = numpy.tile(weights, 2)
            order = numpy.lexsort((tails, heads))
            return x_ids[heads[order]], y_ids[tails[order]], weights[order]

    try:
        import pulp
    except ImportError as error:
        raise ImportError(
            "Custom solvers, return_mip=True, and failed SciPy optimization "
            "require the pulp library"
        ) from error

    mp = pulp.LpProblem("optimal-neargraph", sense=pulp.LpMinimize)
    # a match is as binary decision variable, connecting observation i to observation j
    match_vars = pulp.LpVariable.dicts(
        "match",
        lowBound=0,
        upBound=1,
        indices=zip(row, col, strict=True),
        cat="Continuous" if allow_partial_match else "Binary",
    )
    # we want to minimize the geographic distance of links in the graph
    mp.objective = pulp.lpSum(
        [
            match_vars[i, j] * distance_matrix[i, j]
            for i, j in zip(row, col, strict=True)
        ]
    )

    # for each observation
    for j in range(n_targets):
        # there must be exactly k other matched observations, which might live
        if match_between:
            summand = pulp.lpSum(
                [
                    # over the whole match matrix
                    match_vars[j, i]
                    for i in range(n_sources)
                ]
            )
            sense = 1
        else:
            summand = pulp.lpSum(
                [
                    # in the "upper" triangle, or "lower" triangle
                    match_vars.get((i, j), match_vars.get((j, i)))
                    for i in range(n_sources)
                    if (i != j)
                ]
            )
            sense = int(not allow_partial_match)
        mp += pulp.LpConstraint(summand, sense=sense, rhs=n_matches)
    if match_between:  # but, we may choose to ignore some sources
        for i in range(n_sources):
            summand = pulp.lpSum([match_vars[j, i] for j in range(n_targets)])
            mp += pulp.LpConstraint(summand, sense=-1, rhs=n_matches)

    status = mp.solve(solver)

    if (status != 1) & (not allow_partial_match):
        warnings.warn(
            f"Problem is {pulp.LpStatus[status]}, so edge weights may be non-integer!",
            stacklevel=2,
        )

    edges = [
        (*key, value.value()) for key, value in match_vars.items() if value.value() > 0
    ]
    if not match_between:
        edges.extend([(*tuple(reversed(edge[:-1])), edge[-1]) for edge in edges])

    heads, tails, weights = map(numpy.asarray, zip(*sorted(edges), strict=True))

    if return_mip:
        return x_ids[heads], y_ids[tails], weights, mp
    return x_ids[heads], y_ids[tails], weights

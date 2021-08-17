import functools as _f
from warnings import warn
from ...common import requires as _requires

__all__ = [
    "to_wkb",
    "to_wkt",
    "area",
    "distance",
    "length",
    "boundary",
    "bounds",
    "centroid",
    "representative_point",
    "convex_hull",
    "envelope",
    "buffer",
    "simplify",
    "difference",
    "intersection",
    "symmetric_difference",
    "union",
    "has_z",
    "is_empty",
    "is_ring",
    "is_simple",
    "is_valid",
    "relate",
    "contains",
    "crosses",
    "disjoint",
    "equals",
    "intersects",
    "overlaps",
    "touches",
    "within",
    "equals_exact",
    "almost_equals",
    "project",
    "interpolate",
]


def _atomic_op(df, geom_col="geometry", inplace=False, _func=None, **kwargs):
    """Atomic operation internal function."""

    outval = df[geom_col].apply(lambda x: _func(x, **kwargs))
    outcol = "shape_{}".format(_func.__name__)
    if not inplace:
        new = df.copy()
        new[outcol] = outval
        return new
    df[outcol] = outval


_doc_template = """
Tabular version of pysal.contrib.shapely_ext.{n}

Arguments
---------
df : pandas.DataFrame
    A pandas dataframe with a geometry column.
geom_col : str
    The name of the column in ``df`` containing the geometry.
inplace : bool
    A boolean denoting whether to operate on the dataframe inplace or to
    return a series contaning the results of the computation. If
    operating inplace, the derived column will be under 'shape_{n}'.
**kwargs : dict
    Keyword arguments to be passed to the elementwise functions.

Returns
-------
If inplace, None, and operation is conducted on dataframe in memory. Otherwise,
returns a series.

Notes
-----

Some atomic operations require an 'other' argument.

See Also
--------

pysal.contrib.shapely_ext.{n}

"""

# ensure that the construction of atomics is done only if we can use shapely
_shapely_atomics = {}
try:
    from .. import shapely_ext as _s

    for k in __all__:
        _shapely_atomics.update({k: _f.partial(_atomic_op, _func=_s.__dict__[k])})
        _shapely_atomics[k].__doc__ = _doc_template.format(n=_s.__dict__[k].__name__)
except ImportError:
    pass
globals().update(_shapely_atomics)

##############
# Reductions #
##############


@_requires("shapely")
def cascaded_union(df, geom_col="geometry", **groupby_kws):
    """Returns the cascaded union of a possibly-grouped dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe containing geometry objects which are being unioned.
    geom_col : string
        The name of the column in ``df`` containing the geometry.
    **groupby_kws : dict
        Keyword arguments to pass transparently to the groupby function in ``df``.

    Returns
    -------
    out : {libpysal.cg.{Point, Chain, Polygon}, pandas.DataFrame}
        A PySAL shape or a dataframe of shapes resulting from the union operation.

    See Also
    --------
    
    pysal.shapely_ext.cascaded_union
    pandas.DataFrame.groupby
    
    """

    by = groupby_kws.pop("by", None)
    level = groupby_kws.pop("level", None)
    if by is not None or level is not None:
        df = df.groupby(by=by, level=level, **groupby_kws)
        out = df[geom_col].apply(_s.cascaded_union)
    else:
        out = _s.cascaded_union(df[geom_col].tolist())
    return out


@_requires("shapely")
def unary_union(df, geom_col="geometry", **groupby_kws):
    """Returns the unary union of a possibly-grouped dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe containing geometry objects which are being unioned.
    geom_col : string
        The name of the column in ``df`` containing the geometry.
    **groupby_kws : dict
        Keyword arguments to pass transparently to the groupby function in ``df``.

    Returns
    -------
    out : {libpysal.cg.{Point, Chain, Polygon}, pandas.DataFrame}
        A PySAL shape or a dataframe of shapes resulting from the union operation.

    See Also
    --------
    
    pysal.shapely_ext.unary_union
    pandas.DataFrame.groupby
    
    """

    by = groupby_kws.pop("by", None)
    level = groupby_kws.pop("level", None)
    if by is not None or level is not None:
        df = df.groupby(**groupby_kws)
        out = df[geom_col].apply(_cascaded_union)
    else:
        out = _cascaded_union(df[geom_col].tolist())
    return out


@_requires("shapely")
def _cascaded_intersection(shapes):
    it = iter(shapes)
    outshape = next(it)
    for i, shape in enumerate(it):
        try:
            outshape = _s.intersection(shape, outshape)
        except NotImplementedError:
            warn("An intersection is empty!")
            return None
    return outshape


@_requires("shapely")
def cascaded_intersection(df, geom_col="geometry", **groupby_kws):
    """Returns the cascaded intersection of a possibly-grouped dataframe

    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe containing geometry objects which are being intersectioned.
    geom_col : string
        The name of the column in ``df`` containing the geometry.
    **groupby_kws : dict
        Keyword arguments to pass transparently to the groupby function in ``df``.

    Returns
    -------
    out : {libpysal.cg.{Point, Chain, Polygon}, pandas.DataFrame}
        A PySAL shape or a dataframe of shapes resulting from the intersection operation.

    See Also
    --------
    
    pysal.shapely_ext.cascaded_intersection
    pandas.DataFrame.groupby
    
    """

    by = groupby_kws.pop("by", None)
    level = groupby_kws.pop("level", None)
    if by is not None or level is not None:
        df = df.groupby(**groupby_kws)
        out = df[geom_col].apply(_cascaded_intersection)
    else:
        out = _cascaded_intersection(df[geom_col].tolist())
    return out

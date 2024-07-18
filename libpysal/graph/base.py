import math
from functools import cached_property

import numpy as np
import pandas as pd
from packaging.version import Version
from scipy import __version__ as scipy_version
from scipy import sparse

from libpysal.weights import W

from ._contiguity import (
    _block_contiguity,
    _fuzzy_contiguity,
    _queen,
    _rook,
    _vertex_set_intersection,
)
from ._indices import _build_from_h3
from ._kernel import _distance_band, _kernel
from ._matching import _spatial_matching
from ._network import build_travel_graph as _build_travel_graph
from ._plotting import _explore_graph, _plot
from ._raster import _generate_da, _raster_contiguity
from ._set_ops import SetOpsMixin
from ._spatial_lag import _lag_spatial
from ._summary import GraphSummary
from ._triangulation import _delaunay, _gabriel, _relative_neighborhood, _voronoi
from ._utils import (
    _compute_stats,
    _evaluate_index,
    _neighbor_dict_to_edges,
    _percentile_filtration_grouper,
    _resolve_islands,
    _sparse_to_arrays,
)
from .io._gal import _read_gal, _to_gal
from .io._gwt import _read_gwt, _to_gwt
from .io._parquet import _read_parquet, _to_parquet

ALLOWED_TRANSFORMATIONS = ("O", "B", "R", "D", "V", "C")

# listed alphabetically
__author__ = """"
Martin Fleischmann (martin@martinfleischmann.net)
Eli Knaap (ek@knaaptime.com)
Serge Rey (sjsrey@gmail.com)
Levi John Wolf (levi.john.wolf@gmail.com)
"""

__all__ = [
    "Graph",
    "read_parquet",
    "read_gal",
    "read_gwt",
]


class Graph(SetOpsMixin):
    """Graph class encoding spatial weights matrices

    The :class:`Graph` is currently experimental
    and its API is incomplete and unstable.
    """

    def __init__(self, adjacency, transformation="O", is_sorted=False):
        """Weights base class based on adjacency list

        It is recommenced to use one of the ``from_*`` or ``build_*`` constructors
        rather than invoking ``__init__`` directly.

        Each observation needs to be present in the focal,
        at least as a self-loop with a weight 0.

        Parameters
        ----------
        adjacency : pandas.Series
            A MultiIndexed pandas.Series with ``"focal"`` and ``"neigbor"`` levels
            encoding adjacency, and values encoding weights. By convention,
            isolates are encoded as self-loops with a weight 0.
        transformation : str, default "O"
            weights transformation used to produce the table.

            - **O** -- Original
            - **B** -- Binary
            - **R** -- Row-standardization (global sum :math:`=n`)
            - **D** -- Double-standardization (global sum :math:`=1`)
            - **V** -- Variance stabilizing
            - **C** -- Custom
        is_sorted : bool, default False
            ``adjacency`` capturing the graph needs to be canonically sorted to
            initialize the class. The MultiIndex needs to be ordered i-->j
            on both focal and neighbor levels according to the order of ids in the
            original data from which the Graph is created. Sorting is performed by
            default based on the order of unique values in the focal level. Sorting
            needs to be reflected in both the values of the MultiIndex and also the
            underlying MultiIndex.codes. Set ``is_sorted=True`` to skip this step if the
            adjacency is already canonically sorted and you are certain about it.

        """
        if not isinstance(adjacency, pd.Series):
            raise TypeError(
                f"The adjacency table needs to be a pandas.Series. {type(adjacency)}"
            )
        if not tuple(adjacency.index.names) == ("focal", "neighbor"):
            raise ValueError(
                "The index of the adjacency table needs to be a MultiIndex named "
                "['focal', 'neighbor']."
            )
        if not adjacency.name == "weight":
            raise ValueError(
                "The adjacency needs to be named 'weight'. "
                f"'{adjacency.name}' was given instead."
            )
        if not pd.api.types.is_numeric_dtype(adjacency):
            raise ValueError(
                "The 'weight' needs to be of a numeric dtype. "
                f"'{adjacency.dtype}' dtype was given instead."
            )
        if adjacency.isna().any():
            raise ValueError("The adjacency table cannot contain missing values.")
        if transformation.upper() not in ALLOWED_TRANSFORMATIONS:
            raise ValueError(
                f"'transformation' needs to be one of {ALLOWED_TRANSFORMATIONS}. "
                f"'{transformation}' was given instead."
            )

        if not is_sorted:
            # adjacency always ordered i-->j on both levels
            ids = adjacency.index.get_level_values(0).unique().values
            adjacency = adjacency.reindex(ids, level=0).reindex(ids, level=1)

        self._adjacency = adjacency
        self.transformation = transformation

    def __getitem__(self, item):
        """Easy lookup based on focal index

        Parameters
        ----------
        item : hashable
            hashable represting an index value

        Returns
        -------
        pandas.Series
            subset of the adjacency table for `item`
        """
        if item in self.isolates:
            return pd.Series(
                [],
                index=pd.Index([], name="neighbor"),
                name="weight",
            )
        return self._adjacency.loc[item]

    def _get_ids_repr(self, chars=72):
        if len(self.unique_ids) > 5:
            ids = str(self.unique_ids[:5].tolist())[:-1] + ", "
            if len(ids) > chars:
                ids = str(self.unique_ids[:5].tolist())[:chars]
            return f"{ids}...]"
        else:
            return self.unique_ids.tolist()

    def __repr__(self):
        return (
            f"<Graph of {self.n} nodes and {self.nonzero} nonzero edges indexed by\n"
            f" {self._get_ids_repr()}>"
        )

    def copy(self, deep=True):
        """Make a copy of this Graph's adjacency table and transformation

        Parameters
        ----------
        deep : bool, optional
            Make a deep copy of the adjacency table, by default True

        Returns
        -------
        Graph
            libpysal.graph.Graph as a copy of the original
        """
        return Graph(
            self._adjacency.copy(deep=deep),
            transformation=self.transformation,
            is_sorted=True,
        )

    @cached_property
    def adjacency(self):
        """Return a copy of the adjacency list

        Returns
        -------
        pandas.Series
            Underlying adjacency list
        """
        return self._adjacency.copy()

    @classmethod
    def from_W(cls, w):  # noqa: N802
        """Create an experimental Graph from libpysal.weights.W object

        Parameters
        ----------
        w : libpysal.weights.W

        Returns
        -------
        Graph
            libpysal.graph.Graph from W

        Examples
        --------
        >>> import geopandas as gpd
        >>> from geodatasets import get_path
        >>> nybb = gpd.read_file(get_path("nybb")).set_index("BoroName")
        >>> nybb
                       BoroCode  ...                                           geometry
        BoroName                 ...
        Staten Island         5  ...  MULTIPOLYGON (((970217.022 145643.332, 970227....
        Queens                4  ...  MULTIPOLYGON (((1029606.077 156073.814, 102957...
        Brooklyn              3  ...  MULTIPOLYGON (((1021176.479 151374.797, 102100...
        Manhattan             1  ...  MULTIPOLYGON (((981219.056 188655.316, 980940....
        Bronx                 2  ...  MULTIPOLYGON (((1012821.806 229228.265, 101278...
        [5 rows x 4 columns]

        >>> queen_w = weights.Queen.from_dataframe(nybb, use_index=True)
        >>> queen_graph = graph.Graph.from_W(queen_w)
        >>> queen_graph
        <Graph of 5 nodes and 10 nonzero edges indexed by
         ['Staten Island', 'Queens', 'Brooklyn', 'Manhattan', 'Bronx']>
        """
        return cls.from_weights_dict(dict(w))

    def to_W(self):  # noqa: N802
        """Convert Graph to a libpysal.weights.W object

        Returns
        -------
        libpysal.weights.W
            representation of graph as a weights.W object

        Examples
        --------
        >>> import geopandas as gpd
        >>> from geodatasets import get_path
        >>> nybb = gpd.read_file(get_path("nybb")).set_index("BoroName")
        >>> nybb
                       BoroCode  ...                                           geometry
        BoroName                 ...
        Staten Island         5  ...  MULTIPOLYGON (((970217.022 145643.332, 970227....
        Queens                4  ...  MULTIPOLYGON (((1029606.077 156073.814, 102957...
        Brooklyn              3  ...  MULTIPOLYGON (((1021176.479 151374.797, 102100...
        Manhattan             1  ...  MULTIPOLYGON (((981219.056 188655.316, 980940....
        Bronx                 2  ...  MULTIPOLYGON (((1012821.806 229228.265, 101278...
        [5 rows x 4 columns]

        >>> contiguity = graph.Graph.build_contiguity(nybb)
        >>> contiguity.adjacency
        focal          neighbor
        Staten Island  Staten Island    0
        Queens         Brooklyn         1
                       Manhattan        1
                       Bronx            1
        Brooklyn       Queens           1
                       Manhattan        1
        Manhattan      Queens           1
                       Brooklyn         1
                       Bronx            1
        Bronx          Queens           1
                       Manhattan        1
        Name: weight, dtype: int64

        >>> w = contiguity.to_W()
        >>> w.neighbors
        {'Bronx': ['Queens', 'Manhattan'],
         'Brooklyn': ['Queens', 'Manhattan'],
         'Manhattan': ['Queens', 'Brooklyn', 'Bronx'],
         'Queens': ['Brooklyn', 'Manhattan', 'Bronx'],
         'Staten Island': []}
        """
        grouper = self._adjacency.groupby(level=0, sort=False)
        neighbors = {}
        weights = {}
        for ix, chunk in grouper:
            if ix in self.isolates:
                neighbors[ix] = []
                weights[ix] = []
            else:
                neighbors[ix] = chunk.index.get_level_values("neighbor").tolist()
                weights[ix] = chunk.tolist()

        return W(
            neighbors=neighbors,
            weights=weights,
            id_order=self.unique_ids.tolist(),
            silence_warnings=True,
        )

    @classmethod
    def from_adjacency(
        cls, adjacency, focal_col="focal", neighbor_col="neighbor", weight_col="weight"
    ):
        """Create a Graph from a pandas DataFrame formatted as an adjacency list

        Parameters
        ----------
        adjacency : pandas.DataFrame
            a dataframe formatted as an ajacency list. Should have columns
            "focal", "neighbor", and "weight", or columns that can be mapped
            to these (e.g. origin, destination, cost)
        focal : str, optional
            name of column holding focal/origin index, by default 'focal'
        neighbor : str, optional
            name of column holding neighbor/destination index, by default 'neighbor'
        weight : str, optional
            name of column holding weight values, by default 'weight'

        Returns
        -------
        Graph
            libpysal.graph.Graph
        """
        cols = dict(
            zip(
                [focal_col, neighbor_col, weight_col],
                ["focal_col", "neighbor_col", "weight_col"],
                strict=True,
            )
        )
        for col in cols:
            assert col in adjacency.columns.tolist(), (
                f'"{col}" was given for `{cols[col]}`, but the '
                f"columns available in `adjacency` are:  {adjacency.columns.tolist()}."
            )
        return cls.from_arrays(
            adjacency[focal_col].values,
            adjacency[neighbor_col].values,
            adjacency[weight_col].values,
        )

    @classmethod
    def from_sparse(cls, sparse, ids=None):
        """Convert a ``scipy.sparse`` array to a PySAL ``Graph`` object.

        Parameters
        ----------
        sparse : scipy.sparse array
            sparse representation of a graph
        ids : list-like, default None
            list-like of ids for geometries that is mappable to
            positions from sparse. If None, the positions are used as labels.

        Returns
        -------
        Graph
            libpysal.graph.Graph based on sparse
        """

        return cls(
            _sparse_to_arrays(sparse, ids, return_adjacency=True), is_sorted=True
        )

    @classmethod
    def from_arrays(cls, focal_ids, neighbor_ids, weight, **kwargs):
        """Generate Graph from arrays of indices and weights of the same length

        The arrays needs to be sorted in a way ensuring that focal_ids.unique() is
        equal to the index of original observations from which the Graph is being built

        Parameters
        ----------
        focal_index : array-like
            focal indices
        neighbor_index : array-like
            neighbor indices
        weight : array-like
            weights
        **kwargs
            keyword arguments passed to the class constructor

        Returns
        -------
        Graph
            libpysal.graph.Graph based on arrays
        """

        w = cls(
            pd.Series(
                weight,
                name="weight",
                index=pd.MultiIndex.from_arrays(
                    [focal_ids, neighbor_ids], names=["focal", "neighbor"]
                ),
            ),
            **kwargs,
        )

        return w

    @classmethod
    def from_weights_dict(cls, weights_dict):
        """Generate Graph from a dict of dicts

        Parameters
        ----------
        weights_dict : dictionary of dictionaries
            weights dictionary with the ``{focal: {neighbor: weight}}`` structure.

        Returns
        -------
        Graph
            libpysal.graph.Graph based on weights dictionary of dictionaries
        """
        idx = {f: list(neighbors) for f, neighbors in weights_dict.items()}
        data = {f: list(neighbors.values()) for f, neighbors in weights_dict.items()}
        return cls.from_dicts(idx, data)

    @classmethod
    def from_dicts(cls, neighbors, weights=None):
        """Generate Graph from dictionaries of neighbors and weights

        Parameters
        ----------
        neighbors : dict
            dictionary of neighbors with the ``{focal: [neighbor1, neighbor2]}``
            structure
        weights : dict, optional
            dictionary of neighbors with the ``{focal: [weight1, weight2]}``
            structure. If None, assumes binary weights.

        Returns
        -------
        Graph
            libpysal.graph.Graph based on dictionaries

        Examples
        --------
        >>> neighbors = {
        ...     'Africa': ['Asia'],
        ...     'Asia': ['Africa', 'Europe'],
        ...     'Australia': [],
        ...     'Europe': ['Asia'],
        ...     'North America': ['South America'],
        ...     'South America': ['North America'],
        ... }
        >>> connectivity = graph.Graph.from_dicts(neighbors)
        >>> connectivity.adjacency
        focal          neighbor
        Africa         Asia             1
        Asia           Africa           1
                       Europe           1
        Australia      Australia        0
        Europe         Asia             1
        North America  South America    1
        South America  North America    1
        Name: weight, dtype: float64

        You can also specify weights (for example based
        on the length of the shared border):

        >>> weights = {
        ...     'Africa': [1],
        ...     'Asia': [0.2, 0.8],
        ...     'Australia': [],
        ...     'Europe': [1],
        ...     'North America': [1],
        ...     'South America': [1],
        ... }
        >>> connectivity = graph.Graph.from_dicts(neighbors, weights)
        >>> connectivity.adjacency
        focal          neighbor
        Africa         Asia             1.0
        Asia           Africa           0.2
                       Europe           0.8
        Australia      Australia        0.0
        Europe         Asia             1.0
        North America  South America    1.0
        South America  North America    1.0
        Name: weight, dtype: float64
        """
        head, tail, weight = _neighbor_dict_to_edges(neighbors, weights=weights)
        return cls.from_arrays(head, tail, weight)

    @classmethod
    def build_block_contiguity(cls, regimes):
        """Generate Graph from block contiguity (regime neighbors)

        Block contiguity structures are relevant when defining neighbor relations
        based on membership in a regime. For example, all counties belonging to
        the same state could be defined as neighbors, in an analysis of all
        counties in the US.

        Parameters
        ----------
        regimes : list-like
            list-like of regimes. If pandas.Series, its index is used to encode Graph.
            Otherwise a default RangeIndex is used.

        Returns
        -------
        Graph
            libpysal.graph.Graph encoding block contiguity

        Examples
        --------
        >>> import geopandas as gpd
        >>> from geodatasets import get_path
        >>> france = gpd.read_file(get_path('geoda guerry')).set_index('Dprmnt')

        In the GeoDa Guerry dataset, the Region column reflects the region
        (North, East, West, South or Central) to which each department belongs.

        >>> france[['Region', 'geometry']].head()
                     Region                                           geometry
        Dprtmnt
        Ain               E  POLYGON ((801150.000 2092615.000, 800669.000 2...
        Aisne             N  POLYGON ((729326.000 2521619.000, 729320.000 2...
        Allier            C  POLYGON ((710830.000 2137350.000, 711746.000 2...
        Basses-Alpes      E  POLYGON ((882701.000 1920024.000, 882408.000 1...
        Hautes-Alpes      E  POLYGON ((886504.000 1922890.000, 885733.000 1...

        Using the ``"Region"`` labels as ``regimes`` then identifies all departments
        within the region as neighbors.

        >>> block_contiguity = graph.Graph.build_block_contiguity(france['Region'])
        >>> block_contiguity.adjacency
        focal   neighbor
        Ain     Basses-Alpes       1
                Hautes-Alpes       1
                Aube               1
                Cote-d'Or          1
                Doubs              1
                                  ..
        Vienne  Mayenne            1
                Morbihan           1
                Basses-Pyrenees    1
                Deux-Sevres        1
                Vendee             1
        Name: weight, Length: 1360, dtype: int32
        """
        ids = _evaluate_index(regimes)

        return cls.from_dicts(_block_contiguity(regimes, ids=ids))

    @classmethod
    def build_contiguity(cls, geometry, rook=True, by_perimeter=False, strict=False):
        """Generate Graph from geometry based on contiguity

        Contiguity builder assumes that all geometries are forming a coverage, i.e.
        a non-overlapping mesh and neighbouring geometries share only points or
        segments of their exterior boundaries. In practice, ``build_contiguity`` is
        capable of creating a Graph of partially overlapping geometries when
        ``strict=False, by_perimeter=False``, but that would not strictly follow the
        definition of queen or rook contiguity.

        Parameters
        ----------
        geometry : array-like of shapely.Geometry objects
            Could be geopandas.GeoSeries or geopandas.GeoDataFrame, in which case the
            resulting Graph is indexed by the original index. If an array of
            shapely.Geometry objects is passed, Graph will assume a RangeIndex.
        rook : bool, optional
            Contiguity method. If True, two geometries are considered neighbours if
            they share at least one edge. If False, two geometries are considered
            neighbours if they share at least one vertex. By default True
        by_perimeter : bool, optional
            If True, ``weight`` represents the length of the shared boundary between
            adjacent units, by default False. For row-standardized version of perimeter
            weights, use
            ``Graph.build_contiguity(gdf, by_perimeter=True).transform("r")``.
        strict : bool, optional
            Use the strict topological method. If False, the contiguity is determined
            based on shared coordinates or coordinate sequences representing edges.
            This assumes geometry coverage that is topologically correct. This method
            is faster but can miss some relations. If True, the contiguity is
            determined based on geometric relations that do not require precise
            topology. This method is slower but will result in correct contiguity
            even if the topology of geometries is not optimal. By default False.

        Returns
        -------
        Graph
            libpysal.graph.Graph encoding contiguity weights

        Examples
        --------

        >>> import geopandas as gpd
        >>> from geodatasets import get_path
        >>> nybb = gpd.read_file(get_path("nybb")).set_index("BoroName")
        >>> nybb
                       BoroCode  ...                                           geometry
        BoroName                 ...
        Staten Island         5  ...  MULTIPOLYGON (((970217.022 145643.332, 970227....
        Queens                4  ...  MULTIPOLYGON (((1029606.077 156073.814, 102957...
        Brooklyn              3  ...  MULTIPOLYGON (((1021176.479 151374.797, 102100...
        Manhattan             1  ...  MULTIPOLYGON (((981219.056 188655.316, 980940....
        Bronx                 2  ...  MULTIPOLYGON (((1012821.806 229228.265, 101278...
        [5 rows x 4 columns]

        >>> contiguity = graph.Graph.build_contiguity(nybb)
        >>> contiguity.adjacency
        focal          neighbor
        Staten Island  Staten Island    0
        Queens         Brooklyn         1
                       Manhattan        1
                       Bronx            1
        Brooklyn       Queens           1
                       Manhattan        1
        Manhattan      Queens           1
                       Brooklyn         1
                       Bronx            1
        Bronx          Queens           1
                       Manhattan        1
        Name: weight, dtype: int64

        Weight by perimeter instead of binary weights:

        >>> contiguity_perimeter = graph.Graph.build_contiguity(nybb, by_perimeter=True)
        >>> contiguity_perimeter.adjacency
        focal          neighbor
        Staten Island  Staten Island        0.000000
        Queens         Brooklyn         50867.502055
                       Manhattan          103.745207
                       Bronx                5.777002
        Brooklyn       Queens           50867.502055
                       Manhattan         5736.546898
        Manhattan      Queens             103.745207
                       Brooklyn          5736.546898
                       Bronx             5258.300879
        Bronx          Queens               5.777002
                       Manhattan         5258.300879
        Name: weight, dtype: float64
        """
        ids = _evaluate_index(geometry)

        if hasattr(geometry, "geometry"):
            # potentially cast GeoDataFrame to GeoSeries
            geometry = geometry.geometry

        if strict:
            # use shapely-based constructors
            if rook:
                return cls.from_arrays(
                    *_rook(geometry, ids=ids, by_perimeter=by_perimeter)
                )
            return cls.from_arrays(
                *_queen(geometry, ids=ids, by_perimeter=by_perimeter)
            )

        # use vertex-based constructor
        return cls.from_arrays(
            *_vertex_set_intersection(
                geometry, rook=rook, ids=ids, by_perimeter=by_perimeter
            )
        )

    @classmethod
    def build_distance_band(
        cls, data, threshold, binary=True, alpha=-1.0, kernel=None, bandwidth=None
    ):
        """Generate Graph from geometry based on a distance band

        Parameters
        ----------
        data : numpy.ndarray, geopandas.GeoSeries, geopandas.GeoDataFrame
            geometries containing locations to compute the
            delaunay triangulation. If a geopandas object with Point
            geometry is provided, the .geometry attribute is used. If a numpy.ndarray
            with shapely geometry is used, then the coordinates are extracted and used.
            If a numpy.ndarray of a shape (2,n) is used, it is assumed to contain x, y
            coordinates.
        threshold : float
            distance band
        binary : bool, optional
            If True :math:`w_{ij}=1` if :math:`d_{i,j}<=threshold`, otherwise
            :math:`w_{i,j}=0`.
            If False :math:`wij=dij^{alpha}`, by default True.
        alpha : float, optional
            distance decay parameter for weight (default -1.0)
            if alpha is positive the weights will not decline with
            distance. Ignored if ``binary=True`` or ``kernel`` is not None.
        kernel : str, optional
            kernel function to use in order to weight the output graph. See
            :meth:`Graph.build_kernel` for details. Ignored if ``binary=True``.
        bandwidth : float (default: None)
            distance to use in the kernel computation. Should be on the same scale as
            the input coordinates. Ignored if ``binary=True`` or ``kernel=None``.

        Returns
        -------
        Graph
            libpysal.graph.Graph encoding distance band weights

        Examples
        --------
        >>> import geopandas as gpd
        >>> from geodatasets import get_path
        >>> nybb = gpd.read_file(get_path("nybb")).set_index("BoroName")
        >>> nybb
                       BoroCode  ...                                           geometry
        BoroName                 ...
        Staten Island         5  ...  MULTIPOLYGON (((970217.022 145643.332, 970227....
        Queens                4  ...  MULTIPOLYGON (((1029606.077 156073.814, 102957...
        Brooklyn              3  ...  MULTIPOLYGON (((1021176.479 151374.797, 102100...
        Manhattan             1  ...  MULTIPOLYGON (((981219.056 188655.316, 980940....
        Bronx                 2  ...  MULTIPOLYGON (((1012821.806 229228.265, 101278...
        [5 rows x 4 columns]

        Note that the method requires point geometry (or an array of coordinates
        representing points) as an input.

        The threshold distance is in the units of the geometry projection.
        You can check it using the ``nybb.crs`` property.

        >>> distance_band = graph.Graph.build_distance_band(nybb.centroid, 45000)
        >>> distance_band.adjacency
        focal          neighbor
        Staten Island  Staten Island    0
        Queens         Brooklyn         1
        Brooklyn       Queens           1
        Manhattan      Bronx            1
        Bronx          Manhattan        1
        Name: weight, dtype: int64

        The larger threshold yields more neighbors.

        >>> distance_band = graph.Graph.build_distance_band(nybb.centroid, 110000)
        >>> distance_band.adjacency
        focal          neighbor
        Staten Island  Queens           1
                       Brooklyn         1
                       Manhattan        1
        Queens         Staten Island    1
                       Brooklyn         1
                       Manhattan        1
                       Bronx            1
        Brooklyn       Staten Island    1
                       Queens           1
                       Manhattan        1
                       Bronx            1
        Manhattan      Staten Island    1
                       Queens           1
                       Brooklyn         1
                       Bronx            1
        Bronx          Queens           1
                       Brooklyn         1
                       Manhattan        1
        Name: weight, dtype: int64

        Instead of binary weights you can use inverse distance.

        >>> distance_band = graph.Graph.build_distance_band(
        ...     nybb.centroid,
        ...     45000,
        ...     binary=False,
        ... )
        >>> distance_band.adjacency
        focal          neighbor
        Staten Island  Staten Island    0.000000
        Queens         Brooklyn         0.000024
        Brooklyn       Queens           0.000024
        Manhattan      Bronx            0.000026
        Bronx          Manhattan        0.000026
        Name: weight, dtype: float64

        Or specify the kernel function to derive weight from the distance.

        >>> distance_band = graph.Graph.build_distance_band(
        ...     nybb.centroid,
        ...     45000,
        ...     binary=False,
        ...     kernel='bisquare',
        ...     bandwidth=60000,
        ... )
        >>> distance_band.adjacency
        focal          neighbor
        Staten Island  Staten Island    0.000000
        Queens         Brooklyn         0.232079
        Brooklyn       Queens           0.232079
        Manhattan      Bronx            0.309825
        Bronx          Manhattan        0.309825
        Name: weight, dtype: float64
        """
        ids = _evaluate_index(data)

        dist = _distance_band(data, threshold)

        if binary:
            head, tail, weight = _kernel(
                dist,
                kernel="boxcar",
                metric="precomputed",
                ids=ids,
                bandwidth=np.inf,
            )
        elif kernel is not None:
            head, tail, weight = _kernel(
                dist,
                kernel=kernel,
                metric="precomputed",
                ids=ids,
                bandwidth=bandwidth,
            )
        else:
            head, tail, weight = _kernel(
                dist,
                kernel=lambda distances, alpha: np.power(distances, alpha),
                metric="precomputed",
                ids=ids,
                bandwidth=alpha,
            )

        adjacency = pd.DataFrame.from_dict(
            {"focal": head, "neighbor": tail, "weight": weight}
        ).set_index("focal")

        # drop diagonal
        counts = adjacency.index.value_counts()
        no_isolates = counts[counts > 1]
        adjacency = adjacency[
            ~(
                adjacency.index.isin(no_isolates.index)
                & (adjacency.index == adjacency.neighbor)
            )
        ]
        # set isolates to 0 - distance band should never contain self-weight
        adjacency.loc[~adjacency.index.isin(no_isolates.index), "weight"] = 0

        return cls.from_arrays(
            adjacency.index.values, adjacency.neighbor.values, adjacency.weight.values
        )

    @classmethod
    def build_fuzzy_contiguity(
        cls, geometry, tolerance=None, buffer=None, predicate="intersects", **kwargs
    ):
        """Generate Graph from fuzzy contiguity

        Fuzzy contiguity relaxes the notion of contiguity neighbors
        for the case of geometry collections that violate the condition
        of planar enforcement. It handles three types of conditions present
        in such collections that would result in missing links when using
        the regular contiguity methods.

        The first are edges for nearby polygons that should be shared, but are
        digitized separately for the individual polygons and the resulting edges
        do not coincide, but instead the edges intersect. This case can also be
        covered by ``build_contiguty`` with the ``strict=False`` parameter.

        The second case is similar to the first, only the resultant edges do not
        intersect but are "close". The optional buffering of geometry then closes the
        gaps between the polygons and a resulting intersection is encoded as a link.

        The final case arises when one polygon is "inside" a second polygon but is not
        encoded to represent a hole in the containing polygon.

        It is also possible to create a contiguity based on a custom spatial predicate.

        Parameters
        ----------
        geoms :  array-like of shapely.Geometry objects
            Could be geopandas.GeoSeries or geopandas.GeoDataFrame, in which case the
            resulting Graph is indexed by the original index. If an array of
            shapely.Geometry objects is passed, Graph will assume a RangeIndex.
        tolerance : float, optional
            The percentage of the length of the minimum side of the bounding rectangle
            for the ``geoms`` to use in determining the buffering distance. Either
            ``tolerance`` or ``buffer`` may be specified but not both.
            By default None.
        buffer : float, optional
            Exact buffering distance in the units of ``geoms.crs``. Either
            ``tolerance`` or ``buffer`` may be specified but not both.
            By default None.
        predicate : str, optional
            The predicate to use for determination of neighbors. Default is
            'intersects'. If None is passed, neighbours are determined based
            on the intersection of bounding boxes. See the documentation of
            ``geopandas.GeoSeries.sindex.query`` for allowed predicates.
        **kwargs
            Keyword arguments passed to ``geopandas.GeoSeries.buffer``.

        Returns
        -------
        Graph
            libpysal.graph.Graph encoding fuzzy contiguity


        Examples
        --------
        >>> import geopandas as gpd
        >>> from geodatasets import get_path
        >>> nybb = gpd.read_file(get_path("nybb")).set_index("BoroName")
        >>> nybb
                       BoroCode  ...                                           geometry
        BoroName                 ...
        Staten Island         5  ...  MULTIPOLYGON (((970217.022 145643.332, 970227....
        Queens                4  ...  MULTIPOLYGON (((1029606.077 156073.814, 102957...
        Brooklyn              3  ...  MULTIPOLYGON (((1021176.479 151374.797, 102100...
        Manhattan             1  ...  MULTIPOLYGON (((981219.056 188655.316, 980940....
        Bronx                 2  ...  MULTIPOLYGON (((1012821.806 229228.265, 101278...
        [5 rows x 4 columns]

        Example using the default parameters:

        >>> fuzzy_contiguity = graph.Graph.build_fuzzy_contiguity(nybb)
        >>> fuzzy_contiguity
        <Graph of 5 nodes and 10 nonzero edges indexed by
         ['Staten Island', 'Queens', 'Brooklyn', 'Manhattan', 'Bronx']>

        Example using the tolerance of 0.05:

        >>> fuzzy_contiguity = graph.Graph.build_fuzzy_contiguity(nybb, tolerance=0.05)
        >>> fuzzy_contiguity
        <Graph of 5 nodes and 12 nonzero edges indexed by
         ['Staten Island', 'Queens', 'Brooklyn', 'Manhattan', 'Bronx']>

        Example using a buffer of 10000 feet (CRS of nybb is in feet):

        >>> fuzzy_contiguity = graph.Graph.build_fuzzy_contiguity(nybb, buffer=10000)
        >>> fuzzy_contiguity
        <Graph of 5 nodes and 14 nonzero edges indexed by
         ['Staten Island', 'Queens', 'Brooklyn', 'Manhattan', 'Bronx']>

        """
        ids = _evaluate_index(geometry)

        heads, tails, weights = _fuzzy_contiguity(
            geometry,
            ids,
            tolerance=tolerance,
            buffer=buffer,
            predicate=predicate,
            **kwargs,
        )

        return cls.from_arrays(heads, tails, weights)

    @classmethod
    def build_raster_contiguity(
        cls,
        da,
        rook=False,
        z_value=None,
        coords_labels=None,
        k=1,
        include_nodata=False,
        n_jobs=1,
    ):
        """Generate Graph from ``xarray.DataArray`` raster object

        Create Graph object encoding contiguity of raster cells from
        ``xarray.DataArray`` object. The coordinates are flatten to tuples representing
        the location of each cell within the raster.

        Parameters
        ----------
        da : xarray.DataArray
            Input 2D or 3D DataArray with shape=(z, y, x)
        rook : bool, optional
            Contiguity method. If True, two cells are considered neighbours if
            they share at least one edge. If False, two geometries are considered
            neighbours if they share at least one vertex. By default True
        z_value : {int, str, float}, optional
            Select the z_value of 3D DataArray with multiple layers. By default None
        coords_labels : dict, optional
            Pass dimension labels for coordinates and layers if they do not
            belong to default dimensions, which are (band/time, y/lat, x/lon)
            e.g. ``coords_labels = {"y_label": "latitude", "x_label": "longitude",
            "z_label": "year"}``
            When None, defaults to empty dictionary.
        k : int, optional
            Order of contiguity, this will select all neighbors up to k-th order.
            Default is 1.
        include_nodata : bool, optional
            If True, missing values will be assumed as non-missing when
            selecting higher_order neighbors, Default is False
        n_jobs : int, optional
            Number of cores to be used in the sparse weight construction. If -1,
            all available cores are used. Default is 1. Requires ``joblib``.

        Returns
        -------
        Graph
            libpysal.graph.Graph encoding raster contiguity
        """

        if coords_labels is None:
            coords_labels = {}
        criterion = "rook" if rook else "queen"

        heads, tails, weights, xarray_index = _raster_contiguity(
            da=da,
            criterion=criterion,
            z_value=z_value,
            coords_labels=coords_labels,
            k=k,
            include_nodata=include_nodata,
            n_jobs=n_jobs,
        )
        heads, tails, weights = _resolve_islands(
            heads, tails, xarray_index.to_numpy(), weights
        )
        contig = cls.from_arrays(heads, tails, weights)
        contig._xarray_index_names = xarray_index.names

        if k > 1 and not include_nodata:
            contig = contig.higher_order(k, lower_order=True)

        return contig

    @classmethod
    def build_kernel(
        cls,
        data,
        kernel="gaussian",
        k=None,
        bandwidth=None,
        metric="euclidean",
        p=2,
        coplanar="raise",
    ):
        """Generate Graph from geometry data based on a kernel function

        Parameters
        ----------
        data : numpy.ndarray, geopandas.GeoSeries, geopandas.GeoDataFrame
            geometries over which to compute a kernel. If a geopandas object with Point
            geoemtry is provided, the .geometry attribute is used. If a numpy.ndarray
            with shapely geoemtry is used, then the coordinates are extracted and used.
            If a numpy.ndarray of a shape (2,n) is used, it is assumed to contain x, y
            coordinates. If metric="precomputed", data is assumed to contain a
            precomputed distance metric.
        kernel : string or callable (default: 'gaussian')
            kernel function to apply over the distance matrix computed by `metric`.
            The following kernels are supported:

            - ``"triangular"``:
            - ``"parabolic"``:
            - ``"gaussian"``:
            - ``"bisquare"``:
            - ``"cosine"``:
            - ``'boxcar'``/discrete: all distances less than `bandwidth` are 1, and all
              other distances are 0
            - ``"identity"``/None : do nothing, weight similarity based on raw distance
            - ``callable`` : a user-defined function that takes the distance vector and
              the bandwidth and returns the kernel: kernel(distances, bandwidth)

        k : int (default: None)
            number of nearest neighbors used to truncate the kernel. This is assumed
            to be constant across samples. If None, no truncation is conduted.
        bandwidth : float (default: None)
            distance to use in the kernel computation. Should be on the same scale as
            the input coordinates.
        metric : string or callable (default: 'euclidean')
            distance function to apply over the input coordinates. Supported options
            depend on whether or not scikit-learn is installed. If so, then any
            distance function supported by scikit-learn is supported here. Otherwise,
            only euclidean, minkowski, and manhattan/cityblock distances are admitted.
        p : int (default: 2)
            parameter for minkowski metric, ignored if metric != "minkowski".
        coplanar: str, optional (default "raise")
            Method for handling coplanar points when ``k`` is not None. Options are
            ``'raise'`` (raising an exception when coplanar points are present),
            ``'jitter'`` (randomly displace coplanar points to produce uniqueness), &
            ``'clique'`` (induce fully-connected sub cliques for coplanar points).

        Returns
        -------
        Graph
            libpysal.graph.Graph encoding kernel weights
        """
        ids = _evaluate_index(data)

        head, tail, weight = _kernel(
            data,
            bandwidth=bandwidth,
            metric=metric,
            kernel=kernel,
            k=k,
            p=p,
            ids=ids,
            coplanar=coplanar,
        )

        return cls.from_arrays(head, tail, weight)

    @classmethod
    def build_knn(cls, data, k, metric="euclidean", p=2, coplanar="raise"):
        """Generate Graph from geometry data based on k-nearest neighbors search

        Parameters
        ----------
        data : numpy.ndarray, geopandas.GeoSeries, geopandas.GeoDataFrame
            geometries over which to compute a kernel. If a geopandas object with Point
            geoemtry is provided, the .geometry attribute is used. If a numpy.ndarray
            with shapely geoemtry is used, then the coordinates are extracted and used.
            If a numpy.ndarray of a shape (2,n) is used, it is assumed to contain x, y
            coordinates.
        k : int
            number of nearest neighbors.
        metric : string or callable (default: 'euclidean')
            distance function to apply over the input coordinates. Supported options
            depend on whether or not scikit-learn is installed. If so, then any
            distance function supported by scikit-learn is supported here. Otherwise,
            only euclidean, minkowski, and manhattan/cityblock distances are admitted.
        p : int (default: 2)
            parameter for minkowski metric, ignored if metric != "minkowski".
        coplanar: str, optional (default "raise")
            Method for handling coplanar points. Options include
            ``'raise'`` (raising an exception when coplanar points are present),
            ``'jitter'`` (randomly displace coplanar points to produce uniqueness), &
            ``'clique'`` (induce fully-connected sub cliques for coplanar points).


        Returns
        -------
        Graph
            libpysal.graph.Graph encoding KNN weights

        Examples
        --------
        >>> import geopandas as gpd
        >>> from geodatasets import get_path
        >>> nybb = gpd.read_file(get_path("nybb")).set_index('BoroName')
        >>> nybb
                       BoroCode  ...                                           geometry
        BoroName                 ...
        Staten Island         5  ...  MULTIPOLYGON (((970217.022 145643.332, 970227....
        Queens                4  ...  MULTIPOLYGON (((1029606.077 156073.814, 102957...
        Brooklyn              3  ...  MULTIPOLYGON (((1021176.479 151374.797, 102100...
        Manhattan             1  ...  MULTIPOLYGON (((981219.056 188655.316, 980940....
        Bronx                 2  ...  MULTIPOLYGON (((1012821.806 229228.265, 101278...

        >>> knn3 = graph.Graph.build_knn(nybb.centroid, k=3)
        >>> knn3.adjacency
        focal           neighbor
        Staten Island   Queens           1
                        Brooklyn         1
                        Manhattan        1
        Queens          Brooklyn         1
                        Manhattan        1
                        Bronx            1
        Brooklyn        Staten Island    1
                        Queens           1
                        Manhattan        1
        Manhattan       Queens           1
                        Brooklyn         1
                        Bronx            1
        Bronx           Queens           1
                        Brooklyn         1
                        Manhattan        1
        Name: weight, dtype: int32

        Specifying k=1 identifies the nearest neighbor
        (note that this can be asymmetrical):

        >>> knn1 = graph.Graph.build_knn(nybb.centroid, k=1)
        >>> knn1.adjacency
        focal          neighbor
        Staten Island  Brooklyn     1
        Queens         Brooklyn     1
        Brooklyn       Queens       1
        Manhattan      Bronx        1
        Bronx          Manhattan    1
        Name: weight, dtype: int32
        """
        ids = _evaluate_index(data)

        head, tail, weight = _kernel(
            data,
            bandwidth=np.inf,
            metric=metric,
            kernel="boxcar",
            k=k,
            p=p,
            ids=ids,
            coplanar=coplanar,
        )

        return cls.from_arrays(head, tail, weight)

    @classmethod
    def build_spatial_matches(
        cls,
        data,
        k,
        metric="euclidean",
        solver=None,
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
            geometries that are used as a source for matching. If a geopandas object
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
            provided, pulp's default solver will be used. This is generally
            pulp.COIN(), but this may vary depending on your configuration.
        return_mip : bool (default: False)
            whether or not to return the instance of the pulp.LpProblem. By
            default, the problem is not returned to the user.
        allow_partial_match : bool (default: False)
            whether to allow for partial matching. A partial match may have
            a weight between zero and one, while a "full" match (by default)
            must have a weight of either zero or one. A partial matching may
            have a shorter total distance, but will result in a weighted
            graph.
        """
        head, tail, weight = _spatial_matching(
            x=data,
            metric=metric,
            n_matches=k,
            solver=solver,
            allow_partial_match=allow_partial_match,
            **metric_kwargs,
        )
        # ids need to be addressed here, rather than in the matching
        # because x and y can have different id sets. It's only
        # in W where we *know* we can just use one id vector.
        return cls.from_arrays(head, tail, weight)

    @classmethod
    def build_triangulation(
        cls,
        data,
        method="delaunay",
        bandwidth=np.inf,
        kernel="boxcar",
        clip="bounding_box",
        rook=True,
        coplanar="raise",
    ):
        """Generate Graph from geometry based on triangulation

        Parameters
        ----------
        data : numpy.ndarray, geopandas.GeoSeries, geopandas.GeoDataFrame
            geometries containing locations to compute the
            delaunay triangulation. If a geopandas object with Point
            geoemtry is provided, the .geometry attribute is used. If a numpy.ndarray
            with shapely geoemtry is used, then the coordinates are extracted and used.
            If a numpy.ndarray of a shape (2,n) is used, it is assumed to contain x, y
            coordinates.
        method : str, (default "delaunay")
            method of extracting the weights from triangulation. Supports:

            - ``"delaunay"``
            - ``"gabriel"``
            - ``"relative_neighborhood"``
            - ``"voronoi"``

        bandwidth : float, optional
            distance to use in the kernel computation. Should be on the same scale as
            the input coordinates, by default numpy.inf
        kernel : str, optional
            kernel function to use in order to weight the output graph. See
            :meth:`Graph.build_kernel` for details. By default "boxcar"
        clip : str (default: 'bbox')
            Clipping method when ``method="voronoi"``. Ignored otherwise.
            Default is ``'bounding_box'``. Options are as follows.

            ``None``
                No clip is applied. Voronoi cells may be arbitrarily larger that the
                source map. Note that this may lead to cells that are many orders of
                magnitude larger in extent than the original map. Not recommended.
            ``'bounding_box'``
                Clip the voronoi cells to the bounding box of the input points.
            ``'convex_hull'``
                Clip the voronoi cells to the convex hull of the input points.
            ``'alpha_shape'``
                Clip the voronoi cells to the tightest hull that contains all points
                (e.g. the smallest alpha shape, using
                :func:`libpysal.cg.alpha_shape_auto`).
            ``shapely.Polygon``
                Clip to an arbitrary Polygon.

        rook : bool, optional
            Contiguity method when ``method="voronoi"``. Ignored otherwise.
            If True, two geometries are considered neighbours if they
            share at least one edge. If False, two geometries are considered neighbours
            if they share at least one vertex. By default True
        coplanar: str, optional (default "raise")
            Method for handling coplanar points. Options include
            ``'raise'`` (raising an exception when coplanar points are present),
            ``'jitter'`` (randomly displace coplanar points to produce uniqueness), &
            ``'clique'`` (induce fully-connected sub cliques for coplanar points).

        Returns
        -------
        Graph
            libpysal.graph.Graph encoding triangulation weights

        Examples
        --------

        >>> import geopandas as gpd
        >>> from geodatasets import get_path
        >>> nybb = gpd.read_file(get_path("nybb")).set_index("BoroName")
        >>> nybb
                       BoroCode  ...                                           geometry
        BoroName                 ...
        Staten Island         5  ...  MULTIPOLYGON (((970217.022 145643.332, 970227....
        Queens                4  ...  MULTIPOLYGON (((1029606.077 156073.814, 102957...
        Brooklyn              3  ...  MULTIPOLYGON (((1021176.479 151374.797, 102100...
        Manhattan             1  ...  MULTIPOLYGON (((981219.056 188655.316, 980940....
        Bronx                 2  ...  MULTIPOLYGON (((1012821.806 229228.265, 101278...
        [5 rows x 4 columns]

        Note that the method requires point geometry (or an array of coordinates
        representing points) as an input.

        >>> triangulation = graph.Graph.build_triangulation(nybb.centroid)
        >>> triangulation.adjacency
        focal          neighbor
        Staten Island  Brooklyn         1
                       Manhattan        1
        Queens         Brooklyn         1
                       Manhattan        1
                       Bronx            1
        Brooklyn       Staten Island    1
                       Queens           1
                       Manhattan        1
        Manhattan      Staten Island    1
                       Queens           1
                       Brooklyn         1
                       Bronx            1
        Bronx          Queens           1
                       Manhattan        1
        Name: weight, dtype: int64
        """
        ids = _evaluate_index(data)

        if method == "delaunay":
            head, tail, weights = _delaunay(
                data, ids=ids, bandwidth=bandwidth, kernel=kernel, coplanar=coplanar
            )
        elif method == "gabriel":
            head, tail, weights = _gabriel(
                data, ids=ids, bandwidth=bandwidth, kernel=kernel, coplanar=coplanar
            )
        elif method == "relative_neighborhood":
            head, tail, weights = _relative_neighborhood(
                data, ids=ids, bandwidth=bandwidth, kernel=kernel, coplanar=coplanar
            )
        elif method == "voronoi":
            head, tail, weights = _voronoi(
                data, ids=ids, clip=clip, rook=rook, coplanar=coplanar
            )
        else:
            raise ValueError(
                f"Method '{method}' is not supported. Use one of ['delaunay', "
                "'gabriel', 'relative_neighborhood', 'voronoi']."
            )

        return cls.from_arrays(head, tail, weights)

    @classmethod
    def build_h3(cls, ids, order=1, weight="distance"):
        """Generate Graph from indices of H3 hexagons.

        Encode a graph from a set of H3 hexagons. The graph is generated by
        considering the H3 hexagons as nodes and connecting them based on their
        contiguity. The contiguity is defined by the order parameter, which
        specifies the number of steps to consider as neighbors. The weight
        parameter defines the type of weight to assign to the edges.

        Requires the `h3` library.

        Parameters
        ----------
        ids : array-like
            Array of H3 IDs encoding focal geometries
        order : int, optional
            Order of contiguity, by default 1
        weight : str, optional
            Type of weight. Options are:

            * ``distance``: raw topological distance between cells
            * ``binary``: 1 for neighbors, 0 for non-neighbors
            * ``inverse``: 1 / distance between cells

            By default "distance".

        Returns
        -------
        Graph

        Examples
        --------
        >>> import geopandas as gpd
        >>> from geodatasets import get_path
        >>> from tobler.util import h3fy
        >>> gdf = gpd.read_file(get_path("geoda guerry"))
        >>> h3 = h3fy(gdf, resolution=4)
        >>> h3.head()
                                                                  geometry
        hex_id
        841f94dffffffff  POLYGON ((609346.657 2195981.397, 604556.817 2...
        841fa67ffffffff  POLYGON ((722074.162 2561038.244, 717442.706 2...
        84186a3ffffffff  POLYGON ((353695.287 2121176.341, 329999.974 2...
        8418609ffffffff  POLYGON ((387747.482 2509794.492, 364375.032 2...
        8418491ffffffff  POLYGON ((320872.289 1846157.662, 296923.464 1...

        >>> h3_contiguity = graph.Graph.build_h3(h3.index)
        >>> h3_contiguity
        <Graph of 320 nodes and 1740 nonzero edges indexed by
         ['841f94dffffffff', '841fa67ffffffff', '84186a3ffffffff', ...]>
        """
        neighbors, weights = _build_from_h3(ids, order=order)
        g = cls.from_dicts(neighbors, weights)

        if weight == "distance":
            return g
        elif weight == "binary":
            return g.transform("b")
        elif weight == "inverse":
            return cls(1 / g._adjacency, is_sorted=True)
        else:
            raise ValueError("weight must be one of 'distance', 'binary', or 'inverse'")

    @classmethod
    def build_travel_cost(
        cls, df, network, threshold, kernel=None, mapping_distance=None
    ):
        """Generate a Graph based on shortest travel costs from a pandana.Network

        Parameters
        ----------
        df : geopandas.GeoDataFrame
            geodataframe representing observations which are snapped to the nearest
            node in the pandana.Network. CRS should be the same as the locations
            of ``node_x`` and ``node_y`` in the pandana.Network (usually 4326 if network
            comes from OSM, but sometimes projected to improve snapping quality).
        network : pandana.Network
            pandana Network object describing travel costs between nodes in the study
            area.  See <https://udst.github.io/pandana/> for more
        threshold : int
            threshold representing maximum cost distances. This is measured in the same
            units as the pandana.Network (not influenced by the df.crs in any way). For
            travel modes with relatively constant speeds like walking or biking, this is
            usually distance (e.g. meters if the Network is constructed from OSM). For a
            a multimodal or auto network with variable travel speeds, this is usually
            some measure of travel time
        kernel : str or callable, optional
            kernel transformation applied to the weights. See
            libpysal.graph.Graph.build_kernel for more information on kernel
            transformation options. Default is None, in which case the Graph weight
            is pure distance between focal and neighbor
        mapping_distance : int
            snapping tolerance passed to ``pandana.Network.get_node_ids`` that defines
            the maximum range at which observations are snapped to nearest nodes in the
            network. Default is None

        Returns
        -------
        Graph

        Examples
        ---------
        >>> import geodatasets
        >>> import geopandas as gpd
        >>> import osmnx as ox
        >>> import pandana as pdna

        Read an example geodataframe:

        >>> df = gpd.read_file(geodatasets.get_path("geoda Cincinnati")).to_crs(4326)

        Download a walk network using osmnx

        >>> osm_graph = ox.graph_from_polygon(df.union_all(), network_type="walk")
        >>> nodes, edges = ox.utils_graph.graph_to_gdfs(osm_graph)
        >>> edges = edges.reset_index()

        Generate a routable pandana network from the OSM nodes and edges

        >>> network = pdna.Network(
        >>>     edge_from=edges["u"],
        >>>     edge_to=edges["v"],
        >>>     edge_weights=edges[["length"]],
        >>>     node_x=nodes["x"],
        >>>     node_y=nodes["y"],)

        Use the pandana network to compute shortest paths between gdf centroids and
        generate a Graph

        >>> G = Graph.build_travel_cost(df.set_geometry(df.centroid), network, 500)
        >>> G.adjacency.head()
        focal  neighbor
        0       62          385.609009
                65          309.471985
                115         346.858002
                116           0.000000
                117         333.639008
        Name: weight, dtype: float64
        """
        adj = _build_travel_graph(df, network, threshold, mapping_distance)
        g = cls.from_adjacency(adj)
        if kernel is not None:
            arrays = _kernel(
                g.sparse,
                metric="precomputed",
                kernel=kernel,
                bandwidth=threshold,
                resolve_isolates=False,
                ids=df.index.values,
            )
            return cls.from_arrays(*arrays)
        return g

    @cached_property
    def neighbors(self):
        """Get neighbors dictionary

        Notes
        -----
        It is recommended to work directly with :meth:`Graph.adjacency` rather than
        using the :meth:`Graph.neighbors`.

        Returns
        -------
        dict
            dict of tuples representing neighbors
        """
        grouper = self._adjacency.groupby(level=0, sort=False)
        neighbors = {}
        for ix, chunk in grouper:
            if ix in self.isolates:
                neighbors[ix] = ()
            else:
                neighbors[ix] = tuple(chunk.index.get_level_values("neighbor"))
        return neighbors

    @cached_property
    def weights(self):
        """Get weights dictionary

        Notes
        -----
        It is recommended to work directly with :meth:`Graph.adjacency` rather than
        using the :meth:`Graph.weights`.

        Returns
        -------
        dict
            dict of tuples representing weights
        """
        grouper = self._adjacency.groupby(level=0, sort=False)
        weights = {}
        for ix, chunk in grouper:
            if ix in self.isolates:
                weights[ix] = ()
            else:
                weights[ix] = tuple(chunk)
        return weights

    @cached_property
    def sparse(self):
        """Return a scipy.sparse array (CSR)

        Returns
        -------
        scipy.sparse.CSR
            sparse representation of the adjacency
        """
        # pivot to COO sparse matrix and cast to sparse CRS array
        return sparse.csr_array(
            self._adjacency.astype("Sparse[float]").sparse.to_coo(sort_labels=True)[0]
        )

    def transform(self, transformation):
        """Transformation of weights

        Parameters
        ----------
        transformation : str | callable
            Transformation method. The following are
            valid transformations.

            - **B** -- Binary
            - **R** -- Row-standardization (global sum :math:`=n`)
            - **D** -- Double-standardization (global sum :math:`=1`)
            - **V** -- Variance stabilizing

            Alternatively, you can pass your own callable passed to
            ``self.adjacency.groupby(level=0).transform()``.

        Returns
        -------
        Graph
            transformed weights

        Raises
        ------
        ValueError
            Value error for unsupported transformation
        """
        if isinstance(transformation, str):
            transformation = transformation.upper()

        if self.transformation == transformation:
            return self.copy()

        if transformation == "R":
            standardized = (
                (
                    self._adjacency
                    / self._adjacency.groupby(level=0, sort=False).transform("sum")
                )
                .fillna(0)
                .values
            )  # isolate comes as NaN -> 0

        elif transformation == "D":
            standardized = (self._adjacency / self._adjacency.sum()).values

        elif transformation == "B":
            standardized = self._adjacency.astype(bool).astype(int)

        elif transformation == "V":
            s = self._adjacency.groupby(level=0, sort=False).transform(
                lambda group: group / math.sqrt((group**2).sum())
            )
            n_q = self.n / s.sum()
            standardized = (s * n_q).fillna(0).values  # isolate comes as NaN -> 0

        elif callable(transformation):
            standardized = self._adjacency.groupby(level=0, sort=False).transform(
                transformation
            )
            transformation = "C"

        else:
            raise ValueError(
                f"Transformation '{transformation}' is not supported. "
                f"Use one of {ALLOWED_TRANSFORMATIONS[1:]} or pass a callable."
            )

        standardized_adjacency = pd.Series(
            standardized, name="weight", index=self._adjacency.index
        )
        transformed = Graph(standardized_adjacency, transformation, is_sorted=True)

        if hasattr(self, "_xarray_index_names"):
            transformed._xarray_index_names = self._xarray_index_names
        return transformed

    @cached_property
    def _components(self):
        """helper for n_components and component_labels"""
        return sparse.csgraph.connected_components(self.sparse)

    @cached_property
    def n_components(self):
        """Get a number of connected components

        Returns
        -------
        int
            number of components
        """
        return self._components[0]

    @cached_property
    def component_labels(self):
        """Get component labels per observation

        Returns
        -------
        numpy.array
            Array of component labels
        """
        return pd.Series(
            self._components[1], index=self.unique_ids, name="component labels"
        )

    @cached_property
    def cardinalities(self):
        """Number of neighbors for each observation

        Returns
        -------
        pandas.Series
            Series with a number of neighbors per each observation
        """
        cardinalities = self._adjacency.astype(bool).groupby(level=0, sort=False).sum()
        cardinalities.name = "cardinalities"
        return cardinalities

    @cached_property
    def isolates(self):
        """Index of observations with no neighbors

        Isolates are encoded as a self-loop with
        the weight == 0 in the adjacency table.

        Returns
        -------
        pandas.Index
            Index with a subset of observations that do not have any neighbor
        """
        return self.cardinalities.index[self.cardinalities == 0]

    @cached_property
    def unique_ids(self):
        """Unique IDs used in the Graph"""
        return self._adjacency.index.get_level_values("focal").unique()

    @cached_property
    def n(self):
        """Number of observations."""
        return self.unique_ids.shape[0]

    @cached_property
    def n_nodes(self):
        """Number of nodes."""
        return self.unique_ids.shape[0]

    @cached_property
    def n_edges(self):
        """Number of edges."""
        return self._adjacency.shape[0] - self.isolates.shape[0]

    @cached_property
    def pct_nonzero(self):
        """Percentage of nonzero weights."""
        p = 100.0 * self.sparse.nnz / (1.0 * self.n**2)
        return p

    @cached_property
    def nonzero(self):
        """Number of nonzero weights."""
        return (self._adjacency > 0).sum()

    @cached_property
    def index_pairs(self):
        """Return focal-neighbor index pairs

        Returns
        -------
        tuple(Index, Index)
            tuple of two aligned pandas.Index objects encoding all edges of the Graph
            by their nodes

        Examples
        --------
        >>> import geopandas as gpd
        >>> from geodatasets import get_path
        >>> nybb = gpd.read_file(get_path("nybb")).set_index("BoroName")
        >>> nybb
                       BoroCode  ...                                           geometry
        BoroName                 ...
        Staten Island         5  ...  MULTIPOLYGON (((970217.022 145643.332, 970227....
        Queens                4  ...  MULTIPOLYGON (((1029606.077 156073.814, 102957...
        Brooklyn              3  ...  MULTIPOLYGON (((1021176.479 151374.797, 102100...
        Manhattan             1  ...  MULTIPOLYGON (((981219.056 188655.316, 980940....
        Bronx                 2  ...  MULTIPOLYGON (((1012821.806 229228.265, 101278...
        [5 rows x 4 columns]

        >>> contiguity = graph.Graph.build_contiguity(nybb)
        >>> focal, neighbor = contiguity.index_pairs
        >>> focal
        Index(['Staten Island', 'Queens', 'Queens', 'Queens', 'Brooklyn', 'Brooklyn',
               'Manhattan', 'Manhattan', 'Manhattan', 'Bronx', 'Bronx'],
              dtype='object', name='focal')

        >>> neighbor
        Index(['Staten Island', 'Brooklyn', 'Manhattan', 'Bronx', 'Queens',
               'Manhattan', 'Queens', 'Brooklyn', 'Bronx', 'Queens', 'Manhattan'],
              dtype='object', name='neighbor')
        """
        focal = self._adjacency.index.get_level_values("focal")
        neighbor = self._adjacency.index.get_level_values("neighbor")
        return (focal, neighbor)

    def asymmetry(self, intrinsic=True):
        r"""Asymmetry check.

        Parameters
        ----------
        intrinsic : bool, optional
            Default is ``True``. Intrinsic symmetry is defined as:

            .. math::

                w_{i,j} == w_{j,i}

            If ``intrinsic`` is ``False`` symmetry is defined as:

            .. math::

                i \in N_j \ \& \ j \in N_i

            where :math:`N_j` is the set of neighbors for :math:`j`,
            e.g., ``True`` requires equality of the weight to consider
            two links equal, ``False`` requires only a presence of a link
            with a non-zero weight.

        Returns
        -------
        pandas.Series
            A ``Series`` of ``(i,j)`` pairs of asymmetries sorted
            ascending by the focal observation (index value),
            where ``i`` is the focal and ``j`` is the neighbor.
            An empty ``Series`` is returned if no asymmetries are found.
        """
        if intrinsic:
            wd = self.sparse.transpose() - self.sparse
        else:
            transformed = self.transform("b")
            wd = transformed.sparse.transpose() - transformed.sparse

        ids = np.nonzero(wd)
        if len(ids[0]) == 0:
            return pd.Series(
                index=pd.Index([], name="focal"),
                name="neighbor",
                dtype=self._adjacency.index.dtypes["focal"],
            )
        else:
            i2id = dict(
                zip(np.arange(self.unique_ids.shape[0]), self.unique_ids, strict=True)
            )
            focal, neighbor = np.nonzero(wd)
            focal = focal.astype(self._adjacency.index.dtypes["focal"])
            neighbor = neighbor.astype(self._adjacency.index.dtypes["focal"])
            for i in i2id:
                focal[focal == i] = i2id[i]
                neighbor[neighbor == i] = i2id[i]
            ijs = pd.Series(
                neighbor, index=pd.Index(focal, name="focal"), name="neighbor"
            ).sort_index()
            return ijs

    def summary(self, asymmetries=False):
        """Summary of the Graph properties

        Returns a :class:`GraphSummary` object with the statistical attributes
        summarising the Graph and its basic properties. See the docstring of the
        :class:`GraphSummary` for details and all the available attributes.

        Parameters
        ----------
        asymmetries : bool
            whether to compute ``n_asymmetries``, which is considerably more expensive
            than the other attributes. By default False.

        Returns
        -------
        GraphSummary
            a class containing a summary statisitcs about the graph

        Examples
        --------
        >>> import geopandas as gpd
        >>> from geodatasets import get_path
        >>> nybb = gpd.read_file(get_path("nybb")).set_index("BoroName")
        >>> nybb
                       BoroCode  ...                                           geometry
        BoroName                 ...
        Staten Island         5  ...  MULTIPOLYGON (((970217.022 145643.332, 970227....
        Queens                4  ...  MULTIPOLYGON (((1029606.077 156073.814, 102957...
        Brooklyn              3  ...  MULTIPOLYGON (((1021176.479 151374.797, 102100...
        Manhattan             1  ...  MULTIPOLYGON (((981219.056 188655.316, 980940....
        Bronx                 2  ...  MULTIPOLYGON (((1012821.806 229228.265, 101278...
        [5 rows x 4 columns]

        >>> contiguity = graph.Graph.build_contiguity(nybb)
        >>> contiguity
        <Graph of 5 nodes and 10 nonzero edges indexed by
         ['Staten Island', 'Queens', 'Brooklyn', 'Manhattan', 'Bronx']>

        >>> summary = contiguity.summary(asymmetries=True)
        >>> summary
        Graph Summary Statistics
        ========================
        Graph indexed by:
        ['Staten Island', 'Queens', 'Brooklyn', 'Manhattan', 'Bronx']
        ==============================================================
        Number of nodes:                                             5
        Number of edges:                                            10
        Number of connected components:                              2
        Number of isolates:                                          1
        Number of non-zero edges:                                   10
        Percentage of non-zero edges:                           44.00%
        Number of asymmetries:                                       0
        --------------------------------------------------------------
        Cardinalities
        ==============================================================
        Mean:                       2    25%:                        2
        Standard deviation:         1    50%:                        2
        Min:                        0    75%:                        3
        Max:                        3
        --------------------------------------------------------------
        Weights
        ==============================================================
        Mean:                       1    25%:                        1
        Standard deviation:         0    50%:                        1
        Min:                        0    75%:                        1
        Max:                        1
        --------------------------------------------------------------
        Sum of weights
        ==============================================================
        S0:                                                         10
        S1:                                                         20
        S2:                                                        104
        --------------------------------------------------------------
        Traces
        ==============================================================
        GG:                                                         10
        G'G:                                                        10
        G'G + GG:                                                   20

        >>> summary.s1
        20
        """
        return GraphSummary(self, asymmetries=asymmetries)

    def higher_order(self, k=2, shortest_path=True, diagonal=False, lower_order=False):
        """Contiguity weights object of order :math:`k`.

        Proper higher order neighbors are returned such that :math:`i` and :math:`j`
        are :math:`k`-order neighbors if the shortest path from :math:`i-j` is of
        length :math:`k`.

        Parameters
        ----------
        k : int, optional
            Order of contiguity. By default 2.
        shortest_path : bool, optional
            If True, :math:`i,j` and :math:`k`-order neighbors if the shortest
            path for :math:`i,j` is :math:`k`. If False, :math:`i,j` are
            `k`-order neighbors if there is a path from :math:`i,j` of length
            :math:`k`. By default True.
        diagonal : bool, optional
            If True, keep :math:`k`-order (:math:`i,j`) joins when :math:`i==j`.
            If False, remove :math:`k`-order (:math:`i,j`) joins when
            :math:`i==j`. By default False.
        lower_order : bool, optional
            If True, include lower order contiguities. If False return only weights of
            order :math:`k`. By default False.

        Returns
        -------
        Graph
            higher order weights

        Examples
        --------
        >>> import geopandas as gpd
        >>> from geodatasets import get_path
        >>> gdf = gpd.read_file(get_path("geoda guerry"))
        >>> contiguity = graph.Graph.build_contiguity(gdf)
        >>> contiguity
        <Graph of 85 nodes and 420 nonzero edges indexed by
         [0, 1, 2, 3, 4, ...]>

        >>> contiguity.higher_order(k=2)
        <Graph of 85 nodes and 756 nonzero edges indexed by
         [0, 1, 2, 3, 4, ...]>

        >>> contiguity.higher_order(lower_order=True)
        <Graph of 85 nodes and 1176 nonzero edges indexed by
         [0, 1, 2, 3, 4, ...]>
        """
        if not Version(scipy_version) >= Version("1.12.0"):
            raise ImportError("Graph.higher_order() requires scipy>=1.12.0.")

        binary = self.transform("B")
        sp = binary.sparse

        if lower_order:
            wk = sum(sparse.linalg.matrix_power(sp, x) for x in range(1, k + 1))
            shortest_path = False
        else:
            wk = sparse.linalg.matrix_power(sp, k)

        rk, ck = wk.nonzero()
        sk = set(zip(rk, ck, strict=True))

        if shortest_path:
            for j in range(1, k):
                wj = sparse.linalg.matrix_power(sp, j)
                rj, cj = wj.nonzero()
                sj = set(zip(rj, cj, strict=True))
                sk.difference_update(sj)
        if not diagonal:
            sk = {(i, j) for i, j in sk if i != j}

        higher = Graph.from_sparse(
            sparse.coo_array(
                (
                    np.ones(len(sk), dtype=np.int8),
                    ([s[0] for s in sk], [s[1] for s in sk]),
                ),
                shape=sp.shape,
            ),
            ids=self.unique_ids,
        )
        if hasattr(self, "_xarray_index_names"):
            higher._xarray_index_names = self._xarray_index_names

        return higher

    def lag(self, y, categorical=False, ties="raise"):
        """Spatial lag operator

        Constructs spatial lag based on neighbor relations of the graph.


        Parameters
        ----------
        y : array
            numpy array with dimensionality conforming to w
        categorical : bool
            True if y is categorical, False if y is continuous.
        ties : {'raise', 'random', 'tryself'}, optional
            Policy on how to break ties when a focal unit has multiple
            modes for a categorical lag.
            - 'raise': This will raise an exception if ties are
            encountered to alert the user (Default).
            - 'random': modal label ties Will be broken randomly.
            - 'tryself': check if focal label breaks the tie between label
            modes.  If the focal label does not break the modal tie, the
            tie will be be broken randomly. If the focal unit has a
            self-weight, focal label is not used to break any tie,
            rather any tie will be broken randomly.


        Returns
        -------
        numpy.ndarray
            array of numeric|categorical values for the spatial lag

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> import geopandas as gpd
        >>> from geodatasets import get_path
        >>> aus = gpd.read_file(get_path("abs.australia_states_territories")).set_index(
        ...     "STE_NAME21"
        ... )
        >>> aus = aus[aus.geometry.notna()]
        >>> contiguity = graph.Graph.build_contiguity(aus)

        Spatial lag operator for continuous variables.

        >>> y = np.arange(9)
        >>> contiguity.lag(y)
        array([21.,  3.,  9., 13.,  9.,  0.,  9.,  0.,  0.])

        You can also perform transformation of weights.

        >>> contiguity_r = contiguity.transform("r")
        >>> contiguity_r.lag(y)
        array([4.2, 1.5, 3. , 2.6, 4.5, 0. , 3. , 0. , 0. ])
        """
        return _lag_spatial(self, y, categorical=categorical, ties=ties)

    def to_parquet(self, path, **kwargs):
        """Save Graph to a Apache Parquet

        Graph is serialized to the Apache Parquet using the underlying adjacency
        object stored as a Parquet table and custom metadata containing transformation.

        Requires pyarrow package.

        Parameters
        ----------
        path : str | pyarrow.NativeFile
            path or any stream supported by pyarrow
        **kwargs
            additional keyword arguments passed to pyarrow.parquet.write_table

        See also
        --------
        read_parquet

        Examples
        --------
        >>> import geopandas as gpd
        >>> from geodatasets import get_path
        >>> nybb = gpd.read_file(get_path("nybb")).set_index("BoroName")
        >>> nybb
                       BoroCode  ...                                           geometry
        BoroName                 ...
        Staten Island         5  ...  MULTIPOLYGON (((970217.022 145643.332, 970227....
        Queens                4  ...  MULTIPOLYGON (((1029606.077 156073.814, 102957...
        Brooklyn              3  ...  MULTIPOLYGON (((1021176.479 151374.797, 102100...
        Manhattan             1  ...  MULTIPOLYGON (((981219.056 188655.316, 980940....
        Bronx                 2  ...  MULTIPOLYGON (((1012821.806 229228.265, 101278...
        [5 rows x 4 columns]

        >>> contiguity = graph.Graph.build_contiguity(nybb)
        >>> contiguity.to_parquet("contiguity.parquet")
        """
        _to_parquet(self, path, **kwargs)

    def to_gal(self, path):
        """Save Graph to a GAL file

        Graph is serialized to the GAL file format.

        Parameters
        ----------
        path : str
            path to the GAL file

        See also
        --------
        read_gal

        Examples
        --------
        >>> import geopandas as gpd
        >>> from geodatasets import get_path
        >>> nybb = gpd.read_file(get_path("nybb")).set_index("BoroName")
        >>> nybb
                       BoroCode  ...                                           geometry
        BoroName                 ...
        Staten Island         5  ...  MULTIPOLYGON (((970217.022 145643.332, 970227....
        Queens                4  ...  MULTIPOLYGON (((1029606.077 156073.814, 102957...
        Brooklyn              3  ...  MULTIPOLYGON (((1021176.479 151374.797, 102100...
        Manhattan             1  ...  MULTIPOLYGON (((981219.056 188655.316, 980940....
        Bronx                 2  ...  MULTIPOLYGON (((1012821.806 229228.265, 101278...
        [5 rows x 4 columns]

        >>> contiguity = graph.Graph.build_contiguity(nybb)
        >>> contiguity.to_gal("contiguity.gal")
        """
        _to_gal(self, path)

    def to_gwt(self, path):
        """Save Graph to a GWT file

        Graph is serialized to the GWT file format.

        Parameters
        ----------
        path : str
            path to the GWT file

        See also
        --------
        read_gwt

        Examples
        --------
        >>> import geopandas as gpd
        >>> from geodatasets import get_path
        >>> nybb = gpd.read_file(get_path("nybb")).set_index("BoroName")
        >>> nybb
                       BoroCode  ...                                           geometry
        BoroName                 ...
        Staten Island         5  ...  MULTIPOLYGON (((970217.022 145643.332, 970227....
        Queens                4  ...  MULTIPOLYGON (((1029606.077 156073.814, 102957...
        Brooklyn              3  ...  MULTIPOLYGON (((1021176.479 151374.797, 102100...
        Manhattan             1  ...  MULTIPOLYGON (((981219.056 188655.316, 980940....
        Bronx                 2  ...  MULTIPOLYGON (((1012821.806 229228.265, 101278...
        [5 rows x 4 columns]

        >>> contiguity = graph.Graph.build_contiguity(nybb).transform("r")
        >>> contiguity.to_gwt("contiguity.gwt")
        """
        _to_gwt(self, path)

    def to_networkx(self):
        """Convert Graph to a ``networkx`` graph.

        If Graph is symmetric, returns ``nx.Graph``, otherwise returns a ``nx.DiGraph``.

        Returns
        -------
        networkx.Graph | networkx.DiGraph
            Representation of libpysal Graph as networkx graph

        Examples
        --------
        >>> import geopandas as gpd
        >>> from geodatasets import get_path
        >>> nybb = gpd.read_file(get_path("nybb")).set_index("BoroName")
        >>> nybb
                       BoroCode  ...                                           geometry
        BoroName                 ...
        Staten Island         5  ...  MULTIPOLYGON (((970217.022 145643.332, 970227....
        Queens                4  ...  MULTIPOLYGON (((1029606.077 156073.814, 102957...
        Brooklyn              3  ...  MULTIPOLYGON (((1021176.479 151374.797, 102100...
        Manhattan             1  ...  MULTIPOLYGON (((981219.056 188655.316, 980940....
        Bronx                 2  ...  MULTIPOLYGON (((1012821.806 229228.265, 101278...
        [5 rows x 4 columns]

        >>> contiguity = graph.Graph.build_contiguity(nybb)
        >>> nx_graph = contiguity.to_networkx()
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("NetworkX is required.") from None

        graph_type = nx.Graph if self.asymmetry().empty else nx.DiGraph

        return nx.from_pandas_edgelist(
            self._adjacency.reset_index(),
            source="focal",
            target="neighbor",
            edge_attr="weight",
            create_using=graph_type,
        )

    def plot(
        self,
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
        gdf : geopandas.GeoDataFrame
            Geometries indexed using the same index as Graph. Geometry types other than
            points are converted to centroids encoding start and end point of Graph
            edges.
        focal : hashable | array-like[hashable] | None, optional
            ID or an array-like of IDs of focal geometries whose weights shall be
            plotted. If None, all weights from all focal geometries are plotted.
            By default None
        nodes : bool, optional
            Plot nodes as points, by default True
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

        Notes
        -----
        If you'd like to overlay the actual geometries from the
        ``geopandas.GeoDataFrame``, create an axis by plotting the ``GeoDataFrame``
        and plot the Graph on top.

            ax = gdf.plot()
            gdf_graph.plot(gdf, ax=ax)

        """
        return _plot(
            self,
            gdf,
            focal=focal,
            nodes=nodes,
            color=color,
            node_kws=node_kws,
            edge_kws=edge_kws,
            focal_kws=focal_kws,
            ax=ax,
            figsize=figsize,
            limit_extent=limit_extent,
        )

    def explore(
        self,
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
        return _explore_graph(
            self,
            gdf,
            focal=focal,
            nodes=nodes,
            color=color,
            edge_kws=edge_kws,
            node_kws=node_kws,
            focal_kws=focal_kws,
            m=m,
            **kwargs,
        )

    def subgraph(self, ids):
        """Returns a subset of Graph containing only nodes specified in ids

        The resulting subgraph contains only the nodes in ``ids`` and the edges
        between them or zero-weight self-loops in case of isolates.

        The order of ``ids`` reflects a new canonical order of the resulting
        subgraph. This means ``ids`` should be equal to the index of the DataFrame
        containing data linked to the graph to ensure alignment of sparse representation
        of subgraph.

        Parameters
        ----------
        ids : array-like
            An array of node IDs to be retained

        Returns
        -------
        Graph
            A new Graph that is a subset of the original

        Examples
        --------
        >>> import geopandas as gpd
        >>> from geodatasets import get_path
        >>> nybb = gpd.read_file(get_path("nybb")).set_index("BoroName")
        >>> nybb
                       BoroCode  ...                                           geometry
        BoroName                 ...
        Staten Island         5  ...  MULTIPOLYGON (((970217.022 145643.332, 970227....
        Queens                4  ...  MULTIPOLYGON (((1029606.077 156073.814, 102957...
        Brooklyn              3  ...  MULTIPOLYGON (((1021176.479 151374.797, 102100...
        Manhattan             1  ...  MULTIPOLYGON (((981219.056 188655.316, 980940....
        Bronx                 2  ...  MULTIPOLYGON (((1012821.806 229228.265, 101278...
        [5 rows x 4 columns]

        >>> contiguity = graph.Graph.build_contiguity(nybb)
        >>> contiguity.subgraph(["Queens", "Brooklyn", "Manhattan", "Bronx"])
        <Graph of 4 nodes and 10 nonzero edges indexed by
         ['Queens', 'Brooklyn', 'Manhattan', 'Bronx']>

        Notes
        -----
        Unlike the implementation in ``networkx``, this creates a copy since
        Graphs in ``libpysal`` are immutable.
        """
        masked_adj = self._adjacency.loc[ids, :]
        filtered_adj = masked_adj[
            masked_adj.index.get_level_values("neighbor").isin(ids)
        ]
        sub = Graph.from_arrays(
            *_resolve_islands(
                filtered_adj.index.get_level_values("focal"),
                filtered_adj.index.get_level_values("neighbor"),
                ids,
                filtered_adj.values,
            )
        )

        if hasattr(self, "_xarray_index_names"):
            sub._xarray_index_names = self._xarray_index_names

        return sub

    def eliminate_zeros(self):
        """Remove graph edges with zero weight

        Eliminates edges with weight == 0 that do not encode an
        isolate. This is useful to clean-up edges that will make
        no effect in operations like :meth:`lag`.

        Returns
        -------
        Graph
            subset of Graph with zero-weight edges eliminated
        """
        # substract isolates from mask of zeros
        zeros = (self._adjacency == 0) != np.isin(
            self._adjacency.index.get_level_values(0), self.isolates
        )

        eliminated = Graph(self._adjacency[~zeros], is_sorted=True)
        if hasattr(self, "_xarray_index_names"):
            eliminated._xarray_index_names = self._xarray_index_names

        return eliminated

    def assign_self_weight(self, weight=1):
        """Assign values to edges representing self-weight.

        The value for each ``focal == neighbor`` location in
        the graph is set to ``weight``.

        Parameters
        ----------
        weight : float | array-like
            Defines the value(s) to which the weight representing the relationship with
            itself should be set. If a constant is passed then each self-weight will get
            this value (default is 1). An array of length ``Graph.n`` can be passed to
            set explicit values to each self-weight (assumed to be in the same order as
            original data).

        Returns
        -------
        Graph
            A new ``Graph`` with added self-weights.

        Examples
        --------
        >>> import geopandas as gpd
        >>> from geodatasets import get_path
        >>> nybb = gpd.read_file(get_path("nybb")).set_index("BoroName")
        >>> nybb
                       BoroCode  ...                                           geometry
        BoroName                 ...
        Staten Island         5  ...  MULTIPOLYGON (((970217.022 145643.332, 970227....
        Queens                4  ...  MULTIPOLYGON (((1029606.077 156073.814, 102957...
        Brooklyn              3  ...  MULTIPOLYGON (((1021176.479 151374.797, 102100...
        Manhattan             1  ...  MULTIPOLYGON (((981219.056 188655.316, 980940....
        Bronx                 2  ...  MULTIPOLYGON (((1012821.806 229228.265, 101278...
        [5 rows x 4 columns]

        >>> contiguity = graph.Graph.build_contiguity(nybb)
        >>> contiguity_weights = contiguity.assign_self_weight(0.5)
        >>> contiguity_weights.adjacency
        focal          neighbor
        Staten Island  Staten Island    0.5
        Queens         Queens           0.5
                       Brooklyn         1.0
                       Manhattan        1.0
                       Bronx            1.0
        Brooklyn       Queens           1.0
                       Brooklyn         0.5
                       Manhattan        1.0
        Manhattan      Queens           1.0
                       Brooklyn         1.0
                       Manhattan        0.5
                       Bronx            1.0
        Bronx          Queens           1.0
                       Manhattan        1.0
                       Bronx            0.5
        Name: weight, dtype: float64
        """
        addition = pd.Series(
            weight,
            index=pd.MultiIndex.from_arrays(
                [self.unique_ids, self.unique_ids], names=["focal", "neighbor"]
            ),
            name="weight",
        )
        # drop existing self weights and replace them with a new value
        existing_self_weights = self._adjacency.index[
            self._adjacency.index.codes[0] == self._adjacency.index.codes[1]
        ]
        adj = (
            pd.concat([self._adjacency.drop(existing_self_weights), addition])
            .reindex(self.unique_ids, level=0)
            .reindex(self.unique_ids, level=1)
        )
        assigned = Graph(adj, is_sorted=True)

        if hasattr(self, "_xarray_index_names"):
            assigned._xarray_index_names = self._xarray_index_names

        return assigned

    def apply(self, y, func, **kwargs):
        """Apply a reduction across the neighbor sets

        Applies ``func`` over groups of ``y`` defined by neighbors for each focal.

        Parameters
        ----------
        y : array_like
            array of values to be grouped. Can be 1-D or 2-D and will be coerced to a
            pandas object
        func : function, str, list, dict or None
            Function to use for aggregating the data passed to pandas ``GroupBy.apply``.

        Returns
        -------
        Series | DataFrame
            pandas object indexed by unique_ids
        """
        if not isinstance(y, pd.Series | pd.DataFrame):
            y = pd.DataFrame(y) if hasattr(y, "ndim") and y.ndim == 2 else pd.Series(y)
        grouper = y.take(self._adjacency.index.codes[1]).groupby(
            self._adjacency.index.codes[0], sort=False
        )
        result = grouper.apply(func, **kwargs)
        result.index = self.unique_ids
        if isinstance(result, pd.Series):
            result.name = None
        return result

    def aggregate(self, func):
        """Aggregate weights within a neighbor set

        Apply a custom aggregation function to a group of weights of the same focal
        geometry.

        Parameters
        ----------
        func : callable
            A callable accepted by pandas ``groupby.agg`` method

        Returns
        -------
        pd.Series
            Aggregated weights
        """
        return self._adjacency.groupby(level=0, sort=False).agg(func)

    def describe(
        self,
        y,
        q=None,
        statistics=None,
    ):
        """Describe the distribution of ``y`` values within the neighbors of each node.

        Given the graph, computes the descriptive statistics of values within the
        neighbourhood of each node. Optionally, the values can be limited to a certain
        quantile range before computing the statistics.

        Notes
        -----
        The index of ``values`` must match the index of the graph.

        Weight values do not affect the calculations, only adjacency does.

        Returns numpy.nan for all isolates.

        The numba package is used extensively in this function
        to accelerate the computation of statistics.
        Without numba, these computations may become slow on large data.

        Parameters
        ----------
        y : NDArray[np.float64] | Series
            An 1D array of numeric values to be described.
        q : tuple[float, float] | None, optional
            Tuple of percentages for the percentiles to compute.
            Values must be between 0 and 100 inclusive. When set, values below and above
            the percentiles will be discarded before computation of the statistics.
            The percentiles are computed for each neighborhood. By default None.
        statistics : list[str] | None
            A list of stats functions to compute. If None, compute all
            available functions - "count", "mean", "median",
            "std", "min", "max", "sum", "nunique", "mode". By default None.

        Returns
        -------
        DataFrame
            A DataFrame with descriptive statistics.
        """

        if not isinstance(y, pd.Series):
            y = pd.Series(y, index=self.unique_ids)

        if (y.index != self.unique_ids).all():
            raise ValueError("The values index is not aligned with the graph index.")

        if q is None:
            grouper = y.take(self._adjacency.index.codes[1]).groupby(
                self._adjacency.index.codes[0], sort=False
            )
        else:
            grouper = _percentile_filtration_grouper(y, self._adjacency.index, q=q)

        stat_ = _compute_stats(grouper, statistics)

        stat_.index = self.unique_ids
        if isinstance(stat_, pd.Series):
            stat_.name = None
        # NA isolates
        stat_.loc[self.isolates] = np.nan
        return stat_

    def generate_da(self, y):
        """Creates xarray.DataArray object from passed data aligned with the Graph.

        Parameters
        ----------
        y : array_like
            flat array that shall be reshaped into a DataArray with dimensionality
            conforming to Graph

        Returns
        -------
        xarray.DataArray
            instance of xarray.DataArray that can be aligned with the DataArray from
            which Graph was built
        """
        return _generate_da(self, y)


def _arrange_arrays(heads, tails, weights, ids=None):
    """
    Rearrange input arrays so that observation indices
    are well-ordered with respect to the input ids. That is,
    an "early" identifier should always preceed a "later" identifier
    in the heads, but the tails should be sorted with respect
    to heads *first*, then sorted within the tails.
    """
    if ids is None:
        ids = np.unique(np.hstack((heads, tails)))
    lookup = list(ids).index
    input_df = pd.DataFrame.from_dict(
        {"focal": heads, "neighbor": tails, "weight": weights}
    )
    return (
        input_df.set_index(["focal", "neighbor"])
        .assign(
            focal_loc=input_df.focal.apply(lookup).values,
            neighbor_loc=input_df.neighbor.apply(lookup).values,
        )
        .sort_values(["focal_loc", "neighbor_loc"])
        .reset_index()
        .drop(["focal_loc", "neighbor_loc"], axis=1)
        .values.T
    )


def read_parquet(path, **kwargs):
    """Read Graph from a Apache Parquet

    Read Graph serialized using `Graph.to_parquet()` back into the `Graph` object.
    The Parquet file needs to contain adjacency table with a structure required
    by the `Graph` constructor and optional metadata with the type of transformation.

    Parameters
    ----------
    path : str | pyarrow.NativeFile | file-like object
        path or any stream supported by pyarrow
    **kwargs
        additional keyword arguments passed to pyarrow.parquet.read_table

    Returns
    -------
    Graph
        deserialized Graph

    Examples
    --------
    >>> graph.read_parquet("contiguity.parquet")
    """
    adjacency, transformation, xarray_index_names = _read_parquet(path, **kwargs)
    graph_obj = Graph(adjacency, transformation, is_sorted=True)
    if xarray_index_names is not None:
        graph_obj._xarray_index_names = xarray_index_names
    return graph_obj


def read_gal(path):
    """Read Graph from a GAL file

    The reader tries to infer the dtype of IDs. In case of unsuccessful
    casting to int, it will fall back to string.

    Parameters
    ----------
    path : str
        path to a file

    Returns
    -------
    Graph
        deserialized Graph

    Examples
    --------
    >>> graph.read_parquet("contiguity.gal")
    """
    neighbors = _read_gal(path)
    return Graph.from_dicts(neighbors)


def read_gwt(path):
    """Read Graph from a GWT file

    Parameters
    ----------
    path : str
        path to a file

    Returns
    -------
    Graph
        deserialized Graph

    Examples
    --------
    >>> graph.read_parquet("contiguity.gwt")
    """
    head, tail, weight = _read_gwt(path)
    return Graph.from_arrays(head, tail, weight)

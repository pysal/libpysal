"""
For completeness, we need to test a shuffled dataframe
(i.e. always send unsorted data) with:
- numeric ids
- string ids
- point dataframe
- coordinates
- check two kernel functions
- numba/nonumba
"""
import geodatasets
import geopandas
import numpy
import pytest
import shapely
from scipy import spatial

from libpysal.graph._kernel import _kernel_functions
from libpysal.graph._triangulation import (
    _delaunay,
    _gabriel,
    _relative_neighborhood,
    _voronoi,
)
from libpysal.graph.base import Graph

stores = geopandas.read_file(geodatasets.get_path("geoda liquor_stores")).explode(
    index_parts=False
)
stores_unique = stores.drop_duplicates(subset="geometry")

kernel_functions = [None] + list(_kernel_functions.keys())


def my_kernel(distances, bandwidth):
    output = numpy.cos(distances / distances.max())
    output[distances < bandwidth] = 0
    return output


kernel_functions.append(my_kernel)

# optimal, small, and larger than largest distance.
bandwidths = [None, "auto", 0.5]

numpy.random.seed(6301)
# create a 2-d laplace distribution as a "degenerate"
# over-concentrated distribution
lap_coords = numpy.random.laplace(size=(5, 2))

# create a 2-d cauchy as a "degenerate"
# spatial outlier-y distribution
cau_coords = numpy.random.standard_cauchy(size=(5, 2))

parametrize_ids = pytest.mark.parametrize(
    "ids", [None, "id", "placeid"], ids=["no id", "id", "placeid"]
)

parametrize_kernelfunctions = pytest.mark.parametrize(
    "kernel", kernel_functions, ids=kernel_functions[:-2] + ["None", "custom callable"]
)
parametrize_bw = pytest.mark.parametrize(
    "bandwidth", bandwidths, ids=["no bandwidth", "auto", "fixed"]
)
parametrize_constructors = pytest.mark.parametrize(
    "constructor",
    [_delaunay, _gabriel, _relative_neighborhood, _voronoi],
    ids=["delaunay", "gabriel", "relhood", "voronoi"],
)

# @parametrize_constructors
# @parametrize_ids
# @parametrize_kernelfunctions``
# @parametrize_bw
# def test_option_combinations(constructor, ids, kernel, bandwidth):
#     """
#     NOTE: This does not check for the *validity* of the output, just
#     the structure of the output.
#     """
#     heads, tails, weights = constructor(
#         stores_unique,
#         ids=stores_unique[ids] if ids is not None else None,
#         kernel=kernel,
#         bandwidth=bandwidth
#         )
#     assert heads.dtype == tails.dtype
#     assert heads.dtype == stores_unique.get(ids, stores_unique.index).dtype, 'ids failed to propagate'
#     if kernel is None and bandwidth is None:
#         numpy.testing.assert_array_equal(weights, numpy.ones_like(heads))
#     assert set(zip(heads, tails)) == set(zip(tails, heads)), "all triangulations should be symmetric, this is not"


def test_correctness_voronoi_clipping():
    noclip = _voronoi(lap_coords, clip=None, rook=True)
    extent = _voronoi(lap_coords, clip="extent", rook=True)
    alpha = _voronoi(lap_coords, clip="ashape", rook=True)

    G_noclip = Graph.from_arrays(*noclip)
    G_extent = Graph.from_arrays(*extent)
    G_alpha = Graph.from_arrays(*alpha)

    assert G_alpha < G_extent
    assert G_extent <= G_noclip

    D = spatial.distance.squareform(spatial.distance.pdist(lap_coords))

    extent_known = [
        numpy.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4]),
        numpy.array([1, 2, 3, 4, 0, 3, 4, 0, 3, 0, 1, 2, 0, 1]),
    ]
    alpha_known = [
        numpy.array([0, 0, 0, 1, 2, 2, 3, 3, 4, 4]),
        numpy.array([2, 3, 4, 4, 0, 3, 0, 2, 0, 1]),
    ]

    numpy.testing.assert_array_equal(G_extent.adjacency.index, extent_known[0])
    numpy.testing.assert_array_equal(G_extent.adjacency.neighbor, extent_known[1])

    numpy.testing.assert_array_equal(G_alpha.adjacency.index, alpha_known[0])
    numpy.testing.assert_array_equal(G_alpha.adjacency.neighbor, alpha_known[1])


# TODO: this now fails, probably never worked.
@pytest.mark.xfail
def test_correctness_delaunay_family():
    for i, data in enumerate((cau_coords, lap_coords)):
        voronoi = _voronoi(data, clip=False)
        delaunay = _delaunay(data)
        gabriel = _gabriel(data)
        relneigh = _relative_neighborhood(data)
        G_voronoi = Graph.from_arrays(*voronoi)
        G_delaunay = Graph.from_arrays(*delaunay)
        G_gabriel = Graph.from_arrays(*gabriel)
        G_relneigh = Graph.from_arrays(*relneigh)

        if i:
            known_delaunay = [
                (0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4),
                (1, 2, 4, 0, 2, 3, 0, 1, 3, 4, 1, 2, 4, 0, 2, 3),
            ]
            known_gabriel = [
                (4, 0, 0, 2, 1, 3, 2, 0, 3, 4),
                (0, 4, 3, 0, 4, 0, 3, 2, 2, 1),
            ]
            known_relneigh = [
                (0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4),
                (1, 2, 3, 4, 0, 4, 0, 3, 0, 2, 0, 1),
            ]
        else:
            known_delaunay = [
                (0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4),
                (1, 2, 3, 4, 0, 3, 4, 0, 3, 4, 0, 1, 2, 0, 1, 2),
            ]
            known_gabriel = [
                (4, 0, 0, 2, 1, 3, 2, 0, 3, 4),
                (0, 4, 3, 0, 4, 0, 3, 2, 2, 1),
            ]
            known_relneigh = [
                (0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4),
                (1, 2, 3, 4, 0, 4, 0, 3, 0, 2, 0, 1),
            ]

        assert G_voronoi == G_delaunay
        assert G_gabriel <= G_delaunay
        assert G_relneigh <= G_delaunay
        for i, name in enumerate(["delaunay", "gabriel", "relneigh"]):
            G_known = (known_delaunay, known_gabriel, known_relneigh)[i]
            G_computed = (delaunay, voronoi, gabriel)[i]
            assert G_known == G_computed, (
                f"computed {name} not equivalent to stored {name} for "
                f"{('cauchy', 'laplacian')[i]} coordinates!"
            )


def test_coincident_raise_voronoi():
    with pytest.raises(ValueError) as excinfo:
        _voronoi(stores, clip=False)
    assert "There are" in str(excinfo.value)


def test_coincident_jitter_voronoi():
    G_voronoi_cp_heads, G_voronoi_cp_tails, cp_w = _voronoi(
        stores, clip=False, coincident="jitter"
    )
    G_voronoi_unique_heads, G_voronoi_unique_tails, unique_w = _voronoi(
        stores_unique, clip=False
    )
    assert not numpy.array_equal(G_voronoi_cp_heads, G_voronoi_unique_heads)
    assert not numpy.array_equal(G_voronoi_cp_tails, G_voronoi_unique_tails)
    assert not numpy.array_equal(cp_w, unique_w)


def test_coincident_clique_voronoi():
    G_voronoi_cp_heads, G_voronoi_cp_tails, cp_w = _voronoi(
        stores, clip=False, coincident="clique"
    )
    G_voronoi_unique_heads, G_voronoi_unique_tails, unique_w = _voronoi(
        stores_unique, clip=False
    )
    assert not numpy.array_equal(G_voronoi_cp_heads, G_voronoi_unique_heads)
    assert not numpy.array_equal(G_voronoi_cp_tails, G_voronoi_unique_tails)
    assert not numpy.array_equal(cp_w, unique_w)


class Test_Coincident:
    def setup_method(self):
        self.geom = [
            shapely.Point(0, 0),
            shapely.Point(1, 1),
            shapely.Point(2, 0),
            shapely.Point(3, 1),
            shapely.Point(0, 0),  # coincident point
            shapely.Point(0, 5),
        ]
        self.df_int = geopandas.GeoDataFrame(
            geometry=self.geom,
        )
        self.df_string = geopandas.GeoDataFrame(
            geometry=self.geom, index=["zero", "one", "two", "three", "four", "five"]
        )
        self.mapping = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}

    def test_delaunay_error(self):
        with pytest.raises(
            ValueError,
            match="There are 5 unique locations in the dataset, but 6 observations",
        ):
            _delaunay(self.df_int)

    def test_delaunay_jitter(self):
        heads, tails, weights = _delaunay(self.df_int, coincident="jitter", seed=0)

        exp_heads = numpy.array(
            [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5]
        )
        exp_tails = numpy.array(
            [1, 2, 4, 0, 2, 3, 4, 5, 3, 1, 0, 1, 2, 5, 0, 1, 5, 1, 3, 4]
        )
        exp_w = numpy.ones(exp_heads.shape, dtype="int8")

        numpy.testing.assert_array_equal(heads, exp_heads)
        numpy.testing.assert_array_equal(tails, exp_tails)

        heads, tails, weights = _delaunay(self.df_string, coincident="jitter", seed=0)

        numpy.testing.assert_array_equal(
            heads, numpy.vectorize(self.mapping.get)(exp_heads)
        )
        numpy.testing.assert_array_equal(
            tails, numpy.vectorize(self.mapping.get)(exp_tails)
        )
        numpy.testing.assert_array_equal(weights, exp_w)

    @pytest.mark.xfail
    def test_delaunay_clique(self):
        # TODO: fix the implemntation to make this pass
        heads, tails, weights = _delaunay(self.df_int, coincident="clique", seed=0)

        exp_heads = numpy.array(
            [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5]
        )
        # the order may be different in the end but observations not
        exp_tails = numpy.array(
            [1, 2, 5, 4, 0, 2, 3, 5, 0, 1, 3, 1, 2, 5, 0, 1, 2, 5, 0, 1, 3]
        )
        exp_w = numpy.ones(exp_heads.shape, dtype="int8")

        numpy.testing.assert_array_equal(heads, exp_heads)
        numpy.testing.assert_array_equal(tails, exp_tails)

        heads, tails, weights = _delaunay(self.df_string, coincident="clique", seed=0)

        numpy.testing.assert_array_equal(
            heads, numpy.vectorize(self.mapping.get)(exp_heads)
        )
        numpy.testing.assert_array_equal(
            tails, numpy.vectorize(self.mapping.get)(exp_tails)
        )
        numpy.testing.assert_array_equal(weights, exp_w)

    def test_gabriel_error(self):
        with pytest.raises(
            ValueError,
            match="There are 5 unique locations in the dataset, but 6 observations",
        ):
            _gabriel(self.df_int)

    def test_gabriel_jitter(self):
        heads, tails, weights = _gabriel(self.df_int, coincident="jitter", seed=0)

        exp_heads = numpy.array([0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5])
        exp_tails = numpy.array([4, 2, 2, 5, 3, 4, 1, 0, 3, 1, 2, 5, 0, 1, 1, 3])
        exp_w = numpy.ones(exp_heads.shape, dtype="int8")

        numpy.testing.assert_array_equal(heads, exp_heads)
        numpy.testing.assert_array_equal(tails, exp_tails)

        heads, tails, weights = _gabriel(self.df_string, coincident="jitter", seed=0)

        numpy.testing.assert_array_equal(
            heads, numpy.vectorize(self.mapping.get)(exp_heads)
        )
        numpy.testing.assert_array_equal(
            tails, numpy.vectorize(self.mapping.get)(exp_tails)
        )
        numpy.testing.assert_array_equal(weights, exp_w)

    @pytest.mark.xfail
    def test_gabriel_clique(self):
        # TODO: fix the implemntation to make this pass
        heads, tails, weights = _gabriel(self.df_int, coincident="clique", seed=0)

        exp_heads = numpy.array(
            [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5]
        )
        # the order may be different in the end but observations not
        exp_tails = numpy.array(
            [1, 2, 5, 4, 0, 2, 3, 5, 0, 1, 3, 1, 2, 5, 0, 1, 2, 5, 0, 1, 3]
        )
        exp_w = numpy.ones(exp_heads.shape, dtype="int8")

        numpy.testing.assert_array_equal(heads, exp_heads)
        numpy.testing.assert_array_equal(tails, exp_tails)

        heads, tails, weights = _gabriel(self.df_string, coincident="clique", seed=0)

        numpy.testing.assert_array_equal(
            heads, numpy.vectorize(self.mapping.get)(exp_heads)
        )
        numpy.testing.assert_array_equal(
            tails, numpy.vectorize(self.mapping.get)(exp_tails)
        )
        numpy.testing.assert_array_equal(weights, exp_w)

    def test_relative_neighborhood_error(self):
        with pytest.raises(
            ValueError,
            match="There are 5 unique locations in the dataset, but 6 observations",
        ):
            _relative_neighborhood(self.df_int)

    def test_relative_neighborhood_jitter(self):
        heads, tails, weights = _relative_neighborhood(
            self.df_int, coincident="jitter", seed=0
        )

        exp_heads = numpy.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5])
        exp_tails = numpy.array([1, 2, 4, 0, 3, 4, 5, 3, 0, 1, 2, 5, 0, 1, 5, 3, 1, 4])
        exp_w = numpy.ones(exp_heads.shape, dtype="int8")

        numpy.testing.assert_array_equal(heads, exp_heads)
        numpy.testing.assert_array_equal(tails, exp_tails)

        heads, tails, weights = _relative_neighborhood(
            self.df_string, coincident="jitter", seed=0
        )

        numpy.testing.assert_array_equal(
            heads, numpy.vectorize(self.mapping.get)(exp_heads)
        )
        numpy.testing.assert_array_equal(
            tails, numpy.vectorize(self.mapping.get)(exp_tails)
        )
        numpy.testing.assert_array_equal(weights, exp_w)

    @pytest.mark.xfail
    def test_relative_neighborhood_clique(self):
        # TODO: fix the implemntation to make this pass
        heads, tails, weights = _relative_neighborhood(
            self.df_int, coincident="clique", seed=0
        )

        exp_heads = numpy.array(
            [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5]
        )
        # the order may be different in the end but observations not
        exp_tails = numpy.array(
            [1, 2, 5, 4, 0, 2, 3, 5, 0, 1, 3, 1, 2, 5, 0, 1, 2, 5, 0, 1, 3]
        )
        exp_w = numpy.ones(exp_heads.shape, dtype="int8")

        numpy.testing.assert_array_equal(heads, exp_heads)
        numpy.testing.assert_array_equal(tails, exp_tails)

        heads, tails, weights = _relative_neighborhood(
            self.df_string, coincident="clique", seed=0
        )

        numpy.testing.assert_array_equal(
            heads, numpy.vectorize(self.mapping.get)(exp_heads)
        )
        numpy.testing.assert_array_equal(
            tails, numpy.vectorize(self.mapping.get)(exp_tails)
        )
        numpy.testing.assert_array_equal(weights, exp_w)

    def test_voronoi_error(self):
        with pytest.raises(
            ValueError,
            match="There are 5 unique locations in the dataset, but 6 observations",
        ):
            _voronoi(self.df_int)

    def test_voronoi_jitter(self):
        heads, tails, weights = _voronoi(self.df_int, coincident="jitter", seed=0)

        exp_heads = numpy.array([0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5])
        exp_tails = numpy.array([1, 2, 4, 0, 2, 3, 4, 5, 3, 0, 1, 1, 2, 5, 0, 1, 1, 3])
        exp_w = numpy.ones(exp_heads.shape, dtype="int8")

        numpy.testing.assert_array_equal(heads, exp_heads)
        numpy.testing.assert_array_equal(tails, exp_tails)

        heads, tails, weights = _voronoi(self.df_string, coincident="jitter", seed=0)

        numpy.testing.assert_array_equal(
            heads, numpy.vectorize(self.mapping.get)(exp_heads)
        )
        numpy.testing.assert_array_equal(
            tails, numpy.vectorize(self.mapping.get)(exp_tails)
        )
        numpy.testing.assert_array_equal(weights, exp_w)

    @pytest.mark.xfail
    def test_voronoi_clique(self):
        # TODO: fix the implemntation to make this pass
        heads, tails, weights = _voronoi(self.df_int, coincident="clique", seed=0)

        exp_heads = numpy.array([0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5])
        # the order may be different in the end but observations not
        exp_tails = numpy.array([1, 4, 0, 2, 3, 5, 1, 3, 1, 2, 5, 0, 1, 1, 3])
        exp_w = numpy.ones(exp_heads.shape, dtype="int8")

        numpy.testing.assert_array_equal(heads, exp_heads)
        numpy.testing.assert_array_equal(tails, exp_tails)

        heads, tails, weights = _voronoi(self.df_string, coincident="clique", seed=0)

        numpy.testing.assert_array_equal(
            heads, numpy.vectorize(self.mapping.get)(exp_heads)
        )
        numpy.testing.assert_array_equal(
            tails, numpy.vectorize(self.mapping.get)(exp_tails)
        )
        numpy.testing.assert_array_equal(weights, exp_w)

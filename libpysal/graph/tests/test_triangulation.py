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
import pandas, geopandas, geodatasets, pytest, shapely, numpy
from scipy import spatial
from libpysal.graph._triangulation import (
    _delaunay,
    _gabriel,
    _relative_neighborhood,
    _voronoi,
)
from libpysal.graph._kernel import _kernel_functions
from libpysal.graph.base import Graph

### TODO: is there any way to remove this duplication
### btw. test_triangulation and test_kernel using the
### conftests.py? We need the same data/kernel combos
### and also need to parameterize on numba existing,
### and also have to test coincident points.

stores = geopandas.read_file(
    geodatasets.get_path("geoda liquor_stores")
).explode(index_parts=False)[['id', 'placeid', 'geometry']]
stores_unique = stores.drop_duplicates(subset='geometry')

kernel_functions = [None] + list(_kernel_functions.keys())

def my_kernel(distances, bandwidth):
    output = numpy.cos(distances/distances.max())
    output[distances < bandwidth] = 0
    return output

kernel_functions.append(my_kernel)

# optimal, small, and larger than largest distance.
bandwidths = [None, 'optimal', .5]

numpy.random.seed(6301)
# create a 2-d laplace distribution as a "degenerate"
# over-concentrated distribution
lap_coords = numpy.random.laplace(size=(5,2))

# create a 2-d cauchy as a "degenerate"
# spatial outlier-y distribution
cau_coords = numpy.random.standard_cauchy(size=(5,2))

data = (
    stores,
    lap_coords
    )

parametrize_ids = pytest.mark.parametrize("ids", [None, "id", "placeid"], ids = ["no id", "id", "placeid"])
parametrize_data = pytest.mark.parametrize("data", 
    [stores, lap_coords], 
    ids = ["coords: 2d-laplacian", "coords: 2d-cauchy"]
    )
parametrize_kernelfunctions = pytest.mark.parametrize(
    "kernel",
    kernel_functions,
    ids = kernel_functions[:-2] + ['None', 'custom callable']
    )
parametrize_bw = pytest.mark.parametrize(
    "bandwidth",
    bandwidths,
    ids = ["no bandwidth", 'optimal', 'fixed']
    )
parametrize_constructors = pytest.mark.parametrize(
    "constructor", 
    [_delaunay, _gabriel, _relative_neighborhood, _voronoi],
    ids = ['delaunay', 'gabriel', 'relhood', 'voronoi']
    )

@parametrize_constructors
@parametrize_ids
@parametrize_kernelfunctions
@parametrize_bw
def test_option_combinations(constructor, ids, kernel, bandwidth):
    """
    NOTE: This does not check for the *validity* of the output, just
    the structure of the output. 
    """
    heads, tails, weights = constructor(
        stores_unique, 
        ids=stores_unique[ids] if ids is not None else None, 
        kernel=kernel, 
        bandwidth=bandwidth
        )
    assert heads.dtype == tails.dtype
    assert heads.dtype == stores_unique.get(ids, stores_unique.index).dtype, 'ids failed to propagate'
    if kernel is None:
        numpy.testing.assert_array_equal(weights, numpy.ones_like(heads))
    assert set(zip(heads, tails)) == set(zip(tails, heads)), "all triangulations should be symmetric, this is not"


@pytest.mark.parametrize('kernel', [None, "gaussian"], ids=['no kernel', 'gaussian kernel'])
def test_voronoi(kernel):
    extent = _voronoi(lap_coords, clip='extent', rook=True, kernel=kernel)
    alpha = _voronoi(lap_coords, clip='ashape', rook=True, kernel=kernel)
    G_extent = Graph.from_arrays(*extent)
    G_alpha = Graph.from_arrays(*alpha)
    assert G_alpha < G_extent

    D = spatial.distance.squareform(spatial.distance.pdist(lap_coords))

    extent_known = [
        numpy.array([0,0,0,0,1,1,1,2,2,3,3,3,4,4]),
        numpy.array([1,2,3,4,0,3,4,0,3,0,1,2,0,1]),
    ]
    alpha_known = [
        numpy.array([0,0,0,1,2,2,3,3,4,4]),
        numpy.array([2,3,4,4,0,3,0,2,0,1])
    ]
    if kernel is not None:
        extent_known.append(
            _kernel_functions[kernel](D[extent_known[0], extent_known[1]], 1)
            )
        alpha_known.append(
            _kernel_functions[kernel](D[alpha_known[0], alpha_known[1]], 1)
            )
    else:
        extent_known.append(numpy.ones_like(extent_known[0]))
        alpha_known.append(numpy.ones_like(alpha_known[0]))

    G_extent_known = Graph.from_arrays(
        *extent_known
        )
    G_alpha_known = Graph.from_arrays(
        *alpha_known
        )
    assert G_extent_known == G_extent
    assert G_alpha_known == G_alpha

"""
@parametrize_ids
@parametrize_data
@parametrize_kernelfunctions
@paramterize_bw
def test_delaunay(data, ids, kernel, bandwidth):
    raise NotImplementedError()

@parametrize_ids
@parametrize_data
@parametrize_kernelfunctions
@paramterize_bw
def test_gabriel(data, ids, kernel, bandwidth):
    raise NotImplementedError()

@parametrize_ids
@parametrize_data
@parametrize_kernelfunctions
@paramterize_bw
def test_relative_neighborhood(data, ids, kernel, bandwidth):
    raise NotImplementedError()

@paramterize_graphs
def test_collinear()
"""
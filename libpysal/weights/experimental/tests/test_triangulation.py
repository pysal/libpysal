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
from libpysal.weights.experimental._triangulation import (
    delaunay,
    gabriel,
    relative_neighborhood,
    voronoi,
)

# ### TODO: is there any way to remove this duplication
# ### btw. test_triangulation and test_kernel using the
# ### conftests.py? We need the same data/kernel combos
# ### and also need to parameterize on numba existing,
# ### and also have to test coincident points.

# grocs = geopandas.read_file(
#     geodatasets.get_path("geoda groceries")
# )[['OBJECTID', 'geometry']]
# grocs['strID'] = grocs.OBJECTID.astype(str)
# grocs['intID'] = grocs.OBJECTID.values

# kernel_functions = list(_kernel_functions.keys())

# def my_kernel(distances, bandwidth):
#     output = numpy.cos(distances/distances.max())
#     output[distances < bandwidth] = 0
#     return output

# kernel_functions.append(my_kernel)

# # optimal, small, and larger than largest distance.
# # the optimal bandwidth should be smaller than the
# # max distance for all kernels except identity
# bandwidths = [None, .05, .4]

# numpy.random.seed(6301)
# # create a 2-d laplace distribution as a "degenerate"
# # over-concentrated distribution
# lap_coords = numpy.random.laplace(size=(200,2))

# # create a 2-d cauchy as a "degenerate"
# # spatial outlier-y distribution
# # cau_coords = numpy.random.standard_cauchy(size=(200,2))

# data = (
#     grocs,
#     lap_coords
#     )

# parametrize_ids = pytest.mark.parametrize("ids", [None, "strID", "intID"])
# parametrize_data = pytest.mark.parametrize("data", [grocs, lap_coords, cau_coords], ["groceries", "coords: 2d-laplacian"])
# parametrize_kernelfunctions = pytest.mark.parametrize(
#     "kernel",
#     kernel_functions,
#     kernel_functions[:-2] + ['None', 'custom callable']
#     )
# parametrize_bw = pytest.mark.parametrize(
#     "bandwidth",
#     bandwidth,
#     ['optimal', 'small', 'large']
#     )
# paramterize_graphs = pytest.mark.parametrize(
# 	"graphtype",
# 	(delaunay, gabriel, relative_neighborhood, voronoi),
# 	['delaunay', 'gabriel' 'relative_neighborhood', 'voronoi']
# 	)

# @parametrize_ids
# @parametrize_data
# @parametrize_kernelfunctions
# @paramterize_bw
# def test_voronoi(data, ids, kernel, bandwidth):
# 	raise NotImplementedError()

# @parametrize_ids
# @parametrize_data
# @parametrize_kernelfunctions
# @paramterize_bw
# def test_delaunay(data, ids, kernel, bandwidth):
# 	raise NotImplementedError()

# @parametrize_ids
# @parametrize_data
# @parametrize_kernelfunctions
# @paramterize_bw
# def test_gabriel(data, ids, kernel, bandwidth):
# 	raise NotImplementedError()

# @parametrize_ids
# @parametrize_data
# @parametrize_kernelfunctions
# @paramterize_bw
# def test_relative_neighborhood(data, ids, kernel, bandwidth):
# 	raise NotImplementedError()

# @paramterize_graphs
# def test_collinear()

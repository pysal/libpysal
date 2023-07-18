"""
For completeness, we need to test a shuffled dataframe
(i.e. always send unsorted data) with:
- numeric ids
- string ids
- point dataframe
- coordinates
- check two kernel functions
- check two tree types
- scikit/no scikit
"""
# import pandas, geopandas, geodatasets, pytest, shapely, numpy
# from libpysal.weights.experimental._kernel import (
#     kernel,
#     knn,
#     _optimize_bandwidth,
#     _kernel_functions
# )

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

# metrics = ("euclidean", "haversine")

# # optimal, small, and larger than largest distance.
# # the optimal bandwidth should be smaller than the
# # max distance for all kernels except identity
# bandwidth = [None, .05, .4]

# numpy.random.seed(6301)
# # create a 2-d laplace distribution as a "degenerate"
# # over-concentrated distribution
# # and rescale to match the lenght-scale in groceries
# lap_coords = numpy.random.laplace(size=(200,2)) / 50

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
# parametrize_metrics = pytest.mark.parametrize(
#     "metric",
#     metrics,
#     metrics
#     )
# parametrize_bw = pytest.mark.parametrize(
#     "bandwidth",
#     bandwidth,
#     ['optimal', 'small', 'large']
#     )

# # how do we parameterize conditional on sklearn in env?

# @parametrize_ids
# @parametrize_data
# @parametrize_kernelfunctions
# @parametrize_metrics
# @paramterize_bw
# def test_kernel(data, ids, kernel, metric, bandwidth):
#     raise NotImplementedError()

# @parametrize_ids
# @parametrize_data
# @parametrize_kernelfunctions
# @parametrize_metrics
# @paramterize_bw
# def test_knn(data, ids, kernel, metric, bandwidth):
#     raise NotImplementedError()

# @parametrize_ids
# @parametrize_data
# def test_precomputed(data, ids):
#     raise NotImplementedError()

# @parametrize_data
# def test_coincident(data):
#     raise NotImplementedError()

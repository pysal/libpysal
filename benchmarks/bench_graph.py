import geopandas as gpd
import numpy as np
from scipy.sparse import linalg as spla
from geodatasets import get_path

from libpysal.graph import Graph


class TimeSuite:
    def setup(self, *args, **kwargs):
        self.gdf = gpd.read_file(get_path("geoda south"))
        self.gdf_str = self.gdf.set_index(self.gdf.NAME + " " + self.gdf.STATE_NAME)
        self.gdf_points = self.gdf.set_geometry(self.gdf.representative_point())
        self.gdf_str_points = self.gdf_str.set_geometry(
            self.gdf_str.representative_point()
        )

        self.graphs = {
            "small_int": Graph.build_knn(self.gdf_points, k=10),
            "large_int": Graph.build_knn(self.gdf_points, k=500),
            "small_str": Graph.build_knn(self.gdf_str_points, k=10),
            "large_str": Graph.build_knn(self.gdf_str_points, k=500),
        }
        self.ids = {
            "int": self.gdf.index.to_series().sample(self.gdf.shape[0] // 5).values,
            "str": self.gdf_str.index.to_series()
            .sample(self.gdf_str.shape[0] // 5)
            .values,
        }
        self.sparse_arrays = {k + "_k": v.sparse for k, v in self.graphs.items()}

        self.sparse_arrays = {
            "small_int": Graph.build_knn(self.gdf_points, k=10).sparse,
            "large_int": Graph.build_knn(self.gdf_points, k=500).sparse,
            "queen": Graph.build_contiguity(self.gdf).sparse,
            "rook": Graph.build_contiguity(self.gdf).sparse,
            "delaunay": Graph.build_triangulation(self.gdf).sparse,
            "gabriel": Graph.build_triangulation(self.gdf, method="gabriel").sparse,
            "relneigh": Graph.build_triangulation(
                self.gdf, method="relative_neighborhoood"
            ).sparse,
        }

    def time_queen(self, idx, strict):
        Graph.build_contiguity(
            self.gdf if idx == "int" else self.gdf_str,
            strict=strict,
        )

    time_queen.params = (["int", "str"], [True, False])
    time_queen.param_names = ["index", "strict"]

    def time_knn(self, idx, k):
        Graph.build_knn(self.gdf_points if idx == "int" else self.gdf_str_points, k=k)

    time_knn.params = (["int", "str"], [10, 500])
    time_knn.param_names = ["index", "k"]

    def time_kernel(self, idx):
        Graph.build_kernel(self.gdf_points if idx == "int" else self.gdf_str_points)

    time_kernel.params = ["int", "str"]
    time_kernel.param_names = ["index"]

    def time_assign_self_weight(self, idx, size):
        self.graphs[f"{size}_{idx}"].assign_self_weight()

    time_assign_self_weight.params = (["int", "str"], ["small", "large"])
    time_assign_self_weight.param_names = ["index", "graph_size"]

    def time_sparse(self, idx, size):
        s = self.graphs[f"{size}_{idx}"].sparse

    time_sparse.params = (["int", "str"], ["small", "large"])
    time_sparse.param_names = ["index", "graph_size"]

    def time_subgraph(self, idx, size):
        self.graphs[f"{size}_{idx}"].subgraph(self.ids[idx])

    time_subgraph.params = (["int", "str"], ["small", "large"])
    time_subgraph.param_names = ["index", "graph_size"]

    def time_inverse(self, graph):
        s = self.graphs[f"{graph}"].sparse
        np.linalg.inv(np.eye(s.shape[0]) - 0.5 * s)

    def time_dense_solve(self, graph):
        s = self.graphs[f"{graph}"].sparse
        np.linalg.solve((np.eye(s.shape[0]) - 0.5 * s).todense(), np.arange(w.shape[0]))

    def time_sparse_solve(self, graph):
        s = self.graphs[f"{graph}"].sparse
        spla.spsolve(sp.eye(s.shape[0]) - 0.5 * s, np.arange(w.shape[0]))

    def time_dense_slogdet(self, graph):
        s = self.graphs[f"{graph}"].sparse
        np.linalg.slogdet(np.eye(s.shape[0]) - 0.5 * s)

    def time_sparse_slogdet(self, graph):
        s = self.graphs[f"{graph}"].sparse
        LU = spla.splu(sp.eye(s.shape[0]) - 0.5 * s)
        np.sum(np.log(np.abs(LU.U.diagonal())))


# class MemSuite:
#     def mem_list(self):
#         return [0] * 256

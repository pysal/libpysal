from scipy.spatial import Delaunay as _Delaunay
from scipy import sparse
from numba import njit
from libpysal.weights import W, Rook
import joblib, pandas, pygeos


class Delaunay(W):
    def __init__(self, coordinates, **kwargs):
        edges, _ = self._voronoi_edges(coordinates)
        voronoi_neighbors = pandas.DataFrame(output).groupby(0)[1].apply(list).to_dict()
        W.__init__(self, voronoi_neighbors, **kwargs)

    def _voronoi_edges(self, coordinates):
        dt = _Delaunay(coordinates)
        edges = edges_from_simplices(dt.simplices)
        edges = (
            pandas.DataFrame(numpy.asarray(list(edges)))
            .sort_values([0, 1])
            .drop_duplicates()
            .values
        )
        return edges, dt

    @classmethod
    def from_dataframe(cls, df, **kwargs):
        return cls.__init__(
            pygeos.get_coordinates(pygeos.centroid(df.geometry.data.values))
        )


class Gabriel(Delaunay):
    def __init__(self, coordinates, **kwargs):
        edges, dt = self._voronoi_edges(coordinates)
        droplist = filter_edges(
            edges,
            dt.points,
        )
        output = set(map(tuple, edges)).difference(set(droplist))
        gabriel_neighbors = pandas.DataFrame(output).groupby(0)[1].apply(list).to_dict()
        W.__init__(self, gabriel_neighbors, **kwargs)


@njit
def edges_from_simplices(simplices):
    edges = []
    for simplex in simplices:
        edges.append((simplex[0], simplex[1]))
        edges.append((simplex[1], simplex[0]))
        edges.append((simplex[1], simplex[2]))
        edges.append((simplex[2], simplex[1]))
        edges.append((simplex[2], simplex[0]))
        edges.append((simplex[0], simplex[2]))
    return numpy.asarray(edges)


@njit
def filter_edges(edges, coordinates):
    edge_pointer = 0
    n = edges.max()
    n_edges = len(edges)
    to_drop = []
    while edge_pointer < n_edges:
        edge = edges[edge_pointer]
        cardinality = 0

        for joff in range(edge_pointer, n_edges):
            next_edge = edges[joff]
            if next_edge[0] != edge[0]:
                break
            cardinality += 1
        # let i,j be the diameter of a circle, and k be a third point making a triangle.
        # the right triangle formed when k is on the circle is the limiting case for the
        # gabriel link between ij. If k on the circle, then dij**2 = djk**2 + dki**2,
        # by thales theorem (an inscribed triangle with hypotenuse as diameter is a
        # right triangle.). When dij**2 > djk**2 + dki**2, k is inside of the circle.
        # When dij**2 < dij**2 + djk**2, k is outside of the circle.
        # Therefore, we can take each observation i, iterate over neighbors j,k and remove links
        # where dij**2 > djk**2 + dki**2 to filter a delanuay graph to a gabriel one.
        for ix in range(edge_pointer, edge_pointer + cardinality):
            i, j = edges[ix]
            dij = ((coordinates[i] - coordinates[j]) ** 2).sum() ** 0.5
            for ix2 in range(edge_pointer, edge_pointer + cardinality):
                _, k = edges[ix2]
                if j == k:
                    continue
                dik = ((coordinates[i] - coordinates[k]) ** 2).sum() ** 0.5
                djk = ((coordinates[j] - coordinates[k]) ** 2).sum() ** 0.5

                if dij ** 2 > (dik ** 2 + djk ** 2):
                    to_drop.append((i, j))
                    to_drop.append((j, i))
        edge_pointer += cardinality
    return set(to_drop)


if __name__ == "__main__":
    import libpysal
    import numpy
    import geopandas, pandas
    import pygeos, time

    libpysal.examples.load_example("south.shp")
    df = geopandas.read_file(libpysal.examples.get_path("south.shp"))

    coords = pygeos.get_coordinates(df.geometry.centroid.values.data)
    start1 = time.time()
    voronoi_old = libpysal.weights.Voronoi(coords, clip="extent")
    fin1 = time.time()
    voronoi_old_elapsed = fin1 - start1

    compile_start = time.time()
    edges_from_simplices(numpy.random.randint(0, 4, size=(4, 3)))
    filter_edges(
        numpy.random.randint(0, 4, size=(4, 2)), numpy.random.normal(size=(4, 2))
    )
    compile_finish = time.time()
    compile_time = compile_finish - compile_start

    start2 = time.time()
    dt = _Delaunay(coords)
    edges = edges_from_simplices(dt.simplices)
    edges = (
        pandas.DataFrame(numpy.asarray(list(edges)))
        .sort_values([0, 1])
        .drop_duplicates()
        .values
    )
    droplist = filter_edges(
        edges,
        dt.points,
    )
    output = set(map(tuple, edges)).difference(set(droplist))
    gabriel_neighbors = pandas.DataFrame(output).groupby(0)[1].apply(list).to_dict()
    gabriel_raw = libpysal.weights.W(gabriel_neighbors)
    fin2 = time.time()
    gabriel_raw_elapsed = fin2 - start2

    for focal, gneighbors in gabriel_raw.neighbors.items():
        vneighbors = voronoi_old.neighbors[focal]
        assert set(gneighbors) <= set(vneighbors)

    start3 = time.time()
    delaunay = Delaunay(coords)
    fin3 = time.time()
    voronoi_new_elapsed = fin3 - start3

    start4 = time.time()
    gabriel = Gabriel(coords)
    fin4 = time.time()
    gabriel_elapsed = fin4 - start4
    assert gabriel.n == delaunay.n
    for focal, gneighbors in gabriel.neighbors.items():
        dneighbors = delaunay.neighbors[focal]
        assert set(gneighbors) <= set(dneighbors)
    # for focal, dneighbors in delaunay.neighbors.items():
    #    vold = voronoi_old.neighbors[focal]
    #    assert vold == vneighbors fails due to clipping!
    print(
        f"""
        Old Voronoi: {voronoi_old_elapsed:.2f}
        Raw Gabriel: {gabriel_raw_elapsed:.2f}
        Compile: {compile_time:.2f}
        New Delaunay: {voronoi_new_elapsed:.2f}
        Ref Gabriel: {gabriel_elapsed:.2f}
        """
    )

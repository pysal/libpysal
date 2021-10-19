from scipy.spatial import Delaunay as _Delaunay
from scipy import sparse
from numba import njit
from libpysal.weights import W, WSP
import joblib, pandas, pygeos, numpy

# delaunay graphs and their subgraphs

#### Classes


class Delaunay(W):
    def __init__(self, coordinates, **kwargs):
        edges, _ = self._voronoi_edges(coordinates)
        voronoi_neighbors = pandas.DataFrame(edges).groupby(0)[1].apply(list).to_dict()
        W.__init__(self, voronoi_neighbors, **kwargs)

    def _voronoi_edges(self, coordinates):
        dt = _Delaunay(coordinates)
        edges = _edges_from_simplices(dt.simplices)
        edges = (
            pandas.DataFrame(numpy.asarray(list(edges)))
            .sort_values([0, 1])
            .drop_duplicates()
            .values
        )
        return edges, dt

    @classmethod
    def from_dataframe(cls, df, **kwargs):
        return cls(pygeos.get_coordinates(pygeos.centroid(df.geometry.values.data)))


class Gabriel(Delaunay):
    def __init__(self, coordinates, **kwargs):
        edges, dt = self._voronoi_edges(coordinates)
        droplist = _filter_gabriel(
            edges,
            dt.points,
        )
        output = set(map(tuple, edges)).difference(set(droplist))
        gabriel_neighbors = pandas.DataFrame(output).groupby(0)[1].apply(list).to_dict()
        W.__init__(self, gabriel_neighbors, **kwargs)


class Relative_Neighborhood(Delaunay):
    def __init__(self, coordinates, return_dkmax=False, **kwargs):
        """
        via toussaint, https://doi.org/10.1016/0031-3203(80)90066-7
        """
        edges, dt = self._voronoi_edges(coordinates)
        output, dkmax = _filter_relativehood(
            edges, dt.points, return_dkmax=return_dkmax
        )
        if not return_dkmax:
            del dkmax
        row, col, data = zip(*output)
        sp = sparse.csc_matrix((data, (row, col))) #TODO: faster way than this?
        tmp = WSP(sp).to_W() 
        W.__init__(self, tmp.neighbors, tmp.weights, **kwargs)



#### utilities

@njit
def _edges_from_simplices(simplices):
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
def _filter_gabriel(edges, coordinates):
    edge_pointer = 0
    n = edges.max()
    n_edges = len(edges)
    to_drop = []
    while edge_pointer < n_edges:
        edge = edges[edge_pointer]
        cardinality = 0
        # look ahead to find all neighbors of edge[0]
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
            i, j = edges[ix]  # lookahead ensures that i is always edge[0]
            dij2 = ((coordinates[i] - coordinates[j]) ** 2).sum()
            for ix2 in range(edge_pointer, edge_pointer + cardinality):
                _, k = edges[ix2]
                if j == k:
                    continue
                dik2 = ((coordinates[i] - coordinates[k]) ** 2).sum()
                djk2 = ((coordinates[j] - coordinates[k]) ** 2).sum()

                if dij2 > (dik2 + djk2):
                    to_drop.append((i, j))
                    to_drop.append((j, i))
        edge_pointer += cardinality
    return set(to_drop)


@njit
def _filter_relativehood(edges, coordinates, return_dkmax=False):

    # RNG
    # 1. Compute the delaunay
    # 2. for each edge of the delaunay (i,j), compute
    #    dkmax = max(d(k,i), d(k,j)) for k in 1..n, k != i, j
    # 3. for each edge of the delaunay (i,j), prune
    #    if any dkmax is greater than d(i,j)
    edge_pointer = 0
    n = edges.max()
    n_edges = len(edges)
    out = []
    r = []
    for edge in edges:
        i, j = edge
        pi = coordinates[i]
        pj = coordinates[j]
        dkmax = 0
        dij = ((pi - pj)**2).sum()**.5t 
        prune = False
        for k in range(n):
            pk = coordinates[k]
            dik = ((pi - pk)**2).sum()**.5
            djk = ((pj - pk)**2).sum()**.5
            distances = numpy.array([dik, djk, dkmax])
            dkmax = distances.max()
            prune = dkmax < dij
            if (not return_dkmax) & prune:
                break
        if prune:
            continue
        out.append((i, j, dij))
        if return_dkmax:
            r.append(dkmax)

    return out, r


class Mutual_Reachability(W):
    def __init__(
        self, coordinates, n_max=None, n_core=5, method="kdtree", metric="euclidean"
    ):
        # check to see if metric is supported by method using tree.valid_metrics
        # so, kdtree can only do minkowski
        # balltree can handle more
        # approx can handle a ton, but wait to import pynndescent unless method = approx
        # raise an error if the requested metric is not supported

        # then, compute the core distances for each point
        # finally, for all pairs of points, compute the distances for the nearest n_max
        # points (possibly all points), and censor that by the core distance.
        raise NotImplementedError()


if __name__ == "__main__":
    import numpy, subprocess, geopandas, pandas, pygeos, time, libpysal, os
    from rpy2.robjects import r

    libpysal.examples.load_example("south.shp")
    path = libpysal.examples.get_path("south.shp")
    df = geopandas.read_file(path)

    coords = pygeos.get_coordinates(df.geometry.centroid.values.data)
    start1 = time.time()
    voronoi_old = libpysal.weights.Voronoi(coords, clip="extent")
    fin1 = time.time()
    voronoi_old_elapsed = fin1 - start1

    compile_start = time.time()
    _edges_from_simplices(numpy.random.randint(0, 4, size=(4, 3)))
    _filter_gabriel(
        numpy.random.randint(0, 4, size=(4, 2)), numpy.random.normal(size=(4, 2))
    )
    compile_finish = time.time()
    compile_time = compile_finish - compile_start

    start2 = time.time()
    dt = _Delaunay(coords)
    edges = _edges_from_simplices(dt.simplices)
    edges = (
        pandas.DataFrame(numpy.asarray(list(edges)))
        .sort_values([0, 1])
        .drop_duplicates()
        .values
    )
    droplist = _filter_gabriel(
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

    location = os.path.dirname(__file__)
    with open(f"{location}/gabriel.R", "w") as f:
        f.writelines(
            [
                "library(sf)\n"
                "library(spdep)\n"
                "library(dplyr)\n"
                f"path <- '{path}'\n"
                "df <- st_read(path)\n"
                "network <- df %>% st_centroid() %>% st_coordinates() %>%  gabrielneigh()\n"
                f'network %>% graph2nb(sym=T) %>% write.nb.gal("{location}/gabriel_r.gal")\n'
            ]
        )
    subprocess.call(["/usr/local/bin/Rscript", "--vanilla", f"{location}/gabriel.R"])

    gabriel_r = W.from_file(f"{location}/gabriel_r.gal")

    for i, neighbors in gabriel.neighbors.items():
        iname = str(i + 1)
        nnames = [str(n + 1) for n in neighbors]
        r_neighbors = gabriel_r.neighbors[iname]
        assert set(r_neighbors) == set(nnames)

    os.remove(f"{location}/gabriel_r.gal")
    os.remove(f"{location}/gabriel.R")



    Relative_Neighborhood(coords)

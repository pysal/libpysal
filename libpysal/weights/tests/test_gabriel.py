from .. import gabriel
from ... import examples
import geopandas, numpy

path = examples.get_path("columbus.shp")
df = geopandas.read_file(path)
geoms = df.geometry.centroid
coords = numpy.column_stack((geoms.x, geoms.y))


def test_delaunay():
    a = gabriel.Delaunay(coords)
    b = gabriel.Delaunay.from_dataframe(df)

    assert a.neighbors == b.neighbors

    assert a[13] == {6: 1, 11: 1, 12: 1, 18: 1, 20: 1}


def test_gabriel():
    c = gabriel.Gabriel(coords)
    d = gabriel.Gabriel.from_dataframe(df)
    c2 = gabriel.Delaunay(coords)

    assert c.neighbors == d.neighbors

    assert c[13] == {12: 1, 18: 1}
    for focal, neighbors in c.neighbors.items():
        dneighbors = c2[focal]
        assert set(neighbors) <= set(dneighbors)

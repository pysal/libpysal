from ..shapes import Point, Chain, asShape
from ...io.fileio import FileIO as psopen
from ... import examples as pysal_examples


class Testtest_MultiPloygon:
    def test___init__1(self):
        """Tests conversion of polygons with multiple shells to
        geoJSON multipolygons and back.

        """

        ncovr = pysal_examples.load_example("NCOVR")
        shp = psopen(pysal_examples.get_path("NAT.shp"), "r")
        multipolygons = [p for p in shp if len(p.parts) > 1]
        geoJSON = [p.__geo_interface__ for p in multipolygons]
        for poly in multipolygons:
            json = poly.__geo_interface__
            shape = asShape(json)
            assert json["type"] == "MultiPolygon"
            assert str(shape.holes) == str(poly.holes)
            assert str(shape.parts) == str(poly.parts)


class Testtest_MultiLineString:
    def test_multipart_chain(self):
        vertices = [
            [Point((0, 0)), Point((1, 0)), Point((1, 5))],
            [Point((-5, -5)), Point((-5, 0)), Point((0, 0))],
        ]

        # part A
        chain0 = Chain(vertices[0])
        # part B
        chain1 = Chain(vertices[1])
        # part A and B
        chain2 = Chain(vertices)

        json = chain0.__geo_interface__
        assert json["type"] == "LineString"
        assert len(json["coordinates"]) == 3

        json = chain1.__geo_interface__
        assert json["type"] == "LineString"
        assert len(json["coordinates"]) == 3

        json = chain2.__geo_interface__
        assert json["type"] == "MultiLineString"
        assert len(json["coordinates"]) == 2

        chain3 = asShape(json)
        assert chain2.parts == chain3.parts

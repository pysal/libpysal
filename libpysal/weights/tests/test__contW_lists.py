# ruff: noqa: N999

import os

import geopandas as gpd

from ... import examples as pysal_examples
from ...io.fileio import FileIO
from .._contW_lists import QUEEN, ROOK, ContiguityWeightsLists
from ..weights import W


class TestContiguityWeights:
    def setup_method(self):
        """Setup the binning contiguity weights"""
        shp_obj = FileIO(pysal_examples.get_path("virginia.shp"), "r")
        self.binningW = ContiguityWeightsLists(shp_obj, QUEEN)
        shp_obj.close()

    def test_w_type(self):
        assert isinstance(self.binningW, ContiguityWeightsLists)

    def test_queen(self):
        assert QUEEN == 1

    def test_rook(self):
        assert ROOK == 2

    def test_contiguity_weights_lists(self):
        assert hasattr(self.binningW, "w")
        assert issubclass(dict, type(self.binningW.w))
        assert len(self.binningW.w) == 136

    def test_nested_polygons(self):
        # load queen gal file created using Open Geoda.
        geoda_w = FileIO(pysal_examples.get_path("virginia.gal"), "r").read()
        # build matching W with pysal
        pysal_wb = self.build_w(
            pysal_examples.get_path("virginia.shp"), QUEEN, "POLY_ID"
        )
        # compare output.
        for key in geoda_w.neighbors:
            geoda_neighbors = list(map(int, geoda_w.neighbors[key]))
            pysalb_neighbors = pysal_wb.neighbors[int(key)]
            geoda_neighbors.sort()
            pysalb_neighbors.sort()
            assert geoda_neighbors == pysalb_neighbors

    def test_true_rook(self):
        # load queen gal file created using Open Geoda.
        geoda_w = FileIO(pysal_examples.get_path("rook31.gal"), "r").read()
        # build matching W with pysal
        # pysalW = pysal.rook_from_shapefile(
        #   pysal_examples.get_path('rook31.shp'),','POLY_ID'
        # )
        pysal_wb = self.build_w(pysal_examples.get_path("rook31.shp"), ROOK, "POLY_ID")
        # compare output.
        for key in geoda_w.neighbors:
            geoda_neighbors = list(map(int, geoda_w.neighbors[key]))
            pysalb_neighbors = pysal_wb.neighbors[int(key)]
            geoda_neighbors.sort()
            pysalb_neighbors.sort()
            assert geoda_neighbors == pysalb_neighbors

    def test_true_rook2(self):
        # load queen gal file created using Open Geoda.

        stl = pysal_examples.load_example("stl")
        gal_file = stl.get_path("stl_hom_rook.gal")
        geoda_w = FileIO(gal_file, "r").read()
        # build matching W with pysal
        pysal_wb = self.build_w(stl.get_path("stl_hom.shp"), ROOK, "POLY_ID_OG")
        # compare output.
        for key in geoda_w.neighbors:
            geoda_neighbors = list(map(int, geoda_w.neighbors[key]))
            pysalb_neighbors = pysal_wb.neighbors[int(key)]
            geoda_neighbors.sort()
            pysalb_neighbors.sort()
            assert geoda_neighbors == pysalb_neighbors

    def test_true_rook3(self):
        # load queen gal file created using Open Geoda.
        geoda_w = FileIO(pysal_examples.get_path("virginia_rook.gal"), "r").read()
        # build matching W with pysal
        pysal_wb = self.build_w(
            pysal_examples.get_path("virginia.shp"), ROOK, "POLY_ID"
        )
        # compare output.
        for key in geoda_w.neighbors:
            geoda_neighbors = list(map(int, geoda_w.neighbors[key]))
            pysalb_neighbors = pysal_wb.neighbors[int(key)]
            geoda_neighbors.sort()
            pysalb_neighbors.sort()
            assert geoda_neighbors == pysalb_neighbors

    def test_shapely(self):
        pysalneighbs = ContiguityWeightsLists(
            FileIO(pysal_examples.get_path("virginia.shp")), ROOK
        )
        gdf = gpd.read_file(pysal_examples.get_path("virginia.shp"))
        shplyneighbs = ContiguityWeightsLists(gdf.geometry.tolist(), ROOK)
        assert pysalneighbs.w == shplyneighbs.w
        pysalneighbs = ContiguityWeightsLists(
            FileIO(pysal_examples.get_path("virginia.shp")), QUEEN
        )
        shplyneighbs = ContiguityWeightsLists(gdf.geometry.tolist(), QUEEN)
        assert pysalneighbs.w == shplyneighbs.w

    def build_w(self, shapefile, type_, idVariable=None):  # noqa: N803
        """Building 2 W's the hard way. We need to do this so we
        can test both rtree and binning
        """
        dbname = os.path.splitext(shapefile)[0] + ".dbf"
        db = FileIO(dbname)
        shp_obj = FileIO(shapefile)
        neighbor_data = ContiguityWeightsLists(shp_obj, type_).w
        neighbors = {}
        if idVariable:
            ids = db.by_col[idVariable]
            assert len(ids) == len(set(ids))
            for key in neighbor_data:
                id_ = ids[key]
                if id_ not in neighbors:
                    neighbors[id_] = set()
                neighbors[id_].update([ids[x] for x in neighbor_data[key]])
            for key in neighbors:
                neighbors[key] = list(neighbors[key])
            binning_w = W(neighbors, id_order=ids)
        else:
            neighbors[key] = list(neighbors[key])
            binning_w = W(neighbors)
        return binning_w

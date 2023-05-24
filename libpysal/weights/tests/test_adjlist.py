import unittest as ut
import numpy as np
from .. import adjtools as adj
from ... import weights
from ... import io
from ... import examples
from ..util import lat2W
from ...common import RTOL, ATOL


try:
    import pandas
    import geopandas

    PANDAS_MISSING = False
except ImportError:
    PANDAS_MISSING = True


@ut.skipIf(PANDAS_MISSING, "Pandas is gone")
class Test_Adjlist(ut.TestCase):
    def setUp(self):
        self.knownW = io.open(examples.get_path("columbus.gal")).read()

    def test_round_trip_drop_islands_true(self):
        adjlist = self.knownW.to_adjlist(
            remove_symmetric=False, drop_islands=True
        ).astype(int)
        w_from_adj = weights.W.from_adjlist(adjlist)
        np.testing.assert_allclose(
            w_from_adj.sparse.toarray(), self.knownW.sparse.toarray()
        )

    def test_round_trip_drop_islands_false(self):
        adjlist = self.knownW.to_adjlist(
            remove_symmetric=False, drop_islands=True
        ).astype(int)
        w_from_adj = weights.W.from_adjlist(adjlist)
        np.testing.assert_allclose(
            w_from_adj.sparse.toarray(), self.knownW.sparse.toarray()
        )

    def test_filter(self):
        grid = lat2W(2, 2)
        alist = grid.to_adjlist(remove_symmetric=True, drop_islands=True)
        assert len(alist) == 4
        with self.assertRaises(AssertionError):
            # build this manually because of bug libpysal#322
            alist_neighbors = alist.groupby("focal").neighbor.apply(list).to_dict()
            all_ids = set(alist_neighbors.keys()).union(
                *map(set, alist_neighbors.values())
            )
            for idx in set(all_ids).difference(set(alist_neighbors.keys())):
                alist_neighbors[idx] = []
            badgrid = weights.W(alist_neighbors)
            np.testing.assert_allclose(badgrid.sparse.toarray(), grid.sparse.toarray())
        assert set(alist.focal.unique()) == {0, 1, 2}
        assert set(alist.neighbor.unique()) == {1, 2, 3}
        assert alist.weight.unique().item() == 1
        grid = lat2W(2, 2, id_type="string")
        alist = grid.to_adjlist(remove_symmetric=True, drop_islands=True)
        assert len(alist) == 4
        with self.assertRaises(AssertionError):
            # build this manually because of bug libpysal#322
            alist_neighbors = alist.groupby("focal").neighbor.apply(list).to_dict()
            all_ids = set(alist_neighbors.keys()).union(
                *map(set, alist_neighbors.values())
            )
            for idx in set(all_ids).difference(set(alist_neighbors.keys())):
                alist_neighbors[idx] = []
            badgrid = weights.W(alist_neighbors)
            np.testing.assert_allclose(badgrid.sparse.toarray(), grid.sparse.toarray())
        tuples = set([tuple(t) for t in alist[["focal", "neighbor"]].values])
        full_alist = grid.to_adjlist(drop_islands=True)
        all_possible = set([tuple(t) for t in full_alist[["focal", "neighbor"]].values])
        assert tuples.issubset(all_possible), (
            "the de-duped adjlist has links " "not in the duplicated adjlist."
        )
        complements = all_possible.difference(tuples)
        reversed_complements = set([t[::-1] for t in complements])
        assert reversed_complements == tuples, (
            "the remaining links in the duplicated"
            " adjlist are not the reverse of the links"
            " in the deduplicated adjlist."
        )
        assert alist.weight.unique().item() == 1

    def apply_and_compare_columbus(self, col):
        import geopandas

        df = geopandas.read_file(examples.get_path("columbus.dbf")).head()
        W = weights.Queen.from_dataframe(df)
        alist = adj.adjlist_apply(df[col], W=W, to_adjlist_kws=dict(drop_islands=True))
        right_hovals = alist.groupby("focal").att_focal.unique()
        assert (right_hovals == df[col]).all()
        allpairs = np.subtract.outer(df[col].values, df[col].values)
        flat_diffs = allpairs[W.sparse.toarray().astype(bool)]
        np.testing.assert_allclose(flat_diffs, alist["subtract"].values)
        return flat_diffs

    def test_apply(self):
        self.apply_and_compare_columbus("HOVAL")

    def test_mvapply(self):
        import geopandas

        df = geopandas.read_file(examples.get_path("columbus.dbf")).head()
        W = weights.Queen.from_dataframe(df)

        ssq = lambda x_y: np.sum((x_y[0] - x_y[1]) ** 2).item()
        ssq.__name__ = "sum_of_squares"
        alist = adj.adjlist_apply(
            df[["HOVAL", "CRIME", "INC"]],
            W=W,
            func=ssq,
            to_adjlist_kws=dict(drop_islands=True),
        )
        known_ssq = [
            1301.1639302990804,
            3163.46450914361,
            1301.1639302990804,
            499.52656498472993,
            594.518273032036,
            3163.46450914361,
            499.52656498472993,
            181.79100173844196,
            436.09336916344097,
            594.518273032036,
            181.79100173844196,
            481.89443401250094,
            436.09336916344097,
            481.89443401250094,
        ]  # ugh I hate doing this, but how else?
        np.testing.assert_allclose(
            alist.sum_of_squares.values, np.asarray(known_ssq), rtol=RTOL, atol=ATOL
        )

    def test_map(self):
        atts = ["HOVAL", "CRIME", "INC"]
        df = geopandas.read_file(examples.get_path("columbus.dbf")).head()
        W = weights.Queen.from_dataframe(df)
        hoval, crime, inc = list(map(self.apply_and_compare_columbus, atts))
        mapped = adj.adjlist_map(df[atts], W=W, to_adjlist_kws=dict(drop_islands=True))
        for name, data in zip(atts, (hoval, crime, inc)):
            np.testing.assert_allclose(
                data, mapped["_".join(("subtract", name))].values
            )

    def test_sort(self):
        from libpysal import examples
        from libpysal.weights import Rook

        us = geopandas.read_file(examples.get_path("us48.shp"))
        w = Rook.from_dataframe(us.set_index("STATE_FIPS"), use_index=True)
        unsorted_al = w.to_adjlist(sort_joins=False)
        sorted_al = w.to_adjlist(sort_joins=True)
        sv = ["01"] * 4
        sv.append("04")
        sv = np.array(sv)
        usv = np.array(["53", "53", "30", "30", "30"])
        np.testing.assert_array_equal(unsorted_al.focal.values[:5], usv)
        np.testing.assert_array_equal(sorted_al.focal.values[:5], sv)

    def test_ids(self):
        df = geopandas.read_file(examples.get_path("columbus.dbf")).head()
        df["my_id"] = range(3, len(df) + 3)
        W = weights.Queen.from_dataframe(df, ids="my_id")
        W_adj = W.to_adjlist(drop_islands=True)
        for i in range(3, 8):
            assert i in W_adj.focal
            assert i in W_adj.neighbor
        for i in W_adj.focal:
            assert i in list(range(3, len(df) + 3))
        for i in W_adj.neighbor:
            assert i in list(range(3, len(df) + 3))

    def test_str_ids(self):
        df = geopandas.read_file(examples.get_path("columbus.dbf")).head()
        snakes = ["mamba", "boa", "python", "rattlesnake", "cobra"]
        df["my_str_id"] = snakes
        W = weights.Queen.from_dataframe(df, ids="my_str_id")
        W_adj = W.to_adjlist(drop_islands=True)
        for i in snakes:
            (W_adj.focal == i).any()
            (W_adj.neighbor == i).any()
        for i in W_adj.focal:
            assert i in snakes
        for i in W_adj.neighbor:
            assert i in snakes

    def test_lat2w(self):
        w = lat2W(5, 5)
        manual_neighbors = w.to_adjlist().groupby("focal").neighbor.agg(list).to_dict()
        for focal, neighbors in w.neighbors.items():
            self.assertEqual(set(manual_neighbors[focal]), set(neighbors))

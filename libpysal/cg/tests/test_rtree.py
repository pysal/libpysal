"""pyrtree Unittest."""

from ..rtree import Rect, RTree


class TestPyrtree:
    def setup_method(self):
        k = 10
        w = 20
        objects = {}
        id_ = 0
        for i in range(k):
            mn_y = i * w
            mx_y = mn_y + w
            for j in range(k):
                mn_x = j * w
                mx_x = mn_x + w
                objects[id_] = Rect(mn_x, mn_y, mx_x, mx_y)
                id_ += 1
        self.objects = objects

    def test_rtree(self):
        t = RTree()
        for object_ in self.objects:
            t.insert(object_, self.objects[object_])
        assert len(self.objects) == 100

        qr = Rect(5, 5, 25, 25)

        # find objects with mbrs intersecting with qr
        res = [r.leaf_obj() for r in t.query_rect(qr) if r.is_leaf()]
        assert len(res) == 4
        res.sort()
        assert res == [0, 1, 10, 11]

        # vertices are shared by all coincident rectangles
        res = [r.leaf_obj() for r in t.query_point((20.0, 20.0)) if r.is_leaf()]
        assert len(res) == 4

        res = [r.leaf_obj() for r in t.query_point((21, 20)) if r.is_leaf()]
        assert len(res) == 2

        # single internal point
        res = [r.leaf_obj() for r in t.query_point((21, 21)) if r.is_leaf()]
        assert len(res) == 1

        # single external point
        res = [r.leaf_obj() for r in t.query_point((-12, 21)) if r.is_leaf()]
        assert len(res) == 0

        qr = Rect(5, 6, 65, 7)

        res = [r.leaf_obj() for r in t.query_rect(qr) if r.is_leaf()]
        assert len(res) == 4

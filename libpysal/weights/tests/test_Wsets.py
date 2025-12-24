"""Unit test for set_operations module."""

# ruff: noqa: N999

import numpy as np

from .. import set_operations
from ..util import block_weights, lat2W


class TestSetOperations:
    """Unit test for set_operations module."""

    def test_w_union(self):
        """Unit test"""
        w1 = lat2W(4, 4)
        w2 = lat2W(6, 4)
        w3 = set_operations.w_union(w1, w2)
        assert w1[0] == w3[0]
        assert set(w1.neighbors[15]) == {11, 14}
        assert set(w2.neighbors[15]) == {11, 14, 19}
        assert set(w3.neighbors[15]) == {19, 11, 14}

    def test_w_intersection(self):
        """Unit test"""
        w1 = lat2W(4, 4)
        w2 = lat2W(6, 4)
        w3 = set_operations.w_union(w1, w2)
        assert w1[0] == w3[0]
        assert set(w1.neighbors[15]) == {11, 14}
        assert set(w2.neighbors[15]) == {11, 14, 19}
        assert set(w3.neighbors[15]) == {19, 11, 14}

    def test_w_difference(self):
        """Unit test"""
        w1 = lat2W(4, 4, rook=False)
        w2 = lat2W(4, 4, rook=True)
        w3 = set_operations.w_difference(w1, w2, constrained=False)
        assert w1[0] != w3[0]
        assert set(w1.neighbors[15]) == {10, 11, 14}
        assert set(w2.neighbors[15]) == {11, 14}
        assert set(w3.neighbors[15]) == {10}

    def test_w_symmetric_difference(self):
        """Unit test"""
        w1 = lat2W(4, 4, rook=False)
        w2 = lat2W(6, 4, rook=True)
        w3 = set_operations.w_symmetric_difference(w1, w2, constrained=False)
        assert w1[0] != w3[0]
        assert set(w1.neighbors[15]) == {10, 11, 14}
        assert set(w2.neighbors[15]) == {11, 14, 19}
        assert set(w3.neighbors[15]) == {10, 19}

    def test_w_subset(self):
        """Unit test"""
        w1 = lat2W(6, 4)
        ids = list(range(16))
        w2 = set_operations.w_subset(w1, ids)
        assert w1[0] == w2[0]
        assert set(w1.neighbors[15]) == {11, 14, 19}
        assert set(w2.neighbors[15]) == {11, 14}

    def test_w_clip(self):
        """Unit test for w_clip"""
        w1 = lat2W(3, 2, rook=False)
        w1.transform = "R"
        w2 = block_weights(["r1", "r2", "r1", "r1", "r1", "r2"])
        w2.transform = "R"
        wcs = set_operations.w_clip(w1, w2, outSP=True)
        expected_wcs = np.array(
            [
                [0.0, 0.0, 0.33333333, 0.33333333, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.2, 0.0, 0.0, 0.2, 0.2, 0.0],
                [0.2, 0.0, 0.2, 0.0, 0.2, 0.0],
                [0.0, 0.0, 0.33333333, 0.33333333, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        np.testing.assert_array_equal(
            np.around(wcs.sparse.toarray(), decimals=8), expected_wcs
        )

        wc = set_operations.w_clip(w1, w2, outSP=False)
        np.testing.assert_array_equal(wcs.sparse.toarray(), wc.full()[0])

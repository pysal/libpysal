import numpy as np
import pytest

from ..util import lat2W
from ..weights import W

networkx = pytest.importorskip("networkx")


class TestNetworkXConverter:
    def setup_method(self):
        self.known_nx = networkx.random_regular_graph(4, 10, seed=8879)
        self.known_amat = networkx.to_numpy_array(self.known_nx)
        self.known_W = lat2W(5, 5)

    def test_round_trip(self):
        w_ = W.from_networkx(self.known_nx)
        np.testing.assert_allclose(w_.sparse.toarray(), self.known_amat)
        nx2 = w_.to_networkx()
        np.testing.assert_allclose(networkx.to_numpy_array(nx2), self.known_amat)
        nxsquare = self.known_W.to_networkx()
        np.testing.assert_allclose(
            self.known_W.sparse.toarray(), networkx.to_numpy_array(nxsquare)
        )
        w_square = W.from_networkx(nxsquare)
        np.testing.assert_allclose(
            self.known_W.sparse.toarray(), w_square.sparse.toarray()
        )

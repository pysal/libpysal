import unittest as ut
import numpy as np
import scipy as sp
from packaging.version import Version
from ..util import lat2W
from ..weights import W

try:
    import networkx as nx
except ImportError:
    nx = None

SP_18 = Version(sp.__version__) >= Version('1.8')

@ut.skipIf((nx is None) or (not SP_18), "Missing networkx or old scipy")
class Test_NetworkXConverter(ut.TestCase):
    def setUp(self):
        self.known_nx = nx.random_regular_graph(4, 10, seed=8879)
        self.known_amat = nx.to_numpy_array(self.known_nx)
        self.known_W = lat2W(5, 5)

    def test_round_trip(self):
        W_ = W.from_networkx(self.known_nx)
        np.testing.assert_allclose(W_.sparse.toarray(), self.known_amat)
        nx2 = W_.to_networkx()
        np.testing.assert_allclose(nx.to_numpy_array(nx2), self.known_amat)
        nxsquare = self.known_W.to_networkx()
        np.testing.assert_allclose(
            self.known_W.sparse.toarray(), nx.to_numpy_array(nxsquare)
        )
        W_square = W.from_networkx(nxsquare)
        np.testing.assert_allclose(
            self.known_W.sparse.toarray(), W_square.sparse.toarray()
        )

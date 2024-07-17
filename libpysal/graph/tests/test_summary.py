import geodatasets
import geopandas
import numpy as np
import pytest

from libpysal import graph


@pytest.mark.network
class TestSummary:
    def setup_method(self):
        self.nybb = geopandas.read_file(geodatasets.get_path("nybb"))
        self.G = graph.Graph.build_contiguity(self.nybb)
        self.summary = self.G.summary(True)
        self.no_asymmetries = self.G.summary()

    def test_exactness(self):
        assert self.summary.n_nodes == 5
        assert self.summary.n_edges == 10
        assert self.summary.n_components == 2
        assert self.summary.n_isolates == 1
        assert self.summary.nonzero == 10
        assert self.summary.pct_nonzero == 44.0
        assert self.summary.n_asymmetries == 0
        assert self.summary.cardinalities_mean == 2.0
        assert self.summary.cardinalities_std == 1.224744871391589
        assert self.summary.cardinalities_min == 0
        assert self.summary.cardinalities_25 == 2
        assert self.summary.cardinalities_50 == 2
        assert self.summary.cardinalities_75 == 3
        assert self.summary.cardinalities_max == 3
        assert self.summary.weights_mean == 1
        assert self.summary.weights_std == 0
        assert self.summary.weights_min == 1
        assert self.summary.weights_25 == 1
        assert self.summary.weights_50 == 1
        assert self.summary.weights_75 == 1
        assert self.summary.weights_max == 1
        assert self.summary.s0 == 10
        assert self.summary.s1 == 20.0
        assert self.summary.s2 == 104.0
        np.testing.assert_array_equal(np.array([0, 3, 2, 3, 2]), self.summary.diag_g2)
        np.testing.assert_array_equal(np.array([0, 3, 2, 3, 2]), self.summary.diag_gtg)
        np.testing.assert_array_equal(
            np.array([0, 6, 4, 6, 4]), self.summary.diag_gtg_gg
        )
        assert self.summary.trace_g2 == 10.0
        assert self.summary.trace_gtg == 10.0
        assert self.summary.trace_gtg_gg == 20.0

    def test_repr(self):
        expected = """Graph Summary Statistics
========================
Graph indexed by:
 [0, 1, 2, 3, 4]
==============================================================
Number of nodes:                                             5
Number of edges:                                            10
Number of connected components:                              2
Number of isolates:                                          1
Number of non-zero edges:                                   10
Percentage of non-zero edges:                           44.00%
Number of asymmetries:                                       0
--------------------------------------------------------------
Cardinalities
==============================================================
Mean:                       2    25%:                        2
Standard deviation:         1    50%:                        2
Min:                        0    75%:                        3
Max:                        3
--------------------------------------------------------------
Weights
==============================================================
Mean:                       1    25%:                        1
Standard deviation:         0    50%:                        1
Min:                        1    75%:                        1
Max:                        1
--------------------------------------------------------------
Sum of weights
==============================================================
S0:                                                         10
S1:                                                         20
S2:                                                        104
--------------------------------------------------------------
Traces
==============================================================
GG:                                                         10
G'G:                                                        10
G'G + GG:                                                   20
"""
        assert self.summary.__repr__() == expected

    def test_html_repr(self):
        expected = """
            <table>
                <caption>Graph Summary Statistics</caption>
                <tr>
                    <td>Number of nodes:</td>
                    <td>           5</td>
                </tr>
                <tr>
                    <td>Number of edges:</td>
                    <td>          10</td>
                </tr>
                <tr>
                    <td>Number of connected components:</td>
                    <td>           2</td>
                </tr>
                <tr>
                    <td>Number of isolates:</td>
                    <td>           1</td>
                </tr>
                <tr>
                    <td>Number of non-zero edges:</td>
                    <td>          10</td>
                </tr>
                <tr>
                    <td>Percentage of non-zero edges:</td>
                    <td>      44.00%</td>
                </tr>
                <tr>
                    <td>Number of asymmetries:</td>
                    <td>           0</td>
                </tr>
            </table>
            <table>
                <caption>Cardinalities</caption>
                <tr>
                    <td>Mean:</td>
                    <td>        2</td>
                    <td>25%:</td>
                    <td>        2</td>
                </tr>
                <tr>
                    <td>Standard deviation:</td>
                    <td>        1</td>
                    <td>50%</td>
                    <td>        2
                    <td>
                </tr>
                <tr>
                    <td>Min:</td>
                    <td>        0</td>
                    <td>75%:</td>
                    <td>        3</td>
                </tr>
                <tr>
                    <td>Max:</td>
                    <td>        3</td>
                </tr>
            </table>
            <table>
                <caption>Weights</caption>
                <tr>
                    <td>Mean:</td>
                    <td>        1</td>
                    <td>25%:</td>
                    <td>        1</td>
                </tr>
                <tr>
                    <td>Standard deviation:</td>
                    <td>        0</td>
                    <td>50%</td>
                    <td>        1
                <tr>
                    <td>Min:</td>
                    <td>        1</td>
                    <td>75%:</td>
                    <td>        1</td>
                </tr>
                <tr>
                    <td>Max:</td>
                    <td>        1</td>
                </tr>
            </table>
            <table>
                <caption>Sum of weights and Traces</caption>
                <tr>
                    <td>S0:</td>
                    <td>          10</td>
                    <td>GG:</td>
                    <td>          10</td>
                </tr>
                <tr>
                    <td>S1:</td>
                    <td>          20</td>
                    <td>G'G:</td>
                    <td>          10</td>
                </tr>
                <tr>
                    <td>S3:</td>
                    <td>         104</td>
                    <td>G'G + GG:</td>
                    <td>          20</td>
                </tr>
            </table>
            <div>
                Graph indexed by: <code>[0, 1, 2, 3, 4]</code>
            </div>
            """

        assert self.summary._repr_html_() == expected

    def test_no_asymmetries(self):
        assert not hasattr(self.no_asymmetries, "n_asymmetries")
        _ = self.no_asymmetries.__repr__()
        _ = self.no_asymmetries._repr_html_()

import numpy as np


class GraphSummary:
    r"""Graph Summary

    An object containing the statistical attributes summarising the Graph and its basic
    properties.

    Attributes
    ----------
    n_nodes : int
        number of Graph nodes
    n_edges : int
        number of Graph edges
    n_components : int
        number of connected components
    n_isolates : int
        number of isolates (nodes with no neighbors)
    nonzero : int
        number of edges with nonzero weight
    pct_nonzero : float
        percentage of nonzero weights
    n_asymmetries : int
        number of intrinsic asymmetries
    cardinalities_mean : float
        mean number of neighbors
    cardinalities_std : float
        standard deviation of number of neighbors
    cardinalities_min : float
        minimal number of neighbors
    cardinalities_25 : float
        25th percentile of number of neighbors
    cardinalities_50 : float
        50th percentile (median) of number of neighbors
    cardinalities_75 : float
        75th percentile of number of neighbors
    cardinalities_max : float
        maximal number of neighbors
    weights_mean : float
        mean edge weight
    weights_std : float
        standard deviation of  edge weights
    weights_min : float
        minimal edge weight
    weights_25 : float
        25th percentile of edge weights
    weights_50 : float
        50th percentile (median) of edge weights
    weights_75 : float
        75th percentile of edge weights
    weights_max : float
        maximal edge weight
    s0 : float
        S0 (global) sum of weights

        ``s0`` is defined as

        .. math::
               s0=\sum_i \sum_j w_{i,j}

        :attr:`s0`, :attr:`s1`, and :attr:`s2` reflect interaction between observations
        and are used to compute standard errors for spatial autocorrelation estimators.


    s1 : float
        S1 sum of weights

        ``s1`` is defined as

        .. math::
               s1=1/2 \sum_i \sum_j \Big(w_{i,j} + w_{j,i}\Big)^2

        :attr:`s0`, :attr:`s1`, and :attr:`s2` reflect interaction between observations
        and are used to compute standard errors for spatial autocorrelation estimators.

    s2 : float
        S2 sum of weights

        ``s2`` is defined as

        .. math::
                s2=\sum_j \Big(\sum_i w_{i,j} + \sum_i w_{j,i}\Big)^2

        :attr:`s0`, :attr:`s1`, and :attr:`s2` reflect interaction between observations
        and are used to compute standard errors for spatial autocorrelation estimators.

    diag_g2 : np.ndarray
        diagonal of :math:`GG`
    diag_gtg : np.ndarrray
        diagonal of :math:`G^{'}G`
    diag_gtg_gg : np.ndarray
        diagonal of :math:`G^{'}G + GG`
    trace_g2 : np.ndarray
        trace of :math:`GG`
    trace_gtg : np.ndarrray
        trace of :math:`G^{'}G`
    trace_gtg_gg : np.ndarray
        trace of :math:`G^{'}G + GG`

    Examples
    --------
    >>> import geopandas as gpd
    >>> from geodatasets import get_path
    >>> nybb = gpd.read_file(get_path("nybb")).set_index("BoroName")
    >>> nybb
                    BoroCode  ...                                           geometry
    BoroName                 ...
    Staten Island         5  ...  MULTIPOLYGON (((970217.022 145643.332, 970227....
    Queens                4  ...  MULTIPOLYGON (((1029606.077 156073.814, 102957...
    Brooklyn              3  ...  MULTIPOLYGON (((1021176.479 151374.797, 102100...
    Manhattan             1  ...  MULTIPOLYGON (((981219.056 188655.316, 980940....
    Bronx                 2  ...  MULTIPOLYGON (((1012821.806 229228.265, 101278...
    [5 rows x 4 columns]

    >>> contiguity = graph.Graph.build_contiguity(nybb)
    >>> contiguity
    <Graph of 5 nodes and 10 nonzero edges indexed by
        ['Staten Island', 'Queens', 'Brooklyn', 'Manhattan', 'Bronx']>

    >>> summary = contiguity.summary(asymmetries=True)
    >>> summary
    Graph Summary Statistics
    ========================
    Graph indexed by:
    ['Staten Island', 'Queens', 'Brooklyn', 'Manhattan', 'Bronx']
    ==============================================================
    number of nodes:                                             5
    number of edges:                                            10
    number of connected components:                              2
    number of isolates:                                          1
    number of non-zero edges:                                   10
    Percentage of non-zero edges:                           44.00%
    number of asymmetries:                                       0
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

    >>> summary.s1
    20
    """

    def __init__(self, graph, asymmetries=False):
        """Create GraphSummary

        Parameters
        ----------
        graph : Graph
        asymmetries : bool
            whether to compute ``n_asymmetries``, which is considerably more expensive
            than the other attributes. By default False.
        """
        self._graph = graph
        self.asymmetries = asymmetries

        self.n_nodes = self._graph.n_nodes  # number of nodes / observations
        self.n_edges = self._graph.n_edges  # number of edges excluding isolates
        self.n_components = self._graph.n_components
        self.n_isolates = len(self._graph.isolates)

        # nonzero
        self.nonzero = self._graph.nonzero
        self.pct_nonzero = self._graph.pct_nonzero

        # intrinsic assymetries
        if asymmetries:
            self.n_asymmetries = len(self._graph.asymmetry())

        # statistics of cardinalities
        card_stats = self._graph.cardinalities.describe()
        self.cardinalities_mean = card_stats["mean"]
        self.cardinalities_std = card_stats["std"]
        self.cardinalities_min = card_stats["min"]
        self.cardinalities_25 = card_stats["25%"]
        self.cardinalities_50 = card_stats["50%"]
        self.cardinalities_75 = card_stats["75%"]
        self.cardinalities_max = card_stats["max"]

        # statistics of weights
        weights_stats = self._graph._adjacency.drop(self._graph.isolates).describe()
        self.weights_mean = weights_stats["mean"]
        self.weights_std = weights_stats["std"]
        self.weights_min = weights_stats["min"]
        self.weights_25 = weights_stats["25%"]
        self.weights_50 = weights_stats["50%"]
        self.weights_75 = weights_stats["75%"]
        self.weights_max = weights_stats["max"]

        # sum of weights
        self.s0 = self._s0()
        self.s1 = self._s1()
        self.s2 = self._s2()

        # diags
        self.diag_g2 = self._diag_g2()
        self.diag_gtg = self._diag_gtg()
        self.diag_gtg_gg = self._diag_gtg_gg()

        # traces
        self.trace_g2 = self.diag_g2.sum()
        self.trace_gtg = self.diag_gtg.sum()
        self.trace_gtg_gg = self.diag_gtg_gg.sum()

    def __repr__(self):
        n_asymmetries = f"{self.n_asymmetries:>12.0f}" if self.asymmetries else "NA"
        return f"""Graph Summary Statistics
{'='*24}
Graph indexed by:
 {self._graph._get_ids_repr(57)}
{'='*62}
{"Number of nodes:":<50}{self.n_nodes:>12.0f}
{"Number of edges:":<50}{self.n_edges:>12.0f}
{"Number of connected components:":<50}{self.n_components:>12.0f}
{"Number of isolates:":<50}{self.n_isolates:12.0f}
{"Number of non-zero edges:":<50}{self.nonzero:>12.0f}
{"Percentage of non-zero edges:":<50}{self.pct_nonzero:>11.2f}%
{"Number of asymmetries:":<50}{n_asymmetries}
{'-'*62}
Cardinalities
{'='*62}
{"Mean:":<20}{self.cardinalities_mean:>9.0f}    {"25%:":<20}{self.cardinalities_25:>9.0f}
{"Standard deviation:":<20}{self.cardinalities_std:>9.0f}    {"50%:":<20}{self.cardinalities_50:>9.0f}
{"Min:":<20}{self.cardinalities_min:>9.0f}    {"75%:":<20}{self.cardinalities_75:>9.0f}
{"Max:":<20}{self.cardinalities_max:>9.0f}
{'-'*62}
Weights
{'='*62}
{"Mean:":<20}{self.weights_mean:>9.0f}    {"25%:":<20}{self.weights_25:>9.0f}
{"Standard deviation:":<20}{self.weights_std:>9.0f}    {"50%:":<20}{self.weights_50:>9.0f}
{"Min:":<20}{self.weights_min:>9.0f}    {"75%:":<20}{self.weights_75:>9.0f}
{"Max:":<20}{self.weights_max:>9.0f}
{'-'*62}
Sum of weights
{'='*62}
{"S0:":<50}{self.s0:>12.0f}
{"S1:":<50}{self.s1:>12.0f}
{"S2:":<50}{self.s2:>12.0f}
{'-'*62}
Traces
{'='*62}
{"GG:":<50}{self.trace_g2:>12.0f}
{"G'G:":<50}{self.trace_gtg:>12.0f}
{"G'G + GG:":<50}{self.trace_gtg_gg:>12.0f}
"""  # noqa: E501

    def _repr_html_(self):
        n_asymmetries = f"{self.n_asymmetries:12.0f}" if self.asymmetries else "NA"
        return f"""
            <table>
                <caption>Graph Summary Statistics</caption>
                <tr>
                    <td>Number of nodes:</td>
                    <td>{self.n_nodes:12.0f}</td>
                </tr>
                <tr>
                    <td>Number of edges:</td>
                    <td>{self.n_edges:12.0f}</td>
                </tr>
                <tr>
                    <td>Number of connected components:</td>
                    <td>{self.n_components:12.0f}</td>
                </tr>
                <tr>
                    <td>Number of isolates:</td>
                    <td>{self.n_isolates:12.0f}</td>
                </tr>
                <tr>
                    <td>Number of non-zero edges:</td>
                    <td>{self.nonzero:12.0f}</td>
                </tr>
                <tr>
                    <td>Percentage of non-zero edges:</td>
                    <td>{self.pct_nonzero:11.2f}%</td>
                </tr>
                <tr>
                    <td>Number of asymmetries:</td>
                    <td>{n_asymmetries}</td>
                </tr>
            </table>
            <table>
                <caption>Cardinalities</caption>
                <tr>
                    <td>Mean:</td>
                    <td>{self.cardinalities_mean:9.0f}</td>
                    <td>25%:</td>
                    <td>{self.cardinalities_25:9.0f}</td>
                </tr>
                <tr>
                    <td>Standard deviation:</td>
                    <td>{self.cardinalities_std:9.0f}</td>
                    <td>50%</td>
                    <td>{self.cardinalities_50:9.0f}
                    <td>
                </tr>
                <tr>
                    <td>Min:</td>
                    <td>{self.cardinalities_min:9.0f}</td>
                    <td>75%:</td>
                    <td>{self.cardinalities_75:9.0f}</td>
                </tr>
                <tr>
                    <td>Max:</td>
                    <td>{self.cardinalities_max:9.0f}</td>
                </tr>
            </table>
            <table>
                <caption>Weights</caption>
                <tr>
                    <td>Mean:</td>
                    <td>{self.weights_mean:9.0f}</td>
                    <td>25%:</td>
                    <td>{self.weights_25:9.0f}</td>
                </tr>
                <tr>
                    <td>Standard deviation:</td>
                    <td>{self.weights_std:9.0f}</td>
                    <td>50%</td>
                    <td>{self.weights_50:9.0f}
                <tr>
                    <td>Min:</td>
                    <td>{self.weights_min:9.0f}</td>
                    <td>75%:</td>
                    <td>{self.weights_75:9.0f}</td>
                </tr>
                <tr>
                    <td>Max:</td>
                    <td>{self.weights_max:9.0f}</td>
                </tr>
            </table>
            <table>
                <caption>Sum of weights and Traces</caption>
                <tr>
                    <td>S0:</td>
                    <td>{self.s0:12.0f}</td>
                    <td>GG:</td>
                    <td>{self.trace_g2:12.0f}</td>
                </tr>
                <tr>
                    <td>S1:</td>
                    <td>{self.s1:12.0f}</td>
                    <td>G'G:</td>
                    <td>{self.trace_gtg:12.0f}</td>
                </tr>
                <tr>
                    <td>S3:</td>
                    <td>{self.s2:12.0f}</td>
                    <td>G'G + GG:</td>
                    <td>{self.trace_gtg_gg:12.0f}</td>
                </tr>
            </table>
            <div>
                Graph indexed by: <code>{self._graph._get_ids_repr(57)}</code>
            </div>
            """

    def _s0(self):
        r"""helper to get S0 in downstream

         ``s0`` is defined as

        .. math::

               s0=\sum_i \sum_j w_{i,j}

        :attr:`s0`, :attr:`s1`, and :attr:`s2` reflect interaction between observations
        and are used to compute standard errors for spatial autocorrelation estimators.

        Returns
        -------
        float
            global sum of weights
        """
        return self._graph._adjacency.sum()

    def _s1(self):
        r"""S1 sum of weights

        ``s1`` is defined as

        .. math::

               s1=1/2 \sum_i \sum_j \Big(w_{i,j} + w_{j,i}\Big)^2

        :attr:`s0`, :attr:`s1`, and :attr:`s2` reflect interaction between observations
        and are used to compute standard errors for spatial autocorrelation estimators.

        Returns
        -------
        float
            s1 sum of weights
        """
        t = self._graph.sparse.transpose()
        t = t + self._graph.sparse
        t2 = t * t
        return t2.sum() / 2.0

    def _s2(self):
        r"""S2 sum of weights

        ``s2`` is defined as

        .. math::

                s2=\sum_j \Big(\sum_i w_{i,j} + \sum_i w_{j,i}\Big)^2

        :attr:`s0`, :attr:`s1`, and :attr:`s2` reflect interaction between observations
        and are used to compute standard errors for spatial autocorrelation estimators.

        Returns
        -------
        float
            s2 sum of weights
        """
        s = self._graph.sparse
        return (np.array(s.sum(1) + s.sum(0).transpose()) ** 2).sum()

    def _diag_g2(self):
        """Diagonal of :math:`GG`.

        Returns
        -------
        np.ndarray
        """
        return (self._graph.sparse @ self._graph.sparse).diagonal()

    def _diag_gtg(self):
        """Diagonal of :math:`G^{'}G`.

        Returns
        -------
        np.ndarray
        """
        return (self._graph.sparse.transpose() @ self._graph.sparse).diagonal()

    def _diag_gtg_gg(self):
        """Diagonal of :math:`G^{'}G + GG`.

        Returns
        -------
        np.ndarray
        """
        gt = self._graph.sparse.transpose()
        g = self._graph.sparse
        return (gt @ g + g @ g).diagonal()

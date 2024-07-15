import numpy as np


class GraphSummary:
    """
    s0
    s1
    s2
    n
    n asymmetries
    cardinaities stats
    n_compoments
    n isolates
    diagW2
    diagWtW
    diagWtW_WW
    n nonzero
    perc nonzero
    weights stats
    trcW2
    trcWtW
    trcWtW_WW
    """

    def __init__(self, graph):
        self._graph = graph

        self.n_nodes = self._graph.n_nodes  # number of nodes / observations
        self.n_edges = self._graph.n_edges  # number of edges excluding isolates
        self.n_components = self._graph.n_components
        self.n_isolates = len(self._graph.isolates)

        # nonzero
        self.nonzero = self._graph.nonzero
        self.pct_nonzero = self._graph.pct_nonzero

        # intrinsic assymetries
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
        weights_stats = self._graph._adjacency.describe()
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
{"Number of asymmetries:":<50}{self.n_asymmetries:>12.0f}
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
                    <td>{self.n_asymmetries:12.0f}</td>
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

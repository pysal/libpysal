from scipy import sparse
import numpy as np
import pandas as pd


class W:
    def __init__(self, adjacency, transformation="O"):
        """Weights base class based on adjacency list

        Parameters
        ----------
        adjacency : pandas.Series
            pandas.Series with a MultiIndex with two levels ("focal", "neighbor")
        """
        self.adjacency = adjacency
        self.transformation = transformation

    @classmethod
    def from_w(cls, w):
        return cls(w.to_adjlist().set_index(["focal", "neighbor"]).weight)

    def neighbors(self, ix):
        return self.adjacency[ix].index.values

    def weights(self, ix):
        return self.adjacency[ix].values

    @property
    def sparse(self):
        """Return a scipy.sparse array (COO)

        Also saves self.focal_label and self.neighbor_label capturing
        the original index labels related to their integer representation.

        Returns
        -------
        scipy.sparse.COO
            sparse representation of the adjacency
        """
        focal_int, self.focal_label = self.adjacency.index.get_level_values(
            "focal"
        ).factorize()
        neighbor_int, self.neighbor_label = self.adjacency.index.get_level_values(
            "neighbor"
        ).factorize()
        return sparse.coo_array((self.adjacency.values, (focal_int, neighbor_int)))

    def transform(self, transformation):
        """Transformation of weights

        Parameters
        ----------
        transformation : str
            Transformation method. The following are
            valid transformations.

            - **B** -- Binary
            - **R** -- Row-standardization (global sum :math:`=n`)
            - **D** -- Double-standardization (global sum :math:`=1`)
            - **V** -- Variance stabilizing

        Returns
        -------
        W
            transformed weights

        Raises
        ------
        ValueError
            Value error for unsupported transformation
        """
        transformation = transformation.upper()

        if self.transformation == transformation:
            return self

        if transformation == "R":
            standardized = self.adjacency / self.adjacency.groupby(level=0).sum()

        elif transformation == "D":
            standardized = self.adjacency / self.adjacency.sum()

        elif transformation == "B":
            standardized = pd.Series(
                index=self.adjacency.index,
                data=np.ones(self.adjacency.shape, dtype=int),
            )

        elif transformation == "V":
            standardized = self.adjacency / self.adjacency.groupby(level=0).sum().sqrt()

        else:
            raise ValueError(f"Transformation '{transformation}' is not supported.")

        return W(standardized, transformation)

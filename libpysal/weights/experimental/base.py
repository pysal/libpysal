from scipy import sparse
import numpy as np
import pandas as pd

from scipy.sparse.csgraph import connected_components


class W:
    _cache: dict = {}

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
    def from_old_w(cls, w):
        """Create an experimental W from libpysal.weights.W object

        Parameters
        ----------
        w : libpysal.weights.W

        Returns
        -------
        W
            libpysal.weights.experimentat.W
        """
        return cls(w.to_adjlist().set_index(["focal", "neighbor"]).weight)

    @property
    def neighbors(self):
        """Get neighbors dictionary

        Returns
        -------
        dict
            dict of tuples representing neighbors
        """
        return (
            self.adjacency.reset_index(level=-1)
            .neighbor.groupby(level=0)
            .agg(tuple)
            .to_dict()
        )

    @property
    def weights(self):
        """Get weights dictionary

        Returns
        -------
        dict
            dict of tuples representing weights
        """
        return self.adjacency.groupby(level=0).agg(tuple).to_dict()

    def get_neighbors(self, ix):
        """Get neighbors for a set focal object

        Parameters
        ----------
        ix : hashable
            index of focal object

        Returns
        -------
        array
            array of indices of neighbor objects
        """
        return self.adjacency[ix].index.values

    def get_weights(self, ix):
        """Get weights for a set focal object

        Parameters
        ----------
        ix : hashable
            index of focal object

        Returns
        -------
        array
            array of weights of neighbor object
        """
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

    @property
    def n_components(self):
        """Get a number of connected components

        Returns
        -------
        int
            number of components
        """
        if "n_components" not in self._cache:
            (
                self._cache["n_components"],
                self._cache["component_labels"],
            ) = sparse.csgraph.connected_components(self.sparse)
        return self._cache["n_components"]

    @property
    def component_labels(self):
        """Get component labels per observation

        Returns
        -------
        numpy.array
            Array of component labels
        """
        if "component_labels" not in self._cache:
            (
                self._cache["n_components"],
                self._cache["component_labels"],
            ) = sparse.csgraph.connected_components(self.sparse)
        return self._cache["component_labels"]

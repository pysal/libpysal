from scipy import sparse


class W:
    def __init__(self, adjacency):
        """Weights base class based on adjacency list

        Parameters
        ----------
        adjacency : pandas.Series
            pandas.Series with a MultiIndex with two levels ("focal", "neighbor")
        """
        self.adjacency = adjacency

    @classmethod
    def from_w(cls, w):
        return cls(w.to_adlist())

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

import numpy as np
import pandas

from ._utils import _resolve_islands


class _Set_Mixin:
    """
    This implements common useful set operations on weights and dunder methods.
    """

    # dunders

    def __le__(self, other):  # <=
        return self.issubgraph(other)

    def __ge__(self, other):  # >=
        return other.issubgraph(self)

    def __lt__(self, other):  # <
        return self.issubgraph(other) & (len(self) < len(other))

    def __gt__(self, other):  # >
        return other.issubgraph(self) & (len(self) > len(other))

    def __eq__(self, other):  # ==
        return self.equals(other)

    def __ne__(self, other):  # !=
        return not self.equals(other)

    def __and__(self, other):  # &
        return self.intersection(other)

    def __or__(self, other):  # |
        return self.union(other)

    def __xor__(self, other):  # ^
        return self.symmetric_difference(other)

    def __iand__(self, other):
        raise TypeError("Graphs are immutable.")

    def __ior__(self, other):
        raise TypeError("Graphs are immutable.")

    def __len__(self):
        return self.n_edges

    # methods

    def intersects(self, right):
        """
        Returns True if left and right share at least one link, irrespective of weights
        value.
        """
        intersection = self._adjacency.index.drop(self.isolates).intersection(
            right._adjacency.index.drop(right.isolates)
        )
        if len(intersection) > 0:
            return True
        return False

    def intersection(self, right):
        """
        Returns a binary Graph, that includes only those neighbor pairs that exist
        in both left and right.
        """
        from .base import Graph

        intersection = self._adjacency.index.drop(self.isolates).intersection(
            right._adjacency.index.drop(right.isolates)
        )
        return Graph.from_arrays(
            *_resolve_islands(
                intersection.get_level_values("focal"),
                intersection.get_level_values("neighbor"),
                self.unique_ids,
                np.ones(intersection.shape[0], dtype=np.int8),
            )
        )

    def symmetric_difference(self, right):
        """
        Filter out links that are in both left and right Graph objects.
        """
        from .base import Graph

        if not (self.unique_ids == right.unique_ids).all():
            raise ValueError(
                "Cannot do symmetric difference of Graphs that are based on "
                "different sets of unique IDs."
            )

        sym_diff = self._adjacency.index.drop(self.isolates).symmetric_difference(
            right._adjacency.index.drop(right.isolates)
        )
        return Graph.from_arrays(
            *_resolve_islands(
                sym_diff.get_level_values("focal"),
                sym_diff.get_level_values("neighbor"),
                self.unique_ids,
                np.ones(sym_diff.shape[0], dtype=np.int8),
            )
        )

    def union(self, right):
        """
        Provide the union of two Graph objects, collecing all links that are in either graph.
        """
        from .base import Graph

        if not (self.unique_ids == right.unique_ids).all():
            raise ValueError(
                "Cannot do union of Graphs that are based on different sets of unique IDs."
            )

        union = self._adjacency.index.drop(self.isolates).union(
            right._adjacency.index.drop(right.isolates)
        )
        return Graph.from_arrays(
            *_resolve_islands(
                union.get_level_values("focal"),
                union.get_level_values("neighbor"),
                self.unique_ids,
                np.ones(union.shape[0], dtype=np.int8),
            )
        )

    def difference(self, right):
        """
        Provide the set difference between the graph on the left and the graph on the right.
        This returns all links in the left graph that are not in the right graph.
        """
        from .base import Graph

        diff = self._adjacency.index.drop(self.isolates).difference(
            right._adjacency.index.drop(right.isolates)
        )
        return Graph.from_arrays(
            *_resolve_islands(
                diff.get_level_values("focal"),
                diff.get_level_values("neighbor"),
                self.unique_ids,
                np.ones(diff.shape[0], dtype=np.int8),
            )
        )

    def issubgraph(self, right):
        """
        Return True if every link in the left Graph also occurs in the right Graph.
        This requires both Graphs are labeled equally. Isolates are ignored.
        """
        join = (
            self._adjacency.drop(self.isolates)
            .reset_index(level=1)
            .merge(
                right._adjacency.drop(right.isolates).reset_index(level=1),
                on=("focal", "neighbor"),
                how="outer",
                indicator=True,
            )
        )
        return not (join._merge == "left_only").any()

    def equals(self, right):
        """
        Check that two graphs are identical. This reqiures them to have
        1. the same edge labels and node labels
        2. in the same order
        3. with the same weights

        This is implemented by comparing the underlying adjacency series.

        This is equivalent to checking whether the sorted list of edge tuples
        (focal, neighbor, weight) for the two graphs are the same.
        """
        try:
            pandas.testing.assert_series_equal(
                self._adjacency, right._adjacency, check_dtype=False
            )
        except AssertionError:
            return False
        return True

    def isomorphic(self, right):
        """
        Check that two graphs are isomorphic. This requires that a re-labelling
        can be found to convert one graph into the other graph. Requires networkx.
        """
        try:
            from networkx.algorithms import isomorphism as iso
        except ImportError:
            raise ImportError("NetworkX is required to check for graph isomorphism")

        nxleft = self.to_networkx()
        nxright = right.to_networkx()

        if not iso.faster_could_be_isomorphic(nxleft, nxright):
            return False
        elif not iso.could_be_isomorphic(nxleft, nxright):
            return False
        else:
            return iso.is_isomorphic(nxleft, nxright)

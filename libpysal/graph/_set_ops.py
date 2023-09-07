import pandas


class _Set_Mixin:
    """
    This implements common useful set operations on weights as dunder methods.
    """

    def __le__(self, other):
        return issubgraph(self, other)

    def __ge__(self, other):
        return issubgraph(self, other)

    def __lt__(self, other):
        return issubgraph(self, other) & (len(self) < len(other))

    def __gt__(self, other):
        return issubgraph(self, other) & (len(self) > len(other))

    def __eq__(self, other):
        return label_equals(self, other)

    def __ne__(self, other):
        return not label_equals(self, other)

    def __and__(self, other):
        return intersection(self, other)

    def __or__(self, other):
        return union(self, other)

    def __xor__(self, other):
        return symmetric_difference(self, other)

    def __iand__(self, other):
        raise TypeError("Graphs are immutable")

    def __ior__(self, other):
        raise TypeError("Graphs are immutable")

    def __len__(self):
        return self.n_edges


def intersects(left, right):
    """
    Returns True if left and right share at least one link, irrespective of weights
    value.
    """
    if left._adjacency.index.isin(right._adjacency.index).any():
        return True
    return False


def intersection(left, right):
    """
    Returns a binary weights object, that includes only those neighbor pairs that exist
    in both left and right.
    """
    from .base import Graph

    intersecting = (
        left._adjacency[left._adjacency.index.isin(right._adjacency.index)]
        .astype(bool)
        .astype(int)
    )

    # TODO: do we need to insert isolates? What if left and right have different indices?
    # TODO: do we need to check if left and right are compatible? e.g. using same indices with the same shape of sparse?

    return Graph(intersecting, transformation="b")


def symmetric_difference(left, right):
    """
    Filter out links that are in both left and right Graph objects.
    """
    from .base import Graph

    join = left.adjacency.merge(
        right.adjacency, on=("focal", "neighbor"), how="outer", indicator=True
    )
    return Graph(join[join._merge.str.endswith("only")].drop("_merge", axis=1))


def union(left, right):
    """
    Provide the union of two Graph objects, collecing all links that are in either graph.
    """
    from .base import Graph

    return Graph(
        left.adjacency.merge(right.adjacency, on=("focal", "neighbor"), how="outer")
    )


def difference(left, right):
    """
    Provide the set difference between the graph on the left and the graph on the right.
    This returns all links in the left graph that are not in the right graph.
    """
    from .base import Graph

    join = left.adjacency.merge(
        right.adjacency, on=("focal", "neighbor"), how="outer", indicator=True
    )
    return Graph(join[join._merge == "left_only"].drop("_merge", axis=1))


# TODO: profile the "not intersects(left, right)" vs. the empty join test:


def isdisjoint(left, right):
    """
    Return True if there are no links in the left Graph that also occur in the right Graph. If
    any link in the left Graph occurs in the right Graph, the two are not disjoint.
    """
    return not intersects(left, right)
    join = left.adjacency.join(right.adjacency, on=("focal", "neighbor"), how="inner")
    return join.empty()


def issubgraph(left, right):
    """
    Return True if every link in the left Graph also occurs in the right Graph. This requires
    both Graph are label_equal.
    """
    join = left.adjacency.reset_index(level=1).merge(
        right.adjacency.reset_index(level=1),
        on=("focal", "neighbor"),
        how="outer",
        indicator=True,
    )
    return not (join._merge == "left_only").any()


def issupergraph(left, right):
    """
    Return True if every link in the left Graph also occurs in the right Graph. This requires
    both Graph are label_equal.
    """
    join = left.adjacency.reset_index(level=1).merge(
        right.adjacency.reset_index(level=1),
        on=("focal", "neighbor"),
        how="outer",
        indicator=True,
    )
    return not (join._merge == "right_only").any()


## TODO: Check that each of these statements is true
# identical is the same as checking whether list of edge tuples...
# label_equal is the same as checking whether set of edge tuples...s


def _identical(left, right):
    """
    Check that two graphs are identical. This reqiures them to have
    1. the same edge labels and node labels
    2. in the same order
    3. with the same weights

    This is implemented by comparing the underlying adjacency dataframes.

    This is equivalent to checking whether the list of edge tuples
    (focal, neighbor, weight) for the two graphs are the same.

    This should generally *NOT BE USED*. Label equality and isomorphism
    should be the two "levels" of equality that are commonly-encountered
    by users. Hence, this is a private function, only for developers to
    check serialisation/deserialisation issues as necessary.
    """
    try:
        pandas.testing.assert_frame_equal(left.adjacency, right.adjacency)
    except AssertionError:
        return True
    return False


def label_equals(left, right):
    """
    Check that two graphs have the same labels. This reqiures them to have
    1. the same edge labels and node labels
    2. with the same weights

    This is implemented by comparing the underlying adjacency dataframes
    without respect to ordering.

    This is equivalent to checking whether the set of edge tuples
    (focal, neighbor, weight) for the two graphs are the same.

    See Also
    --------
    isomorphic(left, right) to check if ids in left can be re-labelled to be
    label_equal to right
    """
    try:
        pandas.testing.assert_series_equal(
            left._adjacency.sort_index(),
            right._adjacency.sort_index(),
            check_dtype=False,
        )
    except AssertionError:
        return False
    return True


def isomorphic(left, right):
    """
    Check that two graphs are isomorphic. This requires that a re-labelling
    can be found to convert one graph into the other graph. Requires networkx.
    """
    try:
        from networkx.algorithms import isomorphism as iso
    except ImportError:
        raise ImportError("NetworkX is required to check for graph isomorphism")
    nxleft, nxright = left.to_networkx(), right.to_networkx()
    if not iso.faster_could_be_isomorphic(nxleft, nxright):
        return False
    elif not iso.could_be_isomorphic(nxleft, nxright):
        return False
    else:
        return iso.is_isomorphic(nxleft, nxright)

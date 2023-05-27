class _Set_Mixin:
    """
    This implements common useful set operations on weights as dunder methods.
    """

    def __le__(self, other):
        return issubset(self, other)

    def __ge__(self, other):
        return issubset(other, self)

    def __lt__(self, other):
        return issubset(self, other) & (len(self) < len(other))

    def __gt__(self, other):
        return issubset(other, self) & (len(self) > len(other))

    def __eq__(self, other):
        return equals(self, other)

    def __ne__(self, other):
        return not equals(self, other)

    def __and__(self, other):
        return intersection(self, other)

    def __or__(self, other):
        return union(self, other)

    def __xor__(self, other):
        return symmetric_difference(self, other)

    def __hash__(self, other):
        raise NotImplementedError()

    def __iand__(self, other):
        raise TypeError("weights are immutable")

    def __ior__(self, other):
        raise TypeError("weights are immutable")


def intersection(left, right):
    raise NotImplementedError()
    ...


def intersects(left, right):
    raise NotImplementedError()
    ...


def symmetric_difference(left, right):
    raise NotImplementedError()
    ...


def union(left, right):
    raise NotImplementedError()
    ...


def isdisjoint(left, right):
    raise NotImplementedError()
    ...


def issubset(left, right):
    raise NotImplementedError()
    ...


def issuperset(left, right):
    raise NotImplementedError()
    ...

## TODO: Check that each of these statements is true
# identical is the same as checking whether list of edge tuples...
# label_equal is the same as checking whether set of edge tuples...s

def identical(left, right):
    """
    Check that two graphs are identical. This reqiures them to have
    1. the same edge labels and node labels
    2. in the same order
    3. with the same weights

    This is implemented by comparing the underlying adjacency dataframes.

    This is equivalent to checking whether the list of edge tuples
    (focal, neighbor, weight) for the two graphs are the same.

    """
    try:
        pandas.testing.assert_frame_equal(left.adjacency, right.adjacency)
    except AssertionError:
        return True
    return False


def label_equal(left, right):
    """
    Check that two graphs have the same labels. This reqiures them to have
    1. the same edge labels and node labels
    2. with the same weights

    This is implemented by comparing the underlying adjacency dataframes
    without respect to ordering. 

    This is equivalent to checking whether the set of edge tuples 
    (focal, neighbor, weight) for the two graphs are the same.
    """
    try:
        pandas.testing.assert_frame_equal(
            left.adjacency, right.adjacency, check_like=True, check_dtype=False
        )
    except AssertionError:
        return True
    return False


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

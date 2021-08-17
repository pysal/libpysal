# pylint: disable-msg=C0103, C0301
"""
Pure Python implementation of RTree spatial index.

Adaptation of
http://code.google.com/p/pyrtree/

R-tree.
see doc/ref/r-tree-clustering-split-algo.pdf
"""

__author__ = "Sergio J. Rey"

__all__ = ["RTree", "Rect", "Rtree"]


import array
import numpy
import random
import time

MAXCHILDREN = 10
MAX_KMEANS = 5
BUFFER = numpy.finfo(float).eps


class Rect(object):
    """A rectangle class that stores an axis aligned rectangle and two flags
    (swapped_x and swapped_y). The flags are stored implicitly via swaps in
    the order of minx/y and maxx/y.
    """

    __slots__ = ("x", "y", "xx", "yy", "swapped_x", "swapped_y")

    def __getstate__(self) -> tuple:
        return (self.x, self.y, self.xx, self.yy, self.swapped_x, self.swapped_y)

    def __setstate__(self, state: tuple):
        self.x, self.y, self.xx, self.yy, self.swapped_x, self.swapped_y = state

    def __init__(self, minx: float, miny: float, maxx: float, maxy: float):

        self.swapped_x = maxx < minx
        self.swapped_y = maxy < miny
        self.x = minx
        self.y = miny
        self.xx = maxx
        self.yy = maxy

        if self.swapped_x:
            self.x, self.xx = maxx, minx
        if self.swapped_y:
            self.y, self.yy = maxy, miny

    def coords(self) -> tuple:
        """Return the coordinates of the rectangle."""

        return self.x, self.y, self.xx, self.yy

    def overlap(self, orect):
        """Return the overlapping area of two rectangles.
        
        Parameters
        ----------
        orect : libpysal.cg.Rect
            Another rectangle.
        
        Returns
        -------
        overlapping_area : float
            The area of the overlap between ``orect`` and ``self``.
        
        """

        overlapping_area = self.intersect(orect).area()

        return overlapping_area

    def write_raw_coords(self, toarray, idx: int):
        """Write the raw coordinates of the rectangle."""

        toarray[idx] = self.x
        toarray[idx + 1] = self.y
        toarray[idx + 2] = self.xx
        toarray[idx + 3] = self.yy
        if self.swapped_x:
            toarray[idx] = self.xx
            toarray[idx + 2] = self.x
        if self.swapped_y:
            toarray[idx + 1] = self.yy
            toarray[idx + 3] = self.y

    def area(self) -> float:
        """Calculate the area of the rectangle."""

        w = self.xx - self.x
        h = self.yy - self.y

        return w * h

    def extent(self) -> tuple:
        """Return the extent of the rectangle in the form: (minx, minx, width, height)."""

        x = self.x
        y = self.y

        return (x, y, self.xx - x, self.yy - y)

    def grow(self, amt=None, sf=0.5):
        """Grow the bounds of a rectangle.
        
        Parameters
        ----------
        amt : float
            The amount to grow the rectangle. Default is ``None``, which
            triggers the value of ``BUFFER``.
        sf : float
            The scale factor for ``amt``. Default is ``0.5``.
            
        Returns
        -------
        rect : libpysal.cg.Rect
            A new rectangle grown by ``amt`` and scaled by ``sf``.
        
        """

        if not amt:
            amt = BUFFER
        a = amt * sf
        rect = Rect(self.x - a, self.y - a, self.xx + a, self.yy + a)

        return rect

    def intersect(self, o):
        """Find the intersection of two rectangles.
        
        Parameters
        ----------
        o : libpysal.cg.Rect
            Another rectangle.
            
        Returns
        -------
        intersection : {libpysal.cg.NullRect, libpysal.cg.Rect}
            The intersecting part of ``o`` and ``self``.

        """

        intersection = None

        if self is NullRect:
            intersection = NullRect
        elif o is NullRect:
            intersection = NullRect

        if not intersection:

            nx, ny = max(self.x, o.x), max(self.y, o.y)
            nx2, ny2 = min(self.xx, o.xx), min(self.yy, o.yy)
            w, h = nx2 - nx, ny2 - ny

            if w <= 0 or h <= 0:
                intersection = NullRect
            else:
                intersection = Rect(nx, ny, nx2, ny2)

        return intersection

    def does_contain(self, o):
        """Check whether the rectangle contains the other rectangle.
        
        Parameters
        ----------
        o : libpysal.cg.Rect
            Another rectangle.
        
        Returns
        -------
        dc : bool
            ``True`` if ``self`` contains ``o`` otherwise ``False``.
        
        """

        dc = self.does_containpoint((o.x, o.y)) and self.does_containpoint((o.xx, o.yy))

        return dc

    def does_intersect(self, o):
        """Check whether the rectangles interect.
        
        Parameters
        ----------
        o : libpysal.cg.Rect
            Another rectangle.
        
        Returns
        -------
        dcp : bool
            ``True`` if ``self`` intersects ``o`` otherwise ``False``.
        
        """

        di = self.intersect(o).area() > 0

        return di

    def does_containpoint(self, p):
        """Check whether the rectangle contains a point or not.
        
        Parameters
        ----------
        p : libpysal.cg.Point
            A point.
        
        Returns
        -------
        dcp : bool
            ``True`` if ``self`` contains ``p`` otherwise ``False``.
        
        """

        x, y = p

        dcp = x >= self.x and x <= self.xx and y >= self.y and y <= self.yy

        return dcp

    def union(self, o):
        """Union two rectangles.
        
        Parameters
        ----------
        o : libpysal.cg.Rect
            Another rectangle.
        
        Returns
        -------
        res : libpysal.cg.Rect
            The union of ``o`` and ``self``.

        """

        if o is NullRect:
            res = Rect(self.x, self.y, self.xx, self.yy)
        elif self is NullRect:
            res = Rect(o.x, o.y, o.xx, o.yy)
        else:
            x = self.x
            y = self.y
            xx = self.xx
            yy = self.yy
            ox = o.x
            oy = o.y
            oxx = o.xx
            oyy = o.yy

            nx = x if x < ox else ox
            ny = y if y < oy else oy
            nx2 = xx if xx > oxx else oxx
            ny2 = yy if yy > oyy else oyy

            res = Rect(nx, ny, nx2, ny2)

        return res

    def union_point(self, o):
        """Union the rectangle and a point
        
        Parameters
        ----------
        o : libpysal.cg.Point
            A point.
        
        Returns
        -------
        res : libpysal.cg.Rect
            The union of ``o`` and ``self``.

        """

        x, y = o
        res = self.union(Rect(x, y, x, y))

        return res

    def diagonal_sq(self) -> float:
        """Calculate the squared diagonal of the rectangle."""

        if self is NullRect:
            diag_sq = 0.0

        else:
            w = self.xx - self.x
            h = self.yy - self.y
            diag_sq = w * w + h * h

        return diag_sq

    def diagonal(self) -> float:
        """Calculate the diagonal of the rectangle."""

        return numpy.sqrt(self.diagonal_sq())


NullRect = Rect(0.0, 0.0, 0.0, 0.0)
NullRect.swapped_x = False
NullRect.swapped_y = False


def union_all(kids):
    """Create union of all child rectangles.
    
    Parameters
    ----------
    kids : list
        A list of ``libpysal.cg._NodeCursor`` objects.
    
    Returns
    -------
    cur : {libpysal.cg.Rect, libpysal.cg.NullRect}
        The unioned result of all child rectangles.
    
    """

    cur = NullRect
    for k in kids:
        cur = cur.union(k.rect)

    assert False == cur.swapped_x

    return cur


def Rtree():
    return RTree()


class RTree(object):
    """An RTree for efficiently querying space based on intersecting rectangles.
    
    Attributes
    ----------
    count : int
        The number of nodes in the tree.
    stats : dict
        Tree generation statistics.
    leaf_count : int
        The number of leaves (objects) in the tree.
    rect_pool : array.array
        The pool of rectangles in the tree in the form
        :math:`[ax1, ay1, ax2, ay2, bx1, by1, bx2, by2, ..., nx1, ny1, nx2, ny2]`
        where the first set of 4 coordinates is the bounding box of the root node
        and each successive set of 4 coordinates is the bounding box of a leaf node.
    node_pool : array.array
        The pool of node IDs in the tree.
    leaf_pool : list
        The pool of leaf objects in the tree.
    cursor : libpysal.cg._NodeCursor
        The non-root node and all its children.
    
    Examples
    --------
    
    Instantiate an ``RTree``.
    
    >>> from libpysal.cg import RTree, Chain
    >>> segments = [
    ...     [(0.0, 1.5), (1.5, 1.5)],
    ...     [(1.5, 1.5), (3.0, 1.5)],
    ...     [(1.5, 1.5), (1.5, 0.0)],
    ...     [(1.5, 1.5), (1.5, 3.0)]
    ... ]
    >>> segments = [Chain([p1, p2]) for p1, p2 in segments]
    >>> rt = RTree()
    >>> for segment in segments:
    ...     rt.insert(segment, Rect(*segment.bounding_box).grow(sf=10.))
    
    Examine the tree generation statistics. The statistics here
    are all 0 due to the simple structure of the tree in this example.
    
    >>> rt.stats
    {'overflow_f': 0,
     'avg_overflow_t_f': 0.0,
     'longest_overflow': 0.0,
     'longest_kmeans': 0.0,
     'sum_kmeans_iter_f': 0,
     'count_kmeans_iter_f': 0,
     'avg_kmeans_iter_f': 0.0}
    
    Examine the number of nodes and leaves.
    There five nodes and four leaves (the root plus its four children).
    
    >>> rt.count, rt.leaf_count
    (5, 4)
    
    The pool of nodes are the node IDs in the tree.
    
    >>> rt.node_pool
    array('L', [0, 4, 0, 0, 1, 1, 2, 2, 3, 3])
    
    The pool of leaves are the geometric objects that were inserted into the tree.
    
    >>> rt.leaf_pool[0].vertices
    [(0.0, 1.5), (1.5, 1.5)]
    
    The pool of rectangles are the bounds of partitioned space in the tree.
    Examine the first one.
    
    >>> rt.rect_pool[:4]
    array('d', [-2.220446049250313e-15, -2.220446049250313e-15, 3.000000000000002, 3.000000000000002])
    
    Add the bounding box of a leaf to the tree manually.
    
    >>> rt.add(Chain(((2,2), (4,4))), (2,2,4,4))
    >>> rt.count, rt.leaf_count
    (6, 5)
    
    Query the tree for an intersection. One object is contained in this query.
    
    >>> rt.intersection([.4, 2.1, .9, 2.6])[0].vertices
    [(0.5, 2), (1, 2.5)]
    
    Query the tree with a much larger box. All objects are contained in this query.
    
    >>> len(rt.intersection([-1, -1, 4, 4])) == rt.leaf_count
    True
    
    Query the tree with box outside the tree objects.
    No objects are contained in this query.
    
    >>> rt.intersection([5, 5, 6, 6])
    []
    
    """

    def __init__(self):

        self.count = 0
        self.stats = {
            "overflow_f": 0,
            "avg_overflow_t_f": 0.0,
            "longest_overflow": 0.0,
            "longest_kmeans": 0.0,
            "sum_kmeans_iter_f": 0,
            "count_kmeans_iter_f": 0,
            "avg_kmeans_iter_f": 0.0,
        }

        # This round: not using objects directly -- they take up too much memory, and
        #   efficiency goes down the toilet (obviously) if things start to page. Less
        #   obviously: using object graph directly leads to really long GC pause times,
        #   too. Instead, it uses pools of arrays:
        self.count = 0
        self.leaf_count = 0
        self.rect_pool = array.array("d")
        self.node_pool = array.array("L")

        # leaf objects.
        self.leaf_pool = []

        self.cursor = _NodeCursor.create(self, NullRect)

    def _ensure_pool(self, idx: int):
        """Ensure sufficient slots in rectangle and node pools."""

        bb_len, pool_slot = 4, [0]
        node_len = int(bb_len / 2)
        if len(self.rect_pool) < (bb_len * idx):
            self.rect_pool.extend(pool_slot * bb_len)
            self.node_pool.extend(pool_slot * node_len)

    def insert(self, o, orect):
        """Insert an object and its bounding box into the tree.
        
        Parameters
        ----------
        o : libpysal.cg.{Point, Chain, Rectangle, Polygon}
            The object to insert into the tree.
        orect : ibpysal.cg.Rect
            The object's bounding box.
        
        """

        self.cursor.insert(o, orect)
        assert self.cursor.index == 0

    def query_rect(self, r):
        """Query a rectangle.
        
        Parameters
        ----------
        r : {tuple, libpysal.cg.Point}
            The bounding box of the rectangle in question;
            a :math:`(minx,miny,maxx,maxy)` set of coordinates.
        
        Yields
        ------
        x : generator
            ``libpysal.cg._NodeCursor`` objects.
        """

        for x in self.cursor.query_rect(r):
            yield x

    def query_point(self, p):
        """Query a point.
        
        Parameters
        ----------
        p : {tuple, libpysal.cg.Point}
            The point in question; an :math:`(x,y)` coordinate.
        
        Yields
        ------
        x : generator
            ``libpysal.cg._NodeCursor`` objects.
        """

        for x in self.cursor.query_point(p):
            yield x

    def walk(self, pred):
        """Walk the tree structure with ``pred`` (a function)."""

        return self.cursor.walk(pred)

    def intersection(self, boundingbox):
        """Query for an intersection between leaves in the ``RTree``
        and the bounding box of an object.
        
        Parameters
        ----------
        boundingbox : list
            The bounding box: ``[minx, miny, maxx, maxy]``.
        
        Returns
        -------
        objs : list
            A list of objects whose bounding
            boxes intersect with the query bounding box.

        """

        # grow the bounding box slightly to handle coincident edges
        qr = Rect(*boundingbox).grow(sf=10.0)

        objs = [r.leaf_obj() for r in self.query_rect(qr) if r.is_leaf()]

        return objs

    def add(self, id, boundingbox):
        """Add the bounding box of a leaf to the ``RTree`` manually with a specified ID.

        Parameters
        ----------
        id : int
            An object id.
        boundingbox : list
            The bounding box: ``[minx, miny, maxx, maxy]``.
        
        """

        self.cursor.insert(id, Rect(*boundingbox))


class _NodeCursor(object):
    """An internal class for keeping track of, and reorganizing,
    the structure and composition of the ``RTree``.
    
    Parameters
    ----------
    rooto : libpysal.cg.{Point, Chain, Rectangle, Polygon}
        The object from which the node will be generated.
    index : int
        The ID of the node.
    rect : libpysal.cg.Rect
        The bounding rectangle of the leaf object.
    first_child : int
        The ID of the first child of the node.
    next_sibling : int
        The ID of the sibling of the node.
    
    Attributes
    ----------
    root : libpysal.cg.RTree
        The root node of the tree.
    npool : array.array
        See ``RTree.node_pool``.
    rpool : array.array
        See ``RTree.rect_pool``.
    
    """

    @classmethod
    def create(cls, rooto, rect):
        """Create a node in the tree structure.
        
        Parameters
        ----------
        rooto : libpysal.cg.{Point, Chain, Rectangle, Polygon}
            The object from which the node will be generated.
        index : int
            The ID of the node.
        rect : libpysal.cg.Rect
            The bounding rectangle of the leaf object.
        
        Returns
        -------
        retv : libpysal.cg._NodeCursor
            The generated node.
        
        """

        idx = rooto.count
        rooto.count += 1

        rooto._ensure_pool(idx + 1)

        retv = _NodeCursor(rooto, idx, rect, 0, 0)

        retv._save_back()

        return retv

    @classmethod
    def create_with_children(cls, children, rooto):
        """Create a non-leaf node in the tree structure.
        
        Parameters
        ----------
        children : list
            The child nodes of the node to be generated
        rooto : libpysal.cg.{Point, Chain, Rectangle, Polygon}
            The object from which the node will be generated.
        
        Returns
        -------
        nc : libpysal.cg._NodeCursor
            The generated node with children.
        
        """
        rect = union_all([c for c in children])
        nr = Rect(rect.x, rect.y, rect.xx, rect.yy)

        assert not rect.swapped_x
        nc = _NodeCursor.create(rooto, rect)
        nc._set_children(children)

        assert not nc.is_leaf()
        return nc

    @classmethod
    def create_leaf(cls, rooto, leaf_obj, leaf_rect):
        """Create a leaf node in the tree structure.
        
        Parameters
        ----------
        rooto : libpysal.cg.{Point, Chain, Rectangle, Polygon}
            The object from which the node will be generated.
        leaf_obj : libpysal.cg.{Point, Chain, Rectangle, Polygon}
            The leaf object.
        leaf_rect : libpysal.cg.Rect
            The bounding rectangle of the leaf object.
        
        Returns
        -------
        res : libpysal.cg._NodeCursor
            The generated leaf node.
        
        """

        rect = Rect(leaf_rect.x, leaf_rect.y, leaf_rect.xx, leaf_rect.yy)

        # Mark as leaf by setting the xswap flag.
        rect.swapped_x = True
        res = _NodeCursor.create(rooto, rect)
        idx = res.index
        res.first_child = rooto.leaf_count
        rooto.leaf_count += 1
        res.next_sibling = 0
        rooto.leaf_pool.append(leaf_obj)
        res._save_back()
        res._become(idx)

        assert res.is_leaf()
        return res

    __slots__ = (
        "root",
        "npool",
        "rpool",
        "index",
        "rect",
        "next_sibling",
        "first_child",
    )

    def __getstate__(self) -> tuple:
        return (
            self.root,
            self.npool,
            self.rpool,
            self.index,
            self.rect,
            self.next_sibling,
            self.first_child,
        )

    def __setstate__(self, state: tuple):
        (
            self.root,
            self.npool,
            self.rpool,
            self.index,
            self.rect,
            self.next_sibling,
            self.first_child,
        ) = state

    def __init__(self, rooto, index, rect, first_child, next_sibling):

        self.root = rooto
        self.rpool = rooto.rect_pool
        self.npool = rooto.node_pool
        self.index = index
        self.rect = rect
        self.next_sibling = next_sibling
        self.first_child = first_child

    def walk(self, predicate):
        """Walk the tree structure with ``predicate`` (a function)."""

        if predicate(self, self.leaf_obj()):
            yield self
            if not self.is_leaf():
                for c in self.children():
                    for cr in c.walk(predicate):
                        yield cr

    def query_rect(self, r):
        """Yield objects that intersect with the rectangle (``r``)."""

        def p(o, x):
            return r.does_intersect(o.rect)

        for rr in self.walk(p):
            yield rr

    def query_point(self, point):
        """Yield objects that intersect with the point (``point``)."""

        def p(o, x):
            return o.rect.does_containpoint(point)

        for rr in self.walk(p):
            yield rr

    def lift(self):
        """Promote a node to (potentially) rearrange the
        tree structure for optimal clustering.
        
        Called from ``_NodeCursor._balance()``.
        
        Returns
        -------
        lifted : libpysal.cg._NodeCursor
            The lifted node.
        
        """

        lifted = _NodeCursor(
            self.root, self.index, self.rect, self.first_child, self.next_sibling
        )

        return lifted

    def _become(self, index: int):
        """Have ``self`` become node ``index``."""

        recti = index * 4
        nodei = index * 2
        rp = self.rpool
        x = rp[recti]
        y = rp[recti + 1]
        xx = rp[recti + 2]
        yy = rp[recti + 3]

        if x == 0.0 and y == 0.0 and xx == 0.0 and yy == 0.0:
            self.rect = NullRect
        else:
            self.rect = Rect(x, y, xx, yy)

        self.next_sibling = self.npool[nodei]
        self.first_child = self.npool[nodei + 1]
        self.index = index

    def is_leaf(self) -> bool:
        """Return ``True`` if the node is a leaf, otherwise ``False``."""

        return self.rect.swapped_x

    def has_children(self) -> bool:
        """Return ``True`` if the node has children, otherwise ``False``."""

        return not self.is_leaf() and 0 != self.first_child

    def holds_leaves(self) -> bool:
        """Return ``True`` if the node holds leaves, otherwise ``False``."""

        if 0 == self.first_child:
            return True
        else:
            return self.has_children() and self.get_first_child().is_leaf()

    def get_first_child(self):
        """Get the first child of a node.
        
        Returns
        -------
        c : libpysal.cg._NodeCursor
            The first child of the specified node.
        
        """

        fc = self.first_child
        c = _NodeCursor(self.root, 0, NullRect, 0, 0)
        c._become(self.first_child)

        return c

    def leaf_obj(self):
        """Return the leaf object if the node is a leaf, other return ``None``."""

        if self.is_leaf():
            return self.root.leaf_pool[self.first_child]
        else:
            return None

    def _save_back(self):
        """Save a node back into the tree structure."""

        rp = self.rpool
        recti = self.index * 4
        nodei = self.index * 2

        if self.rect is not NullRect:
            self.rect.write_raw_coords(rp, recti)
        else:
            rp[recti] = 0
            rp[recti + 1] = 0
            rp[recti + 2] = 0
            rp[recti + 3] = 0

        self.npool[nodei] = self.next_sibling
        self.npool[nodei + 1] = self.first_child

    def nchildren(self) -> int:
        """The number of children nodes."""

        i = self.index
        c = 0
        for x in self.children():
            c += 1

        return c

    def insert(self, leafo, leafrect):
        """Insert a leaf object into the tree. See
        ``RTree.insert(o, orect)`` for parameter description.
        
        """

        index = self.index

        # tail recursion, made into loop:
        while True:
            if self.holds_leaves():
                self.rect = self.rect.union(leafrect)
                self._insert_child(_NodeCursor.create_leaf(self.root, leafo, leafrect))

                self._balance()

                # done: become the original again
                self._become(index)
                return
            else:
                # Not holding leaves, move down a level in the tree:

                # ----------------------
                # Micro-optimization:
                #   inlining union() calls -- logic is:
                #       ignored, child = min(
                #           [
                #               ((c.rect.union(leafrect)).area() - c.rect.area(),c.index)
                #               for c in self.children()
                #           ]
                #       )
                child = None
                minarea = -1.0
                for c in self.children():
                    x, y, xx, yy = c.rect.coords()
                    lx, ly, lxx, lyy = leafrect.coords()
                    nx = x if x < lx else lx
                    nxx = xx if xx > lxx else lxx
                    ny = y if y < ly else ly
                    nyy = yy if yy > lyy else lyy
                    a = (nxx - nx) * (nyy - ny)
                    if minarea < 0 or a < minarea:
                        minarea = a
                        child = c.index
                # End micro-optimization
                # ----------------------

                self.rect = self.rect.union(leafrect)
                self._save_back()
                # recurse.
                self._become(child)

    def _balance(self):
        """Balance the leaf layout where possible through ``k_means_cluster()``
        and ``silhouette_coeff()`` for (heuristically) optimal clusterings of
        nodes in the tree structure after the child count of a node has grown
        past the maximum allowed number (see ``MAXCHILDREN``).
        
        Called from ``_NodeCursor.insert()``.
        """

        if self.nchildren() <= MAXCHILDREN:
            return

        t = time.process_time()

        cur_score = -10

        s_children = [c.lift() for c in self.children()]

        clusterings = [
            k_means_cluster(self.root, k, s_children) for k in range(2, MAX_KMEANS)
        ]
        score, bestcluster = max([(silhouette_coeff(c), c) for c in clusterings])

        # generate the (heuristically) optimally-balanced cluster of nodes
        nodes = [
            _NodeCursor.create_with_children(c, self.root)
            for c in bestcluster
            if len(c) > 0
        ]

        self._set_children(nodes)

        dur = time.process_time() - t
        c = float(self.root.stats["overflow_f"])
        oa = self.root.stats["avg_overflow_t_f"]
        self.root.stats["avg_overflow_t_f"] = (dur / (c + 1.0)) + (c * oa / (c + 1.0))
        self.root.stats["overflow_f"] += 1
        self.root.stats["longest_overflow"] = max(
            self.root.stats["longest_overflow"], dur
        )

    def _set_children(self, cs: list):
        """Set up the (new/altered) leaf tree structure.
        
        Called from ``_NodeCursor.create_with_children()``
        and ``_NodeCursor._balance()``.
        
        """

        self.first_child = 0

        if 0 == len(cs):
            return

        pred = None
        for c in cs:
            if pred is not None:
                pred.next_sibling = c.index
                pred._save_back()
            if 0 == self.first_child:
                self.first_child = c.index
            pred = c
        pred.next_sibling = 0
        pred._save_back()
        self._save_back()

    def _insert_child(self, c):
        """Internal function for child node insertion. 
        Called from ``_NodeCursor.insert()``.
        
        Parameters
        ----------
        c : libpysal.cg._NodeCursor
            A child ``libpysal.cg._NodeCursor`` object.
        
        """

        c.next_sibling = self.first_child
        self.first_child = c.index
        c._save_back()
        self._save_back()

    def children(self):
        """Yield the children of a node."""

        if 0 == self.first_child:
            return

        idx = self.index
        fc = self.first_child
        ns = self.next_sibling
        r = self.rect

        self._become(self.first_child)
        while True:
            yield self
            if 0 == self.next_sibling:
                break
            else:
                self._become(self.next_sibling)

        # Go back to becoming the same node we were.
        # self._become(idx)
        self.index = idx
        self.first_child = fc
        self.next_sibling = ns
        self.rect = r


def avg_diagonals(node, onodes):
    """Calculate the mean diagonals.
    
    Parameters
    ----------
    node : libpysal.cg._NodeCursor
        The target node in question.
    onodes : ist
        A list of ``libpysal.cg._NodeCursor`` objects.
    
    Returns
    -------
    diag_avg : float
        The mean diagonal distance of ``node`` and ``onodes``.
    
    """

    nidx = node.index
    sv = 0.0
    diag = 0.0
    memo_tab = {}

    for onode in onodes:
        k1 = (nidx, onode.index)
        k2 = (onode.index, nidx)

        if k1 in memo_tab:
            diag = memo_tab[k1]
        elif k2 in memo_tab:
            diag = memo_tab[k2]
        else:
            diag = node.rect.union(onode.rect).diagonal()
            memo_tab[k1] = diag

        sv += diag

    diag_avg = sv / len(onodes)

    return diag_avg


def silhouette_w(node, cluster, next_closest_cluster):
    """Calculate a silhouette score between a certain node and 2 clusters:
    
    Parameters
    ----------
    node : libpysal.cg._NodeCursor
        The target node in question.
    cluster : list
        A list of ``libpysal.cg._NodeCursor`` objects.
    next_closest_cluster : list
        Another list of ``libpysal.cg._NodeCursor`` objects.
    
    Returns
    -------
    silw : float
        The silhouette score between ``{node, cluster}``
        and ``{node, next_closest_cluster}``.
    
    """

    ndist = avg_diagonals(node, cluster)
    sdist = avg_diagonals(node, next_closest_cluster)

    silw = (sdist - ndist) / max(sdist, ndist)

    return silw


def silhouette_coeff(clustering):
    """Calculate how well defined the clusters are. A score of ``1`` indicates
    the clusters are well defined, a score of ``0`` indicates the clusters are
    undefined, and a score of ``-1`` indicates the clusters are defined
    incorrectly.
    
    Parameters
    ----------
    clustering : list
        A list of ``libpysal.cg._NodeCursor`` objects.
    
    Returns
    -------
    silcoeff : float
        Score for how well defined the clusters are.
        
    """

    # special case for a clustering of 1.0
    if len(clustering) == 1:
        silcoeff = 1.0

    else:
        coeffs = []
        for cluster in clustering:
            others = [c for c in clustering if c is not cluster]
            others_cntr = [center_of_gravity(c) for c in others]
            ws = [
                silhouette_w(node, cluster, others[closest(others_cntr, node)])
                for node in cluster
            ]
            cluster_coeff = sum(ws) / len(ws)
            coeffs.append(cluster_coeff)

        silcoeff = sum(coeffs) / len(coeffs)

    return silcoeff


def center_of_gravity(nodes):
    """Find the center of gravity of multiple nodes.
    
    Parameters
    ----------
    nodes : list
        A list of ``libpysal.cg.RTree`` and ``libpysal.cg._NodeCursor`` objects.
    
    Returns
    -------
    cog : float
        The center of gravity of multiple nodes.
    
    """

    totarea = 0.0
    xs, ys = 0, 0
    for n in nodes:
        if n.rect is not NullRect:
            x, y, w, h = n.rect.extent()
            a = w * h
            xs = xs + (a * (x + (0.5 * w)))
            ys = ys + (a * (y + (0.5 * h)))
            totarea = totarea + a

    cog = (xs / totarea), (ys / totarea)

    return cog


def closest(centroids, node):
    """Find the closest controid to the node's center of gravity.
    
    Parameters
    ----------
    centroids : list
        A list of (x, y) coordinates for the center of other clusters.
    node : libpysal.cg_NodeCursor
        A ``libpysal.cg._NodeCursor`` instance.
    
    Returns
    -------
    ridx : int
        The index of the nearest centroid of other cluster.

    """

    x, y = center_of_gravity([node])
    dist = -1
    ridx = -1

    for (i, (xx, yy)) in enumerate(centroids):
        dsq = ((xx - x) ** 2) + ((yy - y) ** 2)
        if -1 == dist or dsq < dist:
            dist = dsq
            ridx = i

    return ridx


def k_means_cluster(root, k, nodes):
    """Find ``k`` clusters.
    
    Parameters
    ----------
    root : libpysal.cg.RTree
        An ``libpysal.cg.RTree`` instance.
    k : int
        The number clusters to find.
    nodes : list
        A list of ``libpysal.cg.RTree`` and ``libpysal.cg._NodeCursor`` objects.
    
    Returns
    -------
    clusters : list
        Updated versions of ``nodes`` defining new clusters.
    
    """

    t = time.process_time()
    if len(nodes) <= k:
        clusters = [[n] for n in nodes]
        return clusters

    ns = list(nodes)
    root.stats["count_kmeans_iter_f"] += 1

    # Initialize: take n random nodes.
    # random.shuffle(ns)

    cluster_starts = ns[:k]
    cluster_centers = [center_of_gravity([n]) for n in ns[:k]]

    # Loop until stable:
    while True:
        root.stats["sum_kmeans_iter_f"] += 1
        clusters = [[] for c in cluster_centers]

        for n in ns:
            idx = closest(cluster_centers, n)
            clusters[idx].append(n)

        # FIXME HACK TODO: is it okay for there to be empty clusters?
        clusters = [c for c in clusters if len(c) > 0]

        for c in clusters:
            if len(c) == 0:
                print("Error....")
                print(("Nodes: %d, centers: %s." % (len(ns), repr(cluster_centers))))

            assert len(c) > 0

        rest = ns
        first = False

        new_cluster_centers = [center_of_gravity(c) for c in clusters]
        if new_cluster_centers == cluster_centers:
            root.stats["avg_kmeans_iter_f"] = float(
                root.stats["sum_kmeans_iter_f"] / root.stats["count_kmeans_iter_f"]
            )
            root.stats["longest_kmeans"] = max(
                root.stats["longest_kmeans"], (time.process_time() - t)
            )
            return clusters
        else:
            cluster_centers = new_cluster_centers

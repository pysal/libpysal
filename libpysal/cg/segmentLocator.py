# ruff: noqa: A002, N802, N806, N999

import math
import random
import time
import warnings

import numpy
import scipy

from .shapes import LineSegment, Point, Rectangle
from .standalone import get_bounding_box, get_segment_point_dist

dep_msg = "is deprecated and will be removed in a future version of libpysal"

__all__ = ["SegmentGrid", "SegmentLocator", "Polyline_Shapefile_SegmentLocator"]
DEBUG = False


class BruteSegmentLocator:
    def __init__(self, segments):
        self.data = segments
        self.n = len(segments)

    def nearest(self, pt):
        d = self.data
        distances = [get_segment_point_dist(d[i], pt)[0] for i in range(self.n)]
        return numpy.argmin(distances)


class SegmentLocator:
    def __init__(self, segments, nbins=500):
        warnings.warn("SegmentLocator " + dep_msg, FutureWarning, stacklevel=2)
        self.data = segments
        if hasattr(segments, "bounding_box"):
            bbox = segments.bounding_box
        else:
            bbox = get_bounding_box(segments)
        self.bbox = bbox
        res = max((bbox.right - bbox.left), (bbox.upper - bbox.lower)) / float(nbins)
        self.grid = SegmentGrid(bbox, res)
        for i, seg in enumerate(segments):
            self.grid.add(seg, i)

    def nearest(self, pt):
        d = self.data
        possibles = self.grid.nearest(pt)
        distances = [get_segment_point_dist(d[i], pt)[0] for i in possibles]
        # print "possibles",possibles
        # print "distances",distances
        # print "argmin", numpy.argmin(distances)
        return possibles[numpy.argmin(distances)]


class Polyline_Shapefile_SegmentLocator:  # noqa: N801
    def __init__(self, shpfile, nbins=500):
        warnings.warn(
            "Polyline_Shapefile_SegmentLocator " + dep_msg, FutureWarning, stacklevel=2
        )
        self.data = shpfile
        bbox = Rectangle(*shpfile.bbox)
        res = max((bbox.right - bbox.left), (bbox.upper - bbox.lower)) / float(nbins)
        self.grid = SegmentGrid(bbox, res)
        for i, polyline in enumerate(shpfile):
            for p, part in enumerate(polyline.segments):
                for j, seg in enumerate(part):
                    self.grid.add(seg, (i, p, j))

    def nearest(self, pt):
        d = self.data
        possibles = self.grid.nearest(pt)
        distances = [
            get_segment_point_dist(d[i].segments[p][j], pt)[0]
            for (i, p, j) in possibles
        ]
        # print "possibles",possibles
        # print "distances",distances
        # print "argmin", numpy.argmin(distances)
        return possibles[numpy.argmin(distances)]


class SegmentGrid:
    """
    Notes:
        SegmentGrid is a low level Grid class.
        This class does not maintain a copy of the geometry in the grid.
        It returns only approx. Solutions.
        This Grid should be wrapped by a locator.
    """

    def __init__(self, bounds, resolution):
        """
        Returns a grid with specified properties.

        __init__(Rectangle, number) -> SegmentGrid

        Parameters
        ----------
        bounds      : the area for the grid to encompass
        resolution  : the diameter of each bin

        Examples
        --------
        TODO: complete this doctest
        >>> g = SegmentGrid(Rectangle(0, 0, 10, 10), 1)
        """
        warnings.warn("SegmentGrid " + dep_msg, FutureWarning, stacklevel=2)
        if resolution == 0:
            raise Exception("Cannot create grid with resolution 0")
        self.res = resolution
        self.hash = {}
        self._kd = None
        self._kd2 = None
        self._hashKeys = None
        self.x_range = (bounds.left, bounds.right)
        self.y_range = (bounds.lower, bounds.upper)
        try:
            self.i_range = (
                int(math.ceil((self.x_range[1] - self.x_range[0]) / self.res)) + 1
            )
            self.j_range = (
                int(math.ceil((self.y_range[1] - self.y_range[0]) / self.res)) + 1
            )
            self.mask = numpy.zeros((self.i_range, self.j_range), bool)
            self.endMask = numpy.zeros((self.i_range, self.j_range), bool)
        except Exception as e:
            raise Exception(
                "Invalid arguments for SegmentGrid(): ("
                + str(self.x_range)
                + ", "
                + str(self.y_range)
                + ", "
                + str(self.res)
                + ")"
            ) from e

    @property
    def hashKeys(self):
        if self._hashKeys is None:
            self._hashKeys = numpy.array(list(self.hash.keys()), dtype=float)
        return self._hashKeys

    @property
    def kd(self):
        if self._kd is None:
            self._kd = scipy.spatial.cKDTree(self.hashKeys)
        return self._kd

    @property
    def kd2(self):
        if self._kd2 is None:
            self._kd2 = scipy.spatial.KDTree(self.hashKeys)
        return self._kd2

    def in_grid(self, loc):
        """
        Returns whether a 2-tuple location _loc_ lies inside the grid bounds.
        """
        return (
            self.x_range[0] <= loc[0] <= self.x_range[1]
            and self.y_range[0] <= loc[1] <= self.y_range[1]
        )

    def _grid_loc(self, loc):
        i = int((loc[0] - self.x_range[0]) / self.res)  # floored
        j = int((loc[1] - self.y_range[0]) / self.res)  # floored
        # i = min(self.i_range-1, max(int((loc[0] - self.x_range[0])/self.res), 0))
        # j = min(self.j_range-1, max(int((loc[1] - self.y_range[0])/self.res), 0))
        # print "bin:", loc, " -> ", (i,j)
        return (i, j)

    def _real_loc(self, grid_loc):
        x = (grid_loc[0] * self.res) + self.x_range[0]
        y = (grid_loc[1] * self.res) + self.y_range[0]
        return x, y

    def bin_loc(self, loc, id):
        grid_loc = self._grid_loc(loc)
        if grid_loc not in self.hash:
            self.hash[grid_loc] = set()
            self.mask[grid_loc] = True
        self.hash[grid_loc].add(id)
        return grid_loc

    def add(self, segment, id):
        """
        Adds segment to the grid.

        add(segment, id) -> bool

        Parameters
        ----------
        id -- id to be stored int he grid.
        segment -- the segment which identifies where to store 'id' in the grid.

        Examples
        --------
        >>> g = SegmentGrid(Rectangle(0, 0, 10, 10), 1)
        >>> g.add(LineSegment(Point((0.2, 0.7)), Point((4.2, 8.7))), 0)
        True
        """
        if not (self.in_grid(segment.p1) and self.in_grid(segment.p2)):
            raise Exception(
                "Attempt to insert item at location outside grid bounds: "
                + str(segment)
            )
        i, j = self.bin_loc(segment.p1, id)
        i_, j_ = self.bin_loc(segment.p2, id)
        self.endMask[i, j] = True
        self.endMask[i_, j_] = True

        res = self.res
        line = segment.line
        tiny = res / 1000.0
        for i in range(1 + min(i, i_), max(i, i_)):  # noqa: B020
            # print 'i',i
            x = self.x_range[0] + (i * res)
            y = line.y(x)
            self.bin_loc((x - tiny, y), id)
            self.bin_loc((x + tiny, y), id)
        for j in range(1 + min(j, j_), max(j, j_)):  # noqa: B020
            # print 'j',j
            y = self.y_range[0] + (j * res)
            x = line.x(y)
            self.bin_loc((x, y - tiny), id)
            self.bin_loc((x, y + tiny), id)
        self._kd = None
        self._kd2 = None
        return True

    def remove(self, segment):  # noqa: ARG002
        self._kd = None
        self._kd2 = None
        pass

    def nearest(self, pt):
        """
        Return a set of ids.

        The ids identify line segments within a radius of the query point.
        The true nearest segment is guaranteed to be within the set.

        Filtering possibles is the responsibility of the locator not the grid.
        This means the Grid doesn't need to keep a reference to the underlying segments,
        which in turn means the Locator can keep the segments on disk.

        Locators can be customized to different data stores (shape files, SQL, etc.)
        """
        grid_loc = numpy.array(self._grid_loc(pt))
        possibles = set()

        if DEBUG:
            print("in_grid:", self.in_grid(pt))
            i = pylab.matshow(
                self.mask, origin="lower", extent=self.x_range + self.y_range, fignum=1
            )
        # Use KD tree to search out the nearest filled bin.
        # it may be faster to not use kdtree, or at least check grid_loc first
        # The KD tree is build on the keys of self.hash, a dictionary of stored bins.
        dist, i = self.kd.query(grid_loc, 1)

        ### Find non-empty bins within a radius of the query point.
        # Location of Q point
        row, col = grid_loc
        # distance to nearest filled cell +2.
        # +1 returns inconsistent results (compared to BruteSegmentLocator)
        # +2 seems to do the trick.
        radius = int(math.ceil(dist)) + 2
        if radius < 30:
            a, b = numpy.ogrid[
                -radius : radius + 1, -radius : radius + 1
            ]  # build square index arrays centered at 0,0
            index = (
                a**2 + b**2 <= radius**2
            )  # create a boolean mask to filter indicies outside radius
            a, b = index.nonzero()
            # grad the (i,j)'s of the elements within radius.
            rows, cols = (
                row + a - radius,
                col + b - radius,
            )  # recenter the (i,j)'s over the Q point
            #### Filter indicies by bounds of the grid.
            ### filters must be applied one at a time
            ### I havn't figure out a way to group these
            filter_ = rows >= 0
            rows = rows[filter_]
            cols = cols[filter_]  # i >= 0
            filter_ = rows < self.i_range
            rows = rows[filter_]
            cols = cols[filter_]  # i < i_range
            filter_ = cols >= 0
            rows = rows[filter_]
            cols = cols[filter_]  # j >= 0
            filter_ = cols < self.j_range
            rows = rows[filter_]
            cols = cols[filter_]  # j < j_range
            if DEBUG:
                mask_copy = self.mask.copy().astype(float)
                mask_copy += self.endMask.astype(float)
                mask_copy[rows, cols] += 1
                mask_copy[row, col] += 3
                i = pylab.matshow(
                    mask_copy,
                    origin="lower",
                    extent=self.x_range + self.y_range,
                    fignum=1,
                )
                # raw_input('pause')
            ### All that was just setup for this one line...
            idx = self.mask[rows, cols].nonzero()[0]  # Filter out empty bins.
            rows, cols = (
                rows[idx],
                cols[idx],
            )  # (i,j)'s of the filled grid cells within radius.

            for t in zip(rows, cols):  # noqa: B905
                possibles.update(self.hash[t])

            if DEBUG:
                print("possibles", possibles)
        else:
            ### The old way...
            ### previously I was using kd.query_ball_point on,
            ### but the performance was terrible.
            i_ = self.kd2.query_ball_point(grid_loc, radius)
            for i in i_:
                t = tuple(self.kd.data[i])
                possibles.update(self.hash[t])
        return list(possibles)


def random_segments(n):
    segs = []
    for _i in range(n):
        a, b, c, d = (random.random() for x in [1, 2, 3, 4])
        seg = LineSegment(Point((a, b)), Point((c, d)))
        segs.append(seg)
    return segs


def random_points(n):
    return [Point((random.random(), random.random())) for x in range(n)]


def combo_check(bins, segments, qpoints):
    g = SegmentLocator(segments, bins)
    g2 = BruteSegmentLocator(segs)
    for pt in qpoints:
        a = g.nearest(pt)
        b = g2.nearest(pt)
        if a != b:
            print(a, b, a == b)
            global DEBUG
            DEBUG = True
            a = g.nearest(pt)
            print(a)
            a = segments[a]
            b = segments[b]
            print("pt to a (grid)", get_segment_point_dist(a, pt))
            print("pt to b (brut)", get_segment_point_dist(b, pt))
            input()
            pylab.clf()
            DEBUG = False


def brute_check(segments, qpoints):  # noqa: ARG001
    t0 = time.time()
    g2 = BruteSegmentLocator(segs)
    t1 = time.time()
    print("Created Brute in %0.4f seconds" % (t1 - t0))
    t2 = time.time()
    q = list(map(g2.nearest, qpoints))
    t3 = time.time()
    print("Brute Found %d matches in %0.4f seconds" % (len(qpoints), t3 - t2))
    print("Total Brute Time:", t3 - t0)
    print()
    return q


def grid_check(bins, segments, qpoints, visualize=False):
    t0 = time.time()
    g = SegmentLocator(segments, bins)
    t1 = time.time()
    g.grid.kd  # noqa: B018
    t2 = time.time()
    print("Created Grid in %0.4f seconds" % (t1 - t0))
    print("Created KDTree in %0.4f seconds" % (t2 - t1))
    if visualize:
        pylab.matshow(
            g.grid.mask, origin="lower", extent=g.grid.x_range + g.grid.y_range
        )

    t2 = time.time()
    list(map(g.nearest, qpoints))
    t3 = time.time()
    print("Grid Found %d matches in %0.4f seconds" % (len(qpoints), t3 - t2))
    print("Total Grid Time:", t3 - t0)
    qps = len(qpoints) / (t3 - t2)
    print("q/s:", qps)
    # print
    return qps


def binSizeTest():
    q = 100
    minN = 1000
    maxN = 10000
    stepN = 1000
    minB = 250
    maxB = 2000
    stepB = 250
    sizes = list(range(minN, maxN, stepN))
    binSizes = list(range(minB, maxB, stepB))
    results = numpy.zeros((len(sizes), len(binSizes)))
    for row, n in enumerate(sizes):
        segs = random_segments(n)
        qpts = random_points(q)
        for col, bins in enumerate(binSizes):
            print("N, Bins:", n, bins)
            qps = test_grid(bins, segs, qpts)  # noqa: F821
            results[row, col] = qps
    return results


if __name__ == "__main__":
    import pylab

    pylab.ion()

    n = 100
    q = 1000

    t0 = time.time()
    segs = random_segments(n)
    t1 = time.time()
    qpts = random_points(q)
    t2 = time.time()
    print("segments:", t1 - t0)
    print("points:", t2 - t1)
    # test_brute(segs,qpts)
    # test_grid(50, segs, qpts)

    SG = SegmentLocator(segs)
    grid = SG.grid

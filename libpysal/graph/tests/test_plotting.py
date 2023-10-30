import geodatasets
import geopandas
import numpy as np
import pytest
import shapely

from libpysal import graph

matplotlib = pytest.importorskip("matplotlib")


class TestPlotting:
    def setup_method(self):
        self.nybb = geopandas.read_file(geodatasets.get_path("nybb"))
        self.G = graph.Graph.build_contiguity(self.nybb)

        self.nybb_str = self.nybb.set_index("BoroName")
        self.G_str = graph.Graph.build_contiguity(self.nybb_str)

        self.expected_paths = [
            np.array(
                [
                    [941639.45038754, 150931.99114113],
                    [941639.45038754, 150931.99114113],
                ]
            ),
            np.array(
                [
                    [1034578.07840646, 197116.60422991],
                    [998769.11468895, 174169.76072687],
                ]
            ),
            np.array(
                [
                    [1034578.07840646, 197116.60422991],
                    [993336.96493848, 222451.43672456],
                ]
            ),
            np.array(
                [
                    [1034578.07840646, 197116.60422991],
                    [1021174.78976724, 249937.98006968],
                ]
            ),
            np.array(
                [
                    [998769.11468895, 174169.76072687],
                    [993336.96493848, 222451.43672456],
                ]
            ),
            np.array(
                [
                    [993336.96493848, 222451.43672456],
                    [1021174.78976724, 249937.98006968],
                ]
            ),
        ]

    def test_default(self):
        ax = self.G.plot(self.nybb)
        # nodes and edges
        assert len(ax.collections) == 2

        # edge geom
        linecollection = ax.collections[0]
        paths = linecollection.get_paths()
        for i, path in enumerate(paths):
            np.testing.assert_almost_equal(self.expected_paths[i], path.vertices)

        # node geom
        pathcollection = ax.collections[1]
        np.testing.assert_array_equal(
            pathcollection.get_offsets().data,
            shapely.get_coordinates(self.nybb.centroid),
        )

        # edge color
        np.testing.assert_array_equal(
            linecollection.get_color(), np.array([[0.0, 0.0, 0.0, 1.0]])
        )

        # node color
        np.testing.assert_array_equal(
            pathcollection.get_facecolor(), np.array([[0.0, 0.0, 0.0, 1.0]])
        )

    def test_string_id(self):
        ax = self.G_str.plot(self.nybb_str)
        assert len(ax.collections) == 2

        linecollection = ax.collections[0]
        paths = linecollection.get_paths()
        for i, path in enumerate(paths):
            np.testing.assert_almost_equal(self.expected_paths[i], path.vertices)

        pathcollection = ax.collections[1]
        np.testing.assert_array_equal(
            pathcollection.get_offsets().data,
            shapely.get_coordinates(self.nybb.centroid),
        )

    def test_misaligned(self):
        ax = self.G_str.plot(self.nybb_str.sort_index())
        assert len(ax.collections) == 2

        linecollection = ax.collections[0]
        paths = linecollection.get_paths()
        for i, path in enumerate(paths):
            np.testing.assert_almost_equal(self.expected_paths[i], path.vertices)

        pathcollection = ax.collections[1]
        np.testing.assert_array_equal(
            pathcollection.get_offsets().data,
            shapely.get_coordinates(self.nybb.centroid),
        )

    def test_no_nodes(self):
        ax = self.G.plot(self.nybb, nodes=False)
        assert len(ax.collections) == 1

        linecollection = ax.collections[0]
        paths = linecollection.get_paths()
        for i, path in enumerate(paths):
            np.testing.assert_almost_equal(self.expected_paths[i], path.vertices)

    def test_focal(self):
        ax = self.G_str.plot(self.nybb_str, focal="Queens")
        assert len(ax.collections) == 3

        linecollection = ax.collections[0]
        paths = linecollection.get_paths()
        np.testing.assert_almost_equal(self.expected_paths[1], paths[0].vertices)
        np.testing.assert_almost_equal(self.expected_paths[2], paths[1].vertices)
        np.testing.assert_almost_equal(self.expected_paths[3], paths[2].vertices)
        assert len(paths) == 3

        pathcollection = ax.collections[1]
        np.testing.assert_array_equal(
            pathcollection.get_offsets().data,
            shapely.get_coordinates(self.nybb.centroid)[2:],
        )

        pathcollection_focal = ax.collections[2]
        np.testing.assert_array_equal(
            pathcollection_focal.get_offsets().data,
            shapely.get_coordinates(self.nybb.centroid)[[1]],
        )

    def test_focal_array(self):
        ax = self.G_str.plot(self.nybb_str, focal=["Queens", "Bronx"])
        assert len(ax.collections) == 3

        linecollection = ax.collections[0]
        paths = linecollection.get_paths()
        np.testing.assert_almost_equal(self.expected_paths[1], paths[0].vertices)
        np.testing.assert_almost_equal(self.expected_paths[2], paths[1].vertices)
        np.testing.assert_almost_equal(self.expected_paths[3], paths[2].vertices)
        np.testing.assert_almost_equal(self.expected_paths[5], paths[3].vertices)
        assert len(paths) == 4

        pathcollection = ax.collections[1]
        np.testing.assert_array_equal(
            pathcollection.get_offsets().data,
            shapely.get_coordinates(self.nybb.centroid)[1:],
        )

        pathcollection_focal = ax.collections[2]
        np.testing.assert_array_equal(
            pathcollection_focal.get_offsets().data,
            shapely.get_coordinates(self.nybb.centroid)[[1, -1]],
        )

    def test_color(self):
        ax = self.G.plot(self.nybb, color="red")

        linecollection = ax.collections[0]
        np.testing.assert_array_equal(
            linecollection.get_color(), np.array([[1.0, 0.0, 0.0, 1.0]])
        )

        pathcollection = ax.collections[1]
        np.testing.assert_array_equal(
            pathcollection.get_facecolor(), np.array([[1.0, 0.0, 0.0, 1.0]])
        )

    def test_kws(self):
        ax = self.G.plot(
            self.nybb, edge_kws=dict(linestyle="dotted"), node_kws=dict(marker="+")
        )

        linecollection = ax.collections[0]
        assert linecollection.get_linestyle() == [(0.0, [1.5, 2.4749999999999996])]

        pathcollection = ax.collections[1]
        np.testing.assert_array_equal(
            pathcollection.get_paths()[0].vertices,
            np.array([[-0.5, 0.0], [0.5, 0.0], [0.0, -0.5], [0.0, 0.5]]),
        )

    def test_ax(self):
        ax = self.nybb.plot()
        self.G.plot(self.nybb, ax=ax)
        assert len(ax.collections) == 3

        # edge geom
        linecollection = ax.collections[1]
        paths = linecollection.get_paths()
        for i, path in enumerate(paths):
            np.testing.assert_almost_equal(self.expected_paths[i], path.vertices)

        # node geom
        pathcollection = ax.collections[2]
        np.testing.assert_array_equal(
            pathcollection.get_offsets().data,
            shapely.get_coordinates(self.nybb.centroid),
        )

    def test_figsize(self):
        ax = self.G.plot(self.nybb, figsize=(12, 12))
        np.testing.assert_array_equal(
            ax.figure.get_size_inches(), np.array([12.0, 12.0])
        )

    def test_limit_extent(self):
        ax = self.G_str.plot(self.nybb_str)
        self.G_str.plot(
            self.nybb_str, focal="Bronx", limit_extent=True, ax=ax, color="red"
        )
        assert ax.get_ylim() == (194475.53543792566, 252579.0488616723)
        assert ax.get_xlim() == (991274.9092650851, 1036640.134079854)

    def test_focal_kws(self):
        ax = self.G_str.plot(
            self.nybb_str,
            focal="Queens",
            focal_kws=dict(color="blue"),
            node_kws=dict(edgecolor="pink"),
        )

        pathcollection = ax.collections[1]
        np.testing.assert_array_almost_equal(
            pathcollection.get_edgecolor(),
            np.array([[1.0, 0.75294118, 0.79607843, 1.0]]),
        )

        pathcollection_focal = ax.collections[2]
        # inherit node_kws
        np.testing.assert_array_almost_equal(
            pathcollection_focal.get_edgecolor(),
            np.array([[1.0, 0.75294118, 0.79607843, 1.0]]),
        )
        # apply own kws
        np.testing.assert_array_equal(
            pathcollection_focal.get_facecolor(), np.array([[0.0, 0.0, 1.0, 1.0]])
        )

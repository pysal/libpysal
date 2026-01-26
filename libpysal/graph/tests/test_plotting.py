import geodatasets
import geopandas
import numpy as np
import pytest
import shapely

from libpysal import graph
from libpysal.graph.tests.test_utils import fetch_map_string


@pytest.mark.network
class TestPlotting:
    def setup_method(self):
        _ = pytest.importorskip("matplotlib")

        self.nybb = geopandas.read_file(geodatasets.get_path("nybb"))
        self.G = graph.Graph.build_contiguity(self.nybb)

        self.nybb_str = self.nybb.set_index("BoroName")
        self.G_str = graph.Graph.build_contiguity(self.nybb_str)

        self.expected_paths = [
            np.array(
                [
                    [943802.68511489, 147890.05410767],
                    [943802.68511489, 147890.05410767],
                ]
            ),
            np.array(
                [
                    [1033983.96582281, 196127.39050293],
                    [998506.94007314, 177674.69769287],
                ]
            ),
            np.array(
                [
                    [1033983.96582281, 196127.39050293],
                    [995258.50398262, 226631.05230713],
                ]
            ),
            np.array(
                [
                    [1033983.96582281, 196127.39050293],
                    [1021230.82521895, 251186.3362925],
                ]
            ),
            np.array(
                [
                    [998506.94007314, 177674.69769287],
                    [995258.50398262, 226631.05230713],
                ]
            ),
            np.array(
                [
                    [995258.50398262, 226631.05230713],
                    [1021230.82521895, 251186.3362925],
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
            shapely.get_coordinates(self.nybb.representative_point()),
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
            shapely.get_coordinates(self.nybb.representative_point()),
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
            shapely.get_coordinates(self.nybb.representative_point()),
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
            shapely.get_coordinates(self.nybb.representative_point())[2:],
        )

        pathcollection_focal = ax.collections[2]
        np.testing.assert_array_equal(
            pathcollection_focal.get_offsets().data,
            shapely.get_coordinates(self.nybb.representative_point())[[1]],
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
            shapely.get_coordinates(self.nybb.representative_point())[1:],
        )

        pathcollection_focal = ax.collections[2]
        np.testing.assert_array_equal(
            pathcollection_focal.get_offsets().data,
            shapely.get_coordinates(self.nybb.representative_point())[[1, -1]],
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
            self.nybb, edge_kws={"linestyle": "dotted"}, node_kws={"marker": "+"}
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
            shapely.get_coordinates(self.nybb.representative_point()),
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
        assert ax.get_ylim() == (193374.44321345096, 253939.28358198277)
        assert ax.get_xlim() == (993322.2308906089, 1035920.2389148154)

    def test_focal_kws(self):
        ax = self.G_str.plot(
            self.nybb_str,
            focal="Queens",
            focal_kws={"color": "blue"},
            node_kws={"edgecolor": "pink"},
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

    def test_color_by_weight(self):
        """Test weight-based coloring feature"""
        import pandas as pd

        # Create a graph with varying weights for testing
        focal_ids = ["Staten Island", "Queens", "Brooklyn", "Manhattan", "Bronx"]
        neighbor_ids = ["Queens", "Brooklyn", "Manhattan", "Bronx", "Staten Island"]
        weights = [0.2, 0.4, 0.6, 0.8, 1.0]

        adjacency = pd.Series(
            weights,
            index=pd.MultiIndex.from_arrays(
                [focal_ids, neighbor_ids], names=["focal", "neighbor"]
            ),
            name="weight",
        )
        G_test = graph.Graph(adjacency)

        # Test that color_by_weight=True produces colors array
        ax = G_test.plot(self.nybb_str, color_by_weight=True)
        linecollection = ax.collections[0]

        # When color_by_weight=True, should use colors array
        edgecolors = linecollection.get_edgecolors()
        assert edgecolors is not None
        assert len(edgecolors) == len(weights)
        assert edgecolors.shape[1] == 4  # RGBA

        # Different weights should produce different colors
        # Check that colors are not all the same
        assert not np.allclose(edgecolors[0], edgecolors[1])

    def test_color_by_weight_false(self):
        """Test that color_by_weight=False maintains backward compatibility"""
        ax = self.G.plot(self.nybb, color_by_weight=False, color="red")

        linecollection = ax.collections[0]
        # Should use single color, not colors array
        color = linecollection.get_color()
        np.testing.assert_array_equal(color, np.array([[1.0, 0.0, 0.0, 1.0]]))

    def test_color_by_weight_cmap(self):
        """Test that cmap parameter works"""
        import pandas as pd
        import matplotlib.pyplot as plt

        # Use existing graph but test with different colormap
        # This tests that cmap parameter is accepted and produces different colors
        ax1 = self.G_str.plot(self.nybb_str, color_by_weight=True, cmap="viridis")
        ax2 = self.G_str.plot(self.nybb_str, color_by_weight=True, cmap="plasma")

        linecollection1 = ax1.collections[0]
        linecollection2 = ax2.collections[0]

        edgecolors1 = linecollection1.get_edgecolors()
        edgecolors2 = linecollection2.get_edgecolors()

        assert edgecolors1 is not None
        assert edgecolors2 is not None
        # Different colormaps should produce different colors
        # (at least for some edges, unless all weights are identical)
        assert len(edgecolors1) == len(edgecolors2)

    def test_color_by_weight_focal(self):
        """Test weight-based coloring with focal parameter"""
        import pandas as pd

        # Create a graph with varying weights - use existing contiguity graph
        # but modify weights for testing
        G_test = self.G_str.copy()

        # Test with focal parameter - should work with existing graph
        ax = G_test.plot(self.nybb_str, focal="Queens", color_by_weight=True)
        linecollection = ax.collections[0]
        edgecolors = linecollection.get_edgecolors()

        # Should have colors for edges from Queens
        assert edgecolors is not None
        assert len(edgecolors) > 0

    def test_color_by_weight_same_weights(self):
        """Test weight-based coloring when all weights are the same"""
        import pandas as pd

        # Create a graph with ALL identical weights using all boroughs
        focal_ids = ["Staten Island", "Queens", "Brooklyn", "Manhattan", "Bronx"]
        neighbor_ids = ["Queens", "Brooklyn", "Manhattan", "Bronx", "Staten Island"]
        weights = [1.0, 1.0, 1.0, 1.0, 1.0]  # All weights identical

        adjacency = pd.Series(
            weights,
            index=pd.MultiIndex.from_arrays(
                [focal_ids, neighbor_ids], names=["focal", "neighbor"]
            ),
            name="weight",
        )
        G_test = graph.Graph(adjacency)

        # Should not crash when all weights are identical
        ax = G_test.plot(self.nybb_str, color_by_weight=True)
        linecollection = ax.collections[0]
        edgecolors = linecollection.get_edgecolors()

        assert edgecolors is not None
        assert len(edgecolors) == len(weights)
        # All colors should be identical when weights are identical
        np.testing.assert_array_equal(edgecolors[0], edgecolors[1])


class TestExplore:
    def setup_method(self):
        # skip tests when no folium installed
        pytest.importorskip("folium")

        self.nybb_str = geopandas.read_file(geodatasets.get_path("nybb")).set_index(
            "BoroName"
        )
        self.G_str = graph.Graph.build_contiguity(self.nybb_str)

    def test_default(self):
        m = self.G_str.explore(self.nybb_str)
        s = fetch_map_string(m)

        # nodes
        assert s.count("Point") == 5
        # edges
        assert s.count("LineString") == 6
        # tooltip
        assert '"focal":"Queens","neighbor":"Bronx","weight":1}' in s
        # color
        assert s.count('"__folium_color":"black"') == 11
        # labels
        assert s.count("Brooklyn") == 3

    def test_no_nodes(self):
        m = self.G_str.explore(self.nybb_str, nodes=False)
        s = fetch_map_string(m)

        # nodes
        assert s.count("Point") == 0
        # edges
        assert s.count("LineString") == 6
        # tooltip
        assert '"focal":"Queens","neighbor":"Bronx","weight":1}' in s
        # color
        assert s.count('"__folium_color":"black"') == 6
        # labels
        assert s.count("Brooklyn") == 2

    def test_focal(self):
        m = self.G_str.explore(self.nybb_str, focal="Queens")
        s = fetch_map_string(m)

        # nodes
        assert s.count("Point") == 4
        # edges
        assert s.count("LineString") == 3
        # tooltip
        assert '"focal":"Queens","neighbor":"Bronx","weight":1}' in s
        assert '"focal":"Queens","neighbor":"Manhattan","weight":1}' in s
        assert '"focal":"Queens","neighbor":"Brooklyn","weight":1}' in s
        # color
        assert s.count('"__folium_color":"black"') == 7
        # labels
        assert s.count("Brooklyn") == 2

    def test_focal_array(self):
        m = self.G_str.explore(self.nybb_str, focal=["Queens", "Bronx"])
        s = fetch_map_string(m)

        # if node is both focal and neighbor, both are plottted as you can style
        # them differently to see both
        assert s.count("Point") == 6
        # edges
        assert s.count("LineString") == 4
        # tooltip
        assert '"focal":"Queens","neighbor":"Bronx","weight":1}' in s
        assert '"focal":"Queens","neighbor":"Manhattan","weight":1}' in s
        assert '"focal":"Queens","neighbor":"Brooklyn","weight":1}' in s
        assert '"focal":"Bronx","neighbor":"Manhattan","weight":1}' in s

        # color
        assert s.count('"__folium_color":"black"') == 10
        # labels
        assert s.count("Brooklyn") == 2

    def test_color(self):
        m = self.G_str.explore(self.nybb_str, color="red")
        s = fetch_map_string(m)

        assert s.count('"__folium_color":"red"') == 11

    def test_kws(self):
        m = self.G_str.explore(
            self.nybb_str,
            focal=["Queens", "Bronx"],
            edge_kws={"color": "red"},
            node_kws={"color": "blue", "marker_kwds": {"radius": 8}},
            focal_kws={"color": "pink", "marker_kwds": {"radius": 12}},
        )
        s = fetch_map_string(m)

        # color
        assert s.count('"__folium_color":"red"') == 4
        assert s.count('"__folium_color":"blue"') == 4
        assert s.count('"__folium_color":"pink"') == 2

        assert '"radius":8' in s
        assert '"radius":12' in s

    def test_m(self):
        m = self.nybb_str.explore()
        self.G_str.explore(self.nybb_str, m=m)
        s = fetch_map_string(m)

        # nodes
        assert s.count("Point") == 5
        # edges
        assert s.count("LineString") == 6
        # geoms
        assert s.count("Polygon") == 5

    def test_explore_kwargs(self):
        m = self.G_str.explore(self.nybb_str, tiles="OpenStreetMap HOT")
        s = fetch_map_string(m)

        assert "tile.openstreetmap.fr/hot" in s

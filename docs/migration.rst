
W to Graph Member Comparisions
==============================


Overview
--------

This guide compares the members (attributes and methods) from the
`W` class and the `Graph` class.

It is intended for developers. Users interested in migrating to the
new Graph class from W should see the `migration guide <user-guide/graph/w_g_migration.html>`_.


Members common to W and Graph
-----------------------------


+-----------------------------------------------------------------------------------------+------------------+
| Member                                                                                  |   Typee          |
+=========================================================================================+==================+
| `asymmetry <generated/libpysal.graph.Graph.html#libpysal.graph.Graph.asymmetry>`_       |  builtins.method |
+-----------------------------------------------------------------------------------------+------------------+
| `from_sparse <generated/libpysal.graph.Graph.html#libpysal.graph.Graph.from_sparse>`_   |  builtins.method |
+-----------------------------------------------------------------------------------------+------------------+
| `n <generated/libpysal.graph.Graph.html#libpysal.graph.Graph.n>`_                       |  builtins.int    |
+-----------------------------------------------------------------------------------------+------------------+
| `n_components <generated/libpysal.graph.Graph.html#libpysal.graph.Graph.n_components>`_ |  builtins.int    |
+-----------------------------------------------------------------------------------------+------------------+
| `neighbors <generated/libpysal.graph.Graph.html#libpysal.graph.Graph.neighbors>`_       |  builtins.dict   |
+-----------------------------------------------------------------------------------------+------------------+
| `pct_nonzero <generated/libpysal.graph.Graph.html#libpysal.graph.Graph.pct_nonzero>`_   |  builtins.float  |
+-----------------------------------------------------------------------------------------+------------------+
| `plot <generated/libpysal.graph.Graph.html#libpysal.graph.Graph.plot>`_                 |  builtins.method |
+-----------------------------------------------------------------------------------------+------------------+
| `to_networkx <generated/libpysal.graph.Graph.html#libpysal.graph.Graph.to_networkx>`_   |  builtins.method |
+-----------------------------------------------------------------------------------------+------------------+
| `weights <generated/libpysal.graph.Graph.html#libpysal.graph.Graph.weights>`_           |  builtins.dict   |
+-----------------------------------------------------------------------------------------+------------------+


Members common to W and Graph with different types
--------------------------------------------------



+------------------+-----------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------+
| Member           |  Queen Type                                                                                   |  Graph Type                                                                                              |
+==================+===============================================================================================+==========================================================================================================+
| cardinalities    | `builtins.dict <generated/libpysal.weights.W.html#libpysal.weights.W.cardinalities>`_         | `pandas.core.series.Series <generated/libpysal.graph.Graph.html#libpysal.graph.Graph.cardinalities>`_    |
+------------------+-----------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------+
| component_labels | `numpy.ndarray <generated/libpysal.weights.W.html#libpysal.weights.W.component_labels>`_      | `pandas.core.series.Series <generated/libpysal.graph.Graph.html#libpysal.graph.Graph.component_labels>`_ |
+------------------+-----------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------+
| nonzero          | `builtins.int <generated/libpysal.weights.W.html#libpysal.weights.W.nonzero>`_                | `numpy.int64 <generated/libpysal.graph.Graph.html#libpysal.graph.Graph.nonzero>`_                        |
+------------------+-----------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------+
| sparse           | `scipy.sparse._csr.csr_matrix <generated/libpysal.weights.W.html#libpysal.weights.W.sparse>`_ | `scipy.sparse._csr.csr_array <generated/libpysal.graph.Graph.html#libpysal.graph.Graph.sparse>`_         |
+------------------+-----------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------+
| transform        | `builtins.str <generated/libpysal.weights.W.html#libpysal.weights.W.transform>`_              | `builtins.method <generated/libpysal.graph.Graph.html#libpysal.graph.Graph.transform>`_                  |
+------------------+-----------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------+


Members unique to W
-------------------



+---------------------------------------------------------------------------------------------+-----------------+
| Member                                                                                      |   Type          |
+=============================================================================================+=================+
| `asymmetries <generated/libpysal.weights.W.html#libpysal.weights.W.asymmetries>`_           | builtins.list   |
+---------------------------------------------------------------------------------------------+-----------------+
| `diagW2 <generated/libpysal.weights.W.html#libpysal.weights.W.diagW2>`_                     | numpy.ndarray   |
+---------------------------------------------------------------------------------------------+-----------------+
| `diagWtW <generated/libpysal.weights.W.html#libpysal.weights.W.diagWtW>`_                   | numpy.ndarray   |
+---------------------------------------------------------------------------------------------+-----------------+
| `diagWtW_WW <generated/libpysal.weights.W.html#libpysal.weights.W.diagWtW_WW>`_             | numpy.ndarray   |
+---------------------------------------------------------------------------------------------+-----------------+
| `from_WSP <generated/libpysal.weights.W.html#libpysal.weights.W.from_WSP>`_                 | builtins.method |
+---------------------------------------------------------------------------------------------+-----------------+
| `from_adjlist <generated/libpysal.weights.W.html#libpysal.weights.W.from_adjlist>`_         | builtins.method |
+---------------------------------------------------------------------------------------------+-----------------+
| `from_dataframe <generated/libpysal.weights.W.html#libpysal.weights.W.from_dataframe>`_     | builtins.method |
+---------------------------------------------------------------------------------------------+-----------------+
| `from_file <generated/libpysal.weights.W.html#libpysal.weights.W.from_file>`_               | builtins.method |
+---------------------------------------------------------------------------------------------+-----------------+
| `from_iterable <generated/libpysal.weights.W.html#libpysal.weights.W.from_iterable>`_       | builtins.method |
+---------------------------------------------------------------------------------------------+-----------------+
| `from_networkx <generated/libpysal.weights.W.html#libpysal.weights.W.from_networkx>`_       | builtins.method |
+---------------------------------------------------------------------------------------------+-----------------+
| `from_shapefile <generated/libpysal.weights.W.html#libpysal.weights.W.from_shapefile>`_     | builtins.method |
+---------------------------------------------------------------------------------------------+-----------------+
| `from_xarray <generated/libpysal.weights.W.html#libpysal.weights.W.from_xarray>`_           | builtins.method |
+---------------------------------------------------------------------------------------------+-----------------+
| `full <generated/libpysal.weights.W.html#libpysal.weights.W.full>`_                         | builtins.method |
+---------------------------------------------------------------------------------------------+-----------------+
| `get_transform <generated/libpysal.weights.W.html#libpysal.weights.W.get_transform>`_       | builtins.method |
+---------------------------------------------------------------------------------------------+-----------------+
| `histogram <generated/libpysal.weights.W.html#libpysal.weights.W.histogram>`_               | builtins.list   |
+---------------------------------------------------------------------------------------------+-----------------+
| `id2i <generated/libpysal.weights.W.html#libpysal.weights.W.id2i>`_                         | builtins.dict   |
+---------------------------------------------------------------------------------------------+-----------------+
| `id_order <generated/libpysal.weights.W.html#libpysal.weights.W.id_order>`_                 | builtins.list   |
+---------------------------------------------------------------------------------------------+-----------------+
| `id_order_set <generated/libpysal.weights.W.html#libpysal.weights.W.id_order_set>`_         | builtins.bool   |
+---------------------------------------------------------------------------------------------+-----------------+
| `islands <generated/libpysal.weights.W.html#libpysal.weights.W.islands>`_                   | builtins.list   |
+---------------------------------------------------------------------------------------------+-----------------+
| `max_neighbors <generated/libpysal.weights.W.html#libpysal.weights.W.max_neighbors>`_       | builtins.int    |
+---------------------------------------------------------------------------------------------+-----------------+
| `mean_neighbors <generated/libpysal.weights.W.html#libpysal.weights.W.mean_neighbors>`_     | numpy.float64   |
+---------------------------------------------------------------------------------------------+-----------------+
| `min_neighbors <generated/libpysal.weights.W.html#libpysal.weights.W.min_neighbors>`_       | builtins.int    |
+---------------------------------------------------------------------------------------------+-----------------+
| `neighbor_offsets <generated/libpysal.weights.W.html#libpysal.weights.W.neighbor_offsets>`_ | builtins.dict   |
+---------------------------------------------------------------------------------------------+-----------------+
| `remap_ids <generated/libpysal.weights.W.html#libpysal.weights.W.remap_ids>`_               | builtins.method |
+---------------------------------------------------------------------------------------------+-----------------+
| `s0 <generated/libpysal.weights.W.html#libpysal.weights.W.s0>`_                             | numpy.float64   |
+---------------------------------------------------------------------------------------------+-----------------+
| `s1 <generated/libpysal.weights.W.html#libpysal.weights.W.s1>`_                             | numpy.float64   |
+---------------------------------------------------------------------------------------------+-----------------+
| `s2 <generated/libpysal.weights.W.html#libpysal.weights.W.s2>`_                             | numpy.float64   |
+---------------------------------------------------------------------------------------------+-----------------+
| `s2array <generated/libpysal.weights.W.html#libpysal.weights.W.s2array>`_                   | numpy.ndarray   |
+---------------------------------------------------------------------------------------------+-----------------+
| `sd <generated/libpysal.weights.W.html#libpysal.weights.W.sd>`_                             | numpy.float64   |
+---------------------------------------------------------------------------------------------+-----------------+
| `set_shapefile <generated/libpysal.weights.W.html#libpysal.weights.W.set_shapefile>`_       | builtins.method |
+---------------------------------------------------------------------------------------------+-----------------+
| `set_transform <generated/libpysal.weights.W.html#libpysal.weights.W.set_transform>`_       | builtins.method |
+---------------------------------------------------------------------------------------------+-----------------+
| `silence_warnings <generated/libpysal.weights.W.html#libpysal.weights.W.silence_warnings>`_ | builtins.bool   |
+---------------------------------------------------------------------------------------------+-----------------+
| `symmetrize <generated/libpysal.weights.W.html#libpysal.weights.W.symmetrize>`_             | builtins.method |
+---------------------------------------------------------------------------------------------+-----------------+
| `to_WSP <generated/libpysal.weights.W.html#libpysal.weights.W.to_WSP>`_                     | builtins.method |
+---------------------------------------------------------------------------------------------+-----------------+
| `to_adjlist <generated/libpysal.weights.W.html#libpysal.weights.W.to_adjlist>`_             | builtins.method |
+---------------------------------------------------------------------------------------------+-----------------+
| `to_file <generated/libpysal.weights.W.html#libpysal.weights.W.to_file>`_                   | builtins.method |
+---------------------------------------------------------------------------------------------+-----------------+
| `to_sparse <generated/libpysal.weights.W.html#libpysal.weights.W.to_sparse>`_               | builtins.method |
+---------------------------------------------------------------------------------------------+-----------------+
| `transformations <generated/libpysal.weights.W.html#libpysal.weights.W.transformations>`_   | builtins.dict   |
+---------------------------------------------------------------------------------------------+-----------------+
| `trcW2 <generated/libpysal.weights.W.html#libpysal.weights.W.trcW2>`_                       | numpy.float64   |
+---------------------------------------------------------------------------------------------+-----------------+
| `trcWtW <generated/libpysal.weights.W.html#libpysal.weights.W.trcWtW>`_                     | numpy.float64   |
+---------------------------------------------------------------------------------------------+-----------------+
| `trcWtW_WW <generated/libpysal.weights.W.html#libpysal.weights.W.trcWtW_WW>`_               | numpy.float64   |
+---------------------------------------------------------------------------------------------+-----------------+


Members unique to Graph
-----------------------



+---------------------------------------------------------------------------------------------------------------+--------------------------------+
| Member                                                                                                        |   Type                         |
+===============================================================================================================+================================+
| `adjacency <generated/libpysal.graph.Graph.html#libpysal.graph.Graph.adjacency>`_                             | pandas.core.series.Series      |
+---------------------------------------------------------------------------------------------------------------+--------------------------------+
| `aggregate <generated/libpysal.graph.Graph.html#libpysal.graph.Graph.aggregate>`_                             | builtins.method                |
+---------------------------------------------------------------------------------------------------------------+--------------------------------+
| `apply <generated/libpysal.graph.Graph.html#libpysal.graph.Graph.apply>`_                                     | builtins.method                |
+---------------------------------------------------------------------------------------------------------------+--------------------------------+
| `assign_self_weight <generated/libpysal.graph.Graph.html#libpysal.graph.Graph.assign_self_weight>`_           | builtins.method                |
+---------------------------------------------------------------------------------------------------------------+--------------------------------+
| `build_block_contiguity <generated/libpysal.graph.Graph.html#libpysal.graph.Graph.build_block_contiguity>`_   | builtins.method                |
+---------------------------------------------------------------------------------------------------------------+--------------------------------+
| `build_contiguity <generated/libpysal.graph.Graph.html#libpysal.graph.Graph.build_contiguity>`_               | builtins.method                |
+---------------------------------------------------------------------------------------------------------------+--------------------------------+
| `build_distance_band <generated/libpysal.graph.Graph.html#libpysal.graph.Graph.build_distance_band>`_         | builtins.method                |
+---------------------------------------------------------------------------------------------------------------+--------------------------------+
| `build_fuzzy_contiguity <generated/libpysal.graph.Graph.html#libpysal.graph.Graph.build_fuzzy_contiguity>`_   | builtins.method                |
+---------------------------------------------------------------------------------------------------------------+--------------------------------+
| `build_h3 <generated/libpysal.graph.Graph.html#libpysal.graph.Graph.build_h3>`_                               | builtins.method                |
+---------------------------------------------------------------------------------------------------------------+--------------------------------+
| `build_kernel <generated/libpysal.graph.Graph.html#libpysal.graph.Graph.build_kernel>`_                       | builtins.method                |
+---------------------------------------------------------------------------------------------------------------+--------------------------------+
| `build_knn <generated/libpysal.graph.Graph.html#libpysal.graph.Graph.build_knn>`_                             | builtins.method                |
+---------------------------------------------------------------------------------------------------------------+--------------------------------+
| `build_raster_contiguity <generated/libpysal.graph.Graph.html#libpysal.graph.Graph.build_raster_contiguity>`_ | builtins.method                |
+---------------------------------------------------------------------------------------------------------------+--------------------------------+
| `build_spatial_matches <generated/libpysal.graph.Graph.html#libpysal.graph.Graph.build_spatial_matches>`_     | builtins.method                |
+---------------------------------------------------------------------------------------------------------------+--------------------------------+
| `build_triangulation <generated/libpysal.graph.Graph.html#libpysal.graph.Graph.build_triangulation>`_         | builtins.method                |
+---------------------------------------------------------------------------------------------------------------+--------------------------------+
| `copy <generated/libpysal.graph.Graph.html#libpysal.graph.Graph.copy>`_                                       | builtins.method                |
+---------------------------------------------------------------------------------------------------------------+--------------------------------+
| `describe <generated/libpysal.graph.Graph.html#libpysal.graph.Graph.describe>`_                               | builtins.method                |
+---------------------------------------------------------------------------------------------------------------+--------------------------------+
| `difference <generated/libpysal.graph.Graph.html#libpysal.graph.Graph.difference>`_                           | builtins.method                |
+---------------------------------------------------------------------------------------------------------------+--------------------------------+
| `eliminate_zeros <generated/libpysal.graph.Graph.html#libpysal.graph.Graph.eliminate_zeros>`_                 | builtins.method                |
+---------------------------------------------------------------------------------------------------------------+--------------------------------+
| `equals <generated/libpysal.graph.Graph.html#libpysal.graph.Graph.equals>`_                                   | builtins.method                |
+---------------------------------------------------------------------------------------------------------------+--------------------------------+
| `explore <generated/libpysal.graph.Graph.html#libpysal.graph.Graph.explore>`_                                 | builtins.method                |
+---------------------------------------------------------------------------------------------------------------+--------------------------------+
| `from_W <generated/libpysal.graph.Graph.html#libpysal.graph.Graph.from_W>`_                                   | builtins.method                |
+---------------------------------------------------------------------------------------------------------------+--------------------------------+
| `from_adjacency <generated/libpysal.graph.Graph.html#libpysal.graph.Graph.from_adjacency>`_                   | builtins.method                |
+---------------------------------------------------------------------------------------------------------------+--------------------------------+
| `from_arrays <generated/libpysal.graph.Graph.html#libpysal.graph.Graph.from_arrays>`_                         | builtins.method                |
+---------------------------------------------------------------------------------------------------------------+--------------------------------+
| `from_dicts <generated/libpysal.graph.Graph.html#libpysal.graph.Graph.from_dicts>`_                           | builtins.method                |
+---------------------------------------------------------------------------------------------------------------+--------------------------------+
| `from_weights_dict <generated/libpysal.graph.Graph.html#libpysal.graph.Graph.from_weights_dict>`_             | builtins.method                |
+---------------------------------------------------------------------------------------------------------------+--------------------------------+
| `generate_da <generated/libpysal.graph.Graph.html#libpysal.graph.Graph.generate_da>`_                         | builtins.method                |
+---------------------------------------------------------------------------------------------------------------+--------------------------------+
| `higher_order <generated/libpysal.graph.Graph.html#libpysal.graph.Graph.higher_order>`_                       | builtins.method                |
+---------------------------------------------------------------------------------------------------------------+--------------------------------+
| `index_pairs <generated/libpysal.graph.Graph.html#libpysal.graph.Graph.index_pairs>`_                         | builtins.tuple                 |
+---------------------------------------------------------------------------------------------------------------+--------------------------------+
| `intersection <generated/libpysal.graph.Graph.html#libpysal.graph.Graph.intersection>`_                       | builtins.method                |
+---------------------------------------------------------------------------------------------------------------+--------------------------------+
| `intersects <generated/libpysal.graph.Graph.html#libpysal.graph.Graph.intersects>`_                           | builtins.method                |
+---------------------------------------------------------------------------------------------------------------+--------------------------------+
| `isolates <generated/libpysal.graph.Graph.html#libpysal.graph.Graph.isolates>`_                               | pandas.core.indexes.base.Index |
+---------------------------------------------------------------------------------------------------------------+--------------------------------+
| `isomorphic <generated/libpysal.graph.Graph.html#libpysal.graph.Graph.isomorphic>`_                           | builtins.method                |
+---------------------------------------------------------------------------------------------------------------+--------------------------------+
| `issubgraph <generated/libpysal.graph.Graph.html#libpysal.graph.Graph.issubgraph>`_                           | builtins.method                |
+---------------------------------------------------------------------------------------------------------------+--------------------------------+
| `lag <generated/libpysal.graph.Graph.html#libpysal.graph.Graph.lag>`_                                         | builtins.method                |
+---------------------------------------------------------------------------------------------------------------+--------------------------------+
| `n_edges <generated/libpysal.graph.Graph.html#libpysal.graph.Graph.n_edges>`_                                 | builtins.int                   |
+---------------------------------------------------------------------------------------------------------------+--------------------------------+
| `n_nodes <generated/libpysal.graph.Graph.html#libpysal.graph.Graph.n_nodes>`_                                 | builtins.int                   |
+---------------------------------------------------------------------------------------------------------------+--------------------------------+
| `subgraph <generated/libpysal.graph.Graph.html#libpysal.graph.Graph.subgraph>`_                               | builtins.method                |
+---------------------------------------------------------------------------------------------------------------+--------------------------------+
| `summary <generated/libpysal.graph.Graph.html#libpysal.graph.Graph.summary>`_                                 | builtins.method                |
+---------------------------------------------------------------------------------------------------------------+--------------------------------+
| `symmetric_difference <generated/libpysal.graph.Graph.html#libpysal.graph.Graph.symmetric_difference>`_       | builtins.method                |
+---------------------------------------------------------------------------------------------------------------+--------------------------------+
| `to_W <generated/libpysal.graph.Graph.html#libpysal.graph.Graph.to_W>`_                                       | builtins.method                |
+---------------------------------------------------------------------------------------------------------------+--------------------------------+
| `to_gal <generated/libpysal.graph.Graph.html#libpysal.graph.Graph.to_gal>`_                                   | builtins.method                |
+---------------------------------------------------------------------------------------------------------------+--------------------------------+
| `to_gwt <generated/libpysal.graph.Graph.html#libpysal.graph.Graph.to_gwt>`_                                   | builtins.method                |
+---------------------------------------------------------------------------------------------------------------+--------------------------------+
| `to_parquet <generated/libpysal.graph.Graph.html#libpysal.graph.Graph.to_parquet>`_                           | builtins.method                |
+---------------------------------------------------------------------------------------------------------------+--------------------------------+
| `transformation <generated/libpysal.graph.Graph.html#libpysal.graph.Graph.transformation>`_                   | builtins.str                   |
+---------------------------------------------------------------------------------------------------------------+--------------------------------+
| `union <generated/libpysal.graph.Graph.html#libpysal.graph.Graph.union>`_                                     | builtins.method                |
+---------------------------------------------------------------------------------------------------------------+--------------------------------+
| `unique_ids <generated/libpysal.graph.Graph.html#libpysal.graph.Graph.unique_ids>`_                           | pandas.core.indexes.base.Index |
+---------------------------------------------------------------------------------------------------------------+--------------------------------+
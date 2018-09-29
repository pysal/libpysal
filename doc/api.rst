.. _api_ref:

.. currentmodule:: libpysal


libpysal API reference
======================

Spatial Weights
---------------

.. autosummary::
   :toctree: generated/

   libpysal.weights.W

Distance Weights
++++++++++++++++
.. autosummary::
   :toctree: generated/

   libpysal.weights.DistanceBand
   libpysal.weights.Kernel
   libpysal.weights.KNN

Contiguity Weights
++++++++++++++++++

.. autosummary::
   :toctree: generated/

   libpysal.weights.Queen
   libpysal.weights.Rook
   libpysal.weights.Voronoi
   libpysal.weights.W

spint Weights
+++++++++++++

.. autosummary::
   :toctree: generated/

   libpysal.weights.WSP
   libpysal.weights.netW
   libpysal.weights.mat2L
   libpysal.weights.ODW
   libpysal.weights.vecW
   libpysal.weights.lat2W


Weights Util Classes and Functions
++++++++++++++++++++++++++++++++++

.. autosummary::
   :toctree: generated/

   libpysal.weights.block_weights
   libpysal.weights.comb
   libpysal.weights.order
   libpysal.weights.higher_order
   libpysal.weights.shimbel
   libpysal.weights.remap_ids
   libpysal.weights.full2W
   libpysal.weights.full
   libpysal.weights.WSP2W
   libpysal.weights.get_ids
   libpysal.weights.get_points_array_from_shapefile

Weights user Classes and Functions
++++++++++++++++++++++++++++++++++

.. autosummary::
   :toctree: generated/

   libpysal.weights.min_threshold_distance
   libpysal.weights.lat2SW
   libpysal.weights.w_local_cluster
   libpysal.weights.higher_order_sp
   libpysal.weights.hexLat2W
   libpysal.weights.attach_islands
   libpysal.weights.nonplanar_neighbors
   libpysal.weights.fuzzy_contiguity
   libpysal.weights.min_threshold_dist_from_shapefile
   libpysal.weights.build_lattice_shapefile
   libpysal.weights.spw_from_gal


Set Theoretic Weights
+++++++++++++++++++++

.. autosummary::
   :toctree: generated/

   libpysal.weights.w_union
   libpysal.weights.w_intersection
   libpysal.weights.w_difference
   libpysal.weights.w_symmetric_difference
   libpysal.weights.w_subset
   libpysal.weights.w_clip


Spatial Lag
+++++++++++

.. autosummary::
   :toctree: generated/

   libpysal.weights.lag_spatial
   libpysal.weights.lag_categorical
          

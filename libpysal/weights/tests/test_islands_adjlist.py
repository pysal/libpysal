import numpy as np
import pytest

from ... import weights


class TestIslandsAdjlist:
    def test_islands_representation(self):
        # Create a weights object with islands
        neighbors = {0: [1], 1: [0, 2], 2: [1], 3: [], 4: []}  # 3 and 4 are islands
        w_islands = weights.W(neighbors, silence_warnings=True)
        
        # Convert to adjacency list with drop_islands=False to include islands
        adjlist = w_islands.to_adjlist(drop_islands=False)
        
        # Check that islands are represented as (focal, focal, 0)
        islands_in_adjlist = adjlist[(adjlist['focal'] == adjlist['neighbor']) & (adjlist['weight'] == 0)]
        island_ids = set(islands_in_adjlist['focal'].tolist())
        
        # Islands should be 3 and 4
        assert set(w_islands.islands) == {3, 4}
        assert island_ids == {3, 4}
        
        # Convert back from adjacency list
        w_back = weights.W.from_adjlist(adjlist)
        
        # Check that islands are preserved
        assert set(w_back.islands) == {3, 4}
        assert w_back.n == w_islands.n
        
        # Check that non-island relationships are preserved
        for focal in [0, 1, 2]:
            assert set(w_back.neighbors[focal]) == set(w_islands.neighbors[focal])
            np.testing.assert_array_almost_equal(
                np.array(w_back.weights[focal]), 
                np.array(w_islands.weights[focal])
            )
    
    def test_islands_roundtrip(self):
        # Test round-trip conversion preserves islands
        neighbors = {0: [1], 1: [0], 2: []}  # 2 is an island
        w_orig = weights.W(neighbors, silence_warnings=True)
        
        # Round trip: W -> adjlist -> W
        adjlist = w_orig.to_adjlist(drop_islands=False)
        w_new = weights.W.from_adjlist(adjlist)
        
        # Both should have the same islands
        assert set(w_orig.islands) == set(w_new.islands)
        assert w_orig.n == w_new.n
        
        # The islands should be represented as self-loops with weight 0 in adjlist
        island_loops = adjlist[(adjlist['focal'] == adjlist['neighbor']) & (adjlist['weight'] == 0)]
        assert set(island_loops['focal'].tolist()) == set(w_orig.islands)
        
        # Non-island relationships should be preserved
        for focal in w_orig.neighbors:
            if focal not in w_orig.islands:
                assert set(w_new.neighbors[focal]) == set(w_orig.neighbors[focal])
                np.testing.assert_array_almost_equal(
                    np.array(w_new.weights[focal]), 
                    np.array(w_orig.weights[focal])
                )
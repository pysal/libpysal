A new geo-graph module for geographic topology
-----------------------------------------------

[hackmd](https://hackmd.io/kORlTnccR7GmbYfc6sw0Wg?edit)

Next Steps
----------

0. Converter to the different representations 
    - ~xarray-backed pydata.sparse~
    - adjacency table-based on "dataframe" is our target
3. Functionality
    - `.to_sparse()`
    - `.adjacency`
    - stats shim for spreg? 
5. Workbench: Test Suite (speed & correctness)
    - spreg mathematics (eigenvectors, inversion/hat matrix matmul `@`)
    - spopt algorithms
    - lag (categorical and continuous)
    - neighbors querying
    - k components (`scipy.sparse.csgraph` stuff)
    - higher order (n, <=n)
    - daskable?
4. Builders
    2. Target dataframe representation 

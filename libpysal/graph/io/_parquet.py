import json

import libpysal


def _to_parquet(graph_obj, destination, **kwargs):
    """Save adjacency as a Parquet table and add custom metadata

    Metadata contain transformation and the libpysal version used to save the file.

    This allows lossless Parquet IO.

    Parameters
    ----------
    graph_obj : Graph
        Graph to be saved
    destination : str | pyarrow.NativeFile
        path or any stream supported by pyarrow
     **kwargs
        additional keyword arguments passed to pyarrow.parquet.write_table
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except (ImportError, ModuleNotFoundError):
        raise ImportError("pyarrow is required for `to_parquet`.") from None
    table = pa.Table.from_pandas(graph_obj._adjacency.to_frame())

    meta = table.schema.metadata
    d = {"transformation": graph_obj.transformation, "version": libpysal.__version__}
    if hasattr(graph_obj, "_xarray_index_names"):
        d["_xarray_index_names"] = list(graph_obj._xarray_index_names)
    meta[b"libpysal"] = json.dumps(d).encode("utf-8")
    schema = table.schema.with_metadata(meta)

    pq.write_table(table.cast(schema), destination, **kwargs)


def _read_parquet(source, **kwargs):
    """Read libpysal-saved Graph object from Parquet

    Parameters
    ----------
    source : str | pyarrow.NativeFile
        path or any stream supported by pyarrow
    **kwargs
        additional keyword arguments passed to pyarrow.parquet.read_table

    Returns
    -------
    tuple
        tuple of adjacency table, transformation, and xarray_index_names
    """
    try:
        import pyarrow.parquet as pq
    except (ImportError, ModuleNotFoundError):
        raise ImportError("pyarrow is required for `read_parquet`.") from None

    table = pq.read_table(source, **kwargs)
    if b"libpysal" in table.schema.metadata:
        meta = json.loads(table.schema.metadata[b"libpysal"])
        transformation = meta["transformation"]
    else:
        transformation = "O"

    if b"_xarray_index_names" in table.schema.metadata:
        meta = json.loads(table.schema.metadata[b"_xarray_index_names"])
        xarray_index_names = meta["_xarray_index_names"]
    else:
        xarray_index_names = None

    return table.to_pandas()["weight"], transformation, xarray_index_names

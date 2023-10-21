import libpysal
import json


def _to_parquet(G, destination, **kwargs):
    """Save adjacency as a Parquet table and add custom metadata

    Metadata contain transformation and the libpysal version used to save the file.

    This allows lossless Parquet IO.

    Parameters
    ----------
    G : Graph
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
        raise ImportError("pyarrow is required for `to_parquet`.")
    table = pa.Table.from_pandas(G._adjacency.to_frame())

    meta = table.schema.metadata
    d = {"transformation": G.transformation, "version": libpysal.__version__}
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
        tuple of adjacency table and transformation
    """
    try:
        import pyarrow.parquet as pq
    except (ImportError, ModuleNotFoundError):
        raise ImportError("pyarrow is required for `read_parquet`.")

    table = pq.read_table(source, **kwargs)
    if b"libpysal" in table.schema.metadata.keys():
        meta = json.loads(table.schema.metadata[b"libpysal"])
        transformation = meta["transformation"]
    else:
        transformation = "O"

    return table.to_pandas()["weight"], transformation

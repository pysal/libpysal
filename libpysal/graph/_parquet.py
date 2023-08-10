import pyarrow as pa
import pyarrow.parquet as pq
import libpysal
import json

def _to_parquet(G, source):
    table = pa.Table.from_pandas(G._adjacency)

    meta = table.schema.metadata
    d = {"transformation": G.transformation, "version": libpysal.__version__}
    meta[b"libpysal"] = json.dumps(d).encode("utf-8")
    schema = table.schema.with_metadata(meta)

    pq.write_table(table.cast(schema), source)


def _read_parquet(source):
    table = pq.read_table(source)
    if b"libpysal" in table.schema.metadata.keys():
        meta = json.loads(table.schema.metadata[b"libpysal"])
        transformation = meta["transformation"]
    else:
        transformation = "O"

    return table.to_pandas(), transformation

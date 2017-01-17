from ...common import simport, requires
from ...cg import asShape
import pdio
from .. import FileIO
import pandas as pd

@requires('geopandas')
def geopandas(filename, **kw):
    import geopandas
    return geopandas.read_file(filename, **kw)

@requires('fiona')
def fiona(filename, **kw):
    import fiona
    props = {}
    with fiona.open(filename, **kw) as f:
        for i,feat in enumerate(f):
            idx = feat.get('id', i)
            try:
                idx = int(idx)
            except ValueError:
                pass
            props.update({idx:feat.get('properties', dict())})
            props[idx].update({'geometry':asShape(feat['geometry'])})
    return pd.DataFrame().from_dict(props).T

_readers = {'read_shapefile':pdio.read_files, 
            'read_fiona':fiona}
_writers = {'to_shapefile':pdio.write_files}

_pandas_readers = {k:v for k,v in pd.io.api.__dict__.items() if k.startswith('read_')}


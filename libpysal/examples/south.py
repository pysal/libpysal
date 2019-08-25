"""Southern NCOVR dataset.

The version retrieved here comes from:
     https://s3.amazonaws.com/geoda/data/south.zip
"""
from .base import _fetch, RemoteFileMetadata

SOUTH = RemoteFileMetadata(
    filename='south.zip',
    url='https://s3.amazonaws.com/geoda/data/south.zip',
    checksum=(
        '8f151d99c643b187aad37cfb5c3212353e1bc82804a4399a63de369490e56a7a'))

description = """
south
=====

Homicides and selected socio-economic characteristics for Southern U.S. counties (subset of NCOVR national data set). Data for four decennial census years: 1960, 1970, 1980, 1990.
----------------------------------------------------------------------

* south.dbf: attribute data. (k=69)
* south.shp: Polygon shapefile. (n=1412)
* south.shx: spatial index.
* south_q.gal: queen contiguity weights in GAL format.
* south_queen.gal: queen contiguity weights in GAL format.

"""


def fetch_south(meta_data=SOUTH,
                dir_name='south',
                data_home=None,
                download_if_missing=True,
                description=description):
    """Download the south data-set.

    Download it if necessary - will check if it has been fetched already.

    Parameters
    ----------

    meta_data: RemoteFileMetadata
            fields of remote archive
             - filename
             - url
             - checksum

    dir_name: string
            the name of the dataset directory under the examples parent directory

    description: string
            Contents of the README.md file for the example dataset.

    data_home : option, default: None
        Specify another download and cache folder for the datasets. By default
        all libpysal data is stored in ~/pysal_data' subfolders

    download_if_missing : optional, True by default
       If False, raise a IOError if the data is not locally available instead
       of trying to download the data from the source site.

    """

    _fetch(
        meta_data,
        dir_name,
        description,
        data_home=data_home,
        download_if_missing=download_if_missing)

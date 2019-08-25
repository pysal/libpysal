"""National NCOVR dataset.

The version retrieved here comes from:
     https://s3.amazonaws.com/geoda/data/ncovr.zip
"""
from .base import _fetch, RemoteFileMetadata

NAT = RemoteFileMetadata(
    filename='ncovr.zip',
    url='https://s3.amazonaws.com/geoda/data/ncovr.zip',
    checksum=(
        'e8cb04e6da634c6cd21808bd8cfe4dad6e295b22e8d40cc628e666887719cfe9'))

description = """
nat
===

US county homicides 1960-1990
-----------------------------

* NAT.dbf: attribute data for US county homicides. (k=69)
* NAT.shp: Polygon shapefile for US counties. (n=3085)
* NAT.shx: spatial index.
* nat.geojson: shape and attribute data in GeoJSON format.
* nat_queen.gal: queen contiguity weights in GAL format.
* nat_queen_old.gal: queen contiguity weights in GAL format using old polygon index.
* nat_trian_k20.kwt: kernel weights in KWT format.
* natregimes.dbf: attribute data for US county homicides with regimes assigned. (k=73)
* natregimes.shp: Polygon shapefile for US counties. (n=3085)
* natregimes.shx: spatial index.
"""


def fetch_nat(meta_data=NAT,
              dir_name='nat',
              data_home=None,
              download_if_missing=True,
              description=description):
    """Download the nat data-set.

    Download it if necessary - will check if it has already been fetched.

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
        all libpysal data is stored in ~/libpysal_data' subfolders

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

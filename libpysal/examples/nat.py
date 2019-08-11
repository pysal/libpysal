"""National NCOVR dataset.

The version retrieved here comes from:
     https://s3.amazonaws.com/geoda/data/ncovr.zip
"""

from os.path import dirname, exists, join
from os import makedirs, remove
from zipfile import ZipFile
from .base import RemoteFileMetadata
from .base import get_data_home
from .base import _fetch_remote

NAT = RemoteFileMetadata(
    filename='ncovr.zip',
    url='https://s3.amazonaws.com/geoda/data/ncovr.zip',
    checksum=('e8cb04e6da634c6cd21808bd8cfe4dad6e295b22e8d40cc628e666887719cfe9'))

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

def fetch_nat(data_home=None, download_if_missing=True):
    """Load the nat data-set.

    Download it if necessary

    Parameters
    ----------

    data_home : option, default: None
        Specify another download and cache folder for the datasets. By default
        all libpysal data is stored in ~/libpysal_data' subfolders

    download_if_missing : optional, True by default
       If False, raise a IOError if the data is not locally available instead
       of trying to download the data from the source site.

    """
    data_home = get_data_home(data_home=data_home)
    if not exists(data_home):
        makedirs(data_home)
    filepath = join(data_home, 'nat')
    if not exists(filepath):
        if not download_if_missing:
            raise IOError("Data not found and 'download_if_missing' is False")
        else:
            makedirs(filepath)
            print('downloading nat dataset from %s to %s' % (NAT.url, filepath))
            data_path = _fetch_remote(NAT, dirname=filepath)
            file_name = join(filepath,NAT.filename)
            print(data_path)
            print(file_name)
            with ZipFile(file_name, 'r') as archive:
                print('Extracting files....')
                archive.extractall(path=filepath)
            # write README.md from original libpysal
            readme_pth = join(filepath, 'README.md')
            print(readme_pth)
            with open(readme_pth, 'w') as readme:
                readme.write(description)
    else:
        print('already exists, not downloading')


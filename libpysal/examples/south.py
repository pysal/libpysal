"""Southern NCOVR dataset.

The version retrieved here comes from:
     https://s3.amazonaws.com/geoda/data/south.zip
"""

from os.path import dirname, exists, join
from os import makedirs, remove, rename
from zipfile import ZipFile
from .base import RemoteFileMetadata
from .base import get_data_home
from .base import _fetch_remote

SOUTH = RemoteFileMetadata(
    filename='south.zip',
    url='https://s3.amazonaws.com/geoda/data/south.zip',
    checksum=('8f151d99c643b187aad37cfb5c3212353e1bc82804a4399a63de369490e56a7a'))

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

def fetch_south(data_home=None, download_if_missing=True):
    """Load the south data-set.

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
    dataset_path = join(data_home, 'south')
    if not exists(dataset_path):
        if not download_if_missing:
            raise IOError("Data not found and 'download_if_missing' is False")
        else:
            #makedirs(filepath)
            print('downloading dataset from %s to %s' % (SOUTH.url, data_home))
            data_path = _fetch_remote(SOUTH, dirname=data_home)
            file_name = join(data_home, SOUTH.filename)
            print(dataset_path)
            print(file_name)
            with ZipFile(file_name, 'r') as archive:
                print('Extracting files....')
                archive.extractall(path=data_home)

            # write README.md from original libpysal
            readme_pth = join(dataset_path, 'README.md')
            print(readme_pth)
            with open(readme_pth, 'w') as readme:
                readme.write(description)
            # remove zip file
            remove(file_name)
            
    else:
        print('already exists, not downloading')


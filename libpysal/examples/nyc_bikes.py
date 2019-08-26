"""NYC Bike trips

The version retrieved here comes from:
https://github.com/sjsrey/nyc_bikes/archive/master.zip
"""
from .base import _fetch, RemoteFileMetadata

BIKE = RemoteFileMetadata(
    filename='master.zip',
    url='https://github.com/sjsrey/nyc_bikes/archive/master.zip',
    checksum=('159b430476d53cdd6891832c18c575c10cee25e401da96ce7f7aeb049fccd387'))

description = """
NYC Bike Data
=============


- observations: 14042 origin-desination flows
- variables: 27
- support: polygon


Files
-----
nyc_bikes_ct.csv
nyct2010.dbf
nyct2010.prj
nyct2010.shp
nyct2010.shp.xml
nyct2010.shx


Variables
--------

count	number of trips
d_cap   destination tract cap
d_tract destination tract
distance distance
end station latitutde
end station longitude
o_cap   origin tract cap
o_tract origin tract
"""


def fetch_bikes(meta_data=BIKE,
              dir_name='nyc_bikes',
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

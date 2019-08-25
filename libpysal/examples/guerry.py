"""Guerry Moral Statistics dataset.

The version retrieved here comes from:
     https://s3.amazonaws.com/geoda/data/guerry.zip
"""
from .base import _fetch, RemoteFileMetadata

GUERRY = RemoteFileMetadata(
    filename='guerry.zip',
    url='https://s3.amazonaws.com/geoda/data/guerry.zip',
    checksum=('e781a4be1450c9e4fb287440bbb5ffafd9ec63bdf42745158205d92970cca187'))

description = """
guerry
======

Andre-Michel Guerry data on "moral statistics" 1930
---------------------------------------------------
crime, suicide, literacy and other “moral statistics” in 1830s France

* Guerry_documentation.html: metadata.
* Guerry.dbf: attribute data. (k=23)
* Guerry.geojson: shape and attribute data file in geoJSON format.
* Guerry.prj: ESRI projection file.
* Guerry.shp: Polygon shapefile. (n=85)
* Guerry.shx: spatial index.

Angeville, A. (1836). Essai sur la Statistique de la Population française Paris: F. Doufour. 

Guerry, A.-M. (1833). Essai sur la statistique morale de la France Paris: Crochard. English translation: Hugh P. Whitt and Victor W. Reinking, Lewiston, N.Y. : Edwin Mellen Press, 2002. 

Parent-Duchatelet, A. (1836). De la prostitution dans la ville de Paris, 3rd ed, 1857, p. 32, 36 
"""


def fetch_guerry(meta_data=GUERRY,
              dir_name='guerry',
              data_home=None,
              download_if_missing=True,
              description=description):
    """Download the guerry data-set.

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
        download_if_missing=download_if_missing, is_dir=False)

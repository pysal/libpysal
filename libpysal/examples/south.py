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

Homicides and selected socio-economic characteristics for Southern U.S. counties.
---------------------------------------------------------------------------------

- Observations = 1,412
- Variables = 69
- Years = 1960-90s
- Support = polygon

Files
-----
south.gdb     README.md  south.dbf      south.gpkg  south.kml  south.mif  south.shp  south.sqlite
codebook.pdf  south.csv  south.geojson  south.html  south.mid  south.prj  south.shx  south.xlsx

Variables
---------
NAME 	county name
STATE_NAME 	state name
STATE_FIPS 	state fips code (character)
CNTY_FIPS 	county fips code (character)
FIPS 	combined state and county fips code (character)
STFIPS 	state fips code (numeric)
COFIPS 	county fips code (numeric)
FIPSNO 	fips code as numeric variable
SOUTH 	dummy variable for Southern counties (South = 1)
HR** 	homicide rate per 100,000 (1960, 1970, 1980, 1990)
HC** 	homicide count, three year average centered on 1960, 1970, 1980, 1990
PO** 	county population, 1960, 1970, 1980, 1990
RD** 	resource deprivation 1960, 1970, 1980, 1990 (principal component, see Codebook for details)
PS** 	population structure 1960, 1970, 1980, 1990 (principal component, see Codebook for details)
UE** 	unemployment rate 1960, 1970, 1980, 1990
DV** 	divorce rate 1960, 1970, 1980, 1990 (% males over 14 divorced)
MA** 	median age 1960, 1970, 1980, 1990
POL** 	log of population 1960, 1970, 1980, 1990
DNL** 	log of population density 1960, 1970, 1980, 1990
MFIL** 	log of median family income 1960, 1970, 1980, 1990
FP** 	% families below poverty 1960, 1970, 1980, 1990 (see Codebook for details)
BLK** 	% black 1960, 1970, 1980, 1990
GI** 	Gini index of family income inequality 1960, 1970, 1980, 1990
FH** 	% female headed households 1960, 1970, 1980, 1990
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

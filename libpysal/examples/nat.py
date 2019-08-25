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

- Observations = 3,085
- Variables = 69
- Years = 1960-90s
- Support = polygon

Files
-----
ncovr.gdb     NAT.csv  NAT.geojson  NAT.kml  NAT.mif  NAT.shp  NAT.sqlite  ncovr.html
codebook.pdf  NAT.dbf  NAT.gpkg     NAT.mid  NAT.prj  NAT.shx  NAT.xlsx    README.md

Variables
--------
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

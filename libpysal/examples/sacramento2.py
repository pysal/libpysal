"""Zip Code Business Patterns for Sacramento
The version retrieved here comes from:
     https://s3.amazonaws.com/geoda/data/SacramentoMSA2.zip
"""
from .base import _fetch, RemoteFileMetadata

SAC = RemoteFileMetadata(
    filename='SacramentoMSA2.zip',
    url='https://s3.amazonaws.com/geoda/data/SacramentoMSA2.zip',
    checksum=('6446dc044ebb8ce8e63b806bf5202adc80330592f115c8f48d894a706f6481cf'))

description = """
sacramento2
===========

2000 Census Tract Data for Sacramento MSA
-----------------------------------------

- Observations = 83
- Variables = 66
- Years = 1998, 2001
- Support = polygon

Files
-----
 SacramentoMSA2.gdb       SacramentoMSA2.kml   SacramentoMSA2.shp
 README.md                SacramentoMSA2.mid   SacramentoMSA2.shx
 SacramentoMSA2.csv       SacramentoMSA2.mif   SacramentoMSA2.sqlite
 SacramentoMSA2.dbf       SacramentoMSA2.prj   SacramentoMSA2.xlsx
 SacramentoMSA2.geojson   SacramentoMSA2.sbn  'Variable Info for Zip Code File.pdf'
 SacramentoMSA2.gpkg      SacramentoMSA2.sbx

Variables
---------
ZIP ZIP code
PO_NAME 	Name of ZIP code area
STATE 	STATE
MSA 	MSA
CBSA_CODE 	CBSA code
MAN98 	1998 total manufacturing establishments (MSA)
MAN98_12 	1998 total manufacturing establishments, 1-9 employees (MSA)
MAN98_39 	1998 total manufacturing establishments 10+ employees (MSA)
MAN01 	2001 total manufacturing establishments (MSA)
MAN01_12 	2001 total manufacturing establishments, 1-9 employees (MSA)
MAN01_39 	2001 total manufacturing establishments, 10+ employees (MSA)
MAN98US 	1998 total manufacturing establishments (US)
MAN98US12 	1998 total manufacturing establishments, 1-9 employees (US)
MAN98US39 	1998 total manufacturing establishments 10+ employees (US)
MAN01US 	2001 total manufacturing establishments (US)
MAN01US_12 	2001 total manufacturing establishments, 1-9 employees (US)
MAN01US_39 	2001 total manufacturing establishments, 10+ employees (US)
OFF98 	1998 total office establishments (MSA)
OFF98_12 	1998 total office establishments, 1-9 employees (MSA)
OFF98_39 	1998 total office establishments, 10+ employees (MSA)
OFF01 	2001 total office establishments (MSA)
OFF01_12 	2001 total office establishments, 1-9 employees (MSA)
OFF01_39 	2001 total office establishments, 10+ employees (MSA)
OFF98US 	1998 total office establishments (US)
OFF98US12 	1998 total office establishments, 1-9 employees (US)
OFF98US39 	1998 total office establishments, 10+ employees (US)
OFF01US 	2001 total office establishments (US)
OFFUS01_12 	2001 total office establishments, 1-9 employees (US)
OFFUS01_39 	2001 total office establishments, 10+ employees (US)
INFO98 	1998 total information establishments (MSA)
INFO98_12 	1998 total information establishments, 1-9 employees (MSA)
INFO98_39 	1998 total information establishments, 10+ employees (MSA)
INFO01 	2001 total information establishments (MSA)
INFO01_12 	2001 total information establishments, 1-9 employees (MSA)
INFO01_39 	2001 total information establishments, 10+ employees (MSA)
INFO98US 	1998 total information establishments (US)
INFO98US12 	1998 total information establishments, 1-9 employees (US)
INFO98US39 	1998 total information establishments, 10+ employees (US)
INFO01US 	2001 total information establishments (US)
INFO01US_1 	2001 total information establishments, 1-9 employees (US)
INFO01US_3 	2001 total information establishments, 10+ employees (US)
INDEX 	Index
NUMSEC 	Number of sectors represented in ZIP code
EST98 	Total establishments in ZIP code, 1998
EST01 	Total establishments in ZIP code, 2001
PCTNGE 	National growth effect, percent (N)
PCTIME 	Industry mix effect, percent (M)
PCTCSE 	Competitive shift effect, percent (S)
PCTGRO 	Percent growth establishments, 1998-2001 (R)
ID 	Unique ZIP code ID for ID variables in weights matrix creation window

Source: US Census Bureau, 2000 Census (Summary File 3). Extracted from http://factfinder.census.gov in April 2004.
"""


def fetch_sacramento2(meta_data=SAC,
                dir_name='sacramento2',
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

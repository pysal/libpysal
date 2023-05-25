"""Handle remote datasets."""

from bs4 import BeautifulSoup
import requests
import warnings
from .base import Example


# remote_dict holds the metadata for remote datasets from the geoda center
# to update prior to release run _remote_data()

_remote_dict = {'AirBnB': {'download_url': 'https://geodacenter.github.io/data-and-lab//data/airbnb.zip',
  'explain_url': 'https://geodacenter.github.io/data-and-lab//airbnb/',
  'n': '77',
  'k': '20',
  'description': 'Airbnb rentals, socioeconomics, and crime in Chicago'},
 'Atlanta': {'download_url': 'https://geodacenter.github.io/data-and-lab//data/atlanta_hom.zip',
  'explain_url': 'https://geodacenter.github.io/data-and-lab//atlanta_old/',
  'n': '90',
  'k': '23',
  'description': 'Atlanta, GA region homicide counts and rates'},
 'Baltimore': {'download_url': 'https://geodacenter.github.io/data-and-lab//data/baltimore.zip',
  'explain_url': 'https://geodacenter.github.io/data-and-lab//baltim/',
  'n': '211',
  'k': '17',
  'description': 'Baltimore house sales prices and hedonics'},
 'Bostonhsg': {'download_url': 'https://geodacenter.github.io/data-and-lab//data/boston.zip',
  'explain_url': 'https://geodacenter.github.io/data-and-lab//boston-housing/',
  'n': '506',
  'k': '23',
  'description': 'Boston housing and neighborhood data'},
 'Buenosaires': {'download_url': 'https://geodacenter.github.io/data-and-lab//data/buenosaires.zip',
  'explain_url': 'https://geodacenter.github.io/data-and-lab//buenos-aires_old/',
  'n': ' 209',
  'k': ' 21',
  'description': ' Electoral Data for 1999 Argentinean Elections'},
 'Cars': {'download_url': 'https://geodacenter.github.io/data-and-lab//data/Abandoned_Vehicles_Map.csv',
  'explain_url': 'https://geodacenter.github.io/data-and-lab//1-source-and-description/',
  'n': '137,867',
  'k': '21',
  'description': '2011 abandoned vehicles in Chicago (311 complaints).'},
 'Charleston1': {'download_url': 'https://geodacenter.github.io/data-and-lab//data/CharlestonMSA.zip',
  'explain_url': 'https://geodacenter.github.io/data-and-lab//charleston-1_old/',
  'n': ' 117',
  'k': ' 30',
  'description': ' 2000 Census Tract Data for Charleston, SC MSA and counties'},
 'Charleston2': {'download_url': 'https://geodacenter.github.io/data-and-lab//data/CharlestonMSA2.zip',
  'explain_url': 'https://geodacenter.github.io/data-and-lab//charleston2/',
  'n': ' 44',
  'k': ' 97',
  'description': ' 1998 and 2001 Zip Code Business Patterns (Census Bureau) for Charleston, SC MSA'},
 'Chicago Health': {'download_url': 'https://geodacenter.github.io/data-and-lab//data/comarea.zip',
  'explain_url': 'https://geodacenter.github.io/data-and-lab//comarea_vars/',
  'n': ' 77',
  'k': ' 86',
  'description': ' Chicago Health + Socio-Economics'},
 'Chicago commpop': {'download_url': 'https://geodacenter.github.io/data-and-lab//data/chicago_commpop.zip',
  'explain_url': 'https://geodacenter.github.io/data-and-lab//commpop/',
  'n': ' 77',
  'k': ' 8',
  'description': ' Chicago Community Area Population Percent Change for 2000 and 2010'},
 'Chicago parcels': {'download_url': 'https://geodacenter.github.io/data-and-lab//https://uchicago.box.com/s/j2d2ch5uvckse24y8l7vh9198wnq216i',
  'explain_url': 'https://geodacenter.github.io/data-and-lab//parcels/',
  'n': ' 592,521',
  'k': ' 5',
  'description': ' Tax parcel polygons of Cook county'},
 'Chile Labor': {'download_url': 'https://geodacenter.github.io/data-and-lab//data/flma.zip',
  'explain_url': 'https://geodacenter.github.io/data-and-lab//FLMA/',
  'n': '141',
  'k': '62',
  'description': 'Labor Markets in Chile (1982-2002)'},
 'Chile Migration': {'download_url': 'https://geodacenter.github.io/data-and-lab//https://uchicago.box.com/s/yqc97nq23hoeeqo5lkc2grlg98skokgk',
  'explain_url': 'https://geodacenter.github.io/data-and-lab//CHIM/',
  'n': ' 304',
  'k': ' 10',
  'description': ' Internal Migration in Chile (1977-2002)'},
 'Cincinnati': {'download_url': 'https://geodacenter.github.io/data-and-lab//data/walnuthills_updated.zip',
  'explain_url': 'https://geodacenter.github.io/data-and-lab//walnut_hills/',
  'n': ' 457',
  'k': ' 89',
  'description': ' 2008 Cincinnati Crime + Socio-Demographics'},
 'Cleveland': {'download_url': 'https://geodacenter.github.io/data-and-lab//data/cleveland.zip',
  'explain_url': 'https://geodacenter.github.io/data-and-lab//clev_sls_154_core/',
  'n': ' 205',
  'k': ' 9',
  'description': ' 2015 sales prices of homes in Cleveland, OH.'},
 'Columbus': {'download_url': 'https://geodacenter.github.io/data-and-lab//data/columbus.zip',
  'explain_url': 'https://geodacenter.github.io/data-and-lab//columbus/',
  'n': ' 49',
  'k': ' 20',
  'description': ' Columbus neighborhood crime'},
 'Elections': {'download_url': 'https://geodacenter.github.io/data-and-lab//data/election.zip',
  'explain_url': 'https://geodacenter.github.io/data-and-lab//county_election_2012_2016-variables/',
  'n': ' 3,108',
  'k': ' 74',
  'description': ' 2012 and 2016 Presidential Elections'},
 'Grid100': {'download_url': 'https://geodacenter.github.io/data-and-lab//data/grid100.zip',
  'explain_url': 'https://geodacenter.github.io/data-and-lab//grid100/',
  'n': ' 100',
  'k': ' 34',
  'description': ' Grid with simulated variables'},
 'Groceries': {'download_url': 'https://geodacenter.github.io/data-and-lab//data/grocery.zip',
  'explain_url': 'https://geodacenter.github.io/data-and-lab//chicago_sup_vars/',
  'n': ' 148',
  'k': ' 7',
  'description': ' 2015 Chicago supermarkets'},
 'Guerry': {'download_url': 'https://geodacenter.github.io/data-and-lab//data/guerry.zip',
  'explain_url': 'https://geodacenter.github.io/data-and-lab//Guerry/',
  'n': ' 85',
  'k': ' 23',
  'description': ' Moral statistics of France (Guerry, 1833)'},
 'Health+': {'download_url': 'https://geodacenter.github.io/data-and-lab//data/income_diversity.zip',
  'explain_url': 'https://geodacenter.github.io/data-and-lab//co_income_diversity_variables/',
  'n': ' 3,984',
  'k': ' 64',
  'description': ' 2000 Health, Income + Diversity'},
 'Health Indicators': {'download_url': 'https://geodacenter.github.io/data-and-lab//data/healthIndicators.zip',
  'explain_url': 'https://geodacenter.github.io/data-and-lab//healthindicators-variables/',
  'n': ' 77',
  'k': ' 31',
  'description': ' Chicago Health Indicators (2005-11)'},
 'Hickory1': {'download_url': 'https://geodacenter.github.io/data-and-lab//data/HickoryMSA.zip',
  'explain_url': 'https://geodacenter.github.io/data-and-lab//hickory1/',
  'n': ' 68',
  'k': ' 30',
  'description': ' 2000 Census Tract Data for Hickory, NC MSA and counties'},
 'Hickory2': {'download_url': 'https://geodacenter.github.io/data-and-lab//data/HickoryMSA2.zip',
  'explain_url': 'https://geodacenter.github.io/data-and-lab//hickory2/',
  'n': ' 29',
  'k': ' 55',
  'description': ' 1998 and 2001 Zip Code Business Patterns (Census Bureau) for Hickory, NC MSA'},
 'Home Sales': {'download_url': 'https://geodacenter.github.io/data-and-lab//data/kingcounty.zip',
  'explain_url': 'https://geodacenter.github.io/data-and-lab//KingCounty-HouseSales2015/',
  'n': ' 21,613',
  'k': ' 21',
  'description': ' 2014-15 Home Sales in King County, WA'},
 'Houston': {'download_url': 'https://geodacenter.github.io/data-and-lab//data/houston_hom.zip',
  'explain_url': 'https://geodacenter.github.io/data-and-lab//houston/',
  'n': ' 52',
  'k': ' 23',
  'description': ' Houston, TX region homicide counts and rates'},
 'Juvenile': {'download_url': 'https://geodacenter.github.io/data-and-lab//data/juvenile.zip',
  'explain_url': 'https://geodacenter.github.io/data-and-lab//juvenile/',
  'n': ' 168',
  'k': ' 3',
  'description': ' Cardiff juvenile delinquent residences'},
 'Lansing1': {'download_url': 'https://geodacenter.github.io/data-and-lab//data/LansingMSA.zip',
  'explain_url': 'https://geodacenter.github.io/data-and-lab//lansing1/',
  'n': ' 117',
  'k': ' 30',
  'description': ' 2000 Census Tract Data for Lansing, MI MSA and counties'},
 'Lansing2': {'download_url': 'https://geodacenter.github.io/data-and-lab//data/LansingMSA2.zip',
  'explain_url': 'https://geodacenter.github.io/data-and-lab//lansing2/',
  'n': ' 46',
  'k': ' 55',
  'description': ' 1998 and 2001 Zip Code Business Patterns (Census Bureau) for Lansing, MI MSA'},
 'Laozone': {'download_url': 'https://geodacenter.github.io/data-and-lab//data/laozone.zip',
  'explain_url': 'https://geodacenter.github.io/data-and-lab//ozone/',
  'n': ' 32',
  'k': ' 8',
  'description': ' Ozone measures at monitoring stations in Los Angeles basin'},
 'LasRosas': {'download_url': 'https://geodacenter.github.io/data-and-lab//data/lasrosas.zip',
  'explain_url': 'https://geodacenter.github.io/data-and-lab//lasrosas/',
  'n': ' 1,738',
  'k': ' 34',
  'description': ' Corn yield, fertilizer and field data for precision agriculture, Argentina, 1999'},
 'Liquor Stores': {'download_url': 'https://geodacenter.github.io/data-and-lab//data/liquor.zip',
  'explain_url': 'https://geodacenter.github.io/data-and-lab//liq_chicago/',
  'n': ' 571',
  'k': ' 2',
  'description': ' 2015 Chicago Liquor Stores'},
 'Malaria': {'download_url': 'https://geodacenter.github.io/data-and-lab//data/malariacolomb.zip',
  'explain_url': 'https://geodacenter.github.io/data-and-lab//colomb_malaria/',
  'n': ' 1,068',
  'k': ' 50',
  'description': ' Malaria incidence and population (1973, 95, 93 censuses and projections until 2005) \xa0 \xa0 \xa0'},
 'Milwaukee1': {'download_url': 'https://geodacenter.github.io/data-and-lab//data/MilwaukeeMSA.zip',
  'explain_url': 'https://geodacenter.github.io/data-and-lab//milwaukee1/',
  'n': ' 417',
  'k': ' 31',
  'description': ' 2000 Census Tract Data for Milwaukee, WI MSA'},
 'Milwaukee2': {'download_url': 'https://geodacenter.github.io/data-and-lab//data/MilwaukeeMSA2.zip',
  'explain_url': 'https://geodacenter.github.io/data-and-lab//milwaukee2/',
  'n': ' 83',
  'k': ' 55',
  'description': ' 1998 and 2001 Zip Code Business Patterns (Census Bureau) for Milwaukee, WI MSA'},
 'NCOVR': {'download_url': 'https://geodacenter.github.io/data-and-lab//data/ncovr.zip',
  'explain_url': 'https://geodacenter.github.io/data-and-lab//ncovr/',
  'n': '3,085',
  'k': ' 69',
  'description': ' US county homicides 1960-1990'},
 'Natregimes': {'download_url': 'https://geodacenter.github.io/data-and-lab//data/natregimes.zip',
  'explain_url': 'https://geodacenter.github.io/data-and-lab//natregimes/',
  'n': ' 3,085',
  'k': ' 73',
  'description': ' NCOVR with regimes (book/PySAL)'},
 'NDVI': {'download_url': 'https://geodacenter.github.io/data-and-lab//data/ndvi.zip',
  'explain_url': 'https://geodacenter.github.io/data-and-lab//ndvi/',
  'n': ' 49',
  'k': ' 5',
  'description': ' Normalized Difference Vegetation Index grid'},
 'Nepal': {'download_url': 'https://geodacenter.github.io/data-and-lab//data/nepal.zip',
  'explain_url': 'https://geodacenter.github.io/data-and-lab//nepal/',
  'n': ' 75',
  'k': ' 61',
  'description': ' Health, poverty and education indicators for Nepal districts'},
 'NYC': {'download_url': 'https://geodacenter.github.io/data-and-lab///data/nyc.zip',
  'explain_url': 'https://geodacenter.github.io/data-and-lab//nyc/',
  'n': ' 55',
  'k': ' 34',
  'description': ' Demographic and housing data for New York City subboroughs, 2002-09'},
 'NYC Earnings': {'download_url': 'https://geodacenter.github.io/data-and-lab//data/lehd.zip',
  'explain_url': 'https://geodacenter.github.io/data-and-lab//LEHD_Data/',
  'n': ' 108,487',
  'k': ' 70',
  'description': ' Block-level Earnings in NYC (2002-14)'},
 'NYC Education': {'download_url': 'https://geodacenter.github.io/data-and-lab//data/nyc_2000Census.zip',
  'explain_url': 'https://geodacenter.github.io/data-and-lab//NYC-Census-2000/',
  'n': ' 2,216',
  'k': ' 56',
  'description': ' NYC Education (2000)'},
 'NYC Neighborhoods': {'download_url': 'https://geodacenter.github.io/data-and-lab//data/nycnhood_acs.zip',
  'explain_url': 'https://geodacenter.github.io/data-and-lab//NYC-Nhood-ACS-2008-12/',
  'n': ' 195',
  'k': ' 98',
  'description': ' Demographics for New York City neighborhoods'},
 'NYC Socio-Demographics': {'download_url': 'https://geodacenter.github.io/data-and-lab//data/nyctract_acs.zip',
  'explain_url': 'https://geodacenter.github.io/data-and-lab//NYC_Tract_ACS2008_12/',
  'n': ' 2,166',
  'k': ' 113',
  'description': ' NYC Education + Socio-Demographics'},
 'Ohiolung': {'download_url': 'https://geodacenter.github.io/data-and-lab//data/ohiolung.zip',
  'explain_url': 'https://geodacenter.github.io/data-and-lab//ohiolung/',
  'n': ' 88',
  'k': ' 42',
  'description': ' Ohio lung cancer data, 1968, 1978, 1988'},
 'Orlando1': {'download_url': 'https://geodacenter.github.io/data-and-lab//data/OrlandoMSA.zip',
  'explain_url': 'https://geodacenter.github.io/data-and-lab//orlando1/',
  'n': ' 328',
  'k': ' 30',
  'description': ' 2000 Census Tract Data for Orlando, FL MSA and counties'},
 'Orlando2': {'download_url': 'https://geodacenter.github.io/data-and-lab//data/OrlandoMSA2.zip',
  'explain_url': 'https://geodacenter.github.io/data-and-lab//orlando2/',
  'n': ' 94',
  'k': ' 59',
  'description': ' 1998 and 2001 Zip Code Business Patterns (Census Bureau) for Orlando, FL MSA'},
 'Oz9799': {'download_url': 'https://geodacenter.github.io/data-and-lab//data/oz9799.zip',
  'explain_url': 'https://geodacenter.github.io/data-and-lab//oz96/',
  'n': ' 30',
  'k': ' 78',
  'description': ' Monthly ozone data, 1997-99'},
 'Phoenix ACS': {'download_url': 'https://geodacenter.github.io/data-and-lab//data/phx2.zip',
  'explain_url': 'https://geodacenter.github.io/data-and-lab//phx/',
  'n': ' 685',
  'k': ' 17',
  'description': ' Phoenix American Community Survey Data (2010, 5-year averages)'},
 'Pittsburgh': {'download_url': 'https://geodacenter.github.io/data-and-lab//data/pittsburgh.zip',
  'explain_url': 'https://geodacenter.github.io/data-and-lab//pitt93/',
  'n': ' 143',
  'k': ' 8',
  'description': ' Pittsburgh homicide locations'},
 'Police': {'download_url': 'https://geodacenter.github.io/data-and-lab//data/police.zip',
  'explain_url': 'https://geodacenter.github.io/data-and-lab//police/',
  'n': ' 82',
  'k': ' 21',
  'description': ' Police expenditures Mississippi counties'},
 'Sacramento1': {'download_url': 'https://geodacenter.github.io/data-and-lab//data/sacramento.zip',
  'explain_url': 'https://geodacenter.github.io/data-and-lab//sacramento1/',
  'n': ' 403',
  'k': ' 30',
  'description': ' 2000 Census Tract Data for Sacramento MSA'},
 'Sacramento2': {'download_url': 'https://geodacenter.github.io/data-and-lab//data/SacramentoMSA2.zip',
  'explain_url': 'https://geodacenter.github.io/data-and-lab//sacramento2/',
  'n': ' 125',
  'k': ' 53',
  'description': ' 1998 and 2001 Zip Code Business Patterns (Census Bureau) for Sacramento MSA'},
 'SanFran Crime': {'download_url': 'https://geodacenter.github.io/data-and-lab//data/SFCrime_July_Dec2012.zip',
  'explain_url': 'https://geodacenter.github.io/data-and-lab//SFcrimes_vars/',
  'n': ' 3,384',
  'k': ' 13',
  'description': ' July-Dec 2012 crime incidents in San Francisco (points + area) - for CAST'},
 'Savannah1': {'download_url': 'https://geodacenter.github.io/data-and-lab//data/SavannahMSA.zip',
  'explain_url': 'https://geodacenter.github.io/data-and-lab//savannah1/',
  'n': ' 77',
  'k': ' 30',
  'description': ' 2000 Census Tract Data for Savannah, GA MSA and counties'},
 'Savannah2': {'download_url': 'https://geodacenter.github.io/data-and-lab//data/SavannahMSA2.zip',
  'explain_url': 'https://geodacenter.github.io/data-and-lab//savannah2/',
  'n': ' 24',
  'k': ' 55',
  'description': ' 1998 and 2001 Zip Code Business Patterns (Census Bureau) for Savannah, GA MSA'},
 'Scotlip': {'download_url': 'https://geodacenter.github.io/data-and-lab//data/scotlip.zip',
  'explain_url': 'https://geodacenter.github.io/data-and-lab//scotlip/',
  'n': ' 56',
  'k': ' 11',
  'description': ' Male lip cancer in Scotland, 1975-80'},
 'Seattle1': {'download_url': 'https://geodacenter.github.io/data-and-lab//data/SeattleMSA.zip',
  'explain_url': 'https://geodacenter.github.io/data-and-lab//seattle1/',
  'n': ' 664',
  'k': ' 30',
  'description': ' 2000 Census Tract Data for Seattle, WA MSA and counties'},
 'Seattle2': {'download_url': 'https://geodacenter.github.io/data-and-lab//data/SeattleMSA2.zip',
  'explain_url': 'https://geodacenter.github.io/data-and-lab//seattle2/',
  'n': ' 145',
  'k': ' 59',
  'description': ' 1998 and 2001 Zip Code Business Patterns (Census Bureau) for Seattle, WA MSA'},
 'SIDS': {'download_url': 'https://geodacenter.github.io/data-and-lab//data/sids.zip',
  'explain_url': 'https://geodacenter.github.io/data-and-lab//sids/',
  'n': ' 100',
  'k': ' 13',
  'description': ' North Carolina county SIDS death counts'},
 'SIDS2': {'download_url': 'https://geodacenter.github.io/data-and-lab//data/sids2.zip',
  'explain_url': 'https://geodacenter.github.io/data-and-lab//sids2/',
  'n': ' 100',
  'k': ' 17',
  'description': ' North Carolina county SIDS death counts and rates'},
 'Snow': {'download_url': 'https://geodacenter.github.io/data-and-lab//data/snow.zip',
  'explain_url': 'https://geodacenter.github.io/data-and-lab//snow/',
  'n': ' NA',
  'k': ' NA',
  'description': ' John Snow & the 19th Century Cholera Epidemic'},
 'South': {'download_url': 'https://geodacenter.github.io/data-and-lab//data/south.zip',
  'explain_url': 'https://geodacenter.github.io/data-and-lab//south/',
  'n': ' 1,412',
  'k': ' 69',
  'description': ' US Southern county homicides 1960-1990'},
 'Spirals': {'download_url': 'https://geodacenter.github.io/data-and-lab//data/spirals.csv',
  'explain_url': 'https://geodacenter.github.io/data-and-lab//spirals/',
  'n': ' 301',
  'k': ' 2',
  'description': ' Synthetic spiral points'},
 'StLouis': {'download_url': 'https://geodacenter.github.io/data-and-lab//data/stlouis.zip',
  'explain_url': 'https://geodacenter.github.io/data-and-lab//stlouis/',
  'n': ' 78',
  'k': ' 23',
  'description': ' St Louis region county homicide counts and rates'},
 'Tampa1': {'download_url': 'https://geodacenter.github.io/data-and-lab//data/TampaMSA.zip',
  'explain_url': 'https://geodacenter.github.io/data-and-lab//tampa1/',
  'n': ' 547',
  'k': ' 30',
  'description': ' 2000 Census Tract Data for Tampa, FL MSA and counties'},
 'US SDOH': {'download_url': 'https://geodacenter.github.io/data-and-lab//data/us-sdoh-2014.zip',
  'explain_url': 'https://geodacenter.github.io/data-and-lab//us-sdoh/',
  'n': ' 71,901',
  'k': ' 25',
  'description': ' 2014 US Social Determinants of Health Data'}}

def _remote_data():
    """Helper function to get remote metadata for each release.

    Returns
    -------
    datasets : dict
        Remote data sets keyed by the dataset name.
        Values are dictionaries with the following keys
        'download_url', 'explain_url', 'n', 'k', 'description'.
    """

    url = "https://geodacenter.github.io/data-and-lab//"
    try:
        page = requests.get(url)
    except:
        warnings.warn("Remote data sets not available. Check connection.")
        return {}
    soup = BeautifulSoup(page.text, "html.parser")
    samples = soup.find(class_="samples")
    rows = samples.find_all("tr")
    datasets = {}
    for row in rows[1:]:
        data = row.find_all("td")
        name = data[0].text.strip()
        description = data[1].text
        n = data[2].text
        k = data[3].text
        targets = row.find_all("a")
        download_url = url + targets[1].attrs["href"]
        explain_url = url + targets[0].attrs["href"]
        datasets[name] = {'download_url': download_url, 'explain_url': explain_url, 'n': n, 'k': k, 'description': description}
    return datasets


def _build_remotes():
    """Build remote meta data.

    Returns
    -------
    datasets : dict
        Example datasets keyed by the dataset name.

    """
    datasets = {}
    for name in _remote_dict:
        description = _remote_dict[name]["description"]
        n = _remote_dict[name]["n"]
        k = _remote_dict[name]["k"]
        download_url = _remote_dict[name]["download_url"]
        explain_url = _remote_dict[name]["explain_url"]
        datasets[name] = Example(name, description, n, k, download_url, explain_url)
    
    # Other Remotes
    # rio
    name = "Rio Grande do Sul"
    description = "Cities of the Brazilian State of Rio Grande do Sul"
    n = 497
    k = 3
    download_url = "https://github.com/sjsrey/rio_grande_do_sul/archive/master.zip"
    explain_url = (
        "https://raw.githubusercontent.com/sjsrey/rio_grande_do_sul/master/README.md"
    )
    datasets[name] = Example(name, description, n, k, download_url, explain_url)

    # nyc bikes
    name = "nyc_bikes"
    description = "New York City Bike Trips"
    n = 14042
    k = 27
    download_url = "https://github.com/sjsrey/nyc_bikes/archive/master.zip"
    explain_url = "https://raw.githubusercontent.com/sjsrey/nyc_bikes/master/README.md"
    datasets[name] = Example(name, description, n, k, download_url, explain_url)

    # taz
    name = "taz"
    description = "Traffic Analysis Zones in So. California"
    n = 4109
    k = 14
    download_url = "https://github.com/sjsrey/taz/archive/master.zip"
    explain_url = "https://raw.githubusercontent.com/sjsrey/taz/master/README.md"
    datasets[name] = Example(name, description, n, k, download_url, explain_url)

    # clearwater
    name = "clearwater"
    description = "mgwr testing dataset"
    n = 239
    k = 14
    download_url = "https://github.com/sjsrey/clearwater/archive/master.zip"
    explain_url = "https://raw.githubusercontent.com/sjsrey/clearwater/master/README.md"
    datasets[name] = Example(name, description, n, k, download_url, explain_url)

    # newHaven
    name = "newHaven"
    description = "Network testing dataset"
    n = 3293
    k = 5
    download_url = "https://github.com/sjsrey/newHaven/archive/master.zip"
    explain_url = "https://raw.githubusercontent.com/sjsrey/newHaven/master/README.md"
    datasets[name] = Example(name, description, n, k, download_url, explain_url)

    # remove Cars dataset as it is broken
    datasets.pop("Cars")

    return datasets


class Remotes:
    """Remote datasets."""

    def __init__(self):
        """Initialize Remotes."""
        self._datasets = None

    @property
    def datasets(self):
        """Create dictionary of remotes."""
        if self._datasets is None:
            self._datasets = _build_remotes()
        return self._datasets


datasets = Remotes()

"""Handle remote datasets.
"""

from bs4 import BeautifulSoup
import requests
import warnings
from .base import PYSALDATA, Example, get_list_of_files, get_data_home


def poll_remotes():
    """Fetch remote data and generate example datasets.

    Returns
    -------
    datasets : dict
        Example datasets keyed by the dataset name.

    """

    # Geoda Center Data Sets

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


datasets = poll_remotes()


def download(datasets=datasets):
    """
    Download all known remotes
    """

    names = list(datasets.keys())
    names.sort()
    for name in names:
        print(name)
        example = datasets[name]
        try:
            example.download()
        except:
            print("Example not downloaded: {}".format(name))

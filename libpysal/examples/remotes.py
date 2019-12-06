"""
Handle remote datasets
"""
from bs4 import BeautifulSoup
import requests
from .base import (PYSALDATA, Example, get_list_of_files,
                   get_data_home)



# Geoda Center Data Sets
url = "https://geodacenter.github.io/data-and-lab//"
page = requests.get(url)
soup = BeautifulSoup(page.text, 'html.parser')
samples = soup.find(class_='samples')
rows = samples.find_all('tr')
datasets = {}
for row in rows[1:]:
    data = row.find_all('td')
    name = data[0].text.strip()
    description = data[1].text
    n = data[2].text
    k = data[3].text
    targets = row.find_all('a')
    download_url = targets[1].attrs['href']
    explain_url = targets[0].attrs['href']
    datasets[name] = Example(name, description, n, k, download_url, explain_url)


# Other Remotes

name = 'Rio Grande do Sul'
description = 'Cities of the Brazilian State of Rio Grande do Sul' 
n = 497
k = 3
download_url = 'https://github.com/sjsrey/rio_grande_do_sul/archive/master.zip'
explain_url = 'https://raw.githubusercontent.com/sjsrey/rio_grande_do_sul/master/README.md'
datasets[name] = Example(name, description, n, k, download_url, explain_url)

#datasets['Rio Grande do Sul'] = Example('Rio Grande do Sul',
#                          'Cities of the Brazilian State of Rio Grande do Sul',
#                          497, 3,
#                          'https://github.com/sjsrey/rio_grande_do_sul/archive/master.zip',
#                          'https://raw.githubusercontent.com/sjsrey/rio_grande_do_sul/master/README.md')


out_dict = {}
for dataset in datasets:
    out_dict[dataset] = datasets[dataset].json_dict()


def download(datasets=datasets):
    '''
    Download all known remotes
    '''
    names = list(datasets.keys())
    names.sort()
    for name in names:
        print(name)
        example = datasets[name]
        try:
            example.download()
        except:
            print('Example not downloaded: {}'.format(name))

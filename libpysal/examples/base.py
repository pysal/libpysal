"""
Base class for managing example datasets
"""

# Authors: Serge Rey
# License: BSD 3 Clause

import io
import os
import webbrowser
from os import environ, makedirs
from os.path import exists, expanduser, join
import zipfile
import requests
import pandas
from bs4 import BeautifulSoup
from ..io import open as ps_open

PYSALDATA = 'pysal_data'

def get_data_home():
    """Return the path of the libpysal data directory.


    This folder (~/pysal_data)is used by some large dataset loaders to avoid
    downloading the data multiple times.


    Alternatively, it can be set by the 'PYSALDATA' environment variable or
    programmatically by giving an explicit folder path. The '~' symbol is
    expanded to the user home folder

    If the folder does not already exisit, it is automatically created.
    """

    data_home = environ.get('PYSALDATA', join("~", PYSALDATA))
    data_home = expanduser(data_home)
    if not exists(data_home):
        makedirs(data_home)
    return data_home


def get_list_of_files(dir_name):
    """
    create a list of file and sub directories in dir_name
    """
    #names in the given directory
    file_list = os.listdir(dir_name)
    all_files = list()
    # Iterate over all the entries
    for entry in file_list:
        # Create full path
        full_path = os.path.join(dir_name, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(full_path):
            all_files = all_files + get_list_of_files(full_path)
        else:
            all_files.append(full_path)

    return all_files

def type_of_script():
    """Helper function to determine run context"""
    try:
        ipy_str = str(type(get_ipython()))
        if 'zmqshell' in ipy_str:
            return 'jupyter'
        if 'terminal' in ipy_str:
            return 'ipython'
    except:
        return 'terminal'



class Example:
    """
    Example Dataset
    """
    def __init__(self, name, description, n, k, download_url,
                explain_url):
        self.name = name
        self.description = description
        self.n = n
        self.k = k
        self.download_url = download_url
        self.explain_url = explain_url
        self.root = name.replace(" ", "_")
        self.installed = self.downloaded()

    def get_local_path(self, path=get_data_home()):
        """Get local path for example"""
        return join(path, self.root)

    def get_path(self, file_name, verbose=True):
        """
        get path for local file
        """
        file_list = self.get_file_list()
        for file_path in file_list:
            base_name = os.path.basename(file_path)
            if file_name == base_name:
                return file_path
        if verbose:
            print("{} is not a file in this example".format(file_name))
        return None

    def downloaded(self):
        """
        Check if example has already been installed
        """
        path = self.get_local_path()
        if os.path.isdir(path):
            self.installed = True
            return True
        return False


    def explain(self):
        """
        Provide a description of the example.
        """
        file_name = self.explain_url.split("/")[-1]
        if file_name == 'README.md':
            explain_page = requests.get(self.explain_url)
            crawled = BeautifulSoup(explain_page.text, 'html.parser')
            print(crawled.text)
            return None
        if type_of_script() == 'terminal':
            webbrowser.open(self.explain_url)
            return None
        from IPython.display import IFrame
        return IFrame(self.explain_url, width=700, height=350)

    def download(self, path=get_data_home()):
        """
        Download the files for the example.
        """
        if self.downloaded():
            print('Already downloaded')
        else:
            request = requests.get(self.download_url)
            archive = zipfile.ZipFile(io.BytesIO(request.content))
            target = join(path, self.root)
            print('Downloading {} to {}'.format(self.name, target))
            archive.extractall(path=target)
            self.zipfile = archive
            self.installed = True

    def get_file_list(self):
        """
        Get list of local files for the example.
        """
        path = self.get_local_path()
        if os.path.isdir(path):
            return get_list_of_files(path)
        return None


    def json_dict(self):
        """
        container for example meta data
        """
        meta = {}
        meta['name'] = self.name
        meta['description'] = self.description
        meta['download_url'] = self.download_url
        meta['explain_url'] = self.explain_url
        meta['root'] = self.root
        return meta

    def load(self, file_name):
        """
        dispatch to libpysal.io to open file
        """
        pth = self.get_path(file_name)
        if pth:
            return ps_open(pth)




class Examples:
    """
    Manager for pysal example datasets.

    """

    def __init__(self):
        self.datasets = {}


    def add_examples(self, examples):
        """
        add examples to the set of datasets available
        """
        self.datasets.update(examples)


    def explain(self, example_name):
        if example_name in self.datasets:
            return self.datasets[example_name].explain()
        else:
            print('not available')

    def available(self):
        """
        report available datasets
        """
        datasets = self.datasets
        names = list(datasets.keys())
        names.sort()
        rows = []
        for name in names:
            description = datasets[name].description
            installed = datasets[name].installed
            rows.append([name, description, installed])
        datasets = pandas.DataFrame(data = rows, columns=['Name', 'Description', 'Installed'])
        datasets.style.set_properties(subset=['text'], **{'width': '300px'})
        print(datasets.to_string())


    def load(self, example_name):
        """
        load example dataset, download if not locally available
        """
        if example_name in self.datasets:
            example = self.datasets[example_name]
            if example.installed:
                return example
            else:
                "Downloading: {}".format(example_name)
                example.download()
                return example
        else:
            print('Example not available: {}'.format(example_name))
            return None

    def download_remotes(self):
        """
        Dowload all remotes

        """
        names = list(self.remotes.keys())
        names.sort()

        for name in names:
            print(name)
            example = self.remotes[name]
            try:
                example.download()
            except:
                print('Example not downloaded: {}'.format(name))


    def get_installed_names(self):
        """Return names of all currently installed datasets"""
        ds = self.datasets
        return [name for name in ds if ds[name].installed]

example_manager = Examples()


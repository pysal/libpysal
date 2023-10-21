"""
Base class for managing example datasets.
"""

# Authors: Serge Rey
# License: BSD 3 Clause

import io
import os
import tempfile
import webbrowser
from platformdirs import user_data_dir
import zipfile
import requests
import pandas
from bs4 import BeautifulSoup
from ..io import open as ps_open


from typing import Union



def get_data_home():
    """Return the path of the ``libpysal`` data directory. This folder is platform specific.
    If the folder does not already exist, it is automatically created.

    Returns
    -------
    data_home : str
        The system path where the data is/will be stored.

    """

    appname = "pysal"
    appauthor = "pysal"
    data_home = user_data_dir(appname, appauthor)

    try:
        if not os.path.exists(data_home):
            os.makedirs(data_home, exist_ok=True)
    except OSError:
        # Try to fall back to a tmp directory
        data_home = os.path.join(tempfile.gettempdir(), "pysal")
        os.makedirs(data_home, exist_ok=True)

    return data_home


def get_list_of_files(dir_name):
    """Create a list of files and sub-directories in ``dir_name``.

    Parameters
    ----------
    dir_name : str
        The path to the directory or examples.

    Returns
    -------
    all_files : list
        All file and directory paths.

    Raises
    ------
    FileNotFoundError
        If the file or directory is not found.

    """

    # names in the given directory
    all_files = list()
    try:
        file_list = os.listdir(dir_name)
        # Iterate over all the entries
        for entry in file_list:
            # Create full path
            full_path = os.path.join(dir_name, entry)
            # If entry is a directory then get the list of files in this directory
            if os.path.isdir(full_path):
                all_files = all_files + get_list_of_files(full_path)
            else:
                all_files.append(full_path)
    except FileNotFoundError:
        pass

    return all_files


def type_of_script() -> str:
    """Helper function to determine run context."""

    try:
        ipy_str = str(type(get_ipython()))
        if "zmqshell" in ipy_str:
            return "jupyter"
        if "terminal" in ipy_str:
            return "ipython"
    except:
        return "terminal"


class Example:
    """An example dataset.

    Parameters
    ----------
    name : str
        The example dataset name.
    description : str
        The example dataset description.
    n : int
        The number of records in the dataset.
    k : int
        The number of fields in the dataset.
    download_url : str
        The URL to download the dataset.
    explain_url : str
        The URL to the dataset's READEME file.

    Attributes
    ----------
    root : str
        The ``name`` parameter with filled spaces (_).
    installed : bool
        ``True`` if the example is installed, otherwise ``False``.
    zipfile : zipfile.ZipFile
        The archived dataset.

    """

    def __init__(self, name, description, n, k, download_url,
                 explain_url):
        """Initialze Example."""
        self.name = name
        self.description = description
        self.n = n
        self.k = k
        self.download_url = download_url
        self.explain_url = explain_url
        self.root = name.replace(" ", "_")
        self.installed = self.downloaded()

    def get_local_path(self, path=None) -> str:
        """Get the local path for example."""
        path = path or get_data_home()
        return os.path.join(path, self.root)

    def get_path(self, file_name, verbose=True) -> Union[str, None]:
        """Get the path for local file."""
        file_list = self.get_file_list()
        for file_path in file_list:
            base_name = os.path.basename(file_path)
            if file_name == base_name:
                return file_path
        if verbose:
            print(f'{file_name} is not a file in this example.')
        return None

    def downloaded(self) -> bool:
        """Check if the example has already been installed."""
        path = self.get_local_path()
        if os.path.isdir(path):
            self.installed = True
            return True
        return False

    def explain(self) -> None:
        """Provide a description of the example."""

        file_name = self.explain_url.split("/")[-1]
        if file_name == "README.md":
            explain_page = requests.get(self.explain_url)
            crawled = BeautifulSoup(explain_page.text, "html.parser")
            print(crawled.text)
            return None
        if type_of_script() == "terminal":
            webbrowser.open(self.explain_url)
            return None
        from IPython.display import IFrame

        return IFrame(self.explain_url, width=700, height=350)

    def download(self, path=None):
        """Download the files for the example."""
        path = path or get_data_home()

        if not self.downloaded():
            try:
                request = requests.get(self.download_url)
                archive = zipfile.ZipFile(io.BytesIO(request.content))
                target = os.path.join(path, self.root)
                print("Downloading {} to {}".format(self.name, target))
                archive.extractall(path=target)
                self.zipfile = archive
                self.installed = True
            except requests.exceptions.RequestException as e:  
                raise SystemExit(e)
            

    def get_file_list(self) -> Union[list, None]:
        """Get the list of local files for the example."""
        path = self.get_local_path()
        if os.path.isdir(path):
            return get_list_of_files(path)
        return None

    def json_dict(self) -> dict:
        """Container for example meta data."""
        meta = {}
        meta["name"] = self.name
        meta["description"] = self.description
        meta["download_url"] = self.download_url
        meta["explain_url"] = self.explain_url
        meta["root"] = self.root
        return meta

    def load(self, file_name) -> io.FileIO:
        """Dispatch to libpysal.io to open file."""
        pth = self.get_path(file_name)
        if pth:
            return ps_open(pth)


class Examples:
    """Manager for pysal example datasets."""

    def __init__(self, datasets={}):
        self.datasets = datasets

    def add_examples(self, examples):
        """Add examples to the set of datasets available."""
        self.datasets.update(examples)

    def explain(self, example_name) -> str:
        if example_name in self.datasets:
            return self.datasets[example_name].explain()
        else:
            print("not available")

    def available(self):
        """Return df of available datasets."""
        datasets = self.datasets
        names = list(datasets.keys())
        names.sort()
        rows = []
        for name in names:
            description = datasets[name].description
            installed = datasets[name].installed
            rows.append([name, description, installed])
        datasets = pandas.DataFrame(
            data=rows, columns=["Name", "Description", "Installed"]
        )
        datasets.style.set_properties(subset=["text"], **{"width": "300px"})
        return datasets

    def load(self, example_name: str) -> Example:
        """Load example dataset, download if not locally available."""
        if example_name in self.datasets:
            example = self.datasets[example_name]
            if example.installed:
                return example
            else:
                example.download()
                return example
        else:
            print(f'Example not available: {example_name}')
            return None

    def download_remotes(self):
        """Download all remotes."""
        names = list(self.remotes.keys())
        names.sort()

        for name in names:
            print(name)
            example = self.remotes[name]
            try:
                example.download()
            except:
                print(f'Example not downloaded: {name}.')

    def get_installed_names(self) -> list:
        """Return names of all currently installed datasets."""
        ds = self.datasets
        return [name for name in ds if ds[name].installed]

    def get_remote_url(self, name):
        if name in self.datasets: 
            try:
                return self.datasets[name].download_url
            except:
                print(f'{name} is a built-in dataset, no url.')
        else:
            print(f'{name} is not an available dataset.')
                

    def summary(self):
        """Report on datasets."""
        available = self.available()
        n = available.shape[0]
        n_installed = available.Installed.sum()
        n_remote = n - n_installed
        print(f'{n} datasets available, {n_installed} installed, {n_remote} remote.')


example_manager = Examples()

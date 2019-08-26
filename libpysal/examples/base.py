"""
Base class for managing example datasets
"""

# Authors: Serge Rey
# License: BSD 3 Clause

from collections import namedtuple
import os
from os import environ, makedirs, remove, chdir, rename
from os.path import exists, expanduser, isdir, join, isfile
import hashlib
from shutil import rmtree
from zipfile import ZipFile
from urllib.request import urlretrieve

PYSALDATA = 'pysal_data'
RemoteFileMetadata = namedtuple('RemoteFileMetadata',
                                ['filename', 'url', 'checksum'])

class ExampleManager:
    """Manager for example data sets

    Attributes
    ----------
    base_dir : string
      path to built-in local datasets

    data_home : string
      path to fetched datasets

    f_2_dir : dict
      mapping from example file to its path

    example_dirs: dict
      mapping from example to its directory path
    """

    def __init__(self):
        data_home = environ.get("PYSALDATA", join("~", PYSALDATA))
        self.data_home = expanduser(data_home)
        base_dir = os.path.abspath(os.path.dirname(__file__))
        self.base_dir = base_dir
        self.update_datasets()

    def update_datasets(self):
        f_2_dir = {}  # maps files to their directory
        example_dirs = {}  # example name is the key, directory is value
        for parent in [self.data_home, self.base_dir]:
            if exists(parent):  # data_home may not exist initially
                for directory in os.listdir(parent):
                    dir_path = join(parent, directory)
                    if isdir(dir_path):
                        example_dirs[directory] = dir_path
                        files = [
                            f for f in os.listdir(dir_path)
                            if isfile(join(dir_path, f))
                        ]
                        for f in files:
                            f_2_dir[f] = dir_path
        self.f_2_dir = f_2_dir
        self.example_dirs = example_dirs

    def available(self):
        descriptions = []
        for example in self.example_dirs:
            readme = join(self.example_dirs[example], "README.md")
            if exists(readme):
                with open(readme, 'r') as io:
                    lines = io.readlines()
                    if lines[0] == "\n":
                        lines = lines[1:]
                    title = lines[0].strip("\n")
                    short = lines[3].strip("\n")
                    rest = "".join(lines[4:])
                    d = {
                        "name": title,
                        "description": short,
                        "explanation": rest
                    }
            else:
                d = {"name": example, "description": None, "explanation": None}
            descriptions.append(d)
        return [{d["name"]: d["description"] for d in descriptions}]

    def explain(self, name):
        """Explain the dataset

        Parameters
        ----------

        name : string
            Title of the dataset
        """
        if name not in self.example_dirs:
            print(name, " is not a PySAL dataset.")
        pth = join(self.example_dirs[name], "README.md")
        with open(pth, 'r') as readme:
            contents = readme.read()
        print(contents)

    def get_path(self, name):
        """Get the path for a file in an example dataset

        Parameters
        ----------

        name : string
          Name of a file belonging to an example dataset
        """
        if name in self.f_2_dir:
            pth = join(self.f_2_dir[name], name)
            return pth
        print(name, 'not found.')
        return None


example_manager = ExampleManager()


def get_data_home(data_home=None):
    """Return the path of the libpysal data directory.


    This folder is used by some large dataset loaders to avoid downloading the
    data multiple times.


    Alternatively, it can be set by the 'PYSALDATA' environment variable or
    programmatically by giving an explicit folder path. The '~' symbol is
    expanded to the user home folder

    If the folder does not already exisit, it is automatically created.

    Parameters
    ----------

    data_home : str | None
        The path to the libpysal data directory.


    """

    if data_home is None:
        data_home = environ.get('PYSALDATA', join("~", PYSALDATA))
    data_home = expanduser(data_home)
    if not exists(data_home):
        makedirs(data_home)
    return data_home


def _sha256(path):
    """Calculate the sha256 hash of the file at path."""
    sha256hash = hashlib.sha256()
    chunk_size = 8192
    with open(path, "rb") as f:
        while True:
            buffer = f.read(chunk_size)
            if not buffer:
                break
            sha256hash.update(buffer)
    return sha256hash.hexdigest()


def _fetch_remote(remote, dirname=None):
    """Helper function to download a remote dataset into path
    Fetch a dataset pointed by remote's url, save into path using remote's
    filename and ensure its integrity based on the SHA256 Checksum of the
    downloaded file.
    Parameters
    ----------
    remote : RemoteFileMetadata
        Named tuple containing remote dataset meta information: url, filename
        and checksum
    dirname : string
        Directory to save the file to.
    Returns
    -------
    file_path: string
        Full path of the created file.
    """

    file_path = (remote.filename if dirname is None else join(
        dirname, remote.filename))
    urlretrieve(remote.url, file_path)
    checksum = _sha256(file_path)
    if remote.checksum != checksum:
        raise IOError("{} has an SHA256 checksum ({}) "
                      "differing from expected ({}), "
                      "file may be corrupted.".format(file_path, checksum,
                                                      remote.checksum))
    return file_path


def _fetch(metadata,
           dir_name,
           description,
           data_home=None,
           download_if_missing=True,
           is_dir=True):
    """Helper function to download a remote dataset

    Parameters
    ----------
    metadata : RemoteFileMetadata
      Named tuple containing remote dataset meta information: url, filename
      and checksum

    dir_name : string
      Name of the directory to store dataset

    description : string
      Long description of the contents of the dataset

    data_home : string
      Path of parent directory for dir_name

    download_if_missing : boolean
      If file in metadata is not available locally, download and install (Default=True). Raise IOError and do not download if False.

    is_dir : boolean
      The archive create a new directory upon extract (Default=True) or extracts into the parent directory (False)
    """
    data_home = get_data_home(data_home=data_home)
    if not exists(data_home):
        makedirs(data_home)
    dataset_path = join(data_home, dir_name)
    if not exists(dataset_path):
        if not download_if_missing:
            raise IOError("Data not found and 'download_if_missing' is False")

        print('downloading dataset from %s to %s' % (metadata.url,
                                                        data_home))
        _ = _fetch_remote(metadata, dirname=data_home)
        file_name = join(data_home, metadata.filename)
        with ZipFile(file_name, 'r') as archive:
            print('Extracting files....')
            if is_dir:
                archive.extractall(path=data_home)
                info = list(archive.infolist())
                if info[0].is_dir():
                    old_dir = info[0].filename.replace("/", "")
            else:
                archive.extractall(path=join(data_home, dir_name))
                old_dir = metadata.filename.split(".")[0]

        chdir(data_home)
        rename(old_dir, dir_name)

        # write README.md from original libpysal
        readme_pth = join(dataset_path, 'README.md')
        with open(readme_pth, 'w') as readme:
            readme.write(description)

        # remove zip archive
        remove(file_name)

        # remove __MACOSX if it exists
        if is_dir:
            mac = join(data_home, "__MACOSX")
        else:
            mac = join(data_home, dir_name)
            mac = join(mac, "__MACOSX")
        if exists(mac):
            rmtree(mac)

        # update pysal datasets
        example_manager.update_datasets()

    else:
        print('already exists, not downloading')

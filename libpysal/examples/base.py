import os
import csv
import sys
import shutil
import warnings
from collections import namedtuple
from os import environ, listdir, makedirs
from os.path import dirname, exists, expanduser, isdir, join, splitext
import hashlib
from shutil import rmtree
from os.path import dirname, exists, join
from os import makedirs, remove, chdir, rename
from zipfile import ZipFile

PYSALDATA = 'pysal_data'

import numpy as np

from urllib.request import urlretrieve

RemoteFileMetadata = namedtuple('RemoteFileMetadata',
                                ['filename', 'url', 'checksum'])


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
    print(checksum)
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
    data_home = get_data_home(data_home=data_home)
    if not exists(data_home):
        makedirs(data_home)
    dataset_path = join(data_home, dir_name)
    if not exists(dataset_path):
        if not download_if_missing:
            raise IOError("Data not found and 'download_if_missing' is False")
        else:
            print('downloading dataset from %s to %s' % (metadata.url,
                                                         data_home))
            data_path = _fetch_remote(metadata, dirname=data_home)
            file_name = join(data_home, metadata.filename)
            with ZipFile(file_name, 'r') as archive:
                print('Extracting files....')
                if is_dir:
                    archive.extractall(path=data_home)
                else:
                    archive.extractall(path=join(data_home, dir_name))

            chdir(data_home)
            old_dir = metadata.filename.split(".")[0]
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
                print('removing: ', mac)
                rmtree(mac)

    else:
        print('already exists, not downloading')

"""The :mod:`libpysal.examples` module provides example datasets.

The datasets consist of two sets, built-ins which are installed with
this module and remotes that can be downloaded.

This module provides functionality for working with these example datasets.
"""


import pandas as pd
from .base import example_manager
from .remotes import datasets as remote_datasets
from .builtin import datasets as builtin_datasets
from typing import Union


available_datasets = builtin_datasets.copy()
available_datasets.update(remote_datasets.datasets)

__all__ = ["get_path", "available", "explain", "fetch_all",
           "get_url", "load_example", "summary"]

example_manager.add_examples(available_datasets)


def fetch_all():
    """Fetch and install all remote datasets."""
    datasets = remote_datasets.datasets
    names = list(datasets.keys())
    names.sort()
    for name in names:
        example = datasets[name]
        try:
            example.download()
        except:
            print("Example not downloaded: {}".format(name))
    example_manager.add_examples(datasets)


def available() -> pd.DataFrame:
    """Return a dataframe with available datasets."""
    return example_manager.available()


def explain(name: str) -> str:
    """Explain a dataset by name."""
    return example_manager.explain(name)


def get_url(name: str) -> str:
    """Get url for remote dataset."""
    return example_manager.get_remote_url(name)


def load_example(example_name: str) -> Union[base.Example, builtin.LocalExample]:
    """Load example dataset instance."""
    example = example_manager.load(example_name)
    return example


def get_path(file_name: str) -> str:
    """Get the path for a file by searching installed datasets."""
    installed = example_manager.get_installed_names()
    for name in installed:
        example = example_manager.datasets[name]
        pth = example.get_path(file_name, verbose=False)
        if pth:
            return pth
    print("{} is not a file in any installed dataset.".format(file_name))


def summary():
    """Summary of datasets."""
    example_manager.summary()

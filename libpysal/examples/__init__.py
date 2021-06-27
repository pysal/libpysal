""" The :mod:`libpysal.examples` module includes a number of small built-in
    example datasets as well as functions to fetch larger datasets.
"""


import pandas as pd
from .base import example_manager
from .remotes import datasets as remote_datasets
from .builtin import datasets as builtin_datasets


from typing import Union

__all__ = ["get_path", "available", "explain", "fetch_all"]

example_manager.add_examples(builtin_datasets)

def fetch_all():
    """Fetch and install all remote datasets
    """
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
    fetch_all()

    return example_manager.available()


def explain(name: str) -> str:
    """Explain a dataset by name."""

    return example_manager.explain(name)


def load_example(example_name: str) -> Union[base.Example, builtin.LocalExample]:
    """Load example dataset instance."""
    example = example_manager.load(example_name)

    if example is None:
        fetch_all()  # refresh remotes
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

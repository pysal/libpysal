"""
The :mod:`libpysal.examples` module includes a number of small built-in example datasets as well as functions to fetch larger datasets.
"""
from .base import example_manager

__all__ = ['get_path', 'available', 'explain']

def get_path(name):
    """
    Get path of  example folders
    """
    return example_manager.get_path(name)

def available():
    """
    List available datasets
    """
    return example_manager.available()


def explain(name):  # would be nice to use pandas for display here
    """
    Explain a dataset by name
    """
    return example_manager.explain(name)

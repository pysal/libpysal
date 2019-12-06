"""
The :mod:`libpysal.examples` module includes a number of small built-in example datasets as well as functions to fetch larger datasets.
"""
from .base import example_manager
from .remotes import datasets as remotes
#from .nat import fetch_nat
#from .rio_grande_do_sul import fetch_rio
#from .guerry import fetch_guerry
#from .nyc_bikes import fetch_bikes
#from .sacramento2 import fetch_sacramento2
#from .south import fetch_south
#from .taz import fetch_taz

__all__ = [
    'get_path', 'available', 'explain', 'fetch_nat', 'fetch_rio', 'fetch_all'
]


example_manager.remotes = remotes
example_manager.locals = {}


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


def explain(name):  
    """
    Explain a dataset by name
    """
    return example_manager.explain(name)

def load_example(example_name):
    """
    Load example dataset instance
    """
    return example_manager.load(example_name)

#remotes = {}
#remotes['guerry'] = fetch_guerry
#remotes['rio'] = fetch_rio
#remotes['nat'] = fetch_nat
#remotes['nyc_bikes'] = fetch_bikes
#remotes['sacramento2'] = fetch_sacramento2
#remotes['south'] = fetch_south
#remotes['taz'] = fetch_taz
#
#
#def fetch_all():
#    """Fetch all the large remote example datasets"""
#    for data_set in remotes:
#        remotes[data_set]()


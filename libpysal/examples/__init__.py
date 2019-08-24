import os
from os import environ
from os.path import expanduser, join, exists
from .base import PYSALDATA


class DataSets:
    def __init__(self):
        self.data_sets = {}

    def add(self, data_set):
        self.data_sets[data_set.name] = data_set


data_sets = {}


class DataSet:
    def __init__(self, name, fetch=False, path=None, description=None):
        self.name = name
        self.fetch = fetch
        self.description = description
        self.path = path
        self.__register()

    def __register(self):
        data_sets[self.name] = self

    def __repr__(self):
        return "{name:" + self.name + "}"

    def __str__(self):
        if self.description is None:
            return "No description."
        return self.description


base_dir = os.path.abspath(os.path.dirname(__file__))
__all__ = ["get_path", "available", "explain"]
file_2_dir = {}
example_dir = base_dir

dirs = []
for root, subdirs, files in os.walk(example_dir, topdown=False):

    head, tail = os.path.split(root)
    if tail != "examples":
        for f in files:
            file_2_dir[f] = root
        dirs.append(tail)
        desc = None
        if f == "README.md":
            f = join(root, f)
            with open(f, "r") as desc_file:
                desc = desc_file.read()
        data_set = DataSet(tail, path=root, description=desc)

# check if user has cached downloads
# if so, update file list
data_home = environ.get("PYSALDATA", join("~", PYSALDATA))
data_home = expanduser(data_home)
if exists(data_home):
    for root, subdirs, files in os.walk(data_home, topdown=False):
        head, tail = os.path.split(root)
        if tail != "examples":
            dirs.append(tail)
            for f in files:
                file_2_dir[f] = root

# databases no longer included in source: anything > 1m
fetch_datasets = [
    "nat",
    "nyc_bikes",
    "rio_grande_do_sul",
    "taz",
    "clearwater",
    "south",
    "guerry",
    "newHaven",
    "sacramento2",
    "georgia",
    "virginia",
]


def get_path(example_name, raw=False):
    """
    Get path of  example folders
    """
    if type(example_name) != str:
        try:
            example_name = str(example_name)
        except:
            raise KeyError("Cannot coerce requested example name to string")
    if example_name in dirs:
        outpath = os.path.join(example_dir, example_name, example_name)
    elif example_name in file_2_dir:
        d = file_2_dir[example_name]
        outpath = os.path.join(d, example_name)
    elif example_name == "":
        outpath = os.path.join(base_dir, "examples", example_name)
    else:
        raise KeyError(example_name + " not found in PySAL built-in examples.")
    name, ext = os.path.splitext(outpath)
    if (ext == ".zip") and (not raw):
        outpath = "zip://" + outpath
    return outpath


def available(verbose=False):
    """
    List available datasets
    """

    examples = [os.path.join(base_dir, d) for d in dirs]
    if not verbose:
        return [os.path.split(d)[-1] for d in examples]
    examples = [os.path.join(dty, "README.md") for dty in examples]
    descs = [_read_example(path) for path in examples]
    return [{desc["name"]: desc["description"] for desc in descs}]


def _read_example(pth):
    try:
        with open(pth, "r") as io:
            title = io.readline().strip("\n")
            io.readline()  # titling
            io.readline()  # pad
            short = io.readline().strip("\n")
            io.readline()  # subtitling
            io.readline()  # pad
            rest = io.readlines()
            rest = [l.strip("\n") for l in rest if l.strip("\n") != ""]
            d = {"name": title, "description": short, "explanation": rest}
    except IOError:
        base_dirname = os.path.split(pth)[-2]
        dirname = os.path.split(base_dirname)[-1]
        d = {"name": dirname, "description": None, "explanation": None}
    return d


def explain(name):  # would be nice to use pandas for display here
    """
    Explain a dataset by name
    """
    if name in data_sets:
        print(data_sets[name])
    else:
        print(name, "is not a PySAL dataset..")
    # path = os.path.split(get_path(name))[0]
    # fpath = os.path.join(path, "README.md")
    # return _read_example(fpath)

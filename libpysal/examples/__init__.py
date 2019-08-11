import os
from os import environ
from os.path import expanduser, join, exists

base_dir = os.path.abspath(os.path.dirname(__file__))
__all__ = ['get_path', 'available', 'explain']
file_2_dir = {}
example_dir = base_dir

dirs = []
for root, subdirs, files in os.walk(example_dir, topdown=False):
    for f in files:
        file_2_dir[f] = root
    head, tail = os.path.split(root)
    if tail != 'examples':
        dirs.append(tail)

# check if user has cached downloads
# if so, update file list
data_home = environ.get('LIBPYSAL_DATA',
                        join("~", "libpysal_data"))
data_home = expanduser(data_home)
if exists(data_home):
    for root, subdirs, files in os.walk(data_home, topdown=False):
        for f in files:
            file_2_dir[f] = root
        head, tail = os.path.split(root)
        if tail != 'examples':
            dirs.append(tail)


# databases no longer included in source: anything > 1m
fetch_datasets = ["nat", "nyc_bikes", "rio_grande_do_sul",\
                  "taz", "clearwater", "south", "guerry",\
                  "newHaven", "sacramento2", "georgia" "virginia"]


def get_path(example_name, raw=False):
    """
    Get path of  example folders
    """
    if type(example_name) != str:
        try:
            example_name = str(example_name)
        except:
            raise KeyError('Cannot coerce requested example name to string')
    if example_name in dirs:
        outpath =  os.path.join(example_dir, example_name, example_name)
    elif example_name in file_2_dir:
        d = file_2_dir[example_name]
        outpath = os.path.join(d, example_name)
    elif example_name == "":
        outpath = os.path.join(base_dir, 'examples', example_name)
    else:
        raise KeyError(example_name + ' not found in PySAL built-in examples.')
    name,ext = os.path.splitext(outpath)
    if (ext == '.zip') and (not raw):
        outpath = 'zip://'+outpath
    return outpath

def available(verbose=False):
    """
    List available datasets
    """

    examples = [os.path.join(base_dir, d) for d in dirs]
    if not verbose:
        return [os.path.split(d)[-1] for d in examples]
    examples = [os.path.join(dty, 'README.md') for dty in examples]
    descs = [_read_example(path) for path in examples]
    return [{desc['name']:desc['description'] for desc in descs}]


def _read_example(pth):
    try:
        with open(pth, 'r') as io:
            title = io.readline().strip('\n')
            io.readline()  # titling
            io.readline()  # pad
            short = io.readline().strip('\n')
            io.readline()  # subtitling
            io.readline()  # pad
            rest = io.readlines()
            rest = [l.strip('\n') for l in rest if l.strip('\n') != '']
            d = {'name': title, 'description': short, 'explanation': rest}
    except IOError:
        base_dirname = os.path.split(pth)[-2]
        dirname = os.path.split(base_dirname)[-1]
        d = {'name': dirname, 'description': None, 'explanation': None}
    return d


def explain(name):  # would be nice to use pandas for display here
    """
    Explain a dataset by name
    """
    path = os.path.split(get_path(name))[0]
    fpath = os.path.join(path, 'README.md')
    return _read_example(fpath)

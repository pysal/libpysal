# Installation

libpysal supports python \>= [3.11](https://docs.python.org/3.11/) only.
Please make sure that you are operating in a python 3 environment.

## Installing released version

### conda

libpysal is available through conda:

    conda install -c conda-forge libpysal

### pypi

libpysal is available on the [Python Package
Index](https://pypi.org/project/libpysal/). Therefore, you can either
install directly with [pip]{.title-ref} from the command line:

    pip install -U libpysal

or download the source distribution (.tar.gz) and decompress it to your
selected destination. Open a command shell and navigate to the
decompressed folder. Type:

    pip install .

## Installing development version

Potentially, you might want to use the newest features in the
development version of libpysal on github -
[pysal/libpysal](https://github.com/pysal/libpysal) while have not been
incorporated in the Pypi released version. You can achieve that by
installing [pysal/libpysal](https://github.com/pysal/libpysal) by
running the following from a command shell:

    pip install git+https://github.com/pysal/libpysal.git

You can also [fork](https://help.github.com/articles/fork-a-repo/) the
[pysal/libpysal](https://github.com/pysal/libpysal) repo and create a
local clone of your fork. By making changes to your local clone and
submitting a pull request to
[pysal/libpysal](https://github.com/pysal/libpysal), you can contribute
to libpysal development.

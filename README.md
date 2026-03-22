# libpysal: Python Spatial Analysis Library Core

[![Continuous Integration](https://github.com/pysal/libpysal/actions/workflows/unittests.yml/badge.svg)](https://github.com/pysal/libpysal/actions/workflows/unittests.yml)
[![codecov](https://codecov.io/gh/pysal/libpysal/branch/main/graph/badge.svg)](https://codecov.io/gh/pysal/libpysal)
[![PyPI version](https://badge.fury.io/py/libpysal.svg)](https://badge.fury.io/py/libpysal)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/libpysal.svg)](https://anaconda.org/conda-forge/libpysal)
[![DOI](https://zenodo.org/badge/81501824.svg)](https://zenodo.org/badge/latestdoi/81501824)
[![Discord](https://img.shields.io/badge/Discord-join%20chat-7289da?style=flat&logo=discord&logoColor=cccccc&link=https://discord.gg/BxFTEPFFZn)](https://discord.gg/BxFTEPFFZn)
[![SPEC 0 — Minimum Supported Dependencies](https://img.shields.io/badge/SPEC-0-green?labelColor=%23004811&color=%235CA038)](https://scientific-python.org/specs/spec-0000/)

## Introduction

**libpysal** offers five modules that form the building blocks in many upstream packages
in the [PySAL family](https://pysal.org):

- `libpysal.cg` – Computational geometry
- `libpysal.examples` – Built-in example datasets
- `libpysal.graph` – Graph class encoding spatial weights matrices
- `libpysal.io` – Input and output
- `libpysal.weights` – Spatial weights (legacy)

## Documentation

The documentation of `libpysal` is available at
[pysal.org/libpysal](https://pysal.org/libpysal/).

## Development

libpysal development is hosted on [GitHub](https://github.com/pysal/libpysal).

Discussions of development occurs on PySAL [Discord](https://discord.gg/BxFTEPFFZn).

## Contributing

PySAL-libpysal is under active development and contributors are welcome. If you have any
suggestions, feature requests, or bug reports, please open new
[issues](https://github.com/pysal/libpysal/issues) on GitHub. To submit patches, please
review [PySAL's documentation for developers](https://pysal.org/docs/devs/), the PySAL
[development guidelines](https://github.com/pysal/pysal/wiki), and the
[libpysal contributing guidelines](https://github.com/pysal/libpysal/blob/main/.github/CONTRIBUTING.md)
before opening a [pull request](https://github.com/pysal/libpysal/pulls). Once your
changes get merged, you’ll automatically be added to the
[Contributors List](https://github.com/pysal/libpysal/graphs/contributors).

## Citing libpysal

If you use PySAL in a scientific publication, we would appreciate citations to the
following paper:

> 
> [PySAL: A Python Library of Spatial Analytical Methods](http://journal.srsa.org/ojs/index.php/RRS/article/view/134/85),
> *Rey, S.J. and L. Anselin*, Review of Regional Studies 37, 5-27 2007.

Bibtex entry:

```bibtex
@Article{pysal2007,
  author={Rey, Sergio J. and Anselin, Luc},
  title={{PySAL: A Python Library of Spatial Analytical Methods}},
  journal={The Review of Regional Studies},
  year=2007,
  volume={37},
  number={1},
  pages={5-27},
  keywords={Open Source; Software; Spatial}
}
```

## License information

The package is licensed under BSD 3-Clause License (Copyright (c) 2007-, PySAL
Developers).

.. libpysal documentation master file

libpysal: Python Spatial Analysis Library Core
==============================================

.. image:: https://github.com/pysal/libpysal/workflows/.github/workflows/unittests.yml/badge.svg
   :target: https://github.com/pysal/libpysal/actions?query=workflow%3A.github%2Fworkflows%2Funittests.yml

.. image:: https://badges.gitter.im/pysal/pysal.svg
   :target: https://gitter.im/pysal/pysal

.. image:: https://badge.fury.io/py/libpysal.svg
    :target: https://badge.fury.io/py/libpysal

.. raw:: html

    <div class="container-fluid">
      <div class="row equal-height">
        <div class="col-sm-1 col-xs-hidden">
        </div>
        <div class="col-sm-10 col-xs-12">
            <a href="http://nbviewer.jupyter.org/github/pysal/libpysal/blob/main/notebooks/weights.ipynb" class="thumbnail">
                <img src="_static/images/npweights.png" class="img-responsive center-block">
                <div class="caption text-center">
                <h6>Weights for nonplanar enforced geometries</h6>
                </div>
            </a>
        </div>
        <div class="col-sm-1 col-xs-hidden">
        </div>
      </div>
    </div>


************
Introduction
************

**libpysal** offers four modules that form the building blocks in many upstream packages in the `PySAL family <https://pysal.org>`_:

- Spatial Weights: libpysal.weights
- Input-and output: libpysal.io
- Computational geometry: libpysal.cg
- Built-in example datasets libpysal.examples


Examples demonstrating some of **libpysal** functionality are available in the `tutorial <tutorial.html>`_.

Details are available in the `libpysal api <api.html>`_.

For background information see :cite:`pysal2007`.

***********
Development
***********

libpysal development is hosted on github_.

.. _github : https://github.com/pysal/libpysal

Discussions of development occurs on the
`developer list <http://groups.google.com/group/pysal-dev>`_
as well as gitter_.

.. _gitter : https://gitter.im/pysal/pysal?

****************
Getting Involved
****************

If you are interested in contributing to PySAL please see our
`development guidelines  <https://github.com/pysal/pysal/wiki>`_.


***********
Bug reports
***********

To search for or report bugs, please see libpysal's issues_.

.. _issues :  http://github.com/pysal/libpysal/issues


***************
Citing libpysal
***************

If you use PySAL in a scientific publication, we would appreciate citations to the following paper:

  `PySAL: A Python Library of Spatial Analytical Methods <http://journal.srsa.org/ojs/index.php/RRS/article/view/134/85>`_, *Rey, S.J. and L. Anselin*, Review of Regional Studies 37, 5-27 2007.

  Bibtex entry::

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



*******************
License information
*******************

See the file "LICENSE.txt" for information on the history of this
software, terms & conditions for usage, and a DISCLAIMER OF ALL
WARRANTIES.


libpysal
========

Core components of the Python Spatial Analysis Library (`PySAL`_)



.. toctree::
   :hidden:
   :maxdepth: 4
   :caption: Contents:

   Installation <installation>
   Tutorial <tutorial>
   API <api>
   References <references>
   
.. _PySAL: https://github.com/pysal/pysal

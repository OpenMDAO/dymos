.. dymos documentation master file, created by
   sphinx-quickstart on Fri Oct  6 12:34:25 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Dymos: Open-source Optimal Control for Multidisciplinary Systems
====================================================================

Dymos is an open-source tool for solving optimal control problems involving multidisplinary systems.
It is built on top of the `OpenMDAO framework <https://github.com/openmdao/blue>`_.
The goal of Dymos is to allow users to solve optimal control problems using a variety of methods.
Presently, Dymos supports optimal control via the following techniques:

* Gauss-Lobatto collocation method [ConwayHerman1996]_
* Radau-pseudospectral method [Garg2009]_
* Simple ODE integration and shooting via Runge-Kutta

Dymos was designed with generality and extensibility to other optimal control transcriptions in mind.

Table of Contents
=================

.. toctree::
    :maxdepth: 3
    :titlesonly:

    quick_start/quick_start
    feature_reference/feature_reference
    examples/examples
    references/references.rst

Source Docs
===========

.. toctree::
    :maxdepth: 1
    :titlesonly:

    _srcdocs/index.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


References
==========
.. [ConwayHerman1996] ﻿Herman, Albert L, and Bruce A Conway. “Direct Optimization Using Collocation Based on High-Order Gauss-Lobatto Quadrature Rules.” Journal of Guidance, Control, and Dynamics 19.3 (1996): 592–599.
.. [Garg2009] ﻿Garg, Divya et al. “Direct Trajectory Optimization and Costate Estimation of General Optimal Control Problems Using a Radau Pseudospectral Method.” American Institute of Aeronautics and Astronautics, 2009.

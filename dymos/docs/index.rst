.. dymos documentation master file, created by
   sphinx-quickstart on Fri Oct  6 12:34:25 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


|project|: Open-source Optimal Control for Multidisciplinary Systems
====================================================================

|project| is an open-source tool for solving optimal control problems involving multidisplinary systems.
It is built on top of the `OpenMDAO framework <https://github.com/openmdao/blue>`_.
The goal of |project| is to allow users to solve optimal control problems using a variety of methods.
Presently, |project| supports optimal control via the following techniques:

* Gauss-Lobatto collocation method [ConwayHerman1996]_
* Radau-pseudospectral method [Garg2009]_

More direct optimal control transcriptions may be added to |project| in the future.


Table of Contents
=================

.. toctree::
    :maxdepth: 3
    :titlesonly:

    user_guide/user_guide
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

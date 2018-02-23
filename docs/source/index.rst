.. openmdoc documentation master file, created by
   sphinx-quickstart on Fri Oct  6 12:34:25 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

|project|: Open-source Optimal Control for Multidisciplinary Systems
======================================================================

|project| is an open-source tool for solving optimal control problems involving multidisplinary systems.
It is built on top of the `OpenMDAO framework <https://github.com/openmdao/blue>`_.
The goal of |project| is to allow users to solve optimal control problems using a variety of methods, both direct and indirect.
Presently, |project| supports optimal control through three different methods, which can be broadly categorized as either implicit or explicit methods:

* Gauss-Lobatto collocation (implicit)
* Radau-pseudospectral method (implicit)
* Shooting phases (explicit)

.. * Analytic Phases (analytic)

.. toctree::
    :maxdepth: 3
    :titlesonly:

    user_guide/user_guide
    examples/examples
    api_docs



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

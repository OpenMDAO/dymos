.. dymos documentation master file, created by
   sphinx-quickstart on Fri Oct  6 12:34:25 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

|project|: Open-source Optimal Control for Multidisciplinary Systems
======================================================================

|project| is an open-source tool for solving optimal control problems involving multidisplinary systems.
It is built on top of the `OpenMDAO framework <https://github.com/openmdao/blue>`_.
The goal of |project| is to allow users to solve optimal control problems using a variety of methods.
Presently, |project| supports optimal control in three ways:

* Gauss--Lobatto collocation method :

  This method discretizes the ODE using an odd-numbered Legendre--Gauss--Lobatto points (e.g., 5).
  A Hermite interpolant is generated from the odd-indexed points (e.g., 1st, 3rd, and 5th),
  and the remaining points are the collocation points (e.g., 2nd and 4th).

* Radau-pseudospectral method :

  This method discretizes the ODE using Legendre--Gauss--Radau collocation points.
  A Lagrange interpolant is generated to predict derivatives at each collocation point
  that are forced to be equal to the right-hand side of the ODE as evaluated at the same points.

* GLM family of methods :

  The general linear methods (GLM) is a wide-ranging formulation
  that unifies a large family of ODE integrators.
  In Dymos, a long list of explicit and implicit Runge--Kutta methods are made available via the GLM option.
  Internally, the Runge--Kutta are used to solve the ODE as an initial-value problem,
  but Dymos can make use of Runge--Kutta methods to solve general optimal control problems including boundary-value problems.

.. * Analytic Phases (analytic)


Table of Contents
=================

.. toctree::
    :maxdepth: 3
    :titlesonly:

    user_guide/user_guide
    reference_manual/reference_manual
    examples/examples
    api_docs



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

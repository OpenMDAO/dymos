---
title: 'Dymos: Optimal Control of Multidisciplinary Systems'
tags:
  - Python
  - OpenMDAO
  - optimal control
  - trajectory optimization
  - multidisciplinary optimization
authors:
  - name: Rob Falck
    orcid: 0000-0001-9864-4928
    affiliation: 1
  - name: Justin S. Gray
    affiliation: 1
  - name: Kaushik Ponnapalli
    affiliation: 2
affiliations:
 - name: NASA Glenn Research Center.  21000 Brookpark Rd, Cleveland, OH 44135.
   index: 1
-  name: Vantage Partners LLC. 3000 Aerospace Parkway, Brook Park, OH 44142.
-  index: 2
date: 26 June 2020
bibliography: paper.bib
---

# Summary

Dymos is a Python package for solving optimal control problems within
the OpenMDAO [@Gray2019a] software framework.  While there are a number of software
packages available for finding optimal control solutions for various applications,
Dymos was developed to efficiently solve problems wherein the system dynamics
may include expensive, implicit calculations.  By leveraging the OpenMDAO software
package's approach to calculating derivatives for gradient-based optimization,
Dymos can provide significant improvements in performance.

Dymos leverages two common "transcriptions" for direct collocation in its approach to optimal control:
the high-order Gauss-Lobatto transcription developed by Herman and Conway [@Herman1996]
and the Radau pseudospectral method [@Garg2009].  These techniques are implemented
in a way that is nearly transparent to the user so that these different
techniques can be employed with only minor changes to the user's code - typically a single line.

OpenMDAO, the framework on which Dymos is built, can efficiently handle
the inclusion of iterative nonlinear solvers within the user's dynamics model,
as is the case in many problems which involve differential algebraic equations (DAEs).
This enables efficient use of the method differential inclusions [@Seywald1994].
The method of differential inclusions allows the user to parameterize the trajectory
of a system in non-traditional ways and have an embedded nonlinear solver enforce
the system dynamics.  This gives users more freedom to pose dynamics in
more natural ways with less implications on performance.

This greater freedom to utilize nonilinear solvers within the context of optimization
also allows the user to choose whether the dynamics constraints of the direct collocation
techniques are enforced by the optimizer (as is the typical approach), or by a nonlinear
solver.  In the latter case, the direct collocation techniques are transformed into
single or multipe shooting methods, which may provide better performance in some scenarios.
Because shooting methods often pose far fewer constraints for the optimizer than design variables,
we can leverage OpenMDAO's support for adjoint (reverse) differentiation to realize the benefits
of shooting methods without a substantial performance penalty.

By coupling OpenMDAO's unique approach to efficient computation of derivatives to
standard optimal control techniques, Dymos, enables users to solve optimal control
problems which involve potentially expensive iterative techniques in the dynamics.
Dymos has been used to demonstrate the coupling of flight dynamics and subsystem
thermal constraints in electrical aircraft applications [@Falck2017a, @Hariton2020a].
NASA's X-57 "Maxwell" is using Dymos for mission planning to maximize
data collection while abiding the limits of battery storage capacity and
subsystem temperatures [@Schnulo2018a, @Schnulo2019a].  Other authors have
used Dymos to perform studies of aircraft acoustics [@Ingraham2020a] and
the the design of supersonic aircraft with thermal fuel management systems [@Jasa2018a].

# Acknowledgements

Dymos was developed with funding from NASA's Transformative Tools and Technologies ($T^3$) Project.

# References

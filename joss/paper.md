---
title: 'dymos: A Python package for optimal control of multidisciplinary systems'
tags:
  - Python
  - OpenMDAO
  - optimal control
  - trajectory optimization
  - multidisciplinary optimization
  - NASA
authors:
  - name: Rob Falck
    orcid: 0000-0001-9864-4928
    affiliation: 1
  - name: Justin Gray
    affiliation: 1
  - name: Kaushik Ponnapalli
    affiliation: 2
  - name: Ted Wright
    affiliation: 1
affiliations:
  - name: NASA Glenn Research Center
    index: 1
  - name: HX5 LLC
    index: 2
date: 26 June 2020
bibliography: paper.bib
---

# Summary

Dymos is an library for solving optimal-control type optimization problems. 
It contains capabilities typical of optimal control software and can handle a wide range of typical optimal control problems.
Its software design (built on top of NASA's OpenMDAO Framework [@Gray2019a] also allows users to tackle problems where the trajectory is not necessarily central to the optimization and where the system dynamics may include expensive, implicit calculations.
Implicit calculations may be present due to some calculation in the ODE, or more generally because the ODE is posed as a set of differential algebraic equations or differential inclusions [@Seywald1994].

Support for implicit calculations gives users more freedom to pose dynamics in more natural ways, but typically causes numerical and computational cost challenges in an optimization context, especially when finite-differences are used to compute derivatives for the optimizer.
Dymos employs analytic derivatives to overcome these challenges and makes such methods computationally feasible.
To achieve this, dymos makes extensive use of OpenMDAO's built in analytic derivative features, its non-linear solver library, and its support for gradient-based optimization.

Dymos can be used stand-alone, or as a building block in a larger model where a significant portion of the optimization is focused on some non-controls aspect with a trajectory added to enforce some constraint upon the design.
In some contexts, you may hear this kind of problem referred to as co-design, controls-co-design, or multidisciplinary design optimization.

## The dymos perspective on optimal control

In dymos trajectories are subdivided into "phases," and within each phase, boundary or path constraints may be applied to any output of the system dynamics.
In addition to typical collocated dynamic control variables, dymos offers the ability to use variable order polynomial controls that are useful in forcing a lower-order control solution upon a phase of the trajectory.

Dymos is primarily focused on an implicit pseudospectral approach to optimal-control. 
It leverages two common direct collocation transcriptions: the high-order Gauss-Lobatto transcription [@Herman1996] and the Radau pseudospectral method [@Garg2009].
Dymos also provides explicit forms of both of these transcriptions, which provides a single or multiple-shooting approach.
All of these transcriptions are implemented in a way that is independent of the ODE implementation, nearly transparent to the user, and requires very minor code changes - typically a single line in the run-script.
ODE's are implemented as standard OpenMDAO systems which are passed to phases at instantiation time with some additional annotations to identify the states, state-rates, and control inputs.

Dymos does not ship with its own built in optimizer. 
It relies on whatever optimizers you have available in your OpenMDAO installation. 
OpenMDAO ships with an interface to the optimizers in SciPy [@2020SciPy-NMeth], and an additional wrapper for the pyoptsparse [@Perez2012a] library which has more powerful optimizer options such as SNOPT [@GilMS05] and IPOPT [@wachter2006].
For simple problems, Scipy's SLSQP optimizer generally works fine.
On more challenging optimal-control problems, higher quality optimizers are important for getting good performance.

## Statement of Need

Modeling complex multidisciplinary systems often involves the use of iterative nonlinear solvers to converge the design or operational parameters of interacting subsystems.
To speed the design optimization of such systems, the NASA's OpenMDAO software was developed to generalize the calculation of derivatives across models for use in gradient-based optimization.
However, the optimal control of multidisciplinary systems may involve the use of nonlinear solvers within the ODE itself.
The implicit analysis could arise due to the need for some high fidelity physics model (e.g. a vortex lattice method for aerodynamic performance prediction) or from the formulation of a differential inclusion approach.
Despite the application of adjoint differentiation, shooting methods based on explicit time-marching still suffer from a performance standpoint due to the need to reconverge the solvers within the ODE at each time step.
Dymos was developed to leverage the advanced differentiation capabilities of OpenMDAO in combination with modern pseudospectral optimal control techniques to enable optimization of dynamic systems that feature complex interactions between subsystems.
Implicit pseudospectral approaches evaluate the ODE across an entire trajectory simultaneously.
While explicit time-marching requires of the repeated convergence of small, dense systems of equations at a single instant in time, implicit pseudospectral methods converge a larger but more sparse system of equations once across the trajectory per evaluation of the ODE.

Combining nonlinear solvers within the context of optimal-control problems grants the user a lot of flexibility in how to handle the direct collocation problems.
Typically, the collocation defects --- necessary to enforce the physics of the ODE --- are handled by assigning equality constraints to the optimizer.
However it is also possible to use a solver to converge the defects for every optimizer iteration, in effect creating a single or multiple shooting approach that may be beneficial for some problems.
If a solver-based collocation approach is used in combination with finite-differencing to approximate the derivatives needed by the optimizer, then the potential benefits are outweighed by numerical inaccuracies and high computational cost.
Dymos works around this problem by leveraging OpenMDAO's support for adjoint (reverse) differentiation to realize the benefits of shooting methods without a substantial performance penalty.

## Selected applications of dymos

Dymos has been used to demonstrate the coupling of flight dynamics and subsystem thermal constraints in electrical aircraft applications [@Falck2017a; @Hariton2020a].
NASA's X-57 "Maxwell" is using dymos for mission planning to maximize data collection while abiding the limits of battery storage capacity and subsystem temperatures [@Schnulo2018a; @Schnulo2019a].
Other authors have used dymos to perform studies of aircraft acoustics [@Ingraham2020a] and the the design of supersonic aircraft with thermal fuel management systems [@Jasa2018a].

# Acknowledgements

Dymos was developed with funding from NASA's Transformational Tools and Technologies ($T^3$) Project.

# References

# Dymos by Example

The goal of these examples is to walk users through the process of formulating an optimal control problem and solving it using Dymos.

In working through these examples, we'll try to emphasize the following process:

1.  Formulate the optimal control problem
2.  Identify state and control variables, and the ordinary differential equations (ODE) which govern the dynamics.
3.  Build the ODE as an OpenMDAO system.
4.  Test the evaluation of the ODE.
5.  Define the partial derivatives for the ODE.
6.  Test the partial derivatives of the ODE against finite-difference or (preferably) complex-step approximations.


## Prerequisites

These examples assume that the user has a working knowledge of the following:

-   Python
-   The numpy package for numerical computing with Python
-   The matplotlib package for plotting results
